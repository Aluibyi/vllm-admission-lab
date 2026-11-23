from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time

import httpx
import numpy as np
import yaml

from admission_lab.app import DEFAULT, estimate
from admission_lab.sse import SSEParser, content_event


async def run_trace(*, url, model, trace, prompts, output_dir, run_id,
                    config=None, native_priority=False, metrics_url=None,
                    debug_url=None, deadline=60.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists() or (output_dir / "requests.jsonl").exists():
        raise ValueError(f"Refusing to overwrite existing run {output_dir}")
    config = {**DEFAULT, **(config or {})}
    offered = trace["requests"]
    if len(offered) > 30000:
        raise ValueError("Trace too large for single-process open-loop runner")
    start = time.perf_counter() + 0.5
    results = []
    monitoring = []
    monitor_stop = asyncio.Event()
    invalid_reasons = []
    limits = httpx.Limits(max_connections=max(512, len(offered) + 10),
                          max_keepalive_connections=256)
    timeout = httpx.Timeout(deadline, connect=min(10, deadline), pool=min(10, deadline))

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def monitor():
            async with httpx.AsyncClient(timeout=2.0) as diagnostic:
                while not monitor_stop.is_set():
                    sample = {"offset_seconds": time.perf_counter() - start,
                              "client_process_cpu_seconds": time.process_time()}
                    if hasattr(os, "getloadavg"):
                        sample["host_load_average"] = list(os.getloadavg())
                    for label, target in (("backend_metrics", metrics_url), ("gateway", debug_url)):
                        if target:
                            try:
                                response = await diagnostic.get(target)
                                sample[label] = response.json() if label == "gateway" else response.text
                            except Exception as exc:
                                sample[label + "_error"] = type(exc).__name__
                    monitoring.append(sample)
                    try:
                        await asyncio.wait_for(monitor_stop.wait(), 0.5)
                    except asyncio.TimeoutError:
                        pass

        async def one(row):
            await asyncio.sleep(max(0, start + row["scheduled_seconds"] - time.perf_counter()))
            sent = time.perf_counter()
            prompt = prompts[row["prompt_key"]]
            request_id = f"{run_id}-{row['index']:06d}"
            cost = estimate(config, len(prompt), row["max_tokens"])
            result = {**row, "run_id": run_id, "request_id": request_id,
                      "send_offset_seconds": sent - start,
                      "send_lag_ms": (sent - start - row["scheduled_seconds"]) * 1000,
                      "estimated_cost_ms": cost, "http_status": None,
                      "status": "failed", "reason": "not_started",
                      "ttft_ms": None, "e2e_ms": None, "observed_tpot_ms": None,
                      "first_content_offset_seconds": None, "last_content_offset_seconds": None,
                      "actual_output_tokens": None, "actual_prompt_tokens": None,
                      "done": False, "finish_reason": None, "content_events": 0,
                      "metrics_valid": True}
            body = {"model": model, "prompt": prompt, "max_tokens": row["max_tokens"],
                    "temperature": 0, "top_p": 1, "n": 1, "stream": True,
                    "stream_options": {"include_usage": True}, "ignore_eos": True,
                    "return_token_ids": True, "request_id": request_id,
                    "priority": math.ceil(cost) if native_priority else 0}
            first = last = None
            usage = None
            token_ids = 0

            async def stream_request():
                nonlocal first, last, usage, token_ids
                parser = SSEParser()
                async with client.stream("POST", url, json=body,
                                         headers={"x-request-id": request_id}) as response:
                    result["http_status"] = response.status_code
                    if response.status_code >= 400:
                        result["reason"] = f"http_{response.status_code}"
                        return
                    async for chunk in response.aiter_bytes():
                        now = time.perf_counter()
                        for event in parser.feed(chunk):
                            if event == "[DONE]":
                                result["done"] = True
                                continue
                            value, content = content_event(event)
                            if value and value.get("usage"):
                                usage = value["usage"]
                            for choice in (value or {}).get("choices", []):
                                token_ids += len(choice.get("token_ids") or [])
                                if choice.get("finish_reason"):
                                    result["finish_reason"] = choice["finish_reason"]
                            if content:
                                first = now if first is None else first
                                last = now
                                result["content_events"] += 1
                    # A complete SSE event must be delimited; final flush also
                    # detects invalid UTF-8. Normal vLLM sends a trailing blank.
                    trailing = parser.feed(b"", final=True)
                    for event in trailing:
                        if event == "[DONE]":
                            result["done"] = True
                        else:
                            value, content = content_event(event)
                            if value and value.get("usage"):
                                usage = value["usage"]
                            if content:
                                first = time.perf_counter() if first is None else first
                                last = time.perf_counter()
                    if not result["done"] or first is None:
                        result["reason"] = "incomplete_stream"
                    else:
                        result["status"] = "succeeded"
                        result["reason"] = "done"

            try:
                await asyncio.wait_for(stream_request(), deadline)
            except asyncio.TimeoutError:
                result.update(status="timeout", reason="client_deadline")
            except Exception as exc:
                result.update(status="failed", reason=f"client_{type(exc).__name__}")
            terminal = time.perf_counter()
            result["e2e_ms"] = (terminal - sent) * 1000
            result["terminal_offset_seconds"] = terminal - start
            if first is not None:
                result["ttft_ms"] = (first - sent) * 1000
                result["first_content_offset_seconds"] = first - start
                result["last_content_offset_seconds"] = last - start
            if usage:
                result["actual_output_tokens"] = usage.get("completion_tokens")
                result["actual_prompt_tokens"] = usage.get("prompt_tokens")
            elif token_ids:
                result["actual_output_tokens"] = token_ids
                result["actual_prompt_tokens"] = None
            count = result["actual_output_tokens"]
            if count and count > 1 and first is not None:
                result["observed_tpot_ms"] = (last - first) * 1000 / (count - 1)
            if result["status"] == "succeeded":
                if count != row["max_tokens"]:
                    result.update(metrics_valid=False, status="failed", reason="output_budget_mismatch")
                if result["actual_prompt_tokens"] is not None and result["actual_prompt_tokens"] != len(prompt):
                    result.update(metrics_valid=False, status="failed", reason="prompt_token_mismatch")
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

        monitor_task = asyncio.create_task(monitor())
        with (output_dir / "requests.jsonl").open("w", encoding="utf-8", buffering=65536) as handle:
            await asyncio.gather(*(one(row) for row in offered))
            await asyncio.sleep(max(0, start + trace["duration_seconds"] - time.perf_counter()))
        total_seconds = max(trace["duration_seconds"], time.perf_counter() - start)
        monitor_stop.set()
        await monitor_task

    (output_dir / "monitoring.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in monitoring), encoding="utf-8")
    if len(results) != len(offered):
        invalid_reasons.append("request_conservation_failure")
    lag_p95 = float(np.percentile([r["send_lag_ms"] for r in results], 95)) if results else None
    if lag_p95 is not None and lag_p95 > 20:
        invalid_reasons.append("sender_lag_p95_over_20ms")
    if any(not r["metrics_valid"] for r in results):
        invalid_reasons.append("token_accounting_mismatch")
    if any(s.get("gateway", {}).get("log_dropped", 0) for s in monitoring):
        invalid_reasons.append("gateway_logs_dropped")
    manifest = {"run_id": run_id, "url": url, "model": model,
                "trace_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
                "prompts_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
                "load": trace["load"], "seed": trace["seed"], "offered": len(offered),
                "arrival_window_seconds": trace["duration_seconds"],
                "total_seconds_including_drain": total_seconds,
                "send_lag_p95_ms": lag_p95, "native_priority": native_priority,
                "config": config, "valid": not invalid_reasons,
                "invalid_reasons": invalid_reasons,
                "statuses": {s: sum(r["status"] == s for r in results)
                             for s in sorted(set(r["status"] for r in results))}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}), flush=True)
    return results, manifest


async def main_async(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    await run_trace(url=args.url, model=args.model, trace=trace, prompts=prompts,
                    output_dir=args.output, run_id=args.run_id, config=config,
                    native_priority=args.native_priority,
                    metrics_url=args.metrics_url, debug_url=args.debug_url,
                    deadline=args.deadline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="admission-7b")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--prompts", default="artifacts/traces/prompts.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--native-priority", action="store_true")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--debug-url")
    parser.add_argument("--deadline", type=float, default=60)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
