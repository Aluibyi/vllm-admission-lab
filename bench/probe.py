"""Real backend probes: output accounting, low-load cost model and SLOs."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
import numpy as np
from scipy.optimize import nnls
import yaml

from admission_lab.app import DEFAULT
from admission_lab.sse import SSEParser, content_event


async def request_once(client, url, model, tokens, output, request_id, priority=0):
    start = time.perf_counter()
    first = None
    usage = None
    done = False
    finish = None
    body = {"model": model, "prompt": tokens, "max_tokens": output, "temperature": 0,
            "top_p": 1, "stream": True, "n": 1, "ignore_eos": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True,
            "request_id": request_id, "priority": priority}
    parser = SSEParser()
    async with client.stream("POST", url, json=body) as response:
        if response.status_code >= 400:
            text = (await response.aread()).decode(errors="replace")[:1000]
            raise RuntimeError(f"Backend HTTP {response.status_code}: {text}")
        async for chunk in response.aiter_bytes():
            for event in parser.feed(chunk):
                if event == "[DONE]":
                    done = True
                    continue
                value, content = content_event(event)
                if content and first is None:
                    first = time.perf_counter()
                if value and value.get("usage"):
                    usage = value["usage"]
                for choice in (value or {}).get("choices", []):
                    finish = choice.get("finish_reason") or finish
    terminal = time.perf_counter()
    if not done or first is None or not usage:
        raise RuntimeError("Missing DONE, content or usage")
    if usage["completion_tokens"] != output or usage["prompt_tokens"] != len(tokens):
        raise RuntimeError(f"Token accounting mismatch: {usage}")
    return {"prompt_tokens": len(tokens), "max_tokens": output,
            "ttft_ms": (first - start) * 1000, "e2e_ms": (terminal - start) * 1000,
            "usage": usage, "finish_reason": finish}


async def calibrate(args):
    prompts = json.loads(Path(args.prompts).read_text())
    rows = []
    heldout = []
    async with httpx.AsyncClient(timeout=180) as client:
        await request_once(client, args.url, args.model, prompts["S-0"], 128, "warm-short")
        await request_once(client, args.url, args.model, prompts["P-0"], 128, "warm-long")
        for repeat in range(2):
            for p in (512, 4096, 8192):
                for o in (128, 512):
                    row = await request_once(client, args.url, args.model,
                                             prompts[f"P-{repeat}"][:p], o,
                                             f"calib-{repeat}-{p}-{o}")
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        for repeat in range(2):
            for p, o in ((1024, 256), (6144, 384)):
                row = await request_once(client, args.url, args.model, prompts[f"P-{repeat+3}"][:p], o,
                                         f"heldout-{repeat}-{p}-{o}")
                heldout.append(row)
    matrix = np.array([[1, r["prompt_tokens"], r["max_tokens"]] for r in rows], dtype=float)
    target = np.array([r["e2e_ms"] for r in rows])
    coefficients, _ = nnls(matrix, target)
    coefficients = np.maximum(coefficients, [0.01, 0.00001, 0.00001])
    predicted = matrix @ coefficients
    errors = np.abs(predicted - target) / target
    heldout_matrix = np.array([[1, r["prompt_tokens"], r["max_tokens"]] for r in heldout], dtype=float)
    heldout_target = np.array([r["e2e_ms"] for r in heldout])
    heldout_errors = np.abs(heldout_matrix @ coefficients - heldout_target) / heldout_target
    classes = {"S": (512, 128), "P": (8192, 128), "D": (512, 512)}
    slos = {}
    # Predeclared experimental floors, fixed before any GPU calibration results.
    # This avoids using a few-ms idle transport latency as a production-like SLO.
    floors = {"S": {"ttft_ms": 500, "e2e_ms": 5000},
              "P": {"ttft_ms": 2000, "e2e_ms": 8000},
              "D": {"ttft_ms": 500, "e2e_ms": 15000}}
    for cls, shape in classes.items():
        subset = [r for r in rows if (r["prompt_tokens"], r["max_tokens"]) == shape]
        slos[cls] = {name: max(floors[cls][name], 2 * float(np.percentile([r[name] for r in subset], 95)))
                     for name in ("ttft_ms", "e2e_ms")}
    worst = float(np.dot([1, 8192, 512], coefficients))
    config = {**DEFAULT, "cost_intercept_ms": float(coefficients[0]),
              "prompt_ms_per_token": float(coefficients[1]),
              "output_ms_per_token": float(coefficients[2]),
              "work_budget_ms": 2 * worst,
              "tau_ms": max(1, slos["S"]["ttft_ms"] / 2),
              "aging_threshold_ms": max(s["ttft_ms"] for s in slos.values()),
              "queue_timeout_ms": max(30000, max(s["e2e_ms"] for s in slos.values()) * 4),
              "cancel_path_verified": False}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"samples": rows, "heldout_samples": heldout, "coefficients": coefficients.tolist(),
              "median_absolute_percentage_error": float(np.median(errors)),
              "heldout_median_absolute_percentage_error": float(np.median(heldout_errors)),
              "heldout_max_absolute_percentage_error": float(np.max(heldout_errors)),
              "max_absolute_percentage_error": float(np.max(errors)),
              "slo_ms": slos, "slo_floors_ms": floors,
              "slo_definition": "max(predeclared experimental floor, 2x per-class low-load empirical P95); two samples per shape",
              "client_deadline_seconds": max(30, 4 * max(s["e2e_ms"] for s in slos.values()) / 1000),
              "config": config}
    (output / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "gateway.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"samples", "config"}}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="admission-7b")
    parser.add_argument("--prompts", default="artifacts/traces/prompts.json")
    parser.add_argument("--output", default="artifacts/calibration")
    asyncio.run(calibrate(parser.parse_args()))


if __name__ == "__main__":
    main()
