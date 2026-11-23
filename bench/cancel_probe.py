"""Verify real cancellation observationally on an isolated long generation."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx

from admission_lab.sse import SSEParser, content_event
from .matrix import metric_sum, wait_idle
from .probe import request_once


async def probe(args):
    prompts = json.loads(Path(args.prompts).read_text())
    await wait_idle()
    record = {"request_id": f"cancel-probe-{time.time_ns()}", "verified": False,
              "method": "isolated streaming connection close; running before and idle after",
              "samples": []}
    async with httpx.AsyncClient(timeout=60) as client, httpx.AsyncClient(timeout=5) as diagnostic:
        body = {"model": args.model, "prompt": prompts["S-0"], "max_tokens": 8192,
                "stream": True, "n": 1, "ignore_eos": True, "temperature": 0,
                "return_token_ids": True, "request_id": record["request_id"],
                "stream_options": {"include_usage": True}}
        async with client.stream("POST", "http://127.0.0.1:8000/v1/completions", json=body) as response:
            response.raise_for_status()
            parser = SSEParser()
            got_content = False
            for_check = time.perf_counter()
            # Keep the generator alive while diagnostic requests run. Dropping
            # a temporary iterator after break can close the transport early.
            stream_iterator = response.aiter_bytes()
            async for chunk in stream_iterator:
                for event in parser.feed(chunk):
                    if event == "[DONE]":
                        raise RuntimeError("Probe completed before cancellation")
                    if content_event(event)[1]:
                        got_content = True
                if got_content:
                    break
            if not got_content:
                raise RuntimeError("No generated content for cancellation probe")
            # Wait for an observed active metric rather than assuming HTTP
            # response receipt proves a currently running engine request.
            observed_running = False
            for _ in range(20):
                metrics = (await diagnostic.get("http://127.0.0.1:8000/metrics")).text
                running = metric_sum(metrics, "vllm:num_requests_running")
                record["samples"].append({"phase": "before_close", "running": running,
                                           "since_first_seconds": time.perf_counter() - for_check})
                if running is not None and running >= 1:
                    observed_running = True
                    break
                await asyncio.sleep(0.5)
            record["observed_running_before_close"] = observed_running
            closed_at = time.perf_counter()
        await stream_iterator.aclose()
        await wait_idle(timeout=20)
        record["idle_after_close_seconds"] = time.perf_counter() - closed_at
        check = await request_once(client, "http://127.0.0.1:8000/v1/completions", args.model,
                                   prompts["S-1"], 128, "post-cancel-valid-request")
        record["post_cancel_request"] = check
        record["verified"] = observed_running and record["idle_after_close_seconds"] < 10
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    if not record["verified"]:
        raise RuntimeError("Real backend cancellation was not established")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="admission-7b")
    parser.add_argument("--prompts", default="artifacts/traces/prompts.json")
    parser.add_argument("--output", default="artifacts/cancellation_probe.json")
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()
