"""Isolated real-GPU gateway disconnect and reservation cleanup verification."""
import asyncio
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import yaml

from admission_lab.sse import SSEParser, content_event
from .matrix import Backend, metric_sum, stop_owned, wait_idle, wait_ready, warmup
from .probe import request_once
from .cancel_probe import probe as verify_native_cancel


async def execute():
    root = Path("artifacts/gateway_cancel_probe")
    root.mkdir(parents=True, exist_ok=False)
    engine = yaml.safe_load(Path("configs/frozen.yaml").read_text())
    prompts = json.loads(Path(engine["prompts"]).read_text())
    backend = Backend(engine, root)
    gateway = None
    handle = None
    record = {"verified": False, "method": "isolated gateway streaming close; reservation held before, zero after; backend running then idle"}
    try:
        await backend.start("fcfs", "isolated-cancel")
        await warmup(engine, prompts)
        await verify_native_cancel(Namespace(model=engine["model"], prompts=engine["prompts"],
                                             output="artifacts/cancellation_probe.json"))
        config = {**engine["gateway"], "backend_url": "http://127.0.0.1:8000",
                  "event_log": str(root / "gateway_events.jsonl")}
        path = root / "gateway.yaml"
        path.write_text(yaml.safe_dump(config))
        env = dict(os.environ)
        env.update(ADMISSION_CONFIG=str(path.resolve()), PYTHONPATH=str(Path("src").resolve()))
        handle = (root / "gateway.log").open("x")
        gateway = subprocess.Popen([sys.executable, "-m", "uvicorn", "admission_lab.app:create_app", "--factory",
                                    "--host", "127.0.0.1", "--port", "8080", "--workers", "1", "--no-access-log"],
                                   env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        await wait_ready("http://127.0.0.1:8080/health", gateway, timeout=30)
        async with httpx.AsyncClient(timeout=30) as client, httpx.AsyncClient(timeout=5) as diagnostic:
            request_id = f"gateway-cancel-{time.time_ns()}"
            record["request_id"] = request_id
            body = {"model": engine["model"], "prompt": prompts["S-0"], "max_tokens": 512,
                    "stream": True, "n": 1, "ignore_eos": True, "temperature": 0,
                    "return_token_ids": True, "request_id": request_id,
                    "stream_options": {"include_usage": True}}
            async with client.stream("POST", "http://127.0.0.1:8080/v1/completions", json=body,
                                     headers={"x-request-id": request_id}) as response:
                response.raise_for_status()
                parser, content = SSEParser(), False
                stream_iterator = response.aiter_bytes()
                async for chunk in stream_iterator:
                    for event in parser.feed(chunk):
                        if event == "[DONE]":
                            raise RuntimeError("Gateway probe finished before cancellation")
                        content |= bool(content_event(event)[1])
                    if content:
                        break
                if not content:
                    raise RuntimeError("Gateway probe did not produce content")
                observed = False
                for _ in range(20):
                    metrics = (await diagnostic.get("http://127.0.0.1:8000/metrics")).text
                    state = (await diagnostic.get("http://127.0.0.1:8080/debug/state")).json()
                    running = metric_sum(metrics, "vllm:num_requests_running")
                    if running is not None and running >= 1 and state["inflight"] >= 1 and state["reserved_cost_ms"] > 0:
                        observed = True
                        break
                    await asyncio.sleep(0.1)
                record.update(observed_running_and_reserved_before_close=observed, before_close_state=state)
                if not observed:
                    raise RuntimeError("No simultaneous engine running and gateway reservation observation")
                closed_at = time.perf_counter()
            await stream_iterator.aclose()
            while time.perf_counter() - closed_at < 10:
                state = (await diagnostic.get("http://127.0.0.1:8080/debug/state")).json()
                if state["queued"] == state["inflight"] == 0 and abs(state["reserved_cost_ms"]) <= 1e-6:
                    break
                await asyncio.sleep(0.05)
            else:
                raise RuntimeError("Gateway reservation did not clear")
            record["gateway_clear_after_close_seconds"] = time.perf_counter() - closed_at
            await wait_idle(timeout=10)
            record["backend_idle_after_close_seconds"] = time.perf_counter() - closed_at
            record["post_cancel_request"] = await request_once(client, "http://127.0.0.1:8080/v1/completions",
                                                               engine["model"], prompts["S-1"], 128,
                                                               f"post-gateway-cancel-{time.time_ns()}")
            await wait_idle()
            state = (await diagnostic.get("http://127.0.0.1:8080/debug/state")).json()
            record["final_gateway_state"] = state
            record["verified"] = (not any(state[k] for k in ("queued", "inflight", "paused", "log_dropped"))
                                  and abs(state["reserved_cost_ms"]) <= 1e-6
                                  and record["backend_idle_after_close_seconds"] < 10)
            if not record["verified"]:
                raise RuntimeError("Gateway cancellation final checks failed")
    except BaseException as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        Path("artifacts/gateway_cancellation_probe.json").write_text(json.dumps(record, indent=2))
        await stop_owned(gateway)
        if handle:
            handle.close()
        await backend.stop()
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(execute())
