import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import httpx
import pytest
import yaml

from bench.replay import run_trace


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def ready(port, process):
    async with httpx.AsyncClient(timeout=1) as client:
        for _ in range(150):
            if process.poll() is not None:
                raise RuntimeError("Fixture service exited")
            try:
                if (await client.get(f"http://127.0.0.1:{port}/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.03)
    raise TimeoutError("Fixture service not ready")


async def test_real_http_replay_and_disconnect(tmp_path):
    backend_port, gateway_port = free_port(), free_port()
    while gateway_port == backend_port:
        gateway_port = free_port()
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    config = {"backend_url": f"http://127.0.0.1:{backend_port}",
              "event_log": str(tmp_path / "gateway.jsonl"), "cancel_path_verified": True}
    config_path = tmp_path / "gateway.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env["ADMISSION_CONFIG"] = str(config_path)
    processes = []
    handles = []
    try:
        for module, port, factory in (("tests.fake_backend:app", backend_port, False),
                                      ("admission_lab.app:create_app", gateway_port, True)):
            log = (tmp_path / f"{port}.log").open("w")
            handles.append(log)
            command = [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1",
                       "--port", str(port), "--no-access-log"] + (["--factory"] if factory else [])
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                        stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append(process)
            await ready(port, process)
        trace = {"duration_seconds": 0.2, "load": "fixture", "seed": 1,
                 "requests": [{"index": i, "scheduled_seconds": i * 0.03,
                               "class": "S", "prompt_key": "p", "prompt_tokens": 2,
                               "max_tokens": 8} for i in range(3)]}
        rows, manifest = await run_trace(url=f"http://127.0.0.1:{gateway_port}/v1/completions",
                    model="admission-7b", trace=trace, prompts={"p": [1, 2]},
                    output_dir=tmp_path / "fixture-run", run_id="fixture-only",
                    debug_url=f"http://127.0.0.1:{gateway_port}/debug/state", deadline=5)
        assert len(rows) == manifest["offered"] == 3
        assert all(r["status"] == "succeeded" and r["actual_output_tokens"] == 8 for r in rows)
        # This simulated-backend integration test verifies protocol/lifecycle,
        # not the scheduling latency of the developer's OS. The production
        # 20ms validity threshold remains enforced by the replay implementation.
        assert set(manifest["invalid_reasons"]) <= {"sender_lag_p95_over_20ms"}
        assert ("sender_lag_p95_over_20ms" in manifest["invalid_reasons"]) == (manifest["send_lag_p95_ms"] > 20)
        async with httpx.AsyncClient(timeout=5) as client:
            payload = {"model": "admission-7b", "prompt": [1, 2], "max_tokens": 512,
                       "stream": True, "request_id": "disconnect-fixture"}
            async with client.stream("POST", f"http://127.0.0.1:{gateway_port}/v1/completions", json=payload) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        break
            for _ in range(100):
                state = (await client.get(f"http://127.0.0.1:{gateway_port}/debug/state")).json()
                if state["inflight"] == 0:
                    break
                await asyncio.sleep(0.01)
            assert state["inflight"] == 0 and state["reserved_cost_ms"] == 0
    finally:
        for process in reversed(processes):
            process.terminate()
            await asyncio.to_thread(process.wait, 10)
        for handle in handles:
            handle.close()
