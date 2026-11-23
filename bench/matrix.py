"""Sequential, process-owned GPU experiments. Never touches unrelated services."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import site
import socket
import subprocess
import sys
import time

import httpx
import yaml

from .probe import request_once
from .replay import run_trace


def metric_sum(text, prefix):
    values = []
    for line in text.splitlines():
        if line.startswith(prefix + "{") or line.startswith(prefix + " "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values) if values else None


async def wait_ready(url, process, timeout=900):
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=3) as client:
        while time.monotonic() - start < timeout:
            if process.poll() is not None:
                raise RuntimeError(f"Service exited with {process.returncode}: {url}")
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"Service did not become ready: {url}")


async def wait_idle(timeout=120, port=8000):
    start = time.monotonic()
    zeros = 0
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() - start < timeout:
            response = await client.get(f"http://127.0.0.1:{port}/metrics")
            running = metric_sum(response.text, "vllm:num_requests_running")
            waiting = metric_sum(response.text, "vllm:num_requests_waiting")
            if running is None or waiting is None:
                raise RuntimeError("Missing backend idle metrics; cannot verify clean experiment boundary")
            zeros = zeros + 1 if running == 0 and waiting == 0 else 0
            if zeros >= 2:
                return
            await asyncio.sleep(0.5)
    raise TimeoutError("Backend did not drain; stop instead of contaminating the next run")


async def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.to_thread(process.wait, 45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        await asyncio.to_thread(process.wait, 10)


class Backend:
    def __init__(self, frozen, root, port=8000):
        self.frozen = frozen
        self.root = Path(root)
        self.process = None
        self.handle = None
        self.port = port

    async def start(self, group, sequence):
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                check.bind(("127.0.0.1", self.port))
            except OSError as exc:
                raise RuntimeError(f"Backend port {self.port} is occupied; will not use or stop another service") from exc
        native = self.frozen["native"] if group == "native" else {"policy": "fcfs", "extra_args": []}
        executable = str(Path(sys.executable).parent / "vllm")
        command = [executable, "serve", self.frozen["model_path"],
                   "--served-model-name", self.frozen["model"],
                   "--host", "127.0.0.1", "--port", str(self.port),
                   "--scheduling-policy", native["policy"],
                   *self.frozen["backend_args"], *native.get("extra_args", [])]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.frozen["gpu_uuid"]
        env["OMP_NUM_THREADS"] = str(self.frozen.get("cpu_threads", 8))
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
        env["MAX_JOBS"] = "4"
        env["VLLM_NO_USAGE_STATS"] = "1"
        env["DO_NOT_TRACK"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        # NVIDIA documents a CUDA13 tilelang/NVCC vs Torch header mismatch
        # in FlashInfer's sampler JIT. Greedy benchmark uses native sampler;
        # this common setting is identical for every experimental strategy.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = str(self.frozen.get("cpu_threads", 4))
        binary_paths = [str(Path(sys.executable).parent)]
        # Prefer the toolkit shipped with this CUDA13 Python environment over
        # a base image's CUDA12 compiler. The image still supplies GCC/headers.
        cuda_candidates = []
        for base in site.getsitepackages():
            cuda_candidates.extend([Path(base) / "nvidia" / "cu13",
                                    Path(base) / "nvidia" / "cuda_nvcc"])
        cuda_candidates.extend(Path(env[k]) for k in ("CUDA_HOME", "CUDA_PATH") if env.get(k))
        for candidate in cuda_candidates:
            if (candidate / "bin" / "nvcc").is_file():
                env["CUDA_HOME"] = str(candidate)
                env["CUDA_PATH"] = str(candidate)
                binary_paths.append(str(candidate / "bin"))
                libraries = [str(candidate / name) for name in ("lib64", "lib")
                             if (candidate / name).is_dir()]
                old_libraries = [env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []
                env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + old_libraries)
                break
        env["PATH"] = os.pathsep.join(binary_paths + [env.get("PATH", "")])
        log_dir = self.root / "backend_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{sequence}-{group}.log"
        self.handle = log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, env=env, stdout=self.handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        (log.with_suffix(".json")).write_text(json.dumps({"pid": self.process.pid,
                                                          "command": command, "gpu_uuid": env["CUDA_VISIBLE_DEVICES"],
                                                          "cuda_home": env.get("CUDA_HOME"),
                                                          "flashinfer_sampler": env["VLLM_USE_FLASHINFER_SAMPLER"],
                                                          "ninja": shutil.which("ninja", path=env["PATH"])}, indent=2))
        print(json.dumps({"event": "backend_start", "group": group, "pid": self.process.pid,
                          "log": str(log)}), flush=True)
        await wait_ready(f"http://127.0.0.1:{self.port}/health", self.process)
        return self.process

    async def stop(self):
        await stop_owned(self.process)
        self.process = None
        if self.handle:
            self.handle.close()


async def warmup(frozen, prompts, port=8000):
    async with httpx.AsyncClient(timeout=180) as client:
        for index, (cls, output) in enumerate((("S", 128), ("P", 128), ("D", 512))):
            await request_once(client, f"http://127.0.0.1:{port}/v1/completions", frozen["model"],
                               prompts[f"{cls}-0"], output, f"warmup-{time.time_ns()}-{index}")
    await wait_idle(port=port)


async def execute(args):
    frozen_path = Path(args.frozen)
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    all_seeds = frozen.get("seeds", [1101, 1102, 1103])
    slot = args.slot
    slots = frozen.get("gpu_slots", [])
    backend_port, gateway_port = 8000, 8080
    selected_seeds = all_seeds
    if slot is not None:
        resource = slots[slot]
        frozen["gpu_uuid"] = resource["gpu_uuid"]
        backend_port = resource["backend_port"]
        gateway_port = resource["gateway_port"]
        selected_seeds = resource["seeds"]
    if not frozen["gateway"]["cancel_path_verified"]:
        raise ValueError("Confirm real backend cancellation before formal experiments")
    prompts = json.loads(Path(frozen["prompts"]).read_text())
    root = Path(args.output)
    resume_from = getattr(args, "resume_from", None)
    continuation = None
    if resume_from:
        from .checkpoint import plan_continuation
        if root.resolve() == Path(resume_from).resolve():
            raise ValueError("Continuation must use a separate output root")
        continuation = plan_continuation(resume_from, frozen_path)
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    source_root = Path(resume_from) / "provenance" if resume_from else Path("artifacts")
    for source in (source_root / name for name in
                   ("source.tar.gz", "source_manifest.json", "model_manifest.json", "environment.json")):
        path = Path(source)
        if path.exists():
            destination = provenance / path.name
            if destination.exists():
                if destination.read_bytes() != path.read_bytes():
                    raise ValueError(f"Provenance changed inside an existing experiment: {destination}")
                continue
            temporary = provenance / f"{path.name}.{os.getpid()}.tmp"
            shutil.copyfile(path, temporary)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != path.read_bytes():
                    raise ValueError(f"Concurrent provenance differs: {destination}")
            finally:
                temporary.unlink()
    suffix = f"slot-{slot}" if slot is not None else "single"
    execution_sha256 = None
    if continuation:
        # The original benchmark lineage stays intact; archive the exact
        # continuation orchestration separately and expose all changed files.
        execution = root / f"execution_source-{suffix}"
        if execution.exists():
            raise ValueError(f"Continuation output already exists: {execution}")
        execution.mkdir()
        original = json.loads((source_root / "source_manifest.json").read_text())
        original_hashes = {item["path"]: item["sha256"] for item in original["files"]}
        current = []
        project = Path.cwd()
        sources = [p for directory in ("src", "bench", "scripts", "configs")
                   for p in (project / directory).rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".yml"}]
        sources.append(project / "pyproject.toml")
        for path in sorted(sources):
            relative = path.relative_to(project)
            destination = execution / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            current.append({"path": relative.as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        changed = [p["path"] for p in current if p["path"] in original_hashes
                   and p["sha256"] != original_hashes[p["path"]]]
        if any(name not in {"bench/matrix.py", "bench/parallel.py", "bench/analyze.py"} for name in changed):
            raise ValueError(f"Frozen strategy or measurement code changed: {changed}")
        snapshot = {"files": current, "changed_existing_files": changed,
                    "note": "Only orchestration/analysis changes allowed; strategy and replay remain frozen."}
        snapshot_path = execution / "manifest.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
        execution_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        continuation["execution_source_manifest_sha256"] = execution_sha256
        (root / f"continuation-{suffix}.json").write_text(json.dumps(continuation, indent=2))
    (root / f"frozen_config-{suffix}.yaml").write_text(frozen_path.read_text(encoding="utf-8"), encoding="utf-8")
    backend = Backend(frozen, root / suffix, port=backend_port)
    planned = []
    for repetition, seed in enumerate(all_seeds):
        if seed not in selected_seeds:
            continue
        groups = ["fcfs", "native"] if repetition % 2 == 0 else ["native", "fcfs"]
        for group in groups:
            strategies = ["B0", "B2", "P"] if group == "fcfs" else ["B1"]
            jobs = [(s, load, seed) for s in strategies
                    for load in frozen.get("loads", ["light", "near_capacity", "burst"])]
            random.Random(9000 + seed + (group == "native")).shuffle(jobs)
            if continuation:
                jobs = [job for job in jobs if f"{job[0]}-{job[1]}-{job[2]}" in continuation["missing"]]
            if not jobs:
                continue
            planned.append({"repetition": repetition, "seed": seed, "group": group, "jobs": jobs})
    (root / f"run_order-{suffix}.json").write_text(json.dumps(planned, indent=2))
    if getattr(args, "dry_run", False):
        print(json.dumps({"status": "prepared_not_executed", "slot": slot,
                          "selected_jobs": [job for group in planned for job in group["jobs"]]}), flush=True)
        return
    try:
        for index, item in enumerate(planned):
            await backend.start(item["group"], index)
            await warmup(frozen, prompts, port=backend_port)
            for strategy, load, seed in item["jobs"]:
                run_id = f"{strategy}-{load}-{seed}"
                run_root = root / "runs" / run_id
                if (run_root / "manifest.json").exists():
                    raise ValueError(f"Existing result: {run_root}; use a new output root")
                trace = json.loads((Path(frozen["trace_dir"]) / f"{load}-{seed}.json").read_text())
                gateway = None
                gateway_handle = None
                try:
                    await wait_idle(port=backend_port)
                    if strategy in {"B2", "P"}:
                        config = {**frozen["gateway"], "backend_url": f"http://127.0.0.1:{backend_port}",
                                  "policy": "fcfs" if strategy == "B2" else "cost_aging",
                                  "event_log": str(run_root / "gateway_events.jsonl")}
                        run_root.mkdir(parents=True, exist_ok=True)
                        config_path = run_root / "gateway.yaml"
                        config_path.write_text(yaml.safe_dump(config))
                        env = dict(os.environ)
                        env["ADMISSION_CONFIG"] = str(config_path.resolve())
                        env["PYTHONPATH"] = str(Path("src").resolve())
                        gateway_handle = (run_root / "gateway_process.log").open("w")
                        gateway = subprocess.Popen(
                            [sys.executable, "-m", "uvicorn", "admission_lab.app:create_app", "--factory",
                             "--host", "127.0.0.1", "--port", str(gateway_port), "--workers", "1", "--no-access-log"],
                            env=env, stdout=gateway_handle, stderr=subprocess.STDOUT, start_new_session=True)
                        await wait_ready(f"http://127.0.0.1:{gateway_port}/health", gateway, timeout=30)
                    rows, manifest = await run_trace(
                        url=f"http://127.0.0.1:{gateway_port if gateway else backend_port}/v1/completions",
                        model=frozen["model"], trace=trace, prompts=prompts,
                        output_dir=run_root, run_id=run_id, config=config if gateway else frozen["gateway"],
                        native_priority=strategy == "B1" and frozen["native"]["policy"] == "priority",
                        metrics_url=f"http://127.0.0.1:{backend_port}/metrics",
                        debug_url=f"http://127.0.0.1:{gateway_port}/debug/state" if gateway else None,
                        deadline=frozen["client_deadline_seconds"])
                    if gateway:
                        async with httpx.AsyncClient(timeout=5) as client:
                            state = (await client.get(f"http://127.0.0.1:{gateway_port}/debug/state")).json()
                        (run_root / "final_gateway_state.json").write_text(json.dumps(state, indent=2))
                        if state["queued"] or state["inflight"] or abs(state["reserved_cost_ms"]) > 1e-6 or state["log_dropped"] or state["paused"]:
                            manifest["valid"] = False
                            manifest["invalid_reasons"].append("gateway_not_clean_at_end")
                    manifest["frozen_config_sha256"] = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
                    manifest["gpu_uuid"] = frozen["gpu_uuid"]
                    manifest["slot"] = slot
                    manifest["concurrent_gpu_trials"] = frozen.get("concurrent_gpu_trials", slot is not None)
                    if continuation:
                        manifest["continuation"] = {"previous_root": str(Path(resume_from).resolve()),
                                                    "execution_source_manifest_sha256": execution_sha256,
                                                    "session_boundary_may_affect_results": True}
                    for label, source in (("source_manifest", "artifacts/source_manifest.json"),
                                          ("model_manifest", "artifacts/model_manifest.json"),
                                          ("environment", "artifacts/environment.json")):
                        path = provenance / Path(source).name
                        manifest[label + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
                    await wait_idle(port=backend_port)
                    (run_root / "backend_drained.json").write_text(json.dumps({"verified": True, "unix_time": time.time()}))
                    if not manifest["valid"]:
                        raise RuntimeError(f"Invalid measured run {run_id}: {manifest['invalid_reasons']}; evidence retained")
                finally:
                    await stop_owned(gateway)
                    if gateway_handle:
                        gateway_handle.close()
            await backend.stop()
    finally:
        await backend.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default="configs/frozen.yaml")
    parser.add_argument("--output", default="artifacts/formal")
    parser.add_argument("--slot", type=int)
    parser.add_argument("--resume-from", help="Verify prior evidence and run only missing cases in a new root")
    parser.add_argument("--dry-run", action="store_true", help="Validate and save the plan without starting GPU processes")
    args = parser.parse_args()
    async def run():
        task = asyncio.current_task()
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        elif not args.dry_run:
            raise RuntimeError("GPU process ownership requires Linux; Windows supports --dry-run only")
        await execute(args)
    asyncio.run(run())


if __name__ == "__main__":
    main()
