"""Owned, logged development -> frozen paired trials -> isolated validation."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import yaml

from .matrix import stop_owned


class Pipeline:
    def __init__(self, formal_mode="parallel", resume_from=None):
        self.current = None
        self.state = {"started_unix": time.time(), "status": "running", "stages": []}
        self.path = Path("artifacts/pipeline_state.json")
        self.env = dict(os.environ)
        self.env.update({"PYTHONPATH": str(Path("src").resolve()), "OMP_NUM_THREADS": "4",
                         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
                         "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
        self.formal_mode = formal_mode
        self.resume_from = resume_from
        if resume_from:
            previous = json.loads(self.path.read_text())
            freeze_completed = any(s["name"] == "freeze" and s.get("returncode") == 0 for s in previous["stages"])
            if (resume_from not in {"formal_traces", "real_gateway_cancel_verification"} or previous["status"] != "failed"
                    or previous["current_stage"] != resume_from or not freeze_completed
                    or not Path("configs/frozen.yaml").is_file()
                    or any(Path("artifacts/formal").glob("runs/*/*"))):
                raise ValueError("Trace-stage resume requires completed freeze and no formal data")
            archive_name = "pipeline_before_trace_resume.json" if resume_from == "formal_traces" else "pipeline_before_gateway_probe_resume.json"
            with Path("artifacts/setup", archive_name).open("x") as handle:
                json.dump(previous, handle, indent=2)
            self.state = previous
            self.state.setdefault("resumptions", []).append({"unix_time": time.time(), "stage": resume_from,
                                                            "prior_error": previous.get("error")})
            self.state.update(status="running", error=None, finished_unix=None)

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2))
        temporary.replace(self.path)

    async def command(self, name, arguments):
        step = {"name": name, "started_unix": time.time(), "command": arguments}
        self.state["stages"].append(step)
        self.state["current_stage"] = name
        self.save()
        print(json.dumps({"pipeline_stage": name, "event": "start"}), flush=True)
        self.current = subprocess.Popen([sys.executable, *arguments], env=self.env, start_new_session=True)
        try:
            code = await asyncio.to_thread(self.current.wait)
            step.update(returncode=code, finished_unix=time.time())
            self.save()
            if code:
                raise RuntimeError(f"Stage failed: {name}, returncode={code}")
        finally:
            await stop_owned(self.current)
            self.current = None

    async def wait_initial(self):
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=3) as client:
            while time.monotonic() - started < 900:
                try:
                    response = await client.get("http://127.0.0.1:8000/v1/models")
                    if response.status_code == 200 and any(x["id"] == "admission-7b" for x in response.json()["data"]):
                        return
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
                await asyncio.sleep(1)
        raise TimeoutError("Initial backend did not become ready")

    async def stop_initial(self):
        path = Path("artifacts/setup/development_backend.pid")
        if not path.exists():
            return
        pid = int(path.read_text().strip())
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            return
        command = (proc / "cmdline").read_bytes()
        if b"bench.serve" not in command:
            return
        os.kill(pid, signal.SIGTERM)
        for _ in range(120):
            if not proc.exists():
                return
            status = (proc / "status").read_text()
            if "State:\tZ" in status:
                return
            await asyncio.sleep(0.5)
        raise TimeoutError("Owned initial backend wrapper did not stop")

    async def run(self):
        try:
            if not self.resume_from:
                self.state["current_stage"] = "wait_initial_backend"
                self.save()
                await self.wait_initial()
                await self.command("calibration_and_heldout", ["-m", "bench.probe"])
                await self.command("real_cancel_verification", ["-m", "bench.cancel_probe"])
                await self.command("reference_rate", ["-m", "bench.develop", "rate"])
            rate = json.loads(Path("artifacts/development/reference_rate.json").read_text())["reference_rate"]
            self.state["reference_rate"] = rate
            self.save()
            if not self.resume_from:
                await self.command("gateway_tuning_four_candidates", ["-m", "bench.develop", "gateway", "--rate", str(rate)])
                await self.stop_initial()
                await self.command("native_tuning_four_candidates", ["-m", "bench.develop", "native", "--rate", str(rate)])
                await self.command("freeze", ["-m", "bench.develop", "finalize", "--rate", str(rate)])
            await self.command("real_gateway_cancel_verification", ["-m", "bench.gateway_cancel_probe"])
            model_path = yaml.safe_load(Path("configs/engine.yaml").read_text())["model_path"]
            await self.command("formal_traces", ["-m", "bench.workload", "--tokenizer", model_path,
                                                 "--reference-rate", str(rate), "--duration", "90"])
            await self.command("source_snapshot", ["scripts/make_bundle.py"])
            module = "bench.matrix" if self.formal_mode == "single" else "bench.parallel"
            await self.command(f"formal_{self.formal_mode}_three_seeds", ["-m", module, "--frozen", "configs/frozen.yaml",
                                                                       "--output", "artifacts/formal"])
            await self.command("formal_evidence_audit", ["scripts/audit_evidence.py"])
            await self.command("formal_analysis", ["-m", "bench.analyze", "--root", "artifacts/formal", "--require-complete"])
            await self.command("separate_validation_trace", ["-m", "bench.matrix", "--frozen", "configs/validation.yaml",
                                                                  "--slot", "0", "--output", "artifacts/validation"])
            await self.command("validation_analysis", ["-m", "bench.analyze", "--root", "artifacts/validation",
                                                        "--frozen", "configs/validation.yaml", "--output", "reports/validation", "--require-complete"])
            self.state.update(status="completed", current_stage=None, finished_unix=time.time())
            self.save()
            print("PIPELINE_COMPLETE", flush=True)
        except BaseException as exc:
            self.state.update(status="failed", error=type(exc).__name__ + ": " + str(exc), finished_unix=time.time())
            self.save()
            raise
        finally:
            await self.stop_initial()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-mode", choices=["single", "parallel"], default="parallel")
    parser.add_argument("--resume-from", choices=["formal_traces", "real_gateway_cancel_verification"])
    args = parser.parse_args()
    async def run():
        task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await Pipeline(formal_mode=args.formal_mode, resume_from=args.resume_from).run()
    asyncio.run(run())


if __name__ == "__main__":
    main()
