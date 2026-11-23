"""Run paired seeds on separate GPUs; policies stay paired on the same GPU."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from .matrix import stop_owned


async def execute(args):
    config = yaml.safe_load(Path(args.frozen).read_text())
    slots = config["gpu_slots"]
    all_seeds = [seed for slot in slots for seed in slot["seeds"]]
    if len(all_seeds) != len(set(all_seeds)) or set(all_seeds) != set(config["seeds"]):
        raise ValueError("Slots must partition the declared seeds without overlap")
    if len({slot["gpu_uuid"] for slot in slots}) != len(slots):
        raise ValueError("GPU assignment overlaps")
    ports = [slot[key] for slot in slots for key in ("backend_port", "gateway_port")]
    if len(ports) != len(set(ports)):
        raise ValueError("Port assignment overlaps")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    children = []
    handles = []
    started = time.time()
    try:
        for index, slot in enumerate(slots):
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path("src").resolve())
            handle = (root / f"slot-{index}.log").open("w")
            handles.append(handle)
            command = [sys.executable, "-m", "bench.matrix", "--frozen", args.frozen,
                       "--output", args.output, "--slot", str(index)]
            if getattr(args, "resume_from", None):
                command.extend(["--resume-from", args.resume_from])
            if getattr(args, "dry_run", False):
                command.append("--dry-run")
            process = subprocess.Popen(command,
                                       env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
        codes = await asyncio.gather(*(asyncio.to_thread(p.wait) for p in children))
        result = {"started_unix": started, "finished_unix": time.time(),
                  "slots": slots, "pids": [p.pid for p in children], "returncodes": codes,
                  "comparison_unit": "same-GPU paired policies within seed",
                  "shared_host_cpu_interference_possible": True}
        result["dry_run"] = getattr(args, "dry_run", False)
        (root / "parallel_manifest.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if any(codes):
            raise RuntimeError("One or more GPU slots failed; preserve logs and partial runs")
    finally:
        await asyncio.gather(*(stop_owned(p) for p in children), return_exceptions=True)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default="configs/frozen.yaml")
    parser.add_argument("--output", default="artifacts/formal")
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true")
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
