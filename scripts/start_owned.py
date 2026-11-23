"""Start only this experiment's owned backend or pipeline with detached logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("target", choices=["backend", "pipeline"])
parser.add_argument("--resume-from", choices=["formal_traces", "real_gateway_cancel_verification"])
args = parser.parse_args()
root = Path.cwd()
env = dict(os.environ)
env.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
if args.target == "backend":
    command = [sys.executable, "-m", "bench.serve", "--config", "configs/engine.yaml",
               "--logs", "artifacts/development", "--tag", "initial"]
    pidfile, logfile = "development_backend.pid", "initial_backend.log"
else:
    command = [sys.executable, "-m", "bench.pipeline", "--formal-mode", "single"]
    if args.resume_from:
        command.extend(["--resume-from", args.resume_from])
    pidfile, logfile = "pipeline.pid", "pipeline.log"
setup = root / "artifacts/setup"
setup.mkdir(parents=True, exist_ok=True)
if (setup / pidfile).exists():
    if not args.resume_from or args.target != "pipeline":
        raise RuntimeError("Refusing to start another process with an existing ownership record")
    previous_pid = int((setup / pidfile).read_text())
    previous_process = Path(f"/proc/{previous_pid}")
    if previous_process.exists() and "State:\tZ" not in (previous_process / "status").read_text():
        raise RuntimeError("Previous pipeline process is still alive")
    (setup / pidfile).rename(setup / (pidfile + f".before-{args.resume_from}-resume"))
with (setup / logfile).open("a" if args.resume_from else "x") as handle:
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
(setup / pidfile).write_text(str(process.pid))
print(json.dumps({"target": args.target, "pid": process.pid, "log": str(setup / logfile)}), flush=True)
