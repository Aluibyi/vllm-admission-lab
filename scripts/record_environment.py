import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def command(args):
    try:
        value = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return {"returncode": value.returncode, "stdout": value.stdout,
                "stderr": value.stderr}
    except Exception as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)}


def main():
    root = Path("artifacts")
    root.mkdir(exist_ok=True)
    packages = {}
    for name in ("vllm", "torch", "transformers", "fastapi", "uvicorn", "httpx",
                 "numpy", "scipy", "pytest", "pytest-asyncio", "modelscope"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"recorded_unix": time.time(), "platform": platform.platform(),
              "python": sys.version, "executable": sys.executable,
              "cpu_count": os.cpu_count(), "packages": packages,
              "gpu": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,power.limit", "--format=csv"]),
              "git": command(["git", "rev-parse", "HEAD"]),
              "disk": command(["df", "-h", "."])}
    import torch
    result["torch_cuda"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_arch_list"] = torch.cuda.get_arch_list()
    if result["cuda_available"]:
        # The launching script selects exactly one GPU by UUID.
        value = torch.ones(8, device="cuda")
        result["cuda_smoke_sum"] = value.sum().item()
        result["cuda_device_count"] = torch.cuda.device_count()
        result["device_capability"] = torch.cuda.get_device_capability(0)
    (root / "environment.json").write_text(json.dumps(result, indent=2))
    freeze = command([sys.executable, "-m", "pip", "freeze"])
    if freeze.get("returncode") != 0:
        freeze = command([".bootstrap/bin/uv", "pip", "freeze", "--python", sys.executable])
    (root / "requirements.remote.txt").write_text(freeze.get("stdout", ""))
    portable = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                       if d.metadata.get("Name")})
    (root / "requirements.remote.lock.txt").write_text("\n".join(portable) + "\n")
    help_result = command([str(Path(sys.executable).parent / "vllm"), "serve", "--help=all"])
    (root / "vllm_serve_help.txt").write_text(help_result.get("stdout", ""))
    flags = ["--max-num-queued-reqs", "--max-num-queued-tokens", "--max-num-active-seqs",
             "--scheduling-policy", "--disable-prefix-caching", "--enable-chunked-prefill"]
    (root / "capability_matrix.json").write_text(json.dumps({
        "vllm": packages["vllm"], "serve_help_returncode": help_result.get("returncode"),
        "flags": {flag: flag in help_result.get("stdout", "") for flag in flags}}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
