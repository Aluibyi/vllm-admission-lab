"""Read-only target inventory; does not allocate GPUs or read credentials."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys


def command(args, timeout=25):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def main():
    gpu = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
                   "--format=csv"])
    namespace = command(["kubectl", "config", "view", "--minify", "-o", "jsonpath={..namespace}"])
    result = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": sys.version, "gpu": gpu, "namespace": namespace,
              "scope": "Read-only migration snapshot; no CUDA allocations or workload mutations."}
    pods = command(["kubectl", "get", "pods", "-n", "example", "-o", "json", "--request-timeout=15s"])
    if pods.get("returncode") == 0:
        result["pod_gpu_limits"] = [
            {"pod": p["metadata"]["name"], "containers": [
                {"name": c["name"], "image": c["image"],
                 "gpu_limit": c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu")}
                for c in p["spec"]["containers"]]}
            for p in json.loads(pods["stdout"])["items"]]
    else:
        result["pod_inventory_error"] = pods.get("stderr", pods.get("error"))
    target = Path("artifacts/target_environment_20261118.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
