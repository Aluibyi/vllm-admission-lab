"""Small predeclared development budget; final traces use different seeds."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import yaml

from .matrix import Backend, stop_owned, wait_idle, wait_ready, warmup
from .metrics import summarize
from .replay import run_trace
from .workload import build_trace


async def development_run(args, config, name, rate, duration=20, priority=False, gateway=False):
    prompts = json.loads(Path(args.prompts).read_text())
    trace = build_trace(rate, duration, 2201, "calibration")
    rows, manifest = await run_trace(
        url=f"http://127.0.0.1:{8080 if gateway else 8000}/v1/completions",
        model=args.model, trace=trace, prompts=prompts,
        output_dir=Path(args.output) / name, run_id=name, config=config,
        native_priority=priority, metrics_url="http://127.0.0.1:8000/metrics",
        debug_url="http://127.0.0.1:8080/debug/state" if gateway else None,
        deadline=args.deadline)
    calibration = json.loads(Path(args.calibration).read_text())
    stats = summarize(rows, calibration["slo_ms"], manifest["arrival_window_seconds"],
                      manifest["total_seconds_including_drain"])
    await wait_idle(timeout=180)
    return {"name": name, "rate": rate, "config": config,
            "manifest": manifest, "stats": stats}


async def choose_rate(args):
    config = yaml.safe_load(Path(args.gateway_config).read_text())
    candidates = []
    best = None
    upper = None
    for rate in (1, 2, 4, 8, 16, 32):
        result = await development_run(args, config, f"rate-{rate}", rate)
        candidates.append(result)
        m = result["manifest"]
        stable = (m["valid"] and result["stats"]["failure_fraction"] <= 0.02 and
                  m["total_seconds_including_drain"] <= 1.35 * m["arrival_window_seconds"])
        if stable:
            best = rate
        else:
            upper = rate
            break
    if best is None:
        raise RuntimeError("No stable valid rate; fix protocol/environment or extend low-rate calibration")
    if upper and len(candidates) < 6:
        midpoint = (best + upper) / 2
        result = await development_run(args, config, f"rate-refine-{midpoint}", midpoint)
        candidates.append(result)
        if (result["manifest"]["valid"] and result["stats"]["failure_fraction"] <= 0.02
                and result["manifest"]["total_seconds_including_drain"] <= 27):
            best = midpoint
    output = {"reference_rate": best, "definition": "highest short probe with <=2% failure and <=35% drain overhead",
              "attempts": candidates}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "reference_rate.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"reference_rate": best}), flush=True)


async def choose_gateway(args):
    calibration = json.loads(Path(args.calibration).read_text())
    base = calibration["config"]
    if not json.loads(Path("artifacts/cancellation_probe.json").read_text())["verified"]:
        raise ValueError("Cancellation probe did not pass")
    base["cancel_path_verified"] = True
    worst = base["work_budget_ms"] / 2
    attempts = []
    for index, (k, multiple) in enumerate(((8, 1), (16, 2), (32, 4), (64, 8))):
        root = Path(args.output) / f"gateway-{index}"
        root.mkdir(parents=True, exist_ok=True)
        config = {**base, "max_inflight": k, "work_budget_ms": multiple * worst,
                  "event_log": str(root / "gateway_events.jsonl")}
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        env = dict(os.environ)
        env["ADMISSION_CONFIG"] = str(config_path.resolve())
        env["PYTHONPATH"] = str(Path("src").resolve())
        with (root / "process.log").open("w") as handle:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "admission_lab.app:create_app", "--factory",
                                        "--host", "127.0.0.1", "--port", "8080", "--workers", "1", "--no-access-log"],
                                       env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                await wait_ready("http://127.0.0.1:8080/health", process, timeout=30)
                result = await development_run(args, config, f"gateway-{index}/run", args.rate * 1.1,
                                               gateway=True)
                async with httpx.AsyncClient(timeout=5) as client:
                    state = (await client.get("http://127.0.0.1:8080/debug/state")).json()
                result["final_gateway_state"] = state
                if state["inflight"] or state["queued"] or state["paused"] or state["log_dropped"]:
                    result["manifest"]["valid"] = False
                attempts.append(result)
            finally:
                await stop_owned(process)
    valid = [r for r in attempts if r["manifest"]["valid"]]
    if not valid:
        raise RuntimeError("All gateway candidates invalid")
    best = max(valid, key=lambda r: (r["stats"]["cohort_goodput"], -r["stats"]["failure_fraction"]))
    (Path(args.output) / "gateway_selection.json").write_text(json.dumps({"selected": best, "attempts": attempts}, indent=2))
    Path("configs").mkdir(exist_ok=True)
    Path("configs/selected_gateway.yaml").write_text(yaml.safe_dump(best["config"]))
    print(json.dumps({"selected_gateway": best["name"], "max_inflight": best["config"]["max_inflight"],
                      "work_budget_ms": best["config"]["work_budget_ms"]}), flush=True)


async def choose_native(args):
    config = yaml.safe_load(Path(args.engine_config).read_text())
    capabilities = json.loads(Path("artifacts/capability_matrix.json").read_text())["flags"]
    possibilities = [("fcfs", 32, 131072), ("fcfs", 64, 524288),
                     ("priority", 16, 65536), ("priority", 64, 524288)]
    attempts = []
    for index, (policy, active, tokens) in enumerate(possibilities):
        extra = []
        for flag, value in (("--max-num-active-seqs", active), ("--max-num-queued-reqs", 272),
                            ("--max-num-queued-tokens", tokens)):
            if capabilities.get(flag):
                extra.extend([flag, str(value)])
        candidate = {"policy": policy, "extra_args": extra}
        engine = {**config, "native": candidate}
        backend = Backend(engine, Path(args.output) / "native_backend")
        try:
            await backend.start("native", index)
            await warmup(engine, json.loads(Path(args.prompts).read_text()))
            gateway = yaml.safe_load(Path(args.gateway_config).read_text())
            result = await development_run(args, gateway, f"native-{index}", args.rate * 1.1,
                                           priority=policy == "priority")
            result["native"] = candidate
            attempts.append(result)
        finally:
            await backend.stop()
    valid = [r for r in attempts if r["manifest"]["valid"]]
    if not valid:
        raise RuntimeError("All native candidates invalid")
    best = max(valid, key=lambda r: (r["stats"]["cohort_goodput"], -r["stats"]["failure_fraction"]))
    (Path(args.output) / "native_selection.json").write_text(json.dumps({"selected": best, "attempts": attempts}, indent=2))
    Path("configs/selected_native.yaml").write_text(yaml.safe_dump(best["native"]))
    print(json.dumps({"selected_native": best["name"], "native": best["native"]}), flush=True)


def finalize(args):
    engine = yaml.safe_load(Path(args.engine_config).read_text())
    gateway = yaml.safe_load(Path("configs/selected_gateway.yaml").read_text())
    native = yaml.safe_load(Path("configs/selected_native.yaml").read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    frozen = {**engine, "native": native, "gateway": gateway,
              "slo_ms": calibration["slo_ms"], "client_deadline_seconds": calibration["client_deadline_seconds"],
              "reference_rate": args.rate, "trace_dir": "artifacts/traces", "prompts": args.prompts,
              "seeds": [1101, 1102, 1103]}
    Path("configs/frozen.yaml").write_text(yaml.safe_dump(frozen, sort_keys=True))
    validation_uuid = engine.get("validation_gpu_uuid", engine["gpu_uuid"])
    validation = {**frozen, "gpu_uuid": validation_uuid,
                  "seeds": [4401], "loads": ["near_capacity"],
                  "concurrent_gpu_trials": False,
                  "gpu_slots": [{"gpu_uuid": validation_uuid,
                                 "backend_port": 8030, "gateway_port": 8083, "seeds": [4401]}]}
    Path("configs/validation.yaml").write_text(yaml.safe_dump(validation, sort_keys=True))
    trace = build_trace(args.rate, 90, 4401, "near_capacity")
    Path("artifacts/traces/near_capacity-4401.json").write_text(json.dumps(trace, sort_keys=True))
    print("FROZEN_CONFIG_WRITTEN", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["rate", "gateway", "native", "finalize"])
    parser.add_argument("--model", default="admission-7b")
    parser.add_argument("--engine-config", default="configs/engine.yaml")
    parser.add_argument("--gateway-config", default="artifacts/calibration/gateway.yaml")
    parser.add_argument("--calibration", default="artifacts/calibration/calibration.json")
    parser.add_argument("--prompts", default="artifacts/traces/prompts.json")
    parser.add_argument("--output", default="artifacts/development")
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--deadline", type=float, default=60)
    args = parser.parse_args()
    if args.stage == "finalize":
        finalize(args)
    else:
        asyncio.run({"rate": choose_rate, "gateway": choose_gateway, "native": choose_native}[args.stage](args))


if __name__ == "__main__":
    main()
