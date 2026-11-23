"""Plan a continuation without overwriting completed or interrupted experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_jobs(config):
    return {f"{strategy}-{load}-{seed}"
            for strategy in ("B0", "B1", "B2", "P")
            for load in config.get("loads", ["light", "near_capacity", "burst"])
            for seed in config["seeds"]}


def verify_run(directory, frozen_path, config, provenance):
    """Fail closed on incompatible lineage, missing requests or dirty boundaries."""
    directory, provenance = Path(directory), Path(provenance)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    if run_id != directory.name or run_id not in expected_jobs(config):
        raise ValueError(f"Unexpected run identity: {directory}")
    if manifest.get("frozen_config_sha256") != sha256(frozen_path):
        raise ValueError(f"Frozen configuration differs: {run_id}")
    for label, name in (("source_manifest", "source_manifest.json"),
                        ("model_manifest", "model_manifest.json"), ("environment", "environment.json")):
        if manifest.get(label + "_sha256") != sha256(provenance / name):
            raise ValueError(f"{label} provenance differs: {run_id}")
    slots = config.get("gpu_slots", [])
    assigned = [s["gpu_uuid"] for s in slots if manifest["seed"] in s["seeds"]]
    if assigned and assigned != [manifest.get("gpu_uuid")]:
        raise ValueError(f"GPU assignment differs: {run_id}")
    trace_path = Path(config["trace_dir"]) / f'{manifest["load"]}-{manifest["seed"]}.json'
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    prompts = json.loads(Path(config["prompts"]).read_text(encoding="utf-8"))
    for label, value in (("trace", trace), ("prompts", prompts)):
        checksum = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        if manifest.get(label + "_sha256") != checksum:
            raise ValueError(f"{label} differs: {run_id}")
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["offered"] or len(rows) != len(trace["requests"]):
        raise ValueError(f"Request conservation failed: {run_id}")
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate request IDs: {run_id}")
    offered = {r["index"]: r for r in trace["requests"]}
    if {r["index"] for r in rows} != set(offered):
        raise ValueError(f"Request indexes differ: {run_id}")
    for row in rows:
        expected = offered[row["index"]]
        if row["run_id"] != run_id or any(row[k] != v for k, v in expected.items()):
            raise ValueError(f"Offered request differs: {run_id}/{row['index']}")
        if row["status"] == "succeeded" and (
                not row.get("metrics_valid", False) or not row.get("done")
                or row.get("actual_output_tokens") != expected["max_tokens"]
                or row.get("actual_prompt_tokens") != expected["prompt_tokens"]
                or row.get("finish_reason") != "length"):
            raise ValueError(f"Successful stream accounting differs: {run_id}/{row['index']}")
    if not (directory / "backend_drained.json").exists():
        raise ValueError(f"Missing backend drain verification: {run_id}")
    if json.loads((directory / "backend_drained.json").read_text())["verified"] is not True:
        raise ValueError(f"Backend was not drained: {run_id}")
    if run_id.startswith(("B2-", "P-")):
        state = json.loads((directory / "final_gateway_state.json").read_text())
        if any(state[k] for k in ("queued", "inflight", "paused", "log_dropped")) or abs(state["reserved_cost_ms"]) > 1e-6:
            raise ValueError(f"Gateway was not clean: {run_id}")
    return manifest


def plan_continuation(previous, frozen_path):
    previous, frozen_path = Path(previous), Path(frozen_path)
    config = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    complete, invalid, interrupted = [], [], []
    run_root = previous / "runs"
    for directory in sorted(run_root.iterdir() if run_root.exists() else []):
        if not directory.is_dir():
            continue
        if not (directory / "manifest.json").exists():
            interrupted.append(directory.name)
            continue
        manifest = verify_run(directory, frozen_path, config, previous / "provenance")
        (complete if manifest["valid"] else invalid).append(manifest["run_id"])
    # Invalid experiments stay visible; a continuation must not silently replace them.
    if invalid:
        raise ValueError(f"Invalid runs require explicit disposition before continuation: {invalid}")
    planned = expected_jobs(config)
    return {"previous_root": str(previous.resolve()), "frozen_config_sha256": sha256(frozen_path),
            "expected": len(planned), "completed": complete, "interrupted": interrupted,
            "missing": sorted(planned - set(complete)),
            "policy": "Separate output root; preserve prior completed and interrupted evidence.",
            "status": "prepared_not_executed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", required=True)
    parser.add_argument("--frozen", default="configs/frozen.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    plan = plan_continuation(args.previous, args.frozen)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
