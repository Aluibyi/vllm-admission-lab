"""Audit existing GPU evidence; never infer completion from a partial directory."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.checkpoint import plan_continuation, verify_run


root = Path(__file__).resolve().parents[1]
frozen_path = root / "configs/frozen.yaml"
config = yaml.safe_load(frozen_path.read_text())
formal = root / "artifacts/formal"
plan = plan_continuation(formal, frozen_path)
manifests = [verify_run(formal / "runs" / name, frozen_path, config, formal / "provenance")
             for name in plan["completed"]]
original = json.loads((formal / "provenance/source_manifest.json").read_text())
changed = []
for entry in original["files"]:
    path = root / entry["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if actual != entry["sha256"]:
        changed.append(entry["path"])
allowed = {"bench/matrix.py", "bench/parallel.py", "bench/analyze.py", "IMPLEMENTATION_STATUS.md"}
if set(changed) - allowed:
    raise ValueError(f"Unexpected change to frozen source: {sorted(set(changed)-allowed)}")
statuses = Counter()
for manifest in manifests:
    statuses.update(manifest["statuses"])
result = {"audited_utc": datetime.now(timezone.utc).isoformat(),
          "status": "passed_for_present_completed_runs", "matrix_complete": not plan["missing"],
          "expected": plan["expected"], "completed": len(manifests),
          "valid": sum(m["valid"] for m in manifests),
          "offered_requests": sum(m["offered"] for m in manifests), "request_statuses": dict(statuses),
          "max_run_send_lag_p95_ms": max(m["send_lag_p95_ms"] for m in manifests),
          "verified_runs": plan["completed"], "interrupted": plan["interrupted"], "missing": plan["missing"],
          "checked": ["request conservation and unique IDs", "exact offered trace fields",
                      "stream termination, finish reason and token counts on succeeded requests",
                      "prompt/trace/frozen/model/environment/source manifest hashes", "same GPU within seed",
                      "backend drain boundaries", "gateway terminal resource and log state"],
          "changed_existing_frozen_source_files": changed,
          "core_strategy_and_replay_files_unchanged": True,
          "scope": "Files and historical observations only; no new remote/GPU execution."}
target = root / "artifacts/evidence_audit.json"
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in {"checked", "verified_runs"}}, indent=2))
