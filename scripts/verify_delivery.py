"""CPU-only verification of exported evidence; no model weights or GPU needed."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

import yaml

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
from bench.checkpoint import expected_jobs, verify_run

counts = {}
for label in ("formal", "validation"):
    frozen = root / "configs" / ("frozen.yaml" if label == "formal" else "validation.yaml")
    config = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    directory = root / "artifacts" / label
    manifests = [verify_run(path.parent, frozen, config, directory / "provenance")
                 for path in sorted((directory / "runs").glob("*/manifest.json"))]
    assert all(m["valid"] for m in manifests)
    assert {m["run_id"] for m in manifests} == expected_jobs(config)
    counts[label] = {"runs": len(manifests), "requests": sum(m["offered"] for m in manifests)}

provenance = root / "artifacts/formal/provenance"
manifest = json.loads((provenance / "source_manifest.json").read_text(encoding="utf-8"))
cache_roots = {".nv", ".triton", ".humming"}
code_files, generated_files = 0, 0
with tarfile.open(provenance / "source.tar.gz") as archive:
    for entry in manifest["files"]:
        if entry["path"].split("/")[0] in cache_roots:
            generated_files += 1
            continue
        current = root / entry["path"]
        assert hashlib.sha256(current.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
        handle = archive.extractfile(entry["path"])
        assert handle is not None
        assert hashlib.sha256(handle.read()).hexdigest() == entry["sha256"], entry["path"]
        code_files += 1
archive_hash = hashlib.sha256((provenance / "source.tar.gz").read_bytes()).hexdigest()
for path in (root / "artifacts/source.tar.gz", root / "artifacts/validation/provenance/source.tar.gz"):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == archive_hash

assert json.loads((root / "artifacts/final_evidence_audit.json").read_text())["status"] == "passed"
result = {"status": "passed", "matrices": counts, "frozen_source_files_checked": code_files,
          "generated_cache_entries_preserved_in_original_archive_only": generated_files,
          "scope": "Verify exported records and frozen source; historical model rehash is in final_evidence_audit.json.",
          "requires_gpu_or_model_weights": False}
(root / "reports/portable_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
