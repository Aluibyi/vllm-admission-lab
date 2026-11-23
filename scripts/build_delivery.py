"""Package code and original evidence with checksums, excluding runtimes and secrets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def selected_files(root):
    directories = [root / name for name in ("src", "bench", "tests", "scripts", "configs", "reports")]
    directories.extend(root / "artifacts" / name for name in
                       ("calibration", "development", "formal", "validation", "traces", "setup"))
    files = {path for directory in directories if directory.exists()
             for path in directory.rglob("*") if path.is_file()}
    files.update(path for path in root.iterdir() if path.is_file()
                 and (path.suffix in {".md", ".toml"} or path.name == ".gitignore"))
    names = ("environment.json", "model_manifest.json", "capability_matrix.json", "cancellation_probe.json",
             "requirements.remote.lock.txt", "requirements.remote.txt", "tests.remote.txt", "vllm_serve_help.txt",
             "pause_checkpoint.json", "pipeline_state.json", "user_pause_request.json", "resume_plan_20261117.json",
             "evidence_audit.json", "local_test_result.json", "tests.local.xml", "tests.local.txt",
             "connection_status_20261117.json", "source.tar.gz", "source_manifest.json")
    names += ("resource_allocation.json", "generation_smoke.json", "tests.rtx5090.xml", "tests.rtx5090.txt",
              "final_evidence_audit.json", "final_gpu_state.json", "execution_completion.json", "gateway_cancellation_probe.json")
    files.update(root / "artifacts" / name for name in names if (root / "artifacts" / name).is_file())
    for path in sorted(files):
        relative = path.relative_to(root)
        if path.is_symlink() or any(part in {"__pycache__", ".pytest_cache", ".venv", "models", ".git"}
                                   or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pem", ".key", ".pyc", ".zip"} or any(
                value in path.name.lower() for value in ("ed25519", "known_hosts", "ssh_config", "private_key", "refreshed_access")):
            raise ValueError(f"Credential-like path in delivery selection: {relative}")
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    formal = json.loads((root / "reports/completeness.json").read_text())
    validation = json.loads((root / "reports/validation/completeness.json").read_text())
    if not formal["complete"] or not validation["complete"]:
        raise RuntimeError("Refusing to label an incomplete experiment as completed delivery")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "completed_gpu_evidence_delivery", "experiment": "rtx5090-20261118-v1",
                "formal_completed": formal["completed"], "formal_expected": formal["expected"],
                "separate_trace_validation_completed": validation["completed"],
                "note": "New RTX 5090 evidence only; separate validation uses the same allocated GPU.", "files": []}
    # Exclusive creation prevents accidentally replacing a delivered archive.
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected_files(root):
            relative = path.relative_to(root).as_posix()
            value = path.read_bytes()
            archive.writestr("vllm-admission-lab/" + relative, value)
            manifest["files"].append({"path": relative, "bytes": len(value),
                                      "sha256": hashlib.sha256(value).hexdigest()})
        archive.writestr("vllm-admission-lab/DELIVERY_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for entry in manifest["files"]:
            value = archive.read("vllm-admission-lab/" + entry["path"])
            if hashlib.sha256(value).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Member checksum failed: {entry['path']}")
        if any("models/" in name or "/.venv/" in name or "private/" in name for name in archive.namelist()):
            raise RuntimeError("Excluded directory found")
    result = {"path": str(target), "bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "files": len(manifest["files"]), "all_member_checksums_verified": True}
    target.with_suffix(".manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
