"""Create a source-only transfer bundle; credentials and runtimes stay outside."""
from pathlib import Path
import hashlib
import json
import tarfile


root = Path(__file__).resolve().parents[1]
target = root / "artifacts" / "source.tar.gz"
target.parent.mkdir(exist_ok=True)
count = 0
file_hashes = []
with tarfile.open(target, "w:gz") as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(x in {".venv", ".bootstrap", ".python", ".wheels", ".git", ".cache", "models",
                                           "__pycache__", ".pytest_cache", "artifacts"}
                                      or x.endswith(".egg-info") for x in relative.parts):
            continue
        if path.suffix in {".pem", ".key", ".zip", ".gz", ".pyc"} or "ed25519" in path.name:
            continue
        archive.add(path, arcname=str(relative), recursive=False)
        file_hashes.append({"path": relative.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        count += 1
    source_manifest = {"files": file_hashes,
                       "tree_sha256": hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()}
    manifest_path = root / "artifacts" / "source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2))
    archive.add(manifest_path, arcname="artifacts/source_manifest.json", recursive=False)
print(f"SOURCE_BUNDLE files={count} bytes={target.stat().st_size} path={target}")
