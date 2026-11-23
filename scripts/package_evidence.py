"""Source and experiment evidence only; no weights, runtimes or SSH files."""
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    root = Path.cwd().resolve()
    target = root / "artifacts" / "evidence.tar.gz"
    allowed = {"src", "bench", "tests", "scripts", "configs", "reports", "artifacts"}
    forbidden = {".venv", ".bootstrap", ".cache", ".python", ".wheels", "models",
                 "__pycache__", ".pytest_cache", ".git"}
    manifest = []
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or path.resolve() == target:
                continue
            if any(part in forbidden or part.endswith(".egg-info") for part in relative.parts):
                continue
            if len(relative.parts) > 1 and relative.parts[0] not in allowed:
                continue
            if (len(relative.parts) == 1 and path.name != ".gitignore" and
                    path.suffix not in {".md", ".toml", ".yaml", ".yml", ".json", ".py", ".sh", ".txt"}):
                continue
            if path.suffix in {".pem", ".key", ".pyc", ".zip"} or "ed25519" in path.name or path.name == "known_hosts":
                continue
            if path.name.startswith("evidence") and path.name.endswith(".tar.gz"):
                continue
            manifest.append({"path": relative.as_posix(), "bytes": path.stat().st_size})
            archive.add(path, arcname=relative.as_posix(), recursive=False)
    result = {"archive": str(target), "files": len(manifest),
              "uncompressed_bytes": sum(p["bytes"] for p in manifest),
              "compressed_bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "members": manifest}
    (root / "artifacts" / "evidence_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    main()
