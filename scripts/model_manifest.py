"""Hash read-only model files, without loading code or weights into a GPU."""
import hashlib
import json
from pathlib import Path

model = Path("/models/Qwen2.5-7B-Instruct")
files = []
for path in sorted(model.rglob("*")):
    if not path.is_file() or any(part.startswith(".") for part in path.relative_to(model).parts):
        continue
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(16 * 1024 * 1024):
            checksum.update(data)
    files.append({"path": path.relative_to(model).as_posix(), "bytes": path.stat().st_size,
                  "sha256": checksum.hexdigest()})
result = {"model": "Qwen2.5-7B-Instruct", "read_only_mount": str(model), "files": files,
          "total_bytes": sum(item["bytes"] for item in files)}
Path("artifacts/model_manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps({"model_files": len(files), "model_bytes": result["total_bytes"]}), flush=True)
