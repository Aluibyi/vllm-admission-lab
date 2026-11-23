"""Bounded ranged download of pinned Torch wheels, verified against official PyPI.

Only used when large single HTTP transfers stall. The default mirror serves
bytes; filenames, versions and SHA256 always come from official PyPI metadata.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def metadata(name, version=None):
    suffix = f"/{version}" if version else ""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}{suffix}/json", timeout=20) as response:
        return json.load(response)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    cache = Path(".cache/uv/archive-v0")
    prepared = set()
    for path in cache.glob("*/*.dist-info/METADATA"):
        try:
            info = Parser().parsestr(path.read_text(errors="replace"), headersonly=True)
            prepared.add((canonicalize_name(info["Name"]), info["Version"]))
        except Exception:
            continue
    vllm = metadata("vllm", "0.33.0")
    torch_requirement = next(Requirement(r) for r in vllm["info"]["requires_dist"]
                             if Requirement(r).name == "torch")
    torch_version = next(s.version for s in torch_requirement.specifier if s.operator == "==")
    torch = metadata("torch", torch_version)
    selected = [("torch", torch_version, torch)]
    environment = {"python_version": "3.12", "python_full_version": "3.12.16",
                   "sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64", "extra": ""}
    for text in torch["info"]["requires_dist"]:
        req = Requirement(text)
        if req.marker and not req.marker.evaluate(environment):
            continue
        if not (req.name.startswith("nvidia-") or req.name == "triton"):
            continue
        pins = [s.version for s in req.specifier if s.operator == "=="]
        if len(pins) != 1:
            raise RuntimeError(f"Expected exact binary dependency pin: {text}")
        version = pins[0]
        if (canonicalize_name(req.name), version) not in prepared:
            selected.append((req.name, version, metadata(req.name, version)))
    output = Path(".wheels")
    output.mkdir(exist_ok=True)
    records = []
    pending = []
    descriptors = []
    block_size = 4 * 1024 * 1024
    for name, version, data in selected:
        compatible = [f for f in data["urls"] if f["filename"].endswith(".whl")
                      and "x86_64" in f["filename"] and "manylinux" in f["filename"]
                      and ("cp312-" in f["filename"] or "py3-none-" in f["filename"])]
        if len(compatible) != 1:
            raise RuntimeError(f"Expected one Linux CPython3.12 wheel: {name} {version}")
        wheel = compatible[0]
        path = output / wheel["filename"]
        expected = wheel["digests"]["sha256"]
        record = {"name": name, "version": version, "filename": path.name,
                  "bytes": wheel["size"], "official_url": wheel["url"], "sha256": expected}
        records.append(record)
        if path.exists() and sha256(path) == expected:
            continue
        part = path.with_suffix(".part")
        fd = os.open(part, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(fd, wheel["size"])
        descriptors.append((fd, part, path, expected))
        for start in range(0, wheel["size"], block_size):
            pending.append((fd, wheel, start, min(start + block_size, wheel["size"]) - 1))
        print("WHEEL_QUEUED", path.name, wheel["size"], flush=True)

    def download(item):
        fd, wheel, start, end = item
        official = wheel["url"]
        mirror = official.replace("https://files.pythonhosted.org/", "https://pypi.tuna.tsinghua.edu.cn/")
        last = None
        for attempt in range(5):
            url = mirror if attempt < 3 else official
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                               "User-Agent": "vllm-admission-lab/0.1"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError("Server did not honor the exact byte range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("Incorrect range byte count")
                written = 0
                while written < len(data):
                    written += os.pwrite(fd, data[written:], start + written)
                return len(data)
            except Exception as exc:
                last = exc
                time.sleep(min(3, attempt + 1))
        raise RuntimeError(f"Range failed: {wheel['filename']} {start}-{end}: {last}")

    try:
        completed = 0
        last_notice = time.monotonic()
        with ThreadPoolExecutor(max_workers=12) as pool:
            for future in as_completed([pool.submit(download, item) for item in pending]):
                completed += future.result()
                if time.monotonic() - last_notice > 10:
                    print("RANGE_PROGRESS", completed, "bytes", flush=True)
                    last_notice = time.monotonic()
        for fd, part, path, expected in descriptors:
            os.fsync(fd)
            os.close(fd)
            with part.open("rb") as handle:
                actual = hashlib.sha256()
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    actual.update(block)
            if actual.hexdigest() != expected:
                raise RuntimeError(f"Official PyPI SHA256 mismatch: {path.name}")
            part.replace(path)
            print("WHEEL_VERIFIED", path.name, expected, flush=True)
    finally:
        for fd, _, _, _ in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass
    Path("artifacts/setup/wheel_manifest.json").write_text(json.dumps(records, indent=2))
    print("RANGED_WHEELS_READY", len(records), flush=True)


if __name__ == "__main__":
    main()
