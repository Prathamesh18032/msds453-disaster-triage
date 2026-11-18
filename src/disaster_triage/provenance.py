"""Deterministic private provenance. No raw-data discovery."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def save_json(path, value):
    content = json_bytes(value)
    atomic_bytes(path, content)
    return sha256_bytes(content)

def git_revision(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def source_snapshot(root, destination):
    """Sorted uncompressed tar; fixed headers. Explicit roots/extensions only."""
    root = Path(root).resolve()
    selected = []
    for folder, extensions in {
        "src": {".py"}, "scripts": {".py"}, "configs": {".toml"},
        "requirements": {".txt"}, "notebooks": {".ipynb"},
    }.items():
        base = root / folder
        if base.is_symlink():
            raise ValueError("source snapshot rejects symlinks")
        if base.exists():
            for candidate in base.rglob("*"):
                if candidate.is_symlink():
                    raise ValueError("source snapshot rejects symlinks")
                if candidate.is_file() and candidate.suffix in extensions:
                    if not any(p.startswith(".") or p == "__pycache__"
                               for p in candidate.relative_to(base).parts):
                        selected.append(candidate)
    for name in ("pyproject.toml", ".python-version"):
        candidate = root / name
        if candidate.is_symlink():
            raise ValueError("source snapshot rejects symlinks")
        if candidate.is_file():
            selected.append(candidate)
    if not selected:
        raise ValueError("source snapshot is empty; supply the code checkout root")
    output = io.BytesIO()
    names = []
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for candidate in sorted(selected, key=lambda p: p.relative_to(root).as_posix()):
            name = candidate.relative_to(root).as_posix()
            content = candidate.read_bytes()
            header = tarfile.TarInfo(name)
            header.size, header.mode, header.mtime = len(content), 0o644, 0
            header.uid = header.gid = 0
            header.uname = header.gname = ""
            archive.addfile(header, io.BytesIO(content))
            names.append(name)
    atomic_bytes(destination, output.getvalue())
    return {"code_sha256": sha256_bytes(output.getvalue()),
            "code_revision": git_revision(root), "included_files": names}
