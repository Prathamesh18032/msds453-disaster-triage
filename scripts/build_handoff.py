"""Build the direct-import Kaggle handoff with reviewed code and small vendor wheels."""
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
from disaster_triage.provenance import sha256_file, source_snapshot, save_json

root = Path(__file__).resolve().parents[1]
output = root / "artifacts/handoff-v2"
output.mkdir(parents=True, exist_ok=True)
version = "0.1.1"
wheel = root / "dist" / f"disaster_triage-{version}-py3-none-any.whl"
vendor = output / "vendor/datasketch-2.0.0-py3-none-any.whl"
if not wheel.is_file() or not vendor.is_file():
    raise SystemExit("Build the current wheel and download the datasketch vendor wheel first.")
with zipfile.ZipFile(wheel) as package:
    for name in package.namelist():
        if name.startswith("disaster_triage/"):
            if package.read(name) != (root / "src" / name).read_bytes():
                raise SystemExit("Wheel source differs from current code; rebuild it.")
        elif not name.startswith(f"disaster_triage-{version}.dist-info/"):
            raise SystemExit("Unexpected wheel content.")
source = output / "source.tar"
snapshot = source_snapshot(root, source)
staging = output / "bundle"
if staging.exists():
    shutil.rmtree(staging)
(staging / "code").mkdir(parents=True)
with tarfile.open(source) as archive:
    archive.extractall(staging / "code", filter="data")
shutil.copy2(wheel, staging / wheel.name)
shutil.copy2(vendor, staging / vendor.name)
shutil.copy2(root / "notebooks/kaggle-verification.ipynb", staging / "kaggle-verification.ipynb")
shutil.copy2(root / "docs/kaggle-start-here.md", staging / "START-HERE.md")
# Carry forward the six already-recorded synthetic attempts exactly once.
# Only those reviewed verification records are included, never arbitrary artifact traversal.
validation = json.loads((root / "artifacts/slice-2-validation.json").read_text())
for attempt in validation["accounting"]["attempts"]:
    run_id, attempt_id = attempt["run_id"], attempt["attempt_id"]
    if not run_id.startswith(("synthetic-", "pretrained-synthetic-")):
        raise SystemExit("Unexpected non-verification attempt in prior ledger.")
    relative = Path("runs") / run_id / "attempts" / attempt_id / "attempt.json"
    original = root / "artifacts" / relative
    value = json.loads(original.read_text())
    if value["status"] != "completed":
        raise SystemExit("Unfinished prior attempt needs reconciliation.")
    target = staging / "prior-accounting" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, target)
files = {
    path.relative_to(staging).as_posix(): sha256_file(path)
    for path in sorted(staging.rglob("*")) if path.is_file()
}
manifest = {
    "handoff_kind": "slice2-kaggle-v2", "schema_version": 2, "package_version": version,
    "runtime_profile": "kaggle-py313", "execution_validation": "pending user Kaggle run",
    "project_wheel": wheel.name, "datasketch_wheel": vendor.name,
    "source_sha256": snapshot["code_sha256"], "source": snapshot, "files": files,
}
save_json(staging / "handoff-manifest.json", manifest)
target = output / "slice-2-kaggle-handoff.zip"
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(staging).as_posix())
print(json.dumps({"handoff": str(target), "sha256": sha256_file(target),
                  "source_sha256": snapshot["code_sha256"], "files": len(files) + 1,
                  "notebook": str(root / "notebooks/kaggle-verification.ipynb")}, indent=2))
