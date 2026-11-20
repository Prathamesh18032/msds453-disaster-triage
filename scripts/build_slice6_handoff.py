"""Build the verified code-only Kaggle handoff from the current local wheel."""

import argparse
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from disaster_triage.provenance import sha256_file, source_snapshot
from disaster_triage.transformer_data import make_archive, write_index


def build(root, wheel, vendor, output):
    root = root.resolve()
    output = output.resolve()
    if not wheel.name.startswith("disaster_triage-") or vendor.name != "datasketch-2.0.0-py3-none-any.whl":
        raise ValueError("Supply the current project wheel and the pinned vendor wheel.")
    files_on_disk = {path.relative_to(root / "src").as_posix(): path.read_bytes()
                     for path in (root / "src" / "disaster_triage").rglob("*.py")}
    with zipfile.ZipFile(wheel) as package:
        actual = {name: package.read(name) for name in package.namelist() if name.startswith("disaster_triage/")}
        if actual != files_on_disk or any(
            not name.startswith(("disaster_triage/", wheel.name.split("-py3-")[0] + ".dist-info/"))
            for name in package.namelist()
        ):
            raise ValueError("Project wheel must match the reviewed package exactly.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".slice6-code-", dir=output.parent) as pending:
        stage = Path(pending)
        code = stage / "code"
        code.mkdir()
        source = source_snapshot(root, stage / "source.tar")
        with tarfile.open(stage / "source.tar") as archive:
            archive.extractall(code, filter="data")
        top = {
            "slice-6-kaggle.ipynb": root / "notebooks/slice-6-kaggle.ipynb",
            "START-HERE.md": root / "docs/transformer-execution.md",
            wheel.name: wheel, vendor.name: vendor,
        }
        files = {f"code/{name}": code / name for name in source["included_files"]}
        files.update(top)
        manifest = stage / "handoff-manifest.json"
        digest = write_index(manifest, {
            "handoff_kind": "slice2-kaggle-v2", "campaign_kind": "slice-6-code",
            "project_wheel": wheel.name, "datasketch_wheel": vendor.name,
            "package_version": wheel.name.split("-")[1],
            "source": source, "source_sha256": source["code_sha256"],
        }, files)
        make_archive(output, {**files, manifest.name: manifest})
        # Ship a direct-import notebook beside the opaque Kaggle input.
        shutil.copyfile(top["slice-6-kaggle.ipynb"], output.with_name("slice-6-kaggle.ipynb"))
    return {"code_archive": str(output), "code_archive_sha256": sha256_file(output),
            "code_manifest_sha256": digest, "source_sha256": source["code_sha256"]}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.wheel, args.vendor, args.output), indent=2))
