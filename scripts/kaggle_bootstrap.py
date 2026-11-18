"""Standard-library bootstrap embedded verbatim into the Kaggle notebook."""
import hashlib
import importlib.metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import stat
import subprocess
import sys
import zipfile

HANDOFF_KIND = "slice2-kaggle-v2"

def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def extract_zip(path, target):
    """Regular files only; never write outside a fresh staging directory."""
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive paths")
            for item in archive.infolist():
                destination = (target / item.filename).resolve()
                kind = stat.S_IFMT(item.external_attr >> 16)
                if (not destination.is_relative_to(target) or
                    item.filename.startswith("/") or
                    kind not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError("archive contains unsupported paths or links")
            archive.extractall(target)
    except BaseException:
        shutil.rmtree(target)
        raise
    return target

def locate_handoff(input_root, working_root):
    input_root, working_root = Path(input_root), Path(working_root)
    override = os.environ.get("TRIAGE_HANDOFF_DIR")
    if override:
        return Path(override).resolve()
    candidates = []
    for path in input_root.rglob("handoff-manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("handoff_kind") == HANDOFF_KIND:
            candidates.append(path.parent)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("attach one current handoff or set TRIAGE_HANDOFF_DIR")
    archives = list(input_root.rglob("slice-2-kaggle-handoff.zip"))
    if len(archives) != 1:
        raise FileNotFoundError("Add the private handoff dataset to this notebook first")
    target = working_root / ("handoff-" + file_digest(archives[0])[:16])
    if not target.exists():
        extract_zip(archives[0], target)
    return target

def verify_handoff(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "handoff-manifest.json").read_text())
    if manifest.get("handoff_kind") != HANDOFF_KIND:
        raise ValueError("use the updated Kaggle handoff")
    for name, expected in manifest["files"].items():
        raw_path = bundle / name
        path = raw_path.resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or raw_path.is_symlink():
            raise ValueError("handoff member missing or outside bundle")
        if file_digest(path) != expected:
            raise ValueError(f"handoff digest mismatch: {name}")
    # Refuse unchecked executable extras in the source tree.
    actual = {p.relative_to(bundle).as_posix() for p in (bundle / "code").rglob("*") if p.is_file()}
    declared = {name for name in manifest["files"] if name.startswith("code/")}
    if actual != declared:
        raise ValueError("source tree contains unexpected or missing files")
    return manifest

def install_plan(bundle, *, python_version=None, installed=None):
    """Pure plan: drift fails before any package mutation."""
    bundle = Path(bundle).resolve()
    manifest = verify_handoff(bundle)
    profiles = runpy.run_path(str(bundle / "code/src/disaster_triage/profiles.py"))
    installed = profiles["installed_versions"]() if installed is None else dict(installed)
    errors = profiles["profile_errors"](
        "kaggle-py313", python_version or platform.python_version(), installed,
        allow_missing_datasketch=True,
    )
    if errors:
        raise RuntimeError("Runtime differs from the reported Kaggle profile:\n" + "\n".join(errors))
    commands = []
    if installed.get("datasketch") is None:
        commands.append([sys.executable, "-m", "pip", "install", "--no-deps",
                         str(bundle / manifest["datasketch_wheel"])])
    try:
        current_project = package_metadata.version("disaster-triage")
    except package_metadata.PackageNotFoundError:
        current_project = None
    project_matches = current_project == manifest["package_version"]
    if project_matches:
        distribution = package_metadata.distribution("disaster-triage")
        for source in (bundle / "code/src/disaster_triage").glob("*.py"):
            installed_file = Path(distribution.locate_file("disaster_triage/" + source.name))
            if not installed_file.is_file() or file_digest(installed_file) != file_digest(source):
                project_matches = False
                break
    if not project_matches:
        if "disaster_triage" in sys.modules:
            raise RuntimeError("Restart the kernel before replacing an already imported package")
        commands.append([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                         str(bundle / manifest["project_wheel"])])
    return manifest, commands

def prepare_handoff(bundle, working_root, *, run_command=subprocess.check_call):
    manifest, commands = install_plan(bundle)
    for command in commands:
        run_command(command)
    # No already-imported scientific library is replaced. New package imports are safe.
    profiles = runpy.run_path(str(Path(bundle) / "code/src/disaster_triage/profiles.py"))
    errors = profiles["profile_errors"](
        "kaggle-py313", platform.python_version(), profiles["installed_versions"]())
    if errors:
        raise RuntimeError("\n".join(errors))
    code = Path(working_root) / ("code-" + manifest["source_sha256"][:16])
    if not code.exists():
        shutil.copytree(Path(bundle) / "code", code)
    return code, manifest

def dependency_closure():
    """Capture/check the project's installed dependency closure, without upgrades."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending = ["disaster-triage"]
    records, errors = {}, []
    while pending:
        name = canonicalize_name(pending.pop())
        if name in records:
            continue
        try:
            distribution = package_metadata.distribution(name)
        except package_metadata.PackageNotFoundError:
            errors.append(f"missing dependency: {name}")
            continue
        requirements = []
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            requirements.append(str(requirement))
            try:
                version = package_metadata.version(requirement.name)
                if not requirement.specifier.contains(version, prereleases=True):
                    errors.append(f"{requirement.name} {version} does not satisfy {requirement.specifier}")
            except package_metadata.PackageNotFoundError:
                errors.append(f"missing dependency: {requirement.name}")
            pending.append(requirement.name)
        records[name] = {"version": distribution.version, "requires": requirements}
    if errors:
        raise RuntimeError("\n".join(sorted(set(errors))))
    return records


def locate_results(input_root, working_root, override=None):
    if override:
        return Path(override).resolve()
    matches = []
    for path in Path(input_root).rglob("resume-index.json"):
        value = json.loads(path.read_text())
        if value.get("kind") == "slice2-verification-results" and value.get("phase") == "verify":
            matches.append(path.parent)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("attach one first-session result, or set RESULTS_DIR")
    archives = list(Path(input_root).rglob("slice-2-verification-results*.zip"))
    if len(archives) != 1:
        raise FileNotFoundError("Attach the downloaded first-session results as a private input")
    target = Path(working_root) / ("retrieved-" + file_digest(archives[0])[:16])
    if not target.exists():
        extract_zip(archives[0], target)
    return target

def export_results(working_root, *, phase, probes, evidence, source_sha256):
    """Export synthetic artifacts/accounting, excluding public model caches."""
    working_root = Path(working_root)
    artifacts = working_root / "artifacts"
    export = working_root / "return"
    export.mkdir(parents=True, exist_ok=True)
    index = {
        "kind": "slice2-verification-results", "schema_version": 1,
        "phase": phase, "source_sha256": source_sha256,
        "probes": [
            {"device": item["device"], "path": str(
                Path(item["verification_dir"]).relative_to(working_root))}
            for item in probes
        ],
    }
    (export / "resume-index.json").write_text(json.dumps(index, indent=2) + "\n")
    (export / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    # Construct file list before opening output; never include caches or output itself.
    members = [(p, p.relative_to(working_root).as_posix())
               for p in artifacts.rglob("*") if p.is_file()]
    members += [(export / name, name) for name in ("resume-index.json", "evidence.json")]
    digests = {name: file_digest(path) for path, name in members}
    (export / "result-manifest.json").write_text(json.dumps(digests, indent=2) + "\n")
    target = export / "slice-2-verification-results.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in members:
            archive.write(path, name)
        archive.write(export / "result-manifest.json", "result-manifest.json")
    return target

def verify_results(directory):
    directory = Path(directory).resolve()
    digests = json.loads((directory / "result-manifest.json").read_text())
    for name, expected in digests.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.is_symlink():
            raise ValueError("retrieved result path invalid")
        if file_digest(path) != expected:
            raise ValueError(f"retrieved result digest mismatch: {name}")
    index = json.loads((directory / "resume-index.json").read_text())
    if index.get("kind") != "slice2-verification-results" or index.get("phase") != "verify":
        raise ValueError("attach the first-session verification results")
    return index
