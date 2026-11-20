"""Research input and retrievable result packages for the bounded campaign."""

from pathlib import Path
import os
import re
import tempfile

from .accounting import budget_summary, identity, merge_accounting
from .classical_run import _check_source
from .provenance import atomic_bytes, sha256_file, source_snapshot
from .transformer import EPOCH_FILES, KEYS, verify_rolling
from .transformer_data import (
    INPUT_FILES, extract_verified, load_inputs, make_archive, prepare_roles, read_json,
    regular_file, write_index,
)

INDEX = "receipt.json"
RESULT_INDEX = "return-index.json"
RESEARCH_NAME = "slice-6-research.bin"
RETURN_NAME = "slice-6-results.bin"
ID_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}"
ATTEMPT_PATH = rf"runs/{ID_PATTERN}/attempts/{ID_PATTERN}/attempt\.json"
STATIC_STATE_NAMES = ("config.json", "latest.json", "pilot.json", "pilot-progress.json",
                      "complete.json")


def research_member(name):
    return name in INPUT_FILES or re.fullmatch(rf"prior-accounting/{ATTEMPT_PATH}", name) is not None


def result_member(name):
    if re.fullmatch(rf"accounting/{ATTEMPT_PATH}", name):
        return True
    parts = name.split("/")
    if len(parts) < 2 or parts[0] != "state":
        return False
    if len(parts) == 3 and parts[1] == "forecasts":
        return re.fullmatch(r"[a-f0-9]{32}\.json", parts[2]) is not None
    if len(parts) < 3 or parts[1] not in KEYS:
        return False
    if len(parts) == 3:
        return parts[2] in STATIC_STATE_NAMES
    if len(parts) == 4 and parts[2] == "rolling":
        return re.fullmatch(r"[a-f0-9]{32}-\d+-[a-f0-9]{8}\.pt", parts[3]) is not None
    if len(parts) >= 5 and parts[2] == "epochs" and parts[3] in ("e1", "e2", "e3"):
        return "/".join(parts[4:]) in (*EPOCH_FILES, "epoch.json")
    return False


def directory_for(root, campaign):
    identity(campaign)
    if len(campaign) > 65:
        raise ValueError("Campaign ID must contain at most 65 characters.")
    target = Path(root).resolve() / "artifacts" / "transformer" / campaign
    if target.resolve() != target:
        raise ValueError("Campaign artifacts may not contain symbolic links.")
    return target


def prior_members(root):
    summary = budget_summary(root)
    if summary["unresolved_attempts"]:
        raise ValueError("Reconcile attempts before preparing a handoff.")
    sets = {}
    for item in summary["attempts"]:
        name = f"runs/{item['run_id']}/attempts/{item['attempt_id']}/attempt.json"
        sets[f"prior-accounting/{name}"] = regular_file(root, name)
    return sets


def prepare_campaign(root, campaign, freeze, source_root):
    target = directory_for(root, campaign)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("Use a new campaign ID to prepare new immutable inputs.")
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as pending:
        staged = Path(pending) / "campaign"
        staged.mkdir(mode=0o700)
        package = staged / "inputs"
        lineage = prepare_roles(root, freeze, package)
        source = source_snapshot(source_root, staged / "source.tar")
        _check_source(staged / "source.tar")
        load_inputs(package, lineage)
        files = {name: regular_file(package, name) for name in INPUT_FILES}
        for name, original in prior_members(Path(root).resolve() / "artifacts").items():
            atomic_bytes(package / name, original.read_bytes())
            files[name] = package / name
        digest = write_index(package / INDEX, {
            "schema_version": 1, "kind": "slice-6-research", "campaign": campaign,
            "lineage": lineage, "source": source,
        }, files)
        make_archive(staged / RESEARCH_NAME, {**files, INDEX: package / INDEX})
        staged.rename(target)
    return {"campaign": campaign, "research_archive": str(target / RESEARCH_NAME),
            "receipt_sha256": digest, "source_sha256": source["code_sha256"],
            "rows": lineage["rows"]}


def check_index(directory, campaign, expected_sha256=None):
    package = directory / "inputs"
    receipt_path = regular_file(package, INDEX)
    digest = sha256_file(receipt_path)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("Private campaign receipt differs from the supplied digest.")
    receipt = read_json(receipt_path)
    files = receipt["files"]
    if (receipt["schema_version"] != 1 or receipt["kind"] != "slice-6-research"
            or receipt["campaign"] != campaign or not INPUT_FILES <= files.keys()
            or not all(research_member(name) for name in files)):
        raise ValueError("Research input package does not match the campaign contract.")
    inventory = {path.relative_to(package).as_posix() for path in package.rglob("*")
                 if path.is_file() or path.is_symlink()}
    if inventory != {INDEX, *files}:
        raise ValueError("Unexpected private input files.")
    for name, value in files.items():
        if sha256_file(regular_file(package, name)) != value:
            raise ValueError("Private campaign input bytes changed.")
    if sha256_file(regular_file(directory, "source.tar")) != receipt["source"]["code_sha256"]:
        raise ValueError("Campaign source snapshot differs from the preparation receipt.")
    _check_source(directory / "source.tar")
    return receipt, digest, load_inputs(package, receipt["lineage"])


def receive_campaign(root, campaign, archive, receipt_sha256, source_root):
    from .transformer_data import is_digest

    if not is_digest(receipt_sha256):
        raise ValueError("Provide the research receipt SHA-256 issued by local preparation.")
    target = directory_for(root, campaign)
    if target.exists():
        # Inspect the supplied archive receipt even when the working campaign already exists.
        import zipfile
        from .provenance import sha256_bytes
        with zipfile.ZipFile(archive) as bundle:
            if bundle.getinfo(INDEX).file_size > 1024 * 1024 or sha256_bytes(bundle.read(INDEX)) != receipt_sha256:
                raise ValueError("Supplied archive does not belong to this campaign.")
        check_index(target, campaign, receipt_sha256)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".receive-", dir=target.parent) as pending:
        staged = Path(pending)
        (staged / "inputs").mkdir()
        receipt = extract_verified(archive, staged / "inputs", index=INDEX,
                                   allowed=research_member, expected_sha256=receipt_sha256)
        snapshot = source_snapshot(source_root, staged / "source.tar")
        if snapshot["code_sha256"] != receipt["source"]["code_sha256"]:
            raise ValueError("Installed handoff code does not match the local preparation source.")
        check_index(staged, campaign, receipt_sha256)
        # Leave extraction temporary and publish the validated directory as one unit.
        ready = staged / "ready"
        ready.mkdir(mode=0o700)
        (staged / "inputs").rename(ready / "inputs")
        (staged / "source.tar").rename(ready / "source.tar")
        ready.rename(target)


def attempt_files(root, campaign):
    paths = {}
    for key in KEYS:
        run_id = f"transformer-{campaign}-{key}"
        for path in sorted((root / "runs" / run_id / "attempts").glob("*/attempt.json")):
            name = path.relative_to(root).as_posix()
            paths[name] = regular_file(root, name)
    return paths


def state_files(state):
    files = {}
    for key in KEYS:
        directory = state / key
        if not directory.exists():
            continue
        for name in STATIC_STATE_NAMES:
            if (directory / name).exists():
                member = f"{key}/{name}"
                files[member] = regular_file(state, member)
        latest = verify_rolling(directory)
        if latest is not None:
            member = f"{key}/rolling/{latest['file']}"
            files[member] = regular_file(state, member)
        for epoch in range(1, 4):
            if (directory / "epochs" / f"e{epoch}").exists():
                for name in (*EPOCH_FILES, "epoch.json"):
                    member = f"{key}/epochs/e{epoch}/{name}"
                    files[member] = regular_file(state, member)
    for path in (state / "forecasts").glob("*.json"):
        member = path.relative_to(state).as_posix()
        files[member] = regular_file(state, member)
    return files


def export_campaign(campaign, destination):
    directory = campaign.directory
    root = campaign.artifacts
    files = {f"state/{name}": path for name, path in state_files(campaign.state).items()}
    files.update({f"accounting/{name}": path for name, path in attempt_files(root, campaign.id).items()})
    if not all(result_member(name) for name in files):
        raise ValueError("Unexpected result archive members.")
    if any(read_json(path)["status"] == "running"
           for name, path in files.items() if name.startswith("accounting/")):
        raise ValueError("Reconcile interrupted records before exporting an accepted snapshot.")
    with tempfile.TemporaryDirectory(prefix=".return-", dir=directory) as pending:
        manifest = Path(pending) / RESULT_INDEX
        digest = write_index(manifest, {
            "schema_version": 1, "kind": "slice-6-results", "campaign": campaign.id,
            "input_sha256": campaign.receipt_sha256,
        }, files)
        make_archive(destination, {**files, RESULT_INDEX: manifest})
    return {"result_archive": str(destination), "return_index_sha256": digest,
            "result_sha256": sha256_file(destination)}


def import_campaign(campaign, source, *, expected_sha256=None):
    with tempfile.TemporaryDirectory(prefix=".collect-", dir=campaign.directory) as pending:
        staged = Path(pending)
        receipt = extract_verified(source, staged, index=RESULT_INDEX, allowed=result_member,
                                   expected_sha256=expected_sha256)
        if (receipt["schema_version"] != 1 or receipt["kind"] != "slice-6-results"
                or receipt["campaign"] != campaign.id
                or receipt["input_sha256"] != campaign.receipt_sha256):
            raise ValueError("Returned results belong to a different research campaign.")
        staged_state = staged / "state"
        incoming_state = state_files(staged_state)
        incoming_attempts = attempt_files(staged / "accounting", campaign.id)
        expected_files = {f"accounting/{name}" for name in incoming_attempts}
        expected_files.update(f"state/{name}" for name in incoming_state)
        if expected_files != receipt["files"].keys():
            raise ValueError("Returned archive includes unrelated or unreferenced artifacts.")
        if any(read_json(path)["status"] == "running" for path in incoming_attempts.values()):
            raise ValueError("Returned snapshot must contain terminal attempts.")
        campaign.inspect(state=staged_state, accounting=staged / "accounting")
        existing = state_files(campaign.state)
        for name, path in existing.items():
            if "/rolling/" in name:
                continue
            if name not in incoming_state:
                raise ValueError("Returned snapshot does not include previous campaign evidence.")
            if name.endswith("/latest.json"):
                old, new = read_json(path), read_json(incoming_state[name])
                old_step, new_step = (item["progress"]["optimizer_steps"] for item in (old, new))
                if old_step > new_step or (old_step == new_step and old != new):
                    raise ValueError("Returned recovery history regressed or diverged.")
            elif name.endswith("/pilot-progress.json") and name.rsplit("/", 1)[0] + "/pilot.json" not in existing:
                # Resume may overwrite samples for steps recomputed from the last durable boundary.
                continue
            elif sha256_file(path) != sha256_file(incoming_state[name]):
                raise ValueError("Returned snapshot conflicts with previous campaign evidence.")
        for name, path in incoming_attempts.items():
            previous = campaign.artifacts / name
            if previous.exists() and previous.read_bytes() != path.read_bytes():
                raise ValueError("Returned snapshot conflicts with retained attempt history.")
        if set(attempt_files(campaign.artifacts, campaign.id)) - incoming_attempts.keys():
            raise ValueError("Returned snapshot omitted prior attempts.")
        for name, path in sorted(incoming_state.items(), key=lambda item: item[0].endswith("/latest.json")):
            destination = campaign.state / name
            if destination.exists() and sha256_file(destination) == sha256_file(path):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            linked = path.with_name(path.name + ".publish")
            os.link(path, linked)
            linked.replace(destination)
        merge_accounting(staged / "accounting", campaign.artifacts)
        for name, path in existing.items():
            if "/rolling/" in name and name not in incoming_state:
                path.unlink(missing_ok=True)
    return campaign.inspect()
