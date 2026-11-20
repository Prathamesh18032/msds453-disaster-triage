"""Prepare and verify the two permitted DistilBERT roles and exact-byte handoffs."""

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

from .bundles import RoleRow, role_bytes
from .constants import ARCHIVE_SHA256, EVENTS, LABELS
from .dataset import FrozenDataset
from .provenance import atomic_bytes, json_bytes, save_json, sha256_bytes, sha256_file

ROLES = ("train", "model_selection")
INPUT_FILES = {f"{kind}/{role}.jsonl" for kind in ("roles", "texts") for role in ROLES}
FREEZE_ID = "63dd4a283dee0cd16457270365cce8b153b7a41c4c91a12876d81acb56f21a56"
FREEZE_SHA256 = "0d84025e8fef76d3254ba1f78b280e8bfa417748208660787d4fd59dc313b2e3"
ELIGIBLE_SHA256 = "6657462fbc3e541cc596c9b8748dfa257a55fb1e173b5834920eae806180155d"
ROLE_SHA256 = {
    "train": "b95c63dd286795d06bbfbeb813d8bf2d2c6b571555cba502235595d5b5049898",
    "model_selection": "77df04816ca84b1f1ad2881d8d78198dd7bf99d98b0722a92fee6c5a24e8661a",
}
RESEARCH_COUNTS = (1559, 1353, 4090, 2132, 140, 1308, 4874, 1036, 5804, 2876)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def aligned_projection(rows, records, role):
    by_id = {item["row_id"]: item for item in records}
    if len(records) != len(by_id) or set(by_id) != {row.row_id for row in rows}:
        raise ValueError("Transformer text and role inventory differ.")
    pieces = []
    for row in rows:
        item = by_id[row.row_id]
        if (item["event_id"], item["y_true"], item["role"]) != (row.event_id, row.y_true, role):
            raise ValueError("Transformer text and role metadata differ.")
        pieces.append(json_bytes({"row_id": row.row_id, "text": item["transformer_text"]}))
    return b"".join(pieces)


def prepare_roles(root, freeze, destination):
    frozen = FrozenDataset(root, freeze)
    evidence = frozen.verification
    if (evidence["freeze_id"], evidence["freeze_manifest_sha256"],
        evidence["eligible_manifest_sha256"]) != (FREEZE_ID, FREEZE_SHA256, ELIGIBLE_SHA256):
        raise ValueError("Slice 6 requires the accepted exact-byte research freeze.")
    if frozen.descriptor["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Unexpected source archive identity.")
    for role in ROLES:
        path = frozen.directory / "roles" / f"{role}.jsonl"
        content = path.read_bytes()
        if sha256_bytes(content) != ROLE_SHA256[role]:
            raise ValueError("Frozen role manifest digest mismatch.")
        rows = tuple(RoleRow(**json.loads(line)) for line in content.splitlines())
        texts = aligned_projection(rows, frozen.load_role(role), role)
        atomic_bytes(destination / "roles" / path.name, content)
        atomic_bytes(destination / "texts" / path.name, texts)
    return {
        "archive_sha256": ARCHIVE_SHA256,
        "eligible_manifest_sha256": ELIGIBLE_SHA256,
        "freeze_manifest_sha256": FREEZE_SHA256,
        "freeze_id": FREEZE_ID,
        "role_sha256": ROLE_SHA256,
        "rows": {role: evidence["role_counts"][role] for role in ROLES},
        "training_counts": list(RESEARCH_COUNTS),
        "slice8_prediction_rows_per_model": sum(
            evidence["role_counts"][role] for role in ("calibration", "policy", "final_test")
        ),
        "labels": list(LABELS),
    }


@dataclass(frozen=True)
class Inputs:
    lineage: dict
    rows: dict
    texts: dict


def load_inputs(directory, lineage, *, research=True):
    rows, texts = {}, {}
    if research and (
        lineage["freeze_id"] != FREEZE_ID or lineage["archive_sha256"] != ARCHIVE_SHA256
        or lineage["eligible_manifest_sha256"] != ELIGIBLE_SHA256
        or lineage["freeze_manifest_sha256"] != FREEZE_SHA256
        or lineage["role_sha256"] != ROLE_SHA256 or lineage["labels"] != list(LABELS)
        or lineage["rows"] != {"train": 25172, "model_selection": 6376}
        or lineage["training_counts"] != list(RESEARCH_COUNTS)
    ):
        raise ValueError("Inputs do not identify the approved research freeze.")
    if set(lineage["rows"]) != set(ROLES) or set(lineage["role_sha256"]) != set(ROLES):
        raise ValueError("Inputs may only contain train and model-selection roles.")
    for role in ROLES:
        data = (directory / "roles" / f"{role}.jsonl").read_bytes()
        rows[role] = tuple(RoleRow(**json.loads(line)) for line in data.splitlines())
        if (not rows[role] or role_bytes(rows[role]) != data
                or sha256_bytes(data) != lineage["role_sha256"][role]
                or len(rows[role]) != lineage["rows"][role]):
            raise ValueError("Frozen role order, count or digest mismatch.")
        expected_events = EVENTS["train" if role == "train" else "development"]
        if research and {row.event_id for row in rows[role]} != set(expected_events):
            raise ValueError("Frozen role contains incorrect events.")
        projected = [json.loads(line) for line in
                     (directory / "texts" / f"{role}.jsonl").read_bytes().splitlines()]
        if len(projected) != len(rows[role]) or any(
            not isinstance(item, dict) or set(item) != {"row_id", "text"}
            or item["row_id"] != row.row_id or not isinstance(item["text"], str)
            or not item["text"].strip()
            for item, row in zip(projected, rows[role])
        ):
            raise ValueError("Transformer projection does not match frozen role.")
        texts[role] = tuple(item["text"] for item in projected)
    if ({row.row_id for row in rows["train"]} & {row.row_id for row in rows["model_selection"]}
            or {row.event_id for row in rows["train"]} & {row.event_id for row in rows["model_selection"]}):
        raise ValueError("Train and selection roles must have disjoint rows and events.")
    counts = [0] * len(LABELS)
    for row in rows["train"]:
        counts[row.y_true] += 1
    if counts != lineage["training_counts"] or not all(counts):
        raise ValueError("Every fixed label must have the expected positive training count.")
    return Inputs(lineage, rows, texts)


def regular_file(root, name):
    path = root.joinpath(*PurePosixPath(name).parts)
    if not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Handoff members must be regular files within the expected root.")
    return path


def make_archive(destination, members):
    """Publish an archive only after all members are written."""
    import tempfile
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=destination.parent) as staging:
        candidate = Path(staging) / "archive.bin"
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED,
                             strict_timestamps=False) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, arcname=name,
                              compress_type=zipfile.ZIP_STORED if name.endswith(".pt") else None)
        candidate.replace(destination)


def extract_verified(source, destination, *, index, allowed, expected_sha256=None):
    """Verify all names and expected bytes before exposing an extracted archive."""
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or index not in names:
            raise ValueError("Missing index or duplicate archive members.")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            kind = stat.S_IFMT(item.external_attr >> 16)
            if (path.is_absolute() or path.as_posix() != item.filename or ".." in path.parts
                    or kind not in (0, stat.S_IFREG)
                    or (item.filename != index and not allowed(item.filename))):
                raise ValueError("Unexpected archive member or path.")
        if archive.getinfo(index).file_size > 1024 * 1024:
            raise ValueError("Archive receipt exceeds the permitted size.")
        receipt_bytes = archive.read(index)
        if expected_sha256 is not None and sha256_bytes(receipt_bytes) != expected_sha256:
            raise ValueError("Archive index differs from the independently supplied digest.")
        receipt = json.loads(receipt_bytes)
        hashes = receipt["files"]
        if set(hashes) != set(names) - {index} or not all(map(is_digest, hashes.values())):
            raise ValueError("Archive inventory differs from its receipt.")
        if sum(item.file_size for item in archive.infolist()) > shutil.disk_usage(destination).free:
            raise ValueError("Insufficient disk space to verify and extract the private archive.")
        for name, digest in hashes.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            if sha256_file(target) != digest:
                raise ValueError("Archive member digest mismatch.")
        atomic_bytes(destination / index, receipt_bytes)
    return receipt


def write_index(path, value, files):
    return save_json(path, {**value, "files": {
        name: sha256_file(source) for name, source in sorted(files.items())
    }})
