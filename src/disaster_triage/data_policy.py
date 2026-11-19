"""Local corpus ingestion and fixed preparation policy; never log private rows."""
from collections import Counter
from dataclasses import dataclass
import csv
import html
import io
from pathlib import Path, PurePosixPath
import re
import tarfile
import unicodedata

from .constants import ARCHIVE_SHA256, EVENTS, LABELS
from .provenance import json_bytes, sha256_bytes, sha256_file

ARCHIVE_NAME = "HumAID_data_events_set1_47K.tar.gz"
EVENT_ROWS = {
    "canada_wildfires_2016": 2242, "ecuador_earthquake_2016": 1563,
    "hurricane_irma_2017": 9399, "hurricane_maria_2017": 7278,
    "italy_earthquake_aug_2016": 1201, "kaikoura_earthquake_2016": 2195,
    "maryland_floods_2018": 742, "srilanka_floods_2017": 560,
    "greece_wildfires_2018": 1515, "hurricane_harvey_2017": 9112,
    "cyclone_idai_2019": 3933, "puebla_mexico_earthquake_2017": 2015,
    "hurricane_matthew_2016": 1654,
}
EVENT_PARTITION = {e: p for p, events in EVENTS.items() for e in events}
DEV_ROLES = ("model_selection", "calibration", "policy")
DATA_ROLES = ("train", *DEV_ROLES, "final_test")
METADATA_MEMBERS = {
    f"events_set1/{name}" for name in (
        "Licensing.txt", "Readme.txt", ".DS_Store", "._.DS_Store", "._Licensing.txt", "._Readme.txt")
}
POLICY = {
    "version": "slice3-v1", "labels": list(LABELS),
    "events": {p: list(es) for p, es in EVENTS.items()},
    "cleanup": ["html_unescape", "NFKC", "URL_USER", "optional_lowercase", "whitespace"],
    "url_pattern": r"https?://\S+|www\.\S+",
    "mention_pattern": r"(?<!\w)@\w+",
    "minimum_near_length": 30, "shingle_length": 5,
    "num_perm": 128, "minhash_seed": 42, "lsh_threshold": .8,
    "lsh_weights": [.5, .5], "jaccard_numerator": 9, "jaccard_denominator": 10,
    "audit_per_stratum": 3, "audit_census_label": "missing_or_found_people",
    "audit_hash_domain": "audit-v1", "audit_reviewers": 1,
    "audit_exclusions": ["non_english", "unusable"],
    "suspected_annotation_errors": "retain_supplied_label",
    "role_seed": 42, "role_weights": [3, 1, 1],
    "role_tie_order": list(DEV_ROLES),
    "metadata_members": sorted(METADATA_MEMBERS),
}
POLICY_SHA256 = sha256_bytes(json_bytes(POLICY))


@dataclass(frozen=True)
class CorpusSpec:
    """Synthetic specs are for invented integration fixtures, never research CLI."""
    archive_sha256: str = ARCHIVE_SHA256
    archive_bytes: int = 2641818
    event_rows: tuple = tuple(sorted(EVENT_ROWS.items()))
    kind: str = "research"

    def __post_init__(self):
        if self.kind not in ("research", "synthetic"):
            raise ValueError("invalid corpus kind")
        if self.kind == "research" and (
            self.archive_sha256 != ARCHIVE_SHA256 or self.archive_bytes != 2641818
            or self.event_rows != tuple(sorted(EVENT_ROWS.items()))
        ):
            raise ValueError("research snapshot is immutable")
        counts = dict(self.event_rows)
        if len(counts) != len(self.event_rows) or not counts or any(
            e not in EVENT_PARTITION or type(n) is not int or n < 0
            for e, n in self.event_rows
        ):
            raise ValueError("invalid event inventory")


RESEARCH_SPEC = CorpusSpec()


def private_path(root, relative):
    """Confine every input/output to this root's ignored data tree; reject links."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "data":
        raise ValueError("data paths must be local to the execution root")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("data paths cannot contain symlinks")
    if not cursor.resolve().is_relative_to(root):
        raise ValueError("data path escapes execution root")
    return cursor


def resolve_private(root, path):
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path(root).resolve())
        except ValueError:
            raise ValueError("data input is outside execution root") from None
    return private_path(root, path)


def text_views(text):
    value = unicodedata.normalize("NFKC", html.unescape(text))
    value = re.sub(POLICY["url_pattern"], "URL", value, flags=re.IGNORECASE)
    value = re.sub(POLICY["mention_pattern"], "USER", value)
    transformer = " ".join(value.split())
    return transformer.lower(), transformer


@dataclass(frozen=True)
class SourceRow:
    source_key: str
    member: str
    line: int
    source_id: str
    event_id: str
    y_true: int
    raw_text: str
    classical_text: str
    transformer_text: str

    @property
    def numeric_id(self):
        return int(self.source_id)

    @property
    def partition(self):
        return EVENT_PARTITION[self.event_id]

    def metadata(self):
        return {"source_key": self.source_key, "member": self.member, "line": self.line,
                "source_id": self.source_id, "event_id": self.event_id,
                "y_true": self.y_true, "partition": self.partition}


def row_order(row):
    return row.event_id, row.numeric_id, row.member, row.line


@dataclass
class Ingested:
    rows: list
    exclusions: list
    inventory: dict


def read_archive(path, spec=RESEARCH_SPEC):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size != spec.archive_bytes or sha256_file(path) != spec.archive_sha256:
        raise ValueError("archive identity mismatch")
    rows, exclusions, members, metadata_members, seen, data_members = [], [], [], [], set(), set()
    event_counts, partition_counts = Counter(), Counter()
    class_counts = Counter()
    expected_events = dict(spec.event_rows)
    event_counts.update({event: 0 for event in expected_events})
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or member.name in seen or member.issym() or member.islnk()):
                raise ValueError("unsafe or duplicate archive member")
            seen.add(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("unsupported archive member")
            if member.name in METADATA_MEMBERS:
                metadata_members.append({"member": member.name, "bytes": member.size,
                                         "sha256": sha256_bytes(archive.extractfile(member).read())})
                continue
            match = re.fullmatch(r"(.+)_(train|dev|test)\.tsv", name.name)
            if not match or match[1] not in expected_events:
                raise ValueError("unexpected data member")
            event, official_split = match.groups()
            if (event, official_split) in data_members:
                raise ValueError("duplicate canonical data member")
            data_members.add((event, official_split))
            stream = io.TextIOWrapper(archive.extractfile(member), encoding="utf-8-sig", newline="")
            reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            if next(reader, None) != ["tweet_id", "tweet_text", "class_label"]:
                raise ValueError("incompatible member header")
            count = 0
            for line, values in enumerate(reader, 2):
                count += 1
                event_counts[event] += 1
                partition_counts[EVENT_PARTITION[event]] += 1
                key = f"{member.name}:{line}"
                base = {"source_key": key, "member": member.name, "line": line,
                        "event_id": event, "partition": EVENT_PARTITION[event]}
                reasons = []
                if len(values) != 3:
                    reasons.append("malformed_fields")
                else:
                    source_id, text, label = values
                    valid_id = re.fullmatch(r"[0-9]+", source_id) is not None
                    if not valid_id:
                        reasons.append("invalid_identifier")
                    if label not in LABELS:
                        reasons.append("invalid_label")
                    else:
                        class_counts[(event, LABELS.index(label))] += 1
                    if not text.strip():
                        reasons.append("blank_text")
                    classical, transformer = text_views(text)
                    if not transformer:
                        reasons.append("empty_cleaned_text")
                    if not reasons:
                        rows.append(SourceRow(key, member.name, line, source_id, event,
                                              LABELS.index(label), text, classical, transformer))
                if reasons:
                    exclusions.append({**base, "stage": "structural", "reasons": reasons})
            members.append({"member": member.name, "event_id": event,
                            "official_split": official_split, "rows": count,
                            "sha256": sha256_bytes(archive.extractfile(member).read())})
    if data_members != {(e, s) for e in expected_events for s in ("train", "dev", "test")}:
        raise ValueError("incomplete member inventory")
    if dict(event_counts) != expected_events:
        raise ValueError("acquired event counts drifted")
    inventory = {
        "archive_sha256": spec.archive_sha256, "archive_bytes": spec.archive_bytes,
        "corpus_kind": spec.kind, "acquired_rows": sum(event_counts.values()),
        "structurally_valid_rows": len(rows), "structural_exclusions": len(exclusions),
        "event_counts": dict(sorted(event_counts.items())),
        "partition_counts": {p: partition_counts[p] for p in EVENTS},
        "event_class_counts": {e: [class_counts[e, c] for c in range(10)] for e in sorted(expected_events)},
        "members": sorted(members, key=lambda x: x["member"]),
        "metadata_members": sorted(metadata_members, key=lambda x: x["member"]),
    }
    if sha256_file(path) != spec.archive_sha256:
        raise ValueError("archive changed while reading")
    return Ingested(sorted(rows, key=row_order), sorted(exclusions, key=lambda x: x["source_key"]), inventory)
