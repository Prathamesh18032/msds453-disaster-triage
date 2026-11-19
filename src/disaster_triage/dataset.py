"""Immutable local preparation artifacts and verified, role-limited dataset reads."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

from .bundles import RoleRow, role_bytes
from .constants import EVENTS, LABELS
from .data_audit import (apply_decisions, audit_binding, audit_sample, audit_summary,
                         decision_template, validate_decisions, worksheet_html)
from .data_duplicates import duplicate_components
from .data_policy import (ARCHIVE_NAME, DATA_ROLES, DEV_ROLES, POLICY, POLICY_SHA256,
                          RESEARCH_SPEC, CorpusSpec, private_path, read_archive,
                          resolve_private, row_order)
from .provenance import (atomic_bytes, json_bytes, sha256_bytes, sha256_file,
                         source_snapshot)


class FeasibilityError(ValueError):
    def __init__(self, report):
        super().__init__("training class absent; a feasibility amendment is required")
        self.report = report


def _json_load(content):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite JSON value")
    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)


def _jsonl(rows):
    return b"".join(json_bytes(row) for row in rows)


def _runtime():
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(),
            "packages": {name: version(name) for name in (
                "disaster-triage", "datasketch", "numpy", "scipy", "scikit-learn",
                "torch", "transformers", "huggingface-hub", "safetensors")}}


def _provenance(source_root, stage):
    snapshot = source_snapshot(source_root, stage / "source.tar")
    return {"source": snapshot, "runtime": _runtime()}


def _archive(root):
    return private_path(root, Path("data/raw") / ARCHIVE_NAME)


def inspect_data(root):
    return read_archive(_archive(root)).inventory


def _audit_files(ingested, spec):
    sample, counts = audit_sample(ingested.rows)
    content, binding = audit_binding(sample, spec.archive_sha256, spec.kind)
    template = decision_template(sample, binding)
    return sample, counts, content, binding, template


def prepare_audit(root, source_root, *, spec=RESEARCH_SPEC):
    started = time.monotonic()
    ingested = read_archive(_archive(root), spec)
    sample, counts, content, binding, template = _audit_files(ingested, spec)
    if not sample:
        raise ValueError("no audit sample is available")
    base = private_path(root, "data/derived/audits")
    target = private_path(root, Path("data/derived/audits") / binding["audit_id"])
    base.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=base))
        try:
            files = {"sample.jsonl": content, "decisions-template.json": json_bytes(template),
                     "review.html": worksheet_html(sample, template), "policy.json": json_bytes(POLICY),
                     "inventory.json": json_bytes(ingested.inventory),
                     "provenance.json": json_bytes(_provenance(source_root, stage))}
            for name, data in files.items():
                atomic_bytes(stage / name, data)
            hashes = {p.name: sha256_file(p) for p in sorted(stage.iterdir())}
            descriptor = {**binding, **counts, "files": hashes, "status": "audit_pending"}
            atomic_bytes(stage / "audit.json", json_bytes(descriptor))
            atomic_bytes(stage / "publication.json", json_bytes({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "preparation_seconds": time.monotonic() - started,
                "accounting": "CPU data preparation; separate from experiment-job ledger"}))
            try:
                os.rename(stage, target)
            except OSError:
                if not target.is_dir():
                    raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    descriptor = _json_load((target / "audit.json").read_bytes())
    if any(descriptor.get(k) != v for k, v in binding.items()):
        raise ValueError("existing audit binding differs")
    for name, digest in descriptor["files"].items():
        path = private_path(root, target.relative_to(Path(root).resolve()) / name)
        if sha256_file(path) != digest:
            raise ValueError("existing audit artifact differs")
    if (target / "sample.jsonl").read_bytes() != content:
        raise ValueError("existing audit sample differs")
    return {"status": "audit_pending", **binding, **counts,
            "worksheet": str(target / "review.html"),
            "decision_template": str(target / "decisions-template.json"),
            "audit_manifest": str(target / "audit.json")}


def allocation_counts(n):
    counts = [n * weight // 5 for weight in (3, 1, 1)]
    order = sorted(range(3), key=lambda i: (-(n * (3, 1, 1)[i] % 5), i))
    for index in order[:n - sum(counts)]:
        counts[index] += 1
    return counts


def allocate_roles(kept):
    strata = defaultdict(list)
    result = {}
    for row, group_id in kept:
        if row.partition == "development":
            strata[row.event_id, row.y_true].append((row, group_id))
        else:
            result[row.source_key] = row.partition
    for (event, label), values in sorted(strata.items()):
        def rank(item):
            row, _ = item
            text = f"42|{event}|{LABELS[label]}|{row.numeric_id}"
            return sha256_bytes(text.encode("utf-8")), row.source_key
        ordered = sorted(values, key=rank)
        cursor = 0
        for role, count in zip(DEV_ROLES, allocation_counts(len(ordered))):
            for row, _ in ordered[cursor:cursor + count]:
                result[row.source_key] = role
            cursor += count
    return result


def support_tables(inventory, processed):
    event_counts, partition_counts, role_counts, dev_counts = Counter(), Counter(), Counter(), Counter()
    for row in processed:
        c, event, partition, role = row["y_true"], row["event_id"], row["partition"], row["role"]
        event_counts[event, c] += 1
        partition_counts[partition, c] += 1
        role_counts[role, c] += 1
        dev_counts[event, role, c] += 1
    event_records = []
    partition_records = []
    for partition, events in EVENTS.items():
        for c, label in enumerate(LABELS):
            acquired = sum(inventory["event_class_counts"].get(e, [0] * 10)[c] for e in events)
            partition_records.append({"partition": partition, "class_label": label,
                                      "acquired": acquired, "eligible": partition_counts[partition, c]})
        for event in events:
            for c, label in enumerate(LABELS):
                event_records.append({"event_id": event, "partition": partition, "class_label": label,
                                      "acquired": inventory["event_class_counts"].get(event, [0] * 10)[c],
                                      "eligible": event_counts[event, c]})
    focus = []
    for c in (4, 7):
        for category, entities, field in (("partition", EVENTS, "partition"), ("role", DATA_ROLES, "role")):
            for entity in entities:
                matching = [r for r in processed if r[field] == entity and r["y_true"] == c]
                focus.append({"level": category, "entity": entity, "class_label": LABELS[c],
                              "eligible": len(matching),
                              "contributing_events": len({r["event_id"] for r in matching})})
    return {"event_class": event_records, "partition_class": partition_records,
            "role_class": [{"role": role, "class_label": label, "eligible": role_counts[role, c]}
                           for role in DATA_ROLES for c, label in enumerate(LABELS)],
            "development_event_role_class": [
                {"event_id": event, "role": role, "class_label": label, "eligible": dev_counts[event, role, c]}
                for event in EVENTS["development"] for role in DEV_ROLES for c, label in enumerate(LABELS)],
            "focus_classes": focus}


def _assemble(ingested, decision_value, spec):
    sample, _, sample_content, binding, _ = _audit_files(ingested, spec)
    decisions = validate_decisions(decision_value, sample, binding)
    candidates, audit_exclusions = apply_decisions(ingested.rows, decisions)
    kept, duplicate_exclusions, components, edges, duplicate_metadata = duplicate_components(candidates)
    roles = allocate_roles(kept)
    processed, eligible = [], []
    for row, group_id in kept:
        base = {**row.metadata(), "row_id": f"humaid:{row.numeric_id}",
                "group_id": group_id, "role": roles[row.source_key]}
        processed.append({**base, "classical_text": row.classical_text,
                          "transformer_text": row.transformer_text})
        eligible.append({**base, "classical_sha256": sha256_bytes(row.classical_text.encode("utf-8")),
                         "transformer_sha256": sha256_bytes(row.transformer_text.encode("utf-8"))})
    counts = [sum(r["role"] == "train" and r["y_true"] == c for r in processed) for c in range(10)]
    if any(n == 0 for n in counts):
        raise FeasibilityError({"status": "feasibility_amendment_required",
                                "training_counts": counts,
                                "absent_training_labels": [LABELS[c] for c, n in enumerate(counts) if n == 0],
                                "acquired_rows": ingested.inventory["acquired_rows"],
                                "candidate_eligible_rows": len(processed)})
    n_train = sum(counts)
    weights = [n_train / (10 * n) for n in counts]
    exclusions = sorted([*ingested.exclusions, *audit_exclusions, *duplicate_exclusions],
                        key=lambda x: x["source_key"])
    dispositions = [{"source_key": r["source_key"], "status": "excluded", "stage": r["stage"],
                     "reasons": r["reasons"], "group_id": r.get("group_id")} for r in exclusions]
    dispositions.extend({"source_key": r["source_key"], "status": "retained", "stage": "eligible",
                         "reasons": [], "group_id": r["group_id"]} for r in eligible)
    dispositions.sort(key=lambda x: x["source_key"])
    if len(dispositions) != ingested.inventory["acquired_rows"] or len({r["source_key"] for r in dispositions}) != len(dispositions):
        raise ValueError("source dispositions do not reconcile")
    summary = {"acquired_rows": ingested.inventory["acquired_rows"], "eligible_rows": len(processed),
               "structural_exclusions": len(ingested.exclusions), "audit_exclusions": len(audit_exclusions),
               "duplicate_exclusions": len(duplicate_exclusions),
               "role_counts": {role: sum(r["role"] == role for r in processed) for role in DATA_ROLES},
               "training_counts": counts, "training_weights": weights}
    files = {"inventory.json": json_bytes(ingested.inventory),
             "processed.jsonl": _jsonl(processed), "eligible-manifest.jsonl": _jsonl(eligible),
             "exclusions.jsonl": _jsonl(exclusions), "dispositions.jsonl": _jsonl(dispositions),
             "components.jsonl": _jsonl(components), "edges.jsonl": _jsonl(edges),
             "duplicate-metadata.json": json_bytes(duplicate_metadata),
             "support.json": json_bytes(support_tables(ingested.inventory, processed)),
             "audit-sample.jsonl": sample_content, "audit-decisions.json": json_bytes(decision_value),
             "audit-summary.json": json_bytes(audit_summary(sample, decisions)), "policy.json": json_bytes(POLICY)}
    for role in DATA_ROLES:
        rows = tuple(RoleRow(r["row_id"], r["event_id"], r["y_true"])
                     for r in processed if r["role"] == role)
        files[f"roles/{role}.jsonl"] = role_bytes(rows)
        files[f"texts/{role}.jsonl"] = _jsonl(r for r in processed if r["role"] == role)
    return files, summary


def _identity(descriptor):
    return {k: v for k, v in descriptor.items() if k not in ("freeze_id", "complete")}


def freeze_data(root, source_root, decisions_path, *, spec=RESEARCH_SPEC):
    started = time.monotonic()
    ingested = read_archive(_archive(root), spec)
    decision_value = _json_load(resolve_private(root, decisions_path).read_bytes())
    try:
        files, summary = _assemble(ingested, decision_value, spec)
    except FeasibilityError as error:
        report = {**error.report, "archive_sha256": spec.archive_sha256,
                  "policy_sha256": POLICY_SHA256,
                  "decisions_sha256": sha256_bytes(json_bytes(decision_value))}
        content = json_bytes(report)
        report_path = private_path(root, Path("data/derived/feasibility") / (sha256_bytes(content) + ".json"))
        atomic_bytes(report_path, content)
        raise
    base = private_path(root, "data/processed/freezes")
    base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=base))
    try:
        files["provenance.json"] = json_bytes(_provenance(source_root, stage))
        files["source.tar"] = (stage / "source.tar").read_bytes()
        for name, content in files.items():
            atomic_bytes(stage / name, content)
        descriptor = {"schema_version": 1, "corpus_kind": spec.kind,
                      "archive_sha256": spec.archive_sha256, "archive_bytes": spec.archive_bytes,
                      "event_rows": dict(spec.event_rows), "policy_sha256": POLICY_SHA256,
                      "labels": list(LABELS), "events": {p: list(es) for p, es in EVENTS.items()},
                      "files": {name: sha256_bytes(content) for name, content in sorted(files.items())},
                      "summary": summary}
        descriptor["freeze_id"] = sha256_bytes(json_bytes(descriptor))
        descriptor["complete"] = True
        target = private_path(root, Path("data/processed/freezes") / descriptor["freeze_id"])
        atomic_bytes(stage / "publication.json", json_bytes({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "preparation_seconds": time.monotonic() - started,
            "accounting": "CPU data preparation; separate from experiment-job ledger"}))
        atomic_bytes(stage / "freeze.json", json_bytes(descriptor))
        if target.exists():
            if (target / "freeze.json").read_bytes() != json_bytes(descriptor):
                raise ValueError("existing freeze differs")
        else:
            try:
                os.rename(stage, target)
            except OSError:
                if not target.is_dir() or (target / "freeze.json").read_bytes() != json_bytes(descriptor):
                    raise
        return verify_freeze(root, target, mode=spec.kind)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def verify_freeze(root, freeze, *, mode="research"):
    directory = resolve_private(root, freeze)
    descriptor_path = resolve_private(root, directory / "freeze.json")
    descriptor = _json_load(descriptor_path.read_bytes())
    expected_fields = {"schema_version", "corpus_kind", "archive_sha256", "archive_bytes", "event_rows",
                       "policy_sha256", "labels", "events", "files", "summary", "freeze_id", "complete"}
    if (not isinstance(descriptor, dict) or set(descriptor) != expected_fields
            or type(descriptor["schema_version"]) is not int or descriptor["schema_version"] != 1
            or descriptor["complete"] is not True or descriptor["corpus_kind"] != mode):
        raise ValueError("freeze is incomplete or has the wrong corpus kind")
    if descriptor["freeze_id"] != sha256_bytes(json_bytes(_identity(descriptor))) or directory.name != descriptor["freeze_id"]:
        raise ValueError("freeze identity mismatch")
    if (descriptor["labels"] != list(LABELS) or descriptor["policy_sha256"] != POLICY_SHA256
            or descriptor["events"] != {p: list(es) for p, es in EVENTS.items()}):
        raise ValueError("freeze policy mismatch")
    spec = CorpusSpec(descriptor["archive_sha256"], descriptor["archive_bytes"],
                      tuple(sorted(descriptor["event_rows"].items())), descriptor["corpus_kind"])
    ingested = read_archive(_archive(root), spec)
    decisions_path = resolve_private(root, directory / "audit-decisions.json")
    decisions = _json_load(decisions_path.read_bytes())
    expected, summary = _assemble(ingested, decisions, spec)
    names = set(expected) | {"source.tar", "provenance.json"}
    if not isinstance(descriptor["files"], dict) or set(descriptor["files"]) != names:
        raise ValueError("freeze payload inventory mismatch")
    for name, digest in descriptor["files"].items():
        path = resolve_private(root, directory / name)
        if sha256_file(path) != digest:
            raise ValueError("freeze payload digest mismatch")
        if name in expected and path.read_bytes() != expected[name]:
            raise ValueError("freeze lineage, text, component, role, or support invariant failed")
    provenance = _json_load(resolve_private(root, directory / "provenance.json").read_bytes())
    if provenance["source"]["code_sha256"] != descriptor["files"]["source.tar"]:
        raise ValueError("freeze source provenance mismatch")
    if descriptor["summary"] != summary:
        raise ValueError("freeze summary mismatch")
    return {"status": "verified", "freeze_id": descriptor["freeze_id"], "freeze": str(directory),
            "freeze_manifest_sha256": sha256_file(descriptor_path),
            "eligible_manifest_sha256": descriptor["files"]["eligible-manifest.jsonl"],
            "role_manifest_sha256": {role: descriptor["files"][f"roles/{role}.jsonl"] for role in DATA_ROLES},
            **summary}


class FrozenDataset:
    """Verify the entire preparation structurally; expose only permitted role text."""
    def __init__(self, root, freeze, *, mode="research"):
        self.root = Path(root).resolve()
        self.directory = resolve_private(root, freeze)
        self.verification = verify_freeze(self.root, self.directory, mode=mode)
        self._descriptor_bytes = (self.directory / "freeze.json").read_bytes()
        self.descriptor = _json_load(self._descriptor_bytes)

    def load_role(self, role):
        if role == "final_test":
            raise ValueError("final-test content loading awaits the Slice 8 selection/scope lock")
        if role not in ("train", *DEV_ROLES):
            raise ValueError("unsupported dataset role")
        if resolve_private(self.root, self.directory / "freeze.json").read_bytes() != self._descriptor_bytes:
            raise ValueError("freeze changed after verification")
        path = resolve_private(self.root, self.directory / f"texts/{role}.jsonl")
        if sha256_file(path) != self.descriptor["files"][f"texts/{role}.jsonl"]:
            raise ValueError("dataset changed after verification")
        rows = []
        with path.open("rb") as stream:
            for line in stream:
                row = _json_load(line)
                if row["role"] == role:
                    rows.append(row)
        return tuple(rows)
