"""Invented local fixtures only; no acquired text or identifiers."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

import pytest

from disaster_triage import dataset
from disaster_triage.bundles import BundleExpectation, RoleRow, role_bytes
from disaster_triage.cli import main
from disaster_triage.constants import ARCHIVE_SHA256, EVENTS, LABELS, ROLES
from disaster_triage.data_audit import (apply_decisions, audit_binding, audit_sample,
                                      decision_template, validate_decisions, worksheet_html)
from disaster_triage.data_duplicates import duplicate_components, qualifies
from disaster_triage.data_policy import (ARCHIVE_NAME, CorpusSpec, SourceRow, private_path,
                                        read_archive, text_views)
from disaster_triage.provenance import json_bytes, sha256_bytes

TRAIN = EVENTS["train"][0]
DEV = EVENTS["development"][0]
FINAL = EVENTS["final_test"][0]


def row(number, event=TRAIN, label=0, text=None, member=None, line=None):
    text = text or hashlib.sha256(f"invented-{number}".encode()).hexdigest()
    member = member or f"events_set1/{event}_train.tsv"
    line = line or int(number) + 2
    classical, transformer = text_views(text)
    return SourceRow(f"{member}:{line}", member, line, str(number), event, label,
                     text, classical, transformer)


def write_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries:
            if isinstance(content, tarfile.TarInfo):
                archive.addfile(content)
            else:
                content = content.encode() if isinstance(content, str) else content
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def corpus(tmp_path, records=None):
    records = records or ([row(10 + c, label=c) for c in range(10)] +
                          [row(100 + 5*c + i, DEV, c) for c in range(10) for i in range(5)] +
                          [row(1000 + c, FINAL, c) for c in range(10)])
    events = sorted({r.event_id for r in records})
    entries = []
    for event in events:
        for split in ("train", "dev", "test"):
            name = f"events_set1/nesting/{event}_{split}.tsv"
            content = "tweet_id\ttweet_text\tclass_label\n"
            if split == "train":
                content += "".join(f"{r.source_id}\t{r.raw_text}\t{LABELS[r.y_true]}\n"
                                   for r in records if r.event_id == event)
            entries.append((name, content))
    target = tmp_path / "data/raw" / ARCHIVE_NAME
    target.parent.mkdir(parents=True)
    write_archive(target, entries)
    spec = CorpusSpec(sha256_bytes(target.read_bytes()), target.stat().st_size,
                      tuple((e, sum(r.event_id == e for r in records)) for e in events), "synthetic")
    source = tmp_path / "code"
    (source / "src").mkdir(parents=True)
    (source / "src/fixture.py").write_text('"""Invented source snapshot."""\n')
    return target, spec, source


def completed(sample, binding):
    result = decision_template(sample, binding)
    result.update(reviewer="Fixture reviewer", reviewed_at="2026-11-18", protocol_change_requested=False)
    for r in result["rows"]:
        r.update(language="english", usability="usable", annotation="plausible")
    return result


def decisions_for(root, spec):
    ingested = read_archive(root / "data/raw" / ARCHIVE_NAME, spec)
    sample, _ = audit_sample(ingested.rows)
    _, binding = audit_binding(sample, spec.archive_sha256, spec.kind)
    value = completed(sample, binding)
    path = root / "data/derived/decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))
    return path, value


@pytest.mark.parametrize("text,expected", [
    ("NOT &amp; １２ #Help @person https://example.invalid/a  7", ("not & 12 #help user url 7", "NOT & 12 #Help USER URL 7")),
    ("  \t\n", ("", "")), ("I can't go. 42", ("i can't go. 42", "I can't go. 42")),
    ("WWW.Example.invalid @x", ("url user", "URL USER")),
])
def test_text_views(text, expected):
    assert text_views(text) == expected


def test_canonical_events_literal_quotes_and_inventory(tmp_path):
    values = [row(i + 1, event, i % 10, f'Invented "quote {i}')
              for i, event in enumerate(e for events in EVENTS.values() for e in events)]
    path, spec, _ = corpus(tmp_path, values)
    actual = read_archive(path, spec)
    assert len(actual.rows) == 13
    assert len(actual.inventory["members"]) == 39
    assert {r.event_id for r in actual.rows} == {r.event_id for r in values}
    assert sum(actual.inventory["partition_counts"].values()) == 13
    assert all('"quote' in r.raw_text for r in actual.rows)


@pytest.mark.parametrize("mutation", ["path", "symlink", "duplicate", "extra", "header", "missing", "count", "hash"])
def test_archive_rejection(tmp_path, mutation):
    name = f"events_set1/{TRAIN}_train.tsv"
    good = "tweet_id\ttweet_text\tclass_label\n1\tInvented\t" + LABELS[0] + "\n"
    entries = [(name, good), (name.replace("_train", "_dev"), good.splitlines()[0] + "\n"),
               (name.replace("_train", "_test"), good.splitlines()[0] + "\n")]
    if mutation == "path": entries.append(("../bad", ""))
    if mutation == "symlink":
        info = tarfile.TarInfo("link"); info.type = tarfile.SYMTYPE; info.linkname = "target"
        entries.append(("link", info))
    if mutation == "duplicate": entries.append(entries[0])
    if mutation == "extra": entries.append(("unrecognized.txt", ""))
    if mutation == "header": entries[0] = (name, "incorrect\n")
    if mutation == "missing": entries.pop()
    path = tmp_path / "archive.tar.gz"
    write_archive(path, entries)
    spec = CorpusSpec("0" * 64 if mutation == "hash" else sha256_bytes(path.read_bytes()),
                      path.stat().st_size, ((TRAIN, 2 if mutation == "count" else 1),), "synthetic")
    with pytest.raises(ValueError):
        read_archive(path, spec)


def test_row_exclusion_reasons_and_canonical_numeric_ids(tmp_path):
    header = "tweet_id\ttweet_text\tclass_label\n"
    text = (header + f"001\tInvented\t{LABELS[0]}\n" + f"x\tValid text\t{LABELS[0]}\n" +
            f"2\t \t{LABELS[0]}\n" + "3\tValid\tunknown\n" + "broken\tfields\n")
    entries = [(f"events_set1/{TRAIN}_{s}.tsv", text if s == "train" else header)
               for s in ("train", "dev", "test")]
    path = tmp_path / "fixture.tar.gz"; write_archive(path, entries)
    spec = CorpusSpec(sha256_bytes(path.read_bytes()), path.stat().st_size, ((TRAIN, 5),), "synthetic")
    ingested = read_archive(path, spec)
    assert ingested.rows[0].source_id == "001" and ingested.rows[0].numeric_id == 1
    assert len(ingested.exclusions) == 4
    assert {reason for r in ingested.exclusions for reason in r["reasons"]} == {
        "invalid_identifier", "blank_text", "empty_cleaned_text", "invalid_label", "malformed_fields"}


def test_sampling_is_stable_stratified_and_includes_rare_census():
    rows = [row(i + 1, DEV, 0) for i in range(12)] + [row(i + 100, DEV, 4) for i in range(9)]
    sample, counts = audit_sample(rows)
    assert counts == {"sample_rows": 12, "stratified_rows": 6, "rare_census_rows": 9, "overlap_rows": 3}
    assert sample == audit_sample(list(reversed(rows)))[0]
    assert sum(r["y_true"] == 0 for r in sample) == 3
    assert sum(r["y_true"] == 4 for r in sample) == 9


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate", "stale", "unfinished", "reviewer", "date", "amendment", "flag_notes", "reviewer_count"])
def test_decision_gate(mutation):
    sample, _ = audit_sample([row(1)])
    _, binding = audit_binding(sample, "a" * 64, "synthetic")
    value = completed(sample, binding)
    if mutation == "missing": value["rows"] = []
    if mutation == "unknown": value["rows"][0]["source_key"] = "unknown"
    if mutation == "duplicate": value["rows"] *= 2
    if mutation == "stale": value["sample_sha256"] = "0" * 64
    if mutation == "unfinished": value["rows"][0]["language"] = None
    if mutation == "reviewer": value["reviewer"] = ""
    if mutation == "date": value["reviewed_at"] = "2026-02-30"
    if mutation == "amendment": value["protocol_change_requested"] = True
    if mutation == "flag_notes": value["rows"][0]["annotation"] = "questionable"
    if mutation == "reviewer_count": value["reviewer_count"] = True
    with pytest.raises(ValueError): validate_decisions(value, sample, binding)


def test_individual_audit_exclusions_retain_uncertainty_and_labels():
    rows = [row(i, label=i) for i in range(1, 5)]
    sample, _ = audit_sample(rows)
    _, binding = audit_binding(sample, "b" * 64, "synthetic")
    value = completed(sample, binding)
    for r in value["rows"]:
        r.update(reason_codes=["fixture_flag"], notes="Invented audit finding")
    value["rows"][0]["language"] = "non_english"
    value["rows"][1]["usability"] = "unusable"
    value["rows"][2]["language"] = "mixed_uncertain"
    value["rows"][3]["annotation"] = "questionable"
    actual = validate_decisions(value, sample, binding)
    retained, excluded = apply_decisions(rows, actual)
    assert retained == rows[2:] and [r.y_true for r in retained] == [3, 4]
    assert len(excluded) == 2


def test_offline_worksheet_escapes_private_html():
    sample, _ = audit_sample([row(1, text='</script><img src="https://invalid">')])
    _, binding = audit_binding(sample, "b" * 64, "synthetic")
    content = worksheet_html(sample, decision_template(sample, binding)).decode()
    assert '</script><img' not in content
    assert "connect-src 'none'" in content
    assert 'id="annotationPanel" hidden' in content
    assert "localStorage" not in content


def test_components_exclude_transitive_cross_event_and_conflicting_labels():
    rows = [row(9, text="copy"), row(10, text="copy"), row(10, DEV, text="different", line=99),
            row(20, label=1, text="conflict"), row(21, label=2, text="conflict"),
            row(3, text="consistent"), row(2, text="consistent"), row(30, text="unique")]
    kept, excluded, groups, edges, _ = duplicate_components(rows)
    assert [r.numeric_id for r, _ in kept] == [2, 30]
    assert len(excluded) == 6
    assert any(len(g["members"]) == 3 and g["exclusion_reasons"] == ["cross_event_component"] for g in groups)
    assert any(g["exclusion_reasons"] == ["conflicting_label_component"] for g in groups)
    assert duplicate_components(list(reversed(rows)))[:4] == (kept, excluded, groups, edges)


def test_equal_id_representative_tie_breaks_by_member_and_line():
    rows = [row(1, text="first", member="z", line=2), row(1, text="second", member="a", line=3),
            row(1, text="third", member="a", line=2)]
    kept, *_ = duplicate_components(rows)
    assert kept[0][0].raw_text == "third"


def test_exact_jaccard_threshold():
    assert qualifies(set(range(9)), set(range(10)))
    assert not qualifies(set(range(8)), set(range(10)))
    assert not qualifies(set(), set())


def test_near_candidate_confirmation_and_length_boundary(monkeypatch):
    import datasketch
    class EveryCandidate:
        b, r = 1, 1
        def __init__(self, **kwargs): self.keys = []
        def query(self, signature): return self.keys
        def insert(self, key, signature): self.keys.append(key)
    monkeypatch.setattr(datasketch, "MinHashLSH", EveryCandidate)
    rows = [row(1, text="a" * 29), row(2, text="a" * 30), row(3, text="a" * 31),
            row(4, text="b" * 31)]
    kept, excluded, _, edges, meta = duplicate_components(rows)
    assert meta["candidate_pairs"] == 3 and meta["confirmed_near_pairs"] == 1
    assert len(kept) == 3 and len(excluded) == 1
    assert edges == [{"left": rows[1].source_key, "right": rows[2].source_key, "reason": "confirmed_near_duplicate"}]


@pytest.mark.parametrize("n,expected", [(0,[0,0,0]), (1,[1,0,0]), (2,[1,1,0]), (3,[2,1,0]),
                                       (4,[2,1,1]), (5,[3,1,1]), (7,[4,2,1])])
def test_largest_remainders(n, expected):
    assert dataset.allocation_counts(n) == expected


def test_preparation_complete_repeat_verify_role_reader_and_tampering(tmp_path):
    _, spec, source = corpus(tmp_path)
    audit = dataset.prepare_audit(tmp_path, source, spec=spec)
    assert audit["status"] == "audit_pending"
    worksheet = Path(audit["worksheet"])
    original = worksheet.read_bytes()
    assert dataset.prepare_audit(tmp_path, source, spec=spec)["audit_id"] == audit["audit_id"]
    assert worksheet.read_bytes() == original
    decisions, _ = decisions_for(tmp_path, spec)
    result = dataset.freeze_data(tmp_path, source, decisions, spec=spec)
    assert result["eligible_rows"] == 70
    assert result["role_counts"] == {"train":10, "model_selection":30, "calibration":10, "policy":10, "final_test":10}
    directory = Path(result["freeze"])
    assert dataset.freeze_data(tmp_path, source, decisions, spec=spec) == result
    reader = dataset.FrozenDataset(tmp_path, directory, mode="synthetic")
    assert len(reader.load_role("train")) == 10
    with pytest.raises(ValueError, match="Slice 8"): reader.load_role("final_test")
    with pytest.raises(ValueError): dataset.verify_freeze(tmp_path, directory)
    support = json.loads((directory / "support.json").read_bytes())
    assert len(support["event_class"]) == 130 and len(support["partition_class"]) == 30
    assert any(r["eligible"] == 0 for r in support["event_class"])
    role_rows = tuple(RoleRow(r["row_id"], r["event_id"], r["y_true"]) for r in reader.load_role("model_selection"))
    assert role_bytes(role_rows) == (directory / "roles/model_selection.jsonl").read_bytes()
    assert "train" not in ROLES
    # Invented canonical metadata exercises the v1 research expectation contract.
    BundleExpectation("research", "model_selection", role_rows, {
        "archive_sha256": ARCHIVE_SHA256,
        "eligible_manifest_sha256": result["eligible_manifest_sha256"],
        "role_manifest_sha256": result["role_manifest_sha256"]["model_selection"],
        "config_sha256": "a" * 64, "code_sha256": "b" * 64})
    (directory / "texts/train.jsonl").write_bytes(b"corrupt")
    with pytest.raises(ValueError): reader.load_role("train")
    with pytest.raises(ValueError): dataset.verify_freeze(tmp_path, directory, mode="synthetic")


def test_changed_decisions_new_identity_and_preserved_original(tmp_path):
    _, spec, source = corpus(tmp_path)
    path, value = decisions_for(tmp_path, spec)
    first = dataset.freeze_data(tmp_path, source, path, spec=spec)
    prior = (Path(first["freeze"]) / "freeze.json").read_bytes()
    value["reviewer"] = "Another fixture reviewer"
    path.write_bytes(json_bytes(value))
    second = dataset.freeze_data(tmp_path, source, path, spec=spec)
    assert second["freeze_id"] != first["freeze_id"]
    assert (Path(first["freeze"]) / "freeze.json").read_bytes() == prior


def test_zero_training_class_stops_before_publication(tmp_path):
    _, spec, source = corpus(tmp_path, [row(10+c, label=c) for c in range(9)])
    path, _ = decisions_for(tmp_path, spec)
    with pytest.raises(ValueError, match="training class absent"):
        dataset.freeze_data(tmp_path, source, path, spec=spec)
    assert not list(tmp_path.glob("data/processed/freezes/*/freeze.json"))
    reports = list(tmp_path.glob("data/derived/feasibility/*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_bytes())["absent_training_labels"] == [LABELS[9]]


def test_known_os_metadata_is_inventoried_without_becoming_data(tmp_path):
    name = f"events_set1/{TRAIN}"
    header = "tweet_id\ttweet_text\tclass_label\n"
    entries = [(f"{name}_{split}.tsv", header + (f"1\tInvented\t{LABELS[0]}\n" if split == "train" else ""))
               for split in ("train", "dev", "test")]
    entries += [("events_set1/.DS_Store", b"invented metadata"),
                ("events_set1/._Readme.txt", b"invented resource fork")]
    path = tmp_path / "fixture.tar.gz"
    write_archive(path, entries)
    spec = CorpusSpec(sha256_bytes(path.read_bytes()), path.stat().st_size, ((TRAIN,1),), "synthetic")
    actual = read_archive(path, spec)
    assert len(actual.rows) == 1 and len(actual.inventory["metadata_members"]) == 2


def test_incomplete_audit_and_interrupted_publication_leave_no_complete_freeze(tmp_path, monkeypatch):
    _, spec, source = corpus(tmp_path)
    audit = dataset.prepare_audit(tmp_path, source, spec=spec)
    with pytest.raises(ValueError):
        dataset.freeze_data(tmp_path, source, audit["decision_template"], spec=spec)
    path, _ = decisions_for(tmp_path, spec)
    original = dataset.atomic_bytes
    def fail_descriptor(path, content):
        if Path(path).name == "freeze.json": raise OSError("invented interruption")
        return original(path, content)
    monkeypatch.setattr(dataset, "atomic_bytes", fail_descriptor)
    with pytest.raises(OSError): dataset.freeze_data(tmp_path, source, path, spec=spec)
    assert not list(tmp_path.glob("data/processed/freezes/*/freeze.json"))
    assert not list(tmp_path.glob("data/processed/freezes/.pending-*"))


def test_rehashed_semantic_corruption_is_rejected(tmp_path):
    _, spec, source = corpus(tmp_path)
    path, _ = decisions_for(tmp_path, spec)
    result = dataset.freeze_data(tmp_path, source, path, spec=spec)
    directory = Path(result["freeze"])
    support_path = directory / "support.json"
    support = json.loads(support_path.read_bytes())
    support["event_class"][0]["eligible"] += 1
    support_path.write_bytes(json_bytes(support))
    desc_path = directory / "freeze.json"
    desc = json.loads(desc_path.read_bytes())
    desc["files"]["support.json"] = sha256_bytes(support_path.read_bytes())
    desc["freeze_id"] = sha256_bytes(json_bytes(dataset._identity(desc)))
    desc_path.write_bytes(json_bytes(desc))
    renamed = directory.with_name(desc["freeze_id"])
    directory.rename(renamed)
    with pytest.raises(ValueError, match="invariant"):
        dataset.verify_freeze(tmp_path, renamed, mode="synthetic")


def test_input_order_does_not_change_assembled_content(tmp_path):
    path, spec, _ = corpus(tmp_path)
    ingested = read_archive(path, spec)
    _, decisions = decisions_for(tmp_path, spec)
    first = dataset._assemble(ingested, decisions, spec)
    random.Random(99).shuffle(ingested.rows)
    assert dataset._assemble(ingested, decisions, spec) == first


def test_local_paths_and_no_private_cli_diagnostics(tmp_path, capsys):
    with pytest.raises(ValueError): private_path(tmp_path, "../data")
    (tmp_path / "data").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError): private_path(tmp_path, "data/raw")
    assert main(["data", "freeze", "--root", str(tmp_path), "--decisions", "private-sentinel-id.json"]) == 1
    output = capsys.readouterr()
    assert "private-sentinel" not in output.err and "Data preparation failed" in output.err
