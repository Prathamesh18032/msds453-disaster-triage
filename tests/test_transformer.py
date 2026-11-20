import threading
import time
from types import SimpleNamespace
import zipfile

import pytest
import torch

from disaster_triage.bundles import RoleRow, role_bytes
from disaster_triage.constants import LABELS
from disaster_triage.provenance import json_bytes, sha256_bytes
from disaster_triage.transformer import (
    SCREEN_KEYS, do_training, resolved_run, select_screen, steps_per_epoch,
    verify_epoch, verify_rolling,
)
from disaster_triage.transformer_budget import allocation, choose_group, overlap_rate
from disaster_triage.transformer_data import extract_verified, load_inputs
from disaster_triage.transformer_handoff import research_member, result_member


class TinyTokenizer:
    sep_token_id = 2

    def __call__(self, texts, **options):
        return {"input_ids": [[1] + [3] * len(text.split()) + [2] for text in texts]}

    def pad(self, features, **options):
        max_len = max(len(f["input_ids"]) for f in features)
        return {"input_ids": torch.tensor(
            [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in features]
        )}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 5)
        self.dropout = torch.nn.Dropout(.2)
        self.head = torch.nn.Linear(5, 10)

    def forward(self, input_ids):
        values = self.embed(input_ids).sum(dim=1)
        return SimpleNamespace(logits=self.head(self.dropout(values)))


def tiny_assets(_, __, device):
    return TinyTokenizer(), TinyModel().to(device)


@pytest.fixture
def fixture_inputs(tmp_path):
    rows = {
        "train": tuple(RoleRow(f"t-{i}", "synthetic_train", i % 10) for i in range(20)),
        "model_selection": tuple(RoleRow(f"s-{i}", "synthetic_selection", i) for i in range(10)),
    }
    lineage = {
        "rows": {role: len(items) for role, items in rows.items()},
        "role_sha256": {role: sha256_bytes(role_bytes(items)) for role, items in rows.items()},
        "training_counts": [2] * 10, "archive_sha256": "1" * 64, "eligible_manifest_sha256": "2" * 64,
        "freeze_id": "3" * 64, "labels": list(LABELS), "slice8_prediction_rows_per_model": 10,
    }
    for role, items in rows.items():
        (tmp_path / "roles").mkdir(exist_ok=True)
        (tmp_path / "texts").mkdir(exist_ok=True)
        (tmp_path / "roles" / f"{role}.jsonl").write_bytes(role_bytes(items))
        (tmp_path / "texts" / f"{role}.jsonl").write_bytes(b"".join(
            json_bytes({"row_id": r.row_id, "text": "invented example " * (i + 1)})
            for i, r in enumerate(items)
        ))
    return load_inputs(tmp_path, lineage, research=False)


def config_for(inputs, key, *, selection=None):
    return resolved_run(
        key, campaign="tiny", inputs=inputs,
        source={"code_revision": None, "code_sha256": "4" * 64},
        runtime={"versions": {}}, physical_batch=8, device="cpu", selection=selection,
    )


def advance(directory, config, bundle, stage, attempt, latest=None, deadline=None):
    directory.mkdir(exist_ok=True)
    do_training(directory, config, bundle, None, stage, attempt, deadline or time.monotonic() + 15,
                ready_event=threading.Event(), go_event=None, halfway_event=None, latest=latest,
                load_model=tiny_assets)


def test_training_and_new_process_style_resume_same_tensor_updates(tmp_path, fixture_inputs):
    config = config_for(fixture_inputs, SCREEN_KEYS[0])
    pilot_run, uninterrupted = (tmp_path / suffix for suffix in ("pilot_run", "uninterrupted"))
    pilot_attempt, next_attempt = ("a" * 32, "b" * 32)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        advance(pilot_run, config, fixture_inputs, "pilot", pilot_attempt)
        pause = verify_rolling(pilot_run)
        assert pause["progress"] == {"epoch": 1, "optimizer_steps": 2, "next_position": 20}
        assert not (pilot_run / "epochs").exists()
        advance(pilot_run, config, fixture_inputs, "screen", next_attempt, pause)
        advance(uninterrupted, config, fixture_inputs, "screen", "c" * 32)
    finally:
        torch.set_num_threads(previous)
    for epoch in (1, 2, 3):
        left_report, receipt = verify_epoch(pilot_run, config, fixture_inputs, epoch, load_checkpoint=True)
        right_report, _ = verify_epoch(uninterrupted, config, fixture_inputs, epoch, load_checkpoint=True)
        assert left_report["pooled"] == right_report["pooled"]
        if epoch == 1:
            assert receipt["checkpoint_attempt_id"] == pilot_attempt
            assert receipt["attempt_id"] == next_attempt
        outputs = [root / "epochs" / f"e{epoch}" / "checkpoint.pt" for root in (pilot_run, uninterrupted)]
        left, right = [torch.load(path, map_location="cpu", weights_only=True) for path in outputs]
        assert left["progress"] == right["progress"]
        assert left["scheduler"] == right["scheduler"]
        for key, value in left["model"].items():
            torch.testing.assert_close(value, right["model"][key], atol=0, rtol=0)
    path = pilot_run / "epochs/e3/report.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt"):
        verify_epoch(pilot_run, config, fixture_inputs, 3)


def test_repeat_exits_at_selected_epoch_with_screen_schedule(tmp_path, fixture_inputs):
    selection = {"epoch": 1, "learning_rate": 2e-5}
    config = config_for(fixture_inputs, "seed43", selection=selection)
    assert steps_per_epoch(25172) == 1574
    assert config["max_epochs"] == 1
    assert config["scheduler_total_steps"] == 6
    assert config["steps_per_epoch"] == 2


def test_tie_uses_lower_rate_then_earlier_epoch():
    candidates = {(key, epoch): .4 for key in SCREEN_KEYS for epoch in (1, 2, 3)}
    candidates[SCREEN_KEYS[1], 2] = .71
    candidates[SCREEN_KEYS[0], 1] = .709
    candidates[SCREEN_KEYS[0], 2] = .7098
    assert select_screen(candidates) == (SCREEN_KEYS[0], 1)
    candidates.pop((SCREEN_KEYS[1], 1))
    with pytest.raises(ValueError, match="six"):
        select_screen(candidates)


def test_budget_concurrency_worker_ceiling_and_priority():
    available = {"active": 1700, "provider": 10000, "wall": 1000, "segment": 1000}
    candidates = [("seed42-low", 610), ("seed42-high", 580)]
    assert choose_group(candidates, available, 500, True) == ("seed42-low", "seed42-high")
    assert allocation(available, 2, 500) * 2 < 1200
    assert choose_group(candidates, available, 550, True) == ("seed42-low",)
    assert choose_group(candidates, available, 550, False) == ("seed42-low",)
    assert choose_group(candidates, available, 1300, True) == ()


def test_measured_overlap_and_serial_fallback():
    left = {"step_seconds": [[i, i, i + .9] for i in range(1, 101)]}
    right = {"step_seconds": [[i, i + 50, i + 50.9] for i in range(1, 101)]}
    assert overlap_rate(left, right)["pair"]
    right = {"step_seconds": [[i, i + 102, i + 102.9] for i in range(1, 101)]}
    assert not overlap_rate(left, right)["pair"]


@pytest.mark.parametrize("member", ["roles/final_test.jsonl", "texts/policy.jsonl", "../outside"])
def test_receipt_archive_rejects_unpermitted_roles_and_paths(tmp_path, member):
    payload = tmp_path / "input.bin"
    with zipfile.ZipFile(payload, "w") as cache:
        cache.writestr(member, "synthetic")
        cache.writestr("receipt.json", json_bytes({"files": {member: sha256_bytes(b"synthetic")}}))
    with pytest.raises(ValueError, match="Unexpected"):
        extract_verified(payload, tmp_path, index="receipt.json", allowed=research_member)
    assert not result_member("state/seed43/epochs/e1/final_test/predictions.npz")


def test_archive_rejects_stale_index_and_payload(tmp_path):
    payload = tmp_path / "input.bin"
    member = "roles/train.jsonl"
    index = {"files": {member: sha256_bytes(b"expected")}}
    with zipfile.ZipFile(payload, "w") as cache:
        cache.writestr(member, "different")
        cache.writestr("receipt.json", json_bytes(index))
    with pytest.raises(ValueError, match="independently"):
        extract_verified(payload, tmp_path, index="receipt.json", allowed=research_member, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        extract_verified(payload, tmp_path, index="receipt.json", allowed=research_member,
                         expected_sha256=sha256_bytes(json_bytes(index)))


def test_independent_worker_checkpoint_transport_across_fresh_sessions(tmp_path, fixture_inputs, monkeypatch):
    from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    from disaster_triage.transformer_handoff import export_campaign, import_campaign
    from datetime import datetime, timezone
    from disaster_triage import transformer_run
    from disaster_triage.transformer_run import Campaign, _supervise

    monkeypatch.setattr(transformer_run, "utc_now", lambda: datetime(2026, 11, 20, tzinfo=timezone.utc))

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    model_root = tmp_path / "model-assets"
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "invented", "example"]
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(vocab)})
    tokenizer.save_pretrained(model_root)
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=len(vocab), n_layers=1, n_heads=2, dim=8, hidden_dim=16, num_labels=10,
    ))
    model.save_pretrained(model_root)

    def campaign_at(directory):
        campaign = Campaign.__new__(Campaign)
        campaign.id = "tiny"
        campaign.artifacts = directory / "artifacts"
        campaign.directory = campaign.artifacts / "transformer" / campaign.id
        campaign.directory.mkdir(parents=True)
        campaign.state = campaign.directory / "state"
        campaign.inputs = fixture_inputs
        campaign.receipt = {"source": {"code_revision": None, "code_sha256": "4" * 64}}
        campaign.receipt_sha256 = "5" * 64
        return campaign

    def supervise(campaign, stage, serial_key=None):
        keys = SCREEN_KEYS if serial_key is None else (serial_key,)
        jobs = [(key, *campaign.stage_config(key, {"versions": {}}, 8, "cpu", None), None) for key in keys]
        return _supervise(campaign, jobs, stage=stage, model_root=str(model_root), seconds=30,
                          reserve_seconds=2, overlap=False, quota_hours=None, trial="synthetic")

    local = campaign_at(tmp_path / "original")
    assert supervise(local, "pilot", SCREEN_KEYS[0]) == {SCREEN_KEYS[0]: "completed"}
    pilot_archive = local.directory / "pilot.bin"
    result = export_campaign(local, pilot_archive)
    restarted = campaign_at(tmp_path / "fresh")
    import_campaign(restarted, pilot_archive, expected_sha256=result["return_index_sha256"])
    restored = restarted.inspect()
    assert restored["state"][SCREEN_KEYS[0]]["steps"] == 2
    assert restored["state"][SCREEN_KEYS[0]]["status"] == "pilot"
    assert supervise(restarted, "screen") == {key: "completed" for key in SCREEN_KEYS}
    report = restarted.inspect(checkpoints=True)
    assert len(report["reports"]) == 6
    assert report["selection"]["scheduler_total_steps"] == 6
    assert report["cohort"]["n_seeds"] == 1
    finished = restarted.directory / "complete.bin"
    result = export_campaign(restarted, finished)
    for _ in range(2):
        imported = import_campaign(local, finished, expected_sha256=result["return_index_sha256"])
        assert imported == report


def test_archive_accepts_deterministically_dated_source_member(tmp_path):
    import os
    from disaster_triage.transformer_data import make_archive

    source = tmp_path / "source.py"
    source.write_bytes(b"# synthetic\n")
    os.utime(source, (0, 0))
    archive = tmp_path / "code.bin"
    make_archive(archive, {"code/source.py": source})
    with zipfile.ZipFile(archive) as package:
        assert package.read("code/source.py") == source.read_bytes()
