import json
from pathlib import Path
import pytest
from disaster_triage.config import RunConfig
from disaster_triage.smoke import run_offline, resume_probe

def test_complete_synthetic_recovery_and_export(tmp_path):
    source = Path(__file__).resolve().parents[1]
    result = run_offline(RunConfig(), tmp_path, source)
    assert result["recovery_equivalent"] and result["bundle_roundtrips"] == 2
    resumed = resume_probe(RunConfig(), tmp_path, source, result["verification_dir"])
    assert resumed["recovery_equivalent"]
    assert resumed["run_id"] == result["run_id"]

def test_notebook_is_cleared_and_contains_only_synthetic_flow():
    source = Path(__file__).resolve().parents[1]
    notebook = json.loads((source / "notebooks/runtime-verification.ipynb").read_text())
    assert "widgets" not in notebook["metadata"]
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
        assert not cell.get("attachments")
    content = json.dumps(notebook)
    assert "/Users/" not in content
    assert "TORCH_VERSION_PLACEHOLDER" not in content
