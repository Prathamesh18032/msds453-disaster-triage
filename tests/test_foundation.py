from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import pytest
from disaster_triage.config import RunConfig, load_config, resolved_config
from disaster_triage.provenance import json_bytes, save_json, sha256_file, source_snapshot
from disaster_triage.runtime import select_device

@pytest.mark.parametrize("changes", [
    {"seed": 1}, {"seed": True}, {"physical_batch": 4}, {"precision": "fp16"},
    {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
    {"max_epochs": 4}, {"max_length": 64}, {"label_smoothing": .1},
    {"artifact_dir": "../other"}, {"cache_dir": "/tmp/cache"},
    {"data_dir": ""}, {"active_job_hours_cap": 30}, {"device": "cuda"},
])
def test_invalid_configs(changes):
    with pytest.raises(ValueError):
        replace(RunConfig(), **changes)

def test_unknown_fields_and_template_gate(tmp_path):
    file = tmp_path / "bad.toml"
    file.write_text('undocumented = 1\n')
    with pytest.raises(ValueError, match="unknown"):
        load_config(file)
    with pytest.raises(ValueError, match="Slice 3"):
        resolved_config(RunConfig(mode="research_template"), versions={},
                        model_revision="x", training_counts=[1] * 10)

def test_paths_and_no_import_side_effects(tmp_path):
    config = RunConfig()
    assert config.paths(tmp_path)["artifact_dir"] == tmp_path / "artifacts"
    assert list(tmp_path.iterdir()) == []
    subprocess.run([sys.executable, "-c", "import disaster_triage"], cwd=tmp_path, check=True)
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "artifacts").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        config.paths(tmp_path)

def test_deterministic_snapshot_and_json(tmp_path):
    source = tmp_path / "checkout"
    (source / "src/pkg").mkdir(parents=True)
    (source / "src/pkg/a.py").write_text("VALUE = 1\n")
    (source / "artifacts").mkdir()
    (source / "artifacts/private.txt").write_text("unrelated generated value")
    a, b = tmp_path / "a.tar", tmp_path / "b.tar"
    first = source_snapshot(source, a)
    assert first == source_snapshot(source, b)
    assert a.read_bytes() == b.read_bytes()
    assert first["included_files"] == ["src/pkg/a.py"]
    (source / "artifacts/private.txt").write_text("changed generated value")
    assert source_snapshot(source, b)["code_sha256"] == first["code_sha256"]
    (source / "src/pkg/a.py").write_text("VALUE = 2\n")
    assert source_snapshot(source, b)["code_sha256"] != first["code_sha256"]
    assert json_bytes({"b": 1, "a": 2}) == json_bytes({"a": 2, "b": 1})
    digest = save_json(tmp_path / "config.json", {"a": 2})
    assert digest == sha256_file(tmp_path / "config.json")
    with pytest.raises(ValueError):
        json_bytes({"invalid": float("nan")})

def test_snapshot_symlinks_and_missing_backend(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").symlink_to(tmp_path / "absent.py")
    with pytest.raises(ValueError, match="symlink"):
        source_snapshot(tmp_path, tmp_path / "out.tar")
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda:999")
