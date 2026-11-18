"""Standard-library tests also run under Python 3.13 without an ML installation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BOOT = runpy.run_path(str(ROOT / "scripts/kaggle_bootstrap.py"))
PROFILE = runpy.run_path(str(ROOT / "src/disaster_triage/profiles.py"))

class KaggleSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        source = self.bundle / "code/src/disaster_triage"
        source.mkdir(parents=True)
        (source / "profiles.py").write_bytes((ROOT / "src/disaster_triage/profiles.py").read_bytes())
        (self.bundle / "project.whl").write_bytes(b"synthetic project wheel")
        (self.bundle / "datasketch.whl").write_bytes(b"synthetic dependency wheel")
        self.manifest = {
            "handoff_kind": "slice2-kaggle-v2", "package_version": "0.1.1",
            "project_wheel": "project.whl", "datasketch_wheel": "datasketch.whl",
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "files": {p.relative_to(self.bundle).as_posix(): BOOT["file_digest"](p)
                      for p in self.bundle.rglob("*") if p.is_file()},
        }
        self.save_manifest()

    def save_manifest(self):
        (self.bundle / "handoff-manifest.json").write_text(json.dumps(self.manifest))

    def plan(self, installed=None, python="3.13.16", fresh=True):
        import sys
        with patch.dict(sys.modules), patch.object(
            importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError
        ):
            if fresh:
                sys.modules.pop("disaster_triage", None)
            return BOOT["install_plan"](
                self.bundle, python_version=python,
                installed=PROFILE["KAGGLE_313"] if installed is None else installed)

    def test_inventory_accepts_reported_cuda_build(self):
        self.assertEqual(PROFILE["profile_errors"](
            "kaggle-py313", "3.13.16", PROFILE["KAGGLE_313"]), [])

    def test_inventory_rejects_python_or_scientific_package_drift(self):
        for key in PROFILE["KAGGLE_313"]:
            with self.subTest(package=key):
                changed = {**PROFILE["KAGGLE_313"], key: "different"}
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    self.plan(changed)
        with self.assertRaisesRegex(RuntimeError, "Python"):
            self.plan(python="3.12.16")

    def test_setup_preserves_native_packages_and_installs_only_missing_sketch(self):
        inventory = {**PROFILE["KAGGLE_313"], "datasketch": None}
        _, commands = self.plan(inventory)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--no-deps" in command for command in commands))
        self.assertTrue(commands[0][-1].endswith("datasketch.whl"))
        self.assertTrue(commands[1][-1].endswith("project.whl"))
        self.assertFalse(any("--upgrade" in command or "-r" in command for command in commands))

    def test_existing_datasketch_is_not_reinstalled(self):
        _, commands = self.plan()
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0][-1].endswith("project.whl"))

    def test_already_imported_project_requires_restart_before_replacement(self):
        import sys
        with patch.dict(sys.modules, {"disaster_triage": object()}):
            with self.assertRaisesRegex(RuntimeError, "Restart"):
                self.plan(fresh=False)

    def test_wrong_datasketch_version_stops_before_installation(self):
        inventory = {**PROFILE["KAGGLE_313"], "datasketch": "1.6.5"}
        with self.assertRaisesRegex(RuntimeError, "datasketch"):
            self.plan(inventory)

    def test_payload_corruption_and_unchecked_source_fail_preflight(self):
        (self.bundle / "project.whl").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.plan()
        (self.bundle / "project.whl").write_bytes(b"synthetic project wheel")
        (self.bundle / "code/injected.py").write_text("VALUE=1")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.plan()

    def test_flat_and_zipped_input_discovery(self):
        self.assertEqual(BOOT["locate_handoff"](self.root, self.root / "work"), self.bundle)
        zipped_input = self.root / "zipped"
        zipped_input.mkdir()
        archive = zipped_input / "slice-2-kaggle-handoff.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for path in self.bundle.rglob("*"):
                if path.is_file():
                    output.write(path, path.relative_to(self.bundle))
        discovered = BOOT["locate_handoff"](zipped_input, self.root / "stage")
        self.assertEqual(BOOT["verify_handoff"](discovered)["package_version"], "0.1.1")

    def test_archive_rejects_traversal_and_symlinks(self):
        for name, link in (("../outside", False), ("link", True)):
            with self.subTest(name=name):
                archive = self.root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    entry = zipfile.ZipInfo(name)
                    if link:
                        entry.create_system = 3
                        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                    output.writestr(entry, "synthetic")
                target = self.root / "extract"
                with self.assertRaises(ValueError):
                    BOOT["extract_zip"](archive, target)
                self.assertFalse(target.exists())
                self.assertFalse((self.root / "outside").exists())

    def test_return_bundle_keeps_artifacts_and_excludes_cache(self):
        work = self.root / "working"
        probe = work / "artifacts/verification/synthetic"
        probe.mkdir(parents=True)
        (probe / "probe-expectation.json").write_text('{"synthetic":true}')
        cache = work / ".cache"
        cache.mkdir()
        (cache / "model.safetensors").write_bytes(b"excluded public model cache")
        output = BOOT["export_results"](
            work, phase="verify", probes=[{"device": "cpu", "verification_dir": str(probe)}],
            evidence={"synthetic": True}, source_sha256=self.manifest["source_sha256"])
        with zipfile.ZipFile(output) as archive:
            self.assertIn("resume-index.json", archive.namelist())
            self.assertFalse(any(".cache" in name or name.endswith(".safetensors") for name in archive.namelist()))
        inputs = self.root / "result-input"
        inputs.mkdir()
        archive_copy = inputs / output.name
        archive_copy.write_bytes(output.read_bytes())
        retrieved = BOOT["locate_results"](inputs, self.root / "restored")
        index = BOOT["verify_results"](retrieved)
        self.assertEqual(index["probes"][0]["path"], "artifacts/verification/synthetic")
        (retrieved / "evidence.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "digest"):
            BOOT["verify_results"](retrieved)

    def test_notebook_embeds_current_bootstrap_and_is_cleared(self):
        notebook = json.loads((ROOT / "notebooks/kaggle-verification.ipynb").read_text())
        self.assertTrue("".join(notebook["cells"][2]["source"]).startswith(
            (ROOT / "scripts/kaggle_bootstrap.py").read_text()))
        for cell in notebook["cells"]:
            self.assertFalse(cell.get("outputs"))
            self.assertIsNone(cell.get("execution_count"))
            self.assertFalse(cell.get("attachments"))
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<kaggle cell>", "exec")
        self.assertNotIn("widgets", notebook["metadata"])

    def test_package_metadata_keeps_profiles_separate(self):
        import tomllib
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(metadata["requires-python"], ">=3.12,<3.14")
        self.assertIn("torch==2.11.0; python_version == '3.13'", metadata["dependencies"])
        self.assertIn("torch==2.15.0; python_version == '3.12'", metadata["dependencies"])

if __name__ == "__main__":
    unittest.main()
