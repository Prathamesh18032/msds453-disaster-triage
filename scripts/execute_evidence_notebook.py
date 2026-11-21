"""Execute the evidence notebook using a fresh kernel from this interpreter."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient

from disaster_triage.provenance import atomic_bytes


def main():
    source = Path(__file__).resolve().parents[1] / "notebooks" / "study-evidence.ipynb"
    root = Path(os.environ.get("TRIAGE_EVIDENCE_ROOT", Path.cwd())).resolve()
    output_id = os.environ.get("TRIAGE_EVIDENCE_OUTPUT_ID", "study-evidence-synthetic")
    notebook = nbformat.read(source, as_version=4)
    with TemporaryDirectory(prefix="triage-evidence-kernel-") as temporary:
        kernel_dir = Path(temporary) / "kernels" / "triage-evidence"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Disaster Triage evidence", "language": "python",
        }), encoding="utf-8")
        os.environ["JUPYTER_PATH"] = temporary
        os.environ["TRIAGE_EVIDENCE_ROOT"] = str(root)
        NotebookClient(
            notebook, timeout=180, kernel_name="triage-evidence",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    directory = root / "artifacts" / "evidence-notebooks"
    if directory.is_symlink():
        raise ValueError("Notebook output directory must be local.")
    output = directory / f"{output_id}.executed.ipynb"
    atomic_bytes(output, nbformat.writes(notebook).encode("utf-8"))
    print(output)


if __name__ == "__main__":
    main()
