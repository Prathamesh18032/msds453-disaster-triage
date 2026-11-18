"""Execute a fresh notebook kernel, keeping rendered outputs ignored."""
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
import json

root = Path(__file__).resolve().parents[1]
output = root / "artifacts" / "notebook-check"
kernel_dir = output / "kernels" / "python3"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Slice 2 Python 3.12", "language": "python",
}))
os.environ["JUPYTER_PATH"] = str(output)
os.environ["TRIAGE_CODE_ROOT"] = str(root)
os.environ["TRIAGE_ROOT"] = str(output / "working")
for flag in ("TRIAGE_INSTALL", "TRIAGE_PRETRAINED", "TRIAGE_PAIR"):
    os.environ[flag] = "0"
os.environ.pop("TRIAGE_RESUME_DIR", None)
os.environ["TRIAGE_DEVICE"] = "cpu"
notebook = nbformat.read(root / "notebooks/runtime-verification.ipynb", as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                        resources={"metadata": {"path": str(root)}})
client.execute()
nbformat.write(notebook, output / "runtime-verification.executed.ipynb")
print("Fresh Python 3.12 kernel executed the synthetic notebook; outputs are ignored.")
