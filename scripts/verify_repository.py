"""Read-only content/link/notebook/package boundary checks."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
failures = []
for path in (root / "src").rglob("*.py"):
    ast.parse(path.read_text())
for path in (root / "notebooks").glob("*.ipynb"):
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments"):
            failures.append(f"{path.name}: uncleared cell content")
    if "widgets" in notebook.get("metadata", {}):
        failures.append(f"{path.name}: widget state")
for path in [root / "README.md", root / "CONTRIBUTING.md", *(root / "docs").glob("*.md")]:
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
        if "://" not in target and not target.startswith("#"):
            if not (path.parent / target.split("#")[0]).exists():
                failures.append(f"{path.name}: missing local link")
# No dependency on any sibling repository or machine-specific source path.
for path in [*(root / "src").rglob("*.py"), *(root / "configs").glob("*.toml"),
             *(root / "notebooks").glob("*.ipynb")]:
    text = path.read_text()
    if "MSDSP-453-Final-Project" in text or "/Users/" in text:
        failures.append(f"{path.name}: machine/sibling path")
subprocess.run(["git", "-C", str(root), "diff", "--check"], check=True)
if failures:
    raise SystemExit("\n".join(failures))
print("Repository content, links, notebook hygiene, and whitespace checks passed.")
