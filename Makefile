PYTHON ?= .venv/bin/python
.PHONY: check test doctor notebook wheel handoff
check:
	$(PYTHON) scripts/verify_repository.py
	$(PYTHON) -m pip check
test:
	$(PYTHON) -m pytest -q
doctor:
	$(PYTHON) -m disaster_triage doctor
notebook:
	$(PYTHON) scripts/execute_notebook.py
wheel:
	$(PYTHON) -m build --wheel --no-isolation
handoff:
	$(PYTHON) scripts/build_handoff.py
