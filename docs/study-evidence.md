# Study evidence

The CPU renderer uses aggregate model-selection reports in both the command and
the [study notebook](../notebooks/study-evidence.ipynb). Run it with the optional
Matplotlib 3.11.2 evidence dependency. Install the notebook extra for the fresh
kernel walkthrough.

```sh
python -m pip install -e '.[evidence,notebook]'
disaster-triage evidence render --output-id study-evidence-synthetic
python scripts/execute_evidence_notebook.py
```

The default command and notebook render the same invented inputs and verify
an identical result under `artifacts/evidence/study-evidence-synthetic/`. The
synthetic fixtures are scored through the existing evaluator in temporary
storage. `make notebook` continues to run the runtime-verification notebook.

## Development aggregates

Use the accepted SHA-256 values from the Slice 5 and Slice 6 handoffs. Keep
the seven candidate report files in their original subdirectories alongside
`selection.json`; the transformer epoch and seed aggregates are embedded in
`development-summary.json`. These nine JSON files are the complete input.

```sh
export TRIAGE_EVIDENCE_MODE=development
export TRIAGE_EVIDENCE_OUTPUT_ID=development-1
export TRIAGE_EVIDENCE_SELECTION=artifacts/classical/CAMPAIGN/selection.json
export TRIAGE_EVIDENCE_SELECTION_SHA256=ACCEPTED_SELECTION_SHA256
export TRIAGE_EVIDENCE_SUMMARY=artifacts/transformer/CAMPAIGN/development-summary.json
export TRIAGE_EVIDENCE_SUMMARY_SHA256=ACCEPTED_SUMMARY_SHA256

disaster-triage evidence render --mode "$TRIAGE_EVIDENCE_MODE" \
  --output-id "$TRIAGE_EVIDENCE_OUTPUT_ID" \
  --selection "$TRIAGE_EVIDENCE_SELECTION" --selection-sha256 "$TRIAGE_EVIDENCE_SELECTION_SHA256" \
  --summary "$TRIAGE_EVIDENCE_SUMMARY" --summary-sha256 "$TRIAGE_EVIDENCE_SUMMARY_SHA256"

python scripts/execute_evidence_notebook.py
```

Run from the checkout root, or use `--root` for the command and
`TRIAGE_EVIDENCE_ROOT` for the fresh-kernel script. Paths passed to the command
resolve from its working directory. For the notebook, use absolute input paths
or paths relative to the selected root. When switching back to the synthetic
workflow in the same shell, unset the six environment variables above.

The executed notebook is written to
`artifacts/evidence-notebooks/<output-id>.executed.ipynb`. The three CSVs,
three figure pairs (PNG and PDF), and manifest live in
`artifacts/evidence/<output-id>/`. The manifest hashes the inputs and all nine
rendered deliverables and binds the accepted LR and DistilBERT run references.
Reruns compare the complete published file set with fresh output before
returning. If evidence or the rendering environment changes the output bytes,
choose a new output ID.

Macro-F1 always averages the ten canonical labels. The pooled value uses the
saved pooled report. The DistilBERT cohort mean and sample SD summarize three
seeds on the same selection rows. The class table contains all candidates in
pooled and event scope. It displays `N/A` when recall has zero true support or
precision has zero predicted count. Each figure carries its evidence scope and
a numbered caption. Rendering uses no model fits, checkpoint loads or
held-out predictions.
