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

## Held-out aggregates

Final mode reads the accepted Slice 8 manifest and the six JSON files it
references. Keep the manifest, cohort and reports in this layout in an
aggregate-only directory:

```text
manifest.json
transformer-cohort.json
final/majority/report.json
final/lr/report.json
final/seed42/report.json
final/seed43/report.json
final/seed44/report.json
```

Use the independently retained final manifest digest. Run the same
renderer from the CLI or the study notebook:

```sh
export TRIAGE_EVIDENCE_MODE=final
export TRIAGE_EVIDENCE_OUTPUT_ID=slice-9-held-out-evidence
export TRIAGE_EVIDENCE_FINAL_MANIFEST=/local/path/to/manifest.json
export TRIAGE_EVIDENCE_FINAL_MANIFEST_SHA256=ACCEPTED_FINAL_MANIFEST_SHA256

disaster-triage evidence render --mode "$TRIAGE_EVIDENCE_MODE" \
  --output-id "$TRIAGE_EVIDENCE_OUTPUT_ID" \
  --final-manifest "$TRIAGE_EVIDENCE_FINAL_MANIFEST" \
  --final-manifest-sha256 "$TRIAGE_EVIDENCE_FINAL_MANIFEST_SHA256"
python scripts/execute_evidence_notebook.py
```

Clear any development-only input variables before using final mode and
clear both final manifest variables before switching back. Output contains
figures 4–7, five CSVs, an evidence index and a manifest. The exploratory
event comparison averages the same labels with nonzero true support across
all three events. The primary comparison uses all ten labels. Review plots
use realized workload at the frozen thresholds. The final renderer needs
only aggregates, including when run outside the original checkout.

### Private error review

The worksheet is a separate local operator step. The Slice 8 gate verifies
that the imported package matches the exact accepted producer. Install the
retained Slice 8 wheel into a temporary directory with the pinned local
runtime. Use that directory first on `PYTHONPATH` when running the Slice 9
review script. Supply the independently retained digests and use the
implementation checkout as the root.

```sh
PYTHONPATH="$SLICE8_PRODUCER" python scripts/review_slice9.py \
  --root . --campaign-id "$CAMPAIGN" \
  --manifest-sha256 "$MANIFEST_SHA" --selection-lock-sha256 "$LOCK_SHA" \
  --readiness-sha256 "$READY_SHA"
```

Open the resulting offline worksheet locally. Complete one category for
each case and download the decisions JSON. Use the same command with
`--decisions /local/path/to/slice-9-review-decisions.json` to verify coverage
and produce group-level counts. The confusion counts are checked against
the accepted class marginals. The worksheet and decisions stay private
and outside Git and the notebook. The selected seed is the original
screening seed. The targeted sample helps identify error patterns.
