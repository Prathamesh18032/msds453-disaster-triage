# Local CPU classical baselines

The classical runner consumes the accepted dataset freeze and runs the majority
reference and six multinomial logistic-regression candidates. It reads training
and model-selection text. Both final-test gates remain in force.

## Run and replay

Use the pinned local Python 3.12 environment and the
[accepted freeze](data-freeze.md). From the code checkout, set `FREEZE` to
`data/processed/freezes/<accepted-freeze-id>` and `CAMPAIGN` to an opaque name.

```sh
.venv/bin/python -m disaster_triage classical run --root . \
  --source-root . --freeze "$FREEZE" --campaign-id "$CAMPAIGN"
.venv/bin/python -m disaster_triage classical verify --root . \
  --freeze "$FREEZE" --campaign-id "$CAMPAIGN"
```

The runner accepts a verified research freeze within the supplied root. If the
package was installed from a wheel, point `--source-root` at the matching source
checkout and `--root` at the writable local workspace containing the frozen data.
The package files in the runtime and source snapshot must match. The approved
research execution profile is local CPU. No hosted account or model download
is needed.

Run the same command to resume an interrupted campaign after reconciling any
unresolved attempts in the shared budget ledger. Completed stage artifacts,
freeze identity, settings, runtime and source bytes must verify before reuse.
Choose a new campaign ID when source, runtime or the declared data changes.
A hard termination may leave an unresolved ledger entry. Preserve it and
reconcile the interruption with evidence before starting another experiment.

## Outputs

Each ignored `artifacts/classical/<campaign-id>/` directory contains:

- `campaign.json` and `source.tar` for the exact source and runtime identity.
- `features/` with one train-fitted TF-IDF vectorizer and shared sparse train
  and selection features. The classifier consumes float64 CSR features.
- Seven run directories with resolved configuration, convergence metadata,
  `selection/` v1 bundle and aggregate `report.json`. Each LR run also keeps its
  fitted classifier. The majority bundle contains hard labels. LR bundles contain
  raw scores cast to float32 in fixed label order.
- `selection.json` with all candidates and the LR winner. Rank by pooled ten-label
  macro-F1. Compare unrounded scores to the maximum with tolerance 0.001.
  Prefer smaller C, then unweighted within that set.
- A new `replay-*/` record for each successful verification. Replay loads the
  private model files after digest checks, transforms selection text, and
  checks that predictions, aggregate reports, and the LR winner match.

Use `selection.json` and the aggregate reports for evidence rendering. The
serialized vectorizer includes a research-derived vocabulary. Keep it, the
models, feature matrices, row-level v1 bundles, and run artifacts in ignored
local storage. Reload serialized objects only from a trusted local campaign in
the same pinned environment.

The shared `artifacts/runs/` ledger records serial feature generation, fitting
and scoring, and replay. Each started stage is subject to the 24-hour
cumulative cap and November 23 cutoff. The selection winner is the LR
representative. The later family decision, calibration, review policy, and
gated final evaluation use their reserved procedures.
