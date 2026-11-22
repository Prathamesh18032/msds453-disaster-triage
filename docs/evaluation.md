# Evaluation from prediction bundles

The CPU evaluator consumes [prediction-bundle v1](execution-contract.md#5-prediction-bundle-v1)
using NumPy and SciPy. It reads predictions and trusted metadata without access to
dataset text, training code, or model weights. Slice 4 is exercised on invented
fixtures. The default final-test gates remain in force. Slice 8 enables a dedicated
campaign after the selection and readiness records are verified.

## Inputs and results

- Construct a `BundleInput(directory, expected, run_id, model_family, seed)` from
  producer records. Build `BundleExpectation` independently of the prediction
  bundle. Use ordered `RoleRow` records and archive, eligible-manifest and role
  digests from the verified freeze. Use config and source snapshot digests from the
  actual model run. Store per-epoch selection candidates in separate bundle locations.
  The producer's run record must bind each bundle to its checkpoint and epoch.
  The v1 bundle has no checkpoint digest.
- `score_bundle(selection_input)["pooled"]["macro_f1"]` is the Slice 5 ranking
  input. Reports include bundle provenance, canonical labels, pooled and per-event
  scores, and fixed-label precision, recall, F1, true support, and predicted counts.
  The pooled score uses the pooled confusion counts. A zero-row role has a null
  macro-F1. Display recall with true support zero and precision with predicted
  count zero as unavailable, using numeric zeros in fixed-label aggregation.
- For logits, the report includes raw unweighted NLL and 15-bin top-label ECE.
  `fit_review_policy(calibration_input, policy_input)` verifies the two disjoint
  development roles for one run, fits one temperature, and returns thresholds at
  0%, 10%, 20%, and 30% nominal review. The fitted record includes both input
  role-manifest and payload digests. Keep it and its canonical JSON digest.
- Pass the fitted record and independently retained digest to score its policy
  bundle. Reports add calibrated NLL/ECE, the applied digest, and pooled/per-event
  results for each unchanged threshold. Acceptance includes threshold ties. Actual
  review can undershoot the nominal fraction. A group with zero accepted rows
  has null selective error. Request mistakes distinguish accepted false negatives
  from accepted false positives. These are simulated workload and model errors.
  Hard-label majority reports contain only classification outputs.
- `summarize_seeds(reports, cohort=SeedCohort(settings_sha256, epoch, configs))`
  retains the individual DistilBERT reports and computes pooled and per-event
  macro-F1 means and sample SDs when comparable scores exist for at least two
  seeds. Supply a digest of the selected shared training settings and the selected
  epoch, plus the approved config digest for each seed. Config digests are
  seed-specific. The caller takes this cohort from the selected run records.
  Missing seeds mark the summary exploratory. The three-seed family decision
  uses the later selection/scope procedure.

Store records with the canonical JSON writer in ignored artifacts. Keep research
bundles private. Missing/found support is concentrated in one event per partition.
Use class support and the individual event and seed results when interpreting
variation. See the [accepted data freeze](data-freeze.md).

## CPU walkthrough with invented rows

Run this Python block with the project installed. It writes only synthetic bundles
and aggregate JSON under `artifacts/evaluation/`. The same example exercises the
selection interface, fits a policy, reloads that record, and scores the policy
rows using the saved thresholds.

```python
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from disaster_triage.bundles import BundleExpectation, RoleRow, role_bytes, write_bundle
from disaster_triage.constants import LABELS
from disaster_triage.evaluation import BundleInput, fit_review_policy, score_bundle
from disaster_triage.provenance import save_json, sha256_bytes

root = Path("artifacts/evaluation") / f"synthetic-{uuid4().hex}"
run_id = "invented-lr"
family = "tfidf_lr"
requests = LABELS.index("requests_or_urgent_needs")
missing = LABELS.index("missing_or_found_people")
other = LABELS.index("not_humanitarian")


def invented_bundle(role, truth, logits, events):
    rows = tuple(
        RoleRow(f"{role}-{index}", f"synthetic_{event}", label)
        for index, (event, label) in enumerate(zip(events, truth, strict=True))
    )
    expected = BundleExpectation("synthetic", role, rows, {
        "archive_sha256": sha256_bytes(b"invented archive"),
        "eligible_manifest_sha256": sha256_bytes(b"invented eligibility"),
        "role_manifest_sha256": sha256_bytes(role_bytes(rows)),
        "config_sha256": sha256_bytes(b"invented classifier configuration"),
        "code_sha256": sha256_bytes(b"invented producer"),
    })
    directory = root / "bundles" / role
    write_bundle(directory, expected=expected, run_id=run_id, model_family=family, seed=None, arrays={
        "row_id": np.array([row.row_id for row in rows]),
        "event_id": np.array([row.event_id for row in rows]),
        "y_true": np.array(truth, dtype=np.int64),
        "logits": logits,
    })
    return BundleInput(directory, expected, run_id, family, None)


identity_logits = np.eye(len(LABELS), dtype=np.float32) * 2
selection = invented_bundle(
    "model_selection", [requests], identity_logits[[requests]], ["alpha"]
)
calibration = invented_bundle(
    "calibration", [requests, other], np.zeros((2, len(LABELS)), dtype=np.float32),
    ["alpha", "beta"],
)
policy = invented_bundle(
    "policy",
    [requests, requests, other, missing],
    identity_logits[[requests, other, requests, missing]],
    ["alpha", "alpha", "alpha", "beta"],
)
selection_report = score_bundle(selection)
save_json(root / "selection-report.json", selection_report)
record_path = root / "fitted-policy.json"
record_sha256 = save_json(record_path, fit_review_policy(calibration, policy))
report = score_bundle(
    policy,
    fitted_policy=json.loads(record_path.read_text(encoding="utf-8")),
    fitted_policy_sha256=record_sha256,
)
save_json(root / "policy-report.json", report)

assert selection_report["pooled"]["macro_f1"] == 0.1
assert report["pooled"]["macro_f1"] == 0.15
assert report["review"][0]["pooled"]["requests"]["accepted_false_negatives"] == 1
assert report["review"][0]["pooled"]["requests"]["accepted_false_positives"] == 1
assert [scenario["pooled"]["deferred"] for scenario in report["review"]] == [0, 0, 0, 0]
print(root)
```

## Locked held-out campaign

Slice 8 adds a local campaign over the accepted freeze. The ordinary dataset
loader, bundle reader and writer, scorer, and seed summary still reject the
final role by default. The campaign grants access only after checking the
independently retained selection lock and readiness digests, all development
bundles and saved policies, the ordered final role, and the locked predictor
identities.

Build the private selection lock from the accepted receipts with
`scripts/lock_slice8.py --help`. Supply the independently retained Slice 7
ledger snapshot and the owners' joint selection approval. Save the printed
lock digest in the operator record independently of the campaign directory.
Use the pinned Python 3.12 profile with cached assets at the approved
DistilBERT revision. The probe runs on 256 invented inputs, including
max-length batches, with local CPU first and MPS considered only if
necessary. The forecast covers 11,845 rows per trained predictor with a
20% margin. An installed wheel can run outside the checkout. For its
probe, point `--source-root` at the matching checkout.

```sh
disaster-triage evaluation probe --root "$ROOT" --campaign-id "$CAMPAIGN" \
  --source-root "$SOURCE" --selection-lock-sha256 "$LOCK_SHA"
disaster-triage evaluation prepare --root "$ROOT" --campaign-id "$CAMPAIGN" \
  --selection-lock-sha256 "$LOCK_SHA"
disaster-triage evaluation final --root "$ROOT" --campaign-id "$CAMPAIGN" \
  --selection-lock-sha256 "$LOCK_SHA" --readiness-sha256 "$READY_SHA"
disaster-triage evaluation verify --root "$ROOT" --campaign-id "$CAMPAIGN" \
  --selection-lock-sha256 "$LOCK_SHA" --readiness-sha256 "$READY_SHA"
```

Capture the preparation receipt digest in the operator record before running
`final`. Preparation runs four saved predictors on calibration and policy
roles, fits four temperatures, and freezes the four scenarios per predictor.
All roles share the same per-run inference configuration and producer
identity. The separate training configs and exact serialized parent hashes
remain in that inference configuration and in the selection lock.

`final` runs the majority rule, selected LR and all three locked DistilBERT
seeds against the same final rows. It uses the prepared thresholds unchanged.
The majority report contains classification metrics. Trained reports also
contain unweighted NLL and 15-bin ECE, along with review denominators and
counts. Final per-class displays use JSON null for undefined ratios. Macro-F1
still uses zeros in the fixed ten-label mean.

Outputs are under the ignored `artifacts/evaluation/<campaign>` directory.
Each predictor bundle and report is published atomically with a receipt
attested by its attempt record. Ordinary interrupts preserve completed
bundles. Repeat the same command and digests after the ledger is reconciled.
A hard stop may leave a running record. Preserve that record and reconcile
the termination time and elapsed duration with actual process evidence
before retrying. If a completed output requires a correction, keep its
original bytes and record the reviewed defect and correction before any
separate rerun. The shared 24-hour cap applies to evaluation. The November 23
training cutoff still applies to training. `verify` replays aggregate scores
from the final manifest and retained bundles without inference.
