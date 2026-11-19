# Local data preparation and human audit

**Current accepted execution - November 19:** [freeze identities and handoff](data-freeze.md).
The user-approved audit, real preparation, independent verification and installed-
wheel repeat pass. The commands below remain the reusable preparation procedure.

Slice 3 implements preparation under the [execution contract](execution-contract.md).
The owner reviews the private audit before any eligible dataset is frozen. No model,
metric, calibration, policy fitting, or research upload is performed by these commands.

## Storage and input

Transfer the immutable HumAID set-1 archive once into ignored
`data/raw/HumAID_data_events_set1_47K.tar.gz`. Verify the contract's SHA-256 and
2,641,818-byte size. Runtime commands use this root only; no neighboring workspace
is a runtime input. A matching copy can be reused; do not overwrite a mismatching one.

```sh
disaster-triage data inspect --root .
disaster-triage data prepare-audit --root .
```

Use the installed package in the pinned local environment. When running from another
working directory, supply the checkout as `--root`; preparation uses that checkout
for its source snapshot unless an explicit `--source-root` is supplied. All private
paths must remain under its real, nonsymlink `data/` tree. No network is required.

The inspector streams literal TSV records from the archive. It validates all 39
members and canonical basename events, including Greece/Maryland nesting. Six known
metadata members (Readme, Licensing, and four macOS metadata/resource-fork files)
are inventoried and hashed separately. Unknown members, links, unsafe paths,
incompatible headers, archive drift, or event-count drift stop preparation.

## Complete the audit locally

`prepare-audit` prints aggregate counts, an audit ID, and paths to the offline
`review.html`, machine-readable sample, template, and audit manifest under
`data/derived/audits/<audit-id>/`. These files are confidential research derivatives.
Open the HTML page locally in a browser. Do not publish, attach, or upload the page.

The sample selects at most three rows in every event/class stratum, then adds every
missing/found row and removes overlap. On the acquired snapshot this is 500 unique
rows (330 stratified + 179 rare - 9 overlap), before any further exclusions.

1. Choose English, mixed/uncertain, or confirmed non-English before revealing the label.
2. Judge usability and broad supplied-label fit. Annotation labels do not verify facts
   or operational urgency. Explain every flag in the private notes field.
3. Save progress JSON before closing the page. Restore it to continue. The page uses
   no network or browser persistent storage and makes no automatic judgments.
4. Complete the reviewer/date and explicit broader-change disposition, then download
   completed decisions. One reviewer is required; no second review is claimed.
5. Move the downloaded `audit-decisions.json` into the matching ignored audit folder.
   Keep progress/download copies private and include them in eventual retention cleanup.

Only individually confirmed non-English/unusable rows are excluded by this audit.
Mixed/uncertain language and suspected annotation errors retain their supplied labels
and flags. A broader exclusion/relabeling request blocks freezing until an explicit
protocol amendment. Sample findings are descriptive, not a corpus-wide accuracy estimate.

## Freeze and independently verify

Replace `AUDIT_ID` and `FREEZE_ID` below with the printed IDs.

```sh
disaster-triage data freeze --root . --decisions data/derived/audits/AUDIT_ID/audit-decisions.json
disaster-triage data verify --root . --freeze data/processed/freezes/FREEZE_ID
```

Missing/stale decisions, incomplete review, or unresolved broader changes stop before
publication. A lost training class additionally produces an aggregate feasibility
record under `data/derived/feasibility/`; no completed freeze is published.

The fixed cleanup produces lowercase classical/duplicate text and case-preserving
transformer text. Duplicate edges use IDs, normalized equality, and the prescribed
128-permutation/seed-42 MinHash LSH (.80) with exact character-five-gram Jaccard >=.90
confirmation for strings of at least 30 characters. Cross-event or label-conflicting
components are excluded entirely; consistent components keep the lowest numeric ID,
then member/line on equal IDs. LSH candidate retrieval is approximate and can miss
qualifying pairs. No similarity model is fitted.

Development groups use the contract's SHA-256 ordering and integer largest-remainder
3:1:1 allocation, with selection/calibration/policy tie priority. Small-stratum zeros
remain visible. Events and labels stay fixed.

Completed directories contain processed rows, text-bound eligible manifests, source
dispositions, components/edges/exclusions, audit evidence, zero-filled support tables,
ordered role manifests, separate role text files, training counts/weights, policy,
runtime/source provenance, and a completion descriptor. IDs and hashes are derived
from canonical exact bytes. Publication uses a staging directory and writes the
descriptor last; repeat inputs reproduce the identity and preserve earlier freezes.
Creation time/preparation duration are outside deterministic identity.

Verification checks every digest and reconstructs the full preparation from the
immutable archive and decisions, comparing text, groups, exclusions, role ordering,
support and training counts/weights. This structural check includes final-event data;
it does not produce predictions or metrics. It needs the local immutable archive and
the pinned MinHash environment. After a code/runtime change, investigate differences;
do not edit old manifests to make verification pass.

## Reader and handoff

```python
from disaster_triage.dataset import FrozenDataset

data = FrozenDataset(root=".", freeze="data/processed/freezes/FREEZE_ID")
training = data.load_role("train")
selection = data.load_role("model_selection")
```

Initialization verifies the entire freeze structurally. Subsequent role loads check
the descriptor and requested text shard before returning rows. Supported roles are
train, model_selection, calibration, and policy. Final-test loading raises before
opening a role payload and awaits Slice 8. Training is not added to prediction-bundle
`ROLES`; v1 prediction-role records remain only row_id/event_id/y_true.

Pass the same frozen manifests/hashes to later modeling/evaluation work. These
commands preserve the synthetic-only training guard and final prediction gate.
Research data and artifacts never enter the public code repository. Console output
contains only safe aggregate metadata and paths; detailed review notes stay private.

Preparation timings do not reset or rewrite the experiment ledger. Preserve the
reviewed full ledger for future experiment jobs. Hosted processing remains conditional
on the owner's separate source-compatibility, private settings, and retention checks.
The earlier Slice 2 Kaggle handoff is historical synthetic-verification evidence;
it does not contain this new pipeline or a research dataset.
