# DistilBERT training and development selection

Slice 6 fits on the accepted training role and scores model-selection only. The
final data reader and prediction transport gates continue to require Slice 8.
The notebook defaults to setup. The Python runner is the same on the local Mac
and on a compatible, private Kaggle session.

## Prepare once on the local machine

Build the wheel from reviewed sources and create the code-only input.

```sh
make wheel
.venv/bin/python scripts/build_slice6_handoff.py \
  --wheel dist/disaster_triage-0.1.1-py3-none-any.whl \
  --vendor artifacts/handoff-v2/vendor/datasketch-2.0.0-py3-none-any.whl \
  --output artifacts/slice-6-kaggle/slice-6-code.bin
```

Prepare the private research input from the verified freeze. Keep the two
reported manifest digests independently of the archives.

```sh
.venv/bin/python -m disaster_triage transformer prepare --root . --source-root . \
  --freeze "$FREEZE" --campaign-id "$CAMPAIGN"
```

The research input is `artifacts/transformer/<campaign>/slice-6-research.bin`.
It contains the train/selection manifests and transformer text, the input receipt,
and prior attempt metadata. It is confidential. The code input has a separate
allowlist and carries no research payload. Keep the inputs, trained weights,
cache, result archives, and saved notebook outputs out of Git.

## Operate a private Kaggle session

1. Create a private notebook and upload the supplied `slice-6-kaggle.ipynb`.
   Attach `slice-6-code.bin` and `slice-6-research.bin` as separate private inputs.
   Keep the opaque `.bin` filenames so the original archives survive transfer.
   Verify dataset, notebook, collaborator and output privacy. Select T4 x2
   when offered. Enable Internet for the pinned public checkpoint download.
2. Fill the campaign ID, code manifest digest and research receipt digest. With
   `MODE = "setup"`, run the notebook and inspect its dependency closure,
   inventory and free space. If the native profile differs, record the actual
   diagnostics and resolve the mismatch before fitting. The bootstrap preserves
   Kaggle's scientific packages and installs only the wheel and missing pin.
3. Confirm privacy in the parameter cell. Set `MODE = "pilot"` and run. The
   lower-rate worker starts alone. On a matched two-GPU runtime the second
   worker starts when the first reaches step 50. Each pilot pauses at step 100.
   Real updates are reused. With one GPU, a pilot invocation starts one
   screening. The lower-rate pilot alone can forecast both serial screenings.
4. Download the private result and record the printed return-index SHA-256
   before ending the session. Return the file to the operator's local acceptance
   workflow. Keep the most recent validated archive and its digest for recovery.
   The archive is cumulative for this campaign, including independent attempts
   and full recovery state. Preserve any earlier exported checkpoint versions
   until the later snapshot is verified.
5. Choose `MODE = "screen"` for the next admitted screening segment. Re-run the
   stage if the returned state is still incomplete and the forecast permits.
   Use `MODE = "repeat"` after both screenings complete. On a fresh session
   attach the latest archive privately, set `RESTORE = True` and supply
   `RETURN_INDEX_SHA256`. Set `RESTORE = False` when continuing an already
   advanced session from its in-session state.
6. Record displayed quota/session details when available. The runner limits each
   invocation to at most two wall hours by default, stops before the cutoff, and
   counts independent active intervals. Preserve charged intervals from an
   interrupted session and reconcile the evidence before continuing if a session
   was lost. An exception type in the returned ledger is a failed attempt even
   when some checkpoints or epoch reports survived.

The campaign accepts physical batch 8, 4, or 2 with effective batch 16. On a
confirmed memory failure, prepare a new campaign at the next batch setting
using the reconciled ledger. This is an accounted restart of the same scientific
design. A cross-backend restart similarly requires a new campaign. Do not combine
screenings or seed repeats across incompatible profiles or batches. On a compatible
single GPU use the same logical device for recovery. When the pilot cannot
justify pairing, the scheduler admits candidates in protocol order for serial
execution.

## Verify results locally

Run collection from the same installed wheel and local campaign prepared above.

```sh
.venv/bin/python -m disaster_triage transformer collect --root . \
  --campaign-id "$CAMPAIGN" --results "$RETURN" --return-index-sha256 "$RETURN_INDEX"
.venv/bin/python -m disaster_triage transformer verify --root . --campaign-id "$CAMPAIGN"
```

The independent replay checks bundles, metrics, ranking, selected checkpoint
structure and cumulative attempt records. Find the safe result comparison in
`development-summary.json` inside the ignored campaign directory. Each candidate
retains its private resolved configuration, epoch receipts, checkpoints and v1
bundles there. Keep the results private. When training is permanently finished,
`transformer verify --seal` writes the proposed scope and checkpoint references
for the separate Slice 8 lock. Reduced scope uses LR primary and completed
transformers as exploratory. A complete scope leaves the family decision to Slice 8.
