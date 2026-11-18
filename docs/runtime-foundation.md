# Runtime foundation and synthetic handoff

**Returned Kaggle verification reviewed - 2026-11-18.** Both result archives pass
manifest integrity and source/runtime identity checks; the 50-distribution project
closure, actual invented-text DistilBERT GPU backward pass, independent CUDA workers,
eight fixture bundles and all four retrieved CPU/GPU continuations pass review.
Cumulative recorded active work is 0.0445084445 hours across 15 completed attempts.
The detailed review and original ZIPs are retained in ignored
artifacts/received-kaggle/2026-11-18/. Maximum session duration remains unverified;
account quota is a separate dated operational input. Real-data rights, feasibility and results
remain separate gates. The deployed source snapshot and handoff were not changed.

**Current Kaggle handoff:** [start-here instructions](kaggle-start-here.md) and
**notebooks/kaggle-verification.ipynb** support the reported Python 3.13.16,
PyTorch 2.11.0+cu130, CUDA 13.0 / two-T4 runtime. Existing native packages are
preserved. Version 0.1.1 supports Python 3.12 and 3.13 through separate conditional
dependency profiles. Returned execution/recovery evidence has now passed review; see the dated status above.

Slice 2 implements portable execution primitives. Research-data eligibility, real
model schedules, evaluation metrics, calibration and final-test scoring remain later
work. Public files contain code, configuration, technical documentation and invented
fixtures only. No research data is needed to install or test this package.

## Environment

Use Python 3.12. The local verification environment uses Python 3.12.16 on macOS
arm64. Install that interpreter separately if needed, then:

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/macos-py312.txt
.venv/bin/python -m pip install --no-deps -e '.[dev,notebook]'
make doctor
make check
make test
~~~

The exact installed dependency closure is in **requirements/common-py312.txt**.
macOS adds the explicit PyTorch pin through **requirements/macos-py312.txt**.
The earlier Python 3.12 Kaggle candidate is superseded by
**requirements/kaggle-py313.txt**, a reference to the user's reported native inventory.
Use the dedicated Kaggle notebook rather than installing the macOS closure there.
Do not replace Kaggle's PyTorch based on the macOS wheel instructions. Return
runtime evidence if the candidate cannot be satisfied. Normal development remains
local while a hosted runtime is unverified. The notebook checks the installed pins.

**pyproject.toml** defines an installable setuptools package and exact direct pins.
Development/notebook extras are separate. Dependency transitive use of PyYAML does
not introduce YAML configuration. Project configuration is strict TOML.

No import downloads models, starts jobs or creates storage. Explicit model checks
download public model/tokenizer assets into **.cache/huggingface/** under the supplied
execution root. Hugging Face is used for asset download, not remote model inference.

## CLI and configuration

~~~sh
.venv/bin/disaster-triage doctor --root .
.venv/bin/disaster-triage smoke --root . --source-root . --device cpu
.venv/bin/disaster-triage smoke --root . --source-root . --device mps
.venv/bin/disaster-triage smoke --stage pretrained --root . --source-root . --device cpu
.venv/bin/disaster-triage budget-summary --root .
~~~

The first smoke stage is offline and uses a tiny linear/dropout model and invented
labels/features. The pretrained stage is explicit and uses the accepted DistilBERT
architecture with a ten-class head, invented text, dynamic padding and length 128.
Its summary records exact model/tokenizer commit IDs and truncation counts. Neither
stage produces research results or consumes a screening candidate.

Use **configs/synthetic.toml** with **--config** when an explicit file is useful.
**configs/research-template.toml** cannot execute in Slice 2: the eligible-data freeze
and real training counts are not available. Supplied storage paths must be beneath
the root's ignored data, artifacts and cache directories. The root is chosen by the
caller; no runtime file depends on another checkout.

Only 8/2, 4/4 and 2/8 physical-batch/accumulation pairs are accepted. The scientific
settings remain those in [execution-contract.md](execution-contract.md).

## Python interfaces

- **config.RunConfig**, **load_config**, **save_resolved** validate configuration,
  preserve resolved settings/counts/weights, and hash exact deterministic JSON bytes.
- **runtime.doctor**, **select_device**, **seed_runtime** report actual capabilities
  and select CPU, MPS or one CUDA index explicitly. Unavailable backends do not fall
  back silently. FP32 is required; CUDA TF32 and mixed precision are disabled.
- **loss.class_weights**, **backward_window**, **optimizer_window**, **make_optimizer**
  implement the complete-window weighted objective. Supply microbatches as
  **Batch(inputs, labels)**. The denominator covers every label in the window; clipping,
  optimizer stepping and scheduler stepping happen once per window, including short
  final windows. AdamW decay applies to all parameters and is recorded explicitly.
- **checkpoint.save_checkpoint**, **restore_checkpoint**, **checkpoint_due** save
  tensor/state dictionaries and primitive metadata at cleared optimizer boundaries.
  Runtime callers supply independently resolved configuration/source/data/version
  expectations. Recovery restores progress, data order and RNG after object creation.
  Keep each epoch checkpoint and the most recent periodic checkpoint. The later
  research launcher calls **checkpoint_due** every optimizer step.
- **provenance.source_snapshot** creates a deterministic sorted tar with fixed
  headers over Python source/scripts, TOML configs, dependency text files, cleared
  notebooks, project metadata and Python-version file. Symlinks are rejected.
  Data, caches, docs, tests and generated outputs are excluded. A manifest names the
  exact included files; a source digest is required even with uncommitted code.
- **accounting.Attempt**, **budget_summary**, **guard_budget** write independent
  attempt records. Each run retains its identity across resume attempts. Active
  intervals are summed, including failures and synthetic model checks. Ordinary
  installation/unit/workflow checks are separate from this experiment ledger.

The source hash and configuration hash identify bytes, not a claim that a Git
commit contains uncommitted changes. Keep the associated source/config files
privately while retained. Record every future code/config/backend transition.

The later launcher must heartbeat attempts and invoke the shared budget guard at
optimizer boundaries. An unfinished attempt accrues conservatively from its last
heartbeat until explicitly reconciled using evidence. It is not silently discarded.
The foundation is not a distributed resource scheduler; later paired workers must
share one accounting root. Preserve old attempt records when moving between sessions. Synthetic probe directories
include an **accounting/** snapshot; the resume command imports those records
idempotently before checking the budget. The **merge_accounting** helper combines
retained ledgers without double counting; conflicting records require evidence-based
reconciliation. Use one shared ledger root for actual research workers.
The cutoff is November 23, 2026, 23:59 IST, and the cap is 24 summed active job hours.
Wall span, GPU-device time and manually supplied provider quota are distinct.

## Prediction-bundle v1 handoff

**bundles.RoleRow** and **write_role_manifest** use ordered UTF-8 JSONL records with
exactly row_id, event_id and y_true. Keys are sorted, JSON is compact, and each record
ends with LF. Empty roles hash the empty byte string. Slice 3 supplies the actual
ordered role records and frozen eligibility provenance.

Construct **BundleExpectation(mode, role, rows, digests)** independently from predictions.
Required expected digests are archive, eligible manifest, role manifest, configuration
and source snapshot. **write_bundle** writes the accepted manifest/NPZ shape;
**validate_bundle** performs reusable in-memory validation; **read_bundle** is a minimal
transport reader with no metric or selection policy. It verifies payload bytes and
uses **allow_pickle=False**. Priya can build the Slice 4 consumer on these helpers.

Synthetic mode must be explicit and uses synthetic event metadata. Research mode
requires the retained archive digest and the prescribed event allocation.
Final-test reading/export is disabled until Slice 8 connects a completed selection lock.
No application API or v2 schema is introduced.

Offline smoke outputs include logits and majority fixture bundles and
**bundle-expectation.json**. The invented logits are shape/order fixtures, not fitted
logistic-regression results. The majority bundle contains hard labels only.

## Checkpoint retrieval and new-session proof

The offline smoke prints its verification directory. Preserve the entire directory,
particularly **recovery-probe.pt**, **expected-after.pt**, **probe-expectation.json**,
**resolved-config.json**, **source.tar** and **source-manifest.json**, plus its **accounting/** directory.

1. Retrieve a copy into private persistent storage.
2. End the process/session.
3. Start a fresh process/session with identical code, Python/packages and device.
4. Supply the retrieved directory:

~~~sh
.venv/bin/disaster-triage smoke --stage resume --root . --source-root . \
  --device cpu --resume-dir PATH_TO_RETRIEVED_DIRECTORY
~~~

Expectations are rebuilt independently from the supplied configuration, runtime and
source. The resumed dropout update, parameters and subsequent RNG draws must match.
Changing backend is an explicit transition requiring separate compatibility evidence;
identical seeds do not guarantee identical numerics across platforms.

## Notebook and Kaggle checklist

**notebooks/runtime-verification.ipynb** is a thin package client. Execute locally with
**make notebook**. Executed notebooks and kernel files stay in ignored artifacts.
The public notebook remains cleared, without attachments or widget state.

Build a code-only transfer with **make wheel**, download the pinned datasketch
wheel with pip download --no-deps --only-binary=:all: --dest artifacts/handoff-v2/vendor
datasketch==2.0.0, then run **make handoff**. The current ignored ZIP includes a direct-import Kaggle notebook, project/datasketch
wheels, a ready source tree, instructions and hash manifest. See the start-here guide.
The earlier generic local notebook remains useful for the macOS profile.

Notebook environment controls:

| Variable | Behavior |
|---|---|
| TRIAGE_CODE_ROOT | Checkout or extracted source root |
| TRIAGE_ROOT | Writable ignored working root; defaults to Kaggle working storage there |
| TRIAGE_WHEEL / TRIAGE_INSTALL=1 | Explicit pinned bootstrap; restart kernel afterward |
| TRIAGE_DEVICE | cpu by default; explicitly choose mps or cuda:0 |
| TRIAGE_PRETRAINED=1 | Opt into public model/tokenizer download and integration |
| TRIAGE_PAIR=1 | If two CUDA devices exist, run separate synthetic worker processes |
| TRIAGE_RESUME_DIR | Retrieved prior-session probe directory |

Return the actual Python/package/CUDA/device report, account quota before/after,
session limits, CPU/device/pretrained summaries and fresh-session restore summary.
Check dataset, notebook, collaborator and saved-output privacy separately before any
future research upload. No research upload is part of this notebook.

Kaggle handling, accelerator availability, quota/session limits, paired throughput
and fresh-session persistence are **Unknown - user verification required** until the
owner returns evidence. A local MPS pass does not close them.

## Review and checks

Default tests are offline and require no pretrained weights or HumAID. They cover
weighted full/accumulated gradients and updates, short windows, dropout recovery,
step-100/epoch checkpoint cadence, failed atomic writes, provenance, strict bundle
rejection, concurrent/resumed accounting and notebook hygiene. Explicit smoke stages
exercise the actual installed model stack/devices.

Run **make check**, **make test**, **make notebook**, **make wheel** and the reviewed
handoff builder. Verify wheel installation in an isolated environment outside the
checkout. Review the complete file list and diff before any separately authorized
commit/publication. No CI, paid service, research run or final score is added here.
