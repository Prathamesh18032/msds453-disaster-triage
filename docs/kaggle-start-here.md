# Slice 2: start here in Kaggle

This handoff targets the runtime you reported on November 18, 2026:
Python 3.13.16; PyTorch 2.11.0+cu130; CUDA 13.0; two Tesla T4 GPUs;
NumPy 2.1.3; SciPy 1.16.3; scikit-learn 1.6.1; Transformers 5.20.2;
Hugging Face Hub 1.33.0; safetensors 0.8.0. datasketch 2.0.0 is bundled.

The package is version 0.1.1. Setup checks that inventory before any installation.
It installs the bundled project wheel and, only if absent, the bundled datasketch
wheel, using pip with --no-deps. Existing scientific packages/CUDA are preserved.
If the inventory has changed, return the error/output for a revised profile.
The notebook records the actual installed dependency closure for reproducibility.

## What the ZIP contains

- kaggle-verification.ipynb: the notebook to import, directly at the top level.
- disaster_triage-0.1.1-py3-none-any.whl: the installable project.
- datasketch-2.0.0-py3-none-any.whl: the small missing dependency; no internet needed to install it.
- code/: source, configurations, exact profile references and cleared notebooks.
- prior-accounting/: reviewed synthetic-job metadata, retaining the existing compute total.
- handoff-manifest.json: hashes used to check the files and exact source.
- START-HERE.md: these instructions.

There are no datasets, source messages, model weights, credentials or coursework
materials in this ZIP. The six prior accounting entries describe invented-input
verification and contain metadata/hashes only.

## Session one: verify

1. Unzip slice-2-kaggle-handoff.zip on your computer. Import the top-level
   kaggle-verification.ipynb into a private Kaggle notebook.
2. Create a private Kaggle dataset from the same ZIP, then add that dataset as a
   notebook input. Kaggle may expose its extracted files; the setup cell also
   handles the ZIP if it remains intact. Attach only one current handoff.
3. Select GPU T4 x2 and enable Internet for the public DistilBERT download.
   No Hugging Face token is required for that public model.
4. Leave MODE = "verify", RUN_PRETRAINED = True and RUN_PAIR = True.
   Enter the current GPU allowance in GPU_HOURS_BEFORE if shown.
5. Run All. Setup is followed by CPU/tiny-GPU checks, one two-invented-text
   DistilBERT forward/backward update, and independent tiny workers on both GPUs.
6. Enter GPU_HOURS_AFTER and the displayed SESSION_LIMIT if available, then rerun
   the last export cell. Missing values remain unknown rather than inferred.
7. Download slice-2-verification-results.zip from the final link or Kaggle output
   files. Keep this first-session ZIP unchanged. End the active session.

The setup does not upgrade the managed environment. The actual project dependency
closure must be compatible. Full-image pip-check diagnostics are retained separately
so unrelated preexisting conflicts do not trigger bulk upgrades.

If a cell fails, stop and return the error and available outputs. Do not bypass
checks or run research training to diagnose a runtime problem.

## Session two: retrieve and resume

1. Create a new session with the same code handoff, notebook and GPU T4 x2 allocation.
2. Upload the first-session results ZIP as another private dataset/input and attach it.
   The notebook handles extracted files or an intact results ZIP automatically.
3. Change only MODE to "resume". The same package setup may run in the new session.
4. Run All. The notebook skips fresh smoke/pretrained runs and instead verifies
   the retrieved CPU/GPU checkpoints, data order, RNG and next optimizer updates.
5. Download the second-session results ZIP. Return both ZIPs and end the session.

Checkpoint compatibility requires the same code and package versions. A new session
with a changed managed runtime must be diagnosed before recovery can be claimed.

## Meaning of completion

The earlier Mac validation is evidence for its own runtime. The Kaggle inventory is
user-reported evidence of versions and device availability. This revised handoff is
locally checked. The returned November 18 verify/resume archives have now passed
integrity, dependency, GPU, independent-worker and four-probe recovery review for this
handoff/runtime. Maximum session duration remains unknown; real-data feasibility is
still measured in the later training slice.

All inputs here are invented. HumAID eligibility/preprocessing, research training,
calibration, policy fitting and final evaluation are later work.


## If the results input has changed archive layout

An observed second-session path check failed on Kaggle's expanded input. The intact
first-session results remained complete; preserving those bytes and extracting them
explicitly resolved the workflow without waiving any checks. The exact transformed
member was not captured, so do not treat a specific extraction cause as established.

If this happens, rename a copy of the unchanged first-session results ZIP to
slice-2-verification-results.bin, upload it as a private input, and after the setup
cells run the following immediately before the resume cell:

~~~python
files = list(Path("/kaggle/input").rglob("slice-2-verification-results.bin"))
assert len(files) == 1, "Attach the renamed first-session results file"
RESULTS_DIR = WORK / ("original-results-" + file_digest(files[0])[:16])
if not RESULTS_DIR.exists():
    extract_zip(files[0], RESULTS_DIR)
verify_results(RESULTS_DIR)
~~~

Then rerun the resume cell with MODE resume. This changes the input location in the
interactive notebook; it does not alter the frozen code tree or checkpoint/config
identity. Keep the original first-session ZIP and return both results for review.
