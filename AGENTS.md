# Implementation repository instructions

Read `README.md`, `CONTRIBUTING.md`, and applicable technical sources before work.
At every session start and immediately after context compaction, read the local continuation note `artifacts/work-log.md` before further action when it exists, then verify its checkpoint against actual Git/filesystem evidence. Update it before/after long work and before yielding. Keep this note Git-ignored: never stage, commit, copy to public docs, or push it. No sibling repository is a runtime dependency.

This repository contains the implementation for the humanitarian message
classification study. Keep it self-contained: no filesystem links, imports, or
runtime dependencies on neighboring repositories. Do not copy coursework workflows,
rubrics, source PDFs, submission records, or graded report prose into this repository.

Keep data, source and derived research text, secrets, model weights, caches, and
generated artifacts outside Git, including when the remote is private. Use ignored
local storage documented in the README. Follow source rights and retention terms;
preserve retained raw snapshots unchanged. Use synthetic fixtures and examples.
Never print credentials or private dataset rows in logs or tool output.

Inspect the full staged diff and exact file list before committing. Clear notebook
outputs and execution counts; review markdown, attachments, and metadata as well.
Do not force-add ignored files. Use descriptive `feature/<task>` branches without
a `codex` prefix, per the owner's 2026-11-18 instruction. For execution slices, use
`feature/slice-<number>-<purpose>`. Preserve historical branch names in dated records.
Open pull requests into `main`. Respect user authorization for commits
and remote writes; do not force-push, overwrite unrelated work, or delete resources.

Setup does not select a Python version, package manager, dependency set, compute
service, model, split, or experiment policy. No training code, application API,
open-source license, CI, or enforced branch protection is configured. Run relevant
checks and meaningful tests when executable code is added; do not claim that
placeholder folders provide a test suite.

Do not access Canvas or other course systems without explicit user authorization.
Do not draft graded assignment or report prose unless the user requests it.
Do not infer submission or grading state from implementation files. Record an
unverified course requirement as `Unknown - user verification required`.
