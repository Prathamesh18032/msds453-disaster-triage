# Contributing

This is an intentionally public research implementation repository containing code,
synthetic examples and technical documentation. Set up authentication and your own
commit identity using the [README](README.md). Read the self-contained
[execution contract](docs/execution-contract.md) before implementation or evaluation.
Collaborators confirm their clone/workflow and contract acceptance separately from
GitHub write permission.

## Branches and pull requests

Start from an up-to-date `main` and create a descriptive feature branch:

```sh
git switch main
git pull --ff-only
git switch -c feature/describe-your-task
```

Keep changes focused and submit a pull request into `main`. Describe the changed
behavior, validation, and material limitations. Review before merging. Do not
force-push, rewrite shared history, or delete branches without explicit authorization.
The initial scaffold is the sole setup commit directly on `main`; branch protections
and CI have not been configured.

## Slice branches and review

Use one focused feature branch and pull request per execution slice. Branches
use `feature/slice-<number>-<purpose>`, for example
`feature/slice-3-data-partition-freeze`. Do not add a `codex` prefix.
Target `main` and include the slice number,
changed behavior, validation and remaining gates in the PR description.

Start independent slices from current `main`; parallel work uses separate branches.
When a slice depends on an unmerged contract/change, state that dependency explicitly.
After review, prefer squash merging for one coherent main-history entry per slice,
then start subsequent work from updated `main`. Commit/push/PR/merge actions still
require the owner's authorization; opening a PR does not authorize merging it.

## What to commit

Commit reviewed source code, synthetic tests, nonsensitive configurations, technical
documentation, and notebooks whose full contents have been inspected. Stage exact
paths rather than the entire working directory, then review:

```sh
git status --short
git diff --check
git diff --cached --check
git diff --cached
```

Keep datasets, source or derived research text, credentials, weights, caches, generated
reports, and experiment outputs outside Git. `.gitignore` is a convenience, not a
substitute for content review. Never use force-add to bypass these exclusions.
Synthetic test fixtures should be created inline in tests; any future exception to
a data-file ignore rule must be narrow and reviewed.

## Notebooks and research records

Before committing a notebook, clear every code cell's outputs and execution count,
remove widget state, and inspect markdown, attachments, and metadata for private
research text or credentials. Clearing outputs alone does not make its other contents
safe to commit. Keep full run outputs in ignored `artifacts/` storage.

Use synthetic examples in tests and demonstrations. When experiments are added,
document code, dataset and split versions, configuration, seeds, metrics, and artifact
hashes. Freeze model-selection decisions before final-test evaluation. The initial
scaffold selected no runtime or research policy. Current scientific and compute
choices are specified in the execution contract; compatible version pins,
eligible manifests and implementation correctness remain future verification gates.

Run make check and make test in the pinned environment. Integration checks and notebook
execution are documented in [runtime-foundation.md](docs/runtime-foundation.md).
No CI is configured. Distinguish synthetic verification from research results.
