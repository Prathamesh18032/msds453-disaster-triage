# Contributing

This is a private research implementation repository. Set up authentication and your
own commit identity using the [README](README.md). Teammate access is added separately.

## Branches and pull requests

Start from an up-to-date `main` and create a task branch. For agent work:

```sh
git switch main
git pull --ff-only
git switch -c codex/describe-your-task
```

Keep changes focused and submit a pull request into `main`. Describe the changed
behavior, validation, and material limitations. Review before merging. Do not
force-push, rewrite shared history, or delete branches without explicit authorization.
The initial scaffold is the sole setup commit directly on `main`; branch protections
and CI have not been configured.

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
hashes. Freeze model-selection decisions before final-test evaluation. Runtime,
dependencies, compute, models, and splits are not selected by this scaffold.

There is no test suite or CI yet. Add and run meaningful tests with the first executable
implementation; do not describe placeholder folders as passing tests.
