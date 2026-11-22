# Reliable Humanitarian Message Classification Across Unseen Natural Disasters

Public code and technical documentation repository for the MSDS 453 disaster-message classification project.
The research direction is humanitarian information classification on unfamiliar disaster
events, with attention to rare categories and confidence-based human review.

**Current state: local CPU classical baselines and development selection are available.**
The installable package provides strict configuration, provenance, FP32 weighted-loss
accumulation, checkpoint recovery, attempt accounting and prediction-bundle v1.
See [Kaggle start-here instructions](docs/kaggle-start-here.md),
[runtime setup and handoff](docs/runtime-foundation.md) and the unchanged scientific
[execution contract](docs/execution-contract.md). Actual device/model verification
evidence is recorded separately. Returned Kaggle verify/resume archives have now
passed technical review, including both independent workers and four recovered probes.
See the dated status in the runtime guide; research handling/feasibility remain later gates.
The local preparation pipeline validates the immutable archive, generates a bounded
offline human audit, and can publish/independently verify data and role manifests
after completed decisions. See [data preparation and review instructions](docs/data-preparation.md).
The owner approved the 500-row review. The [accepted local data freeze](docs/data-freeze.md)
contains 43,393 eligible rows after 16 cross-event duplicate removals; independent
reconstruction and installed-wheel repeatability pass. The [CPU evaluation engine](docs/evaluation.md)
scores v1 prediction bundles and fits development review policies. The
[classical runner](docs/classical-baselines.md) fits the majority reference and the
declared TF-IDF/logistic regression grid on the local CPU. The evaluator also supports the locked Slice 8 campaign, including saved
review policies and final-test replay. CrisisBench remains future work.
The [DistilBERT runner](docs/transformer-execution.md) prepares the private
train/selection handoff, measures a pilot, and runs the bounded screening and
repeat schedule when its measured forecast fits. The cleared Kaggle notebook
defaults to setup and keeps recovery archives private. The
[study evidence renderer](docs/study-evidence.md) turns saved model-selection
aggregates into reproducible tables and figures on the CPU.

Both partners accepted the scientific contract on November 18. CI and application APIs are not configured.

## Layout

| Location | Purpose |
|---|---|
| `docs/` | Self-contained technical study and prediction-bundle contract |
| `src/disaster_triage/` | Execution primitives, audited data preparation, verified roles, classical baselines, and bundle evaluation |
| `tests/` | Offline tests with inline synthetic fixtures |
| `configs/` | Strict synthetic configuration and gated research template |
| `notebooks/` | Cleared execution, runtime-verification, and study evidence notebooks |
| `scripts/` | Repository, notebook-execution and code-only handoff checks |
| `data/raw/`, `data/processed/`, `data/derived/` | Local research data, ignored by Git |
| `artifacts/` | Local weights, predictions, metrics, and figures, ignored by Git |
| `.cache/` | Local caches, ignored by Git |

The ignored storage directories are created only when needed. This repository must
work independently of other local repositories. Course prompts, rubrics, milestone
tracking, submission records, and report prose belong in the separate coursework
workspace and are not included here.

## Authenticate and clone

This repository is intentionally public, so cloning does not require collaborator
access. Pushing feature branches requires the owner to grant write access. Each
collaborator should confirm a successful clone and their own author identity; GitHub
permission alone does not establish a working local clone.
Install Git and GitHub CLI, then authenticate your own GitHub account if needed:

```sh
gh auth login --hostname github.com --web
gh api user --jq .login
```

From your chosen parent directory, clone using HTTPS without embedding a token:

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' clone https://github.com/Prathamesh18032/msds453-disaster-triage.git
cd msds453-disaster-triage
git config --local credential.helper ''
git config --local --add credential.helper '!gh auth git-credential'
git config --local pull.ff only
git config --local fetch.prune true
git config --local user.name 'YOUR NAME'
git config --local user.email 'YOUR VERIFIED GITHUB OR NOREPLY EMAIL'
```

Use your own author identity. These Git settings apply only to this clone. Credentials
stay in your authenticated GitHub CLI session, never in a remote URL or tracked file.
You may use an existing trusted HTTPS credential manager instead of the CLI helper.
Use the pinned Python 3.12 setup in [runtime instructions](docs/runtime-foundation.md).

## Data and artifacts

Use ignored local storage under the applicable source terms. Follow the
[execution contract](docs/execution-contract.md) for the later one-time immutable
archive transfer/hash verification, fixed scientific policy and prediction-bundle v1.
Runtime scripts must use this repository's own data paths.
Record source URLs, versions, acquisition dates, hashes, rights, transformations, and
split decisions before experiments. Preserve retained raw snapshots unchanged.
Git exclusions apply regardless of repository visibility: keep source text,
derivatives, credentials, weights, caches, and generated outputs outside Git.
Track only reviewed code, nonsensitive configuration, synthetic tests, and technical
documentation. Do not point scripts at a neighboring coursework repository.

## Collaboration and checks

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before changes.
Work on descriptive `feature/<task>` branches without a `codex` prefix and submit
pull requests into `main`. Branch protections and CI are deferred, so this is currently a
documented collaboration policy. Issues are enabled and the wiki is disabled.

Review changes with `git status --short`, `git diff --check`, and `git diff --cached`
before committing. Run make check and make test using the project environment; see the
[runtime verification guide](docs/runtime-foundation.md) for notebook and integration checks.
No open-source license has been assigned.
