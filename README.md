# Reliable Humanitarian Message Classification Across Unseen Natural Disasters

Public code and technical documentation repository for the MSDS 453 disaster-message classification project.
The research direction is humanitarian information classification on unfamiliar disaster
events, with attention to rare categories and confidence-based human review.

**Current state: scaffold plus an adopted execution specification.** No datasets have
been acquired into this repository, and no models have been implemented, trained, or
evaluated here. The [execution contract](docs/execution-contract.md) specifies HumAID
set 1 and ten labels, named whole-event partitions, majority/TF-IDF-logistic-regression/
DistilBERT comparison, weighted transformer loss, FP32 and paired independent T4 jobs.
CrisisBench remains future work. Eligible counts/manifests, compatible runtime and
model revisions, actual throughput and numerical correctness remain unfrozen or
unverified. Both partners have accepted the loss/bundle contract; collaborator
acceptance and branch clone were reported by the owner on November 18.
Python/package installation, training/evaluation code, CI and application APIs
are not configured.

## Layout

| Location | Purpose |
|---|---|
| `docs/` | Self-contained technical study and prediction-bundle contract |
| `src/disaster_triage/` | Reserved for implementation; currently a placeholder, not an installable package |
| `tests/` | Future tests using synthetic fixtures |
| `configs/` | Future nonsensitive, versioned configurations |
| `notebooks/` | Future notebooks with cleared outputs and reviewed contents |
| `scripts/` | Future acquisition, evaluation, and maintenance helpers |
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
There is no runtime installation command until the development environment is chosen.

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
Work on feature branches and submit pull requests into `main`; agent branches use
`codex/<task>`. Branch protections and CI are deferred, so this is currently a
documented collaboration policy. Issues are enabled and the wiki is disabled.

Review changes with `git status --short`, `git diff --check`, and `git diff --cached`
before committing. No automated test suite exists yet; the test directory is a
placeholder. Add appropriate checks when executable code is introduced.
No open-source license has been assigned.
