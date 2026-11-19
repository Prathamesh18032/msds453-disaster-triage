# Accepted local dataset freeze

Date: November 19, 2026. Scientific policy is the [execution contract](execution-contract.md).
The owner reviewed and approved the completed 500-row audit; local preparation,
independent verification and exact-byte repeatability pass. Subsequent
[classical model selection](classical-baselines.md) uses this freeze.

## Identity and counts

Freeze ID: `63dd4a283dee0cd16457270365cce8b153b7a41c4c91a12876d81acb56f21a56`.
Private local directory: `data/processed/freezes/<freeze-id>/`.

- Acquired: 43,409 rows, 13 events, ten labels.
- Eligible: **43,393**. Excluded: **16**, all in eight cross-event duplicate pairs.
- Structural, language/usability audit and conflicting-label exclusions: zero.
- Fixed training/development/final partitions: **25,172 / 10,626 / 7,595**.

| Role | Rows | Requests | Missing/found |
|---|---:|---:|---:|
| train | 25,172 | 1,036 | 140 |
| model_selection | 6,376 | 215 | 12 |
| calibration | 2,129 | 72 | 4 |
| policy | 2,121 | 71 | 4 |
| final_test, structural manifest only | 7,595 | 257 | 19 |

Allocation uses exact largest remainders within development event/class strata,
not independent rounding of pooled totals. All ten training classes survive.
Training counts in the fixed label order are
`[1559,1353,4090,2132,140,1308,4874,1036,5804,2876]`.
The descriptor stores N_train/(10*n_c) weights; missing/found weight is 17.98.
Record the actual FP32 vector again with future model-run configuration.

| Artifact | SHA-256 |
|---|---|
| Raw archive | 5407c09c20286a11a594b5907af89b8319c5b4f365cff0063c4070bed1fd1b44 |
| Approved original decision JSON | ce7089594fa7e358eb4deca82cdfca77f69957ef40f8174b8545d2b4990a60f5 |
| Canonical frozen decision JSON | 38183ea1229636ccb43652aed6451054ffde3426c7ecd58a6dcc37101c6fa636 |
| Preparation source snapshot | 59fa04b3ab38e8b3e467bcd1d72f9520393b3469c25f427b883af9cf5790fd65 |
| Freeze descriptor | 0d84025e8fef76d3254ba1f78b280e8bfa417748208660787d4fd59dc313b2e3 |
| Eligible manifest | 6657462fbc3e541cc596c9b8748dfa257a55fb1e173b5834920eae806180155d |
| Train manifest | b95c63dd286795d06bbfbeb813d8bf2d2c6b571555cba502235595d5b5049898 |
| Selection manifest | 77df04816ca84b1f1ad2881d8d78198dd7bf99d98b0722a92fee6c5a24e8661a |
| Calibration manifest | 5f051b2bbabe4bcb80756c0d445b4e6c2f8ad7f92ec56fc7bc4f923fcc3b5a24 |
| Policy manifest | a1e3ad705662c972245e44295d40d97bddb7acbf8a483522dd1c46d96fa841a3 |
| Final structural manifest | 46735b7dafbb0778d01129d5f55d5e1274c1b651c8515d65d4284d34a4cfdc41 |

The approved JSON's values are preserved; canonical serialization explains the
different stored decision-byte hash. The 31-member runtime source snapshot identifies
the actual uncommitted preparation bytes; the observed Git revision is the earlier
48acec0 merge, not a later publication commit.

## Verification and use

The first preparation, a separate CLI reconstruction and a repeated installed-wheel
preparation outside checkout agree. All 27 freeze files (25 payloads, descriptor,
publication record) remain byte-identical. Independent checks cover every source
disposition, component/edge, representative, exact-fraction role allocation,
disjointness, support, weights, v1 role ordering and actual nonfinal reader loads.
There are 130/30/50/60 event/partition/role/development-event-role class cells,
including 20 explicit event/class zeros. Final-test loading is still rejected.

```python
from disaster_triage.dataset import FrozenDataset

data = FrozenDataset(root=".", freeze="data/processed/freezes/63dd4a283dee0cd16457270365cce8b153b7a41c4c91a12876d81acb56f21a56")
training = data.load_role("train")
selection = data.load_role("model_selection")
```

Keep the immutable raw archive available to the verifier and use the pinned local
MinHash environment. See [preparation commands](data-preparation.md). Downstream
prediction bundles reuse the frozen archive/eligible/role hashes. Their config/code
hashes identify the actual model run, not placeholders or automatically reused data-
preparation source hashes. Classical selection bundles are generated separately.

## Limits and later gates

The audit has 437 plausible, 28 questionable and 35 uncertain category judgments plus
11 additional event-context notes. Supplied labels/events are unchanged. These are
qualitative flags from a stratified sample and rare-class census, not a measured
corpus error rate. Canonical source-member event assignments are not independent
proof of every message's narrated event; preserve the context concerns in interpretation.

Missing/found remains concentrated in one event per partition with support 140/20/19;
calibration and policy have four each. More seeds or weighting do not remove this
limitation. The publisher/acquired count discrepancy remains unresolved.

MinHash uses datasketch 2.0.0, seed42/128 permutations, LSH .80, b/r 9/13, exact
character-five-gram Jaccard >=.90. It retrieved 44 candidate pairs and confirmed eight
near edges, alongside six normalized-equality edges. Retrieval is approximate and
does not prove exhaustive absence of near duplicates.

The [Slice 4 evaluation engine](evaluation.md) is verified on invented bundles.
Slice 5's majority and six LR fits and Slice 6's transformer schedule/timed feasibility remain future work.
The synthetic-only training-config guard and Slice 8 final-prediction gate remain.
Local preparation does not establish compatible private hosted processing. No research
data upload, model result, provider-capacity claim or change to experiment accounting
is made by this freeze. All row-level artifacts and continuation notes stay ignored.
