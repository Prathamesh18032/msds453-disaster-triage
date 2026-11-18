# Execution contract

Version: **2026-11-18**. Status: accepted by the data/model owner and evaluation collaborator; collaborator acceptance was reported by the owner on 2026-11-18. No research implementation, eligible-data freeze, model run or measured result exists yet. Slice 2 adds foundation primitives and synthetic verification; research execution remains pending. Implement and validate this contract before research fitting.

This document is self-contained. Runtime code must not read another repository. Public Git contains reviewed code, nonsensitive configurations, technical documentation, synthetic fixtures and cleared notebooks. Research data/text, private manifests/predictions, weights, secrets, caches and generated artifacts remain outside Git. No application API is defined.

## 1. Study and data contract

Task: classify humanitarian information in messages from complete disaster events unseen during training. Use HumAID v1.0 event-wise set 1 only. CrisisBench is future work; no extra dataset, model family, ensemble, focal-loss comparison or application is required.

- Publisher: [HumAID](https://crisisnlp.qcri.org/humaid_dataset.html); [set-1 archive](https://crisisnlp.qcri.org/data/humaid/HumAID_data_events_set1_47K.tar.gz).
- Archive name: `HumAID_data_events_set1_47K.tar.gz`; acquired 2026-10-05; 2,641,818 bytes; SHA-256 `5407c09c20286a11a594b5907af89b8319c5b4f365cff0063c4070bed1fd1b44`.
- Starting snapshot: **43,409 acquired rows, 13 events, ten observed labels**. Publisher-advertised set-1 count 47,868 differs; do not present it as the acquired count. Final eligible counts remain unfrozen.
- Input columns: `tweet_id`, `tweet_text`, `class_label`. Preserve numeric source IDs privately; no private text/IDs in public examples or logs. Derive canonical event from member basename, not nesting: Greece and Maryland have misleading parent folders.
- Later copy the unchanged archive once into ignored `data/raw/` storage, verify hash, and preserve it unchanged while retained. Runtime uses local data/artifact paths only. No transfer occurs merely by adopting this document.

The zero-based label order is fixed:

```json
[
  "caution_and_advice",
  "displaced_people_and_evacuations",
  "infrastructure_and_utility_damage",
  "injured_or_dead_people",
  "missing_or_found_people",
  "not_humanitarian",
  "other_relevant_information",
  "requests_or_urgent_needs",
  "rescue_volunteering_or_donation_effort",
  "sympathy_and_support"
]
```

Recombine all official per-event train/dev/test archive members. Those suffixes do not define this study's partitions. Whole events remain fixed:

| Partition | Canonical events | Acquired rows | Acquired requests | Acquired missing/found |
|---|---|---:|---:|---:|
| Training | `canada_wildfires_2016`, `ecuador_earthquake_2016`, `hurricane_irma_2017`, `hurricane_maria_2017`, `italy_earthquake_aug_2016`, `kaikoura_earthquake_2016`, `maryland_floods_2018`, `srilanka_floods_2017` | 25,180 | 1,036 | 140 |
| Development | `greece_wildfires_2018`, `hurricane_harvey_2017` | 10,627 | 358 | 20 |
| Final test | `cyclone_idai_2019`, `puebla_mexico_earthquake_2017`, `hurricane_matthew_2016` | 7,602 | 257 | 19 |

Counts precede eligibility/duplicate exclusions. Missing/found is concentrated in Maryland/Greece/Idai, one event per partition. Recompute support after exclusions. Weighting and more seeds do not create independent rare-class test cases. Do not claim unseen-disaster-type/chronological generalization, factual verification or operational urgency from this experiment.

### Eligibility, text views and duplicate groups

1. Validate schema, canonical event, numeric ID, nonblank text and fixed label. Conduct an independent bounded language/annotation audit before eligibility freeze; record findings and every exclusion without silently relabeling data. Audit findings are not yet available.
2. Decode HTML entities; apply NFKC; replace URLs and mentions with fixed `URL`/`USER` tokens; collapse whitespace/trim. The duplicate/classical view lowercases; the transformer view preserves case before the uncased tokenizer. Preserve negation, numbers and meaningful hashtag content. No blanket stemming/stopword deletion or event/source prefix in model input.
3. Edges: identical IDs, identical normalized text, or confirmed near-duplicate similarity. For normalized strings at least 30 characters long, form character five-gram sets; use 128-permutation MinHash, seed 42 and LSH candidate threshold 0.80; confirm exact set Jaccard at least 0.90. Shorter strings use exact matching only. Record approximate candidate-retrieval limitations; fit no similarity vocabulary/model on held-out text.
4. Form connected components. Exclude entire components crossing events or conflicting labels. Within a consistent single-event component retain the lowest numeric ID. Log groups/removals privately before development-role allocation.
5. Recompute event/class and partition/class support with explicit zeros. Keep all ten labels. If any training class disappears, stop before fitting and record a pre-test feasibility amendment. Do not move events or merge/drop labels based on model outcomes. Structural duplicate checks may inspect held-out text/identity, never final prediction performance.

### Development roles and fitting boundary

Within each surviving development event/label stratum, sort retained groups by SHA-256 of UTF-8 `42|canonical_event|class_label|lowest_numeric_group_id`. Allocate counts by largest remainders for **0.6 selection / 0.2 calibration / 0.2 policy**, breaking remainder ties selection, calibration, policy, then assign sorted groups in that order. Record support zeros rather than borrowing rows between roles. Preserve group identity in the private manifest.

Training alone fits vocabulary/IDF, class weights and model parameters. Selection rows choose configurations/epoch count/primary family; calibration rows fit temperature; policy rows choose review thresholds. Do not train/refit on development data. Final data cannot choose any configuration, feasibility scope, temperature or threshold. Freeze eligible/role manifests and runtime/model revisions before fitting.

## 2. Models, loss and selection

| Family | Fixed settings | Budget |
|---|---|---|
| Majority reference | Most frequent eligible training label; alphabetical-order tie break | One hard-label reference; no NLL/calibration/deferral |
| TF-IDF + multinomial logistic regression | Word 1/2-grams; `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, `max_features=100000`; L2, `lbfgs`, `max_iter=2000`; C 0.1/1/10 crossed with class weight none/balanced | Six converged fits |
| DistilBERT | `distilbert/distilbert-base-uncased`, matching tokenizer, ten-class head, length 128, dynamic padding, weighted CE, zero label smoothing; AdamW, LR 2e-5/5e-5, weight decay .01, betas .9/.999, epsilon 1e-8; linear decay, zero warmup, clipping norm 1.0; FP32 | Two seed-42 screenings, at most three epochs each; selected LR/epoch repeated at seeds 43/44; four runs/twelve epochs maximum |

Pin compatible dependencies and model/tokenizer revision hashes after synthetic compatibility tests, before fitting. Python 3.12 is the proposed target; actual compatible runtime/package manager/version pins remain a later runtime decision. Record seeds, truncation frequency, backend and all effective settings. Do not infer weighted loss from the model's default returned loss.

### Weighted objective

After training-data eligibility/deduplication, compute `w_c = N_train / (10 * n_train,c)` in the fixed label order. Stop for any zero training count. For each complete optimizer accumulation window B:

`L(B) = -sum_i(w_yi * log_softmax(logits_i)[y_i]) / sum_i(w_yi)`.

Normalize over the whole window, including the final short window. Averaging independently normalized microbatch losses is incorrect. The denominator depends only on labels/weights; implementations can determine it for the full window before backward calls without retaining all forward graphs. Verify equivalent full-batch/accumulated gradients on synthetic unequal-class batches, with dropout disabled for the equivalence check. Save counts/weights in configuration. No focal loss or oversampling. Calibration/evaluation NLL is unweighted. [PyTorch cross-entropy semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

### Selection and lock

Use pooled fixed-ten-label macro-F1 on model-selection rows; undefined denominators contribute numeric zero. Candidates within 0.001 absolute F1 of the family maximum tie: LR prefers smaller C, then no class weight; DistilBERT prefers lower LR, then earlier epoch. Reuse selected seed-42 checkpoint. Seeds 43/44 use that fixed epoch count, without their own best-epoch selection.

Compare selected transformer mean selection macro-F1 across seeds with LR; prefer LR within 0.001, otherwise higher score. Report both completed families, individual transformer seeds and sample SD when appropriate; deterministic LR has no artificial seed spread. Never choose a best test seed or test-selected ensemble. Before final-test export/evaluation, lock actual completed configurations, seeds, epoch count, primary family and scope; fit calibration/policy on their reserved roles. Corrective post-test bug reruns require documentation and cannot drive new tuning.

## 3. Compute and recovery

Preferred backend: private Kaggle, conditional on source-compatible handling/privacy/deletion verification. NVIDIA T4 x2 is the target allocation, not a future guarantee. Two workers share host resources; measure aggregate throughput rather than assuming double speed. [Kaggle specifications](https://www.kaggle.com/docs/notebooks).

| Stage | GPU 0 independent worker | GPU 1 independent worker | Gate |
|---|---|---|---|
| Screening | Seed 42, LR 2e-5 | Seed 42, LR 5e-5 | Data/runtime/rights checks pass |
| Repetition | Seed 43, selected LR/epochs | Seed 44, selected LR/epochs | Both screenings finish; development selection locked |

- **FP32 throughout**, with physical batch 8/accumulation 2 per worker (effective 16). Memory fallbacks: 4/4 then 2/8. No distributed cross-worker gradient averaging or AMP.
- Use independent run directories, writable checkpoints/logs and RNG/data-order/progress state; no shared mutable checkpoint file. Pair only while reliable and measured aggregate throughput improves over serial; otherwise run serially.
- Save full model/optimizer/scheduler/RNG/data-order/progress/config-hash recovery state every 100 optimizer steps and at epoch boundaries. Restore only compatible state. Preserve retrievable private bundles between sessions; checkpoint existence inside an ephemeral session alone is insufficient recovery evidence.
- Time first 100 optimizer steps on training data and continue the declared screening. Forecast remaining training, evaluation/checkpoint overhead with **20% margin**. Do not create extra tuning candidates.
- **24 cumulative active experiment-job hours**: sum active durations of independent jobs, including CPU fits/evaluation, pilots, failures, recomputation and resumed intervals. Two independent one-hour jobs cost two hours. Record wall time, GPU-device hours and actual provider quota separately; no reset dependence. Shared sequential work is counted once; distinct concurrent work is counted for each job.
- Do preparation, classical models and plots on CPU; stop unused accelerator sessions. Verify actual accelerator access, account quota/session limits before GPU work. The owner provides subsequent quota updates; documentation does not reserve provider capacity.
- No permitted GPU after 30 minutes: time local M4/16 GB MPS. Consider CPU only when measured feasibility fits the cap/cutoff. Record backend changes; cross-backend bitwise equality is not assumed.
- **Training cutoff: 2026-11-23 23:59 IST.** If transformer scope is incomplete, freeze a reduced-scope amendment before final-test inspection: LR primary; completed transformer runs exploratory; suspend the three-seed family comparison explicitly. Report actual completed/failed/interrupted/resumed attempts/configurations/seeds; incomplete screening is not a completed search, and one seed has no sample SD. Calibrate/threshold each reported completed classifier on its reserved development roles.

Run records must capture attempt/run identity, state, parent checkpoint/resume identity, versions/hashes, seed/settings, active intervals and cumulative hours, wall interval, device/quota accounting, progress and artifact hashes. Log no private rows or credentials.

## 4. Evaluation contract

For each completed selected LR/transformer run, use its raw ten-class logits (LR `decision_function` scores). Fit a positive scalar temperature on calibration rows only: bounded minimization of unweighted NLL over log(T) in [-5,5], tolerance 1e-6, stable log-softmax; compare with T=1 and prefer 1 on an NLL tie within 1e-12. Failed optimization/nonfinite values require diagnosis. Temperature preserves argmax, not guaranteed calibration improvement. No class-specific temperature/thresholds.

Confidence is max softmax(logits/T). On sorted policy confidences c and each nominal review r in .10/.20/.30, k=floor(r*n), tau=c[k]. Defer confidence strictly below tau; accept equality. Ties remain together; actual review can undershoot nominal. Include tau=0 unfiltered. Apply thresholds unchanged to final events; do not retune to force final workload. Empty calibration/policy roles stop their fitting with a diagnostic, without borrowing rows or using final data.

Required outputs:

- Pooled and per-event fixed-ten-label macro-F1; per-class precision, recall, F1 and support. Numeric zero for undefined denominators in aggregation; display recall with no true cases as `N/A, support 0`, and distinguish undefined precision using true/predicted counts.
- Requests recall/support; deferred true-request count/fraction; accepted false negatives (true request, other prediction) and accepted false positives (predicted request, other truth). Missing/found metrics/support per event/seed and concentration/uncertainty limitations.
- Unweighted mean NLL from stable log-softmax. Top-label ECE with 15 equal-width bins on [0,1], bin index `min(floor(15*confidence),14)`, nonempty-bin absolute accuracy-confidence gap weighted by bin fraction; empty bins contribute zero. Report before/after calibration.
- At each fixed tau: coverage=accepted/total; review=deferred/total; selective error=wrong accepted/accepted, or `N/A, accepted 0`. Report pooled/per-event actual workloads. Descriptive risk-coverage sweeps cannot select new final thresholds.
- Individual transformer seeds and eligible mean/sample SD; descriptive event variation, with no population significance claim from three selected final events. Deferral simulates workload/errors, not real reviewer accuracy, correction, speed or verified need. Majority has classification metrics only.

## 5. Prediction-bundle v1

Private layout: `artifacts/predictions/<run_id>/<role>/manifest.json` and `predictions.npz`. Each completed bundle covers exactly one role. Producer: model execution; consumer: evaluation/notebook. Implement serializers/readers and synthetic tests later; this section defines the interface now.

### Manifest

UTF-8 JSON object; required keys below. All digest fields are lowercase 64-character SHA-256 hex strings over the exact saved bytes, not local filenames. Keep the hashed source/config/manifests available privately while retained. No text, credentials or machine-specific absolute path is required in the manifest.

| Key | Type / rule |
|---|---|
| `schema_version` | Integer `1` |
| `run_id` | Nonempty stable opaque string identifying configuration/seed; resume attempts retain their parent run identity and have separate attempt logs |
| `model_family` | `majority`, `tfidf_lr`, or `distilbert` |
| `seed` | Integer for DistilBERT (42/43/44); integer or null for deterministic families; do not invent seed variance |
| `prediction_kind` | `logits` for LR/DistilBERT; `hard_labels` for majority |
| `role` | `model_selection`, `calibration`, `policy`, or `final_test` |
| `n_rows` | Nonnegative integer, equal to every payload array's first dimension |
| `labels` | Exact ten strings in section 1 order |
| `archive_sha256` | Retained raw source archive digest; the research dataset must match section 1 |
| `eligible_manifest_sha256` | Frozen private eligible-data manifest digest |
| `role_manifest_sha256` | Frozen private ordered row/event/true-label manifest for this role |
| `config_sha256` | Complete resolved run-configuration file digest, including weights/counts, dependencies, model/tokenizer revisions, seed, precision and effective settings |
| `code_sha256` | Exact saved implementation source-snapshot archive digest; document included files when creating that snapshot |
| `code_revision` | Full Git commit ID or null; record the source snapshot digest even with uncommitted local changes |
| `payload_sha256` | Exact `predictions.npz` bytes digest |

Snapshot/config creation and canonical role-manifest writing belong to the implementation foundation/data freeze. Exporters/readers compare the expected frozen digests supplied by the run/evaluation configuration, not merely a bundle's self-declared values. Test fixtures use explicitly supplied synthetic expected metadata; they cannot pass as research bundles.

### Payload

Read with `numpy.load(..., allow_pickle=False)`. No object arrays. Arrays share the exact frozen role-manifest row order; align by row identity rather than independent positional assumptions between bundles.

| Array | Dtype / shape | Meaning |
|---|---|---|
| `row_id` | NumPy Unicode string, `(N,)` | Unique opaque row identifier from the eligible/role manifests; no message text |
| `event_id` | NumPy Unicode string, `(N,)` | Canonical event from section 1; agree with frozen row metadata |
| `y_true` | int64, `(N,)` | Label indices 0 through 9, exactly matching role manifest |
| `logits` | float32, `(N,10)`; logits kind only | Finite raw uncalibrated scores in fixed column order, not probabilities or temperature-scaled values |
| `y_pred` | int64, `(N,)`; hard-label kind only | Majority predictions 0 through 9; no fabricated probability/logit output |

Require common arrays plus exactly the appropriate prediction array. A zero-row bundle can describe empty support but cannot fit temperature/thresholds or produce ordinary aggregate metrics; report unavailability explicitly. Partial/in-progress payloads are not completed bundles. Publish the manifest only after the payload and its digest are complete; consumers reject mismatches.

### Synthetic walkthrough

The following two invented rows illustrate shape/order only. Event names are synthetic fixture metadata, not additional research events. No model was run and no real message/identifier appears here.

| row_id | event_id | y_true | logits in label order |
|---|---|---:|---|
| `example-001` | `synthetic_event_alpha` | 7 | `[0,0,0,0,0,0,0,2,0,0]` |
| `example-002` | `synthetic_event_alpha` | 5 | `[0,0,0,0,0,2,0,0,0,0]` |

An illustrative logits bundle has schema_version 1, role `model_selection`, n_rows 2, the exact ten labels, row/event/true-label arrays in that order and logits shape (2,10). A majority variant uses `hard_labels` and y_pred `[5,5]` with no logits. Actual fixture serialization must compute real fixture digests and use synthetic expected source/role metadata; do not reuse placeholder hashes or claim this table is a frozen research bundle. Human review checks that index 7 means requests, index 5 means not-humanitarian, and neither role nor label order is inferred from logits.

### Rejection and final-test gate

Reject missing required fields, unsupported version, invalid family/kind/role/type, expected hash or label-order mismatch, corrupt/incomplete payload, duplicate/missing/extra or reordered role rows, mismatched event/true labels, invalid class indices, wrong array shape/dtype, object arrays and nonfinite logits. Real bundles must match the frozen event allocation; synthetic fixtures must be explicitly identified by the caller's synthetic metadata.

Do not export, load for scoring, calibrate against or inspect final-test predictions until the completed selection/scope lock is recorded. Structural eligibility checks are distinct from prediction scoring. Evaluation loads raw bundles without retraining or changing roles; save fitted temperatures/thresholds and derived metrics separately in ignored artifacts, preserving the raw payload. Both partners have accepted this contract; code-level rejection/edge-case tests remain pending.

## 6. Private hosting, retention and later validation

[Source terms](https://crisisnlp.qcri.org/terms-of-use.html) and retained archive Licensing.txt require confidential humanitarian-computing research use, citation and deletion upon request or research completion; IDs may be retained under those terms. Private Kaggle is selected, but hosted compatibility and actual settings are not verified by this document. Before uploading research text, verify compatible terms and separately check dataset/notebook/collaborator/output privacy; [Kaggle](https://www.kaggle.com/docs/datasets) documents distinct dataset/notebook sharing. If unresolved by Nov 19, research data stays local and Kaggle checks use synthetic inputs.

The data/model execution owner coordinates deletion on request/completion, including local/hosted raw/derived text, prediction bundles, checkpoints, caches and saved notebook outputs/versions containing research artifacts; the evaluation collaborator removes her copies and confirms. Check saved/backup/versioned copies separately. Preserve permitted minimal hashes/counts/provenance. Research is ongoing; no deletion is performed by adopting this contract.

Before fitting: validate synthetic forward/backward shapes/finite values, full-window weighted-loss gradients including final short windows, config-compatible checkpoint save/resume, data/event/role disjointness, training-only feature/weight fitting and pinned runtime/revisions. Before consuming model bundles: test manifest/payload rejection, fixed-label/zero-support metrics, empty acceptance, confidence ties, unchanged thresholds and synthetic calibration. Verify reproducible rendering from saved outputs without retraining. Foundation test implementation and actual local evidence are described in
[runtime-foundation.md](runtime-foundation.md); data/evaluation tests and hosted checks remain later gates.

The data/model owner and evaluation collaborator have accepted the weighted objective and prediction-bundle v1 contract. On 2026-11-18 the owner confirmed the collaborator's agreement and successful branch clone/collaboration readiness. This records human acceptance, not implementation or numerical-test evidence. Use normal feature branches and reviewed PRs when publication is authorized; no commit, push or remote change is implied by this document.
