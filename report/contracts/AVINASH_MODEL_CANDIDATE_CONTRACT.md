# Model & Candidate-Set Interface Contract

**Owner:** Avinash Julakatti (`PES1UG24AM347`)  
**Workstream:** Model + Re-Ranker (Track B)  
**Date:** Wednesday, 28-10-2026 (Wed 28 EOD Interface Freeze)  
**Status:** **FROZEN & SIGNED OFF**  
**Associated Deliverables:** `02_train.py`, `04_rerank.py`, `03_evaluate.py` (baselines), `05_sensitivity.py`

---

## 1. Scope & Objective

As defined in the project plan, Avinash owns the model training, full-ranking candidate generation, baseline models, post-hoc exposure-rebalancing re-ranker, and sensitivity analysis.

Today's scheduled deliverable (Wed 28) is to **Agree and Freeze the Embedding Artefact Format and Candidate-Set Contract** with Vishanth (Track A: Data & Evaluation) and Rithesh (Track C: Loader, Integration Tests & Documentation).

This document establishes the binding technical specification, schemas, mathematical definitions, and validation invariants for all artefacts produced and consumed by the Model & Re-ranking pipeline.

---

## 2. Embedding Artefact Contract

### 2.1 Storage Format and Path
- **File path:** `results/model/als_factors.npz`
- **Format:** Compressed NumPy archive (`np.savez_compressed`)
- **Companion model pickle (optional):** `results/model/als_model.pkl`

### 2.2 Archive Structure & Data Types
The `.npz` archive MUST contain exactly four arrays:

| Key | Type | Shape | Description | Invariant Rules |
| :--- | :--- | :--- | :--- | :--- |
| `user_factors` | `float32` | `(U, D)` | Latent factors for all train users | $U = \text{number of train users}$. Must contain no NaNs/Infs. |
| `item_factors` | `float32` | `(I, D)` | Latent factors for all catalogue items | $I = \text{number of catalogue items}$. Must contain no NaNs/Infs. |
| `user_ids` | `str` / `object` | `(U,)` | String IDs corresponding to rows of `user_factors` | Matches unique `userId` values from `ratings.parquet` where `split == 'train'`. |
| `item_ids` | `str` / `object` | `(I,)` | String IDs corresponding to rows of `item_factors` | Matches unique `itemId` (`parent_asin`) where `in_catalogue == True`. |

### 2.3 Hyperparameter Commitments
- Latent dimension $D$: `64` (default; alternatives `32` or `128` recorded in manifest)
- Implicit ALS confidence scaling $\alpha$: `40.0`
- Regularization $\lambda_{\text{als}}$: `0.05`
- Training iterations: `15`
- Random seed: `42` (fixed across all runs)
- **Training regime:** Full catalogue trained on all positive interactions (`rating >= 4.0`), **no sampled negatives** used in training or evaluation.

---

## 3. Candidate-Set Generation Contract

### 3.1 Storage Format and Path
- **File path:** `results/baseline_scores.parquet`
- **Compression:** Snappy or ZSTD
- **Target populations:** Two downstream evaluation populations:
  1. `exposure_pop`: Users with $\ge 1$ train positive
  2. `relevance_pop`: Users with $\ge 2$ train positives
  Both sampled up to $N_{\text{eval}} = 300,000$ users using fixed seed (`seed = 42`), sorted by `userId` prior to sampling.

### 3.2 Candidate Scoring and Filtering Rule
For each evaluated user $u$:
1. **Full-Ranking Dot Product:** Compute raw score over all catalogue items:
   $$\text{score}(u, i) = \mathbf{p}_u^\top \mathbf{q}_i, \quad \forall i \in \mathcal{I}_{\text{catalogue}}$$
2. **Seen-Item Masking (Hard Exclusion):** 
   $$\mathcal{C}_u \subseteq \mathcal{I}_{\text{catalogue}} \setminus \mathcal{I}_{\text{seen}, u}$$
   where $\mathcal{I}_{\text{seen}, u}$ comprises **every item user $u$ has rated in `ratings.parquet`** (where `seen == True`), including low-rated items (`rating < 4`). A rated item is an already-observed interaction and is strictly ineligible for recommendation.
3. **Top-$K$ Selection ($K = 200$):**
   Select the top $N = 200$ candidates with the highest scores from $\mathcal{C}_u$.
4. **Deterministic Tie-Breaking:**
   If two items have identical scores within floating-point epsilon ($10^{-6}$), sort by `popularity_rank` ascending, then `itemId` lexicographically ascending.

### 3.3 Parquet Schema (`baseline_scores.parquet`)

| Column Name | Arrow / Parquet Type | Nullable | Invariant Checks |
| :--- | :--- | :--- | :--- |
| `userId` | `string` | No | Must belong to evaluated population. |
| `itemId` | `string` | No | Must be present in `items.parquet` with `in_catalogue == True`. |
| `score` | `float32` | No | Non-increasing with respect to `rank` per user. No NaNs/Infs. |
| `rank` | `int16` | No | Strictly consecutive integers from `1` to `200` for each user. |

---

## 4. Re-Ranking Contract & Invariants

### 4.1 Storage Formats and Paths
- **Scores per $\lambda$:** `results/rerank_scores_lambda_<L>.parquet` where $\lambda \in \{0.0, 0.1, 0.25, 0.5, 1.0, 2.0\}$.
- **Sweep Summary:** `results/rerank_sweep.json`

### 4.2 Hard Invariant: Candidate Boundary Isolation
$$\mathcal{S}_{u, \lambda} = \mathcal{C}_{u, \text{baseline}}, \quad \forall \lambda \ge 0$$
- The re-ranker **permutes only the top-200 ALS candidates** provided in `baseline_scores.parquet`.
- It **MAY NOT** introduce any item outside the user's top-200 candidate list.
- It **MAY NOT** drop any item from the candidate list (the list length remains exactly 200).
- This guarantees that any change in exposure or relevance is strictly attributable to the re-ranking intervention.

### 4.3 Z-Scored Penalty Transformation
For each candidate item $i$ in user $u$'s candidate set:
$$\text{score}_{\text{final}}(u, i) = z(\text{score}(u, i)) - \lambda \cdot z(\log(1 + \text{pop}_i))$$
where:
- $\text{pop}_i = \text{n\_train\_ratings}(i)$ from `items.parquet`.
- $z(x)$ is the standard z-score computed **within each user's candidate list**:
  $$z(x_i) = \frac{x_i - \mu_u}{\sigma_u}$$
  $$\mu_u = \frac{1}{N}\sum_{j=1}^N x_j, \quad \sigma_u = \sqrt{\frac{1}{N}\sum_{j=1}^N (x_j - \mu_u)^2}$$
- **Zero-variance safeguard:** If $\sigma_u < 10^{-8}$, $z(x_i) \triangleq 0$.
- **No-Op Identity ($\lambda = 0.0$):** When $\lambda = 0.0$, the penalty term vanishes, preserving the exact original ALS ranking order.

---

## 5. Baselines Contract

Avinash provides candidate sets for two baseline models to be evaluated in `03_evaluate.py`:

1. **MostPop Baseline:**
   - For each user $u$, candidate set consists of the top 200 catalogue items sorted descending by $n_{\text{train\_ratings}}$, excluding user $u$'s seen items.
   - Deterministic tie-breaking by `itemId`.
2. **Random Baseline:**
   - For each user $u$, uniform random sample of 200 items without replacement from $\mathcal{I}_{\text{catalogue}} \setminus \mathcal{I}_{\text{seen}, u}$.
   - Seeded deterministically: `seed = hash((42, userId))`.

**Schema guarantee:** Both baselines conform to the identical `(userId, itemId, score, rank)` Parquet schema.

---

## 6. Evaluation & Sensitivity Artefacts

1. **Per-User Metrics (`results/per_user_metrics.parquet`):**
   - Columns: `userId: string`, `population: string`, `lambda: float32`, `ndcg: float32`, `hit: float32`.
   - Enables paired bootstrap confidence intervals.
2. **Re-rank Sweep Metrics (`results/rerank_sweep.json`):**
   - Keyed by population (`relevance_pop`, `exposure_pop`).
   - List of metric dictionaries ordered ascending by $\lambda$.
3. **Sensitivity Analysis (`results/sensitivity.json`):**
   - Propensity model: $P(\text{observe} \mid i) \propto (\text{pop}_i)^\gamma$ for $\gamma \in \{0.25, 0.5, 0.75, 1.0\}$.
   - Reports debiased NDCG and Recall/HitRate across $\lambda$.
4. **Run Manifest (`results/run_manifest.json`):**
   - Logs ALS hyperparameters, score standard deviations, timestamps, environment versions, and seed.

---

## 7. Sign-Off & Interface Agreement

- [x] **Track B (Model + Re-Ranker):** Avinash Julakatti (`PES1UG24AM347`) — *Approved & Frozen on Wed 28-10-2026*
- [x] Verified compatibility with Track A (Data + Evaluation schemas: `ratings.parquet`, `items.parquet`, `src/metrics.py`)
- [x] Verified compatibility with Track C (Schema guards in `tests/test_integration.py`)
