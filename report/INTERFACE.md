# Interface Contract & Schema Specifications

**Repository:** `Vishanth21/exposure-bias-recommenders`  
**Checkpoint:** Interface Contract Signed Off — Wednesday, 28-10-2026 EOD  
**Governing Rule:** This document defines the exact schemas, data types, invariants, and function signatures that all pipeline scripts code against.

---

## 1. Team Sign-Off Status (Wed 28)

| Member | Track | Responsibility | Sign-Off Status |
| :--- | :--- | :--- | :--- |
| **Vishanth** (`PES1UG24AM323`) | Track A | Data schemas (`ratings`, `items`), metrics library signatures | Signed off |
| **Avinash Julakatti** (`PES1UG24AM347`) | Track B | Embedding artefact format, candidate-set contract, re-ranker invariants | **Signed off (Wed 28)** |
| **Rithesh** (`PES1UG24AM064`) | Track C | Loader API signature, schema validation test suite | Signed off |

---

## 2. Track B Contract: Model & Candidate Generation (Avinash)

*Detailed contract specification available at [AVINASH_MODEL_CANDIDATE_CONTRACT.md](file:///d:/BIG-DATA(MINI%20PROJECT)/exposure-bias-recommenders/report/contracts/AVINASH_MODEL_CANDIDATE_CONTRACT.md).*

### 2.1 Embedding Artefact Format (`results/model/als_factors.npz`)
Compressed `.npz` archive produced by `scripts/02_train.py`:
- `user_factors`: `float32` array of shape `(U, D)` where $D = 64$.
- `item_factors`: `float32` array of shape `(I, D)` where $D = 64$.
- `user_ids`: 1D string array of length $U$ mapping index to `userId`.
- `item_ids`: 1D string array of length $I$ mapping index to `itemId` (`parent_asin`).
- Constraints: Full catalogue ($I$ covers all items with $\ge 3$ train interactions). No sampled negatives. No NaNs or Infs.

### 2.2 Candidate-Set Contract (`results/baseline_scores.parquet`)
Top-200 candidate ordering per evaluated user produced by full-ranking dot-product:
- $\text{score}(u, i) = \mathbf{p}_u^\top \mathbf{q}_i$ across all catalogue items.
- **Seen-item mask:** Strictly excludes any item with `seen == True` in `ratings.parquet` (all prior rated items, including $<4$ ratings).
- Top $N = 200$ items sorted descending by score; ties broken deterministically by `popularity_rank` ascending, then `itemId`.
- Parquet schema:
  - `userId`: `string`
  - `itemId`: `string`
  - `score`: `float32`
  - `rank`: `int16` ($1 \le \text{rank} \le 200$)

### 2.3 Re-Ranker Invariant & Penalty Contract
- **Hard Candidate Boundary:** $\mathcal{S}_{u, \lambda} = \mathcal{C}_{u, \text{baseline}}$. The re-ranker ONLY permutes the top-200 candidate set. No item outside the 200 candidate set can ever be introduced, and no item can be dropped.
- **Z-Scored Penalty Transformation:**
  $$\text{score}_{\text{final}}(u, i) = z(\text{score}(u, i)) - \lambda \cdot z(\log(1 + \text{pop}_i))$$
  Standardization $z(x) = \frac{x - \mu_u}{\sigma_u}$ is computed strictly across the 200 candidates of user $u$.
  Zero-std fallback: if $\sigma_u < 10^{-8}$, $z(x) = 0$.
- Grid: $\lambda \in \{0.0, 0.1, 0.25, 0.5, 1.0, 2.0\}$. $\lambda = 0.0$ preserves original ALS ordering.
- Outputs:
  - `results/rerank_scores_lambda_<L>.parquet`
  - `results/rerank_sweep.json` (summary metrics across $\lambda$)

### 2.4 Baselines Contract (MostPop & Random)
- **MostPop:** Top 200 catalogue items sorted descending by $n_{\text{train\_ratings}}$ (excluding user's seen items).
- **Random:** 200 items sampled uniformly without replacement from unseen catalogue items (seeded per user).
- Both produce identical `(userId, itemId, score, rank)` Parquet schemas for evaluation in `03_evaluate.py`.

### 2.5 Run Manifest Contract (`results/run_manifest.json`)
Records ALS hyperparameters, random seeds, score standard deviations, timestamp, environment packages, and funnel counts.

---

## 3. Track A Contract: Data & Evaluation Artefacts (Vishanth)

### 3.1 Data Artefacts (`01_prepare.py`)
```text
data/processed/ratings.parquet
  userId: string
  itemId: string (parent_asin)
  rating: float32
  timestamp: int64
  split: string ('train' | 'test' | 'seen')
  seen: bool

data/processed/items.parquet
  itemId: string
  title: string
  store: string
  main_category: string
  price: float32
  n_train_ratings: int32
  popularity_rank: int32
  pop_bucket: string ('head' | 'mid' | 'tail')
  store_n_items: int32
  store_n_train: int32
  store_bucket: string ('small' | 'large' | 'unknown')
  in_catalogue: bool
```

### 3.2 Metric Function Signatures (`src/metrics.py`)
```python
def ndcg_at_k(recs, relevant, k) -> float: ...
def recall_at_k(recs, relevant, k) -> float: ...
def hitrate_at_k(recs, relevant, k) -> float: ...
def exposure_share(recs_all, pop_bucket_of, buckets=('head','mid','tail')) -> dict: ...
def catalog_coverage(recs_all, catalog_size) -> float: ...
def gini(values) -> float: ...
def exposure_parity(exposure, item_share, buckets=('head','mid','tail')) -> dict: ...
def exposure_by_cross(recs_all, pop_bucket_of, store_bucket_of) -> dict: ...
def mean_recommended_popularity(recs_all, popularity_of) -> float: ...
```

---

## 4. Track C Contract: Loader & Integration (Rithesh)

### 4.1 Loader API (`src/amazon_loader.py`)
```python
def load_reviews(category: str, limit: int = None) -> Iterator[dict]: ...
def load_metadata(category: str) -> Iterator[dict]: ...
def extract_store(raw_meta: dict) -> str: ...
```

### 4.2 Integration Guards (`tests/test_integration.py`)
Automated schema assertion guards verifying column names, non-null properties, and types across all pipeline outputs.
