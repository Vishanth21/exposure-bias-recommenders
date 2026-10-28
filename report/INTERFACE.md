# Interface contract — FROZEN 2026-10-28

**Settled:** `implicit` ALS · Electronics.

**Under review, with the observable that would overturn each:**

- **Category** — Electronics stands unless the funnel shows the `>= 3` catalogue too small to
  rank against, or the `store` null rate high enough that the seller claim cannot be made.
- **Seller size** — the mass rule stands unless the funnel puts fewer than ~100 stores in
  `large`, or `small` holds more than ~95% of stores; then it switches to a fixed percentile.
- **Positive threshold** — `>= 4` stands unless the reported positive fraction is so high
  (~90%+) that the threshold separates almost nothing.

---

## 0. Conventions

| Thing | Convention |
|---|---|
| Dataset | **Amazon Reviews 2023, `Electronics` category.** Loaded via Hugging Face `datasets` with `trust_remote_code=True`. Item identity is **`parent_asin`**, not the variant `asin`. |
| IDs | String IDs (`userId`, `itemId`) are kept as **strings** — Amazon IDs are not numeric. The model layer may factorise them, but the parquet schemas below use the raw strings so artefacts stay joinable and inspectable. |
| Store | `store` from item metadata is the seller/publisher. Empty, `"None"`, or missing → `store = ""` and `store_bucket = "unknown"`. Never silently dropped. |
| `K` | **10** for every reported metric. Candidates are `N = 200` (see §2). |
| Rating → implicit | `rating >= 4` → positive (1). Below → unobserved, **not negative**. The threshold is written as 4 because a 4- or 5-star review is the definition of positive, not as a fractional cut-off. Do **not** assert that ratings are integers or that the value set is documented — the card types `rating` as `float`. The loader logs the observed value distribution on first load and reports the positive fraction, since a skewed scale can make the threshold nearly vacuous. |
| Training population | **No filter.** ALS trains on every user and every item with at least one positive — single-interaction users still inform item factors. |
| Catalogue | Items with **`>= 3` train interactions** are *recommendable*. Items below that are trained on but never recommended: their factors are near-random and would inflate coverage and Gini with noise. Sensitivity at 1 and 5. |
| Evaluation populations | **Exposure pop:** users with `>= 1` train positive (exposure needs no ground truth). **Relevance pop:** users with `>= 2` train positives. Both from the same split; see §3. |
| Excluded candidates | Every item the user has already rated is removed from their candidates, **including low-rated ones** — a rated item is observed, not a recommendation opportunity. |
| Split | Temporal leave-last-out per user: the **last positive** (`rating >= 4`) → `test`, all earlier positives → `train`. Ties broken by `timestamp`, then `itemId` for determinism. |
| Candidate set | Catalogue items (`>= 3` train interactions) minus the user's own rated items. Test items outside the catalogue are **excluded**, and the dropped fraction is reported. |
| Seeds | `random.seed(42)`, `np.random.seed(42)`, ALS `random_state=42`. |
| Paths | All paths relative to repo root. |

---

## 1a. Dataset loader API — `src/amazon_loader.py` 

`01_prepare.py` **never imports `datasets` directly**. It calls this module, so there is exactly
one place that touches Hugging Face and exactly one place that knows Amazon's raw field names.
Signature is frozen.

```python
def load_reviews(category: str = "Electronics", limit: int | None = None) -> pd.DataFrame
```
Returns the raw review table with, at minimum, columns `user_id:str`, `parent_asin:str`,
`rating:float32`, `timestamp:int64` (**epoch milliseconds, as Amazon provides — no unit
conversion here**). `limit` truncates for smoke runs and must be applied *before* any filtering
so a smoke run is deterministic. Extra Amazon columns may be present; consumers ignore them.

```python
def load_metadata(category: str = "Electronics", limit: int | None = None) -> pd.DataFrame
```
Returns one row per `parent_asin` with `parent_asin:str`, `title:str`, `store:str`.
`store` is normalised here, not downstream: missing / `None` / `"None"` / empty → `""`.

```python
def extract_store(meta: pd.DataFrame) -> pd.Series
```
Maps a metadata frame to a `parent_asin`-indexed `store` Series with the same normalisation.
Exists so `01_prepare.py` and the integration tests normalise identically.

**Failure modes the loader must raise on, not swallow:**
- category not present in the HF dataset → `ValueError` naming the category
- review rows whose `parent_asin` has no metadata row → dropped, count logged at WARNING
- `timestamp` not in milliseconds (heuristic: median > 1e11) → `ValueError`, because silently
  treating seconds as milliseconds would corrupt the temporal split and every metric after it

---

## 1. Data artefacts (produced by `01_prepare.py`)

### `data/processed/ratings.parquet`

| column | dtype | notes |
|---|---|---|
| `userId` | string | Amazon `user_id`, as-is |
| `itemId` | string | Amazon `parent_asin` — variant ASINs collapse to the parent |
| `rating` | float32 | 1.0–5.0 |
| `timestamp` | int64 | **epoch milliseconds** (Amazon gives ms — do not treat as seconds) |
| `split` | string | `"train"` \| `"test"` \| `"seen"` |
| `seen` | bool | `True` for every row the user rated, positive or not — the candidate-exclusion mask |

**Non-positives (`rating < 4`) are never `train` or `test`.** They carry `split = "seen"` and
`seen = True`, and exist for exactly one purpose: excluding every item the user has already
rated from their candidate list, since a rated item is observed rather than a recommendation
opportunity. They contribute no training signal and no ground truth. Every positive row also
carries `seen = True`.

**No user or item is dropped at this stage.** Every user with at least one positive gets a
`test` row (their last positive) and their remaining positives as `train`. Users with zero
train positives after the split are kept in the file with a `train` count of 0 and are excluded
downstream by population rule, not here.

The funnel — users before/after, items before/after, catalogue size, positive fraction — is
written to `results/prepare_funnel.json` (schema below). Without it, no exposure number is
interpretable.

### `data/processed/items.parquet`

| column | dtype | notes |
|---|---|---|
| `itemId` | string | `parent_asin` |
| `title` | string | |
| `store` | string | seller/publisher; `""` when missing or `"None"` |
| `main_category` | string | from metadata |
| `price` | float32 | `NaN` when missing (very common — do not drop the item) |
| `n_train_ratings` | int32 | popularity signal, train split only |
| `popularity_rank` | int32 | 0 = most popular; dense over train items |
| `pop_bucket` | string | `"head"` \| `"mid"` \| `"tail"` |
| `store_n_items` | int32 | **catalogue** items (`>= 3` train) belonging to this store — primary seller-size measure |
| `store_n_train` | int32 | total train interactions for this item's store — secondary measure |
| `store_bucket` | string | `"small"` \| `"large"` \| `"unknown"`, defined by `store_n_items` (see rule) |
| `in_catalogue` | bool | `True` if `n_train_ratings >= 3`; only these are recommendable |

**Popularity bucket rule (frozen) — interaction mass, not item percentiles.** Item counts tie
heavily at 1–2 interactions, so percentile cut-offs land inside a tie block and are arbitrary.
Bucket over **catalogue** items, sorted by `n_train_ratings` desc (ties → `itemId` asc),
accumulating interactions until each mass boundary is crossed:

| bucket | definition |
|---|---|
| `head` | items cumulatively covering the first **50%** of catalogue train interactions |
| `mid` | items covering the next **30%** |
| `tail` | items covering the last **20%** |

An item straddling a boundary goes to the bucket it enters. Every exposure table must report,
per bucket, both the **item share** and the **interaction share**, so a reader can see that
`head` is a few items holding half the signal.

**Store bucket rule (frozen) — catalogue footprint.** Primary measure is `store_n_items`, the
number of *catalogue* items (`>= 3` train interactions) the store has. Counting over catalogue
items only means dead listings with no interactions do not inflate a store; the remaining
confound, bulk listing, is part of what "large seller" means for slot allocation, and item
count gives a natural parity benchmark (`exposure share / catalogue share`).

**Mass rule, not a percentile** — most stores hold 1–3 catalogue items, so a "top 20% of
stores" cut-off lands inside a large tie block and is arbitrary, exactly as with items. Rank
stores by `store_n_items` desc (ties → `store` asc) and accumulate catalogue items until the
mass boundary is crossed:

| bucket | definition |
|---|---|
| `large` | stores cumulatively covering the first **80%** of catalogue items |
| `small` | stores covering the last **20%** |

A store straddling the boundary goes to the bucket it enters. Every exposure table reports the
**store share** and the **item share** per bucket, so `large` can be read as "N% of stores
holding 80% of the catalogue".

**This rule is unvalidated until the funnel runs, and it can degenerate.** The boundary is a
*distributional* cut, so the number of stores in `large` is not fixed — it is whatever covers
80% of catalogue items. Two failure modes, with the observed direction of each:

| condition | what happens | severity |
|---|---|---|
| stores are **even** (most hold a similar count) | `large` holds ~50-70% of **stores** | expected, harmless |
| **one store dominates** (>=80% of items) | `large` = **1 store**; `small` = every other store | **degenerate** |

The second is the one to watch. Above a top-store item share of 80%, the rule puts a single
seller in `large` and every remaining seller in `small`, so a `large`-vs-`small` comparison
becomes "that one seller vs the world" and the parity ratio is meaningless.

**The funnel decides this before any exposure number is read.** `store_null_rate` is already
reported; the funnel must also report, per store bucket, `n_stores`, `store_share` and
`item_share`. If `large` has fewer than ~100 stores, or `small` holds more than ~95% of stores,
stop and switch the definition to a fixed store percentile instead — at that point the tie-block
objection no longer applies, because store sizes are no longer tied.

This is a **working default, not a settled rule**. `store_n_train` is a **secondary** measure —
it is collinear with item popularity, so a seller effect measured that way would partly restate
the popularity result.

Items with `store == ""` get `store_bucket = "unknown"` and are **reported separately in every
exposure table** — they are neither small nor large, and must not be folded into either. The
`store` null rate is reported in the funnel.

### `data/processed/user_lookup.parquet` / `item_lookup.parquet`

Optional convenience: `userId` → row index, `itemId` → row index, for the model layer.
The raw string ids in the artefacts above are authoritative.

---

## 2. Model artefacts (produced by `02_train.py`)

### `results/model/als_factors.npz`

```
user_factors : float32 [n_users, factors]
item_factors : float32 [n_items, factors]
```

### `results/baseline_scores.parquet`

Top-`N` candidates per evaluated user, `N = 200`:

| column | dtype |
|---|---|
| `userId` | int32 |
| `itemId` | string |
| `score` | float32 |
| `rank` | int16 (0 = best) |

Frozen: already excludes the user's train items (no leak). Exactly `N` rows per test user
except where fewer candidates exist.

### `results/run_manifest.json`

```json
{
  "generated_utc": "2026-10-29T12:00:00Z",
  "git_commit": "abc1234",
  "seed": 42,
  "python": "3.11.9",
  "packages": {"numpy": "...", "pandas": "...", "implicit": "..."},
  "params": {"factors": 64, "iterations": 15, "regularization": 0.05, "alpha": 20,
             "n_candidates": 200, "k": 10, "min_rating": 4.0, "min_catalogue_interactions": 3,
             "min_train_positives_exposure": 1, "min_train_positives_relevance": 2,
             "n_eval_users": 300000, "n_candidates": 200},
  "counts": {"n_users": 0, "n_items": 0, "n_train": 0, "n_test": 0}
}
```

---

## 3. Result artefacts

### `results/prepare_funnel.json`

One row per pipeline stage. This is what makes an exposure number interpretable: it says how
much catalogue actually survived to be protected.

```json
{
  "category": "Electronics",
  "stages": [
    {"stage": "raw_reviews",        "users": 0, "items": 0, "interactions": 0},
    {"stage": "positives_ge_4",     "users": 0, "items": 0, "interactions": 0},
    {"stage": "non_positives_seen", "users": 0, "items": 0, "interactions": 0},
    {"stage": "after_split",        "users": 0, "items": 0, "interactions": 0},
    {"stage": "catalogue_ge_3",     "users": 0, "items": 0, "interactions": 0},
    {"stage": "eval_exposure_pop",  "users": 0, "items": 0, "interactions": 0},
    {"stage": "eval_relevance_pop", "users": 0, "items": 0, "interactions": 0}
  ],
  "positive_fraction": 0.0,
  "store_null_rate": 0.0,
  "test_items_outside_catalogue_frac": 0.0,
  "buckets": {
    "head": {"n_items": 0, "item_share": 0.0, "interaction_share": 0.0},
    "mid":  {"n_items": 0, "item_share": 0.0, "interaction_share": 0.0},
    "tail": {"n_items": 0, "item_share": 0.0, "interaction_share": 0.0}
  },
  "store_buckets": {
    "large":   {"n_stores": 0, "store_share": 0.0, "item_share": 0.0},
    "small":   {"n_stores": 0, "store_share": 0.0, "item_share": 0.0},
    "unknown": {"n_stores": 0, "store_share": 0.0, "item_share": 0.0}
  }
}
```

`store_buckets` is the **degeneracy check** for the store mass rule: read `n_stores` and
`store_share` per bucket before trusting any seller result. If `large.n_stores` is very small
(< ~100) or `small.store_share` exceeds ~95%, the rule has collapsed and the definition must be
switched to a fixed store percentile — see the store bucket rule above.

`positive_fraction` decides whether the threshold is doing work. `store_null_rate` decides how
strong the seller claim can be. `test_items_outside_catalogue_frac` bounds how much recall is
unreachable by construction.

### `results/baseline_metrics.json`

**Keyed by population.** Exposure is measured on the broad population; relevance only where
there is ground truth. Both carry exposure tables, so we can see whether the `>= 2` restriction
shifts the exposure picture.

```json
{
  "k": 10,
  "n_candidates": 200,
  "catalog_size": 0,
  "exposure_pop": {
    "n_users": 0,
    "exposure_share": {"head": 0.0, "mid": 0.0, "tail": 0.0},
    "store_exposure_share": {"small": 0.0, "large": 0.0, "unknown": 0.0},
    "exposure_share_by_pop": {
      "head": {"small": 0.0, "large": 0.0, "unknown": 0.0},
      "mid":  {"small": 0.0, "large": 0.0, "unknown": 0.0},
      "tail": {"small": 0.0, "large": 0.0, "unknown": 0.0}
    },
    "exposure_parity": {"head": 0.0, "mid": 0.0, "tail": 0.0},
    "coverage": 0.0,
    "gini_popularity": 0.0,
    "mean_recommended_popularity": 0.0
  },
  "relevance_pop": {
    "n_users": 0,
    "ndcg": 0.0,
    "hitrate": 0.0,
    "exposure_share": {"head": 0.0, "mid": 0.0, "tail": 0.0},
    "store_exposure_share": {"small": 0.0, "large": 0.0, "unknown": 0.0},
    "coverage": 0.0,
    "gini_popularity": 0.0
  },
  "baselines": {
    "mostpop": {"exposure_pop": {"ndcg": null, "exposure_share": {}}, "relevance_pop": {"ndcg": 0.0, "hitrate": 0.0}},
    "random":  {"exposure_pop": {"ndcg": null, "exposure_share": {}}, "relevance_pop": {"ndcg": 0.0, "hitrate": 0.0}}
  }
}
```

**Semantics, frozen:**

- `exposure_share` sums to 1.0 (±1e-9); same for `store_exposure_share`. They are measured over
  different partitions of the same slots and each sums independently.
- `exposure_share_by_pop[pop_bucket][store_bucket]` is the **control** for the seller claim:
  if `large` is still over-exposed *within* `head` alone, the seller effect is not just
  popularity. One groupby on data already present.
- `exposure_parity[bucket] = exposure_share[bucket] / item_share[bucket]`; 1.0 = exactly
  proportional. Reported instead of raw shares alone, because a bucket with 2% of items
  receiving 5% of slots is the actual finding.
- `recall` is **not** reported separately. With one test item per user, `recall@10 ==
  hitrate@10` identically; reporting both would look like two results.
- `mostpop.exposure_pop.ndcg` is `null` by design — exposure needs no ground truth.
- `gini_popularity` is Gini over **exposure counts** across catalogue items.
- **Evaluation populations are sampled**, not exhaustive. Both populations are drawn as a
  fixed-seed random sample of `n_eval_users = 300000` from the eligible users (exposure:
  `>= 1` train positive; relevance: `>= 2`). Sampling is without replacement, sorted by
  `userId` before the draw so the result is order-independent. The seed is recorded in
  `run_manifest.json`; the funnel reports both the eligible count and the sampled count per
  population. Without this cap, 200 candidates × millions of users makes
  `baseline_scores.parquet` unmanageable and multiplies per-user metrics by six λ values.
- **Per-user arrays are saved** to `results/per_user_metrics.parquet`
  (`userId, population, lambda, ndcg, hit`) so a paired bootstrap on the cost delta is ~15
  lines whenever there is time. Storing them costs nothing; recomputing them later costs a run.

### `results/rerank_sweep.json`

**Keyed by population**, then a list per population, ordered ascending by `lambda`:

```json
{
  "exposure_pop": [
    {
      "lambda": 0.0,
      "n_users": 0,
      "exposure_share": {"head": 0.0, "mid": 0.0, "tail": 0.0},
      "store_exposure_share": {"small": 0.0, "large": 0.0, "unknown": 0.0},
      "exposure_share_by_pop": {
        "head": {"small": 0.0, "large": 0.0, "unknown": 0.0},
        "mid":  {"small": 0.0, "large": 0.0, "unknown": 0.0},
        "tail": {"small": 0.0, "large": 0.0, "unknown": 0.0}
      },
      "exposure_parity": {"head": 0.0, "mid": 0.0, "tail": 0.0},
      "coverage": 0.0,
      "gini_popularity": 0.0,
      "mean_recommended_popularity": 0.0
    }
  ],
  "relevance_pop": [
    {
      "lambda": 0.0,
      "n_users": 0,
      "ndcg": 0.0,
      "hitrate": 0.0,
      "exposure_share": {"head": 0.0, "mid": 0.0, "tail": 0.0},
      "store_exposure_share": {"small": 0.0, "large": 0.0, "unknown": 0.0},
      "coverage": 0.0,
      "gini_popularity": 0.0,
      "delta_ndcg_vs_baseline": 0.0,
      "delta_ndcg_frac_of_als_minus_random": 0.0,
      "delta_exposure_tail_vs_baseline": 0.0
    }
  ],
  "lambda_grid": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
  "chosen_lambda": null
}
```

`lambda = 0.0` **must** be present in both lists and must reproduce the corresponding block of
`baseline_metrics.json` within `1e-6`. That equality is the self-check that the sweep harness
is wired correctly.

`delta_ndcg_frac_of_als_minus_random` anchors the cost: `ΔNDCG / (NDCG_als − NDCG_random)`.
Without it, "we lost 0.004 NDCG" is uninterpretable; with it, the cost reads as a fraction of
the model's entire edge over a trivial baseline.

`chosen_lambda` is `null` until a human picks one from the curve; `05_report.py` must handle
`null` by plotting the full sweep rather than a chosen point.

### `results/sensitivity.json` — **OPTIONAL**

Consumers must handle the file being absent: `06_report.py` skips the sensitivity panel.

γ-weighted debiased relevance, keyed by population. `gamma` is the exponent on inverse
popularity; weights are normalised to sum to 1 within a user's relevant set.

```json
{
  "exposure_pop":   {"n_users": 0, "gamma_values": [0.5], "metrics": {"0.5": {"ndcg": 0.0, "hitrate": 0.0}}},
  "relevance_pop":  {"n_users": 0, "gamma_values": [0.5], "metrics": {"0.5": {"ndcg": 0.0, "hitrate": 0.0}}}
}
```

`gamma = 0` must reproduce `baseline_metrics.json` exactly — that equality is the self-check
that the weighting is applied correctly, the same way `lambda = 0.0` must reproduce the baseline.
`recall` is not carried here for the same reason it is not carried in `baseline_metrics.json`:
with one test item per user, `recall@10 == hitrate@10` identically.

---

### `results/rerank_scores_lambda_<L>.parquet`

Same schema as `baseline_scores.parquet`, for the chosen lambda(s). `<L>` formatted with
two decimals, e.g. `rerank_scores_lambda_0.50.parquet`.

---

## 4. Metric functions — `src/metrics.py`

Stdlib only. Inputs are plain Python objects.

```python
def ndcg_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float
def recall_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float
def hitrate_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float
def exposure_share(recs_all: Iterable[Sequence[int]], pop_bucket_of: Mapping[int, str],
                   buckets: Sequence[str] = ("head", "mid", "tail")) -> dict[str, float]
def store_exposure_share(recs_all: Iterable[Sequence[int]],
                         store_bucket_of: Mapping[int, str],
                         ) -> dict[str, float]
def catalog_coverage(recs_all: Iterable[Sequence[int]], catalog_size: int) -> float
def gini(values: Iterable[float]) -> float

# added for the two-population design
def exposure_parity(exposure: Mapping[str, float], item_share: Mapping[str, float],
                    buckets: Sequence[str] = ("head", "mid", "tail")) -> dict[str, float]
def exposure_by_cross(recs_all: Iterable[Sequence[int]], pop_bucket_of: Mapping[int, str],
                      store_bucket_of: Mapping[int, str],
                      buckets: Sequence[str] = ("head", "mid", "tail"),
                      store_buckets: Sequence[str] = ("small", "large", "unknown"),
                      ) -> dict[str, dict[str, float]]
def mean_recommended_popularity(recs_all: Iterable[Sequence[int]],
                                popularity_of: Mapping[int, float]) -> float
```

**Semantics, frozen:**

- `recs` is ranked best-first; only the first `k` are used.
- `relevant` may be a list/set; if it contains an item not in the candidate set that is the
  caller's problem — the function does not validate it.
- Binary relevance: an item is relevant if it is in `relevant`. No graded relevance.
- **`ndcg_at_k`** — `DCG = Σ_{i=0..k-1} rel_i / log2(i+2)`; `IDCG` from the ideal ordering,
  i.e. `Σ_{j=0..min(|relevant|,k)-1} 1 / log2(j+2)`. Return `DCG/IDCG`, and `0.0` when
  `IDCG == 0`. This is the **binary** NDCG, which is standard for implicit feedback.
- **`recall_at_k`** — `|top_k ∩ relevant| / |relevant|`; `0.0` if `relevant` is empty.
- **`hitrate_at_k`** — `1.0` if any of top `k` is relevant else `0.0`.
- **`exposure_share`** — counts every slot in every rec list (list length may exceed `k` only
  if the caller passes more; we count what is passed, **not** truncated). Items missing from
  `pop_bucket_of` are **silently skipped** — the caller must bucket the full candidate set;
  the function is total and never raises on a partial mapping. Returns a dict containing a key
  for every bucket in `buckets`, each a fraction of total counted slots; all zeros if no slots.
  Sums to 1.0 ±1e-9.
- **Unknown bucket labels are counted, not dropped.** If a passed mapping yields a label
  outside `buckets`, the function adds that key so the shares still sum to 1.0. The result may
  therefore carry keys beyond `buckets`; callers must not assume the dict is keyed exactly by
  the requested vocabulary. This is a deliberate choice: dropping the slot would silently
  deflate the total and make every share wrong.
- **`store_exposure_share`** — `exposure_share` over the store partition, with the vocabulary
  `("small", "large", "unknown")`. All three keys are **always present** in the result, even
  when the mapping never used one, because `unknown` is a reported line and must not vanish
  when empty. Items absent from `store_bucket_of` are skipped, not folded into `unknown` —
  folding them would bias the headline seller number. Sums to 1.0 ±1e-9 when any slot is
  counted, all zeros otherwise.
- **`catalog_coverage`** — `|distinct items in all rec lists| / catalog_size`; `0.0` if
  `catalog_size == 0`.
- **`gini`** — standard Gini over non-negative values, ascending sort:
  `G = (2·Σ i·x_i)/(n·Σ x_i) - (n+1)/n`, with `i` 1-based. Returns `0.0` for `n == 0` or
  all-zero values. Clamped to `[0, 1]` to absorb float noise.

**Added-function semantics, frozen:**

- **`exposure_parity`** — elementwise `exposure[b] / item_share[b]`, `0.0` where
  `item_share[b] == 0`. Returns a key for every bucket in `buckets`. 1.0 means a bucket takes
  slots in exact proportion to its share of the catalogue.
- **`exposure_by_cross`** — slot counts over the *joint* partition, returned as
  `{pop_bucket: {store_bucket: fraction}}`, each row summing to that pop bucket's share of all
  slots and the whole structure summing to 1.0 ±1e-9. This is the control that separates a
  seller effect from a popularity effect. As with `exposure_share`, an unknown label from
  either mapping is **added** rather than dropped, so the result may carry keys beyond the
  requested vocabularies while the sum-to-one invariant holds.
- **`mean_recommended_popularity`** — mean of `popularity_of[i]` over every recommended slot,
  skipping items absent from the mapping (caller's responsibility to supply the full candidate
  set). `0.0` when there are no slots. Reported in raw interaction counts, so it is readable
  without reference to a bucket rule.

**Efficiency note:** these are per-user or small-set functions. `03_evaluate.py` loops over
test users and aggregates. Do not put pandas groupbys inside the metric functions.

---

## 5. Re-ranker contract — `04_rerank.py`

Input: `baseline_scores.parquet` + `items.parquet` (`n_train_ratings`, `pop_bucket`,
`store_bucket`, `store_n_items`).

**Both terms are z-scored within each user's candidate list before combining.** ALS
dot-product scores have an arbitrary scale, so a fixed λ against raw scores means an unknown
amount of reordering — often none. Z-scoring makes λ readable as *standard deviations of score
traded per standard deviation of popularity*.

```
z_score = (score - mean(score_u)) / std(score_u)
z_pop   = (log1p(n_train_ratings) - mean(log1p(...)_u)) / std(log1p(...)_u)

final_score = z_score - lambda * z_pop
```

`std` of 0 (single candidate, or all-equal) → substitute 1.0 for that user rather than
dividing by zero; log it at DEBUG.

Then re-sort descending and recompute `rank` 0..N-1. **No new items are introduced** — the
re-ranker may only permute the existing candidate set. Hard invariant; assert it.

Tie-break on equal `final_score`: original `score` desc, then `itemId` asc.

`lambda = 0.0` must be a no-op (identity on the candidate set, same ranks) — z-scoring must not
perturb the order of `score` when the penalty is zero, which holds because z-scoring is a
monotone affine transform.

Lambdas to sweep: `[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]`. With z-scoring these are meaningful
without rescaling; record the observed per-user score std in `run_manifest.json` so the report
can state the scale the penalty operated at.

---

## 6. What each stage may rely on

| Producer | Guarantee |
|---|---|
| `01_prepare.py` | Ratings/items parquet exactly as §1, one test row per user, `in_catalogue` set, funnel JSON |
| `02_train.py` | `baseline_scores.parquet` with up to 200 rows/user, no train leaks, no already-rated items |
| `03_evaluate.py` | `MostPop`/`Random` rows in `baseline_metrics.json` |
| `04_rerank.py` | `rerank_sweep.json` keyed by population, `lambda=0.0` matching baseline, per-user arrays saved |
| `06_sensitivity.py` | `sensitivity.json` with `gamma = 0` reproducing the baseline exactly |
| `src/metrics.py` | Functions importable with stdlib only, unit-tested |
| `src/amazon_loader.py` | The three functions in §1a, frozen signatures, no `datasets` import anywhere else |
| `test_integration.py` | Asserts §1–§3 schemas against real artefacts |
| `05_report.py` | Reads only `results/*.json` and `results/*.parquet`; never recomputes metrics |

**Rule:** if a consumer needs a field that is not in this document, the field is added to
this document **first**, then implemented. The document is the source of truth.