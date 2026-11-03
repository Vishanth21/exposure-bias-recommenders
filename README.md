# exposure-bias-recommenders

How much relevance does a popularity debias cost? Measured on Amazon Reviews 2023 (Electronics,
43.9 M reviews). Implicit-feedback ALS generates candidates, and a z-scored popularity penalty
re-ranks them. Exposure and relevance are measured on separate user populations and reported per
popularity bucket and per seller bucket.

**Headline (run `c15fbb8`):** the 2.7 % of items that form the head get **99.96 %** of top-10
slots, and the long tail (81 % of the catalogue) gets **0 %**. At λ = 0.5 the re-ranker cuts mean
recommended popularity by 36 % for **2.1 %** of ALS's NDCG@10 edge over Random. Tail exposure stays
at zero at every λ, because a re-ranker can only reorder candidates that already exist. Full
write-up: [`report/REPORT.md`](report/REPORT.md).

## Architecture

```
                 Hugging Face / datarepo (Amazon Reviews 2023, Electronics)
                                       │
                          src/amazon_loader.py   load_reviews · load_metadata · extract_store
                                       │         (only module that touches the raw schema)
                                       ▼
 01_prepare.py ──▶ data/processed/ratings.parquet, items.parquet ──▶ results/prepare_funnel.json
   temporal leave-last-out · no drops · catalogue (≥3 train) · head/mid/tail · large/small/unknown
                                       │
                                       ▼
 02_train.py ───▶ results/model/als_factors.npz, baseline_scores.parquet (top-200), run_manifest.json
   implicit ALS on all positives · full-catalogue scoring · already-rated items masked
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
 03_evaluate.py               04_rerank.py                 05_sensitivity.py
 ALS · MostPop · Random       final = z(score) − λ·z(pop)   γ-weighted (1/pop^γ) NDCG/HR
 NDCG/HR, exposure, parity,   λ ∈ {0,.1,.25,.5,1,2}         γ = 0 ≡ baseline
 store×pop, coverage, Gini    permute-only, λ = 0 ≡ baseline
        │                              │                         │
 baseline_metrics.json        rerank_sweep.json             sensitivity.json
                              rerank_scores_lambda_*.parquet
                              per_user_metrics.parquet
              └────────────────────────┼─────────────────────────┘
                                       ▼
 06_report.py ──▶ figures/*.png + markdown tables    (reads results/* only, never recomputes)
                                       ▼
            report/results.ipynb ──▶ report/REPORT.md · report/DATA_CARD.md

 src/metrics.py (stdlib only) is shared by 03/04/05.   report/INTERFACE.md is the frozen contract.
 tests/test_integration.py guards every arrow above against INTERFACE.md §1–§3.
```

## Layout

```
report/INTERFACE.md        frozen contract: schemas, metric signatures, loader API
report/DATASET_FACTS.md    dataset and library facts, cited to primary sources
report/DATA_CARD.md        data card (licence, provenance, composition, known gaps)
report/REPORT.md           technical report
report/results.ipynb       executed results notebook (reads results/*.json only)
src/amazon_loader.py       dataset loader (HF streaming, local JSONL, store/price normalisation)
src/metrics.py             ranking and exposure metrics, stdlib only
src/contracts/             model/candidate contract shared by 02 and 04
scripts/01_prepare.py … 06_report.py   pipeline stages, each independently runnable
tests/                     unit tests (metrics, contract, model, re-rank) + integration suite
figures/                   report figures produced by 06_report.py / the notebook
```

## Setup

Python 3.10–3.12. The pinned wheels (numpy 2.1, pyarrow 18, implicit 0.7.2) are not available for
3.13+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

The dataset is not gated, so no token is needed. On first run the loader streams from Hugging
Face and caches `data/raw/*.parquet`. To skip the stream, drop the raw archives
(`Electronics.jsonl.gz`, `meta_Electronics.jsonl.gz`) from
`https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/` into `data/raw/`.

## Running

Run the stages in order from the repo root. Each one reads only the artefacts of the earlier
stages.

```bash
python scripts/01_prepare.py --category Electronics      # add --limit 200000 for a smoke run
python scripts/02_train.py
python scripts/03_evaluate.py
python scripts/04_rerank.py
python scripts/05_sensitivity.py
python scripts/06_report.py                               # tables to stdout, figures to figures/
jupyter nbconvert --to notebook --execute --inplace report/results.ipynb
```

The full Electronics run needs a cloud machine (tens of GB of RAM). Its artefacts are in the
[results folder on Google Drive](https://drive.google.com/drive/folders/16avB4aidRVUhmEpEfITzkUHk2B6qsf4h?usp=sharing).
To rebuild the report without re-running the pipeline, copy the five `*.json` files into
`results/` and run `06_report.py` and the notebook.

## Tests

```bash
python -m pytest
```

- `test_metrics.py`: 52 hand-computed metric cases.
- `test_model_candidate_contract.py`, `test_model_pipeline.py`, `test_rerank_sweep.py`: toy
  end-to-end runs of 02–05.
- `test_integration.py`: schema guards on every artefact boundary.
  - It pushes an Amazon-shaped JSONL slice through the loader and `01_prepare.py` and validates
    `ratings`/`items`/`funnel` against INTERFACE.md.
  - It checks the real `results/*` and `data/processed/*` artefacts, including the λ = 0 and
    γ = 0 self-checks and the store-bucket degeneracy gate.
  - A real-artefact test skips when its file is absent, so a fresh clone is green and a results
    checkout is strict.

## Method notes

- ALS trains on every interaction. Evaluation populations (`>= 1` train positive for exposure,
  `>= 2` for relevance, 300 k sampled each with seed 42) and the catalogue (`>= 3` train
  interactions) are views applied downstream, never drops at prepare time.
- A positive is `rating >= 4`. Below that is unobserved, not negative.
- Popularity buckets are cut on interaction mass (50/30/20), not item percentile, because item
  counts tie heavily at 1–2.
- Seller size is catalogue item count per `store` (mass rule: large = stores covering 80 % of
  catalogue items). An empty store is reported as `unknown`, never folded into `small`.
- Both re-ranker terms are z-scored per user, so λ is in standard deviations of score.
- The relevance cost is measured on popularity-biased observed interactions and is reported as an
  upper bound.

`data/` and `results/` are gitignored and regenerated by the pipeline.

## Team

Vishanth (PES1UG24AM323) — data + evaluation · Avinash Julakatti (PES1UG24AM347) — model +
re-ranker · Rithesh (PES1UG24AM064) — loader, integration tests, report, data card, notebook,
packaging.
