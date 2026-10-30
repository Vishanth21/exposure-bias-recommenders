"""
03_evaluate.py: Evaluation of baseline ALS model and reference baselines (MostPop & Random).

Produces:
- results/baseline_metrics.json

Conforms to report/INTERFACE.md §3 and §4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Mapping, Sequence, Set

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.contracts.model_candidate_contract import (
    DEFAULT_EVAL_K,
    DEFAULT_N_CANDIDATES,
    generate_mostpop_candidates,
    generate_random_candidates,
)
from src.metrics import (
    catalog_coverage,
    exposure_by_cross,
    exposure_parity,
    exposure_share,
    gini,
    hitrate_at_k,
    mean_recommended_popularity,
    ndcg_at_k,
    store_exposure_share,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_baseline_models(
    baseline_scores_path: str = "results/baseline_scores.parquet",
    ratings_path: str = "data/processed/ratings.parquet",
    items_path: str = "data/processed/items.parquet",
    results_dir: str = "results",
    k: int = DEFAULT_EVAL_K,
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> Dict[str, object]:
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading artefacts for evaluation...")
    scores_df = pd.read_parquet(baseline_scores_path)
    ratings_df = pd.read_parquet(ratings_path)
    items_df = pd.read_parquet(items_path)

    # Filter catalogue items
    catalogue_items = items_df[items_df["in_catalogue"]].copy()
    catalog_size = len(catalogue_items)
    catalogue_item_ids = sorted(catalogue_items["itemId"].tolist())

    # Build mappings
    pop_bucket_of: Dict[str, str] = items_df.set_index("itemId")["pop_bucket"].to_dict()
    store_bucket_of: Dict[str, str] = items_df.set_index("itemId")["store_bucket"].to_dict()
    popularity_of: Dict[str, float] = items_df.set_index("itemId")["n_train_ratings"].astype(float).to_dict()

    # Item share per bucket over recommendable catalogue
    cat_pop_counts = catalogue_items["pop_bucket"].value_counts()
    item_share: Dict[str, float] = {
        b: float(cat_pop_counts.get(b, 0) / catalog_size) if catalog_size > 0 else 0.0
        for b in ("head", "mid", "tail")
    }

    # Identify user populations
    train_ratings = ratings_df[ratings_df["split"] == "train"]
    train_pos_counts = train_ratings[train_ratings["rating"] >= 4.0].groupby("userId").size()

    exposure_eligible = set(train_pos_counts[train_pos_counts >= 1].index)
    relevance_eligible = set(train_pos_counts[train_pos_counts >= 2].index)

    # Test ground truth: single held-out test item per user
    test_ratings = ratings_df[ratings_df["split"] == "test"]
    test_ground_truth: Dict[str, str] = test_ratings.set_index("userId")["itemId"].to_dict()

    # Evaluated users present in baseline_scores
    evaluated_users = scores_df["userId"].unique().tolist()
    logger.info("Total evaluated users with candidates: %d", len(evaluated_users))

    # Group candidates by user (ordered by rank)
    scores_df = scores_df.sort_values(["userId", "rank"])
    user_candidates: Dict[str, List[str]] = (
        scores_df.groupby("userId")["itemId"].apply(list).to_dict()
    )

    # Seen items per user
    seen_df = ratings_df[ratings_df["seen"] == True]
    user_seen_dict: Dict[str, Set[str]] = seen_df.groupby("userId")["itemId"].apply(set).to_dict()

    # 1. Evaluate Exposure Population (users in evaluated_users with >= 1 train positive)
    exp_users = [u for u in evaluated_users if u in exposure_eligible]
    exp_recs_k = [user_candidates[u][:k] for u in exp_users if u in user_candidates]

    exp_exposure = exposure_share(exp_recs_k, pop_bucket_of)
    exp_store_exp = store_exposure_share(exp_recs_k, store_bucket_of)
    exp_cross = exposure_by_cross(exp_recs_k, pop_bucket_of, store_bucket_of)
    exp_parity = exposure_parity(exp_exposure, item_share)
    exp_cov = catalog_coverage(exp_recs_k, catalog_size)
    
    # Exposure counts for Gini over all catalogue items
    cat_item_counts = {it: 0 for it in catalogue_item_ids}
    for recs in exp_recs_k:
        for it in recs:
            if it in cat_item_counts:
                cat_item_counts[it] += 1
    exp_gini = gini(cat_item_counts.values())
    exp_mean_pop = mean_recommended_popularity(exp_recs_k, popularity_of)

    # 2. Evaluate Relevance Population (users in evaluated_users with >= 2 train positives)
    rel_users = [u for u in evaluated_users if u in relevance_eligible and u in test_ground_truth]
    rel_recs_k = [user_candidates[u][:k] for u in rel_users if u in user_candidates]

    ndcg_vals = [
        ndcg_at_k(user_candidates[u][:k], [test_ground_truth[u]], k)
        for u in rel_users
        if u in user_candidates
    ]
    hit_vals = [
        hitrate_at_k(user_candidates[u][:k], [test_ground_truth[u]], k)
        for u in rel_users
        if u in user_candidates
    ]
    mean_ndcg = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0
    mean_hit = float(np.mean(hit_vals)) if hit_vals else 0.0

    rel_exposure = exposure_share(rel_recs_k, pop_bucket_of)
    rel_store_exp = store_exposure_share(rel_recs_k, store_bucket_of)
    rel_cov = catalog_coverage(rel_recs_k, catalog_size)
    rel_item_counts = {it: 0 for it in catalogue_item_ids}
    for recs in rel_recs_k:
        for it in recs:
            if it in rel_item_counts:
                rel_item_counts[it] += 1
    rel_gini = gini(rel_item_counts.values())

    # 3. Evaluate MostPop Baseline
    item_train_ratings_int = {it: int(items_df.loc[items_df["itemId"] == it, "n_train_ratings"].values[0]) if it in items_df["itemId"].values else 0 for it in catalogue_item_ids}
    
    mostpop_exp_recs = []
    for u in exp_users:
        seen = user_seen_dict.get(u, set())
        cands = generate_mostpop_candidates(catalogue_item_ids, item_train_ratings_int, seen, n_candidates=k)
        mostpop_exp_recs.append([c[0] for c in cands])
    mostpop_exp_share = exposure_share(mostpop_exp_recs, pop_bucket_of)

    mostpop_rel_ndcg = []
    mostpop_rel_hit = []
    for u in rel_users:
        seen = user_seen_dict.get(u, set())
        cands = generate_mostpop_candidates(catalogue_item_ids, item_train_ratings_int, seen, n_candidates=k)
        recs = [c[0] for c in cands]
        target = [test_ground_truth[u]]
        mostpop_rel_ndcg.append(ndcg_at_k(recs, target, k))
        mostpop_rel_hit.append(hitrate_at_k(recs, target, k))

    # 4. Evaluate Random Baseline
    random_exp_recs = []
    for u in exp_users:
        seen = user_seen_dict.get(u, set())
        seed = int(abs(hash((42, str(u)))) % (2**31))
        cands = generate_random_candidates(catalogue_item_ids, seen, seed=seed, n_candidates=k)
        random_exp_recs.append([c[0] for c in cands])
    random_exp_share = exposure_share(random_exp_recs, pop_bucket_of)

    random_rel_ndcg = []
    random_rel_hit = []
    for u in rel_users:
        seen = user_seen_dict.get(u, set())
        seed = int(abs(hash((42, str(u)))) % (2**31))
        cands = generate_random_candidates(catalogue_item_ids, seen, seed=seed, n_candidates=k)
        recs = [c[0] for c in cands]
        target = [test_ground_truth[u]]
        random_rel_ndcg.append(ndcg_at_k(recs, target, k))
        random_rel_hit.append(hitrate_at_k(recs, target, k))

    # Construct final baseline_metrics.json structure
    metrics_result = {
        "k": k,
        "n_candidates": n_candidates,
        "catalog_size": catalog_size,
        "exposure_pop": {
            "n_users": len(exp_users),
            "exposure_share": exp_exposure,
            "store_exposure_share": exp_store_exp,
            "exposure_share_by_pop": exp_cross,
            "exposure_parity": exp_parity,
            "coverage": float(exp_cov),
            "gini_popularity": float(exp_gini),
            "mean_recommended_popularity": float(exp_mean_pop),
        },
        "relevance_pop": {
            "n_users": len(rel_users),
            "ndcg": float(mean_ndcg),
            "hitrate": float(mean_hit),
            "exposure_share": rel_exposure,
            "store_exposure_share": rel_store_exp,
            "coverage": float(rel_cov),
            "gini_popularity": float(rel_gini),
        },
        "baselines": {
            "mostpop": {
                "exposure_pop": {
                    "ndcg": None,
                    "exposure_share": mostpop_exp_share,
                },
                "relevance_pop": {
                    "ndcg": float(np.mean(mostpop_rel_ndcg)) if mostpop_rel_ndcg else 0.0,
                    "hitrate": float(np.mean(mostpop_rel_hit)) if mostpop_rel_hit else 0.0,
                },
            },
            "random": {
                "exposure_pop": {
                    "ndcg": None,
                    "exposure_share": random_exp_share,
                },
                "relevance_pop": {
                    "ndcg": float(np.mean(random_rel_ndcg)) if random_rel_ndcg else 0.0,
                    "hitrate": float(np.mean(random_rel_hit)) if random_rel_hit else 0.0,
                },
            },
        },
    }

    metrics_path = os.path.join(results_dir, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)
    logger.info("Saved baseline metrics to %s", metrics_path)
    return metrics_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="03_evaluate: Evaluate ALS baseline model and baselines.")
    parser.add_argument("--baseline-scores", type=str, default="results/baseline_scores.parquet")
    parser.add_argument("--ratings-path", type=str, default="data/processed/ratings.parquet")
    parser.add_argument("--items-path", type=str, default="data/processed/items.parquet")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--k", type=int, default=DEFAULT_EVAL_K)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    args = parser.parse_args()

    evaluate_baseline_models(
        baseline_scores_path=args.baseline_scores,
        ratings_path=args.ratings_path,
        items_path=args.items_path,
        results_dir=args.results_dir,
        k=args.k,
        n_candidates=args.n_candidates,
    )
