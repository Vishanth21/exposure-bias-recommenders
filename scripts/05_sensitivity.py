"""
05_sensitivity.py: Gamma-weighted debiased relevance sensitivity analysis.

Owner: Avinash Julakatti (PES1UG24AM347)
Workstream: Model + Re-Ranker (Track B)
Due: Fri 30-10 EOD

Produces:
- results/sensitivity.json

Conforms strictly to report/INTERFACE.md §3 & §6.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Sequence

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.contracts.model_candidate_contract import DEFAULT_EVAL_K
from src.metrics import hitrate_at_k, ndcg_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GAMMA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def compute_sensitivity_analysis(
    baseline_scores_path: str = "results/baseline_scores.parquet",
    ratings_path: str = "data/processed/ratings.parquet",
    items_path: str = "data/processed/items.parquet",
    baseline_metrics_path: str = "results/baseline_metrics.json",
    results_dir: str = "results",
    gamma_values: Sequence[float] = DEFAULT_GAMMA_VALUES,
    k: int = DEFAULT_EVAL_K,
) -> Dict[str, object]:
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading artefacts for sensitivity analysis...")
    scores_df = pd.read_parquet(baseline_scores_path)
    ratings_df = pd.read_parquet(ratings_path)
    items_df = pd.read_parquet(items_path)

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    # Item popularity mapping
    popularity_of: Dict[str, float] = items_df.set_index("itemId")["n_train_ratings"].astype(float).to_dict()

    # Identify user populations
    train_ratings = ratings_df[ratings_df["split"] == "train"]
    train_pos_counts = train_ratings[train_ratings["rating"] >= 4.0].groupby("userId").size()

    exposure_eligible = set(train_pos_counts[train_pos_counts >= 1].index)
    relevance_eligible = set(train_pos_counts[train_pos_counts >= 2].index)

    test_ratings = ratings_df[ratings_df["split"] == "test"]
    test_ground_truth: Dict[str, str] = test_ratings.set_index("userId")["itemId"].to_dict()

    # Pre-organise candidates per user
    scores_df = scores_df.sort_values(["userId", "rank"])
    user_candidates = scores_df.groupby("userId")["itemId"].apply(list).to_dict()

    evaluated_users = sorted(list(user_candidates.keys()))
    exp_users = [u for u in evaluated_users if u in exposure_eligible]
    rel_users = [u for u in evaluated_users if u in relevance_eligible and u in test_ground_truth]

    # Precompute per-user unweighted metrics and test item popularities
    user_metrics: List[Dict[str, float]] = []
    for u in rel_users:
        recs_u = user_candidates[u][:k]
        test_item = test_ground_truth[u]
        target = [test_item]
        
        ndcg_u = ndcg_at_k(recs_u, target, k)
        hit_u = hitrate_at_k(recs_u, target, k)
        pop_item = max(1.0, float(popularity_of.get(test_item, 1.0)))

        user_metrics.append({
            "ndcg": ndcg_u,
            "hit": hit_u,
            "pop": pop_item,
        })

    rel_metrics_by_gamma: Dict[str, Dict[str, float]] = {}
    baseline_ndcg = baseline_metrics["relevance_pop"]["ndcg"]
    baseline_hit = baseline_metrics["relevance_pop"]["hitrate"]

    for gamma in gamma_values:
        gamma_key = f"{gamma:.2f}" if gamma != 0.0 else "0.0"
        if not user_metrics:
            rel_metrics_by_gamma[gamma_key] = {"ndcg": 0.0, "hitrate": 0.0}
            continue

        if abs(gamma) < 1e-12:
            # gamma = 0: exact unweighted mean reproducing baseline
            mean_ndcg = float(np.mean([m["ndcg"] for m in user_metrics]))
            mean_hit = float(np.mean([m["hit"] for m in user_metrics]))
            rel_metrics_by_gamma[gamma_key] = {
                "ndcg": float(mean_ndcg),
                "hitrate": float(mean_hit),
            }
        else:
            # Inverse propensity weights: w = 1 / (pop ^ gamma)
            raw_weights = np.array([1.0 / np.power(m["pop"], gamma) for m in user_metrics], dtype=np.float64)
            w_sum = np.sum(raw_weights)
            if w_sum > 0:
                norm_weights = raw_weights / w_sum
                debiased_ndcg = float(np.sum(norm_weights * np.array([m["ndcg"] for m in user_metrics])))
                debiased_hit = float(np.sum(norm_weights * np.array([m["hit"] for m in user_metrics])))
            else:
                debiased_ndcg = 0.0
                debiased_hit = 0.0

            rel_metrics_by_gamma[gamma_key] = {
                "ndcg": float(debiased_ndcg),
                "hitrate": float(debiased_hit),
            }

    # Exposure pop carries null for relevance metrics (no ground truth needed)
    exp_metrics_by_gamma: Dict[str, Dict[str, object]] = {
        (f"{gamma:.2f}" if gamma != 0.0 else "0.0"): {"ndcg": None, "hitrate": None}
        for gamma in gamma_values
    }

    sensitivity_result = {
        "exposure_pop": {
            "n_users": len(exp_users),
            "gamma_values": list(gamma_values),
            "metrics": exp_metrics_by_gamma,
        },
        "relevance_pop": {
            "n_users": len(rel_users),
            "gamma_values": list(gamma_values),
            "metrics": rel_metrics_by_gamma,
        },
    }

    sensitivity_path = os.path.join(results_dir, "sensitivity.json")
    with open(sensitivity_path, "w", encoding="utf-8") as f:
        json.dump(sensitivity_result, f, indent=2)
    logger.info("Saved sensitivity analysis to %s", sensitivity_path)

    return sensitivity_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="05_sensitivity: Compute debiased relevance sensitivity analysis.")
    parser.add_argument("--baseline-scores", type=str, default="results/baseline_scores.parquet")
    parser.add_argument("--ratings-path", type=str, default="data/processed/ratings.parquet")
    parser.add_argument("--items-path", type=str, default="data/processed/items.parquet")
    parser.add_argument("--baseline-metrics", type=str, default="results/baseline_metrics.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--k", type=int, default=DEFAULT_EVAL_K)
    args = parser.parse_args()

    compute_sensitivity_analysis(
        baseline_scores_path=args.baseline_scores,
        ratings_path=args.ratings_path,
        items_path=args.items_path,
        baseline_metrics_path=args.baseline_metrics,
        results_dir=args.results_dir,
        k=args.k,
    )
