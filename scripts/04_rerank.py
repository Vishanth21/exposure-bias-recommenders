"""
04_rerank.py: Post-hoc popularity taxation re-ranker and lambda parameter sweep.

Owner: Avinash Julakatti (PES1UG24AM347)
Workstream: Model + Re-Ranker (Track B)
Due: Fri 30-10 EOD

Produces:
- results/rerank_sweep.json
- results/rerank_scores_lambda_<L>.parquet
- results/per_user_metrics.parquet

Conforms strictly to:
- report/INTERFACE.md §3 & §5
- report/contracts/AVINASH_MODEL_CANDIDATE_CONTRACT.md §4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Mapping, Sequence, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.contracts.model_candidate_contract import (
    DEFAULT_EVAL_K,
    DEFAULT_LAMBDA_GRID,
    rerank_candidate_list,
    verify_candidate_boundary_invariant,
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


def run_reranking_sweep(
    baseline_scores_path: str = "results/baseline_scores.parquet",
    items_path: str = "data/processed/items.parquet",
    ratings_path: str = "data/processed/ratings.parquet",
    baseline_metrics_path: str = "results/baseline_metrics.json",
    results_dir: str = "results",
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    k: int = DEFAULT_EVAL_K,
) -> Dict[str, object]:
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading artefacts for re-ranking sweep...")
    scores_df = pd.read_parquet(baseline_scores_path)
    items_df = pd.read_parquet(items_path)
    ratings_df = pd.read_parquet(ratings_path)

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    catalogue_items = items_df[items_df["in_catalogue"]].copy()
    catalog_size = len(catalogue_items)
    catalogue_item_ids = sorted(catalogue_items["itemId"].tolist())

    pop_bucket_of: Dict[str, str] = items_df.set_index("itemId")["pop_bucket"].to_dict()
    store_bucket_of: Dict[str, str] = items_df.set_index("itemId")["store_bucket"].to_dict()
    popularity_of: Dict[str, float] = items_df.set_index("itemId")["n_train_ratings"].astype(float).to_dict()

    cat_pop_counts = catalogue_items["pop_bucket"].value_counts()
    item_share: Dict[str, float] = {
        b: float(cat_pop_counts.get(b, 0) / catalog_size) if catalog_size > 0 else 0.0
        for b in ("head", "mid", "tail")
    }

    train_ratings = ratings_df[ratings_df["split"] == "train"]
    train_pos_counts = train_ratings[train_ratings["rating"] >= 4.0].groupby("userId").size()

    exposure_eligible = set(train_pos_counts[train_pos_counts >= 1].index)
    relevance_eligible = set(train_pos_counts[train_pos_counts >= 2].index)

    test_ratings = ratings_df[ratings_df["split"] == "test"]
    test_ground_truth: Dict[str, str] = test_ratings.set_index("userId")["itemId"].to_dict()

    # Pre-organise baseline candidates per user: list of (item_id, score, rank)
    scores_df = scores_df.sort_values(["userId", "rank"])
    user_candidates: Dict[str, List[Tuple[str, float]]] = (
        scores_df.groupby("userId")[["itemId", "score"]]
        .apply(lambda df: list(zip(df["itemId"], df["score"])))
        .to_dict()
    )

    evaluated_users = sorted(list(user_candidates.keys()))
    exp_users = [u for u in evaluated_users if u in exposure_eligible]
    rel_users = [u for u in evaluated_users if u in relevance_eligible and u in test_ground_truth]

    als_baseline_ndcg = baseline_metrics["relevance_pop"]["ndcg"]
    random_baseline_ndcg = baseline_metrics["baselines"]["random"]["relevance_pop"]["ndcg"]
    ndcg_edge = als_baseline_ndcg - random_baseline_ndcg
    baseline_tail_share = baseline_metrics["relevance_pop"]["exposure_share"]["tail"]

    sweep_results: Dict[str, object] = {
        "exposure_pop": [],
        "relevance_pop": [],
        "lambda_grid": list(lambda_grid),
        "chosen_lambda": None,
    }

    per_user_records: List[Dict[str, object]] = []

    for l_val in lambda_grid:
        logger.info("Applying re-ranker for lambda = %.2f", l_val)
        reranked_scores_records: List[Dict[str, object]] = []
        user_reranked_recs: Dict[str, List[str]] = {}

        for u in evaluated_users:
            cands = user_candidates[u]
            c_items = [c[0] for c in cands]
            c_scores = [c[1] for c in cands]
            c_pops = [popularity_of.get(it, 0.0) for it in c_items]

            reranked = rerank_candidate_list(
                candidate_items=c_items,
                candidate_scores=c_scores,
                candidate_popularities=c_pops,
                lambda_val=l_val,
            )

            # Assert boundary invariant: exact permutation of candidates
            reranked_items = [r[0] for r in reranked]
            if not verify_candidate_boundary_invariant(c_items, reranked_items):
                raise AssertionError(f"Candidate boundary invariant violated for user {u} under lambda={l_val}")

            user_reranked_recs[u] = reranked_items

            for new_rank, (it_id, final_sc, _) in enumerate(reranked):
                reranked_scores_records.append({
                    "userId": str(u),
                    "itemId": str(it_id),
                    "score": np.float32(final_sc),
                    "rank": np.int16(new_rank),
                })

        # Save rerank_scores_lambda_<L>.parquet
        l_str = f"{l_val:.2f}"
        rerank_parquet_path = os.path.join(results_dir, f"rerank_scores_lambda_{l_str}.parquet")
        rerank_df = pd.DataFrame(reranked_scores_records)
        rerank_df.to_parquet(rerank_parquet_path, index=False)

        # 1. Evaluate Exposure Population at lambda
        exp_recs_k = [user_reranked_recs[u][:k] for u in exp_users if u in user_reranked_recs]
        exp_exp = exposure_share(exp_recs_k, pop_bucket_of)
        exp_store = store_exposure_share(exp_recs_k, store_bucket_of)
        exp_cross = exposure_by_cross(exp_recs_k, pop_bucket_of, store_bucket_of)
        exp_par = exposure_parity(exp_exp, item_share)
        exp_cov = catalog_coverage(exp_recs_k, catalog_size)

        cat_counts = {it: 0 for it in catalogue_item_ids}
        for recs in exp_recs_k:
            for it in recs:
                if it in cat_counts:
                    cat_counts[it] += 1
        exp_g = gini(cat_counts.values())
        exp_mean_p = mean_recommended_popularity(exp_recs_k, popularity_of)

        exp_block = {
            "lambda": float(l_val),
            "n_users": len(exp_users),
            "exposure_share": exp_exp,
            "store_exposure_share": exp_store,
            "exposure_share_by_pop": exp_cross,
            "exposure_parity": exp_par,
            "coverage": float(exp_cov),
            "gini_popularity": float(exp_g),
            "mean_recommended_popularity": float(exp_mean_p),
        }
        sweep_results["exposure_pop"].append(exp_block)

        # 2. Evaluate Relevance Population at lambda
        rel_recs_k = [user_reranked_recs[u][:k] for u in rel_users if u in user_reranked_recs]
        rel_ndcg_vals = []
        rel_hit_vals = []

        for u in rel_users:
            if u not in user_reranked_recs:
                continue
            recs_u = user_reranked_recs[u][:k]
            target = [test_ground_truth[u]]
            ndcg_u = ndcg_at_k(recs_u, target, k)
            hit_u = hitrate_at_k(recs_u, target, k)
            rel_ndcg_vals.append(ndcg_u)
            rel_hit_vals.append(hit_u)

            per_user_records.append({
                "userId": str(u),
                "population": "relevance_pop",
                "lambda": np.float32(l_val),
                "ndcg": np.float32(ndcg_u),
                "hit": np.float32(hit_u),
            })

        mean_rel_ndcg = float(np.mean(rel_ndcg_vals)) if rel_ndcg_vals else 0.0
        mean_rel_hit = float(np.mean(rel_hit_vals)) if rel_hit_vals else 0.0
        rel_exp = exposure_share(rel_recs_k, pop_bucket_of)
        rel_store = store_exposure_share(rel_recs_k, store_bucket_of)
        rel_cov = catalog_coverage(rel_recs_k, catalog_size)

        rel_counts = {it: 0 for it in catalogue_item_ids}
        for recs in rel_recs_k:
            for it in recs:
                if it in rel_counts:
                    rel_counts[it] += 1
        rel_g = gini(rel_counts.values())

        # Cost & trade-off metrics
        delta_ndcg = mean_rel_ndcg - als_baseline_ndcg
        delta_ndcg_frac = (delta_ndcg / ndcg_edge) if abs(ndcg_edge) > 1e-9 else 0.0
        delta_tail = rel_exp.get("tail", 0.0) - baseline_tail_share

        # Check self-check identity at lambda = 0.0
        if abs(l_val) < 1e-12:
            ndcg_diff = abs(mean_rel_ndcg - als_baseline_ndcg)
            if ndcg_diff > 1e-5:
                logger.warning("Lambda=0.0 NDCG difference vs baseline: %.8f", ndcg_diff)

        rel_block = {
            "lambda": float(l_val),
            "n_users": len(rel_users),
            "ndcg": float(mean_rel_ndcg),
            "hitrate": float(mean_rel_hit),
            "exposure_share": rel_exp,
            "store_exposure_share": rel_store,
            "coverage": float(rel_cov),
            "gini_popularity": float(rel_g),
            "delta_ndcg_vs_baseline": float(delta_ndcg),
            "delta_ndcg_frac_of_als_minus_random": float(delta_ndcg_frac),
            "delta_exposure_tail_vs_baseline": float(delta_tail),
        }
        sweep_results["relevance_pop"].append(rel_block)

    # Save rerank_sweep.json
    sweep_path = os.path.join(results_dir, "rerank_sweep.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    logger.info("Saved re-ranking sweep metrics to %s", sweep_path)

    # Save per_user_metrics.parquet
    per_user_df = pd.DataFrame(per_user_records, columns=["userId", "population", "lambda", "ndcg", "hit"])
    per_user_path = os.path.join(results_dir, "per_user_metrics.parquet")
    per_user_df.to_parquet(per_user_path, index=False)
    logger.info("Saved per-user metrics to %s (%d rows)", per_user_path, len(per_user_df))

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="04_rerank: Run popularity taxation re-ranker sweep.")
    parser.add_argument("--baseline-scores", type=str, default="results/baseline_scores.parquet")
    parser.add_argument("--items-path", type=str, default="data/processed/items.parquet")
    parser.add_argument("--ratings-path", type=str, default="data/processed/ratings.parquet")
    parser.add_argument("--baseline-metrics", type=str, default="results/baseline_metrics.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--k", type=int, default=DEFAULT_EVAL_K)
    args = parser.parse_args()

    run_reranking_sweep(
        baseline_scores_path=args.baseline_scores,
        items_path=args.items_path,
        ratings_path=args.ratings_path,
        baseline_metrics_path=args.baseline_metrics,
        results_dir=args.results_dir,
        k=args.k,
    )
