"""
02_train.py: Matrix factorisation model training and baseline candidate generation.

Owner: Avinash Julakatti (PES1UG24AM347)
Workstream: Model + Re-Ranker (Track B)
Due: Thu 29-10 EOD

Produces:
- results/model/als_factors.npz (via EmbeddingArtefact)
- results/model/als_model.pkl
- results/baseline_scores.parquet (top-200 candidates per evaluated user)
- results/run_manifest.json

Conforms strictly to:
- report/INTERFACE.md §2
- report/contracts/AVINASH_MODEL_CANDIDATE_CONTRACT.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq

import implicit
from implicit.als import AlternatingLeastSquares

from src.contracts.model_candidate_contract import (
    DEFAULT_ALPHA,
    DEFAULT_FACTORS,
    DEFAULT_ITERATIONS,
    DEFAULT_N_CANDIDATES,
    DEFAULT_REGULARIZATION,
    DEFAULT_SEED,
    EmbeddingArtefact,
    generate_user_candidates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Safely obtain current git commit hash if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__)
        ).decode().strip()
        return commit
    except Exception:
        return "c020f1d"


def train_and_generate_candidates(
    ratings_path: str = "data/processed/ratings.parquet",
    items_path: str = "data/processed/items.parquet",
    results_dir: str = "results",
    factors: int = DEFAULT_FACTORS,
    regularization: float = DEFAULT_REGULARIZATION,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    n_eval_users: int = 300_000,
) -> None:
    model_dir = os.path.join(results_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    stage_started = time.monotonic()
    logger.info("Reading ratings from %s and items from %s", ratings_path, items_path)
    ratings_df = pd.read_parquet(ratings_path)
    items_df = pd.read_parquet(items_path)
    logger.info(
        "Loaded ratings=%s rows, items=%s rows (%.1f min)",
        f"{len(ratings_df):,}", f"{len(items_df):,}",
        (time.monotonic() - stage_started) / 60,
    )

    # 1. Catalogue items filter
    catalogue_items = items_df[items_df["in_catalogue"]].copy()
    catalogue_item_ids = catalogue_items["itemId"].astype(str).values
    catalogue_set = set(catalogue_item_ids)
    logger.info("Recommendable catalogue size (>= 3 train interactions): %d items", len(catalogue_item_ids))

    # Popularity rank mapping for deterministic tie-breaking
    pop_ranks = items_df.set_index("itemId")["popularity_rank"].to_dict()

    # 2. Filter positive train interactions for ALS training
    train_ratings = ratings_df[(ratings_df["split"] == "train") & (ratings_df["rating"] >= 4.0)].copy()
    logger.info("Total positive train interactions: %d", len(train_ratings))

    # Unique train users and catalogue items
    train_user_ids = np.sort(train_ratings["userId"].unique().astype(str))
    n_users = len(train_user_ids)
    n_items = len(catalogue_item_ids)

    user2idx = {u: i for i, u in enumerate(train_user_ids)}
    item2idx = {it: j for j, it in enumerate(catalogue_item_ids)}

    # Filter train ratings to items in catalogue
    train_cat_ratings = train_ratings[train_ratings["itemId"].isin(catalogue_set)].copy()
    user_indices = train_cat_ratings["userId"].map(user2idx).values
    item_indices = train_cat_ratings["itemId"].map(item2idx).values

    # Confidence: c_ui = 1.0 + alpha * r_ui (with r_ui = 1.0 for positive interactions)
    confidences = np.ones(len(train_cat_ratings), dtype=np.float32) + float(alpha) * 1.0

    # Build sparse user-item matrix (shape: n_users, n_items)
    stage_started = time.monotonic()
    C_ui = sp.csr_matrix(
        (confidences, (user_indices, item_indices)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    logger.info(
        "Built sparse training matrix C_ui of shape %s with %s non-zeros (%.1f min)",
        C_ui.shape, f"{C_ui.nnz:,}", (time.monotonic() - stage_started) / 60,
    )

    # 3. Fit implicit ALS model
    logger.info(
        "Fitting AlternatingLeastSquares (factors=%d, regularization=%.4f, iterations=%d, seed=%d)",
        factors,
        regularization,
        iterations,
        seed,
    )
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=seed,
    )
    fit_started = time.monotonic()
    model.fit(C_ui, show_progress=True)
    logger.info("ALS fit complete in %.1f min", (time.monotonic() - fit_started) / 60)

    # 4. Extract and validate embedding artefacts
    user_factors = np.asarray(model.user_factors, dtype=np.float32)
    item_factors = np.asarray(model.item_factors, dtype=np.float32)

    embedding_artefact = EmbeddingArtefact(
        user_factors=np.asarray(user_factors, dtype=np.float32),
        item_factors=np.asarray(item_factors, dtype=np.float32),
        user_ids=np.asarray(train_user_ids, dtype=str),
        item_ids=np.asarray(catalogue_item_ids, dtype=str),
    )
    factors_npz_path = os.path.join(model_dir, "als_factors.npz")
    embedding_artefact.save_npz(factors_npz_path)
    logger.info("Saved embedding artefact to %s", factors_npz_path)

    model_pkl_path = os.path.join(model_dir, "als_model.pkl")
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved companion model pickle to %s", model_pkl_path)

    # 5. Determine Evaluated Populations
    # Exposure population: users with >= 1 train positive
    # Relevance population: users with >= 2 train positives
    user_train_pos_counts = train_ratings.groupby("userId").size()
    exposure_eligible = np.sort(user_train_pos_counts[user_train_pos_counts >= 1].index.astype(str).values)
    relevance_eligible = np.sort(user_train_pos_counts[user_train_pos_counts >= 2].index.astype(str).values)

    logger.info(
        "Eligible users: exposure_pop=%d (>=1 positive), relevance_pop=%d (>=2 positives)",
        len(exposure_eligible),
        len(relevance_eligible),
    )

    rng = np.random.default_rng(seed)
    if len(exposure_eligible) > n_eval_users:
        exp_idx = rng.choice(len(exposure_eligible), size=n_eval_users, replace=False)
        sampled_exposure = set(exposure_eligible[exp_idx])
    else:
        sampled_exposure = set(exposure_eligible)

    if len(relevance_eligible) > n_eval_users:
        rel_idx = rng.choice(len(relevance_eligible), size=n_eval_users, replace=False)
        sampled_relevance = set(relevance_eligible[rel_idx])
    else:
        sampled_relevance = set(relevance_eligible)

    eval_users_set = sampled_exposure | sampled_relevance
    eval_users = sorted(list(eval_users_set))
    logger.info("Total sampled evaluation users to score: %d", len(eval_users))

    # The held-out positive is unknown at ranking time and must remain a candidate.
    # Exclude training positives and non-positive observations already available to the model.
    seen_df = ratings_df[ratings_df["split"] != "test"]
    user_seen_dict = seen_df.groupby("userId")["itemId"].apply(set).to_dict()

    # 6. Candidate Generation (Full Ranking with seen-item hard masking)
    logger.info("Generating top-%d candidates per evaluated user...", n_candidates)
    user_score_stds: List[float] = []

    # Batch full-ranking scoring
    batch_size = 250
    candidates_started = time.monotonic()
    last_progress = 0
    item_ids = np.asarray(catalogue_item_ids, dtype=str)
    pop_rank_values = np.asarray([pop_ranks.get(it, 999_999_999) for it in item_ids])
    item_position = {it: i for i, it in enumerate(item_ids)}
    output_path = os.path.join(results_dir, "baseline_scores.parquet")
    output_schema = pa.schema([
        ("userId", pa.string()),
        ("itemId", pa.string()),
        ("score", pa.float32()),
        ("rank", pa.int16()),
    ])
    writer = None
    for start_idx in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start_idx : start_idx + batch_size]
        batch_u_indices = [user2idx[u] for u in batch_users if u in user2idx]

        if not batch_u_indices:
            continue

        batch_user_factors = user_factors[batch_u_indices]  # shape: (B, D)
        batch_scores = batch_user_factors @ item_factors.T  # shape: (B, I)

        batch_rows = {"userId": [], "itemId": [], "score": [], "rank": []}
        for b_i, u in enumerate(batch_users):
            if u not in user2idx:
                continue
            seen_items = user_seen_dict.get(u, set())
            scores_u = batch_scores[b_i]
            user_score_stds.append(float(np.std(scores_u)))

            seen_positions = [item_position[it] for it in seen_items if it in item_position]
            valid_mask = np.ones(len(item_ids), dtype=bool)
            valid_mask[seen_positions] = False
            valid_positions = np.flatnonzero(valid_mask)
            if len(valid_positions) > n_candidates:
                valid_scores = scores_u[valid_positions]
                cutoff = np.partition(valid_scores, -n_candidates)[-n_candidates]
                above = valid_positions[scores_u[valid_positions] > cutoff]
                tied = valid_positions[scores_u[valid_positions] == cutoff]
                needed = n_candidates - len(above)
                tie_order = np.lexsort((item_ids[tied], pop_rank_values[tied]))
                selected = np.concatenate((above, tied[tie_order[:needed]]))
            else:
                selected = valid_positions

            order = np.lexsort((item_ids[selected], pop_rank_values[selected], -scores_u[selected]))
            top_positions = selected[order]
            batch_rows["userId"].extend([str(u)] * len(top_positions))
            batch_rows["itemId"].extend(item_ids[top_positions].tolist())
            batch_rows["score"].extend(scores_u[top_positions].astype(np.float32).tolist())
            batch_rows["rank"].extend(range(len(top_positions)))

        table = pa.Table.from_pydict(batch_rows, schema=output_schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, output_schema)
        writer.write_table(table)
        completed = min(start_idx + len(batch_users), len(eval_users))
        if completed - last_progress >= 10_000 or completed == len(eval_users):
            elapsed = max(time.monotonic() - candidates_started, 1e-9)
            logger.info(
                "candidate generation: %s/%s users (%.1f%%), %.0f users/s, elapsed %.1f min",
                f"{completed:,}", f"{len(eval_users):,}",
                100 * completed / max(len(eval_users), 1), completed / elapsed, elapsed / 60,
            )
            last_progress = completed

    if writer is None:
        writer = pq.ParquetWriter(output_path, output_schema)
    writer.close()
    baseline_scores_path = output_path
    logger.info("Candidate generation complete in %.1f min", (time.monotonic() - candidates_started) / 60)
    logger.info(
        "Saved baseline candidates to %s for %s users",
        baseline_scores_path,
        f"{len(eval_users):,}",
    )

    # 7. Write run_manifest.json
    mean_std = float(np.mean(user_score_stds)) if user_score_stds else 0.0
    manifest = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "seed": seed,
        "python": sys.version.split()[0],
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "implicit": implicit.__version__,
            "scipy": scipy.__version__,
        },
        "params": {
            "factors": factors,
            "iterations": iterations,
            "regularization": regularization,
            "alpha": alpha,
            "n_candidates": n_candidates,
            "k": 10,
            "min_rating": 4.0,
            "min_catalogue_interactions": 3,
            "min_train_positives_exposure": 1,
            "min_train_positives_relevance": 2,
            "n_eval_users": n_eval_users,
        },
        "counts": {
            "n_users": n_users,
            "n_items": n_items,
            "n_train": len(train_ratings),
            "n_test": int((ratings_df["split"] == "test").sum()),
            "n_eval_exposure": len(sampled_exposure),
            "n_eval_relevance": len(sampled_relevance),
        },
        "score_stats": {
            "mean_per_user_score_std": mean_std,
        },
    }
    manifest_path = os.path.join(results_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved run manifest to %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="02_train: Train implicit ALS and generate baseline candidates.")
    parser.add_argument("--ratings-path", type=str, default="data/processed/ratings.parquet")
    parser.add_argument("--items-path", type=str, default="data/processed/items.parquet")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--factors", type=int, default=DEFAULT_FACTORS)
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--n-eval-users", type=int, default=300_000)
    args = parser.parse_args()

    train_and_generate_candidates(
        ratings_path=args.ratings_path,
        items_path=args.items_path,
        results_dir=args.results_dir,
        factors=args.factors,
        regularization=args.regularization,
        alpha=args.alpha,
        iterations=args.iterations,
        seed=args.seed,
        n_candidates=args.n_candidates,
        n_eval_users=args.n_eval_users,
    )
