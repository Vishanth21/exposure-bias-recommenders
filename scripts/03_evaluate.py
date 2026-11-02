"""Stage 03 — score ALS and reference recommenders for the evaluation populations."""

from __future__ import annotations

from bisect import bisect_right
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metrics import (  # noqa: E402
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

log = logging.getLogger("evaluate")

PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_RESULTS = REPO_ROOT / "results"

K = 10
N_CANDIDATES = 200
N_EVAL_USERS = 300_000
SEED = 42

POP_BUCKETS = ("head", "mid", "tail")
STORE_BUCKETS = ("small", "large", "unknown")


# --------------------------------------------------------------------------- populations


def build_populations(train: pd.DataFrame) -> dict[str, pd.Index]:
    """Sample the two evaluation populations from train-positive counts.

    Sorting before the draw makes the sample independent of file order; the same seed then
    reproduces it across runs. The cap keeps candidate generation and the six-lambda sweep
    finite.
    """
    counts = train.groupby("userId").size()
    rng = np.random.default_rng(SEED)

    out: dict[str, pd.Index] = {}
    for name, minimum in (("exposure_pop", 1), ("relevance_pop", 2)):
        eligible = np.sort(counts.index[counts >= minimum].to_numpy())
        if len(eligible) > N_EVAL_USERS:
            picked = rng.choice(eligible, size=N_EVAL_USERS, replace=False)
            eligible = np.sort(picked)
        out[name] = pd.Index(eligible, name="userId")
        log.info(
            "%s: %d eligible, %d sampled", name, int((counts >= minimum).sum()), len(eligible)
        )
    return out


def _catalogue_index_for_unseen_rank(rank: int, excluded_positions: list[int]) -> int:
    """Map a rank in the unseen catalogue back to the full catalogue index."""
    low = rank
    high = rank + len(excluded_positions)
    while low < high:
        middle = (low + high) // 2
        available_through_middle = middle + 1 - bisect_right(excluded_positions, middle)
        if available_through_middle <= rank:
            low = middle + 1
        else:
            high = middle
    return low


def _random_unseen_items(
    catalogue: np.ndarray,
    excluded_positions: list[int],
    rng: np.random.Generator,
    candidate_count: int,
    k: int,
) -> list[str]:
    """Sample a random ordered prefix without scanning the catalogue per user."""
    available_count = len(catalogue) - len(excluded_positions)
    if available_count <= candidate_count:
        excluded = set(excluded_positions)
        return [
            item
            for position, item in enumerate(catalogue)
            if position not in excluded
        ][:k]

    unseen_ranks = rng.choice(available_count, size=candidate_count, replace=False)
    catalogue_indices = [
        _catalogue_index_for_unseen_rank(int(rank), excluded_positions)
        for rank in unseen_ranks[:k]
    ]
    return catalogue[catalogue_indices].tolist()


# --------------------------------------------------------------------------- recommendation construction


def build_recommendations(
    users: pd.Index,
    catalogue: pd.Index,
    seen: dict[str, set[str]],
    als_recs: dict[str, list[str]],
    popularity_rank: dict[str, int],
    candidate_count: int,
    k: int,
    seed: int = SEED,
) -> dict[str, dict[str, list[str]]]:
    """Keep only each user's metric prefix; the larger candidate pool is temporary."""
    catalogue_arr = catalogue.to_numpy()
    catalogue_position = {item: position for position, item in enumerate(catalogue_arr)}
    mostpop_order = sorted(catalogue_arr.tolist(), key=lambda item: (popularity_rank[item], item))
    rng = np.random.default_rng(seed)
    out = {"als": {}, "mostpop": {}, "random": {}}
    started_at = time.monotonic()
    last_progress = 0

    for position, user in enumerate(users, start=1):
        known = seen.get(user, set())
        excluded_positions = sorted(
            catalogue_position[item] for item in known if item in catalogue_position
        )
        out["als"][user] = als_recs.get(user, [])[:k]

        best = []
        for item in mostpop_order:
            if item not in known:
                best.append(item)
                if len(best) == k:
                    break
        out["mostpop"][user] = best

        out["random"][user] = _random_unseen_items(
            catalogue_arr, excluded_positions, rng, candidate_count, k
        )

        if position - last_progress >= 10_000 or position == len(users):
            elapsed = max(time.monotonic() - started_at, 1e-9)
            log.info(
                "recommendations: %s/%s users (%.1f%%), %.0f users/s, elapsed %.1f min",
                f"{position:,}", f"{len(users):,}",
                100 * position / max(len(users), 1), position / elapsed, elapsed / 60,
            )
            last_progress = position
    return out


# --------------------------------------------------------------------------- scoring


def score_population(
    users: pd.Index,
    recs: dict[str, list[int]],
    relevant: dict[str, set[str]],
    pop_bucket_of: dict[str, str],
    store_bucket_of: dict[str, str],
    popularity_of: dict[str, float],
    item_share: dict[str, float],
    catalogue: pd.Index,
    with_relevance: bool,
    k: int = K,
) -> dict:
    """Aggregate one recommender over one population.

    ``with_relevance`` is False on the exposure population: those users have no held-out
    positive, so NDCG and hitrate would be 0.0 by construction rather than by measurement.
    """
    rec_lists = [recs[u][:k] for u in users if recs.get(u)]
    per_user_ndcg = {}
    per_user_hit = {}
    metrics_started = time.monotonic()
    last_progress = 0
    for position, user in enumerate(users, start=1):
        user_relevant = relevant.get(user, set())
        user_recs = recs.get(user, [])
        per_user_ndcg[user] = ndcg_at_k(user_recs, user_relevant, k)
        per_user_hit[user] = hitrate_at_k(user_recs, user_relevant, k)
        if position - last_progress >= 10_000 or position == len(users):
            elapsed = max(time.monotonic() - metrics_started, 1e-9)
            log.info(
                "per-user metrics: %s/%s users (%.1f%%), %.0f users/s, elapsed %.1f min",
                f"{position:,}", f"{len(users):,}",
                100 * position / max(len(users), 1), position / elapsed, elapsed / 60,
            )
            last_progress = position

    exposure = exposure_share(rec_lists, pop_bucket_of, buckets=POP_BUCKETS)
    out: dict = {"n_users": int(len(users))}

    if with_relevance:
        out["ndcg"] = float(np.mean(list(per_user_ndcg.values()))) if len(users) else 0.0
        out["hitrate"] = float(np.mean(list(per_user_hit.values()))) if len(users) else 0.0

    out["exposure_share"] = exposure
    out["store_exposure_share"] = store_exposure_share(rec_lists, store_bucket_of)
    out["exposure_share_by_pop"] = exposure_by_cross(
        rec_lists, pop_bucket_of, store_bucket_of, POP_BUCKETS, STORE_BUCKETS
    )
    out["exposure_parity"] = exposure_parity(exposure, item_share, POP_BUCKETS)
    out["coverage"] = catalog_coverage(rec_lists, len(catalogue))
    out["gini_popularity"] = gini(
        # Exposure counts per catalogue item, including never-recommended ones.
        _exposure_counts(rec_lists, catalogue)
    )
    out["mean_recommended_popularity"] = mean_recommended_popularity(
        rec_lists, popularity_of
    )

    out["_per_user"] = {"ndcg": per_user_ndcg, "hit": per_user_hit}
    return out


def _exposure_counts(rec_lists: list[list[int]], catalogue: pd.Index) -> list[int]:
    """Slot count per catalogue item, zeros included.

    Gini over recommended items alone would measure concentration among winners; the
    never-recommended items are most of the inequality. The universe is the catalogue, not
    the items that carry a bucket label: every catalogue item has one.
    """
    counts = dict.fromkeys(catalogue, 0)
    for recs in rec_lists:
        for item in recs:
            if item in counts:
                counts[item] += 1
    return list(counts.values())


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    parser.add_argument("--ratings-path", type=Path, default=None)
    parser.add_argument("--items-path", type=Path, default=None)
    parser.add_argument("--baseline-scores", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.monotonic()
    stage_started = time.monotonic()
    ratings_path = args.ratings_path or (args.data_dir / "ratings.parquet")
    items_path = args.items_path or (args.data_dir / "items.parquet")
    ratings = pd.read_parquet(ratings_path)
    items = pd.read_parquet(items_path)
    log.info(
        "loaded ratings=%s and items=%s in %.1f min",
        f"{len(ratings):,}", f"{len(items):,}", (time.monotonic() - stage_started) / 60,
    )

    train = ratings[ratings["split"] == "train"]
    test = ratings[ratings["split"] == "test"]
    catalogue = pd.Index(items.loc[items["in_catalogue"], "itemId"])
    catalog_size = len(catalogue)

    pop_bucket_of = dict(zip(items["itemId"], items["pop_bucket"]))
    store_bucket_of = dict(zip(items["itemId"], items["store_bucket"]))
    popularity_of = dict(zip(items["itemId"], items["n_train_ratings"].astype(float)))
    popularity_rank = dict(zip(items["itemId"], items["popularity_rank"]))

    cat = items[items["in_catalogue"]]
    item_share = {
        b: float(len(cat[cat["pop_bucket"] == b]) / catalog_size) if catalog_size else 0.0
        for b in POP_BUCKETS
    }

    # The held-out positive is unknown at ranking time and must remain recommendable.
    # Training positives and non-positive observations are already known and excluded.
    observed_before_test = ratings[ratings["split"] != "test"]
    seen: dict[str, set[str]] = observed_before_test.groupby("userId")["itemId"].agg(set).to_dict()
    relevant = test.groupby("userId")["itemId"].agg(set).to_dict()

    populations = build_populations(train)
    stage_started = time.monotonic()
    baseline_path = args.baseline_scores or (results_dir / "baseline_scores.parquet")
    als_top = pd.read_parquet(
        baseline_path,
        columns=["userId", "itemId", "rank"],
        filters=[("rank", "<", args.k)],
    ).sort_values(["userId", "rank"])
    als_recs = als_top.groupby("userId")["itemId"].apply(list).to_dict()
    log.info(
        "loaded top-%d ALS candidates for %s users in %.1f min",
        args.k, f"{len(als_recs):,}", (time.monotonic() - stage_started) / 60,
    )

    result: dict = {
        "k": args.k,
        "n_candidates": args.n_candidates,
        "catalog_size": catalog_size,
    }
    per_user_rows: list[pd.DataFrame] = []

    for pop_name, users in populations.items():
        log.info("scoring %s (%d users)", pop_name, len(users))
        stage_started = time.monotonic()
        model_recs = build_recommendations(
            users, catalogue, seen, als_recs, popularity_rank,
            args.n_candidates, args.k,
        )
        log.info("%s recommendations ready in %.1f min", pop_name, (time.monotonic() - stage_started) / 60)
        with_relevance = pop_name == "relevance_pop"

        blocks: dict[str, dict] = {}
        for model_name, recs in model_recs.items():
            stage_started = time.monotonic()
            scored = score_population(
                users, recs, relevant, pop_bucket_of, store_bucket_of, popularity_of,
                item_share, catalogue, with_relevance, args.k,
            )
            per_user = scored.pop("_per_user")
            blocks[model_name] = scored
            log.info(
                "%s/%s metrics complete in %.1f min",
                pop_name, model_name, (time.monotonic() - stage_started) / 60,
            )
            # One row per user; only the ALS rows belong in the baseline per-user table,
            # since the baselines are not the sweep's subject.
            if model_name == "als":
                per_user_rows.append(
                    pd.DataFrame(
                        {
                            "userId": list(per_user["ndcg"].keys()),
                            "population": pop_name,
                            "lambda": np.nan,  # filled by 04_rerank.py
                            "ndcg": list(per_user["ndcg"].values()),
                            "hit": list(per_user["hit"].values()),
                        }
                    )
                )

        als_block = blocks["als"]
        # baseline_metrics.json keeps the ALS block flat and nests the baselines, per §3.
        result[pop_name] = als_block
        for name, block in blocks.items():
            if name == "als":
                continue
            # Exposure needs no ground truth, so a baseline's exposure-pop ndcg is null by
            # design rather than a zero that reads as measured.
            result.setdefault("baselines", {}).setdefault(name, {})[pop_name] = {
                "ndcg": block.get("ndcg"),
                "hitrate": block.get("hitrate"),
                "exposure_share": block["exposure_share"],
            }

    per_user_frame = pd.concat(per_user_rows, ignore_index=True)
    per_user_frame.to_parquet(results_dir / "per_user_metrics.parquet", index=False)
    log.info("wrote %s (%d rows)", results_dir / "per_user_metrics.parquet", len(per_user_frame))

    (results_dir / "baseline_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("wrote %s", results_dir / "baseline_metrics.json")
    log.info("evaluation complete in %.1f min", (time.monotonic() - run_started) / 60)
    return 0




def evaluate_baseline_models(
    baseline_scores_path: str | Path = "results/baseline_scores.parquet",
    ratings_path: str | Path = "data/processed/ratings.parquet",
    items_path: str | Path = "data/processed/items.parquet",
    results_dir: str | Path = "results",
    k: int = K,
    n_candidates: int = N_CANDIDATES,
) -> dict:
    argv = [
        "--k", str(k),
        "--n-candidates", str(n_candidates),
        "--results-dir", str(results_dir),
        "--baseline-scores", str(baseline_scores_path),
        "--ratings-path", str(ratings_path),
        "--items-path", str(items_path),
    ]
    main(argv)
    output_path = Path(results_dir) / "baseline_metrics.json"
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    raise SystemExit(main())
