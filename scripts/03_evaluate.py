"""Stage 03 — score ALS and reference recommenders for the evaluation populations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
RESULTS = REPO_ROOT / "results"

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


# --------------------------------------------------------------------------- candidate sets


def candidate_sets(
    users: pd.Index,
    catalogue: pd.Index,
    seen: dict[str, set[str]],
) -> dict[str, np.ndarray]:
    """Catalogue items minus each user's rated items, as a positional index array.

    One array per user. Every item the user rated is excluded, low ratings included: a rated
    item is observed, not a recommendation opportunity.
    """
    catalogue_arr = catalogue.to_numpy()
    catalogue_set = set(catalogue_arr)
    out: dict[str, np.ndarray] = {}
    for user in users:
        excluded = seen.get(user, set()) & catalogue_set
        out[user] = catalogue_arr[~np.isin(catalogue_arr, list(excluded))] if excluded else catalogue_arr
    return out


def top_n(ordered_items: np.ndarray, ordered_scores: np.ndarray, n: int) -> list[int]:
    """Top ``n`` item positions, ties broken by item position ascending for determinism."""
    if len(ordered_items) <= n:
        return ordered_items.tolist()
    return ordered_items[:n].tolist()


# --------------------------------------------------------------------------- recommenders


def als_recommendations(
    candidates: dict[str, np.ndarray],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_index: dict[str, int],
    item_index: dict[str, int],
    catalogue: pd.Index,
    n: int = N_CANDIDATES,
) -> dict[str, list[int]]:
    """Full-ranking ALS: score every catalogue item, take the top ``n``.

    Only the candidate set is scored, so the result is a ranking over what the user could
    still be shown. No negatives are sampled, which keeps NDCG unconditional on a sampled
    subset.
    """
    out: dict[str, list[int]] = {}
    for user, cand in candidates.items():
        if not len(cand):
            out[user] = []
            continue
        cand_pos = np.array([item_index[i] for i in cand])
        scores = item_factors[cand_pos] @ user_factors[user_index[user]]
        order = np.argsort(-scores, kind="stable")
        out[user] = top_n(cand[order], scores[order], n)
    return out


def mostpop_recommendations(
    candidates: dict[str, np.ndarray],
    popularity_rank: dict[str, int],
    n: int = N_CANDIDATES,
) -> dict[str, list[int]]:
    """Most popular catalogue items the user has not rated.

    Ranked by ``popularity_rank`` so ties at low interaction counts are broken by the same
    rule the popularity buckets use.
    """
    out: dict[str, list[int]] = {}
    for user, cand in candidates.items():
        order = sorted(cand.tolist(), key=lambda i: (popularity_rank[i], i))
        out[user] = order[:n]
    return out


def random_recommendations(
    candidates: dict[str, np.ndarray], n: int = N_CANDIDATES, seed: int = SEED
) -> dict[str, list[int]]:
    """Uniform sample of the candidate set, drawn from one seeded generator.

    Users are iterated in a fixed order so the draw does not depend on dict construction.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {}
    for user in sorted(candidates):
        cand = candidates[user]
        if len(cand) <= n:
            out[user] = cand.tolist()
        else:
            out[user] = cand[rng.choice(len(cand), size=n, replace=False)].tolist()
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
    per_user_ndcg = {
        u: ndcg_at_k(recs.get(u, []), relevant.get(u, set()), k) for u in users
    }
    per_user_hit = {
        u: hitrate_at_k(recs.get(u, []), relevant.get(u, set()), k) for u in users
    }

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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    RESULTS.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_parquet(PROCESSED / "ratings.parquet")
    items = pd.read_parquet(PROCESSED / "items.parquet")
    factors = np.load(RESULTS / "model" / "als_factors.npz")
    baseline_scores = pd.read_parquet(RESULTS / "baseline_scores.parquet")

    train = ratings[ratings["split"] == "train"]
    test = ratings[ratings["split"] == "test"]
    catalogue = pd.Index(items.loc[items["in_catalogue"], "itemId"])
    catalog_size = len(catalogue)

    pop_bucket_of = dict(zip(items["itemId"], items["pop_bucket"]))
    store_bucket_of = dict(zip(items["itemId"], items["store_bucket"]))
    popularity_of = dict(zip(items["itemId"], items["n_train_ratings"].astype(float)))
    popularity_rank = dict(zip(items["itemId"], items["popularity_rank"]))
    item_index = {item: i for i, item in enumerate(items["itemId"])}

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

    user_index = {u: i for i, u in enumerate(_user_order(baseline_scores, factors))}

    populations = build_populations(train)

    result: dict = {
        "k": args.k,
        "n_candidates": args.n_candidates,
        "catalog_size": catalog_size,
    }
    per_user_rows: list[pd.DataFrame] = []

    for pop_name, users in populations.items():
        log.info("scoring %s (%d users)", pop_name, len(users))
        candidates = candidate_sets(users, catalogue, seen)
        with_relevance = pop_name == "relevance_pop"

        model_recs: dict[str, dict[str, list[int]]] = {
            "als": als_recommendations(
                candidates, factors["user_factors"], factors["item_factors"],
                user_index, item_index, catalogue, args.n_candidates,
            ),
            "mostpop": mostpop_recommendations(
                candidates, popularity_rank, args.n_candidates
            ),
            "random": random_recommendations(
                candidates, args.n_candidates
            ),
        }

        blocks: dict[str, dict] = {}
        for model_name, recs in model_recs.items():
            scored = score_population(
                users, recs, relevant, pop_bucket_of, store_bucket_of, popularity_of,
                item_share, catalogue, with_relevance, args.k,
            )
            per_user = scored.pop("_per_user")
            blocks[model_name] = scored
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
    per_user_frame.to_parquet(RESULTS / "per_user_metrics.parquet", index=False)
    log.info("wrote %s (%d rows)", RESULTS / "per_user_metrics.parquet", len(per_user_frame))

    (RESULTS / "baseline_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("wrote %s", RESULTS / "baseline_metrics.json")
    return 0


def _user_order(baseline_scores: pd.DataFrame, factors) -> list[str]:
    """ALS user row order, as recorded by 02_train.py.

    ``user_factors`` rows are positional; the training stage writes the order it used to
    ``results/model/user_order.json``.
    """
    path = RESULTS / "model" / "user_order.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; 02_train.py writes it alongside als_factors.npz"
        )
    return json.loads(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
