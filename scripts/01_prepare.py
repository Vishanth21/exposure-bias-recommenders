"""Stage 01 — build ratings/items parquet and the prepare funnel.

Writes, per ``report/INTERFACE.md`` §1 and §3:

* ``data/processed/ratings.parquet`` — one row per review, tagged ``train`` / ``test`` /
  ``seen``.
* ``data/processed/items.parquet`` — one row per item, with both bucket rules.
* ``results/prepare_funnel.json`` — the counts and bucket shares every later stage reads.
"""

from __future__ import annotations

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

from amazon_loader import load_metadata, load_reviews  # noqa: E402

log = logging.getLogger("prepare")

PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "results"

POSITIVE_THRESHOLD = 4.0
CATALOGUE_MIN_TRAIN = 3
N_EVAL_USERS = 300_000

POP_BUCKETS = ("head", "mid", "tail")
# (bucket, upper edge of the cumulative interaction mass); last edge closes at 1.0.
POP_MASS_EDGES = (("head", 0.50), ("mid", 0.80), ("tail", 1.00))
STORE_MASS_EDGE = 0.80

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLIT_SEEN = "seen"


# --------------------------------------------------------------------------- split


def assign_splits(reviews: pd.DataFrame) -> pd.DataFrame:
    """Tag each row ``train`` / ``test`` / ``seen``.

    Only positives enter the temporal split: the user's last positive by
    ``(timestamp, itemId)`` is ``test``, their earlier positives are ``train``, and every
    non-positive is ``seen``. A non-positive never becomes ground truth or training signal.
    """
    df = reviews.copy()
    df["_positive"] = df["rating"] >= POSITIVE_THRESHOLD
    df["split"] = SPLIT_SEEN

    pos = df[df["_positive"]]
    if len(pos):
        # Sort key order defines "last": timestamp first, itemId settles equal timestamps.
        ordered = pos.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")
        last_idx = ordered.groupby("user_id", sort=False).tail(1).index
        df.loc[last_idx, "split"] = SPLIT_TEST
        df.loc[pos.index.difference(last_idx), "split"] = SPLIT_TRAIN

    df["seen"] = True
    return df.drop(columns=["_positive"])


# --------------------------------------------------------------------------- buckets


def assign_pop_buckets(n_train: pd.Series) -> pd.Series:
    """Bucket catalogue items by cumulative interaction mass.

    ``n_train`` is indexed by ``itemId`` and holds train counts for catalogue items only.
    Items are taken in descending count (``itemId`` ascending breaks ties) and assigned to
    the bucket whose cumulative mass interval they enter, so a boundary-straddling item
    lands in the bucket it enters rather than being split.

    A percentile cut is not usable here: counts tie heavily at 1–2, so the cut would land
    inside a tie block and be decided by ``itemId``.
    """
    counts = n_train.sort_index()
    counts = counts.sort_values(ascending=False, kind="mergesort")

    total = float(counts.sum())
    if total <= 0.0:
        return pd.Series("tail", index=counts.index, dtype=object)

    # Mass *after* including each item, so the item crossing an edge is inside that bucket.
    cum_share = counts.cumsum() / total

    labels = np.empty(len(counts), dtype=object)
    prev_edge = 0.0
    position = 0
    values = cum_share.to_numpy()
    for bucket, edge in POP_MASS_EDGES:
        # Items whose cumulative share sits in (prev_edge, edge].
        in_bucket = (values > prev_edge) & (values <= edge)
        labels[position : position + int(in_bucket.sum())] = bucket
        position += int(in_bucket.sum())
        prev_edge = edge
    # Float residue can leave the last item one ulp past 1.0.
    labels[position:] = POP_MASS_EDGES[-1][0]

    return pd.Series(labels, index=counts.index, dtype=object)


def assign_store_buckets(store_n_items: pd.Series) -> pd.Series:
    """Bucket stores by cumulative catalogue footprint.

    ``store_n_items`` is indexed by ``store`` (non-empty). Stores are ranked by catalogue
    item count descending (``store`` ascending breaks ties) and accumulate until 80% of
    catalogue items is covered: those stores are ``large``, the remainder ``small``. A store
    crossing the boundary is placed in the bucket it enters.

    The number of stores in ``large`` is therefore a result, not a parameter; the funnel
    reports it because a single dominant store degenerates the comparison.
    """
    sizes = store_n_items.sort_index()
    sizes = sizes.sort_values(ascending=False, kind="mergesort")

    total = float(sizes.sum())
    if total <= 0.0:
        return pd.Series("small", index=sizes.index, dtype=object)

    cum_items = sizes.cumsum().to_numpy()
    boundary = STORE_MASS_EDGE * total
    # First index whose cumulative coverage reaches the boundary; that store is inside large.
    n_large = int(np.searchsorted(cum_items, boundary, side="left")) + 1
    n_large = min(n_large, len(sizes))

    labels = np.empty(len(sizes), dtype=object)
    labels[:n_large] = "large"
    labels[n_large:] = "small"
    return pd.Series(labels, index=sizes.index, dtype=object)


def _bucket_mass_table(items: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Item share and interaction share per popularity bucket, catalogue items only."""
    cat = items[items["in_catalogue"]]
    n_items = len(cat)
    total_interactions = float(cat["n_train_ratings"].sum())

    out: dict[str, dict[str, float]] = {}
    for bucket in POP_BUCKETS:
        rows = cat[cat["pop_bucket"] == bucket]
        out[bucket] = {
            "n_items": int(len(rows)),
            "item_share": (len(rows) / n_items) if n_items else 0.0,
            "interaction_share": (
                float(rows["n_train_ratings"].sum()) / total_interactions
                if total_interactions
                else 0.0
            ),
        }
    return out


def _store_bucket_table(items: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Store count, store share and catalogue item share per seller bucket."""
    cat = items[items["in_catalogue"]]
    total_items = len(cat)

    # "" is a real bucket (unknown) and one store for counting purposes.
    stores = cat["store"].fillna("")
    per_store = stores.groupby(stores).size()
    bucket_of_store = cat.groupby(stores)["store_bucket"].first()

    out: dict[str, dict[str, float]] = {}
    for bucket in ("large", "small", "unknown"):
        store_ids = bucket_of_store.index[bucket_of_store == bucket]
        n_stores = int(len(store_ids))
        n_items = int(per_store.reindex(store_ids).sum()) if n_stores else 0
        out[bucket] = {
            "n_stores": n_stores,
            "store_share": (n_stores / len(per_store)) if len(per_store) else 0.0,
            "item_share": (n_items / total_items) if total_items else 0.0,
        }
    return out


# --------------------------------------------------------------------------- build


def build_items(
    reviews: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(ratings, items)``.

    ``ratings`` is the tagged review table in the frozen column order. ``items`` carries one
    row per ``parent_asin`` seen in the reviews, joined to metadata.
    """
    train = reviews[reviews["split"] == SPLIT_TRAIN]
    n_train_ratings = train.groupby("parent_asin").size()

    items = meta.drop_duplicates(subset=["parent_asin"]).set_index("parent_asin")
    review_ids = pd.Index(reviews["parent_asin"].unique(), name="parent_asin")

    # Review ids missing from metadata keep a row: dropping them would silently remove
    # interactions from the popularity and seller counts.
    items = items.reindex(review_ids)
    n_missing_meta = int(items["title"].isna().sum())
    if n_missing_meta:
        log.warning("%d items have no metadata row; kept with empty fields", n_missing_meta)

    items["title"] = items["title"].fillna("")
    items["store"] = items["store"].fillna("")
    items["main_category"] = items["main_category"].fillna("")
    items["price"] = items["price"].astype("float32")

    items["n_train_ratings"] = (
        n_train_ratings.reindex(items.index).fillna(0).astype("int32")
    )
    items["in_catalogue"] = items["n_train_ratings"] >= CATALOGUE_MIN_TRAIN

    catalogue = items[items["in_catalogue"]]
    # rank() is 1-based; the contract has 0 = most popular. Only relative order is used downstream.
    ranks = catalogue["n_train_ratings"].rank(method="first", ascending=False) - 1
    # Unranked (non-catalogue) items get -1 so the column stays int32 without a nullable dtype.
    items["popularity_rank"] = ranks.reindex(items.index).fillna(-1).astype("int32")

    pop_bucket = assign_pop_buckets(catalogue["n_train_ratings"])
    items["pop_bucket"] = pop_bucket.reindex(items.index).fillna("tail")

    cat = items[items["in_catalogue"]]
    known = cat[cat["store"] != ""]
    store_n_items = known.groupby("store").size()
    store_n_train = known.groupby("store")["n_train_ratings"].sum()
    store_bucket = assign_store_buckets(store_n_items)

    items["store_n_items"] = (
        items["store"].map(store_n_items).fillna(0).astype("int32")
    )
    items["store_n_train"] = (
        items["store"].map(store_n_train).fillna(0).astype("int32")
    )
    # An empty store is not a small store: it is reported on its own line everywhere.
    items["store_bucket"] = pd.Series(
        np.where(
            items["store"] == "",
            "unknown",
            items["store"].map(store_bucket).fillna("small"),
        ),
        index=items.index,
        dtype=object,
    )

    items = items.reset_index()[
        [
            "parent_asin",
            "title",
            "store",
            "main_category",
            "price",
            "n_train_ratings",
            "popularity_rank",
            "pop_bucket",
            "store_n_items",
            "store_n_train",
            "store_bucket",
            "in_catalogue",
        ]
    ].rename(columns={"parent_asin": "itemId"})

    ratings = reviews.rename(columns={"user_id": "userId", "parent_asin": "itemId"})[
        ["userId", "itemId", "rating", "timestamp", "split", "seen"]
    ].copy()
    ratings["rating"] = ratings["rating"].astype("float32")
    ratings["timestamp"] = ratings["timestamp"].astype("int64")
    ratings["seen"] = ratings["seen"].astype(bool)

    return ratings, items


def build_funnel(
    reviews: pd.DataFrame,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
) -> dict:
    """Stage counts and shares, in the order the pipeline removes information."""
    positives = ratings[ratings["split"] != SPLIT_SEEN]
    non_positives = ratings[ratings["split"] == SPLIT_SEEN]
    train = ratings[ratings["split"] == SPLIT_TRAIN]
    test = ratings[ratings["split"] == SPLIT_TEST]

    catalogue = items[items["in_catalogue"]]
    test_items = set(test["itemId"])
    out_of_catalogue = len(test_items - set(catalogue["itemId"]))

    train_positives = train.groupby("userId").size()
    exposure_users = train_positives.index[train_positives >= 1]
    relevance_users = train_positives.index[train_positives >= 2]

    def stage(name: str, frame: pd.DataFrame, user_col: str, item_col: str) -> dict:
        return {
            "stage": name,
            "users": int(frame[user_col].nunique()),
            "items": int(frame[item_col].nunique()),
            "interactions": int(len(frame)),
        }

    stages = [
        stage("raw_reviews", reviews, "user_id", "parent_asin"),
        stage("positives_ge_4", positives, "userId", "itemId"),
        stage("non_positives_seen", non_positives, "userId", "itemId"),
        stage("after_split", ratings, "userId", "itemId"),
        stage("catalogue_ge_3", catalogue, "itemId", "itemId"),
        stage(
            "eval_exposure_pop",
            train[train["userId"].isin(exposure_users)],
            "userId",
            "itemId",
        ),
        stage(
            "eval_relevance_pop",
            train[train["userId"].isin(relevance_users)],
            "userId",
            "itemId",
        ),
    ]
    for row in stages[-2:]:
        row["sampled_users"] = min(row["users"], N_EVAL_USERS)
    # catalogue_ge_3 has no user column of its own; users are those with a train row on one.
    catalogue_users = train[train["itemId"].isin(catalogue["itemId"])]["userId"].nunique()
    stages[4]["users"] = int(catalogue_users)

    return {
        "category": None,  # filled by main(), which knows the requested category
        "stages": stages,
        "positive_fraction": float(
            (reviews["rating"] >= POSITIVE_THRESHOLD).mean() if len(reviews) else 0.0
        ),
        "store_null_rate": float(
            (items["store"] == "").mean() if len(items) else 0.0
        ),
        "test_items_outside_catalogue_frac": float(
            out_of_catalogue / len(test_items) if test_items else 0.0
        ),
        "buckets": _bucket_mass_table(items),
        "store_buckets": _store_bucket_table(items),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Electronics")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="truncate reviews/metadata before filtering, for a smoke run",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    run_started = time.monotonic()
    log.info(
        "prepare started: category=%s review_limit=%s metadata_limit=%s",
        args.category,
        args.limit if args.limit is not None else "all",
        args.limit if args.limit is not None else "all",
    )

    stage_started = time.monotonic()
    reviews = load_reviews(category=args.category, limit=args.limit)
    log.info(
        "reviews ready: %s rows, %s users, %s items (%.1f min)",
        f"{len(reviews):,}",
        f"{reviews['user_id'].nunique():,}",
        f"{reviews['parent_asin'].nunique():,}",
        (time.monotonic() - stage_started) / 60,
    )

    stage_started = time.monotonic()
    meta = load_metadata(category=args.category, limit=args.limit)
    log.info(
        "metadata ready: %s rows, %s distinct items (%.1f min)",
        f"{len(meta):,}",
        f"{meta['parent_asin'].nunique():,}",
        (time.monotonic() - stage_started) / 60,
    )

    # Ratings are the observed rating distribution: the >= 4 threshold is only informative
    # if it is not near-vacuous, and the card does not state the value set.
    log.info(
        "rating distribution: %s",
        reviews["rating"].value_counts().sort_index().to_dict(),
    )

    stage_started = time.monotonic()
    reviews = assign_splits(reviews)
    log.info(
        "temporal split ready: train=%s test=%s seen=%s (%.1f min)",
        f"{int((reviews['split'] == SPLIT_TRAIN).sum()):,}",
        f"{int((reviews['split'] == SPLIT_TEST).sum()):,}",
        f"{int((reviews['split'] == SPLIT_SEEN).sum()):,}",
        (time.monotonic() - stage_started) / 60,
    )

    stage_started = time.monotonic()
    ratings, items = build_items(reviews, meta)
    log.info(
        "tables built: ratings=%s rows, items=%s rows, catalogue=%s (%.1f min)",
        f"{len(ratings):,}",
        f"{len(items):,}",
        f"{int(items['in_catalogue'].sum()):,}",
        (time.monotonic() - stage_started) / 60,
    )

    stage_started = time.monotonic()
    ratings.to_parquet(PROCESSED / "ratings.parquet", index=False)
    items.to_parquet(PROCESSED / "items.parquet", index=False)
    log.info("parquet files written (%.1f min)", (time.monotonic() - stage_started) / 60)

    stage_started = time.monotonic()
    funnel = build_funnel(reviews, ratings, items)
    funnel["category"] = args.category
    (RESULTS / "prepare_funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")
    log.info("prepare funnel written (%.1f min)", (time.monotonic() - stage_started) / 60)

    large = funnel["store_buckets"]["large"]
    log.info(
        "catalogue %d items; large stores %d (%.3f of stores, %.3f of items)",
        int(items["in_catalogue"].sum()),
        large["n_stores"],
        large["store_share"],
        large["item_share"],
    )
    log.info("prepare complete in %.1f min", (time.monotonic() - run_started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
