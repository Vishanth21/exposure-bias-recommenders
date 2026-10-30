"""
01_prepare.py: Data preparation stage for Amazon Reviews 2023.

Transforms raw reviews and metadata into contract-compliant artefacts:
- data/processed/ratings.parquet
- data/processed/items.parquet
- results/prepare_funnel.json

Conforms to report/INTERFACE.md §1 & §3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.amazon_loader import extract_store, load_metadata, load_reviews

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(
    category: str = "Electronics",
    output_data_dir: str = "data/processed",
    output_results_dir: str = "results",
    limit: int | None = None,
) -> None:
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_results_dir, exist_ok=True)

    logger.info("Loading reviews and metadata for category: %s", category)
    reviews_df = load_reviews(category=category, limit=limit)
    meta_df = load_metadata(category=category, limit=limit)

    raw_users_cnt = int(reviews_df["user_id"].nunique())
    raw_items_cnt = int(reviews_df["parent_asin"].nunique())
    raw_interactions_cnt = int(len(reviews_df))

    # Match reviews with metadata: drop review rows whose parent_asin has no metadata row
    valid_meta_items = set(meta_df["parent_asin"].unique())
    before_drop = len(reviews_df)
    reviews_df = reviews_df[reviews_df["parent_asin"].isin(valid_meta_items)].copy()
    dropped_no_meta = before_drop - len(reviews_df)
    if dropped_no_meta > 0:
        logger.warning("Dropped %d review rows with missing item metadata.", dropped_no_meta)

    # Threshold for positive interactions
    positive_mask = reviews_df["rating"] >= 4.0
    positives_df = reviews_df[positive_mask].copy()
    pos_users_cnt = int(positives_df["user_id"].nunique())
    pos_items_cnt = int(positives_df["parent_asin"].nunique())
    pos_interactions_cnt = int(len(positives_df))
    positive_fraction = float(pos_interactions_cnt / len(reviews_df)) if len(reviews_df) > 0 else 0.0

    # Temporal leave-last-out split per user on positive interactions
    # Sort positives by timestamp asc, tie break by parent_asin asc
    positives_df = positives_df.sort_values(["timestamp", "parent_asin"], ascending=[True, True])
    
    # Identify the last positive interaction per user
    last_pos_indices = positives_df.groupby("user_id").tail(1).index

    # Construct split column in reviews_df
    reviews_df["split"] = "seen"
    reviews_df.loc[positive_mask, "split"] = "train"
    reviews_df.loc[last_pos_indices, "split"] = "test"
    reviews_df["seen"] = True

    # Rename columns to contract schema
    ratings_parquet = pd.DataFrame({
        "userId": reviews_df["user_id"].astype(str),
        "itemId": reviews_df["parent_asin"].astype(str),
        "rating": reviews_df["rating"].astype(np.float32),
        "timestamp": reviews_df["timestamp"].astype(np.int64),
        "split": reviews_df["split"].astype(str),
        "seen": reviews_df["seen"].astype(bool),
    })

    # Items preparation
    train_ratings = ratings_parquet[ratings_parquet["split"] == "train"]
    item_train_counts = train_ratings.groupby("itemId").size().to_dict()

    meta_dict = meta_df.drop_duplicates("parent_asin").set_index("parent_asin").to_dict("index")
    all_item_ids = sorted(list(valid_meta_items | set(ratings_parquet["itemId"].unique())))

    # Populate item fields
    item_records = []
    for item_id in all_item_ids:
        m = meta_dict.get(item_id, {})
        store_val = str(m.get("store", "") or "").strip()
        if store_val in ("None", "nan", "null"):
            store_val = ""
        n_train = int(item_train_counts.get(item_id, 0))
        in_cat = bool(n_train >= 3)
        price_val = float(m.get("price", np.nan)) if not pd.isna(m.get("price", np.nan)) else np.nan

        item_records.append({
            "itemId": item_id,
            "title": str(m.get("title", "")),
            "store": store_val,
            "main_category": str(m.get("main_category", category)),
            "price": np.float32(price_val),
            "n_train_ratings": np.int32(n_train),
            "in_catalogue": in_cat,
        })
    items_df = pd.DataFrame(item_records)

    # Popularity ranking (0 = most popular, dense over train items)
    train_items_df = items_df[items_df["n_train_ratings"] > 0].sort_values(
        by=["n_train_ratings", "itemId"], ascending=[False, True]
    ).reset_index(drop=True)
    train_items_df["popularity_rank"] = np.arange(len(train_items_df), dtype=np.int32)
    
    pop_rank_map = train_items_df.set_index("itemId")["popularity_rank"].to_dict()
    items_df["popularity_rank"] = items_df["itemId"].map(lambda x: pop_rank_map.get(x, np.int32(999_999_999)))

    # Interaction-mass popularity buckets over CATALOGUE items (n_train_ratings >= 3)
    catalogue_df = items_df[items_df["in_catalogue"]].sort_values(
        by=["n_train_ratings", "itemId"], ascending=[False, True]
    ).copy()
    
    total_cat_interactions = int(catalogue_df["n_train_ratings"].sum())
    catalogue_df["cum_interactions"] = catalogue_df["n_train_ratings"].cumsum()

    head_limit = 0.50 * total_cat_interactions
    mid_limit = 0.80 * total_cat_interactions  # 50% + 30% = 80%

    def _assign_pop_bucket(cum_int: float) -> str:
        if cum_int <= head_limit:
            return "head"
        elif cum_int <= mid_limit:
            return "mid"
        else:
            return "tail"

    catalogue_df["pop_bucket"] = catalogue_df["cum_interactions"].map(_assign_pop_bucket)
    pop_bucket_map = catalogue_df.set_index("itemId")["pop_bucket"].to_dict()
    items_df["pop_bucket"] = items_df["itemId"].map(lambda x: pop_bucket_map.get(x, "tail"))

    # Store footprint metrics & buckets
    # Count catalogue items per store
    cat_stores = catalogue_df[catalogue_df["store"] != ""]
    store_cat_item_counts = cat_stores.groupby("store").size().to_dict()
    store_cat_train_counts = cat_stores.groupby("store")["n_train_ratings"].sum().to_dict()

    items_df["store_n_items"] = items_df["store"].map(lambda s: np.int32(store_cat_item_counts.get(s, 0)))
    items_df["store_n_train"] = items_df["store"].map(lambda s: np.int32(store_cat_train_counts.get(s, 0)))

    # Store bucket: mass rule covering first 80% of catalogue items
    store_summary = []
    for s, count in store_cat_item_counts.items():
        store_summary.append({"store": s, "n_items": count})
    store_summary_df = pd.DataFrame(store_summary).sort_values(by=["n_items", "store"], ascending=[False, True]).reset_index(drop=True)
    
    total_store_cat_items = int(store_summary_df["n_items"].sum()) if len(store_summary_df) > 0 else 0
    store_summary_df["cum_items"] = store_summary_df["n_items"].cumsum()
    store_large_limit = 0.80 * total_store_cat_items

    store_bucket_map = {}
    for _, row in store_summary_df.iterrows():
        s_name = row["store"]
        if row["cum_items"] <= store_large_limit:
            store_bucket_map[s_name] = "large"
        else:
            store_bucket_map[s_name] = "small"

    def _assign_store_bucket(store_name: str) -> str:
        if not store_name:
            return "unknown"
        return store_bucket_map.get(store_name, "small")

    items_df["store_bucket"] = items_df["store"].map(_assign_store_bucket)

    # Save processed parquets
    ratings_path = os.path.join(output_data_dir, "ratings.parquet")
    items_path = os.path.join(output_data_dir, "items.parquet")

    ratings_parquet.to_parquet(ratings_path, index=False)
    items_df.to_parquet(items_path, index=False)
    logger.info("Saved ratings to %s (%d rows)", ratings_path, len(ratings_parquet))
    logger.info("Saved items to %s (%d rows)", items_path, len(items_df))

    # Funnel computation
    user_train_pos = ratings_parquet[ratings_parquet["split"] == "train"].groupby("userId").size()
    exposure_pop_users = int((user_train_pos >= 1).sum())
    relevance_pop_users = int((user_train_pos >= 2).sum())

    cat_items_set = set(catalogue_df["itemId"])
    test_ratings = ratings_parquet[ratings_parquet["split"] == "test"]
    test_outside_cat = int((~test_ratings["itemId"].isin(cat_items_set)).sum())
    test_outside_cat_frac = float(test_outside_cat / len(test_ratings)) if len(test_ratings) > 0 else 0.0

    store_null_rate = float((items_df["store"] == "").mean())

    # Bucket statistics
    bucket_stats = {}
    total_cat_n = len(catalogue_df)
    for b in ("head", "mid", "tail"):
        b_df = catalogue_df[catalogue_df["pop_bucket"] == b]
        b_n = len(b_df)
        b_ints = int(b_df["n_train_ratings"].sum())
        bucket_stats[b] = {
            "n_items": b_n,
            "item_share": float(b_n / total_cat_n) if total_cat_n > 0 else 0.0,
            "interaction_share": float(b_ints / total_cat_interactions) if total_cat_interactions > 0 else 0.0,
        }

    # Store bucket statistics
    store_bucket_stats = {}
    total_stores_cnt = len(store_summary_df) + (1 if (items_df["store"] == "").any() else 0)
    for sb in ("large", "small", "unknown"):
        sb_items = items_df[items_df["store_bucket"] == sb]
        sb_n_items = len(sb_items)
        if sb == "unknown":
            sb_n_stores = 1 if (items_df["store"] == "").any() else 0
        else:
            sb_n_stores = sum(1 for s, b in store_bucket_map.items() if b == sb)
        store_bucket_stats[sb] = {
            "n_stores": sb_n_stores,
            "store_share": float(sb_n_stores / total_stores_cnt) if total_stores_cnt > 0 else 0.0,
            "item_share": float(sb_n_items / len(items_df)) if len(items_df) > 0 else 0.0,
        }

    funnel = {
        "category": category,
        "stages": [
            {"stage": "raw_reviews", "users": raw_users_cnt, "items": raw_items_cnt, "interactions": raw_interactions_cnt},
            {"stage": "positives_ge_4", "users": pos_users_cnt, "items": pos_items_cnt, "interactions": pos_interactions_cnt},
            {"stage": "non_positives_seen", "users": raw_users_cnt, "items": raw_items_cnt, "interactions": raw_interactions_cnt - pos_interactions_cnt},
            {"stage": "after_split", "users": int(ratings_parquet["userId"].nunique()), "items": int(ratings_parquet["itemId"].nunique()), "interactions": len(ratings_parquet)},
            {"stage": "catalogue_ge_3", "users": int(ratings_parquet[ratings_parquet["itemId"].isin(cat_items_set)]["userId"].nunique()), "items": total_cat_n, "interactions": total_cat_interactions},
            {"stage": "eval_exposure_pop", "users": exposure_pop_users, "items": total_cat_n, "interactions": total_cat_interactions},
            {"stage": "eval_relevance_pop", "users": relevance_pop_users, "items": total_cat_n, "interactions": total_cat_interactions},
        ],
        "positive_fraction": positive_fraction,
        "store_null_rate": store_null_rate,
        "test_items_outside_catalogue_frac": test_outside_cat_frac,
        "buckets": bucket_stats,
        "store_buckets": store_bucket_stats,
    }

    funnel_path = os.path.join(output_results_dir, "prepare_funnel.json")
    with open(funnel_path, "w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=2)
    logger.info("Saved prepare funnel to %s", funnel_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="01_prepare: Prepare ratings and items for exposure-bias study.")
    parser.add_argument("--category", type=str, default="Electronics", help="Amazon category")
    parser.add_argument("--limit", type=int, default=None, help="Truncate for smoke testing")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Output directory for parquets")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for funnel")
    args = parser.parse_args()

    prepare_dataset(
        category=args.category,
        output_data_dir=args.data_dir,
        output_results_dir=args.results_dir,
        limit=args.limit,
    )
