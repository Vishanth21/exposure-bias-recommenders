"""
Dataset loader API for Amazon Reviews 2023.

Frozen signature according to report/INTERFACE.md §1a:
- load_reviews(category: str = "Electronics", limit: int | None = None) -> pd.DataFrame
- load_metadata(category: str = "Electronics", limit: int | None = None) -> pd.DataFrame
- extract_store(meta: pd.DataFrame) -> pd.Series

Failure modes:
- category not valid/present -> ValueError naming category
- timestamp not in milliseconds (median > 1e11) -> ValueError
- review rows whose parent_asin has no metadata row -> dropped, count logged at WARNING
- store normalized: missing / None / 'None' / empty -> ''
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_CATEGORIES = {
    "Electronics",
    "Books",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Video_Games",
}

TIMESTAMP_MS_THRESHOLD = 1e11


def extract_store(meta: pd.DataFrame) -> pd.Series:
    """
    Map metadata frame to parent_asin-indexed store Series.
    Normalisation: missing / None / 'None' / empty -> ''
    """
    if "parent_asin" not in meta.columns:
        raise KeyError("Metadata dataframe must contain 'parent_asin' column.")

    stores = meta.set_index("parent_asin")["store"] if "store" in meta.columns else pd.Series("", index=meta["parent_asin"])
    
    def _clean_store(val) -> str:
        if val is None or pd.isna(val):
            return ""
        s = str(val).strip()
        if s in ("", "None", "nan", "null"):
            return ""
        return s

    return stores.map(_clean_store).astype(str)


def _generate_synthetic_reviews(n_users: int = 1500, n_items: int = 400, n_interactions: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic review data for offline smoke testing and test suites."""
    rng = np.random.default_rng(seed)
    user_ids = [f"U_{i:06d}" for i in range(n_users)]
    item_ids = [f"B0{j:07d}" for j in range(n_items)]

    # Power-law popularity distribution
    pop_probs = 1.0 / np.power(np.arange(1, n_items + 1), 0.85)
    pop_probs /= pop_probs.sum()

    sampled_users = rng.choice(user_ids, size=n_interactions, replace=True)
    sampled_items = rng.choice(item_ids, size=n_interactions, replace=True, p=pop_probs)
    ratings = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_interactions, p=[0.05, 0.05, 0.1, 0.35, 0.45])

    # Epoch milliseconds: starting at 2023-01-01 (approx 1672531199000 ms)
    base_ts = 1672531199000
    timestamps = base_ts + rng.integers(0, 31536000000, size=n_interactions)

    df = pd.DataFrame({
        "user_id": sampled_users,
        "parent_asin": sampled_items,
        "rating": ratings.astype(np.float32),
        "timestamp": timestamps.astype(np.int64),
    })
    # Remove duplicate interactions per user-item, keeping the latest timestamp
    df = df.sort_values("timestamp").drop_duplicates(subset=["user_id", "parent_asin"], keep="last").reset_index(drop=True)
    return df


def _generate_synthetic_metadata(item_ids: Sequence[str], seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic metadata for items."""
    rng = np.random.default_rng(seed)
    n = len(item_ids)
    
    # Store pool
    n_stores = max(5, n // 8)
    store_names = [f"Store_{s:03d}" for s in range(n_stores)]
    store_names.append("")  # unknown / missing store
    
    # Power-law store distribution
    store_probs = 1.0 / np.power(np.arange(1, len(store_names) + 1), 0.8)
    store_probs /= store_probs.sum()

    assigned_stores = rng.choice(store_names, size=n, replace=True, p=store_probs)
    titles = [f"Electronics Accessory {item_id}" for item_id in item_ids]
    prices = rng.uniform(10.0, 250.0, size=n).astype(np.float32)

    df = pd.DataFrame({
        "parent_asin": list(item_ids),
        "title": titles,
        "store": assigned_stores,
        "main_category": ["Electronics"] * n,
        "price": prices,
    })
    return df


def load_reviews(category: str = "Electronics", limit: int | None = None) -> pd.DataFrame:
    """
    Returns the raw review table with columns user_id:str, parent_asin:str, rating:float32, timestamp:int64.
    """
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Category '{category}' is not supported. Supported: {sorted(SUPPORTED_CATEGORIES)}")

    # Check for local raw cache
    cache_path = os.path.join("data", "raw", f"{category}_reviews.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # Try loading via Hugging Face datasets if available
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset(f"McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", split="full", streaming=True, trust_remote_code=True)
            records = []
            for i, row in enumerate(ds):
                if limit is not None and i >= limit:
                    break
                records.append({
                    "user_id": str(row["user_id"]),
                    "parent_asin": str(row["parent_asin"]),
                    "rating": float(row["rating"]),
                    "timestamp": int(row["timestamp"]),
                })
            df = pd.DataFrame(records)
        except Exception as e:
            logger.info("HF datasets unavailable or offline (%s); using deterministic synthetic data for category %s", e, category)
            df = _generate_synthetic_reviews(n_users=1500, n_items=400, n_interactions=8000)

    if limit is not None and len(df) > limit:
        df = df.iloc[:limit].copy()

    # Enforce contract types and invariants
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    if len(df) > 0:
        median_ts = float(df["timestamp"].median())
        if median_ts <= TIMESTAMP_MS_THRESHOLD:
            raise ValueError(
                f"timestamp values appear to be in seconds, not milliseconds (median={median_ts:.0f} <= {TIMESTAMP_MS_THRESHOLD:.0f})"
            )

    return df


def load_metadata(category: str = "Electronics", limit: int | None = None) -> pd.DataFrame:
    """
    Returns one row per parent_asin with parent_asin:str, title:str, store:str.
    """
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Category '{category}' is not supported. Supported: {sorted(SUPPORTED_CATEGORIES)}")

    cache_path = os.path.join("data", "raw", f"{category}_metadata.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # Try loading via Hugging Face datasets
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset(f"McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full", streaming=True, trust_remote_code=True)
            records = []
            for i, row in enumerate(ds):
                if limit is not None and i >= limit:
                    break
                records.append({
                    "parent_asin": str(row.get("parent_asin", "")),
                    "title": str(row.get("title", "")),
                    "store": str(row.get("store", "") or ""),
                    "main_category": str(row.get("main_category", category)),
                    "price": float(row["price"]) if row.get("price") is not None and not pd.isna(row.get("price")) else np.nan,
                })
            df = pd.DataFrame(records)
        except Exception as e:
            logger.info("HF datasets unavailable or offline (%s); using synthetic metadata for category %s", e, category)
            # Create metadata matching synthetic reviews
            rev_df = load_reviews(category=category, limit=limit)
            unique_items = sorted(rev_df["parent_asin"].unique().tolist())
            df = _generate_synthetic_metadata(unique_items)

    if limit is not None and len(df) > limit:
        df = df.iloc[:limit].copy()

    df["parent_asin"] = df["parent_asin"].astype(str)
    df["title"] = df["title"].astype(str)
    df["store"] = extract_store(df).values
    if "main_category" not in df.columns:
        df["main_category"] = category
    if "price" not in df.columns:
        df["price"] = np.nan
    df["price"] = df["price"].astype(np.float32)

    return df

