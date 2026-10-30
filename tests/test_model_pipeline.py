"""
Comprehensive pipeline tests for Avinash Julakatti's Model & Candidate Generation (02_train.py).

Verifies:
- End-to-end execution of train_and_generate_candidates on a toy dataset.
- Embedding artefact format, types, and invariants (als_factors.npz).
- Candidate schema and seen-item hard masking in baseline_scores.parquet.
- Run manifest contents and hyperparameter logging.
"""

import importlib
import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

train_mod = importlib.import_module("scripts.02_train")
train_and_generate_candidates = train_mod.train_and_generate_candidates

from src.contracts.model_candidate_contract import EmbeddingArtefact


@pytest.fixture
def toy_dataset_dir(tmp_path):
    """Create a minimal valid ratings.parquet and items.parquet fixture."""
    data_dir = tmp_path / "data" / "processed"
    results_dir = tmp_path / "results"
    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    # 6 catalogue items: i0..i4 (catalogue >=3 train ratings), i5 (non-catalogue: 1 rating)
    items_data = [
        {"itemId": "i0", "title": "Item 0", "store": "StoreA", "main_category": "Electronics", "price": 10.0, "n_train_ratings": 5, "popularity_rank": 0, "pop_bucket": "head", "store_n_items": 3, "store_n_train": 12, "store_bucket": "large", "in_catalogue": True},
        {"itemId": "i1", "title": "Item 1", "store": "StoreA", "main_category": "Electronics", "price": 20.0, "n_train_ratings": 4, "popularity_rank": 1, "pop_bucket": "mid", "store_n_items": 3, "store_n_train": 12, "store_bucket": "large", "in_catalogue": True},
        {"itemId": "i2", "title": "Item 2", "store": "StoreB", "main_category": "Electronics", "price": 30.0, "n_train_ratings": 3, "popularity_rank": 2, "pop_bucket": "tail", "store_n_items": 2, "store_n_train": 6, "store_bucket": "small", "in_catalogue": True},
        {"itemId": "i3", "title": "Item 3", "store": "StoreA", "main_category": "Electronics", "price": 15.0, "n_train_ratings": 3, "popularity_rank": 3, "pop_bucket": "tail", "store_n_items": 3, "store_n_train": 12, "store_bucket": "large", "in_catalogue": True},
        {"itemId": "i4", "title": "Item 4", "store": "StoreB", "main_category": "Electronics", "price": 25.0, "n_train_ratings": 3, "popularity_rank": 4, "pop_bucket": "tail", "store_n_items": 2, "store_n_train": 6, "store_bucket": "small", "in_catalogue": True},
        {"itemId": "i5", "title": "Item 5", "store": "StoreB", "main_category": "Electronics", "price": 40.0, "n_train_ratings": 1, "popularity_rank": 5, "pop_bucket": "tail", "store_n_items": 2, "store_n_train": 6, "store_bucket": "small", "in_catalogue": False},
    ]
    items_df = pd.DataFrame(items_data)
    items_path = str(data_dir / "items.parquet")
    items_df.to_parquet(items_path, index=False)

    # 3 users:
    # u0: rated i0, i1 (train positives), i2 (test). Has unseen catalogue items: i3, i4
    # u1: rated i1, i3 (train positives), i0 (test). Has unseen catalogue items: i2, i4
    # u2: rated i0 (train positive), i5 (seen). Has unseen catalogue items: i1, i2, i3, i4
    ratings_data = [
        {"userId": "u0", "itemId": "i0", "rating": 5.0, "timestamp": 1000000000000, "split": "train", "seen": True},
        {"userId": "u0", "itemId": "i1", "rating": 4.5, "timestamp": 1000000001000, "split": "train", "seen": True},
        {"userId": "u0", "itemId": "i2", "rating": 5.0, "timestamp": 1000000002000, "split": "test", "seen": True},
        {"userId": "u1", "itemId": "i1", "rating": 5.0, "timestamp": 1000000000000, "split": "train", "seen": True},
        {"userId": "u1", "itemId": "i3", "rating": 4.0, "timestamp": 1000000001000, "split": "train", "seen": True},
        {"userId": "u1", "itemId": "i0", "rating": 5.0, "timestamp": 1000000002000, "split": "test", "seen": True},
        {"userId": "u2", "itemId": "i0", "rating": 4.0, "timestamp": 1000000000000, "split": "train", "seen": True},
        {"userId": "u2", "itemId": "i5", "rating": 2.0, "timestamp": 1000000001000, "split": "seen", "seen": True},
    ]
    ratings_df = pd.DataFrame(ratings_data)
    ratings_path = str(data_dir / "ratings.parquet")
    ratings_df.to_parquet(ratings_path, index=False)

    return {
        "ratings_path": ratings_path,
        "items_path": items_path,
        "results_dir": str(results_dir),
    }


def test_train_and_candidate_generation_end_to_end(toy_dataset_dir):
    train_and_generate_candidates(
        ratings_path=toy_dataset_dir["ratings_path"],
        items_path=toy_dataset_dir["items_path"],
        results_dir=toy_dataset_dir["results_dir"],
        factors=8,
        regularization=0.01,
        iterations=5,
        seed=42,
        n_candidates=2,
    )

    results_dir = toy_dataset_dir["results_dir"]
    factors_npz = os.path.join(results_dir, "model", "als_factors.npz")
    model_pkl = os.path.join(results_dir, "model", "als_model.pkl")
    baseline_scores_parquet = os.path.join(results_dir, "baseline_scores.parquet")
    run_manifest_json = os.path.join(results_dir, "run_manifest.json")

    assert os.path.exists(factors_npz)
    assert os.path.exists(model_pkl)
    assert os.path.exists(baseline_scores_parquet)
    assert os.path.exists(run_manifest_json)

    # 1. Verify Embedding Artefact
    artefact = EmbeddingArtefact.load_npz(factors_npz)
    assert artefact.user_factors.shape[1] == 8
    assert artefact.item_factors.shape[1] == 8
    assert len(artefact.user_ids) == 3  # u0, u1, u2
    assert len(artefact.item_ids) == 5  # catalogue items: i0..i4

    # 2. Verify Candidate Scores Parquet Schema & Invariants
    scores_df = pd.read_parquet(baseline_scores_parquet)
    assert list(scores_df.columns) == ["userId", "itemId", "score", "rank"]
    assert scores_df["score"].dtype == np.float32
    assert scores_df["rank"].dtype == np.int16

    # Verify Seen-Item Masking: u0 rated i0 and i1 in train, so neither can be recommended to u0
    u0_recs = scores_df[scores_df["userId"] == "u0"]["itemId"].tolist()
    assert "i0" not in u0_recs
    assert "i1" not in u0_recs

    # Verify Non-Catalogue Items are Never Recommended (i5 is non-catalogue)
    assert "i5" not in scores_df["itemId"].values

    # 3. Verify Run Manifest
    with open(run_manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["seed"] == 42
    assert manifest["params"]["factors"] == 8
    assert manifest["counts"]["n_users"] == 3
    assert manifest["counts"]["n_items"] == 5
