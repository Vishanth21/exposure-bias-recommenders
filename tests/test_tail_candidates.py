"""
Unit tests for isolated tail-candidate augmentation (scripts/02b_tail_candidates.py).

Verifies:
- build_tail_augmented_candidates produces exactly N_CANDIDATES (200) rows per user.
- The 201-candidate fallback over-append bug is resolved.
- Top-10 candidates are preserved.
- Uniqueness and exclusion of seen items are strictly maintained.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

tail_cand_mod = importlib.import_module("scripts.02b_tail_candidates")
N_CANDIDATES = tail_cand_mod.N_CANDIDATES
build_tail_augmented_candidates = tail_cand_mod.build_tail_augmented_candidates
from src.contracts.model_candidate_contract import EmbeddingArtefact


def test_tail_augmented_candidates_exact_200(tmp_path: Path):
    data_dir = tmp_path / "data" / "processed"
    source_results = tmp_path / "results"
    results_b = tmp_path / "results_b"
    data_dir.mkdir(parents=True)
    source_results.mkdir(parents=True)
    (source_results / "model").mkdir(parents=True)

    # 1. Create items (250 items: 150 head, 50 mid, 50 tail)
    n_items = 250
    items_list = []
    for i in range(n_items):
        item_id = f"item_{i:03d}"
        if i < 150:
            bucket = "head"
        elif i < 200:
            bucket = "mid"
        else:
            bucket = "tail"
        items_list.append({
            "itemId": item_id,
            "pop_bucket": bucket,
            "popularity_rank": i,
            "in_catalogue": True,
        })
    items_df = pd.DataFrame(items_list)
    items_path = data_dir / "items.parquet"
    items_df.to_parquet(items_path, index=False)

    # 2. Create ratings (1 user: user_0 rated item_249 in train)
    user = "user_0"
    ratings_df = pd.DataFrame([
        {"userId": user, "itemId": "item_249", "split": "train", "rating": 5.0, "seen": True},
        {"userId": user, "itemId": "item_248", "split": "test", "rating": 5.0, "seen": True},
    ])
    ratings_path = data_dir / "ratings.parquet"
    ratings_df.to_parquet(ratings_path, index=False)

    # 3. Create baseline_scores.parquet with 200 items for user_0
    # Items 0..199 (mostly head and mid, plus some tail)
    baseline_items = [f"item_{i:03d}" for i in range(N_CANDIDATES)]
    baseline_scores = [float(200 - i) for i in range(N_CANDIDATES)]
    scores_df = pd.DataFrame({
        "userId": [user] * N_CANDIDATES,
        "itemId": baseline_items,
        "score": np.array(baseline_scores, dtype=np.float32),
        "rank": np.arange(N_CANDIDATES, dtype=np.int16),
    })
    scores_path = source_results / "baseline_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)

    # 4. Create als_factors.npz
    rng = np.random.default_rng(42)
    user_factors = rng.normal(size=(1, 16)).astype(np.float32)
    item_factors = rng.normal(size=(n_items, 16)).astype(np.float32)
    item_ids = np.array([f"item_{i:03d}" for i in range(n_items)])
    user_ids = np.array([user])
    factors = EmbeddingArtefact(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
    )
    factors.save_npz(str(source_results / "model" / "als_factors.npz"))

    # 5. Create source run_manifest and prepare_funnel
    manifest = {
        "params": {"tail_candidate_quota": 0},
    }
    (source_results / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source_results / "prepare_funnel.json").write_text(json.dumps({"stage": "test"}), encoding="utf-8")

    # 6. Run tail candidate augmentation with quota = 20
    build_tail_augmented_candidates(
        ratings_path=ratings_path,
        items_path=items_path,
        source_results=source_results,
        results_dir=results_b,
        tail_candidate_quota=20,
    )

    output_scores = results_b / "baseline_scores.parquet"
    assert output_scores.exists()

    out_df = pd.read_parquet(output_scores)
    # MUST have exactly 200 candidates, NOT 201
    assert len(out_df) == N_CANDIDATES
    assert out_df["userId"].iloc[0] == user
    assert out_df["itemId"].nunique() == N_CANDIDATES
    assert list(out_df["rank"]) == list(range(N_CANDIDATES))

    # Top 10 must be preserved identically
    assert list(out_df["itemId"][:10]) == baseline_items[:10]

    # No known items (e.g. item_249) in candidates
    assert "item_249" not in set(out_df["itemId"])
