"""
Model & Candidate-Set Interface Contract implementation.

Owner: Avinash Julakatti (PES1UG24AM347)
Workstream: Model + Re-Ranker (Track B)
Date: Wednesday, 28-10-2026

This module provides the definitive schema definitions, invariant checks,
and mathematical transformations agreed upon in the Wed 28 Interface Freeze:
1. Embedding Artefact Contract (als_factors.npz)
2. Candidate-Set Schema & Masking Contract (baseline_scores.parquet)
3. Re-Ranker Candidate Boundary Invariant & Z-Scored Penalty Transformation
4. MostPop & Random Baseline Candidate Generators
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants & Contract Commitments
# ---------------------------------------------------------------------------

DEFAULT_FACTORS = 64
DEFAULT_ALPHA = 40.0
DEFAULT_REGULARIZATION = 0.05
DEFAULT_ITERATIONS = 15
DEFAULT_SEED = 42

DEFAULT_N_CANDIDATES = 200
DEFAULT_EVAL_K = 10
DEFAULT_LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
EPSILON_STD = 1e-8


# ---------------------------------------------------------------------------
# 1. Embedding Artefact Contract
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingArtefact:
    """Represents the frozen embedding artefact format (als_factors.npz)."""
    user_factors: np.ndarray  # Shape: (n_users, n_factors), float32
    item_factors: np.ndarray  # Shape: (n_items, n_factors), float32
    user_ids: np.ndarray      # Shape: (n_users,), string
    item_ids: np.ndarray      # Shape: (n_items,), string

    def validate(self) -> None:
        """Enforce strict dimensional and numerical invariants."""
        if not isinstance(self.user_factors, np.ndarray) or not isinstance(self.item_factors, np.ndarray):
            raise TypeError("user_factors and item_factors must be numpy ndarrays.")
        if not isinstance(self.user_ids, np.ndarray) or not isinstance(self.item_ids, np.ndarray):
            raise TypeError("user_ids and item_ids must be numpy ndarrays.")

        if self.user_factors.ndim != 2:
            raise ValueError(f"user_factors must be 2D, got shape {self.user_factors.shape}")
        if self.item_factors.ndim != 2:
            raise ValueError(f"item_factors must be 2D, got shape {self.item_factors.shape}")

        if self.user_factors.shape[1] != self.item_factors.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: user {self.user_factors.shape[1]} != item {self.item_factors.shape[1]}"
            )

        if len(self.user_ids) != self.user_factors.shape[0]:
            raise ValueError(
                f"user_ids length {len(self.user_ids)} does not match user_factors rows {self.user_factors.shape[0]}"
            )

        if len(self.item_ids) != self.item_factors.shape[0]:
            raise ValueError(
                f"item_ids length {len(self.item_ids)} does not match item_factors rows {self.item_factors.shape[0]}"
            )

        if np.isnan(self.user_factors).any() or np.isinf(self.user_factors).any():
            raise ValueError("user_factors contains NaN or Inf values.")
        if np.isnan(self.item_factors).any() or np.isinf(self.item_factors).any():
            raise ValueError("item_factors contains NaN or Inf values.")

    def save_npz(self, filepath: str) -> None:
        """Save compressed npz archive conforming to the contract."""
        self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            user_factors=self.user_factors.astype(np.float32),
            item_factors=self.item_factors.astype(np.float32),
            user_ids=self.user_ids.astype(str),
            item_ids=self.item_ids.astype(str),
        )

    @classmethod
    def load_npz(cls, filepath: str) -> EmbeddingArtefact:
        """Load and validate npz archive conforming to the contract."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        data = np.load(filepath, allow_pickle=True)
        required_keys = {"user_factors", "item_factors", "user_ids", "item_ids"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f"Missing required keys in embedding npz: {missing}")

        artefact = cls(
            user_factors=data["user_factors"].astype(np.float32),
            item_factors=data["item_factors"].astype(np.float32),
            user_ids=data["user_ids"].astype(str),
            item_ids=data["item_ids"].astype(str),
        )
        artefact.validate()
        return artefact


# ---------------------------------------------------------------------------
# 2. Candidate-Set Generation Contract
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """A single candidate record for baseline_scores or rerank_scores."""
    user_id: str
    item_id: str
    score: float
    rank: int


def generate_user_candidates(
    user_vector: np.ndarray,
    item_factors: np.ndarray,
    item_ids: Sequence[str],
    seen_item_ids: Set[str],
    n_candidates: int = DEFAULT_N_CANDIDATES,
    item_popularity_ranks: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, float, int]]:
    """
    Score catalogue items via full-ranking dot product and select top-N candidates.
    
    Hard invariant: Excludes all items already rated/seen by the user.
    Deterministic tie-breaking: score desc -> popularity_rank asc -> item_id asc.
    Returns: list of (item_id, score, rank).
    """
    if user_vector.ndim == 1:
        user_vector = user_vector.reshape(1, -1)
    
    raw_scores = (user_vector @ item_factors.T).flatten()
    
    candidates: List[Tuple[float, int, str]] = []
    pop_ranks = item_popularity_ranks or {}

    for idx, item_id in enumerate(item_ids):
        if item_id in seen_item_ids:
            continue
        score = float(raw_scores[idx])
        pop_rank = pop_ranks.get(item_id, 999_999_999)
        candidates.append((score, pop_rank, item_id))

    # Sort descending by score, ascending by pop_rank (more popular first on tie), ascending by item_id
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    top_candidates = candidates[:n_candidates]
    return [
        (item_id, float(score), rank + 1)
        for rank, (score, _, item_id) in enumerate(top_candidates)
    ]


# ---------------------------------------------------------------------------
# 3. Re-Ranking & Candidate Invariant Contract
# ---------------------------------------------------------------------------

def compute_zscore(values: Sequence[float], eps: float = EPSILON_STD) -> np.ndarray:
    """
    Compute sample standard score z(x) = (x - mean) / std.
    Zero-variance fallback: if std < eps, returns array of zeros.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.array([], dtype=np.float32)
    std = float(np.std(arr))
    if std < eps:
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(np.mean(arr))
    return ((arr - mean) / std).astype(np.float32)


def rerank_candidate_list(
    candidate_items: Sequence[str],
    candidate_scores: Sequence[float],
    candidate_popularities: Sequence[int | float],
    lambda_val: float,
    eps: float = EPSILON_STD,
) -> List[Tuple[str, float, int]]:
    """
    Apply post-hoc popularity taxation re-ranker.
    
    Formula:
        score_final = z(score) - lambda * z(log(1 + pop))
    
    Invariants:
    1. Candidate set boundary isolation: candidate_items set is preserved exactly.
    2. Lambda = 0.0 is an exact no-op preserving initial relative rank order.
    Returns: list of (item_id, final_score, rank) sorted descending by final_score.
    """
    n = len(candidate_items)
    if n == 0:
        return []

    # If lambda is 0, preserve exact original order
    if abs(lambda_val) < 1e-12:
        return [
            (candidate_items[i], float(candidate_scores[i]), i + 1)
            for i in range(n)
        ]

    z_scores = compute_zscore(candidate_scores, eps=eps)
    log_pops = [np.log1p(max(0.0, float(p))) for p in candidate_popularities]
    z_pops = compute_zscore(log_pops, eps=eps)

    penalized_scores = z_scores - (float(lambda_val) * z_pops)

    # Sort descending by final score; break ties with original rank
    indexed_items = [
        (penalized_scores[i], i, candidate_items[i])
        for i in range(n)
    ]
    indexed_items.sort(key=lambda x: (-x[0], x[1]))

    return [
        (item_id, float(score), new_rank + 1)
        for new_rank, (score, _, item_id) in enumerate(indexed_items)
    ]


def verify_candidate_boundary_invariant(
    baseline_items: Sequence[str],
    reranked_items: Sequence[str],
) -> bool:
    """
    Enforce hard boundary invariant:
    Re-ranker MUST permute the identical candidate item set without additions or deletions.
    """
    if len(baseline_items) != len(reranked_items):
        return False
    return set(baseline_items) == set(reranked_items)


# ---------------------------------------------------------------------------
# 4. Baselines Contract (MostPop & Random)
# ---------------------------------------------------------------------------

def generate_mostpop_candidates(
    catalogue_items: Sequence[str],
    item_train_ratings: Dict[str, int],
    seen_item_ids: Set[str],
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> List[Tuple[str, float, int]]:
    """
    Generate MostPop baseline candidates:
    Catalogue items with highest train interaction count, excluding seen items.
    """
    unseen_items = [item for item in catalogue_items if item not in seen_item_ids]
    # Sort descending by interaction count, then item_id
    unseen_items.sort(key=lambda item: (-item_train_ratings.get(item, 0), item))

    top_items = unseen_items[:n_candidates]
    return [
        (item, float(item_train_ratings.get(item, 0)), rank + 1)
        for rank, item in enumerate(top_items)
    ]


def generate_random_candidates(
    catalogue_items: Sequence[str],
    seen_item_ids: Set[str],
    seed: int,
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> List[Tuple[str, float, int]]:
    """
    Generate Random baseline candidates:
    Uniform random sample of catalogue items without replacement, excluding seen items.
    """
    unseen_items = [item for item in catalogue_items if item not in seen_item_ids]
    k = min(n_candidates, len(unseen_items))
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(unseen_items), size=k, replace=False)
    
    return [
        (unseen_items[idx], 1.0 / (rank + 1), rank + 1)
        for rank, idx in enumerate(sampled_indices)
    ]
