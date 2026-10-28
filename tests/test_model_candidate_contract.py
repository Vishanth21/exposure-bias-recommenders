"""
Unit tests for Avinash's Model & Candidate-Set Contract (Wed 28 Interface Freeze).

Verifies:
- EmbeddingArtefact npz serialization, dimensions, and invariants.
- Candidate generation with full-ranking dot product and seen-item hard masking.
- Z-score computation including edge cases (zero variance).
- Re-ranker candidate boundary isolation invariant.
- Re-ranker z-score penalty math and lambda=0 no-op preservation.
- MostPop and Random baseline contracts.
"""

import os
import tempfile
import numpy as np
import pytest

from src.contracts.model_candidate_contract import (
    EmbeddingArtefact,
    compute_zscore,
    generate_mostpop_candidates,
    generate_random_candidates,
    generate_user_candidates,
    rerank_candidate_list,
    verify_candidate_boundary_invariant,
)


class TestEmbeddingArtefactContract:
    def test_valid_embedding_artefact_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "als_factors.npz")
            n_users, n_items, n_factors = 10, 20, 64
            
            user_factors = np.random.randn(n_users, n_factors).astype(np.float32)
            item_factors = np.random.randn(n_items, n_factors).astype(np.float32)
            user_ids = np.array([f"user_{i}" for i in range(n_users)])
            item_ids = np.array([f"item_{j}" for j in range(n_items)])

            artefact = EmbeddingArtefact(
                user_factors=user_factors,
                item_factors=item_factors,
                user_ids=user_ids,
                item_ids=item_ids,
            )
            artefact.save_npz(filepath)
            assert os.path.exists(filepath)

            loaded = EmbeddingArtefact.load_npz(filepath)
            assert loaded.user_factors.shape == (n_users, n_factors)
            assert loaded.item_factors.shape == (n_items, n_factors)
            assert len(loaded.user_ids) == n_users
            assert len(loaded.item_ids) == n_items
            np.testing.assert_allclose(loaded.user_factors, user_factors, rtol=1e-5)
            np.testing.assert_allclose(loaded.item_factors, item_factors, rtol=1e-5)

    def test_dimension_mismatch_raises(self):
        artefact = EmbeddingArtefact(
            user_factors=np.ones((5, 32), dtype=np.float32),
            item_factors=np.ones((10, 64), dtype=np.float32),  # mismatch 32 != 64
            user_ids=np.array(["u1", "u2", "u3", "u4", "u5"]),
            item_ids=np.array([f"i{j}" for j in range(10)]),
        )
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            artefact.validate()

    def test_nan_or_inf_raises(self):
        user_factors = np.ones((2, 16), dtype=np.float32)
        user_factors[0, 0] = np.nan
        artefact = EmbeddingArtefact(
            user_factors=user_factors,
            item_factors=np.ones((3, 16), dtype=np.float32),
            user_ids=np.array(["u1", "u2"]),
            item_ids=np.array(["i1", "i2", "i3"]),
        )
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            artefact.validate()


class TestCandidateSetContract:
    def test_full_ranking_candidate_generation_and_seen_masking(self):
        user_vec = np.array([1.0, 0.0], dtype=np.float32)
        item_factors = np.array([
            [1.0, 0.0],   # item_0: dot=1.0 (seen -> should be masked!)
            [0.8, 0.0],   # item_1: dot=0.8
            [0.5, 0.0],   # item_2: dot=0.5
            [0.9, 0.0],   # item_3: dot=0.9
            [0.2, 0.0],   # item_4: dot=0.2
        ], dtype=np.float32)
        item_ids = ["item_0", "item_1", "item_2", "item_3", "item_4"]
        seen = {"item_0"}

        candidates = generate_user_candidates(
            user_vector=user_vec,
            item_factors=item_factors,
            item_ids=item_ids,
            seen_item_ids=seen,
            n_candidates=3,
        )

        assert len(candidates) == 3
        # item_0 is masked. The remaining highest are item_3 (0.9), item_1 (0.8), item_2 (0.5)
        assert candidates[0][0] == "item_3"
        assert candidates[0][2] == 1  # rank 1
        assert candidates[1][0] == "item_1"
        assert candidates[1][2] == 2  # rank 2
        assert candidates[2][0] == "item_2"
        assert candidates[2][2] == 3  # rank 3
        assert "item_0" not in [c[0] for c in candidates]

    def test_deterministic_tie_breaking(self):
        user_vec = np.array([1.0], dtype=np.float32)
        item_factors = np.array([[1.0], [1.0]], dtype=np.float32)  # equal scores
        item_ids = ["item_b", "item_a"]
        pop_ranks = {"item_a": 1, "item_b": 2}  # item_a is more popular (rank 1)

        candidates = generate_user_candidates(
            user_vector=user_vec,
            item_factors=item_factors,
            item_ids=item_ids,
            seen_item_ids=set(),
            n_candidates=2,
            item_popularity_ranks=pop_ranks,
        )
        assert candidates[0][0] == "item_a"
        assert candidates[1][0] == "item_b"


class TestZScoreAndReRankerContract:
    def test_compute_zscore_hand_values(self):
        values = [10.0, 20.0, 30.0]
        # mean = 20.0, std = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2)/3) = sqrt(200/3) = 8.1649658
        z = compute_zscore(values)
        assert len(z) == 3
        assert z[1] == pytest.approx(0.0, abs=1e-5)
        assert z[0] == pytest.approx(-z[2], abs=1e-5)
        assert z[2] == pytest.approx((30.0 - 20.0) / np.sqrt(200.0 / 3.0), abs=1e-5)

    def test_compute_zscore_zero_variance(self):
        values = [5.0, 5.0, 5.0, 5.0]
        z = compute_zscore(values)
        assert np.all(z == 0.0)

    def test_candidate_boundary_invariant(self):
        baseline = ["item_1", "item_2", "item_3"]
        reranked_valid = ["item_3", "item_1", "item_2"]
        reranked_invalid_added = ["item_1", "item_2", "item_999"]
        reranked_invalid_dropped = ["item_1", "item_2"]

        assert verify_candidate_boundary_invariant(baseline, reranked_valid) is True
        assert verify_candidate_boundary_invariant(baseline, reranked_invalid_added) is False
        assert verify_candidate_boundary_invariant(baseline, reranked_invalid_dropped) is False

    def test_rerank_lambda_zero_is_exact_noop(self):
        items = ["item_a", "item_b", "item_c"]
        scores = [3.5, 2.1, 1.0]
        pops = [1000, 10, 500]

        reranked = rerank_candidate_list(
            candidate_items=items,
            candidate_scores=scores,
            candidate_popularities=pops,
            lambda_val=0.0,
        )
        assert [r[0] for r in reranked] == items
        assert [r[2] for r in reranked] == [1, 2, 3]

    def test_rerank_penalty_taxes_popular_item(self):
        # Two items: item_pop (high score, huge pop), item_tail (slightly lower score, low pop)
        items = ["item_pop", "item_tail"]
        scores = [2.05, 1.95]  # pop item has higher score initially
        pops = [100000, 1]

        # Under strong penalty lambda = 2.0, item_tail should overtake item_pop
        reranked = rerank_candidate_list(
            candidate_items=items,
            candidate_scores=scores,
            candidate_popularities=pops,
            lambda_val=2.0,
        )
        assert reranked[0][0] == "item_tail"
        assert reranked[0][2] == 1
        assert reranked[1][0] == "item_pop"
        assert reranked[1][2] == 2


class TestBaselinesContract:
    def test_mostpop_baseline(self):
        catalogue = ["item_1", "item_2", "item_3", "item_4"]
        counts = {"item_1": 100, "item_2": 500, "item_3": 50, "item_4": 1000}
        seen = {"item_4"}  # item_4 is seen

        candidates = generate_mostpop_candidates(
            catalogue_items=catalogue,
            item_train_ratings=counts,
            seen_item_ids=seen,
            n_candidates=2,
        )
        assert len(candidates) == 2
        # item_4 is skipped; item_2 (500) then item_1 (100)
        assert candidates[0][0] == "item_2"
        assert candidates[0][2] == 1
        assert candidates[1][0] == "item_1"
        assert candidates[1][2] == 2

    def test_random_baseline(self):
        catalogue = [f"item_{i}" for i in range(50)]
        seen = {"item_0", "item_1"}

        cand1 = generate_random_candidates(catalogue, seen, seed=42, n_candidates=10)
        cand2 = generate_random_candidates(catalogue, seen, seed=42, n_candidates=10)
        cand_diff_seed = generate_random_candidates(catalogue, seen, seed=99, n_candidates=10)

        assert len(cand1) == 10
        assert all(c[0] not in seen for c in cand1)
        assert [c[0] for c in cand1] == [c[0] for c in cand2]  # deterministic with seed
        assert [c[0] for c in cand1] != [c[0] for c in cand_diff_seed]
