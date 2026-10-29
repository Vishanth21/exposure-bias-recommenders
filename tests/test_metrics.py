"""Unit tests for src/metrics.py.

Expected values are hand-computed in the comments, so each test checks the definition
rather than re-deriving what the implementation does.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (  # noqa: E402
    catalog_coverage,
    exposure_by_cross,
    exposure_parity,
    exposure_share,
    gini,
    hitrate_at_k,
    mean_recommended_popularity,
    ndcg_at_k,
    recall_at_k,
    store_exposure_share,
)

LOG2 = math.log2


class TestNdcg(unittest.TestCase):
    def test_perfect_ranking(self):
        # relevant item at position 0 only: DCG = 1/log2(2) = 1, IDCG = 1 -> 1.0
        self.assertAlmostEqual(ndcg_at_k([7, 8, 9], {7}, 3), 1.0, places=12)

    def test_relevant_at_second_position(self):
        # DCG = 1/log2(3); IDCG = 1 -> 1/log2(3)
        expected = 1.0 / LOG2(3)
        self.assertAlmostEqual(ndcg_at_k([8, 7, 9], {7}, 3), expected, places=12)

    def test_two_relevant_both_found(self):
        # both relevant occupy the first two slots -> DCG == IDCG
        self.assertAlmostEqual(ndcg_at_k([7, 8, 9], {7, 8}, 3), 1.0, places=12)

    def test_two_relevant_one_missed(self):
        # DCG = 1; IDCG = 1 + 1/log2(3)
        expected = 1.0 / (1.0 / LOG2(2) + 1.0 / LOG2(3))
        self.assertAlmostEqual(ndcg_at_k([7, 9, 10], {7, 8}, 3), expected, places=12)

    def test_k_truncates(self):
        self.assertEqual(ndcg_at_k([9, 9, 7], {7}, 2), 0.0)

    def test_empty_relevant_returns_zero(self):
        self.assertEqual(ndcg_at_k([1, 2, 3], set(), 3), 0.0)

    def test_k_zero(self):
        self.assertEqual(ndcg_at_k([1, 2, 3], {1}, 0), 0.0)

    def test_k_larger_than_recs(self):
        self.assertAlmostEqual(ndcg_at_k([7], {7}, 10), 1.0, places=12)


class TestRecall(unittest.TestCase):
    def test_half(self):
        # 2 hits / 4 relevant
        self.assertAlmostEqual(recall_at_k([1, 2, 9, 10], {1, 2, 3, 4}, 4), 0.5, places=12)

    def test_all(self):
        self.assertAlmostEqual(recall_at_k([1, 2], {1, 2}, 10), 1.0, places=12)

    def test_none(self):
        self.assertEqual(recall_at_k([8, 9], {1, 2}, 10), 0.0)

    def test_empty_relevant(self):
        self.assertEqual(recall_at_k([1, 2], set(), 10), 0.0)

    def test_k_truncation_matters(self):
        self.assertEqual(recall_at_k([9, 1, 2], {1, 2}, 1), 0.0)
        self.assertEqual(recall_at_k([9, 1, 2], {1, 2}, 3), 1.0)


class TestHitRate(unittest.TestCase):
    def test_hit(self):
        self.assertEqual(hitrate_at_k([9, 1, 2], {1, 2}, 2), 1.0)

    def test_miss(self):
        self.assertEqual(hitrate_at_k([9, 1, 2], {1, 2}, 1), 0.0)

    def test_empty(self):
        self.assertEqual(hitrate_at_k([1], set(), 1), 0.0)


class TestExposureShare(unittest.TestCase):
    def test_even_split(self):
        # 2 head slots, 2 tail slots, 4 total
        bucket_of = {1: "head", 2: "head", 3: "tail"}
        out = exposure_share([[1, 2], [3, 3]], bucket_of)
        self.assertAlmostEqual(out["head"], 0.5, places=12)
        self.assertAlmostEqual(out["mid"], 0.0, places=12)
        self.assertAlmostEqual(out["tail"], 0.5, places=12)

    def test_sums_to_one(self):
        bucket_of = {1: "head", 2: "mid", 3: "tail", 4: "tail"}
        out = exposure_share([[1, 2, 3], [4]], bucket_of)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=12)

    def test_all_slots_same_bucket(self):
        bucket_of = {i: "head" for i in range(5)}
        out = exposure_share([[0, 1, 2], [3, 4]], bucket_of)
        self.assertAlmostEqual(out["head"], 1.0, places=12)
        self.assertAlmostEqual(out["tail"], 0.0, places=12)

    def test_empty_input(self):
        out = exposure_share([], {1: "head"})
        self.assertEqual(out["head"], 0.0)
        self.assertEqual(out["mid"], 0.0)
        self.assertEqual(out["tail"], 0.0)

    def test_unbucketed_items_skipped(self):
        # item 99 unmapped -> skipped, so both counted slots are head
        bucket_of = {1: "head", 2: "head"}
        out = exposure_share([[1, 99, 2]], bucket_of)
        self.assertAlmostEqual(out["head"], 1.0, places=12)

    def test_lists_not_truncated(self):
        bucket_of = {1: "head", 2: "tail"}
        out = exposure_share([[1, 2, 2, 2]], bucket_of)
        self.assertAlmostEqual(out["head"], 0.25, places=12)
        self.assertAlmostEqual(out["tail"], 0.75, places=12)


class TestStoreExposureShare(unittest.TestCase):

    def test_three_buckets_sum_to_one(self):
        bucket_of = {0: "small", 1: "small", 2: "large", 3: "large"}
        recs = [[0, 2], [1, 3]]
        out = store_exposure_share(recs, bucket_of)
        self.assertAlmostEqual(out["small"], 0.5)
        self.assertAlmostEqual(out["large"], 0.5)
        self.assertAlmostEqual(out["unknown"], 0.0)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_unknown_always_present_and_counted(self):
        bucket_of = {0: "small", 1: "unknown"}
        out = store_exposure_share([[0, 1]], bucket_of)
        self.assertEqual(set(out), {"small", "large", "unknown"})
        self.assertAlmostEqual(out["small"], 0.5)
        self.assertAlmostEqual(out["unknown"], 0.5)
        self.assertAlmostEqual(out["large"], 0.0)

    def test_missing_bucket_still_returns_vocabulary(self):
        out = store_exposure_share([], {})
        self.assertEqual(set(out), {"small", "large", "unknown"})
        self.assertTrue(all(v == 0.0 for v in out.values()))

    def test_unmapped_items_skipped_not_counted_as_unknown(self):
        out = store_exposure_share([[0, 9]], {0: "small"})
        self.assertAlmostEqual(out["small"], 1.0)
        self.assertAlmostEqual(out["unknown"], 0.0)


class TestCoverage(unittest.TestCase):
    def test_quarter(self):
        # 2 distinct items of 8
        self.assertAlmostEqual(catalog_coverage([[1, 2], [1]], 8), 0.25, places=12)

    def test_full(self):
        self.assertAlmostEqual(catalog_coverage([[1], [2], [3], [4]], 4), 1.0, places=12)

    def test_empty_catalog(self):
        self.assertEqual(catalog_coverage([[1, 2]], 0), 0.0)

    def test_no_recs(self):
        self.assertEqual(catalog_coverage([], 10), 0.0)


class TestGini(unittest.TestCase):
    def test_perfect_equality(self):
        self.assertAlmostEqual(gini([5, 5, 5, 5]), 0.0, places=12)

    def test_known_two_values(self):
        # x = [0, 10]: weighted = 1*0 + 2*10 = 20 ; total = 10 ; n = 2
        # G = (2*20)/(2*10) - (2+1)/2 = 2.0 - 1.5 = 0.5
        self.assertAlmostEqual(gini([0, 10]), 0.5, places=12)

    def test_known_three_values(self):
        # x = [1, 2, 3]: weighted = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        # total = 6 ; n = 3
        # G = (2*14)/(3*6) - 4/3 = 28/18 - 1.3333... = 1.5555... - 1.3333... = 0.2222...
        expected = 28.0 / 18.0 - 4.0 / 3.0
        self.assertAlmostEqual(gini([1, 2, 3]), expected, places=12)

    def test_max_concentration(self):
        # one value holds everything: G = 1 - 1/n -> for n=100, 0.99
        vals = [0] * 99 + [1000]
        self.assertAlmostEqual(gini(vals), 0.99, places=10)

    def test_all_zero(self):
        self.assertEqual(gini([0, 0, 0]), 0.0)

    def test_empty(self):
        self.assertEqual(gini([]), 0.0)

    def test_order_independent(self):
        self.assertAlmostEqual(gini([3, 1, 2]), gini([1, 2, 3]), places=12)


class TestCrossChecks(unittest.TestCase):
    """Consistency properties that must hold for any sane implementation."""

    def test_perfect_ranking_gives_max_all_metrics(self):
        recs = [1, 2, 3, 4, 5]
        rels = set(recs)
        self.assertAlmostEqual(ndcg_at_k(recs, rels, 5), 1.0, places=12)
        self.assertAlmostEqual(recall_at_k(recs, rels, 5), 1.0, places=12)
        self.assertEqual(hitrate_at_k(recs, rels, 5), 1.0)

    def test_ndcg_monotone_in_position(self):
        # moving the relevant item later can never increase NDCG
        vals = [ndcg_at_k([9] * p + [1], {1}, 6) for p in range(6)]
        for earlier, later in zip(vals, vals[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_more_head_slots_means_higher_head_share(self):
        # 1 of 4 slots head  vs  3 of 4 slots head
        bucket_of = {0: "head", 1: "mid", 2: "mid", 3: "tail"}
        flat = exposure_share([[0, 1, 2, 3]], bucket_of)   # head = 1/4
        skew = exposure_share([[0, 1, 0, 3]], bucket_of)   # head = 2/4
        self.assertAlmostEqual(flat["head"], 0.25, places=12)
        self.assertAlmostEqual(skew["head"], 0.50, places=12)
        self.assertGreater(skew["head"], flat["head"])
        # and the long tail loses share in the skewed case
        self.assertAlmostEqual(flat["tail"], 0.25, places=12)
        self.assertAlmostEqual(skew["tail"], 0.25, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestExposureParity(unittest.TestCase):
    """Hand-computed: exposure / catalogue share."""

    def test_exact_proportionality_is_one(self):
        # 50% of items take 50% of slots -> parity 1.0 for every bucket.
        exposure = {"head": 0.5, "mid": 0.3, "tail": 0.2}
        share = {"head": 0.5, "mid": 0.3, "tail": 0.2}
        got = exposure_parity(exposure, share)
        for b in ("head", "mid", "tail"):
            self.assertAlmostEqual(got[b], 1.0, places=12)

    def test_over_exposure(self):
        # head: 2% of items, 5% of slots -> 0.05/0.02 = 2.5
        # mid:  18% of items, 25% of slots -> 0.25/0.18 = 1.3888...
        # tail: 80% of items, 70% of slots -> 0.70/0.80 = 0.875
        got = exposure_parity(
            {"head": 0.05, "mid": 0.25, "tail": 0.70},
            {"head": 0.02, "mid": 0.18, "tail": 0.80},
        )
        self.assertAlmostEqual(got["head"], 2.5, places=12)
        self.assertAlmostEqual(got["mid"], 0.25 / 0.18, places=12)
        self.assertAlmostEqual(got["tail"], 0.875, places=12)

    def test_zero_catalogue_share_yields_zero(self):
        got = exposure_parity({"head": 1.0}, {"head": 0.0})
        self.assertEqual(got["head"], 0.0)

    def test_all_requested_buckets_present(self):
        got = exposure_parity({"head": 1.0}, {"head": 1.0, "mid": 0.0, "tail": 0.0})
        self.assertEqual(set(got), {"head", "mid", "tail"})


class TestExposureByCross(unittest.TestCase):
    """Hand-computed joint partition over 6 slots."""

    def test_known_partition(self):
        # slots: head-large x2, head-small x1, tail-small x3 = 6 slots
        recs = [[1, 2], [3, 4], [5, 6]]
        pop = {1: "head", 2: "head", 3: "head", 4: "tail", 5: "tail", 6: "tail"}
        store = {1: "large", 2: "large", 3: "small", 4: "small", 5: "small", 6: "small"}

        got = exposure_by_cross([recs[0], recs[1], recs[2]], pop, store)

        # head: large 2/6, small 1/6 ; tail: small 3/6
        self.assertAlmostEqual(got["head"]["large"], 2 / 6, places=12)
        self.assertAlmostEqual(got["head"]["small"], 1 / 6, places=12)
        self.assertAlmostEqual(got["tail"]["small"], 3 / 6, places=12)
        self.assertEqual(got["tail"]["large"], 0.0)

    def test_sums_to_one(self):
        recs = [[1, 2, 3]]
        pop = {1: "head", 2: "mid", 3: "tail"}
        store = {1: "small", 2: "large", 3: "small"}
        got = exposure_by_cross(recs, pop, store)
        total = sum(v for row in got.values() for v in row.values())
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_all_keys_present_when_empty(self):
        got = exposure_by_cross([], {}, {})
        for pb in ("head", "mid", "tail"):
            for sb in ("small", "large", "unknown"):
                self.assertIn(sb, got[pb])
                self.assertEqual(got[pb][sb], 0.0)

    def test_unknown_store_preserved_as_own_bucket(self):
        # An item with unknown store must NOT be folded into small or large.
        recs = [[1, 2]]
        pop = {1: "head", 2: "head"}
        store = {1: "large", 2: "unknown"}
        got = exposure_by_cross(recs, pop, store)
        self.assertAlmostEqual(got["head"]["large"], 0.5, places=12)
        self.assertAlmostEqual(got["head"]["unknown"], 0.5, places=12)
        self.assertEqual(got["head"]["small"], 0.0)


class TestMeanRecommendedPopularity(unittest.TestCase):
    """Hand-computed means in raw interaction counts."""

    def test_known_mean(self):
        recs = [[1, 2], [3]]
        pop = {1: 10.0, 2: 20.0, 3: 30.0}
        self.assertAlmostEqual(mean_recommended_popularity(recs, pop), 20.0, places=12)

    def test_empty_is_zero(self):
        self.assertEqual(mean_recommended_popularity([], {}), 0.0)

    def test_missing_items_skipped(self):
        # item 2 absent -> mean over 10 and 30
        recs = [[1, 2, 3]]
        pop = {1: 10.0, 3: 30.0}
        self.assertAlmostEqual(mean_recommended_popularity(recs, pop), 20.0, places=12)

    def test_repeated_item_weighted_by_slot(self):
        # counted per slot: (5 + 5 + 15)/3
        recs = [[1, 2], [1]]
        pop = {1: 5.0, 2: 15.0}
        self.assertAlmostEqual(
            mean_recommended_popularity(recs, pop), 25.0 / 3, places=12
        )

