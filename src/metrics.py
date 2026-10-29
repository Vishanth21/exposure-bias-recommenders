"""Ranking and exposure metrics for the exposure-bias study.

Signatures and semantics are frozen in ``report/INTERFACE.md`` §4.

* ``recs`` is ranked best-first; only the first ``k`` entries are scored.
* Relevance is binary.
* Empty/zero-denominator cases return ``0.0`` rather than raising.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "ndcg_at_k",
    "recall_at_k",
    "hitrate_at_k",
    "exposure_share",
    "store_exposure_share",
    "exposure_parity",
    "exposure_by_cross",
    "mean_recommended_popularity",
    "catalog_coverage",
    "gini",
]

_LOG2 = math.log2


def _top_k(recs: Sequence[int], k: int) -> list[int]:
    """Return the first ``k`` recommendations; a negative ``k`` yields ``[]``."""
    if k <= 0:
        return []
    return list(recs[:k])


def ndcg_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary NDCG@k.

    ``DCG = sum_{i=0..k-1} rel_i / log2(i + 2)``; ``IDCG`` is the same sum in the ideal
    ordering. ``0.0`` when nothing is relevant (NDCG undefined there).
    """
    rel_set = set(relevant)
    if not rel_set:
        return 0.0

    top = _top_k(recs, k)
    dcg = 0.0
    for i, item in enumerate(top):
        if item in rel_set:
            dcg += 1.0 / _LOG2(i + 2)

    n_ideal = min(len(rel_set), k)
    idcg = 0.0
    for j in range(n_ideal):
        idcg += 1.0 / _LOG2(j + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the user's relevant items that appear in the top ``k``."""
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    top = _top_k(recs, k)
    hits = sum(1 for item in top if item in rel_set)
    return hits / len(rel_set)


def hitrate_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """1.0 if at least one of the top ``k`` is relevant, else 0.0."""
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    top = _top_k(recs, k)
    return 1.0 if any(item in rel_set for item in top) else 0.0


def exposure_share(
    recs_all: Iterable[Sequence[int]],
    pop_bucket_of: Mapping[int, str],
    buckets: Sequence[str] = ("head", "mid", "tail"),
) -> dict[str, float]:
    """Share of recommendation slots per popularity bucket.

    Counts every slot as passed — no truncation; that is the caller's choice. Items
    absent from ``pop_bucket_of`` are skipped, so this never raises on a partial
    mapping. Unknown labels are counted too, so shares still sum to 1.0; the result may
    therefore carry keys beyond ``buckets``.
    """
    counts: dict[str, int] = {b: 0 for b in buckets}
    total = 0

    for recs in recs_all:
        for item in recs:
            bucket = pop_bucket_of.get(item)
            if bucket is None:
                continue
            if bucket not in counts:
                counts[bucket] = 0
            counts[bucket] += 1
            total += 1

    if total == 0:
        return {b: 0.0 for b in counts}
    return {b: c / total for b, c in counts.items()}


def store_exposure_share(
    recs_all: Iterable[Sequence[int]],
    store_bucket_of: Mapping[int, str],
) -> dict[str, float]:
    """Share of recommendation slots per store bucket.

    ``unknown`` is always present: Amazon's ``store`` field is sometimes missing, and
    folding those items into ``small`` or ``large`` would bias the headline number.
    """
    shares = exposure_share(
        recs_all, store_bucket_of, buckets=("small", "large", "unknown")
    )
    return {b: shares.get(b, 0.0) for b in ("small", "large", "unknown")}


def catalog_coverage(recs_all: Iterable[Sequence[int]], catalog_size: int) -> float:
    """Fraction of the catalogue that appears in at least one recommendation list."""
    if catalog_size <= 0:
        return 0.0
    seen: set[int] = set()
    for recs in recs_all:
        seen.update(recs)
    return len(seen) / catalog_size


def gini(values: Iterable[float]) -> float:
    """Gini coefficient of non-negative values.

        G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n

    ``x`` ascending, ``i`` 1-based. ``0.0`` for empty or all-zero input; clamped to
    [0, 1] to absorb float noise.
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0

    total = sum(xs)
    if total <= 0.0:
        return 0.0

    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    g = (2.0 * weighted) / (n * total) - (n + 1) / n

    if g < 0.0:
        return 0.0
    if g > 1.0:
        return 1.0
    return g


def exposure_parity(
    exposure: Mapping[str, float],
    item_share: Mapping[str, float],
    buckets: Sequence[str] = ("head", "mid", "tail"),
) -> dict[str, float]:
    """Exposure share divided by catalogue share, per bucket.

    ``1.0`` is proportional; a bucket holding 2% of items and taking 5% of slots has
    parity 2.5. Zero catalogue share yields ``0.0`` — an empty bucket is a data
    property, not an error.
    """
    out: dict[str, float] = {}
    for b in buckets:
        share = item_share.get(b, 0.0)
        out[b] = 0.0 if share == 0.0 else exposure.get(b, 0.0) / share
    return out


def exposure_by_cross(
    recs_all: Iterable[Sequence[int]],
    pop_bucket_of: Mapping[int, str],
    store_bucket_of: Mapping[int, str],
    buckets: Sequence[str] = ("head", "mid", "tail"),
    store_buckets: Sequence[str] = ("small", "large", "unknown"),
) -> dict[str, dict[str, float]]:
    """Slot shares over the joint (popularity x store) partition.

    The control that separates a seller effect from a popularity effect: if large stores
    are over-exposed *within* the head bucket alone, the seller claim is not just
    restating popularity. Returns ``{pop_bucket: {store_bucket: fraction}}``; the whole
    structure sums to 1.0, and each row sums to that popularity bucket's share. Items
    missing from either mapping are skipped, as in :func:`exposure_share`.
    """
    counts: dict[str, dict[str, int]] = {
        pb: {sb: 0 for sb in store_buckets} for pb in buckets
    }
    total = 0

    for recs in recs_all:
        for item in recs:
            pb = pop_bucket_of.get(item)
            sb = store_bucket_of.get(item)
            if pb is None or sb is None:
                continue
            if pb not in counts:
                counts[pb] = {x: 0 for x in store_buckets}
            if sb not in counts[pb]:
                counts[pb][sb] = 0
            counts[pb][sb] += 1
            total += 1

    if total == 0:
        return {pb: {sb: 0.0 for sb in row} for pb, row in counts.items()}
    return {pb: {sb: c / total for sb, c in row.items()} for pb, row in counts.items()}


def mean_recommended_popularity(
    recs_all: Iterable[Sequence[int]],
    popularity_of: Mapping[int, float],
) -> float:
    """Mean popularity of every recommended slot, in raw interaction counts.

    Raw counts rather than bucket labels, so the number reads without a bucket rule:
    "the average slot went to an item with 412 training interactions". Items absent from
    ``popularity_of`` are skipped; ``0.0`` when no slots are counted.
    """
    total = 0.0
    n = 0
    for recs in recs_all:
        for item in recs:
            pop = popularity_of.get(item)
            if pop is None:
                continue
            total += pop
            n += 1
    return 0.0 if n == 0 else total / n

