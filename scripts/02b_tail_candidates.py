"""Build a separate candidate set with a per-user quota of ALS-scored tail items."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.contracts.model_candidate_contract import EmbeddingArtefact

log = logging.getLogger("tail_candidates")
N_CANDIDATES = 200
BATCH_USERS = 250


def _top_positions(
    scores: np.ndarray,
    item_ids: np.ndarray,
    popularity_ranks: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or len(scores) == 0:
        return np.array([], dtype=np.int64)
    count = min(count, len(scores))
    if count == len(scores):
        positions = np.arange(len(scores))
    else:
        cutoff = np.partition(scores, -count)[-count]
        above = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)
        needed = count - len(above)
        tie_order = np.lexsort((item_ids[tied], popularity_ranks[tied]))
        positions = np.concatenate((above, tied[tie_order[:needed]]))
    order = np.lexsort((item_ids[positions], popularity_ranks[positions], -scores[positions]))
    return positions[order]


def build_tail_augmented_candidates(
    ratings_path: Path,
    items_path: Path,
    source_results: Path,
    results_dir: Path,
    tail_candidate_quota: int = 20,
) -> None:
    if not 0 < tail_candidate_quota < N_CANDIDATES:
        raise ValueError(f"tail candidate quota must be between 1 and {N_CANDIDATES - 1}")
    output_path = results_dir / "baseline_scores.parquet"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    partial_output_path = results_dir / "baseline_scores.parquet.partial"
    if partial_output_path.exists():
        partial_output_path.unlink()

    source_scores = source_results / "baseline_scores.parquet"
    source_manifest = source_results / "run_manifest.json"
    source_funnel = source_results / "prepare_funnel.json"
    source_factors = source_results / "model" / "als_factors.npz"
    for path in (source_scores, source_manifest, source_funnel, source_factors):
        if not path.exists():
            raise FileNotFoundError(f"Required source artefact is missing: {path}")

    results_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading ALS factors and catalogue metadata")
    factors = EmbeddingArtefact.load_npz(str(source_factors))
    items = pd.read_parquet(
        items_path, columns=["itemId", "pop_bucket", "popularity_rank"]
    )
    bucket_by_item = dict(zip(items["itemId"].astype(str), items["pop_bucket"]))
    rank_by_item = dict(zip(items["itemId"].astype(str), items["popularity_rank"]))

    item_ids = np.asarray(factors.item_ids, dtype=str)
    user_ids = np.asarray(factors.user_ids, dtype=str)
    tail_positions = np.fromiter(
        (i for i, item in enumerate(item_ids) if bucket_by_item.get(item) == "tail"),
        dtype=np.int64,
    )
    tail_item_ids = item_ids[tail_positions]
    tail_item_factors = factors.item_factors[tail_positions]
    tail_popularity_ranks = np.asarray(
        [rank_by_item[item] for item in tail_item_ids], dtype=np.int64
    )
    tail_position_by_item = {item: i for i, item in enumerate(tail_item_ids)}
    user_position = {user: i for i, user in enumerate(user_ids)}
    log.info("Tail catalogue: %s items; quota=%d", f"{len(tail_item_ids):,}", tail_candidate_quota)

    source_reader = pq.ParquetFile(source_scores)
    eval_users: list[str] = []
    previous_user: str | None = None
    for batch in source_reader.iter_batches(batch_size=1_000_000, columns=["userId"]):
        values = batch.column("userId").to_numpy(zero_copy_only=False)
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        for value in values[starts]:
            user = str(value)
            if user != previous_user:
                eval_users.append(user)
                previous_user = user
    eval_user_set = set(eval_users)
    log.info("Candidate users: %s", f"{len(eval_users):,}")

    ratings = pd.read_parquet(ratings_path, columns=["userId", "itemId", "split"])
    known = ratings[
        ratings["userId"].isin(eval_user_set) & (ratings["split"] != "test")
    ]
    user_seen = known.groupby("userId")["itemId"].agg(set).to_dict()
    del ratings, known
    log.info("Loaded seen-item sets for %s evaluation users", f"{len(user_seen):,}")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_quota = manifest.get("params", {}).get("tail_candidate_quota", 0)
    if source_quota:
        raise ValueError("Source candidates already use a tail quota")
    manifest["params"]["tail_candidate_quota"] = tail_candidate_quota
    manifest["params"]["candidate_policy"] = "global_top_n_minus_quota_plus_top_tail"
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "prepare_funnel.json").write_bytes(source_funnel.read_bytes())
    log.info("Saved B-run manifest and prepare funnel to %s", results_dir)

    output_schema = pa.schema(
        [
            ("userId", pa.string()),
            ("itemId", pa.string()),
            ("score", pa.float32()),
            ("rank", pa.int16()),
        ]
    )
    writer = pq.ParquetWriter(partial_output_path, output_schema)
    global_keep = N_CANDIDATES - tail_candidate_quota
    started = time.monotonic()
    completed = 0
    tail_rows_written = 0
    reader = pq.ParquetFile(source_scores)

    for batch in reader.iter_batches(
        batch_size=BATCH_USERS * N_CANDIDATES,
        columns=["userId", "itemId", "score", "rank"],
    ):
        user_values = batch.column("userId").to_numpy(zero_copy_only=False)
        item_values = batch.column("itemId").to_numpy(zero_copy_only=False)
        score_values = batch.column("score").to_numpy(zero_copy_only=False).astype(np.float32)
        rank_values = batch.column("rank").to_numpy(zero_copy_only=False)
        starts = np.r_[0, np.flatnonzero(user_values[1:] != user_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(user_values)]
        batch_users = [str(user_values[start]) for start in starts]
        factor_rows = [user_position[user] for user in batch_users]
        tail_scores = factors.user_factors[factor_rows] @ tail_item_factors.T

        out = {"userId": [], "itemId": [], "score": [], "rank": []}
        for batch_row, (start, end, user) in enumerate(zip(starts, ends, batch_users)):
            if int(end - start) != N_CANDIDATES:
                raise ValueError(
                    f"Expected {N_CANDIDATES} source rows for user {user}, got {end - start}"
                )
            if not np.array_equal(rank_values[start:end], np.arange(N_CANDIDATES)):
                raise ValueError(f"Source ranks are incomplete for user {user}")

            keep_end = start + global_keep
            kept_ids = item_values[start:keep_end].astype(str)
            kept_scores = score_values[start:keep_end]
            known_items = user_seen.get(user, set())
            excluded_items = known_items.union(kept_ids.tolist())
            excluded_tail_positions = {
                tail_position_by_item[item]
                for item in excluded_items
                if item in tail_position_by_item
            }
            overfetch = min(
                len(tail_item_ids), tail_candidate_quota + len(excluded_tail_positions)
            )
            selected_tail = _top_positions(
                tail_scores[batch_row],
                tail_item_ids,
                tail_popularity_ranks,
                overfetch,
            )
            selected_tail = [
                int(position)
                for position in selected_tail
                if int(position) not in excluded_tail_positions
            ][:tail_candidate_quota]

            candidate_ids = kept_ids.tolist()
            candidate_scores = kept_scores.tolist()
            candidate_popularity = [rank_by_item[item] for item in candidate_ids]
            selected_items = set(candidate_ids)
            for tail_position in selected_tail:
                item = str(tail_item_ids[tail_position])
                if item in selected_items:
                    continue
                candidate_ids.append(item)
                candidate_scores.append(float(tail_scores[batch_row, tail_position]))
                candidate_popularity.append(tail_popularity_ranks[tail_position])
                selected_items.add(item)

            fallback_start = keep_end
            for source_position in range(fallback_start, int(end)):
                if len(candidate_ids) >= N_CANDIDATES:
                    break
                item = str(item_values[source_position])
                if item not in selected_items:
                    candidate_ids.append(item)
                    candidate_scores.append(float(score_values[source_position]))
                    candidate_popularity.append(rank_by_item[item])
                    selected_items.add(item)

            if len(candidate_ids) != N_CANDIDATES:
                raise ValueError(
                    f"Expected {N_CANDIDATES} candidates for user {user}, got {len(candidate_ids)}"
                )
            if len(selected_items) != N_CANDIDATES:
                raise ValueError(f"Duplicate candidates found for user {user}")
            if selected_items.intersection(known_items):
                raise ValueError(f"Known items found in candidates for user {user}")

            candidate_ids_array = np.asarray(candidate_ids, dtype=str)
            candidate_scores_array = np.asarray(candidate_scores, dtype=np.float32)
            candidate_popularity_array = np.asarray(candidate_popularity, dtype=np.int64)
            order = np.lexsort(
                (candidate_ids_array, candidate_popularity_array, -candidate_scores_array)
            )
            ordered_ids = candidate_ids_array[order]
            ordered_scores = candidate_scores_array[order]
            if not np.array_equal(ordered_ids[:10], item_values[start:start + 10]):
                raise ValueError(f"Tail augmentation changed the original top 10 for user {user}")
            out["userId"].extend([user] * len(order))
            out["itemId"].extend(ordered_ids.tolist())
            out["score"].extend(ordered_scores.tolist())
            out["rank"].extend(range(len(order)))
            tail_rows_written += sum(
                bucket_by_item[item] == "tail" for item in ordered_ids
            )

        writer.write_table(pa.Table.from_pydict(out, schema=output_schema))
        completed += len(batch_users)
        if completed % 10_000 == 0 or completed == len(eval_users):
            elapsed = max(time.monotonic() - started, 1e-9)
            log.info(
                "tail candidate scoring: %s/%s users (%.1f%%), %.0f users/s, elapsed %.1f min",
                f"{completed:,}", f"{len(eval_users):,}",
                100 * completed / max(len(eval_users), 1), completed / elapsed, elapsed / 60,
            )

    writer.close()
    partial_output_path.replace(output_path)
    log.info("Saved tail-augmented candidates to %s", output_path)
    log.info(
        "Tail candidates written: %s total (%.2f per user on average)",
        f"{tail_rows_written:,}", tail_rows_written / max(completed, 1),
    )
    log.info("tail candidate augmentation complete in %.1f min", (time.monotonic() - started) / 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings-path", type=Path, default=Path("data/processed/ratings.parquet"))
    parser.add_argument("--items-path", type=Path, default=Path("data/processed/items.parquet"))
    parser.add_argument("--source-results", type=Path, default=Path("results"))
    parser.add_argument("--results-dir", type=Path, default=Path("results_b"))
    parser.add_argument("--tail-candidate-quota", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_tail_augmented_candidates(
        ratings_path=args.ratings_path,
        items_path=args.items_path,
        source_results=args.source_results,
        results_dir=args.results_dir,
        tail_candidate_quota=args.tail_candidate_quota,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
