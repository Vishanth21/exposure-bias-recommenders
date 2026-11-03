"""Interactive CLI demonstration of exposure-bias recommendation & debiasing.

Usage:
    python scripts/demo_interactive.py
    python scripts/demo_interactive.py --user U_000000 --lambda 0.5
    python scripts/demo_interactive.py --results-dir results_b --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


def display_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    header_str = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    print(header_str)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        print(line)
    print(sep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live demo of the Recommender & Exposure Re-ranking System")
    parser.add_argument("--user", type=str, default="U_000000", help="User ID to view recommendations for")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5, choices=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0], help="Re-ranking penalty lambda")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory (results or results_b)")
    parser.add_argument("--items-path", type=Path, default=Path("data/processed/items.parquet"))
    parser.add_argument("--top-k", type=int, default=10, help="Number of items to display")
    args = parser.parse_args()

    items_path = args.items_path
    if not items_path.exists():
        print(f"Error: items metadata not found at {items_path}")
        return 1

    items_df = pd.read_parquet(items_path).set_index("itemId")

    lam_str = f"{args.lam:.2f}"
    rerank_path = args.results_dir / f"rerank_scores_lambda_{lam_str}.parquet"
    baseline_path = args.results_dir / "rerank_scores_lambda_0.00.parquet"
    if not baseline_path.exists():
        baseline_path = args.results_dir / "baseline_scores.parquet"

    if not rerank_path.exists():
        print(f"Error: rerank file {rerank_path} not found. Run scripts/04_rerank.py first.")
        return 1

    print("=" * 80)
    print(f"  EXPOSURE BIAS RECOMMENDER DEMO")
    print(f"  Directory: {args.results_dir}  |  User: {args.user}  |  Lambda (debiasing penalty): {args.lam}")
    print("=" * 80)

    # Load baseline vs debiased
    base_df = pd.read_parquet(baseline_path)
    base_user = base_df[base_df["userId"] == args.user].head(args.top_k)

    rerank_df = pd.read_parquet(rerank_path)
    rerank_user = rerank_df[rerank_df["userId"] == args.user].head(args.top_k)

    if base_user.empty or rerank_user.empty:
        available_users = base_df["userId"].unique()[:5]
        print(f"User {args.user} not found. Available sample users: {list(available_users)}")
        return 1

    print("\n--- BASELINE RECOMMENDATIONS (lambda = 0.0, unconstrained ALS) ---")
    rows = []
    for rank, row in enumerate(base_user.itertuples(), start=1):
        item_meta = items_df.loc[row.itemId] if row.itemId in items_df.index else None
        title = item_meta["title"][:35] if item_meta is not None else row.itemId
        store = item_meta["store"][:18] if item_meta is not None and item_meta["store"] else "Unknown"
        store_bucket = item_meta["store_bucket"] if item_meta is not None else "unknown"
        pop_bucket = item_meta["pop_bucket"] if item_meta is not None else "unknown"
        score_val = f"{row.score:.4f}"
        rows.append([str(rank), str(row.itemId), title, store, store_bucket, pop_bucket, score_val])
    display_table(["Rank", "Item ID", "Title", "Seller/Store", "Store Size", "Pop Tier", "Score"], rows)

    base_pop_counts = {"head": 0, "mid": 0, "tail": 0}
    base_store_counts = {"large": 0, "small": 0, "unknown": 0}
    for row in rows:
        base_store_counts[row[4]] = base_store_counts.get(row[4], 0) + 1
        base_pop_counts[row[5]] = base_pop_counts.get(row[5], 0) + 1

    print(f"Baseline Top-{args.top_k} Composition: Head={base_pop_counts.get('head',0)}, Mid={base_pop_counts.get('mid',0)}, Tail={base_pop_counts.get('tail',0)} | Large Sellers={base_store_counts.get('large',0)}, Small Sellers={base_store_counts.get('small',0)}")

    if args.lam > 0:
        print(f"\n--- DEBIASED RECOMMENDATIONS (lambda = {args.lam}) ---")
        rows = []
        for rank, row in enumerate(rerank_user.itertuples(), start=1):
            item_meta = items_df.loc[row.itemId] if row.itemId in items_df.index else None
            title = item_meta["title"][:35] if item_meta is not None else row.itemId
            store = item_meta["store"][:18] if item_meta is not None and item_meta["store"] else "Unknown"
            store_bucket = item_meta["store_bucket"] if item_meta is not None else "unknown"
            pop_bucket = item_meta["pop_bucket"] if item_meta is not None else "unknown"
            score_val = f"{row.score:.4f}"
            rows.append([str(rank), str(row.itemId), title, store, store_bucket, pop_bucket, score_val])
        display_table(["Rank", "Item ID", "Title", "Seller/Store", "Store Size", "Pop Tier", "Score"], rows)

        rerank_pop_counts = {"head": 0, "mid": 0, "tail": 0}
        rerank_store_counts = {"large": 0, "small": 0, "unknown": 0}
        for row in rows:
            rerank_store_counts[row[4]] = rerank_store_counts.get(row[4], 0) + 1
            rerank_pop_counts[row[5]] = rerank_pop_counts.get(row[5], 0) + 1

        print(f"Debiased Top-{args.top_k} Composition: Head={rerank_pop_counts.get('head',0)}, Mid={rerank_pop_counts.get('mid',0)}, Tail={rerank_pop_counts.get('tail',0)} | Large Sellers={rerank_store_counts.get('large',0)}, Small Sellers={rerank_store_counts.get('small',0)}")
        delta_tail = rerank_pop_counts.get('tail', 0) - base_pop_counts.get('tail', 0)
        delta_small = rerank_store_counts.get('small', 0) - base_store_counts.get('small', 0)
        print(f"\n[Shift with lambda={args.lam}]: Tail items shift: {delta_tail:+d} | Small seller items shift: {delta_small:+d}")

    print("\nTip: Try different lambdas (e.g. --lambda 1.0 or --lambda 2.0) or users (--user U_000001) to observe debiasing behavior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

