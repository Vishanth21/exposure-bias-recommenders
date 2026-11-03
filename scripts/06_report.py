"""Summarise pipeline result artefacts and write report figures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required result file is missing: {path}")
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_lambda_zero_match(
    baseline: dict[str, Any], rerank: dict[str, Any]
) -> None:
    lambda_zero = next(
        (row for row in rerank["exposure_pop"] if math.isclose(row["lambda"], 0.0)),
        None,
    )
    relevance_zero = next(
        (row for row in rerank["relevance_pop"] if math.isclose(row["lambda"], 0.0)),
        None,
    )
    if lambda_zero is None or relevance_zero is None:
        raise ValueError("rerank_sweep.json must contain lambda=0.0 for both populations")

    pairs = (
        (baseline["exposure_pop"], lambda_zero),
        (baseline["relevance_pop"], relevance_zero),
    )
    for baseline_block, sweep_block in pairs:
        for key, expected in baseline_block.items():
            if key not in sweep_block:
                continue
            actual = sweep_block[key]
            if isinstance(expected, dict):
                for child, expected_value in expected.items():
                    actual_value = actual.get(child)
                    if isinstance(expected_value, dict):
                        for nested, nested_expected in expected_value.items():
                            nested_actual = actual_value.get(nested)
                            if not math.isclose(
                                float(nested_expected), float(nested_actual), abs_tol=1e-6
                            ):
                                raise ValueError(
                                    f"lambda=0 mismatch at {key}.{child}.{nested}: "
                                    f"{nested_actual} != {nested_expected}"
                                )
                    elif not math.isclose(
                        float(expected_value), float(actual_value), abs_tol=1e-6
                    ):
                        raise ValueError(
                            f"lambda=0 mismatch at {key}.{child}: "
                            f"{actual_value} != {expected_value}"
                        )
            elif isinstance(expected, (int, float)) and not math.isclose(
                float(expected), float(actual), abs_tol=1e-6
            ):
                raise ValueError(
                    f"lambda=0 mismatch at {key}: {actual} != {expected}"
                )


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n### {title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _lambda_axis(ax, lambdas: list[float]) -> list[int]:
    """Evenly spaced λ positions: the grid is non-uniform, so real spacing crushes 0 and 0.1."""
    positions = list(range(len(lambdas)))
    ax.set_xticks(positions, [f"{value:g}" for value in lambdas])
    ax.set_xlabel("λ")
    return positions


def _format_metric(value: float) -> str:
    return f"{value:.4f}" if value >= 1e-3 else f"{value:.1e}"


def plot_baseline_quality(baseline: dict[str, Any], output: Path) -> None:
    names = ["ALS", "MostPop", "Random"]
    metrics = {name: baseline["relevance_pop"] for name in names[:1]}
    metrics.update(
        {
            "MostPop": baseline["baselines"]["mostpop"]["relevance_pop"],
            "Random": baseline["baselines"]["random"]["relevance_pop"],
        }
    )
    ndcg = [float(metrics[name]["ndcg"]) for name in names]
    hitrate = [float(metrics[name]["hitrate"]) for name in names]

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    x = list(range(len(names)))
    width = 0.36
    # Scores are ~0.01 on a full-catalogue ranking, so the axis is scaled to the data
    # (a 0-1 axis renders every bar as empty) and each bar carries its value.
    for offset, values, label in ((-width / 2, ndcg, "NDCG@10"), (width / 2, hitrate, "HitRate@10")):
        bars = ax.bar([position + offset for position in x], values, width, label=label)
        ax.bar_label(bars, labels=[_format_metric(value) for value in values], padding=3, fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("Score (full ranking over the catalogue)")
    ax.set_ylim(0, max(hitrate) * 1.2)
    ax.set_title("Baseline relevance quality (relevance population)")
    ax.legend(loc="upper right")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_reranker_tradeoff(rerank: dict[str, Any], chosen: float | None, output: Path) -> None:
    exposure = sorted(rerank["exposure_pop"], key=lambda row: row["lambda"])
    relevance = sorted(rerank["relevance_pop"], key=lambda row: row["lambda"])
    lambdas = [float(row["lambda"]) for row in exposure]
    popularity = [float(row["mean_recommended_popularity"]) for row in exposure]
    tail = [float(row["exposure_share"]["tail"]) for row in exposure]
    ndcg = [float(row["ndcg"]) for row in relevance]

    fig, left = plt.subplots(figsize=(8, 4.8), layout="constrained")
    right = left.twinx()
    x = _lambda_axis(left, lambdas)
    # Tail share is identically 0 across the sweep, so it is stated as text rather than
    # drawn as a flat line on an axis centred on zero, which reads as a broken plot.
    left.plot(x, popularity, marker="o", color="tab:blue", label="Mean recommended popularity")
    right.plot(x, ndcg, marker="s", color="tab:orange", label="NDCG@10")
    if chosen is not None and chosen in lambdas:
        left.axvline(lambdas.index(chosen), color="0.4", linestyle="--", linewidth=1, label="Chosen λ")
    left.set_ylim(0, max(popularity) * 1.1)
    left.set_ylabel("Mean train interactions per recommended item")
    right.set_ylabel("NDCG@10")
    left.set_title("Re-ranking trade-off across the full λ sweep")
    tail_note = "Tail exposure share = 0 at every λ" if max(tail) == 0 else f"Max tail share {max(tail):.2%}"
    left.text(0.02, 0.04, tail_note, transform=left.transAxes, fontsize=9, color="0.3")
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="lower left", bbox_to_anchor=(0, 0.1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_popularity_exposure(
    rerank: dict[str, Any], funnel: dict[str, Any], output: Path
) -> None:
    rows = sorted(rerank["exposure_pop"], key=lambda row: row["lambda"])
    lambdas = [float(row["lambda"]) for row in rows]
    bucket_names = ("head", "mid", "tail")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, bucket in zip(axes, bucket_names):
        x = _lambda_axis(ax, lambdas)
        values = [float(row["exposure_share"][bucket]) for row in rows]
        item_share = float(funnel["buckets"][bucket]["item_share"])
        ax.plot(x, values, marker="o", label="Exposure share")
        ax.axhline(item_share, color="0.45", linestyle="--", label="Catalogue item share")
        ax.set_title(f"{bucket.capitalize()} (λ=2: {values[-1]:.2%} of slots)")
    axes[0].set_ylabel("Share of recommendation slots")
    axes[-1].legend(loc="center right", fontsize="small")
    fig.suptitle("Exposure by popularity bucket")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_store_exposure(
    rerank: dict[str, Any], funnel: dict[str, Any], output: Path
) -> None:
    rows = sorted(rerank["exposure_pop"], key=lambda row: row["lambda"])
    lambdas = [float(row["lambda"]) for row in rows]
    colors = {"large": "tab:blue", "small": "tab:orange", "unknown": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    x = _lambda_axis(ax, lambdas)
    for bucket, color in colors.items():
        values = [float(row["store_exposure_share"][bucket]) for row in rows]
        item_share = float(funnel["store_buckets"][bucket]["item_share"])
        ax.plot(x, values, marker="o", color=color, label=f"{bucket} exposure")
        ax.axhline(item_share, color=color, linestyle="--", linewidth=1, label=f"{bucket} catalogue share")
    ax.set_ylabel("Share of recommendation slots")
    ax.set_title("Seller-bucket exposure and catalogue share")
    ax.legend(loc="center right", fontsize="small")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def gamma_key(value: float) -> str:
    return "0.0" if math.isclose(value, 0.0) else f"{value:.2f}"


def plot_sensitivity(sensitivity: dict[str, Any], output: Path) -> None:
    block = sensitivity["relevance_pop"]
    gammas = [float(value) for value in block["gamma_values"]]
    metrics = [block["metrics"][gamma_key(gamma)] for gamma in gammas]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(gammas, [float(row["ndcg"]) for row in metrics], marker="o", label="NDCG@10")
    ax.plot(gammas, [float(row["hitrate"]) for row in metrics], marker="s", label="HitRate@10")
    ax.set_xlabel("γ")
    ax.set_ylabel("Weighted metric")
    ax.set_title("Popularity-weighted relevance sensitivity")
    ax.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def print_summary(
    baseline: dict[str, Any],
    rerank: dict[str, Any],
    funnel: dict[str, Any],
    sensitivity: dict[str, Any] | None,
) -> None:
    als = baseline["relevance_pop"]
    mostpop = baseline["baselines"]["mostpop"]["relevance_pop"]
    random = baseline["baselines"]["random"]["relevance_pop"]
    gate = float(als["ndcg"]) > float(mostpop["ndcg"])

    print(f"Category: {funnel['category']}")
    print(
        "Catalogue: {:,} items; positive fraction: {:.3f}; test items outside catalogue: {:.3f}".format(
            int(funnel["stages"][4]["items"]),
            float(funnel["positive_fraction"]),
            float(funnel["test_items_outside_catalogue_frac"]),
        )
    )
    print(
        "Store metadata missing: {:.3f}; large stores: {:,} ({:.3f} of stores, {:.3f} of items)".format(
            float(funnel["store_null_rate"]),
            int(funnel["store_buckets"]["large"]["n_stores"]),
            float(funnel["store_buckets"]["large"]["store_share"]),
            float(funnel["store_buckets"]["large"]["item_share"]),
        )
    )
    print(
        "ALS exceeds MostPop on relevance NDCG@10: {} ({:.6f} vs {:.6f})".format(
            "yes" if gate else "no", float(als["ndcg"]), float(mostpop["ndcg"])
        )
    )
    if gate:
        print(
            "Observed-interaction relevance cost is an upper bound because held-out labels "
            "reflect popularity-biased exposure."
        )
    else:
        print("ALS does not exceed MostPop on relevance NDCG@10; there is no model edge to spend.")

    print_table(
        "Baseline relevance quality",
        ["Model", "Users", "NDCG@10", "HitRate@10"],
        [
            [
                name,
                f"{int(als['n_users']):,}",
                f"{float(block['ndcg']):.6f}",
                f"{float(block['hitrate']):.6f}",
            ]
            for name, block in (
                ("ALS", als),
                ("MostPop", mostpop),
                ("Random", random),
            )
        ],
    )

    headers = ["λ", "Tail exposure", "NDCG@10", "HitRate@10"]
    if gate:
        headers.append("ΔNDCG / ALS–Random edge")
    rows = []
    relevance_by_lambda = {float(row["lambda"]): row for row in rerank["relevance_pop"]}
    for row in sorted(rerank["exposure_pop"], key=lambda item: item["lambda"]):
        value = float(row["lambda"])
        rel = relevance_by_lambda[value]
        cells = [
            f"{value:g}",
            f"{float(row['exposure_share']['tail']):.6f}",
            f"{float(rel['ndcg']):.6f}",
            f"{float(rel['hitrate']):.6f}",
        ]
        if gate:
            cells.append(f"{float(rel['delta_ndcg_frac_of_als_minus_random']):.3f}")
        rows.append(cells)
    print_table("Re-ranking sweep", headers, rows)

    if sensitivity is not None:
        block = sensitivity["relevance_pop"]
        print_table(
            "Popularity-weighted relevance sensitivity",
            ["γ", "NDCG@10", "HitRate@10"],
            [
                [
                    f"{float(gamma):g}",
                    f"{float(block['metrics'][gamma_key(float(gamma))]['ndcg']):.6f}",
                    f"{float(block['metrics'][gamma_key(float(gamma))]['hitrate']):.6f}",
                ]
                for gamma in block["gamma_values"]
            ],
        )
    else:
        print("Sensitivity result absent; no sensitivity figure generated.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    baseline = load_json(args.results_dir / "baseline_metrics.json")
    rerank = load_json(args.results_dir / "rerank_sweep.json")
    funnel = load_json(args.results_dir / "prepare_funnel.json")
    sensitivity = load_json(args.results_dir / "sensitivity.json", required=False)
    assert baseline is not None and rerank is not None and funnel is not None
    require_lambda_zero_match(baseline, rerank)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        args.figures_dir / "baseline_relevance.png",
        args.figures_dir / "reranker_tradeoff.png",
        args.figures_dir / "popularity_exposure.png",
        args.figures_dir / "store_exposure.png",
    ]
    plot_baseline_quality(baseline, figure_paths[0])
    plot_reranker_tradeoff(
        rerank, rerank.get("chosen_lambda"), figure_paths[1]
    )
    plot_popularity_exposure(rerank, funnel, figure_paths[2])
    plot_store_exposure(rerank, funnel, figure_paths[3])
    if sensitivity is not None:
        figure_paths.append(args.figures_dir / "sensitivity.png")
        plot_sensitivity(sensitivity, figure_paths[-1])

    print_summary(baseline, rerank, funnel, sensitivity)
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
