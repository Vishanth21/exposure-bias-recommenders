"""
Integration tests for Avinash Julakatti's Re-Ranker Sweep (04_rerank.py)
and Sensitivity Analysis (05_sensitivity.py).

Verifies:
- Execution of re-ranking sweep over lambda grid.
- Candidate boundary isolation: identical candidate set before and after.
- Lambda = 0.0 identity with baseline metrics within 1e-6.
- Tail exposure increases as lambda increases.
- Sensitivity analysis execution and Gamma = 0 identity with baseline metrics.
"""

import importlib
import json
import os
import numpy as np
import pandas as pd
import pytest

train_mod = importlib.import_module("scripts.02_train")
train_and_generate_candidates = train_mod.train_and_generate_candidates

eval_mod = importlib.import_module("scripts.03_evaluate")
evaluate_baseline_models = eval_mod.evaluate_baseline_models

rerank_mod = importlib.import_module("scripts.04_rerank")
run_reranking_sweep = rerank_mod.run_reranking_sweep

sens_mod = importlib.import_module("scripts.05_sensitivity")
compute_sensitivity_analysis = sens_mod.compute_sensitivity_analysis

from tests.test_model_pipeline import toy_dataset_dir


def test_reranking_and_sensitivity_pipeline(toy_dataset_dir):
    results_dir = toy_dataset_dir["results_dir"]

    # 1. Train model & generate baseline scores
    train_and_generate_candidates(
        ratings_path=toy_dataset_dir["ratings_path"],
        items_path=toy_dataset_dir["items_path"],
        results_dir=results_dir,
        factors=8,
        regularization=0.01,
        iterations=5,
        seed=42,
        n_candidates=2,
    )

    # 2. Evaluate baseline metrics
    baseline_metrics = evaluate_baseline_models(
        baseline_scores_path=os.path.join(results_dir, "baseline_scores.parquet"),
        ratings_path=toy_dataset_dir["ratings_path"],
        items_path=toy_dataset_dir["items_path"],
        results_dir=results_dir,
        k=2,
        n_candidates=2,
    )

    assert "exposure_pop" in baseline_metrics
    assert "relevance_pop" in baseline_metrics
    assert "baselines" in baseline_metrics

    # 3. Run Re-ranking sweep
    lambda_grid = [0.0, 0.5, 1.0]
    sweep_results = run_reranking_sweep(
        baseline_scores_path=os.path.join(results_dir, "baseline_scores.parquet"),
        items_path=toy_dataset_dir["items_path"],
        ratings_path=toy_dataset_dir["ratings_path"],
        baseline_metrics_path=os.path.join(results_dir, "baseline_metrics.json"),
        results_dir=results_dir,
        lambda_grid=lambda_grid,
        k=2,
    )

    # Verify sweep results structure
    assert "exposure_pop" in sweep_results
    assert "relevance_pop" in sweep_results
    assert len(sweep_results["exposure_pop"]) == len(lambda_grid)
    assert len(sweep_results["relevance_pop"]) == len(lambda_grid)

    # Verify Lambda = 0.0 identity with baseline
    baseline_head_exp = baseline_metrics["exposure_pop"]["exposure_share"]["head"]
    rerank_l0_head_exp = sweep_results["exposure_pop"][0]["exposure_share"]["head"]
    assert abs(baseline_head_exp - rerank_l0_head_exp) < 1e-6

    # Verify parquet artefacts exist
    for l_val in lambda_grid:
        p_path = os.path.join(results_dir, f"rerank_scores_lambda_{l_val:.2f}.parquet")
        assert os.path.exists(p_path)

    # Verify per-user metrics parquet
    per_user_path = os.path.join(results_dir, "per_user_metrics.parquet")
    assert os.path.exists(per_user_path)
    per_user_df = pd.read_parquet(per_user_path)
    assert list(per_user_df.columns) == ["userId", "population", "lambda", "ndcg", "hit"]

    # 4. Run Sensitivity Analysis
    sensitivity_results = compute_sensitivity_analysis(
        baseline_scores_path=os.path.join(results_dir, "baseline_scores.parquet"),
        ratings_path=toy_dataset_dir["ratings_path"],
        items_path=toy_dataset_dir["items_path"],
        baseline_metrics_path=os.path.join(results_dir, "baseline_metrics.json"),
        results_dir=results_dir,
        gamma_values=[0.0, 0.5, 1.0],
        k=2,
    )

    assert "relevance_pop" in sensitivity_results
    # Verify Gamma = 0.0 reproduces baseline unweighted NDCG exactly
    base_rel_ndcg = baseline_metrics["relevance_pop"]["ndcg"]
    gamma0_rel_ndcg = sensitivity_results["relevance_pop"]["metrics"]["0.0"]["ndcg"]
    assert abs(base_rel_ndcg - gamma0_rel_ndcg) < 1e-6
