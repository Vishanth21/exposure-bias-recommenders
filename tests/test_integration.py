"""
Integration tests — schema guards on every artefact boundary (report/INTERFACE.md §1–§3).

Owner: Rithesh (PES1UG24AM064)

Two layers, sharing the same validators:

1. **Offline slice.** A tiny gzip JSONL slice of Amazon-shaped rows is written to a temp
   ``data/raw/``, read through ``src/amazon_loader.py`` and pushed through
   ``scripts/01_prepare.py``. The produced ``ratings.parquet``, ``items.parquet`` and
   ``prepare_funnel.json`` are checked against the contract. Always runs; no network.
2. **Real artefacts.** Every file the cloud run produced under ``data/processed/`` and
   ``results/`` is checked against the same contract, plus the cross-file self-checks
   (``lambda = 0`` and ``gamma = 0`` reproduce the baseline). A test skips when its artefact
   is absent, so the suite is green on a fresh clone and strict on a results checkout.

Large parquet files are checked from their footer schema and first row group only.
"""

from __future__ import annotations

import gzip
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import amazon_loader

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PROCESSED = REPO_ROOT / "data" / "processed"

POP_BUCKETS = ("head", "mid", "tail")
STORE_BUCKETS = ("small", "large", "unknown")
SPLITS = {"train", "test", "seen"}
LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
FUNNEL_STAGES = [
    "raw_reviews",
    "positives_ge_4",
    "non_positives_seen",
    "after_split",
    "catalogue_ge_3",
    "eval_exposure_pop",
    "eval_relevance_pop",
]
SUM_TOL = 1e-6
SELF_CHECK_TOL = 1e-6

RATINGS_SCHEMA = {
    "userId": "string",
    "itemId": "string",
    "rating": "float32",
    "timestamp": "int64",
    "split": "string",
    "seen": "bool",
}
ITEMS_SCHEMA = {
    "itemId": "string",
    "title": "string",
    "store": "string",
    "main_category": "string",
    "price": "float32",
    "n_train_ratings": "int32",
    "popularity_rank": "int32",
    "pop_bucket": "string",
    "store_n_items": "int32",
    "store_n_train": "int32",
    "store_bucket": "string",
    "in_catalogue": "bool",
}
SCORES_SCHEMA = {"userId": "string", "itemId": "string", "score": "float32", "rank": "int16"}
PER_USER_SCHEMA = {
    "userId": "string",
    "population": "string",
    "lambda": "float64",
    "ndcg": "float64",
    "hit": "float64",
}


# --------------------------------------------------------------------------- helpers


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"artefact not present: {path.relative_to(REPO_ROOT)}")
    return path


def _load_json(path: Path) -> dict:
    with _require(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _arrow_kind(arrow_type) -> str:
    """Collapse arrow types to the names the contract uses."""
    name = str(arrow_type)
    if name in ("string", "large_string") or name.startswith("dictionary<values=string"):
        return "string"
    return {"float": "float32", "double": "float64"}.get(name, name)


def _pandas_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "string"
    return str(series.dtype)


def _assert_schema(actual: dict[str, str], expected: dict[str, str], name: str) -> None:
    missing = sorted(set(expected) - set(actual))
    assert not missing, f"{name}: missing columns {missing}"
    wrong = {
        col: (actual[col], want) for col, want in expected.items() if actual[col] != want
    }
    assert not wrong, f"{name}: dtype mismatch (actual, expected) {wrong}"


def _assert_parquet_schema(path: Path, expected: dict[str, str]) -> None:
    schema = pq.read_schema(path)
    actual = {field.name: _arrow_kind(field.type) for field in schema}
    _assert_schema(actual, expected, path.name)


def _first_row_group(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.ParquetFile(path).read_row_group(0, columns=columns).to_pandas()


def _assert_shares(shares: dict, keys, name: str, *, allow_all_zero: bool = False) -> None:
    for key in keys:
        assert key in shares, f"{name}: missing key {key!r}"
    values = [float(v) for v in shares.values()]
    assert all(0.0 <= v <= 1.0 + SUM_TOL for v in values), f"{name}: share out of [0,1] {shares}"
    total = sum(values)
    if allow_all_zero and total == 0.0:
        return
    assert math.isclose(total, 1.0, abs_tol=SUM_TOL), f"{name}: sums to {total}, not 1"


def _assert_close_tree(expected, actual, path: str) -> None:
    """Every numeric leaf in ``expected`` must match ``actual`` within the self-check tol."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, f"{path}.{key} missing from sweep row"
            _assert_close_tree(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(float(expected), float(actual), abs_tol=SELF_CHECK_TOL), (
            f"{path}: {actual} != {expected}"
        )


# --------------------------------------------------------------------------- validators
# Used by both the offline slice and the real artefacts, so the two cannot drift.


def validate_ratings(ratings: pd.DataFrame) -> None:
    _assert_schema(
        {c: _pandas_kind(ratings[c]) for c in ratings.columns}, RATINGS_SCHEMA, "ratings"
    )
    assert set(ratings["split"].unique()) <= SPLITS
    assert ratings["seen"].all(), "every rated row must carry seen=True"
    assert ratings["rating"].between(1.0, 5.0).all()
    assert (ratings["timestamp"] > amazon_loader.TIMESTAMP_MS_THRESHOLD).all(), (
        "timestamps must be epoch milliseconds"
    )

    positive = ratings["rating"] >= 4.0
    assert (ratings.loc[positive, "split"] != "seen").all(), "positives must be train/test"
    assert (ratings.loc[~positive, "split"] == "seen").all(), "non-positives must be seen"

    tests = ratings[ratings["split"] == "test"]
    assert not tests["userId"].duplicated().any(), "more than one test row for a user"
    users_with_positive = set(ratings.loc[positive, "userId"])
    assert set(tests["userId"]) == users_with_positive, "every positive user needs a test row"

    # Temporal leave-last-out: no train positive is later than the user's test positive.
    train = ratings[ratings["split"] == "train"]
    last_train = train.groupby("userId")["timestamp"].max()
    test_ts = tests.set_index("userId")["timestamp"]
    joined = pd.concat([last_train.rename("train"), test_ts.rename("test")], axis=1).dropna()
    assert (joined["train"] <= joined["test"]).all(), "train positive after the test positive"


def validate_items(items: pd.DataFrame, min_catalogue: int = 3) -> None:
    _assert_schema({c: _pandas_kind(items[c]) for c in items.columns}, ITEMS_SCHEMA, "items")
    assert not items["itemId"].duplicated().any(), "items.parquet must have one row per item"
    assert set(items["pop_bucket"].unique()) <= set(POP_BUCKETS)
    assert set(items["store_bucket"].unique()) <= set(STORE_BUCKETS)
    assert (items["in_catalogue"] == (items["n_train_ratings"] >= min_catalogue)).all()

    # store normalisation: "" <=> unknown, and the literal "None" never survives the loader
    assert not items["store"].isin(["None", "nan", "null"]).any()
    assert ((items["store"] == "") == (items["store_bucket"] == "unknown")).all()

    catalogue = items[items["in_catalogue"]]
    if len(catalogue):
        ranks = catalogue["popularity_rank"].sort_values().to_numpy()
        assert (ranks == np.arange(len(catalogue))).all(), "catalogue rank must be dense 0..n-1"
        # Head is the most popular end: no tail item outranks a head item.
        head_min = catalogue.loc[catalogue["pop_bucket"] == "head", "n_train_ratings"].min()
        tail_max = catalogue.loc[catalogue["pop_bucket"] == "tail", "n_train_ratings"].max()
        if not (pd.isna(head_min) or pd.isna(tail_max)):
            assert head_min >= tail_max
        known = catalogue[catalogue["store"] != ""]
        per_store = known.groupby("store")["store_n_items"].first()
        assert (per_store == known.groupby("store").size()).all(), "store_n_items mismatch"


def validate_funnel(funnel: dict) -> None:
    assert isinstance(funnel["category"], str) and funnel["category"]
    assert [s["stage"] for s in funnel["stages"]] == FUNNEL_STAGES
    for stage in funnel["stages"]:
        for key in ("users", "items", "interactions"):
            assert isinstance(stage[key], int) and stage[key] >= 0, (stage["stage"], key)
    by_name = {s["stage"]: s for s in funnel["stages"]}
    for name in ("eval_exposure_pop", "eval_relevance_pop"):
        stage = by_name[name]
        assert 0 < stage["sampled_users"] <= min(stage["users"], 300_000)
    assert by_name["eval_relevance_pop"]["users"] <= by_name["eval_exposure_pop"]["users"]
    assert by_name["positives_ge_4"]["interactions"] + by_name["non_positives_seen"][
        "interactions"
    ] == by_name["raw_reviews"]["interactions"], "positives + seen must equal raw reviews"

    for key in ("positive_fraction", "store_null_rate", "test_items_outside_catalogue_frac"):
        assert 0.0 <= float(funnel[key]) <= 1.0, key

    buckets = funnel["buckets"]
    assert set(buckets) == set(POP_BUCKETS)
    _assert_shares({b: buckets[b]["item_share"] for b in POP_BUCKETS}, POP_BUCKETS, "item_share")
    _assert_shares(
        {b: buckets[b]["interaction_share"] for b in POP_BUCKETS},
        POP_BUCKETS,
        "interaction_share",
    )
    stores = funnel["store_buckets"]
    assert set(stores) == set(STORE_BUCKETS)
    for bucket in STORE_BUCKETS:
        assert {"n_stores", "store_share", "item_share"} <= set(stores[bucket])
    _assert_shares(
        {b: stores[b]["item_share"] for b in STORE_BUCKETS}, STORE_BUCKETS, "store item_share"
    )


def validate_exposure_block(block: dict, name: str, *, cross: bool = True) -> None:
    assert int(block["n_users"]) > 0, name
    _assert_shares(block["exposure_share"], POP_BUCKETS, f"{name}.exposure_share")
    _assert_shares(block["store_exposure_share"], STORE_BUCKETS, f"{name}.store_exposure_share")
    assert 0.0 <= float(block["coverage"]) <= 1.0
    assert 0.0 <= float(block["gini_popularity"]) <= 1.0
    if cross and "exposure_share_by_pop" in block:
        by_pop = block["exposure_share_by_pop"]
        total = sum(float(v) for row in by_pop.values() for v in row.values())
        assert math.isclose(total, 1.0, abs_tol=SUM_TOL), f"{name}: cross-tab sums to {total}"
        for bucket in POP_BUCKETS:
            row_sum = sum(float(v) for v in by_pop[bucket].values())
            assert math.isclose(row_sum, float(block["exposure_share"][bucket]), abs_tol=SUM_TOL), (
                f"{name}: cross-tab row {bucket} != exposure_share"
            )


# --------------------------------------------------------------------------- offline slice


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _slice_rows(seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Amazon-shaped review + metadata rows covering every normalisation edge case."""
    rng = np.random.default_rng(seed)
    n_items, n_users = 40, 120
    items = [f"B{j:09d}" for j in range(n_items)]
    weights = 1.0 / np.arange(1, n_items + 1) ** 0.9
    weights /= weights.sum()

    reviews, seen_pairs = [], set()
    base_ts = 1_600_000_000_000
    for u in range(n_users):
        for _ in range(int(rng.integers(1, 9))):
            item = items[int(rng.choice(n_items, p=weights))]
            if (u, item) in seen_pairs:
                continue
            seen_pairs.add((u, item))
            reviews.append(
                {
                    "user_id": f"AU{u:05d}",
                    "parent_asin": item,
                    "asin": item + "V",  # variant asin, must be ignored
                    "rating": float(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.1, 0.3, 0.5])),
                    "timestamp": int(base_ts + rng.integers(0, 10**10)),
                    "text": "ok",
                }
            )

    odd_stores = [None, "None", "", "  "]  # every form the card warns about
    metadata = []
    for j, item in enumerate(items):
        row = {
            "parent_asin": item,
            "title": f"Cable {j}" if j % 7 else None,
            "main_category": "All Electronics",
            "price": "None" if j % 3 == 0 else f"{10 + j}.99",
            "store": odd_stores[j % 4] if j % 5 == 0 else f"Store{j % 6}",
        }
        if j == 1:
            del row["store"]  # missing key entirely
        metadata.append(row)
    return reviews, metadata


@pytest.fixture
def raw_slice(tmp_path, monkeypatch):
    reviews, metadata = _slice_rows()
    _write_jsonl_gz(tmp_path / "data" / "raw" / "Electronics.jsonl.gz", reviews)
    _write_jsonl_gz(tmp_path / "data" / "raw" / "meta_Electronics.jsonl.gz", metadata)
    monkeypatch.chdir(tmp_path)  # the loader resolves data/raw relative to cwd
    return {"root": tmp_path, "reviews": reviews, "metadata": metadata}


def test_loader_reviews_contract(raw_slice):
    df = amazon_loader.load_reviews("Electronics")
    assert {"user_id", "parent_asin", "rating", "timestamp"} <= set(df.columns)
    assert df["rating"].dtype == np.float32
    assert df["timestamp"].dtype == np.int64
    assert pd.api.types.is_object_dtype(df["user_id"]) or pd.api.types.is_string_dtype(df["user_id"])
    assert len(df) == len(raw_slice["reviews"])


def test_loader_limit_applies_before_filtering(raw_slice):
    df = amazon_loader.load_reviews("Electronics", limit=25)
    assert len(df) == 25
    expected = [r["user_id"] for r in raw_slice["reviews"][:25]]
    assert df["user_id"].tolist() == expected, "limit must truncate in file order"


def test_loader_metadata_normalisation(raw_slice):
    meta = amazon_loader.load_metadata("Electronics")
    assert {"parent_asin", "title", "store", "main_category", "price"} <= set(meta.columns)
    assert meta["price"].dtype == np.float32
    assert not meta["store"].isin(["None", "nan", "null"]).any()
    assert (meta["store"] == meta["store"].str.strip()).all()
    assert not meta["title"].isin(["None", "nan"]).any(), "missing title must become ''"

    by_id = meta.set_index("parent_asin")
    items = [m["parent_asin"] for m in raw_slice["metadata"]]
    assert by_id.loc[items[0], "store"] == ""  # None
    assert by_id.loc[items[1], "store"] == ""  # key missing
    assert by_id.loc[items[5], "store"] == ""  # "None"
    assert by_id.loc[items[10], "store"] == ""  # ""
    assert by_id.loc[items[15], "store"] == ""  # whitespace only
    assert np.isnan(by_id.loc[items[0], "price"])  # "None" price
    assert math.isclose(float(by_id.loc[items[2], "price"]), 12.99, rel_tol=1e-6)
    assert by_id.loc[items[0], "title"] == ""


def test_extract_store_matches_loader(raw_slice):
    meta = amazon_loader.load_metadata("Electronics")
    stores = amazon_loader.extract_store(meta)
    assert stores.index.name == "parent_asin"
    assert (stores.values == meta["store"].values).all()


def test_loader_rejects_unknown_category():
    with pytest.raises(ValueError, match="Not_A_Category"):
        amazon_loader.load_reviews("Not_A_Category")
    with pytest.raises(ValueError, match="Not_A_Category"):
        amazon_loader.load_metadata("Not_A_Category")


def test_loader_rejects_second_timestamps(tmp_path, monkeypatch):
    rows = [
        {"user_id": "u", "parent_asin": f"B{i}", "rating": 5.0, "timestamp": 1_600_000_000 + i}
        for i in range(5)
    ]
    _write_jsonl_gz(tmp_path / "data" / "raw" / "Electronics.jsonl.gz", rows)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="milliseconds"):
        amazon_loader.load_reviews("Electronics")


@pytest.fixture
def prepared_slice(raw_slice, monkeypatch):
    prepare = importlib.import_module("scripts.01_prepare")
    processed = raw_slice["root"] / "data" / "processed"
    results = raw_slice["root"] / "results"
    monkeypatch.setattr(prepare, "PROCESSED", processed)
    monkeypatch.setattr(prepare, "RESULTS", results)
    assert prepare.main(["--category", "Electronics"]) == 0
    return {
        "ratings": pd.read_parquet(processed / "ratings.parquet"),
        "items": pd.read_parquet(processed / "items.parquet"),
        "funnel": json.loads((results / "prepare_funnel.json").read_text(encoding="utf-8")),
        "processed": processed,
        "raw": raw_slice,
    }


def test_prepare_ratings_boundary(prepared_slice):
    validate_ratings(prepared_slice["ratings"])
    _assert_parquet_schema(prepared_slice["processed"] / "ratings.parquet", RATINGS_SCHEMA)
    # Nothing is dropped at prepare time.
    assert len(prepared_slice["ratings"]) == len(prepared_slice["raw"]["reviews"])


def test_prepare_items_boundary(prepared_slice):
    items = prepared_slice["items"]
    validate_items(items)
    _assert_parquet_schema(prepared_slice["processed"] / "items.parquet", ITEMS_SCHEMA)
    assert set(items["itemId"]) == set(prepared_slice["ratings"]["itemId"])


def test_prepare_funnel_boundary(prepared_slice):
    funnel = prepared_slice["funnel"]
    validate_funnel(funnel)
    raw = prepared_slice["raw"]["reviews"]
    assert funnel["stages"][0]["interactions"] == len(raw)
    expected_pos = sum(r["rating"] >= 4 for r in raw) / len(raw)
    assert math.isclose(funnel["positive_fraction"], expected_pos, rel_tol=1e-9)


# --------------------------------------------------------------------------- real artefacts


def test_real_ratings_parquet():
    path = _require(PROCESSED / "ratings.parquet")
    _assert_parquet_schema(path, RATINGS_SCHEMA)
    validate_ratings(pd.read_parquet(path))


def test_real_items_parquet():
    path = _require(PROCESSED / "items.parquet")
    _assert_parquet_schema(path, ITEMS_SCHEMA)
    validate_items(pd.read_parquet(path))


def test_real_prepare_funnel():
    funnel = _load_json(RESULTS / "prepare_funnel.json")
    validate_funnel(funnel)
    # Degeneracy gate for the store mass rule (INTERFACE §1): if this fails the seller
    # definition must switch to a fixed percentile before any seller number is reported.
    stores = funnel["store_buckets"]
    total_stores = stores["large"]["n_stores"] + stores["small"]["n_stores"] + stores["unknown"]["n_stores"]
    if total_stores >= 100:
        assert stores["large"]["n_stores"] >= 100, "large store bucket has collapsed"
    else:
        assert stores["large"]["n_stores"] > 0, "large store bucket has collapsed"
    assert stores["small"]["store_share"] <= 0.95, "small store bucket holds ~all stores"


def test_real_run_manifest():
    manifest = _load_json(RESULTS / "run_manifest.json")
    assert manifest["seed"] == 42
    params = manifest["params"]
    assert params["k"] == 10
    assert params["n_candidates"] == 200
    assert params["min_rating"] == 4.0
    assert params["min_catalogue_interactions"] == 3
    assert params["min_train_positives_exposure"] == 1
    assert params["min_train_positives_relevance"] == 2
    assert params["n_eval_users"] == 300_000
    for package in ("numpy", "pandas", "implicit"):
        assert manifest["packages"].get(package), f"package version missing: {package}"
    assert manifest["score_stats"]["mean_per_user_score_std"] > 0


def test_real_baseline_metrics():
    baseline = _load_json(RESULTS / "baseline_metrics.json")
    assert baseline["k"] == 10 and baseline["n_candidates"] == 200
    assert int(baseline["catalog_size"]) > 0
    validate_exposure_block(baseline["exposure_pop"], "exposure_pop")
    validate_exposure_block(baseline["relevance_pop"], "relevance_pop")
    rel = baseline["relevance_pop"]
    assert 0.0 <= rel["ndcg"] <= rel["hitrate"] <= 1.0, "binary NDCG@k cannot exceed HitRate@k"
    assert "recall" not in rel, "recall@10 == hitrate@10 with one test item; not reported"

    for name in ("mostpop", "random"):
        model = baseline["baselines"][name]
        assert model["exposure_pop"]["ndcg"] is None, "exposure pop carries no ground truth"
        _assert_shares(model["exposure_pop"]["exposure_share"], POP_BUCKETS, f"{name} exposure")
        assert 0.0 <= model["relevance_pop"]["ndcg"] <= model["relevance_pop"]["hitrate"]


def test_real_baseline_matches_funnel():
    baseline = _load_json(RESULTS / "baseline_metrics.json")
    funnel = _load_json(RESULTS / "prepare_funnel.json")
    stages = {s["stage"]: s for s in funnel["stages"]}
    assert baseline["exposure_pop"]["n_users"] == stages["eval_exposure_pop"]["sampled_users"]
    assert baseline["relevance_pop"]["n_users"] == stages["eval_relevance_pop"]["sampled_users"]
    assert baseline["catalog_size"] == stages["catalogue_ge_3"]["items"]
    # Parity is exposure share over catalogue item share.
    for bucket in POP_BUCKETS:
        share = funnel["buckets"][bucket]["item_share"]
        expected = baseline["exposure_pop"]["exposure_share"][bucket] / share if share else 0.0
        actual = baseline["exposure_pop"]["exposure_parity"][bucket]
        assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9), bucket


def test_real_rerank_sweep():
    sweep = _load_json(RESULTS / "rerank_sweep.json")
    assert sweep["lambda_grid"] == LAMBDA_GRID
    assert sweep["chosen_lambda"] is None or sweep["chosen_lambda"] in LAMBDA_GRID
    for population in ("exposure_pop", "relevance_pop"):
        rows = sweep[population]
        assert [row["lambda"] for row in rows] == LAMBDA_GRID, f"{population} not ascending"
        for row in rows:
            validate_exposure_block(row, f"{population}[lambda={row['lambda']}]")
    for row in sweep["relevance_pop"]:
        for key in (
            "delta_ndcg_vs_baseline",
            "delta_ndcg_frac_of_als_minus_random",
            "delta_exposure_tail_vs_baseline",
        ):
            assert key in row, key
    # Taxing popularity can only lower mean recommended popularity.
    mrp = [row["mean_recommended_popularity"] for row in sweep["exposure_pop"]]
    assert all(a >= b for a, b in zip(mrp, mrp[1:])), "mean popularity rose with lambda"


def test_real_lambda_zero_reproduces_baseline():
    baseline = _load_json(RESULTS / "baseline_metrics.json")
    sweep = _load_json(RESULTS / "rerank_sweep.json")
    for population in ("exposure_pop", "relevance_pop"):
        zero = next(row for row in sweep[population] if row["lambda"] == 0.0)
        shared = {k: v for k, v in baseline[population].items() if k in zero}
        _assert_close_tree(shared, zero, f"{population}[lambda=0]")
    zero_rel = next(row for row in sweep["relevance_pop"] if row["lambda"] == 0.0)
    assert zero_rel["delta_ndcg_vs_baseline"] == 0.0


def test_real_delta_fraction_is_consistent():
    baseline = _load_json(RESULTS / "baseline_metrics.json")
    sweep = _load_json(RESULTS / "rerank_sweep.json")
    edge = baseline["relevance_pop"]["ndcg"] - baseline["baselines"]["random"]["relevance_pop"]["ndcg"]
    assert edge > 0
    for row in sweep["relevance_pop"]:
        delta = row["ndcg"] - baseline["relevance_pop"]["ndcg"]
        assert math.isclose(row["delta_ndcg_vs_baseline"], delta, abs_tol=1e-9)
        assert math.isclose(
            row["delta_ndcg_frac_of_als_minus_random"], delta / edge, rel_tol=1e-6, abs_tol=1e-9
        )


def test_real_sensitivity_gamma_zero_reproduces_baseline():
    sensitivity = _load_json(RESULTS / "sensitivity.json")
    baseline = _load_json(RESULTS / "baseline_metrics.json")
    block = sensitivity["relevance_pop"]
    assert block["n_users"] == baseline["relevance_pop"]["n_users"]
    assert 0.0 in block["gamma_values"]
    zero = block["metrics"]["0.0"]
    for metric in ("ndcg", "hitrate"):
        assert math.isclose(zero[metric], baseline["relevance_pop"][metric], abs_tol=SELF_CHECK_TOL)
    for gamma_key, metrics in block["metrics"].items():
        assert 0.0 <= metrics["ndcg"] <= metrics["hitrate"] <= 1.0, gamma_key
    for gamma_key, metrics in sensitivity["exposure_pop"]["metrics"].items():
        assert metrics["ndcg"] is None, "exposure pop carries no ground truth"


def test_real_baseline_scores_parquet():
    path = _require(RESULTS / "baseline_scores.parquet")
    _assert_parquet_schema(path, SCORES_SCHEMA)
    sample = _first_row_group(path)
    assert sample["rank"].between(0, 199).all()
    first_user = sample[sample["userId"] == sample["userId"].iloc[0]].sort_values("rank")
    assert not first_user["itemId"].duplicated().any(), "duplicate candidate for a user"
    assert (np.diff(first_user["score"].to_numpy()) <= 1e-6).all(), "rank 0 must be best"


def test_real_baseline_scores_exclude_train_items():
    scores_path = _require(RESULTS / "baseline_scores.parquet")
    ratings_path = _require(PROCESSED / "ratings.parquet")
    sample = _first_row_group(scores_path, ["userId", "itemId"])
    users = set(sample["userId"].unique())
    ratings = pd.read_parquet(ratings_path, columns=["userId", "itemId", "split"])
    known = ratings[ratings["userId"].isin(users) & (ratings["split"] != "test")]
    leaked = sample.merge(known, on=["userId", "itemId"])
    assert leaked.empty, f"{len(leaked)} already-rated items leaked into candidates"


@pytest.mark.parametrize("lam", LAMBDA_GRID)
def test_real_rerank_scores_parquet(lam):
    path = _require(RESULTS / f"rerank_scores_lambda_{lam:.2f}.parquet")
    _assert_parquet_schema(path, SCORES_SCHEMA)
    sample = _first_row_group(path)
    assert sample["rank"].between(0, 199).all()


def test_real_rerank_permutes_candidates_only():
    base_path = _require(RESULTS / "baseline_scores.parquet")
    rerank_path = _require(RESULTS / "rerank_scores_lambda_2.00.parquet")
    base = _first_row_group(base_path, ["userId", "itemId"])
    rerank = _first_row_group(rerank_path, ["userId", "itemId"])
    users = sorted(set(base["userId"]) & set(rerank["userId"]))[:500]
    assert users, "no overlapping users in the first row groups"
    base_sets = base[base["userId"].isin(users)].groupby("userId")["itemId"].apply(frozenset)
    rerank_sets = rerank[rerank["userId"].isin(users)].groupby("userId")["itemId"].apply(frozenset)
    assert (base_sets == rerank_sets.reindex(base_sets.index)).all(), "re-ranker introduced items"


def test_real_per_user_metrics_parquet():
    path = _require(RESULTS / "per_user_metrics.parquet")
    _assert_parquet_schema(path, PER_USER_SCHEMA)
    frame = pd.read_parquet(path)
    assert set(frame["population"].unique()) <= {"exposure_pop", "relevance_pop"}
    assert set(frame["lambda"].unique()) <= set(LAMBDA_GRID)
    assert frame["hit"].isin([0.0, 1.0]).all()
    assert (frame["ndcg"] <= frame["hit"] + 1e-12).all()

    sweep_path = RESULTS / "rerank_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        rel = frame[frame["population"] == "relevance_pop"]
        for row in sweep["relevance_pop"]:
            per_lambda = rel[rel["lambda"] == row["lambda"]]
            if per_lambda.empty:
                continue
            assert math.isclose(per_lambda["ndcg"].mean(), row["ndcg"], abs_tol=SELF_CHECK_TOL)
