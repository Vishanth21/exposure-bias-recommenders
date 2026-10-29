# Amazon Reviews 2023 — dataset facts

What the pipeline depends on, sourced to the dataset card, the dataset docs site, `implicit`
source, and Hugging Face `datasets` docs.

## Review records

Fields: `rating` (float), `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`,
`timestamp` (int), `helpful_vote` (int), `verified_purchase` (bool).
[Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md),
[docs field table](https://amazon-reviews-2023.github.io/main.html)

`timestamp` is Unix **milliseconds**. Sample row: `1588687728923`, which is 2020-05-05. Read as
seconds it would be the year 52295, so a pipeline asserting a plausible date range catches the
mistake.
[Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md)

`rating` is typed `float` and documented as ranging 1.0 to 5.0. Published samples are all whole
stars. Whether fractional values occur is unsettled, so log the observed distribution on the
first full load; the positive fraction at `rating >= 4` falls out of the same pass.
[Docs field table](https://amazon-reviews-2023.github.io/main.html)

## Item metadata

Fields: `main_category`, `title`, `average_rating` (float), `rating_number` (int), `features`,
`description`, `price`, `images`, `videos`, `store`, `categories`, `details`, `parent_asin`,
`bought_together`.
[Docs field table](https://amazon-reviews-2023.github.io/main.html),
[Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md)

Two declared types mislead: `price` is **string**, not float — it holds `"None"` when absent —
and `details` is a **JSON-encoded string**, not a dict, so it needs `json.loads` before use.

`store` is the storefront/brand label, not a seller identity. It arrives empty, `None`, or as the
literal `"None"`; normalise all four cases to `""` at the loader.
[Docs field table](https://amazon-reviews-2023.github.io/main.html),
[card sample](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md)

`parent_asin` is the join key: variants differing by colour, style or size share a parent, and
"**Please use parent ID to find product meta**". The variant `asin` collapses into it.
[Docs field table](https://amazon-reviews-2023.github.io/main.html)

## Category statistics

Counts are taken over reviews, not metadata, and some items have no metadata.
[Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md)

| Category | Users | Items | Ratings | Ratings/item | Ratings/user |
|---|---:|---:|---:|---:|---:|
| Electronics | 18.3M | 1.6M | 43.9M | ~27.4 | ~2.4 |
| Books | 10.3M | 4.4M | 29.5M | ~6.7 | ~2.9 |

## Loading

Reviews and metadata are separate subsets, `raw_review_<Category>` and `raw_meta_<Category>`.
Each has one split, `full`.

```python
load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", trust_remote_code=True)
load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Electronics", split="full", trust_remote_code=True)
```

`trust_remote_code=True` is required: "For security reasons, 🤗 Datasets do not allow running
dataset loading scripts by default". The script mechanism is legacy — the docs call scripts
"likely not needed" for JSON/JSONL/Parquet, and the page is published only on `main`. This
repository ships a script, which is also why its viewer is disabled with
`DatasetWithScriptNotSupportedError`. Pin the `datasets` version on cloud so an upstream removal
cannot break loading quietly.
[datasets docs](https://huggingface.co/docs/datasets/main/en/dataset_script),
[dataset page](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

The dataset is not gated; no HF token is needed.

Raw per-category archives come from datarepo:

- reviews: `https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/<Category>.jsonl.gz`
- metadata: `https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz`

The datarepo metadata prefix is `meta_`; the Hub subset prefix is `raw_meta_`. Mixing them 404s.
[Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md)

## `implicit` ALS

`implicit.als.AlternatingLeastSquares` takes `factors` (100), `regularization` (0.01), `alpha`
(1.0), `iterations` (15) and `random_state` (None), alongside `dtype`, `use_native`, `use_cg`,
`use_gpu`, `calculate_training_loss` and `num_threads`.
[als.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/als.py)

`fit(user_items)` wants a CSR matrix of shape **users × items** holding confidences: "the rows of
the matrix are the users, the columns are the items liked by that user, and the value is the
confidence that the user liked the item." Unset entries are read as `P=0, C=1`; negative values
express a dislike.
[cpu/als.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/cpu/als.py)

`recommend(userid, user_items, N=10, filter_already_liked_items=True, ...)` returns
`(itemids, scores)`, 1-D per user and 2-D over a batch. `filter_already_liked_items` defaults to
`True`.
[cpu/matrix_factorization_base.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/cpu/matrix_factorization_base.py),
[recommender_base.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/recommender_base.py)

`alpha` is "the weight to give to positive examples", applied as `Cui = alpha * Cui` before
solving. The default is `1.0`; binarising to 0/1 and setting `alpha=20` is a project choice.
[cpu/als.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/cpu/als.py),
[als.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/als.py)

After `fit`, `user_factors` and `item_factors` hold the latent factors as ndarrays.
`similar_items(itemid, N=10, ...)` is the item-item API.
[cpu/als.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/cpu/als.py),
[recommender_base.py](https://raw.githubusercontent.com/benfred/implicit/main/implicit/recommender_base.py)

## Licence

No licence is stated for this dataset. It is open to download without an access request, so use
is governed by the citation the authors request:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

For an explicit licence statement, ask the authors: `yphou AT ucsd.edu`.
[Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md)