# Data card — Amazon Reviews 2023 (Electronics)

Prepared by Rithesh (PES1UG24AM064) · co-signed by Vishanth (PES1UG24AM323) ·
built from the primary-source notes in [`DATASET_FACTS.md`](DATASET_FACTS.md) and the observed
counts in `results/prepare_funnel.json` (run `c15fbb8`, 2026-11-02).

## At a glance

| | |
|---|---|
| Name | Amazon Reviews 2023 — `raw_review_Electronics`, `raw_meta_Electronics` |
| Publisher | McAuley Lab, UC San Diego |
| Source | [Hugging Face `McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) · raw archives at `datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/` |
| Licence | **None stated.** Openly downloadable without an access request; use is governed by the requested citation (below). For an explicit licence, contact the authors (`yphou AT ucsd.edu`). |
| Access | Not gated; no token. The HF loader needs `trust_remote_code=True` (legacy script). |
| Used in this project | All of Electronics: 43,886,944 reviews · 18,286,191 users · 1,609,860 items |
| Used for | Training an implicit ALS recommender and measuring how exposure is distributed across item popularity and storefront size. Academic coursework only. |

## Motivation

The dataset was released to support research on recommendation and language–item retrieval. We
use it because, unlike MovieLens or Netflix-style data, its item metadata carries a `store` field.
That lets "large sellers get more exposure" be *measured* on a real storefront label, not proxied
by popularity.

## Composition

**Review record** (one per user–item review): `rating` (float, 1.0–5.0), `title`, `text`,
`images`, `asin`, `parent_asin`, `user_id`, `timestamp` (int, **Unix milliseconds**),
`helpful_vote`, `verified_purchase`. We read only `user_id`, `parent_asin`, `rating`,
`timestamp`.

**Metadata record** (one per `parent_asin`): `main_category`, `title`, `average_rating`,
`rating_number`, `features`, `description`, `price`, `images`, `videos`, `store`, `categories`,
`details`, `parent_asin`, `bought_together`. We read `parent_asin`, `title`, `store`,
`main_category`, `price`.

**Item identity** is `parent_asin`. Colour, size and style variants (`asin`) collapse into their
parent, as the dataset authors instruct.

**Observed distribution (Electronics, full pass):**

| Quantity | Value |
|---|---:|
| Reviews | 43,886,944 |
| Users / items in reviews | 18,286,191 / 1,609,860 |
| Positive fraction (`rating >= 4`) | 0.761 |
| Positives / non-positives | 33,387,473 / 10,499,471 |
| Items with ≥ 3 train positives (catalogue) | 538,448 |
| Head items holding 50 % of catalogue interactions | 14,607 (2.71 %) |
| Tail items holding the last 20 % | 437,915 (81.33 %) |
| Catalogue items with missing `store` | 0.59 % of items · 0.35 % of catalogue |
| Distinct stores in catalogue | 64,579 known (9,992 `large`, 54,587 `small`) |
| Distinct test items outside the catalogue | 51.6 % |

Ratings per user average ~2.4 and per item ~27.4, so the matrix is extremely sparse. Most users
have a single review.

## Collection process

Reviews and metadata were collected from Amazon.com by the dataset authors and cover reviews up
to 2023. We did not collect, scrape or augment anything. The data reflects what Amazon displayed
and what reviewers chose to write. That is the exposure bias this project studies: items Amazon
already surfaced are more likely to have reviews at all.

## Preprocessing in this project

All preprocessing lives in `src/amazon_loader.py` and `scripts/01_prepare.py`:

- **`store`** normalised at the loader: missing, `None`, the literal `"None"`, blank and
  whitespace-only all become `""`, which maps to `store_bucket = "unknown"`. It is never dropped
  or merged into `small`.
- **`price`** is declared numeric but arrives as a string (`"None"` when absent). Unparseable →
  `NaN`; the item is kept.
- **`title`, `main_category`** missing → `""`.
- **`timestamp`** kept in milliseconds. The loader raises if the median looks like seconds,
  because that would silently corrupt the temporal split.
- **Positive** = `rating >= 4`. Lower ratings are kept only as `seen` rows, to exclude
  already-rated items from candidates.
- **No user or item filtering.** Reviews whose `parent_asin` has no metadata row keep an item row
  with empty fields, so they still count toward popularity.
- **Split:** temporal leave-last-out per user (the last positive → test).

## Known gaps and caveats

1. **`store` is a storefront/brand label, not a seller identity.** One brand may be sold by many
   merchants, and one merchant may list under several labels. Our "seller size" is the catalogue
   footprint of a label.
2. **Store null rate is low (0.59 %)**, so the seller analysis rests on almost all items. This was
   the gate that decided whether the seller claim could be made at all.
3. **Reviews ≠ purchases ≠ impressions.** A review is a sparse, self-selected signal of
   interaction. Non-reviewers and items that were shown but not reviewed are invisible.
4. **Popularity-biased labels.** Held-out positives come from a platform that already
   over-exposed popular items, so offline relevance metrics favour popular recommendations (see
   REPORT §4.3).
5. **Rating scale.** Published samples are whole stars, but the card types `rating` as float and
   does not state the value set. With a positive fraction of 0.761, `>= 4` still separates a
   meaningful quarter of reviews.
6. **Declared types mislead:** `price` is a string, and `details` is a JSON-encoded string.
7. **Temporal coverage is uneven** across years and sub-categories. We do not correct for
   drift.
8. **Personal data.** `user_id` is an opaque identifier. Review text may contain personal content,
   but this project never reads it. No artefact we publish contains review text.

## Intended and out-of-scope uses

**Intended here:** offline research on exposure distribution and debiasing in recommenders.
**Out of scope:** profiling individual users or sellers, making claims about specific brands, or
treating storefront labels as legal seller entities.

## Distribution and maintenance

We redistribute no raw data. `data/` and `results/` are gitignored and regenerated by the
pipeline. The dataset is maintained by the McAuley Lab. Pin `datasets==3.1.0` so that an upstream
change to the legacy loading script cannot break loading silently.

## Citation

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

---

Sign-off: Rithesh ☑ · Vishanth ☐
