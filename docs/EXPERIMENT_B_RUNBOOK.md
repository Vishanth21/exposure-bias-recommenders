# Experiment B — re-run on the cloud machine

## What B tests

Run A found that ALS's top-200 candidate lists contain **zero tail items** (0 of 116,984,800
slots), so the re-ranker had nothing in the tail to promote. B asks the obvious follow-up: **if the
tail is in the candidate set, does the re-ranker move exposure there, and at what cost?**

B reuses run A's trained ALS model; nothing is retrained. Each user keeps their original top 180
candidates, gets their 20 highest-scoring unseen tail items, and is topped back up to 200 from
original ranks 180–199. Tail items always score below every original candidate, so they land at
the bottom of the list. The original top 10 is asserted unchanged, which means **B at λ = 0 must
reproduce run A's baseline exactly**.

Code: branch `experiment-tail-candidates-b`. The "Expected 200 candidates, got 201" crash from the
first attempt is fixed in commit `902d737`. The fallback loop now stops at 200, and the
uniqueness, known-item and top-10 assertions are intact. Outputs go to `results_b/`, so run A's
`results/` is never touched.

## 0. Before starting (2 min)

```bash
cd ~/exposure-bias-recommenders
source .venv/bin/activate
git fetch origin
git switch experiment-tail-candidates-b
git pull --ff-only
git log --oneline -1                       # expect 7b24a6f or later
ls results/baseline_scores.parquet results/model/als_factors.npz data/processed/*.parquet
ls results_b/ 2>/dev/null                  # leftovers from the failed attempt
df -h .                                    # B needs ~6 GB free
```

If `results_b/baseline_scores.parquet` exists from an earlier attempt, move it away first, because
the script refuses to overwrite it. A leftover `.partial` file, `run_manifest.json` or
`prepare_funnel.json` is fine; those are replaced automatically.

## 1. Run stages 02b → 05 in one detached job

This survives disconnecting from SSH. Each stage starts only if the previous one succeeded.

```bash
mkdir -p logs_b
nohup bash -c '
  set -eo pipefail
  cd ~/exposure-bias-recommenders && source .venv/bin/activate
  python scripts/02b_tail_candidates.py                           > logs_b/02b.log 2>&1
  python scripts/03_evaluate.py --results-dir results_b           > logs_b/03.log  2>&1
  python scripts/04_rerank.py --baseline-scores results_b/baseline_scores.parquet \
      --baseline-metrics results_b/baseline_metrics.json --results-dir results_b \
                                                                  > logs_b/04.log  2>&1
  python scripts/05_sensitivity.py --baseline-scores results_b/baseline_scores.parquet \
      --baseline-metrics results_b/baseline_metrics.json --results-dir results_b \
                                                                  > logs_b/05.log  2>&1
  echo OK > logs_b/DONE
' > logs_b/runner.log 2>&1 < /dev/null &
echo "started PID $!"
```

To check progress later: `tail -f logs_b/02b.log`, then the next log. `logs_b/DONE` appears when
everything has finished. If a stage fails, `logs_b/runner.log` and that stage's log show the error.

**Rough timings:** 02b about 30–60 min (it scores 437,915 tail items for each of 584,924 users;
its log prints users/s, so the estimate firms up quickly). 03 about 10 min. 04 about 15–20 min
(about 2.3 min per λ in run A). 05 a few minutes. **Total: about 1–1.5 h.**

## 2. Checks that say the run is valid

| Where | Look for | Means |
|---|---|---|
| `02b.log`, last lines | `Tail candidates written: … (20.00 per user on average)` | Every user got the full tail quota |
| `03.log` / `results_b/baseline_metrics.json` | ALS `ndcg` = **0.013341209**, `hitrate` = **0.022936667** (same as run A) | Top 10 untouched; B is a clean comparison |
| `04.log` | No `Lambda=0.0 NDCG difference` warning | Sweep harness aligned with the baseline |
| `results_b/rerank_sweep.json` | `exposure_share.tail` > 0 for some λ > 0 | The re-ranker now reaches the tail |

If the λ = 0 numbers differ from run A, **stop and send the logs**: something other than the
candidate set changed.

## 3. What to send back

Upload these five small files to the Drive folder, in a `results_b/` subfolder:

```
results_b/baseline_metrics.json
results_b/rerank_sweep.json
results_b/sensitivity.json
results_b/run_manifest.json
results_b/prepare_funnel.json
```

plus `logs_b/*.log`. The large parquet files can stay on the machine. Stage 06 (figures and tables)
is run locally from the JSONs with the fixed plotting code, so it doesn't need to run on the cloud.

## 4. What happens with the results (locally)

```bash
python scripts/06_report.py --results-dir results_b --figures-dir figures_b
```

Then add a "Run B: tail in the candidate set" subsection to `report/REPORT.md` §4.2 and one slide
after "The tail stays at 0%". How to read the outcome:

| Outcome | What we say |
|---|---|
| Tail share rises with λ, cost similar to run A | The re-ranker works once the tail is a candidate. The bottleneck in run A was candidate generation, as claimed. |
| Tail share rises but NDCG drops much faster than in A | Reaching the tail is possible but expensive on observed metrics. Point to the sensitivity analysis (the cost is an upper bound). |
| Tail share still ≈ 0 at λ = 2 | 20 tail candidates aren't enough against a 2σ penalty. Report it honestly; a larger quota is the next step. |

Run A's results stay the headline either way. B is reported as a candidate-generation sensitivity
experiment, not a replacement.
