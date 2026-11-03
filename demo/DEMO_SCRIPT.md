# Demo video — 3 minutes

One minute per person, in pipeline order: the finding, the fix, then the engineering and the
takeaway. Record each segment separately and join them; every segment opens on its own slide
from the deck, so the cuts are clean.

| Time | Who | Covers | Opens on slide |
|---|---|---|---|
| 0:00–1:00 | Vishanth | The question, the data, Finding 1 (head takes 99.96%, tail 0%) | 2 → 6 |
| 1:00–2:00 | Avinash | The re-ranker, the λ trade-off, Finding 2 (tail stays at 0%) | 9 → 12 |
| 2:00–3:00 | **Rithesh** | How we know the numbers are right, the report, the takeaways | 5 → live → 14 |

Vishanth's and Avinash's rows are a suggested split for them to confirm. The speaker notes on
their slides work as a script.

---

## Rithesh — 2:00 to 3:00 (fully scripted)

About 150 words, at a relaxed speaking pace. Timings are cumulative within the segment.

### Before you hit record

- Open a terminal in the repo root, font at 18 pt or bigger, and activate the venv first:
  `.venv\Scripts\activate`. Clear the screen.
- Run both commands below once off camera so nothing is slow on the first take.
- Have the deck open on slide 5 (Pipeline), and `report/results.ipynb` open in VS Code or Jupyter,
  scrolled to section 5 (the sweep table).

### Shot 1 · 0:00–0:12 · deck, slide 5 (Pipeline)

> "I built the dataset loader and the test suite that holds this pipeline together. Every
> handoff between these six stages has a frozen schema, and the tests check all of them."

### Shot 2 · 0:12–0:32 · terminal

Type and run:

```
python -m pytest tests/test_integration.py -v
```

> "This feeds a small slice of Amazon-shaped data through the loader and the prepare stage,
> then checks the real cloud results. That includes the self-check that λ zero reproduces the
> baseline exactly. Seventeen pass. The skips are the multi-gigabyte files that stay on the
> cloud machine. Writing these tests caught two real bugs before the report."

Let the green `17 passed` line sit on screen for a second before cutting.

### Shot 3 · 0:32–0:47 · results notebook, section 5 (sweep table)

> "The report and this notebook read only the result files and never recompute a metric.
> Here's the cost curve: at λ 0.5 we spend two percent of the model's edge and cut recommended
> popularity by a third."

Point at (or highlight) the λ = 0.5 row.

### Shot 4 · 0:47–1:00 · deck, slide 14 (Three takeaways)

> "So: exposure bias at catalogue scale is extreme, a simple penalty is cheap at moderate λ,
> but it can't reach the tail. That fix has to happen upstream. Thanks for watching."

---

## Recording options on this laptop

- **Windows Snipping Tool** (Windows 11): `Win + Shift + R`, then drag over the screen region
  and use the record button. Saves an MP4 and records your microphone.
- **Xbox Game Bar:** `Win + Alt + R` starts and stops recording of the active window. It can't
  record the desktop or switch windows mid-take, so record each shot separately.
- **PowerPoint**, if the deck is exported to `.pptx`: Record → From current slide records
  narration over the slides; insert the terminal and notebook shots as clips.

Join the three segments in Clipchamp (built into Windows 11), check the total is at most
3:00, and export at 1080p.
