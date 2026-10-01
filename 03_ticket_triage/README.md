# 03 Ticket Triage

## How to run

All commands run from `03_ticket_triage/src/` (Python 3.12):

```
pip install -r requirements-app.txt
python start.py
```

Then open `http://127.0.0.1:8501`. Ctrl+C stops both the API and the front end.

`src/` contains everything the MVP needs: the code (`app/`, `src/`), the trained model bundle (`models/bundle_v3/`),
the pack's `artifacts/` (cost matrix, capacity) and `eval/` files. The training data is not needed to run the MVP or
regenerate the submissions.

| What | Command (from `03_ticket_triage/src/`) |
|---|---|
| Start the MVP (front end + API) | `python start.py`, then open `http://127.0.0.1:8501` |
| Regenerate `submission.csv` and `submission_curveball.csv` | `python src/make_submission.py` (writes to `src/outputs/final/`; copy both files to `03_ticket_triage/`) |
| API smoke tests (including `/triage`) | `python -m pytest tests -v` |

`start.py` launches the API on port 8000 (interactive docs at `http://127.0.0.1:8000/docs`) and the front end on
port 8501. Both bind to 127.0.0.1. Use `--api-port` / `--ui-port` if those ports are taken.

**Front end** (Streamlit, `app/frontend.py`):
- **Single Ticket tab:** paste a subject and body, then pick a product and channel. The result shows a ROUTE or
  SEND TO HUMAN badge with the reason, plus the team, category, priority and sentiment, each with its confidence.
  It also flags an unhappy customer and any guard that fired.
- **Batch CSV tab:** upload a CSV (`ticket_id, channel, subject, body, product`) to get a results table and the
  hand-off rate against the 15% cap. A button downloads the results as a submission-format CSV.
- **Sidebar:** the headline validation metrics.

**API** (FastAPI, `app/api.py`):
- `GET /health`
- `GET /model_info`
- `POST /triage`, same as `POST /predict`: JSON with subject, body, product and channel. Returns all four labels
  with probabilities, `p_correct`, expected costs, the decision and a reason string.
- `POST /predict_batch`: CSV upload. Applies the 15% cap and returns JSON plus a submission-format CSV.

**Files in `03_ticket_triage/`:**
- `submission.csv` and `submission_curveball.csv`: the scored submissions.
- `submission_with_tickets.csv` and `submission_curveball_with_tickets.csv`: the same predictions next to the
  original ticket fields, for reading only.

The experiment pipeline and its reports (training, selection rounds, probes, the parity test against stored
validation outputs) live in the full project. They need the training data and about 900 MB of run outputs, so they
are not included here. File paths quoted below such as `outputs/...` refer to that project.

## Approach

Data: 30,000 labelled tickets split once into TRAIN 21,000 / VAL 4,500 / TEST 4,500 (stratified, seed 42). TEST is
locked. Every model choice was made on VAL using pre-written selection rules: paired bootstrap, 1,000 resamples,
95% CI must exclude 0 (`outputs/round2c/selection_rules.md`).

**Models, one per label.** Every model reads `channel + product + subject + body` as text.

| Label | Model |
|---|---|
| Category | word TF-IDF (1–2-grams) + logistic regression. Adding char n-grams gave no significant gain. |
| Priority | Ordinal model: three cumulative logistic regressions for P(≤P1), P(≤P2), P(≤P3) on word + char_wb(3–5) TF-IDF |
| Team | Tree stacker: gradient-boosted trees (HistGradientBoosting) over the category probabilities, a word+char team model's probabilities, and product/channel one-hot. It learns the product exceptions, e.g. switch firmware goes to Network-Support. |
| Sentiment | word+char TF-IDF → chi² top-2,000 features → HistGradientBoosting |

**Calibration.** Each label's probabilities are temperature-scaled, with T fit on 5-fold out-of-fold TRAIN
predictions (category 0.74, priority 0.75, team 1.14, sentiment 1.03).

**Correctness model.** A logistic regression predicts `p_correct`, the probability that the routed team is right.
Its inputs are team-probability features (top probability, margin, entropy), the priority probabilities, the
top category probability, product and channel. It is trained on out-of-fold predictions. It ranks misroutes
better than raw team confidence (VAL AUROC 0.756 vs 0.745).

**Abstention, policy f (expected cost).** For each ticket, with W/A the misroute/hand-off costs from the cost matrix:

- expected misroute cost = (1 − p_correct) × Σₖ P(priority = k) · W(k)
- expected hand-off cost = Σₖ P(priority = k) · A(k)
- savings = misroute − hand-off

In a batch, the tickets with the highest positive savings are sent to a human, up to the 15% cap. For a single
ticket in the app, a ticket goes to a human whenever savings > 0.

*Worked example:* "Alletra 6010 went offline after the firmware update; live payments cluster is down".
The model gives P(P1, P2, P3, P4) = (0.859, 0.140, 0.000, 0.001) and p_correct = 0.946.

- Expected misroute cost = 0.054 × (0.859·8 + 0.140·2 + 0.001·1) = **0.38**
- Expected hand-off cost = 0.859·1 + 0.140·0.5 + 0.001·0.3 = **0.93**
- Savings −0.55 → **route to Storage-Support**.

With team confidence 0.85 instead, the misroute cost would be 0.15 × 7.15 = 1.07 > 0.93, and the ticket would be
handed off. Each app response carries this reasoning as a reason string.

**P1 label threshold 0.225.** The priority label is P1 when calibrated P(P1) ≥ 0.225; otherwise it is the argmax
over P2–P4. This is the lowest threshold whose OOF priority macro-F1 stays within 0.005 of argmax. On VAL, P1
recall rises from 0.598 to 0.809 and macro-F1 from 0.767 to 0.770. Routing uses the full probabilities, not the
label, so routing cost is unchanged (asserted).

**Input guards.** Two guards force a hand-off and are named in the reason:

- a product not seen in training (proxy for an unseen category / unknown owner)
- an empty or very short body (< 20 characters after removing the portal severity prefix)

In a batch, guard-flagged tickets use cap slots first.

**Parity test.** `src/bundle.py` refits every component from the run configs, and its out-of-fold predictions match
the stored ones to within 5e-9. `tests/test_parity.py` scores VAL through the app's inference code. The calibrated
probabilities for all four labels and `p_correct` match the evaluation outputs to within 5e-12. Policy f with the
15% cap reproduces the same 426 hand-offs and the same cost, 0.141733. In the full project all 17 tests pass (API smoke tests plus parity); the 6 API smoke tests are included here and pass.

## Results

VAL (n = 4,500). Baseline = word TF-IDF + logistic regression per label. Final = combined_v3 with the P1 threshold.

| Metric | Baseline | Final |
|---|---|---|
| Category macro-F1 | 0.8667 | 0.8667 |
| Priority macro-F1 | 0.7428 | 0.7700 (argmax: 0.7674) |
| Team macro-F1 (all tickets) | 0.8973 | 0.9053 |
| Team accuracy | 0.9042 | 0.9122 |
| Sentiment macro-F1 | 0.8805 | 0.8857 |
| Escalation-risk F1 | 0.8518 | 0.8607 |
| P1 precision / recall / F1 | 0.799 / 0.528 / 0.636 | 0.658 / 0.809 / 0.726 |
| Routing cost, no abstention | 0.2096 | 0.1767 |
| Routing cost, policy b (15% least confident) | 0.1562 (abstain 15.0%) | 0.1538 (abstain 15.0%) |
| **Routing cost, policy f (expected cost)** | 0.1484 (abstain 10.1%) | **0.1417 (abstain 9.5%)** |
| Oracle floor (perfect abstention, cap 15%) | 0.0431 | 0.0380 |

The policy-f cost of 0.1417 vs the baseline's 0.1484 is a difference of −0.0067, 95% CI [−0.0157, +0.0019]. VAL was
used for selection, so these numbers are slightly optimistic.

TEST (n = 4,500, scored once after locking; nothing was tuned on it; `outputs/final/test_report.txt`):

| Metric | Final |
|---|---|
| Category / priority / team / sentiment macro-F1 | 0.8620 / 0.7686 / 0.8984 / 0.8859 |
| P1 precision / recall / F1 | 0.651 / 0.838 / 0.733 |
| Escalation-risk F1 | 0.8578 |
| Routing cost, no abstention | 0.1700 |
| Routing cost, policy b (15% least confident) | 0.1350 (abstain 15.0%) |
| **Routing cost, policy f** | **0.1331 (abstain 9.7%)** |

TEST matches VAL closely, so the VAL-based selection did not overfit noticeably.

Eval submission (`submission.csv`, 6,000 tickets): 691 abstains (11.5%, under the 900-ticket cap), 96 of them
forced by the short-body guard. Curveball file: 2 of 15 abstained.

## Tested and not adopted

- **Severity token** (parsed customer severity as an extra token): no change in any F1, and cost −0.0002
  [−0.0016, +0.0011]. The model already reads "severity: critical" as n-grams.
- **C tuning** (0.3–30 per label): no label improved significantly, so C = 1 was kept everywhere.
- **LinearSVC + softmax**: priority macro-F1 dropped 0.035 and category F1 dropped 0.005; the cost gain's CI
  includes 0.
- **Team cascade** (P(category) × P(team | category, product)): team F1 0.891 vs 0.897, cost +0.0007.
- **P1 sample weighting vs threshold**: weight 3 on P1 reached recall 0.827 but precision 0.601 and macro-F1 0.744.
  The ordinal model plus the 0.225 threshold gives recall 0.809, precision 0.658 and macro-F1 0.770, so it is
  better on both precision and macro-F1.
- **Richer correctness model** (agreement with a second model, a category→team lookup, category entropy): AUROC
  +0.002 [−0.003, +0.007] and cost +0.0032. The signal is redundant with the existing features.
- **Factor interactions** (scope × environment × workaround): pairwise terms add only 0.0019 log-loss and 0.0015
  AUROC for P1, so the effects are additive.
- **Priority stacker (E12), near miss**: P1 PR-AUC +0.0084 [−0.0017, +0.0198] missed the CI rule; macro-F1
  +0.0105; cost vs final +0.0004.
- **Novelty guard** (hand off tickets whose closest TRAIN ticket has cosine similarity below the 99.5th-percentile
  VAL cut-off): on eval it flagged 226 tickets. Most were typo-heavy or non-English tickets on known topics, and it
  missed most of the unseen-category tickets, so it was not deployed. The code remains in `src/novelty.py` and
  `src/guards.py`, but it is inactive unless the novelty files are in the bundle.

## Probe: what drives priority

Counterfactual edits on all applicable VAL tickets, using the priority model's calibrated probabilities.
"Up" means more urgent.

| Edit (all applicable tickets) | n | mean ΔP(P1) | mean ΔP(P1 or P2) | share up | share down |
|---|---|---|---|---|---|
| Scope up (single user → whole site) | 1,204 | +0.138 | +0.274 | 41% | 0% |
| Scope down | 160 | −0.211 | −0.318 | 0% | 48% |
| Environment down (production → test) | 1,573 | −0.108 | −0.216 | 0% | 41% |
| Add a workaround | 4,071 | −0.040 | −0.117 | 0% | 19% |
| Tone only (angrier wording) | 4,446 | +0.035 | +0.036 | 8% | 0% |
| Severity tag low → critical | 223 | +0.020 | +0.082 | 17% | 0% |
| Severity tag critical → low | 191 | −0.127 | −0.109 | 0% | 27% |
| Add a "critical" tag | 3,190 | +0.036 | +0.064 | 12% | 0% |
| Urgency claim ("Marking this urgent") | 3,805 | +0.006 | +0.009 | 4% | 0% |

Scope-up by environment: production ΔP(P1) +0.178 (n = 517), non-production +0.073 (n = 228).

The model responds to business impact: scope, environment and workaround. Tone and urgency claims barely move it,
which is the intended behaviour for "urgent" claims and angry-but-minor tickets.

## Limitations

- **VAL reused:** about 37 runs were compared on the same VAL set, so the selected results are slightly
  optimistic. TEST is held out for one final check.
- **70% of the data:** the final models are trained on TRAIN only (70%). A refit on all rows
  (`--rows all --i-have-locked`) is planned after the TEST check.
- **Template-based data:** the tickets are built from templates, so performance on free-form real tickets will
  be lower.
- **Missing scope/environment phrases:** about 40% of tickets have no recognisable scope phrase (40%) or
  environment phrase (43%). Priority is harder to infer for these.
- **Noisy labels:** the pack says some training labels are wrong, and we did no relabelling. Sentiment F1
  plateaus around 0.886 across all models.
- **Unseen category only partly handled:** the guards catch unknown products and short bodies only. The unseen
  category in the eval set is data-centre facilities: PDUs, CRAC/cooling, UPS, water leaks and power feeds.
  - A keyword scan finds about 195 such eval tickets. Only about 29 are abstained (by the cost rule); the rest are
    routed, mostly to Server-HW.
  - Next step: a dedicated facilities guard, separating rack and room power or cooling from server PSU and fan faults.
- **Information-free tickets above 20 characters still route:** for example "same thing as before, it's doing it
  again".
- **No embeddings or LLM:** non-English and sarcastic tickets rely on character n-grams only.

**Responsible AI.**

- The data contains no personal data beyond synthetic names in ticket text, and nothing is sent to external
  services.
- No LLM is involved in any decision, so there is no hallucination risk. Every output comes from deterministic
  local models and can be traced to probabilities and a cost calculation.
- Abstention is the safety valve: uncertain, high-stakes and out-of-scope tickets go to a human, with the reason
  shown.

**Tools.** Python, scikit-learn, rapidfuzz, FastAPI, Streamlit. No pretrained models or LLM gateway were used. Claude Code was
used as an AI coding assistant.
