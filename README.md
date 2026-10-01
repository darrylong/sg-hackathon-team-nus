# Team sg_hackathon_team_nus

Members: Ho Jone Mun, Koen Goh

```
<your team>/
  README.md                 this file - fill it in
  slides/                   ONE slide deck (PDF or PPTX, about 6-10 slides) covering both problems
  03_ticket_triage/         submission files (overwrite them) + src/ (your code and full-stack MVP)
  05_channel_analytics/     submission files (overwrite them) + src/ (your code and full-stack MVP)
```

## How to run

| Problem | Start the MVP (one command) | Regenerate the submission files |
|---|---|---|
| 03 Ticket Triage | `cd 03_ticket_triage/src && python start.py` then open `http://127.0.0.1:8501` (API docs at `http://127.0.0.1:8000/docs`) | `cd 03_ticket_triage/src && python src/make_submission.py` (writes `outputs/final/`; copy `submission.csv` and `submission_curveball.csv` up into `03_ticket_triage/`) |
| 05 Channel Analytics | `cd 05_channel_analytics/src && CHANNEL_DATA_DIR=/path/to/pack/05_channel_analytics/data ./start.sh` then open `http://localhost:8000` (API docs at `http://localhost:8000/docs`) | `cd 05_channel_analytics/src && CHANNEL_DATA_DIR=/path/to/pack/05_channel_analytics/data .venv/bin/python -m src.run_all` (about 20 s; writes `submission/`; copy both CSVs up into `05_channel_analytics/`) |

Setup:

- **03 Ticket Triage:** Python 3.11+. Run `pip install -r requirements-app.txt` once, from `03_ticket_triage/src/`.
  Everything the MVP and the submission script need is in `src/`: the trained model bundle (`models/bundle_v3/`),
  the pack's `artifacts/` (cost matrix, capacity) and the `eval/` tickets. The pack's training data is not needed.
  Run the API tests with `python -m pytest tests -v`. Details are in `03_ticket_triage/README.md`.
- **05 Channel Analytics:** Python 3.9+. Node.js is not needed because the built React frontend is included. On the
  first run, `start.sh` creates `.venv` and installs `requirements.txt` (needs internet, about 1-2 min). Run
  `start.sh` once before the regenerate command, because that command uses `.venv`. The code reads the pack's three
  CSVs (`partner_sales.csv`, `partner_master.csv`, `targets.csv`) from `CHANNEL_DATA_DIR`. Always set it in this
  layout: the default (`../05_channel_analytics/data`, relative to `src/`) does not exist here. The pipeline
  recomputes everything else, including the health scores and the forecast, at startup (about 10 s). Run the 39 tests
  (leakage, submission format, API) with `CHANNEL_DATA_DIR=... .venv/bin/python -m pytest -q`. The last step of
  `run_all` compares against our earlier v1 folder, which is not part of the hand-in, so it is skipped with a note.
  Details are in `05_channel_analytics/src/README.md`.

## Approach

### 03 Ticket Triage

Full write-up: `03_ticket_triage/README.md`.

- **Split:** 30,000 labelled tickets split once into TRAIN 21,000 / VAL 4,500 / TEST 4,500 (stratified, seed 42).
  Every model choice was made on VAL with a pre-written rule: paired bootstrap, 95% CI must exclude 0. TEST was
  scored once at the end.
- **Models (one per label, all reading `channel + product + subject + body` as text):**
  - Category: word TF-IDF + logistic regression.
  - Priority: ordinal model (three cumulative logistic regressions) on word + character n-gram TF-IDF. The label is
    P1 when P(P1) ≥ 0.225, which raises P1 recall on VAL from 0.60 to 0.81.
  - Team: gradient-boosted tree stacker over the category and team probabilities plus product/channel. It learns
    product exceptions such as switch firmware going to Network-Support.
  - Sentiment: word + character TF-IDF, chi² feature selection, gradient-boosted trees.
  - All probabilities are temperature-calibrated.
- **How we decide to abstain:** a correctness model predicts `p_correct`, the chance the routed team is right. For
  each ticket we compare the expected misroute cost, (1 − p_correct) × the priority-weighted misroute cost, with the
  expected hand-off cost from the cost matrix. Tickets where a hand-off is cheaper go to a human, highest savings
  first, up to the 15% cap. Two guards always force a hand-off: a product not seen in training, and an empty or very
  short body. Every decision comes with a plain-language reason.
- **Noisy labels:** we did not relabel or clean the training labels. Calibration keeps the probabilities honest, and
  the abstention rule sends low-confidence tickets to a human. Sentiment macro-F1 levels off at about 0.886 across
  every model we tried, which suggests label noise sets the ceiling.
- **Sarcasm and mixed languages:** no translation, embeddings or LLM. Character n-grams pick up product names, error
  codes and word fragments in any language. On the curveball set, the Malay (CB-003) and German (CB-011) tickets were
  routed to the correct team. Sarcasm is not handled: CB-001 ("Wow, what an achievement… rebooted itself four times")
  was scored as Neutral sentiment.

### 05 Channel Analytics

Full write-up: `05_channel_analytics/src/README.md`. Running the pipeline also writes reports to `src/outputs/`
(`eda/eda_report.md`, `models/at_risk_report.md`, `models/forecast_report.md`).

- **Data:** 273,250 order lines from 2,600 partners, Jan 2024 to Jun 2026. One line (PT-00018, EMEA GreenLake,
  $37.8M) is 47 times the next-largest line. It is flagged as an outlier and left out of every aggregate; with it,
  EMEA 2026-Q1 would show +11.5% QoQ instead of −2.6%.
- **Key insights.** Each one is an insight card on the dashboard, and every number on it is computed from the data:
  - **Growth comes mostly from AMS.** Like-for-like H1 revenue grew +35% in AMS but only +5-6% in APJ and EMEA, and
    42-43% of APJ/EMEA partners are shrinking. Existing APJ/EMEA partners are almost flat (+2-4% YoY), so most growth
    there comes from partners onboarded since 2024.
  - **Growth is volume, not price.** No product family's unit price moved more than 2.2% in 30 months.
  - **Margin depends on mix.** GreenLake earns 21.7% against ProLiant's 9.8%, and Silver partners 16.5% against
    Platinum's 10.6%. Blended margin is nearly flat, so it hides shifts in mix.
  - **Revenue is concentrated.** The top 10% of partners bring 58% of revenue, and the top 20% bring 74%.
  - **Close to target overall, but most partners miss.** The channel reaches 92-99% of target, yet only about 45% of
    partners hit their own target in a typical quarter.
  - **Strong seasonality.** The last month of a quarter is 1.5 times the other two, and Q3 is below Q2 every year.
  - **Most revenue collapses recover.** Of partners whose quarter fell below 10% of their usual revenue, 71-94%
    recovered the next quarter; most are small Business-tier resellers. A naive churn label is mostly noise.
- **How the health score works:**
  - Label (the judges' definition): in the next quarter, the partner has no orders, or revenue below 10% of its
    trailing 4-quarter average.
  - 57 point-in-time features in 7 groups: revenue trend, order activity, target attainment, margin, product breadth,
    year-on-year, and static attributes. Each feature uses only data up to its cutoff; a test checks that deleting
    future rows does not change it.
  - Model: the mean of a logistic regression and a gradient-boosting classifier, trained on 6 quarterly snapshots
    (cutoffs 2024-Q4 to 2026-Q1, 14,146 rows). The probabilities are Platt-calibrated on out-of-fold predictions,
    then scaled up for Q3, because Q3 leave rates are 1.27 times the average.
  - `health_score` = 100 × (1 − P(leave in 2026-Q3)).
  - `at_risk` flags the top N partners, where N = 1.5 × the expected number of leavers. The 1.5 gave the best F1 in
    the backtest. The model expects 60 leavers, so 90 partners are flagged (85 Business, 5 Silver), about 0.3% of
    channel revenue.
  - **Reasons:** for each partner, each driver group (recency, frequency, size, momentum, YoY, attainment, product
    breadth, margin, volatility) is reset in turn to the median of healthy partners in the same tier. The drivers
    are ranked by how much the risk drops, and the top 3 are written as sentences with the partner's own numbers,
    for example "Orders infrequently: active in 1 of the last 6 months, 0.2 order lines a month (typical Business
    partner: 1.7)". No LLM is used, so no figures can be invented.
  - **Quietly declining partners** (274 partners, $45.7M in Q2 revenue; H1 YoY ≤ −25% and down 2+ quarters) are
    shown as a separate segment and not flagged. Historically they did not leave more often than partners of the
    same size and tier.
- **Forecast method:**
  - Six methods per region: seasonal ratio, YoY growth, OLS on monthly log revenue, ARIMA airline model, SARIMAX with
    targets, and projected target × the usual attainment for that quarter of the year. `targets.csv` ends at
    2026-Q2, so the Q3 target is projected from earlier targets. The backtest also uses projected targets and never
    reads the real target for the quarter being forecast.
  - Ensemble: the equal-weight mean of all six. We fixed the rule before looking at results: use all six only if
    their leave-one-quarter-out error is no worse than our first version's three-method ensemble. It was better,
    1.7% vs 2.6%.
  - Bias correction: −2.0%, the mean backtest log error. Recent forecasts ran high because growth is slowing.
- **Interval:** the 80% interval is a t-quantile × the spread of the backtest log errors. Each region uses the
  larger of the pooled spread and its own. The result is ±2.5% for ALL and ±3.3% to ±4.3% for the regions.
- **Link to the at-risk model:** the at-risk probabilities give an excess-churn adjustment of −$2.0M for ALL. In the
  backtest it made the error slightly worse, so it is not applied; the dashboard shows it as revenue at risk.

## Results

### 03 Ticket Triage

TEST hold-out (n = 4,500, scored once after all choices were locked on VAL):

| Metric | Result |
|---|---|
| Category / priority / team / sentiment macro-F1 | 0.862 / 0.769 / 0.898 / 0.886 |
| P1 precision / recall / F1 | 0.651 / 0.838 / 0.733 |
| Escalation-risk F1 | 0.858 |
| Routing cost, no abstention | 0.1700 |
| Routing cost, abstain on the 15% least confident | 0.1350 |
| **Routing cost, expected-cost abstention (ours)** | **0.1331 (9.7% handed off)** |

The four macro-F1 scores on TEST are within 0.007 of VAL, so selecting on VAL did not overfit noticeably.
On the eval file, 691 of 6,000 tickets (11.5%) are handed off; 2 of the 15 curveball tickets are handed off.

### 05 Channel Analytics

**At-risk model:** rolling-origin backtest. Each quarter is scored by a model trained only on earlier quarters.

| Metric | Result |
|---|---|
| ROC-AUC, mean over 4 hold-out quarters (cutoffs 2025-Q2 to 2026-Q1) | 0.857 |
| ROC-AUC, latest hold-out (cutoff 2026-Q1, labelled by 2026-Q2) | 0.851 |
| Average precision (base rate 1-2% per quarter) | 0.084 |
| Precision / recall of the `at_risk` flag | 0.103 / 0.171 |
| Baseline: one feature (order lines per month), ROC-AUC | 0.856 |

The model barely beats the one-feature baseline: the signal is mostly how often and how much a partner orders. The
ensemble's gain is stability across quarters. XGBoost, LightGBM, deeper trees and up-weighting decliners all scored
0.821-0.848. The best challenger tied: paired-bootstrap 95% CI for the difference was −0.004 to +0.004.

**Forecast:** leave-one-quarter-out over 5 quarters (2025-Q2 to 2026-Q2). Each quarter is forecast from earlier data
only.

| Area | MAPE (bias-corrected) | Actual inside 80% interval | 2026-Q3 forecast (80% interval) |
|---|---|---|---|
| APJ | 1.6% | 5 of 5 | $192.4M ($186.2M - $198.8M) |
| EMEA | 1.9% | 4 of 5 | $228.3M ($220.8M - $236.2M) |
| AMS | 2.0% | 3 of 5 | $718.9M ($689.0M - $750.0M) |
| **ALL** | **1.3%** | 3 of 5 | **$1,139.6M ($1,112.0M - $1,167.9M)** |

Single methods, raw MAPE for ALL: target × attainment 1.2%, seasonal ratio 2.5%, SARIMAX 2.6%, YoY growth 2.7%,
OLS 3.3%, ARIMA 8.0%. Our first version's three-method ensemble scored 2.6% mean leave-one-out MAPE against 1.7% for
the six-method ensemble used here. The ALL forecast is −3.7% vs 2026-Q2, matching the usual Q3 dip, and +28.6% YoY.

## Limitations and next steps

### 03 Ticket Triage

- **Unseen category only partly caught:** the eval set adds data-centre facilities tickets (cooling, PDUs, UPS,
  water leaks). About 195 eval tickets look like this, and only about 29 are handed off; most go to Server-HW (for
  example curveball CB-004, "server room aircon dead"). Next step: a facilities guard that tells room power and
  cooling apart from server PSU and fan faults.
- **Tickets with no real content still route** if they are over 20 characters, for example "same thing as before,
  it's doing it again" (CB-015). Next step: a guard for content-free tickets.
- **Sarcasm and non-English text** rely on character n-grams only. Next step: a multilingual sentence-embedding
  model as an extra feature.
- **Trained on 70% of the data:** the final models use TRAIN only. Next step: refit on all 30,000 rows.
- **Template-built data:** performance on real, free-form tickets will be lower.

### 05 Channel Analytics

- **Leaving is hard to predict:** only about 1 in 10 flagged partners actually leaves. The history has almost no
  permanent leavers, so the label mostly catches small, irregular partners who skip a quarter and come back. Next
  step: confirm real leavers with the channel team and retrain on them.
- **Slowly declining partners are not flagged:** the 274 quietly declining partners are only a segment. If 2026-Q3
  leavers come from this group, the model will miss them. Next step: a second score that predicts next-quarter
  revenue change rather than a leave/stay label.
- **Short history for the forecast:** 30 months gives only 5 backtest quarters, so the ensemble choice and the
  interval rest on few points. The ALL interval covered the actual value in only 3 of 5 backtest quarters, so it
  may be too narrow. ARIMA and SARIMAX are fitted on 15-29 monthly points, and their parameters are unstable. Next
  step: recheck coverage every quarter, and widen the interval (for example with conformal intervals) if it keeps
  missing.
- **Assumptions inside the forecast:** the target-based methods depend on our projection of the Q3 target, and the
  −2.0% bias correction assumes growth keeps slowing at the recent pace. Next step: use the real Q3 targets once
  they are published.
- **No tier history:** tier and region come from the current master file. Reasons explain the model's score, not
  why the partner behaves that way.
- **Next step for the dashboard:** let an LLM rephrase each partner's reasons, grounded on the same computed numbers.

## Tools and models used

- **03 Ticket Triage:** Python, scikit-learn, pandas, rapidfuzz, FastAPI, Streamlit. No pretrained models, no LLM
  or gateway; nothing is sent to external services. Claude Code was used as an AI coding assistant.
- **05 Channel Analytics:** Python, pandas, NumPy, scikit-learn (logistic regression, gradient boosting, Platt
  calibration), statsmodels (ARIMA / SARIMAX), SciPy, matplotlib, FastAPI + uvicorn. Frontend: React 18 + TypeScript,
  Grommet with the HPE theme (`grommet-theme-hpe`), Recharts, Vite. XGBoost and LightGBM were tested but not used. No
  pretrained models, no LLM or gateway; nothing is sent to external services. Claude Code was used as an AI coding
  assistant.
