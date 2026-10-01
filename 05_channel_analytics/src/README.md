# Partner Channel Performance Analytics (Problem 05) - v2

## Changes vs v1

v1 is frozen in `../Hackathon`. v2 changes only the forecast; the at-risk model and its submission are identical.

- **New forecasting methods:** ARIMA (SARIMA(0,1,1)(0,1,1)_3 airline model), SARIMAX (regression with ARIMA errors;
  exog = month-in-quarter dummies + log quarter target from `targets.csv`), and target x attainment.
- **targets.csv in the forecast.** The 2026-Q3 target isn't in the data, so it is projected from earlier targets.
  The backtest projects too, and never reads the forecast quarter's real target.
- **Ensemble chosen by a rule fixed in advance:** all 6 methods are used only if their leave-one-quarter-out error
  is no worse than v1's 3. Result: all6, 1.7% vs 2.6%.
- **Churn link:** an excess-churn adjustment from the at-risk model, -$2.0M for ALL. Its backtest showed slightly
  higher error, so the rule did not apply it. It is reported as revenue at risk.
- **Full-stack MVP:** FastAPI backend + React/Grommet (HPE theme) dashboard, started with `./start.sh`
  (see below).
- **v1 vs v2 comparison:** `src/compare_versions.py` -> `outputs/version_comparison.md`. v2 reproduces v1's backtest
  exactly for the shared methods (consistency check), then compares forecasts, accuracy and the at-risk output.


## Run the dashboard (one command)

```bash
./start.sh
```

Then open **http://localhost:8000**. On first run, `start.sh` creates `.venv` and installs `requirements.txt`. If
`frontend/dist` is missing, it builds it with npm (prebuilt `dist/` is included, so Node is not needed). Then it
starts the server. The backend loads the data and computes health scores, reasons, forecast and insights in memory
(about 10 s) before the first page loads. Use `PORT=9000 ./start.sh` for another port.

| Page | What it shows |
|---|---|
| Overview | KPIs (revenue, YoY, margin, attainment, active partners, at-risk count, Q3 forecast), revenue by region, segments, top insights |
| Performance | Filters (region, tier, product, partner type, quarter range), stacked revenue by any dimension, breakdown table with margin / YoY / attainment, margin trend, searchable partner table |
| Insights | 10 insight cards, each with a chart, the key number and the takeaway, all computed from the data |
| At-risk partners | Flag summary, main drivers, how the score works, tables (at risk / quietly declining / all). Clicking a row opens a slide-over with every reason and the partner's history |
| Partner page | `/partners/PT-xxxxx`: the same content as a linkable full page |
| Q3 forecast | Forecast and 80% range per region, history + forecast band chart, per-method forecasts, backtest accuracy, ensemble choice, churn adjustment |
| Methodology | Data decisions, label, model, reasons, forecast, limitations |

**Refresh analysis** (header) re-runs the analysis on the server: `POST /api/pipeline/run` returns 202, and the UI
polls `/api/pipeline/status` every 2 s. The old results stay on screen until the new ones are ready; on failure they
are kept and the error is shown. The light/dark toggle is in the header.

**API** (JSON, served by FastAPI on the same port; interactive docs at `/docs`):
- `GET /api/meta`, `/api/kpis`, `/api/trend`, `/api/breakdown`, `/api/partners`, `/api/partners/{id}`, `/api/at-risk`,
  `/api/forecast`, `/api/insights`
- `POST /api/pipeline/run`, `GET /api/pipeline/status`
- Filters: `region`, `tier`, `product_family`, `partner_type` (comma-separated), `q_from`, `q_to`

**Develop the frontend:** run `python -m src.serve` (API on :8000) and `cd frontend && npm run dev` (Vite on :5173,
proxies `/api`). After changes, `npm run build`.

## Setup on a new computer

1. Install Python 3.9 or newer. Node.js isn't needed, because the built frontend ships with the project.
2. Copy `Hackathon_v2` without `.venv` and `frontend/node_modules`. It's about 2 MB.
3. Put the pack's `05_channel_analytics` folder next to it, or point to the data with `CHANNEL_DATA_DIR=/path/to/data`.
4. Run it:

```bash
./start.sh
```

The first run needs internet for about 1–2 minutes of package installs. After that, open http://localhost:8000.

`start.sh` now checks the Python version, checks that the three data files are there (with a clear message if not), and rebuilds a broken copied `.venv` automatically.

## Manual setup (pipeline scripts only)

Requires Python 3.9+. By default the data is read from `../05_channel_analytics/data`. To use a different folder, set
`CHANNEL_DATA_DIR` or pass `--data-dir`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Regenerate everything, including both submission files and the v1 comparison (about 20 s):

```bash
.venv/bin/python -m src.run_all
```

Or run the steps one by one:

```bash
.venv/bin/python -m src.data_prep   # load, clean, merge, aggregate -> outputs/processed/*.parquet
.venv/bin/python -m src.eda         # EDA -> outputs/eda/eda_report.md, charts/, tables/
.venv/bin/python -m src.features    # at-risk features + proxy labels -> outputs/features/
.venv/bin/python -m src.at_risk     # health score, at-risk flags, reasons -> submission/submission_at_risk.csv
.venv/bin/python -m src.forecast    # 2026-Q3 forecast + 80% interval -> submission/submission_forecast.csv
.venv/bin/python -m pytest -q       # 39 tests: leakage, submissions, v1 consistency, API
```

The two files in `submission/` go into the team folder's `05_channel_analytics/` unchanged.

## Code

| File | What it does |
|---|---|
| `src/data_prep.py` | Loads the 3 CSVs; parses dates; adds month/quarter labels, margin $ and unit price; flags outlier lines; merges partner attributes; builds the aggregate tables. `build_all()` returns them all as a dict, for reuse by models and the API. |
| `src/eda.py` | Data-quality checks and 11 analysis sections. Each section produces a table (CSV), a chart (PNG) where useful, and a takeaway. |
| `src/features.py` | Point-in-time feature matrix for the at-risk model: 57 features in 7 groups, built for any cutoff quarter using only data up to that date. `FEATURE_DOCS` documents each one. |
| `src/labels.py` | Proxy churn labels for historical cutoffs (`left_zero`, `left_token` = primary, `left_persistent`). |
| `src/feature_diagnostics.py` | Label base rates, missing values, univariate AUC, redundant pairs, a sanity-check baseline model and the declining-cohort analysis -> `outputs/features/feature_report.md`. |
| `src/at_risk.py` | Rolling-origin backtest, final calibrated ensemble, health score, at-risk flags, plain-language reasons, partner segments -> `outputs/models/partner_scores.csv`, `at_risk_report.md`. |
| `src/forecast.py` | Three forecasting methods + ensemble, rolling backtest, bias correction, 80% interval -> `outputs/models/forecast_report.md`, `charts/forecast_fan.png`. |
| `src/churn_adjustment.py` | Excess-churn adjustment linking the at-risk probabilities to the forecast, with its own backtest. |
| `src/compare_versions.py` | v1 vs v2: consistency check, forecasts, leave-one-out accuracy, at-risk diff, verdict -> `outputs/version_comparison.md`. v1 location: `V1_DIR` env var, default `../Hackathon`. |
| `src/run_all.py` | Runs every step in order; the at-risk result is passed to the forecast in memory. |
| `src/api/state.py` | `AppState`: loads the data, then computes the at-risk model, forecast, declining list and insights in memory at startup. Only the data load is mandatory; a failing section shows "Data unavailable" in the UI. |
| `src/api/analytics.py` | Pure query functions behind each endpoint (filters, KPIs, trends, breakdowns, partner list and detail, at-risk, forecast). |
| `src/api/insights.py` | The 10 insight cards, each with chart data and a takeaway built from computed numbers. |
| `src/api/app.py` | FastAPI routes, background pipeline with atomic state swap, SPA serving with deep-link fallback. |
| `src/serve.py` | `python -m src.serve` starts uvicorn. |
| `frontend/` | React 18 + TypeScript + Grommet with `grommet-theme-hpe` (HPE Design System) + Recharts. `src/chartColors.ts` holds the validated chart palette (light and dark). |
| `start.sh` | One-command launcher. |
| `experiments/model_comparison.py` | Decliner check + XGBoost / LightGBM / deeper trees / weighting vs the current ensemble, with a paired bootstrap -> `outputs/models/model_comparison.md`. Needs `xgboost`, `lightgbm` and `brew install libomp` (not installed in v2's venv); kept as a record of the v1 experiment. |
| `src/plot_style.py` | Shared chart styling (palette, fonts, formatters). |
| `tests/test_features.py` | Leakage test (features unchanged when future rows are deleted), coverage, bounds, label definition. |
| `tests/test_models.py` | Submission format checks, every flagged partner has a reason, forecast uses only past data, in-memory compute reproduces the submission files. |
| `tests/test_api.py` | Every endpoint (200, keys, no NaN), filter consistency, empty filters, at-risk/forecast match the submissions, 404s, SPA fallback, pipeline 202/409/failure handling. |

### Tables from `build_all()`

| Table | Grain |
|---|---|
| `sales` | every order line, plus `month`, `quarter`, `margin_usd`, `unit_price`, `is_outlier`, `partner_type`, `country`, `onboarded_date`, `cohort`, `tenure_months` |
| `master`, `targets` | cleaned inputs (`targets` gains quarter start/end dates) |
| `partner_month` | partner x month from onboarding to Jun 2026, **zero-filled** (revenue, margin, quantity, order lines, products, active) |
| `partner_quarter` | partner x quarter with target, attainment, margin %, product breadth |
| `partner_summary` | one row per partner as of 2026-06-30: first/last order, typical gap between orders, recency vs that gap, H1 YoY growth, last quarter vs trailing 4 quarters, ramping flag |
| `region_month`, `region_quarter`, `tier_quarter`, `product_quarter`, `region_tier_product_quarter` | revenue, margin $, margin %, quantity, order lines, active partners |

## Data decisions

- **Outlier:** one line (PT-00018, EMEA GreenLake, qty 12,800, $37.8M on 2026-03-27) is about 47x the next-largest line.
  It stays in `sales` with `is_outlier=True` and is excluded from every aggregate. Pass `--keep-outliers` to
  `data_prep` to include it.
- **Ramping partners:** onboarded less than 12 months before 2026-06-30 (`RAMPING_MONTHS`).
- **Margin %** is always revenue-weighted (`margin_usd / revenue_usd`), never an average of line percentages.

## At-risk features

**Snapshots.** Training cutoffs are 2024-Q4 .. 2026-Q1, each labelled with the following quarter. 2026-Q1 (labelled by
2026-Q2) is the holdout. The 2026-Q2 snapshot is scored for the 2026-Q3 submission and covers all 2,600 partners.
Outputs: `train_panel.parquet` (14,146 rows), `score_2026Q2.parquet`, `feature_dictionary.csv`.

**Feature groups** (full list in `outputs/features/feature_dictionary.csv`):

| Group | Examples |
|---|---|
| Revenue trend | avg revenue over 1/2/4 quarters, QoQ and 2Q growth, CV, slope, drawdown from peak |
| Order activity | order days, lines per month, active-month share, days since last order, typical gap, recency vs gap |
| Target attainment | attainment over 1/2/4 quarters, trend, consecutive misses, target YoY |
| Profit quality | revenue-weighted margin % over 2/4 quarters, margin trend, margin vs tier median |
| Product diversity | families ordered, change in families, revenue share per family, HHI |
| Year-on-year | YoY for the last quarter and the last 2 quarters (= H1 26 vs H1 25 at the scoring cutoff), decline streak, YoY vs region median |
| Static | partner type, tier, region, tenure, ramping flag |

**Robustness.** Growth rates are symmetric, `(new - old) / (new + old)`, bounded in [-1, 1]. Ratios are capped at 5. Slopes
and CVs are winsorized at 1%/99% inside each snapshot, so the raw tables are never changed. The outlier line is excluded.

**Labels.** `left_token` matches the judges' definition: no orders, or revenue below 10% of the partner's trailing
4-quarter average, in the next quarter. Historical rate 0.9-2.1% per quarter. The 2025-Q2 cutoff, the seasonal analogue
of 2026-Q3, is 2.1%.

**Known limitations.**
- The outlier threshold is computed on the full data. It affects 1 line in 2026-03.
- Tier and region come from the current master; no tier history is available.
- Most historical "leavers" are small, irregular partners who come back. The 73-partner declining cohort (YoY decline
  4+ quarters, mostly Gold/Silver in EMEA) is barely represented in the historical labels, so the model needs a
  second signal for it.

## At-risk model

- **Model:** the mean of a logistic regression and a gradient-boosting classifier (scikit-learn), trained on the 6
  historical snapshots with the `left_token` label. Probabilities are Platt-calibrated on out-of-fold backtest
  predictions, then scaled for seasonality: labels in Q3 have a leave rate 1.27x the average.
- **Backtest** (rolling origin; each quarter scored by a model trained only on earlier quarters): mean ROC-AUC
  **0.857** over 2025-Q2 .. 2026-Q1. A single feature (order lines per month) scores 0.856, so the signal is mostly
  ordering frequency and size. The model adds stability across quarters, not much lift.
- **Alternatives tested** (`experiments/model_comparison.py`): XGBoost (0.834 with scale_pos_weight=50, 0.846 without), LightGBM (0.840; 0.848 regularised; 0.821 with is_unbalance), deeper trees (0.845) and up-weighting decliners (0.838) all score below the current ensemble. The best challenger, LR + regularised LightGBM (0.858), ties with it: paired-bootstrap 95% CI for the difference is -0.004 to +0.004. Historical decliners (YoY down 2-4+ quarters) did not leave more often than partners of the same size and tier, so they are reported as a segment rather than flagged.
- **health_score** = 100 x (1 - P(no or token orders in 2026-Q3)).
- **at_risk:** the top N partners, where N = 1.5 x expected leavers. The multiplier maximised F1 in the backtest.
  The model expects 60 leavers in 2026-Q3, so **90 partners are flagged** (85 Business, 5 Silver), about 0.3% of
  channel revenue. Backtest precision is about 10% and recall about 17%: leaving is hard to predict because most
  partners who skip a quarter come back.
- **Reasons:** for every partner, each driver group (recency, frequency, size, momentum, YoY, attainment, product
  breadth, margin, volatility) is reset to the median of healthy partners in the same tier. The drop in risk ranks
  the drivers, and the top 3 are written as sentences from the partner's own numbers. No LLM is involved, so no
  figures can be invented. An LLM can later rephrase these sentences in the dashboard, grounded on the same numbers.
- **Segments:** At risk (90), Quietly declining (274: H1 YoY <= -25% and down 2+ quarters), Growing (1,075:
  YoY >= +15%), Ramping (261: onboarded < 12 months), Stable (900).

## Forecast

- **Six methods per region:**
  - seasonal ratio
  - YoY growth
  - OLS on monthly log revenue
  - ARIMA airline model (period 3 = the quarter-end spike)
  - SARIMAX with targets: AIC picks the order from (1,0,0)+trend, (0,1,1) and (1,1,0)
  - projected target x usual attainment for that quarter of the year
- **Backtest** (2025-Q2 .. 2026-Q2, raw MAPE for ALL):

  | Method | ALL MAPE |
  |---|---|
  | target x attainment | 1.2% |
  | seasonal ratio | 2.5% |
  | SARIMAX | 2.6% |
  | YoY growth | 2.7% |
  | OLS regression | 3.3% |
  | ARIMA | 8.0% (misses the Q3/Q4 pattern) |

- **Selection:** leave-one-quarter-out after bias correction, all6 scores 1.7% (regions + ALL) against core3's 2.6%,
  so all6 is used. Dropping ARIMA gives the same 1.7%; that row is reported for information only.
- **Bias correction:** -2.0%, because growth is slowing.
- **80% interval:** t-quantile x the spread of backtest log errors. Each region uses the larger of the pooled spread
  and its own.
- **Churn adjustment:** not applied (see Changes vs v1).

| Region | Forecast | 80% interval |
|---|---|---|
| APJ | $192.4M | $186.2M - $198.8M |
| EMEA | $228.3M | $220.8M - $236.2M |
| AMS | $718.9M | $689.0M - $750.0M |
| ALL | $1,139.6M | $1,112.0M - $1,167.9M |
