"""2026-Q3 revenue forecast by region with an 80% interval (v2).

    python -m src.forecast

Six methods per region:
    seasonal_ratio     last quarter x the same quarter-on-quarter ratio in previous years
    yoy_growth         same quarter last year x the recent year-on-year growth rate
    regression         OLS on monthly log revenue: trend + quarter-of-year + month-in-quarter effects
    arima              SARIMA(0,1,1)(0,1,1)_3 "airline" model on monthly log revenue (period 3 = quarter-end spike)
    sarimax            regression with ARIMA errors on monthly log revenue; exog = month-in-quarter dummies +
                       log(quarter target from targets.csv); ARIMA order picked by AIC
    target_attainment  projected quarter target x the usual attainment for that quarter of the year

targets.csv stops at 2026-Q2, so the target of the forecast quarter is always projected from earlier targets
(also in the backtest - the real target of the forecast quarter is never used).

Two ensembles are backtested: core3 (the v1 methods) and all6. Fixed rule: all6 is used only if its
leave-one-quarter-out bias-corrected error is no worse than core3's. The chosen ensemble is bias-corrected, then the
excess-churn adjustment from the at-risk model is added if its own backtest says it helps (src/churn_adjustment.py).
The 80% interval comes from the chosen ensemble's backtest errors. ALL = sum of the regions.

Writes outputs/models/forecast_backtest.csv, forecast_report.md, charts/forecast_fan.png and
submission/submission_forecast.csv.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import churn_adjustment as churn
from . import plot_style as ps
from .data_prep import REGIONS, ROOT, build_all
from .eda import md_table
from .features import shift_quarter

MODEL_DIR = ROOT / "outputs" / "models"
SUBMISSION_DIR = ROOT / "submission"
TARGET = "2026-Q3"
BACKTEST_QUARTERS = ["2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"]
CORE_METHODS = ["seasonal_ratio", "yoy_growth", "regression"]
NEW_METHODS = ["arima", "sarimax", "target_attainment"]
METHODS = CORE_METHODS + NEW_METHODS
ENSEMBLES = {"core3": CORE_METHODS, "all6": METHODS}
# Reported for information only; never eligible for selection (choosing it after seeing results would overfit).
SENSITIVITY = {"all6 without arima (info only)": [m for m in METHODS if m != "arima"]}
COVERAGE = 0.80
AREAS = REGIONS + ["ALL"]
SARIMAX_CANDIDATES = [((1, 0, 0), "ct"), ((0, 1, 1), "t"), ((1, 1, 0), "t")]


def quarter_series(region_month: pd.DataFrame) -> pd.DataFrame:
    rm = region_month.copy()
    rm["quarter"] = rm["month"].dt.year.astype(str) + "-Q" + rm["month"].dt.quarter.astype(str)
    return rm.pivot_table(index="quarter", columns="region", values="revenue_usd", aggfunc="sum", observed=True)[REGIONS]


def region_targets(tables: dict) -> pd.DataFrame:
    """Rows = quarter, cols = region: sum of partner targets."""
    pq = tables["partner_quarter"]
    return pq.pivot_table(index="quarter", columns="region", values="target_usd", aggfunc="sum", observed=True)[REGIONS]


def _quarter_start(q: str) -> pd.Timestamp:
    return pd.Period(q.replace("-", ""), freq="Q").start_time


def _quarter_of(months: pd.DatetimeIndex) -> list[str]:
    return [f"{m.year}-Q{m.quarter}" for m in months]


# ---------------------------------------------------------------- v1 methods

def seasonal_ratio(q: pd.Series, target: str) -> float:
    prev = shift_quarter(target, -1)
    ratios = []
    for k in (1, 2, 3):
        a, b = shift_quarter(target, -4 * k), shift_quarter(target, -4 * k - 1)
        if a in q.index and b in q.index:
            ratios.append(q[a] / q[b])
    return q[prev] * np.mean(ratios) if ratios and prev in q.index else np.nan


def yoy_growth(q: pd.Series, target: str) -> float:
    ly = shift_quarter(target, -4)
    recent = [shift_quarter(target, -1), shift_quarter(target, -2)]
    year_ago = [shift_quarter(c, -4) for c in recent]
    if ly not in q.index:
        return np.nan
    if all(c in q.index for c in recent + year_ago):
        g = q[recent].sum() / q[year_ago].sum()
    elif recent[0] in q.index and year_ago[0] in q.index:
        g = q[recent[0]] / q[year_ago[0]]
    else:
        return np.nan
    return q[ly] * g


def _design(months: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    t = (months.year - origin.year) * 12 + (months.month - origin.month)
    qoy = months.quarter
    miq = (months.month - 1) % 3 + 1
    return np.column_stack([np.ones(len(months)), t,
                            qoy == 2, qoy == 3, qoy == 4, miq == 2, miq == 3]).astype(float)


def regression(m: pd.Series, target: str) -> float:
    """m: monthly revenue indexed by month start, already restricted to months before the target quarter."""
    if len(m) < 12:
        return np.nan
    origin = m.index[0]
    X, y = _design(m.index, origin), np.log(m.to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_var = np.sum((y - X @ beta) ** 2) / max(len(y) - X.shape[1], 1)
    future = pd.date_range(_quarter_start(target), periods=3, freq="MS")
    return float(np.exp(_design(future, origin) @ beta + resid_var / 2).sum())


# ---------------------------------------------------------------- v2 methods

def project_target(tgt: pd.Series, target: str) -> float:
    """Target of the forecast quarter, projected from earlier targets only: target(T-4) x target(T-1) / target(T-5)."""
    ly, prev, prev_ly = shift_quarter(target, -4), shift_quarter(target, -1), shift_quarter(target, -5)
    if not all(c in tgt.index for c in (ly, prev, prev_ly)):
        return np.nan
    return float(tgt[ly] * tgt[prev] / tgt[prev_ly])


def target_attainment(q: pd.Series, tgt: pd.Series, target: str) -> float:
    proj = project_target(tgt, target)
    att = [q[c] / tgt[c] for c in (shift_quarter(target, -4), shift_quarter(target, -8))
           if c in q.index and c in tgt.index]
    return proj * float(np.mean(att)) if att and not np.isnan(proj) else np.nan


def _sum_exp_forecast(res, steps: int, exog=None) -> float:
    fc = res.get_forecast(steps=steps, exog=exog)
    mean, var = np.asarray(fc.predicted_mean), np.asarray(fc.var_pred_mean)
    return float(np.exp(mean + var / 2).sum())


def arima(m: pd.Series, target: str) -> float:
    if len(m) < 15:
        return np.nan
    y = np.log(m.to_numpy())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 3)).fit(disp=False)
        return _sum_exp_forecast(res, 3)
    except Exception:  # noqa: BLE001 - a failed fit just drops this method for this quarter
        return np.nan


def _sarimax_exog(months: pd.DatetimeIndex, quarter_target: dict[str, float]) -> np.ndarray:
    miq = (months.month - 1) % 3 + 1
    log_t = np.log([quarter_target[q] / 3 for q in _quarter_of(months)])
    return np.column_stack([miq == 2, miq == 3, log_t]).astype(float)


def sarimax(m: pd.Series, tgt: pd.Series, target: str) -> float:
    proj = project_target(tgt, target)
    if len(m) < 15 or np.isnan(proj):
        return np.nan
    qt = tgt.to_dict()
    qt[target] = proj
    if any(q not in qt for q in _quarter_of(m.index)):
        return np.nan
    y = np.log(m.to_numpy())
    X = _sarimax_exog(m.index, qt)
    X_future = _sarimax_exog(pd.date_range(_quarter_start(target), periods=3, freq="MS"), qt)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for order, trend in SARIMAX_CANDIDATES:
            try:
                res = SARIMAX(y, exog=X, order=order, trend=trend).fit(disp=False, maxiter=200)
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(res.aic) and (best is None or res.aic < best.aic):
                best = res
    if best is None:
        return np.nan
    return _sum_exp_forecast(best, 3, exog=X_future)


# ---------------------------------------------------------------- forecast + backtest

def forecast_all(quarterly: pd.DataFrame, monthly: pd.DataFrame, targets_q: pd.DataFrame, target: str) -> pd.DataFrame:
    """Forecast `target` for each region using only data before it. Rows = regions + ALL, cols = methods + ensembles."""
    q_hist = quarterly[quarterly.index < target]
    m_hist = monthly[monthly.index < _quarter_start(target)]
    t_hist = targets_q[targets_q.index < target]
    rows = {}
    for r in REGIONS:
        rows[r] = {"seasonal_ratio": seasonal_ratio(q_hist[r], target),
                   "yoy_growth": yoy_growth(q_hist[r], target),
                   "regression": regression(m_hist[r], target),
                   "arima": arima(m_hist[r], target),
                   "sarimax": sarimax(m_hist[r], t_hist[r], target),
                   "target_attainment": target_attainment(q_hist[r], t_hist[r], target)}
    out = pd.DataFrame(rows).T[METHODS]
    for name, cols in {**ENSEMBLES, **SENSITIVITY}.items():
        out[name] = out[cols].mean(axis=1)
    out.loc["ALL"] = out.sum(axis=0, min_count=len(REGIONS))
    return out


def backtest(quarterly: pd.DataFrame, monthly: pd.DataFrame, targets_q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tq in BACKTEST_QUARTERS:
        fc = forecast_all(quarterly, monthly, targets_q, tq)
        actual = quarterly.loc[tq].copy()
        actual["ALL"] = actual.sum()
        for area in AREAS:
            for method in METHODS + list(ENSEMBLES) + list(SENSITIVITY):
                f = fc.loc[area, method]
                rows.append({"quarter": tq, "area": area, "method": method, "forecast": f, "actual": actual[area],
                             "pct_error": f / actual[area] - 1, "log_error": np.log(actual[area] / f)})
    return pd.DataFrame(rows)


def bias_and_scale(bt: pd.DataFrame, ensemble: str, exclude: str | None = None) -> tuple[float, dict[str, float]]:
    """Bias correction and 80% log-scale half-width from an ensemble's backtest errors.

    bias  = mean regional log error log(actual / forecast); negative = the methods over-forecast.
    width = t-quantile x std of the regional errors around that bias: the larger of the pooled std (15 errors)
            and the region's own std. ALL uses its own errors, floored at half the pooled regional width.
    `exclude` drops one backtest quarter (for the leave-one-out check)."""
    ens = bt[(bt["method"] == ensemble) & (bt["quarter"] != exclude)]
    reg = ens[ens["area"] != "ALL"]["log_error"]
    all_err = ens[ens["area"] == "ALL"]["log_error"]
    bias = float(reg.mean())
    t_reg = stats.t.ppf(0.5 + COVERAGE / 2, df=len(reg) - 1)
    t_all = stats.t.ppf(0.5 + COVERAGE / 2, df=len(all_err) - 1)
    own = ens[ens["area"] != "ALL"].groupby("area")["log_error"].std(ddof=1)
    half = {r: t_reg * max(reg.std(ddof=1), own[r]) for r in REGIONS}
    half["ALL"] = max(t_all * all_err.std(ddof=1), 0.5 * t_reg * reg.std(ddof=1))
    return bias, half


def loo_check(bt: pd.DataFrame, ensemble: str) -> pd.DataFrame:
    """Leave one backtest quarter out: estimate bias/width on the others, then score the held-out quarter.
    ALL's corrected forecast is the sum of the corrected regions."""
    rows = []
    ens = bt[bt["method"] == ensemble]
    for q in BACKTEST_QUARTERS:
        bias, half = bias_and_scale(bt, ensemble, exclude=q)
        g = ens[ens["quarter"] == q].set_index("area")
        corrected = g.loc[REGIONS, "forecast"] * np.exp(bias)
        corrected["ALL"] = corrected.sum()
        for area in AREAS:
            actual, raw = g.loc[area, "actual"], g.loc[area, "forecast"]
            rows.append({"quarter": q, "area": area, "actual": actual, "forecast_corrected": corrected[area],
                         "ape_raw": abs(raw / actual - 1), "ape_bias_corrected": abs(corrected[area] / actual - 1),
                         "inside_80": abs(np.log(actual / corrected[area])) <= half[area]})
    return pd.DataFrame(rows)


def choose_ensemble(bt: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Fixed rule: all6 only if its leave-one-out bias-corrected MAPE (regions + ALL) is <= core3's."""
    scores = pd.DataFrame({e: loo_check(bt, e).groupby("area")["ape_bias_corrected"].mean().reindex(AREAS)
                           for e in ENSEMBLES}).T
    scores["mean"] = scores[AREAS].mean(axis=1)
    chosen = "all6" if scores.loc["all6", "mean"] <= scores.loc["core3", "mean"] else "core3"
    for name in SENSITIVITY:
        scores.loc[name, AREAS] = loo_check(bt, name).groupby("area")["ape_bias_corrected"].mean().reindex(AREAS)
        scores.loc[name, "mean"] = scores.loc[name, AREAS].mean()
    return chosen, scores


def validate_forecast(sub: pd.DataFrame) -> None:
    assert list(sub.columns) == ["region", "forecast_revenue_usd", "lo80_usd", "hi80_usd"], sub.columns
    assert list(sub["region"]) == ["APJ", "EMEA", "AMS", "ALL"], sub["region"].tolist()
    assert (sub["lo80_usd"] <= sub["forecast_revenue_usd"]).all() and (sub["forecast_revenue_usd"] <= sub["hi80_usd"]).all()
    assert np.isfinite(sub[["forecast_revenue_usd", "lo80_usd", "hi80_usd"]].to_numpy()).all()
    assert abs(sub.set_index("region").loc[["APJ", "EMEA", "AMS"], "forecast_revenue_usd"].sum()
               - sub.set_index("region").loc["ALL", "forecast_revenue_usd"]) < 1


# ---------------------------------------------------------------- main

def load_inputs(tables: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = tables or build_all()
    monthly = tables["region_month"].pivot(index="month", columns="region", values="revenue_usd")[REGIONS]
    return quarter_series(tables["region_month"]), monthly, region_targets(tables)


@dataclass
class ForecastResult:
    """Everything the forecast step produces, kept in memory (used by the API)."""
    sub: pd.DataFrame
    fc: pd.DataFrame
    bt: pd.DataFrame
    loo: pd.DataFrame
    chosen: str
    selection: pd.DataFrame
    bias: float
    half: dict
    churn_bt: pd.DataFrame
    apply_churn: bool
    churn_live: dict
    quarterly: pd.DataFrame
    targets_q: pd.DataFrame


def compute(tables: dict | None = None, at_risk_result=None) -> ForecastResult:
    """Backtest, select, forecast and adjust - no file I/O except reading the at-risk outputs when no
    in-memory at-risk result is passed."""
    quarterly, monthly, targets_q = load_inputs(tables)

    bt = backtest(quarterly, monthly, targets_q)
    chosen, selection = choose_ensemble(bt)
    bias, half = bias_and_scale(bt, chosen)
    loo = loo_check(bt, chosen)

    fc = forecast_all(quarterly, monthly, targets_q, TARGET)
    base = fc.loc[REGIONS, chosen] * np.exp(bias)

    # Excess-churn adjustment (link to the at-risk model).
    if at_risk_result is not None:
        oof, scores = at_risk_result.oof, at_risk_result.scores
    else:
        oof = pd.read_parquet(MODEL_DIR / "at_risk_oof.parquet")
        scores = pd.read_csv(MODEL_DIR / "partner_scores.csv")
    shares = churn.historical_shares(oof)
    churn_bt = churn.backtest(shares, loo.set_index(["quarter", "area"]).rename(columns={"forecast_corrected": "forecast"}))
    apply_churn = churn.decide(churn_bt)
    churn_live = churn.live(scores, shares, base)
    final = base + (churn_live["adjustment"][REGIONS] if apply_churn else 0)

    fc["bias_corrected"] = np.append(base.to_numpy(), base.sum())
    fc["final"] = np.append(final.to_numpy(), final.sum())

    sub = pd.DataFrame({"region": ["APJ", "EMEA", "AMS", "ALL"]})
    sub["forecast_revenue_usd"] = [fc.loc[a, "final"] for a in sub["region"]]
    sub["lo80_usd"] = [f * np.exp(-half[a]) for a, f in zip(sub["region"], sub["forecast_revenue_usd"])]
    sub["hi80_usd"] = [f * np.exp(half[a]) for a, f in zip(sub["region"], sub["forecast_revenue_usd"])]
    cols = ["forecast_revenue_usd", "lo80_usd", "hi80_usd"]
    sub[cols] = sub[cols].round(0)
    sub.loc[sub["region"] == "ALL", "forecast_revenue_usd"] = sub.loc[sub["region"] != "ALL", "forecast_revenue_usd"].sum()
    sub[cols] = sub[cols].astype("int64")
    validate_forecast(sub)
    return ForecastResult(sub=sub, fc=fc, bt=bt, loo=loo, chosen=chosen, selection=selection, bias=bias, half=half,
                          churn_bt=churn_bt, apply_churn=apply_churn, churn_live=churn_live, quarterly=quarterly,
                          targets_q=targets_q)


def run(tables: dict | None = None, at_risk_result=None) -> pd.DataFrame:
    """compute() plus every file the pipeline writes."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    r = compute(tables, at_risk_result)
    r.bt.to_csv(MODEL_DIR / "forecast_backtest.csv", index=False)
    r.sub.to_csv(SUBMISSION_DIR / "submission_forecast.csv", index=False)
    r.churn_bt.to_csv(MODEL_DIR / "churn_adjustment_backtest.csv", index=False)
    r.loo.to_csv(MODEL_DIR / "forecast_loo.csv", index=False)
    meta = {"chosen_ensemble": r.chosen, "bias_log": float(r.bias),
            "half_width_log": {k: float(v) for k, v in r.half.items()},
            "churn_adjustment_applied": bool(r.apply_churn),
            "churn_adjustment_usd": {k: float(v) for k, v in r.churn_live["adjustment"].items()}}
    (MODEL_DIR / "forecast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    fan_chart(r.quarterly, r.sub)
    write_report(r.bt, r.fc, r.sub, r.half, r.quarterly, r.bias, r.loo, r.chosen, r.selection, r.churn_bt,
                 r.apply_churn, r.churn_live)
    return r.sub


def fan_chart(quarterly: pd.DataFrame, sub: pd.DataFrame) -> None:
    fig, axes = ps.plt.subplots(1, 4, figsize=(15, 3.8))
    qs = list(quarterly.index)
    for ax, area in zip(axes, ["AMS", "APJ", "EMEA", "ALL"]):
        hist = quarterly.sum(axis=1) if area == "ALL" else quarterly[area]
        row = sub.set_index("region").loc[area]
        color = ps.REGION_COLORS.get(area, ps.SERIES[3])
        ax.plot(qs, hist.values, color=color, marker="o", markersize=4)
        ax.plot([qs[-1], TARGET], [hist.iloc[-1], row["forecast_revenue_usd"]], color=color, linestyle="--")
        ax.errorbar([TARGET], [row["forecast_revenue_usd"]],
                    yerr=[[row["forecast_revenue_usd"] - row["lo80_usd"]], [row["hi80_usd"] - row["forecast_revenue_usd"]]],
                    fmt="o", color=color, markersize=7, capsize=5, markeredgecolor=ps.SURFACE, markeredgewidth=2)
        ax.yaxis.set_major_formatter(ps.money_m)
        ax.set_xticks([qs[0], qs[3], qs[7], TARGET])
        ax.set_title(f"{area}: \\${row['forecast_revenue_usd'] / 1e6:,.0f}M "
                     f"(\\${row['lo80_usd'] / 1e6:,.0f}-{row['hi80_usd'] / 1e6:,.0f}M)", fontsize=11)
    fig.suptitle(f"Quarterly revenue and {TARGET} forecast with 80% interval", x=0.01, ha="left", fontweight="semibold")
    ps.save(fig, MODEL_DIR / "charts" / "forecast_fan.png")


def write_report(bt, fc, sub, half, quarterly, bias, loo, chosen, selection, churn_bt, apply_churn, churn_live) -> None:
    L = ["# 2026-Q3 forecast report (v2)", ""]

    def t(df):
        L.extend([md_table(df), ""])

    pct = lambda v: f"{v:.1%}"  # noqa: E731
    money = lambda v: f"${v / 1e6:,.1f}M"  # noqa: E731

    order = METHODS + list(ENSEMBLES) + list(SENSITIVITY)
    acc = bt.assign(ape=bt["pct_error"].abs()).groupby(["method", "area"])["ape"].mean().unstack()[AREAS].reindex(order)
    acc["mean_regions"] = acc[REGIONS].mean(axis=1)
    L += ["## 1. Backtest: mean absolute % error (raw, before bias correction)", "",
          f"Quarters {', '.join(BACKTEST_QUARTERS)}, each forecast with data before it only. Targets for the forecast "
          "quarter are projected, never read from targets.csv.", ""]
    t(acc.map(pct).reset_index())
    L += ["ALL-region % error by quarter and method (positive = too high):", ""]
    all_err = bt[(bt["area"] == "ALL") & bt["method"].isin(order)].pivot(index="method", columns="quarter",
                                                                         values="pct_error").reindex(order)
    t(all_err.map(lambda v: f"{v:+.1%}").reset_index())

    L += ["## 2. Ensemble selection (rule fixed in advance)", "",
          "Leave-one-quarter-out, bias-corrected mean absolute % error. all6 is used only if it is no worse than core3. "
          "The last row is information only and cannot be selected.", ""]
    t(selection.map(pct).reset_index().rename(columns={"index": "ensemble"}))
    L += [f"**Chosen: {chosen}.**", ""]

    lsum = loo.groupby("area")[["ape_raw", "ape_bias_corrected", "inside_80"]].mean().reindex(AREAS)
    L += ["## 3. Bias correction and interval", "",
          f"Mean regional log error of {chosen}: {bias:+.3f} ({np.exp(bias) - 1:+.1%}); the forecast is multiplied by "
          "this factor. Leave-one-quarter-out check:", ""]
    t(lsum.reset_index().assign(ape_raw=lambda d: d["ape_raw"].map(pct),
                                ape_bias_corrected=lambda d: d["ape_bias_corrected"].map(pct),
                                inside_80=lambda d: d["inside_80"].map(lambda v: f"{v:.0%}")))

    L += ["## 4. Churn adjustment (link to the at-risk model)", "",
          "Expected revenue-loss share = sum of p_leave x average quarterly revenue / regional revenue. The time series "
          "already contains the normal share, so only the excess is applied.", ""]
    shares = pd.DataFrame({"normal share (backtest mean)": churn_live["share_normal"],
                           "share now": churn_live["share_now"],
                           "adjustment": churn_live["adjustment"][REGIONS],
                           "flagged partners' avg quarterly revenue": churn_live["flagged_revenue_avg_4q"]})
    shares.loc["ALL"] = [np.nan, np.nan, churn_live["adjustment"]["ALL"], churn_live["flagged_revenue_avg_4q"].sum()]
    t(shares.reset_index().rename(columns={"index": "area"}).assign(
        **{"normal share (backtest mean)": lambda d: d["normal share (backtest mean)"].map(lambda v: "-" if pd.isna(v) else f"{v:.3%}"),
           "share now": lambda d: d["share now"].map(lambda v: "-" if pd.isna(v) else f"{v:.3%}"),
           "adjustment": lambda d: d["adjustment"].map(money),
           "flagged partners' avg quarterly revenue": lambda d: d["flagged partners' avg quarterly revenue"].map(money)}))
    if len(churn_bt):
        cb = churn_bt.groupby("area")[["ape_without", "ape_with"]].mean().reindex(AREAS)
        L += ["Backtest (quarters after each out-of-fold cutoff, normal share estimated leave-one-out):", ""]
        t(cb.map(lambda v: f"{v:.2%}").reset_index())
    L += [f"**Churn adjustment applied: {'yes' if apply_churn else 'no'}** "
          f"(rule: apply only if the backtest error does not increase).", ""]

    L += [f"## 5. {TARGET} forecast by method", ""]
    t(fc.map(money).reset_index().rename(columns={"index": "area"}))

    s = sub.copy()
    q2 = quarterly.loc["2026-Q2"].to_dict()
    q2["ALL"] = sum(q2.values())
    q3ly = quarterly.loc["2025-Q3"].to_dict()
    q3ly["ALL"] = sum(q3ly.values())
    s["vs 2026-Q2"] = [f / q2[a] - 1 for a, f in zip(s["region"], s["forecast_revenue_usd"])]
    s["vs 2025-Q3"] = [f / q3ly[a] - 1 for a, f in zip(s["region"], s["forecast_revenue_usd"])]
    s["interval"] = [f"-{1 - np.exp(-half[a]):.1%} / +{np.exp(half[a]) - 1:.1%}" for a in s["region"]]
    for c in ["forecast_revenue_usd", "lo80_usd", "hi80_usd"]:
        s[c] = s[c].map(money)
    for c in ["vs 2026-Q2", "vs 2025-Q3"]:
        s[c] = s[c].map(lambda v: f"{v:+.1%}")
    L += ["## 6. Submission", ""]
    t(s)
    L += ["![Forecast](charts/forecast_fan.png)", "",
          "## Limitations", "",
          "- 30 months of history: five backtest quarters, so method and ensemble choices rest on few points.",
          "- ARIMA/SARIMAX are fitted on 15-29 monthly points; with so little data their parameters are unstable, "
          "which is why they enter as ensemble members rather than replacing the simple methods.",
          "- The Q3 target is projected (targets.csv ends at 2026-Q2); the target-based methods inherit that projection's error.",
          "- The bias correction assumes growth keeps slowing at the recent pace.", ""]
    (MODEL_DIR / "forecast_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    run()
    print((MODEL_DIR / "forecast_report.md").read_text())


if __name__ == "__main__":
    main()
