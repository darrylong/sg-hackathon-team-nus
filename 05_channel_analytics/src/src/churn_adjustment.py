"""Excess-churn adjustment: link the at-risk model to the revenue forecast.

The time-series forecasts already contain the churn that normally happens, so only the *excess* is applied:

    expected loss share  s  = sum_i p_i x E_i / sum_i E_i          (per region)
        p_i = calibrated probability that partner i leaves next quarter (no seasonal scaling, so it is comparable
              with the historical out-of-fold probabilities; the time series already carries seasonality)
        E_i = partner i's average quarterly revenue over the last 4 quarters
    normal share         s0 = mean of s over the historical backtest cutoffs
    adjustment              = -(s - s0) x baseline forecast        (negative when more churn than usual)

Backtested on 2025-Q3 .. 2026-Q2 with out-of-fold probabilities (s0 estimated leave-one-out) and applied to the
submission only if it does not increase the error.
"""
from __future__ import annotations

import pandas as pd

from .data_prep import REGIONS
from .features import shift_quarter


def loss_shares(df: pd.DataFrame, p_col: str, e_col: str) -> pd.Series:
    """Expected revenue-loss share per region for one snapshot."""
    g = df.assign(loss=df[p_col] * df[e_col].fillna(0), e=df[e_col].fillna(0)).groupby("region")
    return (g["loss"].sum() / g["e"].sum()).reindex(REGIONS)


def historical_shares(oof: pd.DataFrame) -> pd.DataFrame:
    """Rows = cutoff, cols = region: expected loss share from out-of-fold probabilities."""
    return pd.DataFrame({c: loss_shares(g, "p", "baseline_4q_revenue") for c, g in oof.groupby("cutoff")}).T


def adjustment(share_now: pd.Series, share_normal: pd.Series, forecast: pd.Series) -> pd.Series:
    """Per-region $ adjustment plus ALL (= sum of regions)."""
    adj = -(share_now - share_normal) * forecast.reindex(REGIONS)
    adj["ALL"] = adj[REGIONS].sum()
    return adj


def backtest(shares: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    """Score the adjustment on quarters that follow each OOF cutoff.

    corrected: rows (quarter, area) with columns forecast (bias-corrected, leave-one-out) and actual."""
    rows = []
    for cutoff in shares.index:
        q = shift_quarter(cutoff, 1)
        if q not in corrected.index.get_level_values("quarter"):
            continue
        normal = shares.drop(index=cutoff).mean()
        f = corrected.loc[q, "forecast"]
        actual = corrected.loc[q, "actual"]
        adj = adjustment(shares.loc[cutoff], normal, f[REGIONS])
        f_all = f[REGIONS].sum()
        for area in REGIONS + ["ALL"]:
            base = f_all if area == "ALL" else f[area]
            rows.append({"quarter": q, "area": area, "adjustment": adj[area],
                         "ape_without": abs(base / actual[area] - 1),
                         "ape_with": abs((base + adj[area]) / actual[area] - 1)})
    return pd.DataFrame(rows)


def live(scores: pd.DataFrame, shares: pd.DataFrame, forecast: pd.Series) -> dict:
    """Adjustment for the target quarter from the current partner scores."""
    now = loss_shares(scores, "p_leave_unseasonal", "revenue_avg_4q")
    normal = shares.mean()
    adj = adjustment(now, normal, forecast)
    flagged = scores[scores["at_risk"] == 1]
    return {
        "share_now": now, "share_normal": normal, "adjustment": adj,
        "flagged_revenue_avg_4q": flagged.groupby("region")["revenue_avg_4q"].sum().reindex(REGIONS).fillna(0),
        "expected_loss_seasonal": (scores["p_leave"] * scores["revenue_avg_4q"].fillna(0))
        .groupby(scores["region"]).sum().reindex(REGIONS),
    }


def decide(bt: pd.DataFrame) -> bool:
    """Apply the adjustment only if it does not increase the mean absolute % error."""
    return bool(len(bt)) and bt["ape_with"].mean() <= bt["ape_without"].mean() + 1e-12

