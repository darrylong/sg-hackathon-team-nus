"""Query functions behind the API: pure functions of (state, parameters) returning JSON-ready dicts."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..data_prep import PARTNER_TYPES, PRODUCTS, REGIONS, TIERS, aggregate
from ..features import shift_quarter

DIMS = ["region", "tier", "product_family", "partner_type"]
PARTNER_DIMS = ["region", "tier", "partner_type"]
SEGMENTS = ["At risk", "Quietly declining", "Ramping (new)", "Stable", "Growing"]
PARTNER_SORTS = ["revenue_range", "health_score", "p_leave", "risk_rank", "yoy_h1_pct", "margin_pct",
                 "days_since_last_order", "partner_name", "revenue_q2_2026"]
METHOD_LABELS = {
    "seasonal_ratio": "Seasonal ratio", "yoy_growth": "YoY growth", "regression": "Monthly regression (OLS)",
    "arima": "ARIMA (airline)", "sarimax": "SARIMAX with targets", "target_attainment": "Target x attainment",
    "core3": "Ensemble: v1 methods (3)", "all6": "Ensemble: all methods (6)",
    "all6 without arima (info only)": "Ensemble without ARIMA (info only)",
    "bias_corrected": "Chosen ensemble, bias-corrected", "final": "Final forecast",
}


# ---------------------------------------------------------------- JSON + filters

def clean(obj):
    """Recursively convert numpy/pandas values to JSON-safe Python values; NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray, pd.Index)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) or math.isinf(obj) else float(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):  # dates in this data have no time part
        return obj.date().isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    return obj


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict("records"))


@dataclass
class Filters:
    region: list[str] = field(default_factory=list)
    tier: list[str] = field(default_factory=list)
    product_family: list[str] = field(default_factory=list)
    partner_type: list[str] = field(default_factory=list)
    q_from: str | None = None
    q_to: str | None = None

    @classmethod
    def parse(cls, region=None, tier=None, product_family=None, partner_type=None, q_from=None, q_to=None) -> "Filters":
        split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else []  # noqa: E731
        return cls(split(region), split(tier), split(product_family), split(partner_type), q_from or None, q_to or None)

    def no_range(self) -> "Filters":
        return replace(self, q_from=None, q_to=None)


def filter_sales(state, f: Filters) -> pd.DataFrame:
    df = state.sales
    for dim in DIMS:
        vals = getattr(f, dim)
        if vals:
            df = df[df[dim].isin(vals)]
    if f.q_from:
        df = df[df["quarter"] >= f.q_from]
    if f.q_to:
        df = df[df["quarter"] <= f.q_to]
    return df


def filter_pq(state, f: Filters) -> pd.DataFrame:
    df = state.pq
    for dim in PARTNER_DIMS:
        vals = getattr(f, dim)
        if vals:
            df = df[df[dim].isin(vals)]
    if f.q_from:
        df = df[df["quarter"] >= f.q_from]
    if f.q_to:
        df = df[df["quarter"] <= f.q_to]
    return df


def filter_partner_rows(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    for dim in PARTNER_DIMS:
        vals = getattr(f, dim)
        if vals:
            df = df[df[dim].isin(vals)]
    return df


def all_quarters(state) -> list[str]:
    return sorted(state.sales["quarter"].unique())


def last_quarter(state, f: Filters) -> str:
    qs = all_quarters(state)
    return min(f.q_to, qs[-1]) if f.q_to else qs[-1]


# ---------------------------------------------------------------- meta + KPIs

def meta(state) -> dict:
    s = state.tables["sales"]
    return clean({
        "regions": REGIONS, "tiers": TIERS, "products": PRODUCTS, "partner_types": PARTNER_TYPES,
        "quarters": all_quarters(state), "segments": SEGMENTS,
        "data_from": s["order_date"].min(), "data_to": s["order_date"].max(),
        "partners": len(state.tables["master"]), "order_lines": len(s),
        "outlier_lines_excluded": int(s["is_outlier"].sum()),
        "generated_at": state.generated_at, "errors": state.errors,
        "sections": {"at_risk": state.at_risk is not None, "forecast": state.forecast is not None,
                     "insights": state.insights is not None},
    })


def kpis(state, f: Filters) -> dict:
    lq = last_quarter(state, f)
    ly = shift_quarter(lq, -4)
    s_nr = filter_sales(state, f.no_range())
    cur, prev = s_nr[s_nr["quarter"] == lq], s_nr[s_nr["quarter"] == ly]
    rev, rev_ly = cur["revenue_usd"].sum(), prev["revenue_usd"].sum()
    s_range = filter_sales(state, f)

    attainment = None
    if not f.product_family:
        pq = filter_pq(state, f.no_range())
        pq = pq[(pq["quarter"] == lq) & pq["target_usd"].notna()]
        if pq["target_usd"].sum() > 0:
            attainment = pq["revenue_usd"].sum() / pq["target_usd"].sum()

    at_risk = None
    if state.at_risk is not None:
        sc = filter_partner_rows(state.at_risk.scores, f)
        if f.product_family:
            sc = sc[sc["partner_id"].isin(s_range["partner_id"].unique())]
        at_risk = {"flagged": int(sc["at_risk"].sum()), "partners": len(sc),
                   "declining": int((sc["segment"] == "Quietly declining").sum())}

    fc = None
    if state.forecast is not None and not (f.tier or f.product_family or f.partner_type):
        area = f.region[0] if len(f.region) == 1 else ("ALL" if not f.region else None)
        if area:
            row = state.forecast.sub.set_index("region").loc[area]
            fc = {"area": area, "quarter": "2026-Q3", "forecast": row["forecast_revenue_usd"],
                  "lo80": row["lo80_usd"], "hi80": row["hi80_usd"]}

    return clean({
        "quarter": lq, "revenue": rev, "revenue_last_year": rev_ly,
        "yoy": rev / rev_ly - 1 if rev_ly > 0 else None,
        "margin_pct": cur["margin_usd"].sum() / rev * 100 if rev > 0 else None,
        "active_partners": cur["partner_id"].nunique(),
        "attainment": attainment, "attainment_note": "not available with a product filter (targets are per partner)"
        if f.product_family else None,
        "revenue_in_range": s_range["revenue_usd"].sum(),
        "at_risk": at_risk, "forecast": fc,
    })


# ---------------------------------------------------------------- performance

def trend(state, f: Filters, grain: str = "quarter", by: str = "region") -> dict:
    if by not in DIMS:
        raise ValueError(f"by must be one of {DIMS}")
    s = filter_sales(state, f)
    col = "quarter" if grain == "quarter" else "month"
    if s.empty:
        return {"grain": grain, "by": by, "keys": [], "rows": []}
    rev = s.groupby([col, by])["revenue_usd"].sum().unstack(by).fillna(0)
    tot = s.groupby(col).agg(revenue=("revenue_usd", "sum"), margin=("margin_usd", "sum"))
    order = {"region": REGIONS, "tier": TIERS, "product_family": PRODUCTS, "partner_type": PARTNER_TYPES}[by]
    keys = [k for k in order if k in rev.columns]
    rows = []
    for period, r in rev[keys].iterrows():
        label = period if grain == "quarter" else pd.Timestamp(period).strftime("%Y-%m")
        rows.append({"period": label, **r.to_dict(), "total": tot.loc[period, "revenue"],
                     "margin_pct": tot.loc[period, "margin"] / tot.loc[period, "revenue"] * 100})
    return clean({"grain": grain, "by": by, "keys": keys, "rows": rows})


def breakdown(state, f: Filters, dim: str = "region") -> dict:
    if dim not in DIMS:
        raise ValueError(f"dim must be one of {DIMS}")
    s = filter_sales(state, f)
    if s.empty:
        return {"dim": dim, "rows": [], "growth_label": None, "attainment_label": None}
    agg = aggregate(s, dim).set_index(dim)
    lq = last_quarter(state, f)
    recent = [shift_quarter(lq, -1), lq]
    prior = [shift_quarter(q, -4) for q in recent]
    s_nr = filter_sales(state, f.no_range())
    h_now = s_nr[s_nr["quarter"].isin(recent)].groupby(dim)["revenue_usd"].sum()
    h_prev = s_nr[s_nr["quarter"].isin(prior)].groupby(dim)["revenue_usd"].sum()
    out = pd.DataFrame({
        "revenue": agg["revenue_usd"], "share": agg["revenue_usd"] / agg["revenue_usd"].sum(),
        "margin_pct": agg["margin_pct"], "partners": agg["partners"], "order_lines": agg["order_lines"],
        "growth_yoy": h_now / h_prev.where(h_prev > 0) - 1,
    })
    att_label = None
    if dim in PARTNER_DIMS and not f.product_family:
        last4 = [shift_quarter(lq, -i) for i in range(3, -1, -1)]
        pq = filter_pq(state, f.no_range())
        pq = pq[pq["quarter"].isin(last4) & pq["target_usd"].notna()]
        g = pq.groupby(dim)
        out["attainment"] = g["revenue_usd"].sum() / g["target_usd"].sum()
        out["pct_partner_quarters_hit"] = g["attainment"].apply(lambda a: (a >= 1).mean())
        att_label = f"{last4[0]} .. {last4[-1]}"
    out = out.sort_values("revenue", ascending=False).rename_axis("key").reset_index()
    return clean({"dim": dim, "rows": records(out),
                  "growth_label": f"{recent[0]}..{recent[1]} vs {prior[0]}..{prior[1]}", "attainment_label": att_label})


def partners(state, f: Filters, search: str | None = None, segment: str | None = None, sort: str = "revenue_range",
             desc: bool = True, page: int = 1, page_size: int = 25) -> dict:
    if state.at_risk is None:
        return {"total": 0, "page": page, "page_size": page_size, "rows": [], "available": False}
    df = filter_partner_rows(state.at_risk.scores, f)
    s = filter_sales(state, f)
    if f.product_family:
        df = df[df["partner_id"].isin(s["partner_id"].unique())]
    per = s.groupby("partner_id").agg(revenue_range=("revenue_usd", "sum"), margin=("margin_usd", "sum"))
    df = df.assign(revenue_range=df["partner_id"].map(per["revenue_range"]).fillna(0),
                   margin_pct=df["partner_id"].map(per["margin"] / per["revenue_range"].where(per["revenue_range"] > 0) * 100))
    if search:
        q = search.strip().lower()
        df = df[df["partner_name"].str.lower().str.contains(q, regex=False) | df["partner_id"].str.lower().str.contains(q, regex=False)]
    if segment:
        df = df[df["segment"] == segment]
    sort = sort if sort in PARTNER_SORTS else "revenue_range"
    df = df.sort_values(sort, ascending=not desc, na_position="last")
    total = len(df)
    page = max(page, 1)
    page_size = min(max(page_size, 1), 200)
    rows = df.iloc[(page - 1) * page_size: page * page_size]
    cols = ["partner_id", "partner_name", "region", "tier", "partner_type", "country", "segment", "health_score",
            "p_leave", "at_risk", "risk_rank", "revenue_range", "margin_pct", "revenue_q2_2026", "yoy_h1_pct",
            "days_since_last_order", "reason_1"]
    return clean({"total": total, "page": page, "page_size": page_size, "rows": records(rows[cols]), "available": True})


def partner_detail(state, pid: str) -> dict | None:
    master = state.tables["master"].set_index("partner_id")
    if pid not in master.index:
        return None
    m = master.loc[pid]
    profile = {"partner_id": pid, "partner_name": m["partner_name"], "partner_type": str(m["partner_type"]),
               "tier": str(m["tier"]), "region": str(m["region"]), "country": m["country"],
               "onboarded_date": m["onboarded_date"]}

    score, reasons, signals = None, [], None
    if state.at_risk is not None:
        sc = state.at_risk.scores.set_index("partner_id").loc[pid]
        score = {k: sc[k] for k in ["health_score", "p_leave", "risk_rank", "at_risk", "segment", "revenue_avg_4q",
                                    "revenue_q2_2026", "yoy_h1_pct", "last_order", "days_since_last_order"]}
        score["partners"] = len(state.at_risk.scores)
        for k in (1, 2, 3):
            text = sc.get(f"reason_{k}")
            if isinstance(text, str) and text:
                reasons.append({"text": text, "driver": sc.get(f"driver_{k}"), "points": sc.get(f"driver_{k}_pts")})
        X = state.at_risk.X.set_index("partner_id")
        x = X.loc[pid]
        peers = X[X["tier"].astype(str) == profile["tier"]]
        sig_cols = {"lines_per_month_6m": "Order lines per month (6 m)", "active_month_share_6m": "Months with orders (6 m)",
                    "days_since_last_order": "Days since last order", "typical_gap_days": "Typical days between orders",
                    "att_avg_4q": "Target attainment (4 q)", "n_products_2q": "Product families (2 q)",
                    "margin_pct_4q": "Margin % (4 q)", "rev_cv_6m": "Revenue volatility (CV, 6 m)",
                    "tenure_months": "Months since onboarding"}
        signals = [{"key": k, "label": v, "value": x[k], "tier_median": peers[k].median()} for k, v in sig_cols.items()]

    pq = state.pq[state.pq["partner_id"] == pid]
    quarterly = pq[["quarter", "revenue_usd", "target_usd", "attainment", "margin_pct", "order_lines", "n_products"]]
    pm = state.tables["partner_month"]
    pm = pm[pm["partner_id"] == pid].assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
    monthly = pm[["month", "revenue_usd", "order_lines"]]
    s = state.sales[state.sales["partner_id"] == pid]
    last4 = all_quarters(state)[-4:]
    mix = s[s["quarter"].isin(last4)].groupby("product_family")["revenue_usd"].sum()
    mix = pd.DataFrame({"product": mix.index, "revenue": mix.values, "share": (mix / mix.sum()).values}) \
        if mix.sum() > 0 else pd.DataFrame(columns=["product", "revenue", "share"])
    return clean({"profile": profile, "score": score, "reasons": reasons, "signals": signals,
                  "quarterly": records(quarterly), "monthly": records(monthly),
                  "product_mix": records(mix.sort_values("revenue", ascending=False)), "mix_period": f"{last4[0]}..{last4[-1]}"})


# ---------------------------------------------------------------- at-risk + forecast

AT_RISK_COLS = ["risk_rank", "partner_id", "partner_name", "region", "tier", "partner_type", "country", "health_score",
                "p_leave", "segment", "revenue_avg_4q", "revenue_q2_2026", "yoy_h1_pct", "days_since_last_order",
                "lines_per_month_6m", "reason_1", "driver_1", "driver_1_pts", "reason_2", "driver_2", "driver_2_pts",
                "reason_3", "driver_3", "driver_3_pts"]


def at_risk_summary(state) -> dict:
    ar = state.at_risk
    if ar is None:
        return {"available": False, "error": state.errors.get("at_risk")}
    sc = ar.scores
    flagged = sc[sc["at_risk"] == 1]
    fs = ar.flag_summary.loc[ar.best_mult]
    segs = sc.groupby("segment").agg(partners=("partner_id", "size"), revenue_q2=("revenue_q2_2026", "sum"),
                                     avg_health=("health_score", "mean")).reindex(SEGMENTS).dropna(how="all")
    drivers = flagged["driver_1"].replace("", np.nan).dropna().value_counts()
    declining = state.declining
    dec_cols = [c for c in AT_RISK_COLS if c not in ("driver_1_pts", "driver_2_pts", "driver_3_pts")] + ["yoy_decline_streak", "rev_drawdown"]
    bins = np.histogram(sc["health_score"], bins=[0, 50, 60, 70, 80, 90, 95, 97, 98, 99, 100])
    return clean({
        "available": True,
        "summary": {"partners": len(sc), "flagged": len(flagged), "expected_leavers": ar.expected,
                    "flagged_revenue_avg_4q": flagged["revenue_avg_4q"].sum(),
                    "flagged_share_of_revenue": flagged["revenue_avg_4q"].sum() / sc["revenue_avg_4q"].sum(),
                    "season_factor": ar.season, "flag_multiplier": ar.best_mult,
                    "backtest_auc": ar.summary.loc["ensemble", "auc"],
                    "backtest_auc_heuristic": ar.summary.loc["heuristic: -lines_per_month_6m", "auc"],
                    "backtest_precision": fs["precision"], "backtest_recall": fs["recall"]},
        "drivers": [{"driver": k, "partners": v} for k, v in drivers.items()],
        "segments": records(segs.rename_axis("segment").reset_index()),
        "health_histogram": [{"from": a, "to": b, "partners": n} for a, b, n in zip(bins[1][:-1], bins[1][1:], bins[0])],
        "flagged": records(flagged[AT_RISK_COLS]),
        "declining": records(declining[dec_cols]) if declining is not None else None,
    })


def forecast_payload(state) -> dict:
    fr = state.forecast
    if fr is None:
        return {"available": False, "error": state.errors.get("forecast")}
    hist = fr.quarterly.copy()
    hist["ALL"] = hist[REGIONS].sum(axis=1)
    areas = REGIONS + ["ALL"]
    bt = fr.bt.assign(ape=fr.bt["pct_error"].abs())
    mape = bt.groupby(["method", "area"])["ape"].mean().unstack()[areas]
    order = [m for m in METHOD_LABELS if m in mape.index]
    mape = mape.reindex(order)
    loo = fr.loo.groupby("area")[["ape_bias_corrected", "inside_80"]].mean().reindex(areas)
    cl = fr.churn_live
    churn_bt = fr.churn_bt.groupby("area")[["ape_without", "ape_with"]].mean().reindex(areas) if len(fr.churn_bt) else None
    sub = fr.sub.rename(columns={"forecast_revenue_usd": "forecast", "lo80_usd": "lo80", "hi80_usd": "hi80"})
    methods = fr.fc.rename_axis("area").reset_index()
    return clean({
        "available": True, "target_quarter": "2026-Q3",
        "submission": records(sub),
        "history": records(hist.rename_axis("quarter").reset_index()),
        "methods": records(methods), "method_labels": METHOD_LABELS,
        "backtest_mape": records(mape.rename_axis("method").reset_index()),
        "selection": records(fr.selection.rename_axis("ensemble").reset_index()),
        "chosen": fr.chosen, "bias_pct": math.exp(fr.bias) - 1,
        "interval_pct": {a: {"down": 1 - math.exp(-h), "up": math.exp(h) - 1} for a, h in fr.half.items()},
        "loo": records(loo.rename_axis("area").reset_index()),
        "backtest_quarters": sorted(fr.bt["quarter"].unique()),
        "churn": {"applied": fr.apply_churn, "adjustment": cl["adjustment"].to_dict(),
                  "share_now": cl["share_now"].to_dict(), "share_normal": cl["share_normal"].to_dict(),
                  "flagged_revenue_avg_4q": cl["flagged_revenue_avg_4q"].to_dict(),
                  "backtest": records(churn_bt.rename_axis("area").reset_index()) if churn_bt is not None else []},
    })
