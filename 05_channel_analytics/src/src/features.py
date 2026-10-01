"""Point-in-time partner features for the at-risk (churn) model.

Every feature for a cutoff quarter Q is computed only from data on or before the last day of Q,
so the same function builds historical training snapshots and the final 2026-Q2 scoring snapshot.

    python -m src.features      # writes outputs/features/*.parquet + feature_dictionary.csv + diagnostics

    from src.features import build_features, build_panel
    X = build_features(tables, "2026-Q2")                # one row per partner onboarded by the cutoff
    panel = build_panel(tables, TRAIN_CUTOFFS)           # stacked snapshots with labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import PRODUCTS, RAMPING_MONTHS, ROOT, TIERS, build_all, month_start

FEATURE_DIR = ROOT / "outputs" / "features"

# Snapshots with a known next quarter (labels) and the snapshot we score for 2026-Q3.
TRAIN_CUTOFFS = ["2024-Q4", "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1"]
SCORE_CUTOFF = "2026-Q2"

DECLINE_YOY = -0.10  # symmetric YoY growth below this counts as a "down" quarter
RATIO_CAP = 5.0      # cap for unbounded ratios (x / baseline)
WINSOR_Q = (0.01, 0.99)

# (name, group, window, description). Order here is the column order of the feature matrix.
FEATURE_DOCS = [
    # Revenue trend
    ("rev_q1_log", "revenue_trend", "last quarter", "log(1 + revenue in the cutoff quarter)"),
    ("rev_avg_2q_log", "revenue_trend", "last 2 quarters", "log(1 + average quarterly revenue)"),
    ("rev_avg_4q_log", "revenue_trend", "last 4 quarters", "log(1 + average quarterly revenue)"),
    ("rev_growth_qoq", "revenue_trend", "last 2 quarters", "symmetric growth (Q - Q-1) / (Q + Q-1), in [-1, 1]"),
    ("rev_growth_2q", "revenue_trend", "last 4 quarters", "symmetric growth of last 2 quarters vs the 2 before"),
    ("rev_q1_vs_prev4q", "revenue_trend", "last 5 quarters", "cutoff-quarter revenue / average of the 4 before (capped at 5)"),
    ("rev_cv_4q", "revenue_trend", "last 4 quarters", "coefficient of variation of quarterly revenue"),
    ("rev_cv_6m", "revenue_trend", "last 6 months", "coefficient of variation of monthly revenue"),
    ("rev_slope_4q", "revenue_trend", "last 4 quarters", "linear slope of quarterly revenue / its mean"),
    ("rev_slope_6m", "revenue_trend", "last 6 months", "linear slope of monthly revenue / its mean"),
    ("rev_drawdown", "revenue_trend", "all history", "latest 2-quarter average / peak 2-quarter average (1 = at peak)"),
    # Order activity
    ("order_days_6m", "order_activity", "last 6 months", "distinct days with an order"),
    ("lines_per_month_6m", "order_activity", "last 6 months", "average order lines per month"),
    ("active_month_share_6m", "order_activity", "last 6 months", "share of months with at least one order"),
    ("lines_last_month", "order_activity", "last month", "order lines in the final month of the cutoff quarter"),
    ("lines_3m_vs_prev3m", "order_activity", "last 6 months", "symmetric growth of order lines, last 3 months vs prior 3"),
    ("lines_q1_vs_prev4q", "order_activity", "last 5 quarters", "cutoff-quarter order lines / average of the 4 before (capped at 5)"),
    ("days_since_last_order", "order_activity", "all history", "days from last order to the cutoff date"),
    ("typical_gap_days", "order_activity", "last 12 months", "median days between order days"),
    ("recency_vs_gap", "order_activity", "last 12 months", "days since last order / typical gap (capped at 20)"),
    ("avg_line_value_6m_log", "order_activity", "last 6 months", "log(1 + revenue per order line)"),
    ("line_value_3m_vs_prev3m", "order_activity", "last 6 months", "symmetric growth of revenue per line, last 3 months vs prior 3"),
    ("never_ordered", "order_activity", "all history", "1 if the partner has no order on or before the cutoff"),
    # Target attainment
    ("att_q1", "target_attainment", "last quarter", "revenue / target in the cutoff quarter (capped at 5)"),
    ("att_avg_2q", "target_attainment", "last 2 quarters", "sum revenue / sum target (capped at 5)"),
    ("att_avg_4q", "target_attainment", "last 4 quarters", "sum revenue / sum target (capped at 5)"),
    ("att_trend_4q", "target_attainment", "last 4 quarters", "linear slope of quarterly attainment"),
    ("att_miss_streak", "target_attainment", "all history", "consecutive quarters below target, ending at the cutoff"),
    ("att_misses_4q", "target_attainment", "last 4 quarters", "quarters below target"),
    ("target_yoy", "target_attainment", "Q vs Q-4", "symmetric YoY growth of the target"),
    # Profit quality
    ("margin_pct_4q", "profit_quality", "last 4 quarters", "revenue-weighted margin %"),
    ("margin_pct_2q", "profit_quality", "last 2 quarters", "revenue-weighted margin %"),
    ("margin_trend", "profit_quality", "last 4 quarters", "margin % of last 2 quarters minus the 2 before (pp)"),
    ("margin_vs_tier", "profit_quality", "last 4 quarters", "margin % minus the median of partners in the same tier (pp)"),
    ("margin_usd_4q_log", "profit_quality", "last 4 quarters", "log(1 + margin dollars)"),
    # Product diversity
    ("n_products_q1", "product_diversity", "last quarter", "product families ordered"),
    ("n_products_2q", "product_diversity", "last 2 quarters", "product families ordered"),
    ("n_products_change", "product_diversity", "last 4 quarters", "families in last 2 quarters minus families in the 2 before"),
    *[(f"share_{p.lower()}_2q", "product_diversity", "last 2 quarters", f"{p} share of revenue") for p in PRODUCTS],
    ("product_hhi_2q", "product_diversity", "last 2 quarters", "Herfindahl index of product revenue shares (1 = single product)"),
    # Year-on-year
    ("yoy_q1", "year_on_year", "Q vs Q-4", "symmetric YoY revenue growth of the cutoff quarter"),
    ("yoy_2q", "year_on_year", "last 2Q vs same 2Q a year earlier", "symmetric YoY revenue growth (H1 26 vs H1 25 at the 2026-Q2 cutoff)"),
    ("lines_yoy_2q", "year_on_year", "last 2Q vs same 2Q a year earlier", "symmetric YoY growth of order lines"),
    ("yoy_down_quarters_4q", "year_on_year", "last 4 quarters", f"quarters with YoY growth below {DECLINE_YOY:+.0%}"),
    ("yoy_decline_streak", "year_on_year", "last 6 quarters", f"consecutive quarters with YoY growth below {DECLINE_YOY:+.0%}, ending at the cutoff"),
    ("yoy_2q_vs_region", "year_on_year", "last 2Q vs same 2Q a year earlier", "yoy_2q minus the median yoy_2q of the partner's region"),
    # Static
    ("partner_type", "static", "single value", "Distributor / Reseller / Service Provider"),
    ("tier", "static", "single value", "Platinum / Gold / Silver / Business"),
    ("region", "static", "single value", "AMS / APJ / EMEA"),
    ("tier_rank", "static", "single value", "Business=0, Silver=1, Gold=2, Platinum=3"),
    ("tenure_months", "static", "single value", "months from onboarding to the cutoff date"),
    ("is_ramping", "static", "single value", f"1 if onboarded less than {RAMPING_MONTHS} months before the cutoff"),
    ("quarters_observed", "static", "single value", "quarters since onboarding inside the data window, up to the cutoff"),
]
FEATURES = [name for name, *_ in FEATURE_DOCS]
CATEGORICAL = ["partner_type", "tier", "region"]
WINSORIZE = ["rev_cv_4q", "rev_cv_6m", "rev_slope_4q", "rev_slope_6m", "att_trend_4q", "margin_trend", "margin_vs_tier"]


# ---------------------------------------------------------------- helpers

def _period(q: str) -> pd.Period:
    return pd.Period(q.replace("-", ""), freq="Q")


def quarter_end(q: str) -> pd.Timestamp:
    return _period(q).end_time.normalize()


def quarter_start(q: str) -> pd.Timestamp:
    return _period(q).start_time


def shift_quarter(q: str, n: int) -> str:
    p = _period(q) + n
    return f"{p.year}-Q{p.quarter}"


def sym_growth(new: pd.Series, old: pd.Series) -> pd.Series:
    """(new - old) / (new + old): bounded in [-1, 1], robust to small bases; NaN when both are 0 or missing."""
    denom = new + old
    return ((new - old) / denom).where(denom > 0)


def capped_ratio(num: pd.Series, den: pd.Series, cap: float = RATIO_CAP) -> pd.Series:
    return (num / den.where(den > 0)).clip(upper=cap)


def slope(mat: pd.DataFrame) -> pd.Series:
    """Least-squares slope per row over equally spaced columns (NaN if any value is missing)."""
    y = mat.to_numpy(dtype=float)
    x = np.arange(y.shape[1]) - (y.shape[1] - 1) / 2
    return pd.Series((y * x).sum(axis=1) / (x ** 2).sum(), index=mat.index)


def norm_slope(mat: pd.DataFrame) -> pd.Series:
    mu = mat.mean(axis=1, skipna=False)
    return slope(mat) / mu.where(mu > 0)


def cv(mat: pd.DataFrame, min_obs: int) -> pd.Series:
    mu = mat.mean(axis=1)
    out = mat.std(axis=1, ddof=0) / mu.where(mu > 0)
    return out.where(mat.notna().sum(axis=1) >= min_obs)


def trailing_streak(flags: pd.DataFrame) -> pd.Series:
    """Consecutive True values ending at the last column (columns ordered oldest -> newest)."""
    f = flags.fillna(False).to_numpy(dtype=bool)[:, ::-1]
    streak = np.where(f.all(axis=1), f.shape[1], f.argmin(axis=1))
    return pd.Series(streak, index=flags.index)


def winsorize(df: pd.DataFrame, cols: list[str], q: tuple[float, float] = WINSOR_Q) -> pd.DataFrame:
    """Clip heavy tails inside the snapshot, so the raw tables are never modified."""
    out = df.copy()
    for c in cols:
        lo, hi = out[c].quantile(q[0]), out[c].quantile(q[1])
        out[c] = out[c].clip(lo, hi)
    return out


# ---------------------------------------------------------------- features

def build_features(tables: dict[str, pd.DataFrame], cutoff: str) -> pd.DataFrame:
    """One row per partner onboarded on or before the end of `cutoff`, using only data up to that date."""
    end = quarter_end(cutoff)
    pop = tables["master"][tables["master"]["onboarded_date"] <= end].set_index("partner_id")
    idx = pop.index

    # Point-in-time views: nothing after the cutoff date gets past these filters.
    pq = tables["partner_quarter"][tables["partner_quarter"]["quarter"] <= cutoff]
    pm = tables["partner_month"][tables["partner_month"]["month"] <= end]
    sales = tables["sales"]
    sales = sales[(~sales["is_outlier"]) & (sales["order_date"] <= end)]

    qs = [shift_quarter(cutoff, -i) for i in range(9, -1, -1)]  # last 10 quarters, oldest -> newest
    q = lambda i: qs[-1 - i]  # noqa: E731  q(0) = cutoff quarter, q(1) = the one before, ...

    def qpivot(col: str) -> pd.DataFrame:
        return pq.pivot(index="partner_id", columns="quarter", values=col).reindex(index=idx, columns=qs)

    rev, lines = qpivot("revenue_usd"), qpivot("order_lines")
    att, tgt, mgn, nprod = qpivot("attainment"), qpivot("target_usd"), qpivot("margin_usd"), qpivot("n_products")

    months = pd.date_range(end=month_start(pd.Series([end])).iloc[0], periods=12, freq="MS")
    m6, last3, prev3 = months[-6:], months[-3:], months[-6:-3]
    R = pm.pivot(index="partner_id", columns="month", values="revenue_usd").reindex(index=idx, columns=months)
    L = pm.pivot(index="partner_id", columns="month", values="order_lines").reindex(index=idx, columns=months)

    f = pd.DataFrame(index=idx)
    last2q, last4q, prev4q = [q(1), q(0)], [q(3), q(2), q(1), q(0)], [q(4), q(3), q(2), q(1)]

    # Revenue trend
    f["rev_q1_log"] = np.log1p(rev[q(0)])
    f["rev_avg_2q_log"] = np.log1p(rev[last2q].mean(axis=1))
    f["rev_avg_4q_log"] = np.log1p(rev[last4q].mean(axis=1))
    f["rev_growth_qoq"] = sym_growth(rev[q(0)], rev[q(1)])
    f["rev_growth_2q"] = sym_growth(rev[q(0)] + rev[q(1)], rev[q(2)] + rev[q(3)])
    f["rev_q1_vs_prev4q"] = capped_ratio(rev[q(0)], rev[prev4q].mean(axis=1))
    f["rev_cv_4q"] = cv(rev[last4q], min_obs=2)
    f["rev_cv_6m"] = cv(R[m6], min_obs=3)
    f["rev_slope_4q"] = norm_slope(rev[last4q])
    f["rev_slope_6m"] = norm_slope(R[m6])
    roll2 = rev.T.rolling(2).mean().T
    f["rev_drawdown"] = roll2[q(0)] / roll2.max(axis=1).where(lambda p: p > 0)

    # Order activity
    s6 = sales[sales["order_date"] >= m6[0]]
    f["order_days_6m"] = s6.groupby("partner_id")["order_date"].nunique().reindex(idx).fillna(0)
    f["lines_per_month_6m"] = L[m6].mean(axis=1)
    f["active_month_share_6m"] = (L[m6] > 0).sum(axis=1) / L[m6].notna().sum(axis=1).where(lambda n: n > 0)
    f["lines_last_month"] = L[months[-1]]
    f["lines_3m_vs_prev3m"] = sym_growth(L[last3].sum(axis=1, min_count=1), L[prev3].sum(axis=1, min_count=1))
    f["lines_q1_vs_prev4q"] = capped_ratio(lines[q(0)], lines[prev4q].mean(axis=1))
    last_order = sales.groupby("partner_id")["order_date"].max().reindex(idx)
    f["days_since_last_order"] = (end - last_order).dt.days
    days = sales.loc[sales["order_date"] > end - pd.Timedelta(days=365), ["partner_id", "order_date"]].drop_duplicates()
    days = days.sort_values(["partner_id", "order_date"])
    gaps = days.groupby("partner_id")["order_date"].diff().dt.days
    f["typical_gap_days"] = gaps.groupby(days["partner_id"]).median().reindex(idx)
    f["recency_vs_gap"] = (f["days_since_last_order"] / f["typical_gap_days"].clip(lower=1)).clip(upper=20)
    f["avg_line_value_6m_log"] = np.log1p(R[m6].sum(axis=1) / L[m6].sum(axis=1).where(lambda n: n > 0))
    value_last3 = R[last3].sum(axis=1) / L[last3].sum(axis=1).where(lambda n: n > 0)
    value_prev3 = R[prev3].sum(axis=1) / L[prev3].sum(axis=1).where(lambda n: n > 0)
    f["line_value_3m_vs_prev3m"] = sym_growth(value_last3, value_prev3)
    f["never_ordered"] = last_order.isna().astype(int)

    # Target attainment
    f["att_q1"] = att[q(0)].clip(upper=RATIO_CAP)
    f["att_avg_2q"] = capped_ratio(rev[last2q].sum(axis=1, min_count=1), tgt[last2q].sum(axis=1, min_count=1))
    f["att_avg_4q"] = capped_ratio(rev[last4q].sum(axis=1, min_count=1), tgt[last4q].sum(axis=1, min_count=1))
    f["att_trend_4q"] = slope(att[last4q].clip(upper=RATIO_CAP))
    f["att_miss_streak"] = trailing_streak(att < 1)
    f["att_misses_4q"] = (att[last4q] < 1).sum(axis=1)
    f["target_yoy"] = sym_growth(tgt[q(0)], tgt[q(4)])

    # Profit quality (revenue-weighted margins)
    def margin_pct(cols: list[str]) -> pd.Series:
        r = rev[cols].sum(axis=1, min_count=1)
        return mgn[cols].sum(axis=1, min_count=1) / r.where(r > 0) * 100

    f["margin_pct_4q"] = margin_pct(last4q)
    f["margin_pct_2q"] = margin_pct(last2q)
    f["margin_trend"] = f["margin_pct_2q"] - margin_pct([q(3), q(2)])
    f["margin_vs_tier"] = f["margin_pct_4q"] - f["margin_pct_4q"].groupby(pop["tier"], observed=True).transform("median")
    f["margin_usd_4q_log"] = np.log1p(mgn[last4q].sum(axis=1).clip(lower=0))

    # Product diversity
    s2q = sales[sales["order_date"] >= quarter_start(q(1))]
    sp2 = sales[(sales["order_date"] >= quarter_start(q(3))) & (sales["order_date"] < quarter_start(q(1)))]
    f["n_products_q1"] = nprod[q(0)]
    f["n_products_2q"] = s2q.groupby("partner_id")["product_family"].nunique().reindex(idx).fillna(0)
    n_prev2 = sp2.groupby("partner_id")["product_family"].nunique().reindex(idx).fillna(0)
    f["n_products_change"] = f["n_products_2q"] - n_prev2
    mix = (s2q.assign(product_family=s2q["product_family"].astype(str))
           .pivot_table(index="partner_id", columns="product_family", values="revenue_usd", aggfunc="sum")
           .reindex(index=idx, columns=PRODUCTS).fillna(0))
    total = mix.sum(axis=1)
    shares = mix.div(total.where(total > 0), axis=0)
    for p in PRODUCTS:
        f[f"share_{p.lower()}_2q"] = shares[p]
    f["product_hhi_2q"] = (shares ** 2).sum(axis=1, min_count=1)

    # Year-on-year
    yoy = pd.DataFrame({q(i): sym_growth(rev[q(i)], rev[q(i + 4)]) for i in range(5, -1, -1)})
    down = yoy.lt(DECLINE_YOY) & yoy.notna()
    f["yoy_q1"] = yoy[q(0)]
    f["yoy_2q"] = sym_growth(rev[q(0)] + rev[q(1)], rev[q(4)] + rev[q(5)])
    f["lines_yoy_2q"] = sym_growth(lines[q(0)] + lines[q(1)], lines[q(4)] + lines[q(5)])
    f["yoy_down_quarters_4q"] = down[last4q].sum(axis=1)
    f["yoy_decline_streak"] = trailing_streak(down)
    f["yoy_2q_vs_region"] = f["yoy_2q"] - f["yoy_2q"].groupby(pop["region"], observed=True).transform("median")

    # Static
    for c in CATEGORICAL:
        f[c] = pop[c]
    f["tier_rank"] = pop["tier"].map({t: len(TIERS) - 1 - i for i, t in enumerate(TIERS)}).astype(int)
    f["tenure_months"] = ((end - pop["onboarded_date"]).dt.days / 30.44).round(1)
    f["is_ramping"] = (f["tenure_months"] < RAMPING_MONTHS).astype(int)
    f["quarters_observed"] = rev.notna().sum(axis=1)

    f = winsorize(f[FEATURES], WINSORIZE)
    f.insert(0, "cutoff", cutoff)
    return f.reset_index()


# ---------------------------------------------------------------- labels

def build_panel(tables: dict[str, pd.DataFrame], cutoffs: list[str]) -> pd.DataFrame:
    """Stacked feature snapshots with next-quarter labels (see src/labels.py)."""
    from .labels import build_labels

    parts = []
    for c in cutoffs:
        X = build_features(tables, c)
        y = build_labels(tables, c)
        parts.append(X.merge(y.drop(columns="cutoff"), on="partner_id", how="left", validate="one_to_one"))
    panel = pd.concat(parts, ignore_index=True)
    for col in CATEGORICAL:  # concat can drop the categorical dtype
        panel[col] = pd.Categorical(panel[col], categories=tables["master"][col].cat.categories)
    return panel


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DOCS, columns=["feature", "group", "window", "description"])


def main() -> None:
    from .feature_diagnostics import run_diagnostics

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_all()
    panel = build_panel(tables, TRAIN_CUTOFFS)
    score = build_features(tables, SCORE_CUTOFF)

    panel.to_parquet(FEATURE_DIR / "train_panel.parquet", index=False)
    score.to_parquet(FEATURE_DIR / f"score_{SCORE_CUTOFF.replace('-', '')}.parquet", index=False)
    feature_dictionary().to_csv(FEATURE_DIR / "feature_dictionary.csv", index=False)
    print(f"train_panel: {len(panel):,} rows ({len(TRAIN_CUTOFFS)} snapshots) x {len(FEATURES)} features")
    print(f"score snapshot {SCORE_CUTOFF}: {len(score):,} partners")
    run_diagnostics(tables, panel, score)


if __name__ == "__main__":
    main()
