"""At-risk model: health score, at-risk flag, plain-language reasons and partner segments.

    python -m src.at_risk

1. Rolling-origin backtest of candidate models on the historical snapshots (label = left_token next quarter).
2. Final model = mean of logistic regression and gradient boosting probabilities, refit on all snapshots,
   then Platt-calibrated on out-of-fold backtest predictions so probabilities are honest.
3. health_score = 100 x (1 - P(leave in 2026-Q3)). at_risk = top-N partners, N chosen in the backtest.
4. Reasons: for each partner, each driver group (recency, frequency, size, ...) is reset to the values of a
   typical healthy partner of the same tier; the drop in risk measures how much that group drives the score.
   The biggest drivers are rendered as sentences from the partner's own numbers (no LLM needed).

Writes outputs/models/partner_scores.csv, at_risk_report.md, at_risk_model.joblib, at_risk_oof.parquet
(out-of-fold probabilities for the forecast's churn adjustment) and submission/submission_at_risk.csv.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_prep import ROOT, TIERS, build_all
from .eda import fp, md_table
from .features import CATEGORICAL, FEATURES, SCORE_CUTOFF, TRAIN_CUTOFFS, build_features, build_panel

MODEL_DIR = ROOT / "outputs" / "models"
SUBMISSION_DIR = ROOT / "submission"
LABEL = "left_token"
EVAL_CUTOFFS = TRAIN_CUTOFFS[2:]  # each evaluated with a model trained on all earlier cutoffs
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
FLAG_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # flag N = multiplier x expected leavers

# Driver groups used for explanations (profile attributes are context, not drivers).
REASON_GROUPS = {
    "recency": ["days_since_last_order", "recency_vs_gap", "never_ordered", "lines_last_month"],
    "frequency": ["lines_per_month_6m", "order_days_6m", "active_month_share_6m", "typical_gap_days"],
    "volatility": ["rev_cv_6m", "rev_cv_4q"],
    "size": ["rev_q1_log", "rev_avg_2q_log", "rev_avg_4q_log", "margin_usd_4q_log", "avg_line_value_6m_log"],
    "momentum": ["rev_growth_qoq", "rev_growth_2q", "rev_q1_vs_prev4q", "rev_slope_4q", "rev_slope_6m",
                 "rev_drawdown", "lines_3m_vs_prev3m", "lines_q1_vs_prev4q", "line_value_3m_vs_prev3m"],
    "yoy": ["yoy_q1", "yoy_2q", "lines_yoy_2q", "yoy_down_quarters_4q", "yoy_decline_streak", "yoy_2q_vs_region"],
    "attainment": ["att_q1", "att_avg_2q", "att_avg_4q", "att_trend_4q", "att_miss_streak", "att_misses_4q", "target_yoy"],
    "breadth": ["n_products_q1", "n_products_2q", "n_products_change", "share_proliant_2q", "share_synergy_2q",
                "share_storage_2q", "share_aruba_2q", "share_greenlake_2q", "product_hhi_2q"],
    "margin": ["margin_pct_4q", "margin_pct_2q", "margin_trend", "margin_vs_tier"],
}


# ---------------------------------------------------------------- models

def make_lr():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.05, max_iter=5000))


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, categorical_features="from_dtype", random_state=0)


class RiskModel:
    """Mean of logistic-regression and gradient-boosting probabilities, optionally Platt-calibrated."""

    def __init__(self) -> None:
        self.lr, self.hgb = make_lr(), make_hgb()
        self.calibrator: LogisticRegression | None = None

    def fit(self, df: pd.DataFrame) -> "RiskModel":
        self.lr.fit(df[FEATURES], df[LABEL])
        self.hgb.fit(df[FEATURES], df[LABEL])
        return self

    def components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        lr = self.lr.predict_proba(df[FEATURES])[:, 1]
        hgb = self.hgb.predict_proba(df[FEATURES])[:, 1]
        return {"logistic": lr, "boosting": hgb, "ensemble": (lr + hgb) / 2}

    def calibrate(self, raw: np.ndarray, y: np.ndarray) -> "RiskModel":
        self.calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw)[:, None], y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.components(df)["ensemble"]
        return raw if self.calibrator is None else self.calibrator.predict_proba(_logit(raw)[:, None])[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def money(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"${v:,.0f}" if abs(v) < 1e3 else (f"${v / 1e3:,.0f}K" if abs(v) < 1e6 else f"${v / 1e6:,.1f}M")


def seasonal_factor(panel: pd.DataFrame, label_quarter: int) -> float:
    """Leave rate for labels in this quarter of the year / average leave rate over all labelled quarters."""
    rates = panel.groupby("cutoff")[LABEL].mean()
    label_q = rates.index.map(lambda c: int(c[-1]) % 4 + 1)
    return float(rates[label_q == label_quarter].mean() / rates.mean())


def scale_odds(p: np.ndarray, factor: float) -> np.ndarray:
    """Multiply the odds by a factor that moves the mean probability by roughly `factor` (for small p)."""
    return p * factor / (1 - p + p * factor)


def prf(y: np.ndarray, flag: np.ndarray) -> tuple[float, float, float]:
    tp = (flag & (y == 1)).sum()
    prec = tp / flag.sum() if flag.sum() else 0.0
    rec = tp / (y == 1).sum() if (y == 1).sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def top_n_flags(p: np.ndarray, n: int) -> np.ndarray:
    flag = np.zeros(len(p), dtype=bool)
    flag[np.argsort(-p)[:max(n, 0)]] = True
    return flag


# ---------------------------------------------------------------- backtest

def backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling origin: for each eval cutoff, train on all earlier cutoffs. Returns metrics, OOF preds, flag study."""
    metrics, oof, flags = [], [], []
    for c in EVAL_CUTOFFS:
        train = panel[panel["cutoff"] < c]
        test = panel[panel["cutoff"] == c]
        y = test[LABEL].to_numpy()
        model = RiskModel().fit(train)
        preds = model.components(test)
        preds["heuristic: -lines_per_month_6m"] = -test["lines_per_month_6m"].fillna(0).to_numpy()
        for name, p in preds.items():
            metrics.append({"cutoff": c, "model": name, "auc": roc_auc_score(y, p),
                            "avg_precision": average_precision_score(y, p)})
        oof.append(pd.DataFrame({"cutoff": c, "partner_id": test["partner_id"].values, "y": y,
                                 "raw": preds["ensemble"]}))
        expected = train[LABEL].mean() * len(test)
        for m in FLAG_MULTIPLIERS:
            n = int(round(m * expected))
            prec, rec, f1 = prf(y, top_n_flags(preds["ensemble"], n))
            flags.append({"cutoff": c, "multiplier": m, "flagged": n, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(metrics), pd.concat(oof, ignore_index=True), pd.DataFrame(flags)


# ---------------------------------------------------------------- explanations

def sym_to_pct(s: float) -> float:
    """Symmetric growth (n - o) / (n + o) -> ordinary % change n / o - 1."""
    return np.nan if pd.isna(s) or s >= 1 else (1 + s) / (1 - s) - 1


def driver_effects(model: RiskModel, X: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Risk drop when each driver group is reset to the median of healthy same-tier partners."""
    healthy = X[p <= np.median(p)]
    ref = healthy.groupby("tier", observed=True)[NUMERIC].median()
    ref_rows = ref.reindex(X["tier"].astype(str)).set_index(X.index)
    effects = {}
    for group, feats in REASON_GROUPS.items():
        Xg = X.copy()
        Xg[feats] = ref_rows[feats].to_numpy()
        effects[group] = p - model.predict(Xg)
    return pd.DataFrame(effects, index=X.index)


def render_reason(group: str, r: pd.Series, peer: dict) -> str:
    q = SCORE_CUTOFF
    if group == "recency":
        if r["never_ordered"] == 1:
            return f"Has not placed an order since onboarding {r['tenure_months']:.0f} months ago"
        days, gap = r["days_since_last_order"], r["typical_gap_days"]
        if pd.notna(gap) and days > gap:
            return f"No order for {days:.0f} days - usually orders every {gap:.0f} days"
        if days > 60:
            return f"No order for {days:.0f} days"
        n = r["lines_last_month"]
        return f"Quiet at quarter end: {n:.0f} order line{'' if n == 1 else 's'} in the final month of {q}"
    if group == "frequency":
        months = r["active_month_share_6m"] * 6
        if months == 0:
            return f"No orders in the last 6 months (last order {r['days_since_last_order']:.0f} days ago)"
        return (f"Orders infrequently: active in {months:.0f} of the last 6 months, "
                f"{r['lines_per_month_6m']:.1f} order lines a month (typical {r['tier']} partner: "
                f"{peer['lines_per_month_6m'][r['tier']]:.1f})")
    if group == "volatility":
        return (f"Very irregular buying: monthly revenue varies {r['rev_cv_6m']:.1f}x its average "
                f"(typical {r['tier']} partner: {peer['rev_cv_6m'][r['tier']]:.1f}x)")
    if group == "size":
        rev, typical = np.expm1(r["rev_q1_log"]), money(np.expm1(peer["rev_q1_log"][r["tier"]]))
        if rev == 0:
            return (f"No revenue in {q} (averaged {money(np.expm1(r['rev_avg_4q_log']))} a quarter over the last year; "
                    f"typical {r['tier']} partner: {typical})")
        return f"Small account: {money(rev)} revenue in {q} vs {typical} for a typical {r['tier']} partner"
    if group == "momentum":
        parts = []
        if pd.notna(r["rev_q1_vs_prev4q"]):
            parts.append(f"revenue in {q} was {r['rev_q1_vs_prev4q']:.0%} of its previous 4-quarter average")
        lines = sym_to_pct(r["lines_3m_vs_prev3m"])
        if pd.notna(lines):
            parts.append(f"order lines {lines:+.0%} in the last 3 months vs the 3 before")
        return ("Slowing down: " + "; ".join(parts)) if parts else "Recent revenue trend is weakening"
    if group == "yoy":
        pct = sym_to_pct(r["yoy_2q"])
        text = f"Revenue {pct:+.0%} year on year (H1 2026 vs H1 2025)" if pd.notna(pct) else "Revenue down year on year"
        if r["yoy_decline_streak"] >= 2:
            text += f", down {r['yoy_decline_streak']:.0f} quarters in a row"
        return text
    if group == "attainment":
        streak = r["att_miss_streak"]
        if streak >= 1:
            return (f"Below target {streak:.0f} quarter{'' if streak == 1 else 's'} in a row; "
                    f"{r['att_avg_2q']:.0%} of target over the last 2 quarters")
        return f"Target attainment slipping: {r['att_avg_2q']:.0%} of target over the last 2 quarters"
    if group == "breadth":
        n, chg = r["n_products_2q"], r["n_products_change"]
        text = f"Narrow product mix: {n:.0f} product famil{'y' if n == 1 else 'ies'} in the last 2 quarters"
        return text + (f" (down from {n - chg:.0f})" if chg < 0 else "")
    if group == "margin":
        return f"Margin {r['margin_pct_4q']:.1f}% vs {peer['margin_pct_4q'][r['tier']]:.1f}% for its tier"
    raise KeyError(group)


def explain(model: RiskModel, X: pd.DataFrame, p: np.ndarray, top: int = 3) -> pd.DataFrame:
    eff = driver_effects(model, X, p)
    peer = {c: X.groupby("tier", observed=True)[c].median().to_dict() for c in
            ["lines_per_month_6m", "rev_cv_6m", "rev_q1_log", "margin_pct_4q"]}
    rows = []
    for i in X.index:
        r = X.loc[i]
        e = eff.loc[i].sort_values(ascending=False)
        e = e[(e > 0.002) & (e > 0.1 * p[X.index.get_loc(i)])].head(top)
        reasons = [(g, render_reason(g, r, peer), d) for g, d in e.items()]
        if not reasons:
            reasons = [("profile", f"{r['tier']} {r['partner_type']} - risk comes from many small signals, "
                                   f"no single dominant driver", 0.0)]
        row = {"partner_id": r["partner_id"]}
        for k in range(top):
            g, text, d = reasons[k] if k < len(reasons) else ("", "", np.nan)
            row[f"reason_{k + 1}"], row[f"driver_{k + 1}"], row[f"driver_{k + 1}_pts"] = text, g, d * 100
        rows.append(row)
    return pd.DataFrame(rows)


def segment(X: pd.DataFrame, at_risk: np.ndarray) -> pd.Series:
    yoy = X["yoy_2q"].map(sym_to_pct)
    seg = np.select(
        [at_risk == 1,
         X["is_ramping"] == 1,
         (yoy <= -0.25) & (X["yoy_decline_streak"] >= 2),
         yoy >= 0.15],
        ["At risk", "Ramping (new)", "Quietly declining", "Growing"], default="Stable")
    return pd.Series(seg, index=X.index)


# ---------------------------------------------------------------- submission

def validate_at_risk(sub: pd.DataFrame, master: pd.DataFrame) -> None:
    assert list(sub.columns) == ["partner_id", "health_score", "at_risk"], sub.columns
    assert sub["partner_id"].is_unique, "duplicate partner ids"
    assert set(sub["partner_id"]) == set(master["partner_id"]), "partner ids differ from partner_master"
    assert np.isfinite(sub["health_score"]).all(), "non-finite health score"
    assert sub["at_risk"].isin([0, 1]).all(), "at_risk must be 0/1"


# ---------------------------------------------------------------- main

@dataclass
class AtRiskResult:
    """Everything the at-risk step produces, kept in memory (used by the API and by the forecast)."""
    scores: pd.DataFrame        # one row per partner, same columns as partner_scores.csv
    oof: pd.DataFrame           # calibrated out-of-fold probabilities per historical cutoff
    X: pd.DataFrame             # 2026-Q2 feature snapshot
    panel: pd.DataFrame         # historical labelled snapshots
    model: RiskModel
    metrics: pd.DataFrame
    summary: pd.DataFrame
    flags: pd.DataFrame
    flag_summary: pd.DataFrame
    raw_oof: pd.DataFrame
    best_mult: float
    expected: float
    n_flag: int
    season: float


def compute(tables: dict) -> AtRiskResult:
    """Train, calibrate, score and explain - no file I/O."""
    panel = build_panel(tables, TRAIN_CUTOFFS)
    X = build_features(tables, SCORE_CUTOFF)

    metrics, oof, flags = backtest(panel)
    summary = metrics.groupby("model")[["auc", "avg_precision"]].mean().sort_values("auc", ascending=False)
    flag_summary = flags.groupby("multiplier")[["flagged", "precision", "recall", "f1"]].mean()
    best_mult = flag_summary["f1"].idxmax()

    model = RiskModel().fit(panel).calibrate(oof["raw"].to_numpy(), oof["y"].to_numpy())
    season = seasonal_factor(panel, label_quarter=3)
    p_base = model.predict(X)
    p = scale_odds(p_base, season)

    # Out-of-fold calibrated probabilities per historical cutoff, for the forecast's churn adjustment.
    oof_cal = oof.assign(p=model.calibrator.predict_proba(_logit(oof["raw"].to_numpy())[:, None])[:, 1])
    oof_cal = oof_cal.merge(panel[["cutoff", "partner_id", "region", "baseline_4q_revenue"]],
                            on=["cutoff", "partner_id"], how="left", validate="one_to_one")
    oof_cal["region"] = oof_cal["region"].astype(str)
    expected = p.sum()
    n_flag = int(round(best_mult * expected))
    at_risk = top_n_flags(p, n_flag).astype(int)

    reasons = explain(model, X, p)
    summ = tables["partner_summary"].set_index("partner_id")
    scores = X[["partner_id", "region", "tier", "partner_type"]].copy()
    for c in ["region", "tier", "partner_type"]:
        scores[c] = scores[c].astype(str)
    scores.insert(1, "partner_name", scores["partner_id"].map(summ["partner_name"]))
    scores["country"] = scores["partner_id"].map(summ["country"])
    scores["p_leave"] = p
    scores["p_leave_unseasonal"] = p_base
    scores["health_score"] = (100 * (1 - p)).round(4)
    scores["risk_rank"] = scores["p_leave"].rank(ascending=False, method="first").astype(int)
    scores["at_risk"] = at_risk
    scores["segment"] = segment(X, at_risk).values
    scores["revenue_q2_2026"] = np.expm1(X["rev_q1_log"]).round(0)
    scores["revenue_avg_4q"] = np.expm1(X["rev_avg_4q_log"]).round(0)
    scores["yoy_h1_pct"] = X["yoy_2q"].map(sym_to_pct)
    scores["last_order"] = scores["partner_id"].map(summ["last_order"])
    scores["days_since_last_order"] = X["days_since_last_order"]
    scores["lines_per_month_6m"] = X["lines_per_month_6m"].round(2)
    scores = scores.merge(reasons, on="partner_id").sort_values("risk_rank").reset_index(drop=True)
    return AtRiskResult(scores=scores, oof=oof_cal, X=X, panel=panel, model=model, metrics=metrics, summary=summary, flags=flags,
                        flag_summary=flag_summary, raw_oof=oof, best_mult=best_mult, expected=expected,
                        n_flag=n_flag, season=season)


def submission_frame(res: AtRiskResult) -> pd.DataFrame:
    return res.scores[["partner_id", "health_score", "at_risk"]].sort_values("partner_id").reset_index(drop=True)


def run(tables: dict | None = None) -> AtRiskResult:
    """compute() plus every file the pipeline writes."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    tables = tables or build_all()
    res = compute(tables)
    res.oof.to_parquet(MODEL_DIR / "at_risk_oof.parquet", index=False)
    res.scores.to_csv(MODEL_DIR / "partner_scores.csv", index=False)
    sub = submission_frame(res)
    validate_at_risk(sub, tables["master"])
    sub.to_csv(SUBMISSION_DIR / "submission_at_risk.csv", index=False)
    joblib.dump(res.model, MODEL_DIR / "at_risk_model.joblib")
    write_report(res.metrics, res.summary, res.raw_oof, res.flags, res.flag_summary, res.best_mult, res.expected,
                 res.n_flag, res.scores, res.model, res.season)
    return res


def write_report(metrics, summary, oof, flags, flag_summary, best_mult, expected, n_flag, scores, model, season) -> None:
    L = ["# At-risk model report", ""]

    def t(df):
        L.extend([md_table(df), ""])

    L += ["## 1. Rolling-origin backtest", "",
          f"Label `{LABEL}` (no orders, or a token order, in the next quarter). Each cutoff is scored by a model "
          f"trained only on earlier cutoffs. Mean over {', '.join(EVAL_CUTOFFS)}:", ""]
    t(summary.reset_index().round(3))
    L += ["Per cutoff:", ""]
    t(metrics.pivot(index="model", columns="cutoff", values="auc").round(3).reset_index())

    cal = oof.groupby("cutoff").agg(partners=("y", "size"), actual_leavers=("y", "sum"))
    cal["expected_calibrated"] = [model.calibrator.predict_proba(_logit(g["raw"].to_numpy())[:, None])[:, 1].sum()
                                  for _, g in oof.groupby("cutoff")]
    L += ["## 2. Calibration", "",
          "Ensemble probabilities are Platt-calibrated on the out-of-fold predictions above (in-sample for this table):", ""]
    t(cal.reset_index().round(1))
    L += [f"Leaving is seasonal: labels falling in Q3 have a leave rate {season:.2f}x the average (Q3 is the low "
          f"season, so more partners skip it). The 2026-Q3 probabilities are scaled by this factor on the odds scale.", ""]

    L += ["## 3. How many partners to flag", "",
          "Flag the top N partners, N = multiplier x expected leavers (base rate of the training data x partners). "
          "Mean over the backtest cutoffs:", ""]
    t(flag_summary.reset_index().round(3))
    L += [f"Chosen multiplier **{best_mult}**. For 2026-Q3 the calibrated model expects **{expected:.1f}** leavers, "
          f"so **{n_flag}** partners are flagged.", ""]

    L += ["## 4. Segments (2026-Q2 snapshot)", ""]
    seg = scores.groupby("segment").agg(partners=("partner_id", "size"), revenue_q2=("revenue_q2_2026", "sum"))
    seg["revenue_q2"] = seg["revenue_q2"].map(lambda v: f"${v / 1e6:,.1f}M")
    t(seg.reset_index())

    flagged = scores[scores["at_risk"] == 1]
    L += ["## 5. Flagged partners: main drivers", ""]
    drv = flagged["driver_1"].value_counts().rename_axis("main driver").reset_index(name="partners")
    t(drv)
    exposure = flagged["revenue_avg_4q"].sum()
    dormant = scores[scores["revenue_q2_2026"] == 0]
    L += [f"Flagged partners average {money(flagged['revenue_avg_4q'].median())} a quarter (median); together "
          f"{money(exposure)} a quarter, {fp(exposure / scores['revenue_avg_4q'].sum())} of channel revenue.", "",
          f"Partners with no revenue in {SCORE_CUTOFF}: {len(dormant)}, of which {int(dormant['at_risk'].sum())} flagged "
          f"(median risk rank {dormant['risk_rank'].median():.0f}).", ""]
    top = flagged.head(20)[["risk_rank", "partner_id", "partner_name", "region", "tier", "p_leave", "reason_1", "reason_2"]].copy()
    top["p_leave"] = top["p_leave"].map(fp)
    L += ["Top 20 by risk:", ""]
    t(top)

    L += ["## Limitations", "",
          "- History has almost no permanent leavers: the label mostly captures small, irregular partners skipping a "
          "quarter (most come back). If 2026-Q3 leavers follow a different process, precision will suffer.",
          "- The partners with long YoY decline are shown as a separate segment ('Quietly declining'); historically they "
          "did not leave more often than average, so the model does not flag them for decline alone.",
          "- Reasons measure how much each driver group moves the model's risk; they explain the model, not causation.",
          ""]
    (MODEL_DIR / "at_risk_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    scores = run().scores
    flagged = scores[scores["at_risk"] == 1]
    print((MODEL_DIR / "at_risk_report.md").read_text())
    print(f"\nFlagged {len(flagged)} of {len(scores)} partners -> {SUBMISSION_DIR / 'submission_at_risk.csv'}")


if __name__ == "__main__":
    main()
