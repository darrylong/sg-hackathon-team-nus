"""Checks on the feature matrix and proxy labels, written to outputs/features/feature_report.md.

Called by `python -m src.features`. Sections:
    1. label base rates per snapshot (and sensitivity to the token threshold)
    2. missing values in the scoring snapshot
    3. univariate ROC-AUC of every feature against the primary label (history vs holdout)
    4. near-duplicate feature pairs
    5. sanity-check baseline model (time split) - shows the features carry signal; not the final model
    6. the "quietly declining" cohort at the scoring cutoff
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from . import plot_style as ps
from .eda import fp, md_table
from .features import CATEGORICAL, FEATURE_DIR, FEATURES, SCORE_CUTOFF, TRAIN_CUTOFFS, feature_dictionary
from .labels import TOKEN_SHARE, build_labels

LABEL = "left_token"
HOLDOUT = TRAIN_CUTOFFS[-1]
CHARTS = FEATURE_DIR / "charts"
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def auc(y: pd.Series, x: pd.Series) -> float:
    m = x.notna()
    y, x = y[m], x[m]
    return roc_auc_score(y, x) if y.nunique() == 2 else np.nan


class Doc:
    def __init__(self) -> None:
        self.lines = ["# Feature engineering report", "",
                      f"Training snapshots: {', '.join(TRAIN_CUTOFFS)} (label = next quarter). "
                      f"Holdout snapshot: {HOLDOUT}. Scoring snapshot: {SCORE_CUTOFF}.", ""]

    def h(self, s: str) -> None:
        self.lines += [f"## {s}", ""]
        print(f"\n=== {s} ===")

    def p(self, s: str) -> None:
        self.lines += [s, ""]
        print(s)

    def t(self, df: pd.DataFrame) -> None:
        self.lines += [md_table(df), ""]
        print(df.to_string(index=False))

    def img(self, path, alt: str) -> None:
        self.lines += [f"![{alt}](charts/{path.name})", ""]


def label_rates(tables: dict, panel: pd.DataFrame, d: Doc) -> None:
    d.h("1. Proxy label base rates")
    rows = []
    for c in TRAIN_CUTOFFS:
        g = panel[panel["cutoff"] == c]
        row = {"cutoff": c, "label quarter": g.attrs.get("nq", ""), "partners": len(g),
               "left_zero": g["left_zero"].mean(), "left_token": g["left_token"].mean(),
               "left_persistent": g["left_persistent"].mean()}
        for share in (0.05, 0.20):
            row[f"token @{share:.0%}"] = build_labels(tables, c, token_share=share)["left_token"].mean()
        rows.append(row)
    tbl = pd.DataFrame(rows).drop(columns="label quarter")
    show = tbl.copy()
    for col in show.columns[2:]:
        show[col] = show[col].map(lambda v: fp(v) if pd.notna(v) else "-")
    d.p(f"Primary label `{LABEL}`: no orders in the next quarter, or revenue below {TOKEN_SHARE:.0%} of the "
        "partner's trailing 4-quarter average. Rates are shares of partners in each snapshot.")
    d.t(show)
    tbl.to_csv(FEATURE_DIR / "label_rates.csv", index=False)


def missingness(score: pd.DataFrame, d: Doc) -> None:
    d.h(f"2. Missing values in the {SCORE_CUTOFF} snapshot")
    miss = score[FEATURES].isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    d.p("Missing values are expected, not errors: they come from partners with too little history (new partners have "
        "no YoY; partners without recent orders have no line value). Tree models handle NaN natively; "
        "impute before linear models.")
    d.t(pd.DataFrame({"feature": miss.index, "missing": miss.map(fp).values}))


def univariate(panel: pd.DataFrame, d: Doc) -> pd.DataFrame:
    d.h(f"3. Univariate ROC-AUC vs `{LABEL}`")
    hist = panel[panel["cutoff"] != HOLDOUT]
    hold = panel[panel["cutoff"] == HOLDOUT]
    docs = feature_dictionary().set_index("feature")
    rows = []
    for f in NUMERIC:
        a_tr, a_ho = auc(hist[LABEL], hist[f]), auc(hold[LABEL], hold[f])
        rows.append({"feature": f, "group": docs.loc[f, "group"], "auc_history": a_tr, "auc_holdout": a_ho,
                     "strength": abs(a_ho - 0.5), "risk_when": "higher" if a_ho > 0.5 else "lower",
                     "coverage": panel[f].notna().mean()})
    tbl = pd.DataFrame(rows).sort_values("strength", ascending=False)
    tbl.to_csv(FEATURE_DIR / "univariate_auc.csv", index=False)
    d.p("AUC on non-missing rows. 0.5 = no signal; the further from 0.5 the stronger. `risk_when` says whether "
        "high or low values go with leaving. History = pooled snapshots before the holdout.")
    show = tbl.head(25).copy()
    for c in ["auc_history", "auc_holdout"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    show["coverage"] = show["coverage"].map(fp)
    d.t(show.drop(columns="strength"))

    top = tbl.head(20).iloc[::-1]
    fig, ax = ps.plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["strength"] + 0.5, left=0, color=ps.SERIES[0], height=0.6,
            edgecolor=ps.SURFACE, linewidth=2)
    for y, (v, w) in enumerate(zip(top["strength"] + 0.5, top["risk_when"])):
        ax.annotate(f"{v:.2f} ({w} = risk)", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=ps.INK_2)
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title(f"Top 20 features by holdout AUC ({HOLDOUT} snapshot)")
    d.img(ps.save(fig, CHARTS / "top_features_auc.png"), "Top features by AUC")
    return tbl


def redundancy(score: pd.DataFrame, d: Doc) -> None:
    d.h("4. Near-duplicate features (|Spearman| > 0.95)")
    corr = score[NUMERIC].corr(method="spearman").abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(NUMERIC) for b in NUMERIC[i + 1:] if corr.loc[a, b] > 0.95]
    if pairs:
        d.p("Keep both for tree models; drop one of each pair for linear models.")
        d.t(pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_spearman"]).round(3))
    else:
        d.p("None.")


def sanity_model(panel: pd.DataFrame, d: Doc) -> None:
    d.h("5. Sanity-check baseline model (not the final model)")
    train = panel[panel["cutoff"] != HOLDOUT]
    test = panel[panel["cutoff"] == HOLDOUT]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
    model.fit(train[FEATURES], train[LABEL])
    p = model.predict_proba(test[FEATURES])[:, 1]
    rows = [("Gradient boosting, all features", roc_auc_score(test[LABEL], p), average_precision_score(test[LABEL], p))]
    for f, sign in [("recency_vs_gap", 1), ("rev_q1_vs_prev4q", -1), ("lines_per_month_6m", -1)]:
        x = (sign * test[f]).fillna((sign * test[f]).max())
        rows.append((f"Single feature: {f}", roc_auc_score(test[LABEL], x), average_precision_score(test[LABEL], x)))
    res = pd.DataFrame(rows, columns=["model", "holdout_auc", "holdout_avg_precision"])
    d.p(f"Trained on {', '.join(TRAIN_CUTOFFS[:-1])}; tested on {HOLDOUT} (label = {LABEL} in the following quarter). "
        f"Holdout base rate {test[LABEL].mean():.1%} ({int(test[LABEL].sum())} of {len(test)}).")
    d.t(res.round(3))

    imp = permutation_importance(model, test[FEATURES], test[LABEL], scoring="roc_auc", n_repeats=5, random_state=0)
    top = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).head(12)
    d.p("Permutation importance on the holdout (drop in AUC when the feature is shuffled):")
    d.t(pd.DataFrame({"feature": top.index, "auc_drop": top.values.round(4)}))
    res.to_csv(FEATURE_DIR / "sanity_model.csv", index=False)


def decline_cohort(tables: dict, score: pd.DataFrame, d: Doc) -> None:
    d.h(f"6. The quietly declining cohort ({SCORE_CUTOFF})")
    s = score.set_index("partner_id")
    streak = s["yoy_decline_streak"]
    zero_now = s["rev_q1_log"].fillna(0) == 0
    counts = pd.DataFrame({
        "YoY decline streak >=": [2, 3, 4, 5],
        "partners": [int((streak >= k).sum()) for k in [2, 3, 4, 5]],
        f"of which zero revenue in {SCORE_CUTOFF}": [int(((streak >= k) & zero_now).sum()) for k in [2, 3, 4, 5]],
    })
    d.t(counts)
    cohort = streak.index[streak >= 4]
    prof = s.loc[cohort]
    d.p(f"Streak >= 4 cohort ({len(cohort)} partners): tiers {prof['tier'].value_counts().to_dict()}, "
        f"regions {prof['region'].value_counts().to_dict()}. These partners have shrunk year on year for a year or "
        f"more but still order every month, so a next-quarter label built from history barely sees them. Treat this "
        f"as a second risk mode next to the intermittent, already-silent partners.")

    pq = tables["partner_quarter"]
    rev = pq.pivot(index="partner_id", columns="quarter", values="revenue_usd")
    qs = sorted(rev.columns)
    in_c = rev.index.isin(cohort)
    fig, ax = ps.plt.subplots(figsize=(9, 3.8))
    for i, (mask, label) in enumerate([(in_c, f"Declining cohort ({in_c.sum()})"), (~in_c, "All other partners")]):
        med = rev[mask].median()
        idx = med / med[qs[:4]].mean() * 100
        ax.plot(qs, idx.values, color=ps.SERIES[i], label=label, marker="o", markersize=4)
    ax.axhline(100, color=ps.AXIS, linewidth=1)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Median quarterly revenue, indexed to the 2024 average = 100")
    ax.legend(loc="upper left")
    d.img(ps.save(fig, CHARTS / "declining_cohort.png"), "Declining cohort trajectory")
    d.p("Read with care: the cohort is selected on four down quarters ending at the cutoff, so the drop after "
        "2025-Q2 is partly by construction. The YoY growth distribution is one continuous spread, not a separate "
        "cluster, and historically these partners left at about the average rate (see the at-risk report).")
    prof.reset_index()[["partner_id", "tier", "region", "yoy_decline_streak", "yoy_2q", "rev_drawdown",
                        "att_miss_streak"]].to_csv(FEATURE_DIR / "declining_cohort.csv", index=False)


def run_diagnostics(tables: dict, panel: pd.DataFrame, score: pd.DataFrame) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    d = Doc()
    label_rates(tables, panel, d)
    missingness(score, d)
    univariate(panel, d)
    redundancy(score, d)
    sanity_model(panel, d)
    decline_cohort(tables, score, d)
    path = FEATURE_DIR / "feature_report.md"
    path.write_text("\n".join(d.lines), encoding="utf-8")
    print(f"\nReport: {path}")
