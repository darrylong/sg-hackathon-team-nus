"""Experiment: does XGBoost, deeper trees or up-weighting decliners beat the current at-risk ensemble?

    .venv/bin/python -m experiments.model_comparison

Same rolling-origin backtest as src/at_risk.py (each cutoff scored by a model trained on earlier cutoffs).
Writes outputs/models/model_comparison.md. Needs xgboost (not part of requirements.txt; see the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, roc_auc_score

from src.at_risk import EVAL_CUTOFFS, LABEL, MODEL_DIR, make_hgb, make_lr
from src.data_prep import build_all
from src.eda import md_table
from src.features import FEATURES, TRAIN_CUTOFFS, build_panel


def make_hgb_deep():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=5, min_samples_leaf=40,
                                          l2_regularization=1.0, categorical_features="from_dtype", random_state=0)


def make_xgb(scale_pos_weight: float):
    return xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=1.0, reg_alpha=0.5, scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                             tree_method="hist", enable_categorical=True, random_state=0, verbosity=0)


def make_lgbm(**overrides):
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1)
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


def decliner_weight(df: pd.DataFrame, w: float = 3.0) -> np.ndarray:
    return np.where(df["yoy_decline_streak"] >= 2, w, 1.0)


VARIANTS = {
    "current: LR + HGB": [("lr", make_lr, False), ("hgb", make_hgb, False)],
    "HGB": [("hgb", make_hgb, False)],
    "HGB max_depth=5": [("hgb", make_hgb_deep, False)],
    "HGB, decliners weighted 3x": [("hgb", make_hgb, True)],
    "XGB (plan: scale_pos_weight=50)": [("xgb", lambda: make_xgb(50), False)],
    "XGB (scale_pos_weight=1)": [("xgb", lambda: make_xgb(1), False)],
    "LR + XGB (spw=1)": [("lr", make_lr, False), ("xgb", lambda: make_xgb(1), False)],
    "LGBM": [("lgbm", make_lgbm, False)],
    "LGBM (is_unbalance)": [("lgbm", lambda: make_lgbm(is_unbalance=True), False)],
    "LGBM (regularised: 7 leaves, 80 min samples)": [
        ("lgbm", lambda: make_lgbm(num_leaves=7, min_child_samples=80, n_estimators=500, learning_rate=0.02), False)],
    "LR + LGBM": [("lr", make_lr, False), ("lgbm", make_lgbm, False)],
    "LR + LGBM (regularised)": [
        ("lr", make_lr, False),
        ("lgbm", lambda: make_lgbm(num_leaves=7, min_child_samples=80, n_estimators=500, learning_rate=0.02), False)],
    "LR + HGB + LGBM (regularised)": [
        ("lr", make_lr, False), ("hgb", make_hgb, False),
        ("lgbm", lambda: make_lgbm(num_leaves=7, min_child_samples=80, n_estimators=500, learning_rate=0.02), False)],
    "LR": [("lr", make_lr, False)],
}


def paired_bootstrap(store: dict, a: str, b: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean-over-cutoffs AUC difference (b - a), resampling partners within each cutoff. Returns diff, lo95, hi95."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        d = []
        for c in EVAL_CUTOFFS:
            y, pa = store[(c, a)]
            _, pb = store[(c, b)]
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() == y[idx].max():
                continue
            d.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
        diffs.append(np.mean(d))
    point = np.mean([roc_auc_score(store[(c, b)][0], store[(c, b)][1]) - roc_auc_score(store[(c, a)][0], store[(c, a)][1])
                     for c in EVAL_CUTOFFS])
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def decliner_check(panel: pd.DataFrame) -> pd.DataFrame:
    """Step 1: did historical decliners leave more often than partners of the same size and tier?"""
    P = panel.copy()
    P["size_decile"] = P.groupby("cutoff")["rev_avg_4q_log"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    groups = {
        "YoY decline streak >= 4": P["yoy_decline_streak"] >= 4,
        "YoY decline streak >= 3": P["yoy_decline_streak"] >= 3,
        "YoY decline streak >= 2": P["yoy_decline_streak"] >= 2,
        "last 2Q YoY <= -25%": P["yoy_2q"] <= -1 / 7,  # symmetric growth for -25%
        "revenue < 50% of peak": P["rev_drawdown"] < 0.5,
    }
    rows = []
    for name, mask in groups.items():
        g, rest = P[mask], P[~mask]
        ref = rest.groupby(["cutoff", "tier", "size_decile"], observed=True)[LABEL].mean()
        expected = g.set_index(["cutoff", "tier", "size_decile"]).index.map(ref).to_series()
        expected = expected.fillna(rest[LABEL].mean()).sum()
        a, n, c, m = int(g[LABEL].sum()), len(g), int(rest[LABEL].sum()), len(rest)
        rows.append({"group": name, "partner_snapshots": n, "leavers": a,
                     "expected_at_same_size_tier": round(expected, 1), "ratio": round(a / expected, 2),
                     "fisher_p": round(fisher_exact([[a, n - a], [c, m - c]])[1], 3)})
    return pd.DataFrame(rows)


def main() -> None:
    panel = build_panel(build_all(), TRAIN_CUTOFFS)
    rows, store = [], {}
    for c in EVAL_CUTOFFS:
        train, test = panel[panel["cutoff"] < c], panel[panel["cutoff"] == c]
        y = test[LABEL].to_numpy()
        decl = (test["yoy_decline_streak"] >= 2).to_numpy()
        for name, parts in VARIANTS.items():
            preds = []
            for _, factory, weighted in parts:
                m = factory()
                kw = {"sample_weight": decliner_weight(train)} if weighted else {}
                m.fit(train[FEATURES], train[LABEL], **kw)
                preds.append(m.predict_proba(test[FEATURES])[:, 1])
            p = np.mean(preds, axis=0)
            store[(c, name)] = (y, p)
            rows.append({"cutoff": c, "variant": name, "auc": roc_auc_score(y, p),
                         "avg_precision": average_precision_score(y, p),
                         "decliners_flagged_top100": int(decl[np.argsort(-p)[:100]].sum())})
    res = pd.DataFrame(rows)
    summ = res.groupby("variant").agg(auc_mean=("auc", "mean"), auc_min=("auc", "min"),
                                      avg_precision=("avg_precision", "mean"),
                                      decliners_in_top100=("decliners_flagged_top100", "mean"))
    summ = summ.reindex(list(VARIANTS)).round(3)
    per = res.pivot(index="variant", columns="cutoff", values="auc").reindex(list(VARIANTS)).round(3)
    base = "current: LR + HGB"
    challengers = [v for v in summ.sort_values("auc_mean", ascending=False).index if v != base][:4]
    boot = pd.DataFrame([(v, *paired_bootstrap(store, base, v)) for v in challengers],
                        columns=["challenger", "auc_diff_vs_current", "ci95_low", "ci95_high"]).round(4)

    L = ["# Model comparison", "",
         "## Step 1: did historical decliners leave more often?", "",
         md_table(decliner_check(panel)), "",
         "Decline streaks show no excess leaving once size and tier are accounted for. Distance from peak revenue "
         "does (about 1.35x), and is already a model feature (`rev_drawdown`).", "",
         f"## Step 2-3: model variants (rolling-origin backtest over {', '.join(EVAL_CUTOFFS)}, label `{LABEL}`)", "",
         md_table(summ.reset_index()), "", "AUC per cutoff:", "", md_table(per.reset_index()), "",
         "Paired bootstrap (1,000 resamples of partners within each cutoff) of the mean AUC difference against the "
         "current model, for the four best challengers. An interval containing 0 means no real difference:", "",
         md_table(boot), "",
         "## Decision", "",
         "Keep the current logistic regression + gradient boosting ensemble. The best challengers (LR + regularised "
         "LightGBM, LR + XGBoost) tie with it within bootstrap noise, and it keeps the best worst-case cutoff. Single "
         "boosted models (HGB, XGBoost, LightGBM) all trail the ensembles, and class re-weighting (XGBoost "
         "scale_pos_weight=50, LightGBM is_unbalance) or up-weighting decliners lowers AUC. XGBoost and LightGBM are not "
         "added to requirements.txt: on macOS both need the OpenMP runtime (`brew install libomp`), which would make "
         "the one-command setup fragile for the judges for no measurable gain.", ""]
    (MODEL_DIR / "model_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
