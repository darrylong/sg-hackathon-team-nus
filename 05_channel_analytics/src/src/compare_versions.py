"""Compare v1 (frozen folder) with v2 (this folder): forecast, backtest accuracy and at-risk output.

    python -m src.compare_versions

v1 is only read, never written. Its location is V1_DIR (env var), default ../Hackathon next to this folder.
Writes outputs/version_comparison.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_prep import REGIONS, ROOT
from .eda import md_table
from .forecast import AREAS, METHODS, loo_check

V1_DIR = Path(os.environ.get("V1_DIR", ROOT.parent / "Hackathon"))
OUT = ROOT / "outputs" / "version_comparison.md"
TOL = 1.0  # dollars: v2 core3 must reproduce v1's ensemble to within rounding


def _paths(base: Path) -> dict[str, Path]:
    return {"forecast": base / "submission" / "submission_forecast.csv",
            "at_risk": base / "submission" / "submission_at_risk.csv",
            "backtest": base / "outputs" / "models" / "forecast_backtest.csv",
            "scores": base / "outputs" / "models" / "partner_scores.csv"}


def money(v: float) -> str:
    return f"${v / 1e6:,.1f}M"


def pct(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:.1%}"


def consistency(bt1: pd.DataFrame, bt2: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """v2's core3 ensemble and the three v1 methods must equal v1's backtest numbers."""
    rows = []
    for m1, m2 in [("ensemble", "core3"), ("seasonal_ratio", "seasonal_ratio"), ("yoy_growth", "yoy_growth"),
                   ("regression", "regression")]:
        a = bt1[bt1["method"] == m1].set_index(["quarter", "area"])["forecast"]
        b = bt2[bt2["method"] == m2].set_index(["quarter", "area"])["forecast"]
        diff = (a - b.reindex(a.index)).abs()
        rows.append({"v1 method": m1, "v2 method": m2, "rows": len(a), "max abs diff ($)": round(float(diff.max()), 2)})
    tbl = pd.DataFrame(rows)
    return bool((tbl["max abs diff ($)"] <= TOL).all()), tbl


def run() -> Path:
    p1, p2 = _paths(V1_DIR), _paths(ROOT)
    L = ["# v1 vs v2 comparison", "", f"v1: `{V1_DIR}` (read only)  \nv2: `{ROOT}`", ""]

    def t(df):
        L.extend([md_table(df), ""])

    missing = [k for k, v in p1.items() if not v.exists()]
    if missing:
        L += [f"**v1 files missing ({', '.join(missing)}); comparison skipped.** Run `python -m src.run_all` in v1 first.", ""]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(L), encoding="utf-8")
        return OUT

    meta = json.loads((ROOT / "outputs" / "models" / "forecast_meta.json").read_text())
    bt1, bt2 = pd.read_csv(p1["backtest"]), pd.read_csv(p2["backtest"])

    # 1. consistency
    ok, cons = consistency(bt1, bt2)
    L += ["## 1. Consistency check", "",
          "v2 must reproduce v1's backtest for the methods both versions share, otherwise the comparison below is not "
          "like for like.", ""]
    t(cons)
    L += ["**PASS** - v2's core3 reproduces v1 exactly." if ok else
          "**FAIL - v2 does not reproduce v1; treat the accuracy comparison with caution.**", ""]

    # 2. forecast side by side
    f1 = pd.read_csv(p1["forecast"]).set_index("region")
    f2 = pd.read_csv(p2["forecast"]).set_index("region")
    order = ["APJ", "EMEA", "AMS", "ALL"]
    fc = pd.DataFrame({
        "v1 forecast": f1["forecast_revenue_usd"], "v1 80% range": f1["lo80_usd"].map(money) + " - " + f1["hi80_usd"].map(money),
        "v1 width": (f1["hi80_usd"] - f1["lo80_usd"]) / f1["forecast_revenue_usd"],
        "v2 forecast": f2["forecast_revenue_usd"], "v2 80% range": f2["lo80_usd"].map(money) + " - " + f2["hi80_usd"].map(money),
        "v2 width": (f2["hi80_usd"] - f2["lo80_usd"]) / f2["forecast_revenue_usd"],
    }).reindex(order)
    fc["v2 - v1"] = fc["v2 forecast"] - fc["v1 forecast"]
    fc["v2 vs v1"] = fc["v2 forecast"] / fc["v1 forecast"] - 1
    show = fc.copy()
    for c in ["v1 forecast", "v2 forecast", "v2 - v1"]:
        show[c] = show[c].map(money)
    for c in ["v1 width", "v2 width"]:
        show[c] = show[c].map(lambda v: f"±{v / 2:.1%}")
    show["v2 vs v1"] = show["v2 vs v1"].map(lambda v: f"{v:+.1%}")
    L += ["## 2. 2026-Q3 forecast", ""]
    t(show.reset_index())

    # 3. accuracy, like for like (same quarters, same leave-one-out procedure)
    loo1 = loo_check(bt1, "ensemble")
    loo2 = loo_check(bt2, meta["chosen_ensemble"])
    acc = pd.DataFrame({
        "v1 MAPE (core3, bias-corrected)": loo1.groupby("area")["ape_bias_corrected"].mean(),
        f"v2 MAPE ({meta['chosen_ensemble']}, bias-corrected)": loo2.groupby("area")["ape_bias_corrected"].mean(),
        "v1 inside 80%": loo1.groupby("area")["inside_80"].mean(),
        "v2 inside 80%": loo2.groupby("area")["inside_80"].mean(),
    }).reindex(AREAS)
    acc.loc["mean"] = acc.mean()
    L += ["## 3. Backtest accuracy (leave-one-quarter-out, 2025-Q2 .. 2026-Q2)", ""]
    t(acc.map(pct).reset_index().rename(columns={"index": "area"}))

    raw = bt2.assign(ape=bt2["pct_error"].abs()).groupby(["method", "area"])["ape"].mean().unstack()[AREAS]
    raw = raw.reindex(METHODS + ["core3", "all6"])
    L += ["Raw (not bias-corrected) MAPE of every v2 method - v1 is the core3 row:", ""]
    t(raw.map(pct).reset_index())

    cb_path = ROOT / "outputs" / "models" / "churn_adjustment_backtest.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path)
        if len(cb):
            c = cb.groupby("area")[["ape_without", "ape_with"]].mean().reindex(AREAS)
            L += [f"Churn adjustment backtest (applied in v2: {'yes' if meta['churn_adjustment_applied'] else 'no'}; "
                  f"live adjustment for ALL {money(meta['churn_adjustment_usd']['ALL'])}):", ""]
            t(c.map(lambda v: f"{v:.2%}").reset_index())

    # 4. at-risk
    a1 = pd.read_csv(p1["at_risk"]).set_index("partner_id")
    a2 = pd.read_csv(p2["at_risk"]).set_index("partner_id").reindex(a1.index)
    flags1, flags2 = set(a1.index[a1["at_risk"] == 1]), set(a2.index[a2["at_risk"] == 1])
    rho = spearmanr(a1["health_score"], a2["health_score"]).correlation
    identical = a1.equals(a2)
    max_diff = float((a1["health_score"] - a2["health_score"]).abs().max())
    risk = pd.DataFrame([
        ("partners", f"{len(a1):,}", f"{len(a2):,}"), ("flagged", str(len(flags1)), str(len(flags2))),
        ("flagged in both versions", str(len(flags1 & flags2)), ""),
        ("Spearman correlation of health scores", f"{rho:.4f}", ""),
        ("largest health-score difference", f"{max_diff:.4f}", ""),
    ], columns=["metric", "v1", "v2"])
    L += ["## 4. At-risk submission", ""]
    t(risk)
    L += ["Identical files: **yes** (v2 did not change the at-risk model)." if identical else
          "Files differ - see the table above.", ""]

    # 5. verdict
    m1 = acc.loc["mean", "v1 MAPE (core3, bias-corrected)"]
    m2 = acc.loc["mean", f"v2 MAPE ({meta['chosen_ensemble']}, bias-corrected)"]
    c1, c2 = acc.loc["mean", "v1 inside 80%"], acc.loc["mean", "v2 inside 80%"]
    better = m2 < m1
    verdict = (f"v2's forecast is more accurate in the backtest (mean leave-one-out MAPE {m2:.1%} vs {m1:.1%}) "
               if better else f"v2's forecast is not more accurate (mean leave-one-out MAPE {m2:.1%} vs {m1:.1%}) ")
    verdict += (f"with the same backtest coverage ({c2:.0%} vs {c1:.0%}) " if np.isclose(c1, c2) else
                f"with backtest coverage {c2:.0%} vs {c1:.0%} ")
    verdict += (f"and a narrower interval (ALL ±{fc.loc['ALL', 'v2 width'] / 2:.1%} vs ±{fc.loc['ALL', 'v1 width'] / 2:.1%}). "
                if fc.loc["ALL", "v2 width"] < fc.loc["ALL", "v1 width"] else
                f"and a wider interval (ALL ±{fc.loc['ALL', 'v2 width'] / 2:.1%} vs ±{fc.loc['ALL', 'v1 width'] / 2:.1%}). ")
    verdict += ("Recommendation: submit v2's forecast. " if better and ok else
                "Recommendation: keep v1's forecast. " if not better else
                "Recommendation: fix the consistency check before deciding. ")
    verdict += ("The at-risk file is identical in both versions. " if identical else "")
    verdict += ("Caveat: five backtest quarters only; a narrower interval raises the risk of missing the actual value "
                "if 2026-Q3 behaves unlike the backtest quarters.")
    L += ["## 5. Verdict", "", verdict, ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    return OUT


def main() -> None:
    print(run().read_text())


if __name__ == "__main__":
    main()
