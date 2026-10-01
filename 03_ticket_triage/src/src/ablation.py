"""Collect metrics.json of evaluated runs into outputs/round2b/ablation.csv.

python src/ablation.py [run ...]   (default: every run evaluated against baseline_lr)
"""
import json
import sys

import pandas as pd

from data import ROOT
from runs import RUNS_DIR

OUT = ROOT / "outputs" / "round2b" / "ablation.csv"


def row(run):
    m = json.loads((RUNS_DIR / run / "metrics.json").read_text())
    cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
    t, r, ci = m["targets"], m["routing"], m["bootstrap_vs_ref"]["policy_f_cost"]
    return {
        "run": run, "change": cfg.get("change", cfg.get("description", "")),
        "category_F1": t["category"]["macro_f1"], "priority_F1": t["priority"]["macro_f1"],
        "P1_recall": m["P1"]["recall"], "team_acc": t["assigned_team"]["accuracy"],
        "team_F1": t["assigned_team"]["macro_f1"], "sentiment_F1": t["sentiment"]["macro_f1"],
        "escalation_F1": m["escalation"]["f1"], "ECE_team": t["assigned_team"]["ece_cal"],
        "team_error_AUROC": m["team_error_auroc"]["p_correct"],
        "cost_b": r["b_naive_low_conf"]["mean_cost"], "cost_f": r["f_expcost_pcorrect"]["mean_cost"],
        "abstain_rate_f": r["f_expcost_pcorrect"]["abstain_rate"],
        "cost_f_CI_vs_baseline": f"{ci['diff']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]",
    }


def main(runs=None):
    if not runs:
        runs = sorted(p.name for p in RUNS_DIR.iterdir() if (p / "metrics.json").exists()
                      and json.loads((p / "metrics.json").read_text()).get("bootstrap_vs_ref", {}).get("ref") == "baseline_lr")
        runs = ["baseline_lr"] + [r for r in runs if r != "baseline_lr"]
    df = pd.DataFrame([row(r) for r in runs])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = main(sys.argv[1:])
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40):
        print(df.drop(columns="change").round(4).to_string(index=False))
