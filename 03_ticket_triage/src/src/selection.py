"""Round 2C selection rules and paired-bootstrap comparisons between runs (VAL only)."""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data import ROOT, TARGETS, load_splits, load_tickets
from evaluate import analyse, fast_macro_f1
from runs import RUNS_DIR

BASELINE = "baseline_lr"
COST_SLACK = 0.002
N_BOOT, SEED = 1000, 42
RULES_MD = ROOT / "outputs" / "round2c" / "selection_rules.md"

RULES = {
    "priority": [
        "(a) VAL policy-f cost point estimate <= baseline_lr + 0.002",
        "(b) P1 PR-AUC improves vs the current priority choice, paired-bootstrap 95% CI excluding 0",
        "(c) priority macro F1 does not drop significantly vs the current choice (CI upper bound >= 0)",
    ],
    "category": ["macro F1 improves vs the current choice, 95% CI excluding 0"],
    "sentiment": ["macro F1 improves vs the current choice, 95% CI excluding 0"],
    "assigned_team": ["Accept a team change if EITHER (i) OR (ii):",
                      "(i) VAL policy-f cost improves vs the current choice, 95% CI excluding 0, AND team macro F1 "
                      "does not drop significantly vs the current choice (CI upper bound >= 0)",
                      "(ii) VAL policy-f cost point estimate <= baseline_lr + 0.002 AND team macro F1 improves vs "
                      "the current choice, 95% CI excluding 0"],
    "notes": ["Bootstrap: 1000 paired resamples of VAL tickets, seed 42.",
              "Priority label = argmax of the chosen run's calibrated probabilities; no P1 threshold.",
              "TEST split stays locked; repeated VAL comparisons make selected results slightly optimistic."],
}

_val = None
_cache = {}


def val_frame():
    global _val
    if _val is None:
        _val = load_splits(load_tickets())["val"]
    return _val


def get(run):
    if run not in _cache:
        _cache[run] = analyse(run, val_frame())
    return _cache[run]


def cost_f(run):
    return float(get(run)["routing"]["per_ticket"]["f_expcost_pcorrect"].mean())


def _p1(res):
    pr = res["targets"]["priority"]
    return pr["p_cal"][:, pr["classes"].index("P1")]


def compare(run, ref) -> dict:
    """{metric: (diff, lo, hi)} for run - ref, paired bootstrap on VAL."""
    a, b = get(run), get(ref)
    y_p1 = val_frame()["priority"].to_numpy() == "P1"
    idx = np.random.default_rng(SEED).integers(0, len(y_p1), size=(N_BOOT, len(y_p1)))
    out = {}

    def add(name, fa, fb):
        d = np.array([fa(i) - fb(i) for i in idx])
        full = np.arange(len(y_p1))
        out[name] = (float(fa(full) - fb(full)), *map(float, np.percentile(d, [2.5, 97.5])))

    for t in TARGETS:
        ta, tb = a["targets"][t], b["targets"][t]
        k = len(ta["classes"])
        pa, pb, y = ta["p_cal"].argmax(1), tb["p_cal"].argmax(1), ta["y"]
        add(f"{t}_macro_f1", lambda i: fast_macro_f1(y[i], pa[i], k), lambda i: fast_macro_f1(y[i], pb[i], k))
    p1a, p1b = _p1(a), _p1(b)
    add("P1_PR_AUC", lambda i: average_precision_score(y_p1[i], p1a[i]),
        lambda i: average_precision_score(y_p1[i], p1b[i]))
    add("P1_AUROC", lambda i: roc_auc_score(y_p1[i], p1a[i]), lambda i: roc_auc_score(y_p1[i], p1b[i]))
    ca = a["routing"]["per_ticket"]["f_expcost_pcorrect"]
    cb = b["routing"]["per_ticket"]["f_expcost_pcorrect"]
    add("policy_f_cost", lambda i: ca[i].mean(), lambda i: cb[i].mean())
    return out


def check_priority(run, current):
    c = compare(run, current)
    checks = {"a_cost": cost_f(run) <= cost_f(BASELINE) + COST_SLACK,
              "b_pr_auc": c["P1_PR_AUC"][1] > 0,
              "c_macro_f1": c["priority_macro_f1"][2] >= 0}
    return all(checks.values()), checks, c


def check_team(run, current):
    c = compare(run, current)
    team_f1, cost = c["assigned_team_macro_f1"], c["policy_f_cost"]
    checks = {"i_cost_improves": cost[2] < 0, "i_team_f1_no_drop": team_f1[2] >= 0,
              "ii_cost_within_slack": cost_f(run) <= cost_f(BASELINE) + COST_SLACK, "ii_team_f1_improves": team_f1[1] > 0}
    passed = (checks["i_cost_improves"] and checks["i_team_f1_no_drop"]) or              (checks["ii_cost_within_slack"] and checks["ii_team_f1_improves"])
    return passed, checks, c


def check_f1(run, current, target):
    c = compare(run, current)
    checks = {"f1": c[f"{target}_macro_f1"][1] > 0}
    return all(checks.values()), checks, c


def write_rules_md():
    RULES_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Selection rules (round 2C)", ""]
    for k, items in RULES.items():
        lines.append(f"## {k}")
        lines += [f"- {s}" for s in items] + [""]
    RULES_MD.write_text("\n".join(lines), encoding="utf-8")


def stamp_config(run):
    path = RUNS_DIR / run / "config.json"
    cfg = json.loads(path.read_text())
    cfg["selection_rules"] = {"file": "outputs/round2c/selection_rules.md", **RULES}
    path.write_text(json.dumps(cfg, indent=2, default=str))


def stamp_all():
    for p in RUNS_DIR.iterdir():
        if (p / "config.json").exists():
            stamp_config(p.name)
