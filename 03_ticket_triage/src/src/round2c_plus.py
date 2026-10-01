"""Round 2C+: richer team-correctness model on combined_v3.

python src/round2c_plus.py
Builds run combined_v3_corr = combined_v3's probability files (copied, unchanged) + correctness.csv from
correctness_ext.py, runs evaluate.py --run combined_v3_corr --ref combined_v3, then applies the team rule
(selection.check_team, read-only) vs combined_v3. Writes outputs/round2c_plus/round2c_plus_report.txt and
decisions.json. Nothing in outputs/runs/combined_v3/ is written.
"""
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import selection as sel
from correctness_ext import E1_RUN, NEW_FEATURES, run_correctness_ext
from data import ROOT
from runs import RUNS_DIR, load_run_frame

OUT = ROOT / "outputs" / "round2c_plus"
SRC_RUN, RUN = "combined_v3", "combined_v3_corr"
COPY = ["oof_train.csv", "val.csv", "oof_calibrated.csv", "val_calibrated.csv", "calibration.json"]
POLICY = "f_expcost_pcorrect"


def fmt(ci):
    return f"{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]"


def build_run():
    src, dst = RUNS_DIR / SRC_RUN, RUNS_DIR / RUN
    dst.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copyfile(src / f, dst / f)
    cfg = json.loads((src / "config.json").read_text())
    cfg.update({"run_name": RUN, "source_run": SRC_RUN,
                "change": f"{SRC_RUN} probabilities unchanged (files copied); correctness.csv from "
                          f"correctness_ext.py (+{', '.join(NEW_FEATURES)}; E1 run {E1_RUN})"})
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, default=str))


def evaluate():
    res = subprocess.run([sys.executable, str(ROOT / "src" / "evaluate.py"), "--run", RUN, "--ref", SRC_RUN],
                         capture_output=True, text=True, cwd=ROOT)
    (RUNS_DIR / RUN / "evaluate_stdout.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"evaluate failed:\n{res.stderr[-2000:]}")


def auroc_compare(correct):
    """Team-error AUROC of old vs new p_correct on VAL, with paired bootstrap CI of new - old."""
    is_err = ~correct
    old = -load_run_frame(SRC_RUN, "correctness.csv")["p_correct"].to_numpy()
    new = -load_run_frame(RUN, "correctness.csv")["p_correct"].to_numpy()
    idx = np.random.default_rng(sel.SEED).integers(0, len(is_err), size=(sel.N_BOOT, len(is_err)))
    d = np.array([roc_auc_score(is_err[i], new[i]) - roc_auc_score(is_err[i], old[i]) for i in idx])
    a_old, a_new = roc_auc_score(is_err, old), roc_auc_score(is_err, new)
    return a_old, a_new, (a_new - a_old, *np.percentile(d, [2.5, 97.5]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_run()
    info = run_correctness_ext(RUN)
    evaluate()

    old_info = json.loads((RUNS_DIR / SRC_RUN / "correctness.json").read_text())
    m_new = json.loads((RUNS_DIR / RUN / "metrics.json").read_text())
    m_old = json.loads((RUNS_DIR / SRC_RUN / "metrics.json").read_text())
    ok, checks, c = sel.check_team(RUN, SRC_RUN)
    r_new = sel.get(RUN)["routing"]
    a_old, a_new, a_ci = auroc_compare(r_new["correct"])
    base_cost = sel.cost_f(sel.BASELINE)
    L = []

    L.append(f"===== Round 2C+: {RUN} ({SRC_RUN} probabilities, extended correctness model) =====")
    L.append(f"New features: {', '.join(NEW_FEATURES)} (E1 run {E1_RUN}); all base correctness.py features kept.")
    L.append(f"Sanity: team macro F1 diff vs {SRC_RUN} = {fmt(c['assigned_team_macro_f1'])} (identical probabilities)")

    L.append("\n----- Team-error AUROC (higher = better at ranking misroutes) -----")
    rows = [{"split": "TRAIN OOF", "p_ok": info["train_oof"]["auroc_p_ok"],
             "old_p_correct": old_info["train_oof"]["auroc_p_correct"], "new_p_correct": info["train_oof"]["auroc_p_correct"],
             "old_logloss": old_info["train_oof"]["logloss_p_correct"], "new_logloss": info["train_oof"]["logloss_p_correct"]},
            {"split": "VAL", "p_ok": info["val"]["auroc_p_ok"], "old_p_correct": a_old, "new_p_correct": a_new,
             "old_logloss": old_info["val"]["logloss_p_correct"], "new_logloss": info["val"]["logloss_p_correct"]}]
    L.append(pd.DataFrame(rows).round(4).to_string(index=False))
    L.append(f"VAL AUROC new - old (paired bootstrap): {fmt(a_ci)}")

    L.append("\n----- Policy f (expected cost with p_correct) on VAL -----")
    rows = []
    for name, m in ((SRC_RUN, m_old), (RUN, m_new)):
        f = m["routing"][POLICY]
        rows.append({"run": name, "cost_f": f["mean_cost"], "abstain_rate": f["abstain_rate"],
                     "n_abstain": f["n_abstain"],
                     **{f"misr_{p}": f[f"misr_{p}"] for p in ("P1", "P2", "P3", "P4")}})
    L.append(pd.DataFrame(rows).round(4).to_string(index=False))
    L.append(f"cost_f {RUN} - {SRC_RUN} (selection.compare): {fmt(c['policy_f_cost'])}")
    b = m_new["bootstrap_vs_ref"]["policy_f_cost"]
    L.append(f"cost_f {RUN} - {SRC_RUN} (evaluate.py --ref):   {fmt((b['diff'], b['lo'], b['hi']))}")
    L.append(f"team-error AUROC as reported by evaluate.py: {SRC_RUN} {m_old['team_error_auroc']['p_correct']:.4f}, "
             f"{RUN} {m_new['team_error_auroc']['p_correct']:.4f}")

    L.append("\n----- Policy-f efficiency by TRUE priority -----")
    for name, m in ((SRC_RUN, m_old), (RUN, m_new)):
        L.append(f"{name}:\n" + pd.DataFrame(m["efficiency"][POLICY]).round(4).to_string(index=False))

    L.append("\n----- New-feature groups (team error rate by agreement flag) -----")
    for k, v in info["feature_groups"].items():
        L.append(f"{k:24s} " + "  ".join(f"{g}: n={d['n']} err={d['team_error_rate']:.4f}" for g, d in v.items()))

    L.append("\n----- Correctness-model coefficients (full-TRAIN fit; numeric features standardized) -----")
    coef = pd.Series(info["coefficients_standardized"]).rename("coef").reset_index().rename(columns={"index": "feature"})
    coef["new"] = coef["feature"].isin(NEW_FEATURES).map({True: "*", False: ""})
    L.append(coef.round(4).to_string(index=False))

    L.append(f"\n----- Team rule vs {SRC_RUN} (applied to cost; probabilities unchanged) -----")
    L.append(f"(i)  cost_f CI upper < 0: {checks['i_cost_improves']}  ({fmt(c['policy_f_cost'])}); "
             f"team F1 no significant drop: {checks['i_team_f1_no_drop']}")
    L.append(f"(ii) cost_f {sel.cost_f(RUN):.4f} <= baseline_lr + {sel.COST_SLACK} = {base_cost + sel.COST_SLACK:.4f}: "
             f"{checks['ii_cost_within_slack']}; team F1 improves: {checks['ii_team_f1_improves']}")
    verdict = "ACCEPT combined_v3_corr" if ok else "REJECT combined_v3_corr; keep combined_v3"
    L.append(f"==> {verdict}")

    decisions = {"run": RUN, "ref": SRC_RUN, "accepted": bool(ok), "team_rule": {k: bool(v) for k, v in checks.items()},
                 "policy_f_cost": {"run": sel.cost_f(RUN), "ref": sel.cost_f(SRC_RUN),
                                   "diff_ci": list(c["policy_f_cost"])},
                 "val_team_error_auroc": {"old": a_old, "new": a_new, "diff_ci": list(a_ci)},
                 "correctness_choice": RUN if ok else SRC_RUN}
    (OUT / "round2c_plus_report.txt").write_text("\n".join(L), encoding="utf-8")
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
