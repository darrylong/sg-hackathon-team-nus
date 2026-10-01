"""P1 ranking / thresholding analysis on VAL (read-only on runs).

python src/p1_analysis.py  -> outputs/p1/p1_report.txt, p1_metrics.csv
- P1-vs-rest AUROC and PR-AUC (calibrated) for baseline_lr, E1_char_tfidf, combined_v1 (final), E11_p1_weighted.
- P1 P/R/F1 at argmax and at an F2-optimal threshold chosen on TRAIN OOF calibrated P(P1).
  Threshold rule: predict P1 if P(P1) >= thr, else argmax over P2..P4.
- Paired bootstrap (1000, seed 42) of E11 - final for P1 PR-AUC / AUROC.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

from data import ROOT, load_splits, load_tickets
from runs import RUNS_DIR, load_run_frame, proba_matrix

OUT = ROOT / "outputs" / "p1"
RUNS = ["baseline_lr", "E1_char_tfidf", "combined_v1", "E11_p1_weighted"]
FINAL, CANDIDATE = "combined_v1", "E11_p1_weighted"
N_BOOT, SEED = 1000, 42


def f2_threshold(y_p1, p1):
    prec, rec, thr = precision_recall_curve(y_p1, p1)
    f2 = 5 * prec * rec / np.clip(4 * prec + rec, 1e-12, None)
    i = np.nanargmax(f2[:-1])
    return float(thr[i])


def threshold_predict(classes, p, thr):
    i1 = classes.index("P1")
    rest = p.copy()
    rest[:, i1] = -1
    pred = np.array(classes)[rest.argmax(1)]
    pred[p[:, i1] >= thr] = "P1"
    return pred


def p1_prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=["P1"], zero_division=0)
    f2 = fbeta_score(y == "P1", pred == "P1", beta=2, zero_division=0)
    return p[0], r[0], f[0], f2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_tickets()
    parts = load_splits(df)
    y_val, y_tr = parts["val"]["priority"].to_numpy(), parts["train"]["priority"].to_numpy()
    rows, probs = [], {}
    for run in RUNS:
        classes, p_val = proba_matrix(load_run_frame(run, "val_calibrated.csv"), "priority")
        _, p_oof = proba_matrix(load_run_frame(run, "oof_calibrated.csv"), "priority")
        i1 = classes.index("P1")
        probs[run] = p_val[:, i1]
        thr = f2_threshold(y_tr == "P1", p_oof[:, i1])
        pred_arg = np.array(classes)[p_val.argmax(1)]
        pred_thr = threshold_predict(classes, p_val, thr)
        m = json.loads((RUNS_DIR / run / "metrics.json").read_text())
        ci = m["bootstrap_vs_ref"]["policy_f_cost"]
        a, t = p1_prf(y_val, pred_arg), p1_prf(y_val, pred_thr)
        rows.append({
            "run": run, "P1_AUROC": roc_auc_score(y_val == "P1", p_val[:, i1]),
            "P1_PR_AUC": average_precision_score(y_val == "P1", p_val[:, i1]),
            "argmax_P": a[0], "argmax_R": a[1], "argmax_F1": a[2], "argmax_F2": a[3],
            "F2_thr_oof": thr, "thr_P": t[0], "thr_R": t[1], "thr_F1": t[2], "thr_F2": t[3],
            "prio_macroF1_argmax": f1_score(y_val, pred_arg, average="macro"),
            "prio_macroF1_thr": f1_score(y_val, pred_thr, average="macro"),
            "cost_f": m["routing"]["f_expcost_pcorrect"]["mean_cost"],
            "cost_f_vs_baseline": f"{ci['diff']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]",
        })
    table = pd.DataFrame(rows)

    # paired bootstrap: candidate - final
    yb = (y_val == "P1")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(yb), size=(N_BOOT, len(yb)))
    boot = {}
    for name, fn in (("PR_AUC", average_precision_score), ("AUROC", roc_auc_score)):
        d = np.array([fn(yb[i], probs[CANDIDATE][i]) - fn(yb[i], probs[FINAL][i]) for i in idx])
        boot[name] = (fn(yb, probs[CANDIDATE]) - fn(yb, probs[FINAL]), *np.percentile(d, [2.5, 97.5]))
    cost_ci = json.loads((RUNS_DIR / CANDIDATE / "metrics.json").read_text())["bootstrap_vs_ref"]["policy_f_cost"]
    pr_ok = boot["PR_AUC"][1] > 0
    cost_ok = cost_ci["hi"] < 0
    decision = (f"USE {CANDIDATE} for priority" if pr_ok and cost_ok else
                f"KEEP unweighted priority model ({FINAL}) + F2 threshold "
                f"{table.set_index('run').at[FINAL, 'F2_thr_oof']:.3f}")

    L = ["P1 ranking / weighting (VAL, calibrated probabilities; F2 threshold chosen on TRAIN OOF)", "",
         table.round(4).to_string(index=False), "",
         f"Paired bootstrap {CANDIDATE} - {FINAL} ({N_BOOT} resamples, seed {SEED}):"]
    L += [f"  P1 {k:6s} diff={v[0]:+.4f}  CI=[{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in boot.items()]
    L += [f"  policy-f cost vs baseline_lr: {cost_ci['diff']:+.4f} [{cost_ci['lo']:+.4f}, {cost_ci['hi']:+.4f}]", "",
          f"Rule: PR-AUC CI excludes 0 -> {pr_ok}; policy-f cost CI excludes 0 (selection rule) -> {cost_ok}",
          f"DECISION: {decision}"]
    (OUT / "p1_report.txt").write_text("\n".join(L), encoding="utf-8")
    table.to_csv(OUT / "p1_metrics.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
