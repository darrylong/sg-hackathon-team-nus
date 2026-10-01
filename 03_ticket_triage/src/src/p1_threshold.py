"""P1 label threshold sweep for E5b_ordinal_char (label only; routing unchanged). Report only, no pick.

python src/p1_threshold.py -> outputs/round2c/p1_threshold/{threshold_curve.csv, threshold_curve.png, options.csv, report.txt}
Rule: P1 if P(P1) >= t, else argmax over P2..P4. Calibrated probabilities.
Escalation F1 uses the final system's sentiment (combined_v3 = E9_tree_selected) for every row.
"""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, fbeta_score, precision_recall_fscore_support  # noqa: E402

from data import ROOT, load_splits, load_tickets  # noqa: E402
from evaluate import analyse, escalation, routing_policies  # noqa: E402
from runs import load_run_frame, proba_matrix  # noqa: E402

RUN, REF_RUN, SYSTEM = "E5b_ordinal_char", "E11_p1_weighted", "combined_v3"
OUT = ROOT / "outputs" / "round2c" / "p1_threshold"
THRESHOLDS = np.round(np.arange(0.10, 0.50 + 1e-9, 0.025), 3)


def label(classes, p, t=None):
    if t is None:
        return np.array(classes)[p.argmax(1)]
    i1 = classes.index("P1")
    rest = p.copy()
    rest[:, i1] = -1
    pred = np.array(classes)[rest.argmax(1)]
    pred[p[:, i1] >= t] = "P1"
    return pred


def scores(y, pred, sent_true=None, sent_pred=None):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=["P1"], zero_division=0)
    out = {"P1_P": p[0], "P1_R": r[0], "P1_F1": f[0],
           "P1_F2": fbeta_score(y == "P1", pred == "P1", beta=2, zero_division=0),
           "prio_macro_F1": f1_score(y, pred, average="macro")}
    if sent_true is not None:
        out["escalation_F1"] = f1_score(escalation(y, sent_true), escalation(pred, sent_pred), zero_division=0)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = load_splits(load_tickets())
    train, val = parts["train"], parts["val"]
    cls, p_oof = proba_matrix(load_run_frame(RUN, "oof_calibrated.csv"), "priority")
    _, p_val = proba_matrix(load_run_frame(RUN, "val_calibrated.csv"), "priority")
    y_tr, y_val = train["priority"].to_numpy(), val["priority"].to_numpy()
    sys_res = analyse(SYSTEM, val)
    sent_pred = sys_res["targets"]["sentiment"]["pred"]
    sent_true = val["sentiment"].to_numpy()

    curve = []
    for t in THRESHOLDS:
        curve.append({"t": t, **{f"oof_{k}": v for k, v in scores(y_tr, label(cls, p_oof, t)).items()},
                      **{f"val_{k}": v for k, v in scores(y_val, label(cls, p_val, t), sent_true, sent_pred).items()}})
    curve = pd.DataFrame(curve)
    curve.to_csv(OUT / "threshold_curve.csv", index=False)

    oof_arg = scores(y_tr, label(cls, p_oof))
    t_b = float(curve.loc[curve["oof_P1_F2"].idxmax(), "t"])
    ok_c = curve[curve["oof_prio_macro_F1"] >= oof_arg["prio_macro_F1"] - 0.005]
    t_c = float(ok_c["t"].min()) if len(ok_c) else None

    rows = [{"option": "A argmax", "run": RUN, "t": None, **scores(y_val, label(cls, p_val), sent_true, sent_pred)},
            {"option": "B max OOF F2", "run": RUN, "t": t_b, **scores(y_val, label(cls, p_val, t_b), sent_true, sent_pred)},
            {"option": "C lowest t, OOF macroF1 >= argmax-0.005", "run": RUN, "t": t_c,
             **(scores(y_val, label(cls, p_val, t_c), sent_true, sent_pred) if t_c is not None else {})}]
    rcls, rp = proba_matrix(load_run_frame(REF_RUN, "val_calibrated.csv"), "priority")
    rows.append({"option": "ref E11 argmax", "run": REF_RUN, "t": None,
                 **scores(y_val, label(rcls, rp), sent_true, sent_pred)})
    options = pd.DataFrame(rows)
    options.to_csv(OUT / "options.csv", index=False)

    # Routing must not depend on the priority LABEL: recompute policies with each label and compare
    base = {k: v.mean() for k, v in sys_res["routing"]["per_ticket"].items()}
    for t in (t_b, t_c):
        if t is None:
            continue
        res = copy.copy(sys_res)
        res["targets"] = {**sys_res["targets"], "priority": {**sys_res["targets"]["priority"],
                                                             "pred": label(cls, p_val, t)}}
        new = {k: v.mean() for k, v in routing_policies(res, val)["per_ticket"].items()}
        assert all(np.isclose(base[k], new[k]) for k in ("b_naive_low_conf", "f_expcost_pcorrect")), (t, base, new)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ("oof", "val")):
        for k in ("P1_P", "P1_R", "P1_F1", "P1_F2", "prio_macro_F1"):
            ax.plot(curve["t"], curve[f"{split}_{k}"], marker=".", label=k)
        for t, name in ((t_b, "B"), (t_c, "C")):
            if t is not None:
                ax.axvline(t, ls="--", c="grey", lw=0.8)
                ax.text(t, 0.02, name, ha="center")
        ax.set(title=f"{RUN}: {split.upper()}", xlabel="P(P1) threshold t")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_curve.png", dpi=120)

    L = [f"P1 label threshold for {RUN} (calibrated; P1 if P(P1)>=t else argmax over P2..P4). Report only.",
         f"OOF argmax: {', '.join(f'{k}={v:.4f}' for k, v in oof_arg.items())}",
         f"B: t={t_b} (max OOF P1 F2);  C: t={t_c} (lowest t with OOF macro F1 >= {oof_arg['prio_macro_F1'] - 0.005:.4f})",
         f"Escalation F1 uses {SYSTEM} sentiment for every row.",
         f"Routing check: policy b={base['b_naive_low_conf']:.4f}, f={base['f_expcost_pcorrect']:.4f} unchanged "
         "under thresholds B and C (asserted).", "", "VAL options:", options.round(4).to_string(index=False), "",
         "Curve:", curve.round(4).to_string(index=False)]
    (OUT / "report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:6]) + "\n\n" + options.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
