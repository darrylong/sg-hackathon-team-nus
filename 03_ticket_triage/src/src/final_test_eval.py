"""One-time TEST evaluation of the locked final system (models/bundle_v3, P1 threshold 0.225, policy f,
guards, batch cap 0.15). Nothing is tuned on these numbers.

python src/final_test_eval.py -> outputs/final/test_report.txt, test_metrics.json
"""
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from data import ROOT, load_splits, load_tickets
from inference import TARGETS, Triage, escalation
from make_submission import BUNDLE, CAP, decide_capped
from routing import cap_count, realized_cost, top_k_mask

OUT = ROOT / "outputs" / "final"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = load_splits(load_tickets())["test"]
    tri = Triage.load(BUNDLE)
    scored = tri.score(test[["ticket_id", "channel", "subject", "body", "product"]])
    pred, notes = decide_capped(tri, scored, CAP)
    assert list(pred["ticket_id"]) == list(test["ticket_id"])

    m = {"split": "TEST", "n": len(test), "bundle": tri.manifest["run"], "p1_threshold": tri.threshold,
         "cap": CAP, "targets": {}}
    for t in TARGETS:
        m["targets"][t] = {"macro_f1": f1_score(test[t], pred[t], average="macro"),
                           "accuracy": accuracy_score(test[t], pred[t])}
    p, r, f, _ = precision_recall_fscore_support(test["priority"], pred["priority"], labels=["P1"], zero_division=0)
    m["P1"] = {"precision": p[0], "recall": r[0], "f1": f[0]}
    ep, er, ef, _ = precision_recall_fscore_support(escalation(test["priority"], test["sentiment"]),
                                                    escalation(pred["priority"], pred["sentiment"]),
                                                    average="binary", zero_division=0)
    m["escalation"] = {"precision": ep, "recall": er, "f1": ef}

    true_prio = test["priority"].to_numpy()
    correct = (pred["assigned_team"] == test["assigned_team"]).to_numpy()
    team_conf = pred[[c for c in pred.columns if c.startswith("proba_assigned_team_")]].to_numpy().max(1)
    k = cap_count(CAP, len(test))
    policies = {"no_abstain": np.zeros(len(test), bool),
                "b_naive_low_conf": top_k_mask(-team_conf, k),
                "f_expcost_pcorrect": pred["abstain"].to_numpy().astype(bool)}
    m["routing"] = {}
    for name, mask in policies.items():
        cost = realized_cost(correct, mask, true_prio, tri.W, tri.A)
        m["routing"][name] = {"mean_cost": float(cost.mean()), "abstain_rate": float(mask.mean()),
                              "n_abstain": int(mask.sum()),
                              "misroutes_routed": int((~mask & ~correct).sum())}
    m["guarded"] = int(pred["guarded"].sum())
    m["notes"] = notes

    L = [f"FINAL TEST EVALUATION (run once; nothing tuned on it) - bundle {m['bundle']}, n={m['n']}, "
         f"P1 threshold {m['p1_threshold']}, cap {CAP}", ""]
    for t in TARGETS:
        L.append(f"{t:14s} macro F1={m['targets'][t]['macro_f1']:.4f}  accuracy={m['targets'][t]['accuracy']:.4f}")
    L.append(f"P1 precision={p[0]:.4f} recall={r[0]:.4f} F1={f[0]:.4f}")
    L.append(f"Escalation precision={ep:.4f} recall={er:.4f} F1={ef:.4f}")
    L.append("")
    for name, v in m["routing"].items():
        L.append(f"{name:20s} mean cost={v['mean_cost']:.4f}  abstain={v['n_abstain']} ({v['abstain_rate']:.2%})  "
                 f"routed misroutes={v['misroutes_routed']}")
    L.append(f"guard-flagged tickets: {m['guarded']}; notes: {notes or 'none'}")
    (OUT / "test_report.txt").write_text("\n".join(L), encoding="utf-8")
    (OUT / "test_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
