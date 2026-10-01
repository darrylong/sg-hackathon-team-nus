"""Evaluate a run on VAL.  python src/evaluate.py --run <name> [--ref <name>]

Requires val.csv and val_calibrated.csv (run src/calibration.py first; done automatically if missing).
Writes report.txt, metrics.json, cost_curve.csv, cost_curve.png into the run folder and appends
a row to outputs/experiments.csv.
"""
import argparse
import json
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, log_loss, precision_recall_fscore_support)

from sklearn.metrics import roc_auc_score  # noqa: E402

from calibration import calibrate_run  # noqa: E402
from correctness import run_correctness  # noqa: E402
from data import ROOT, TARGETS, load_splits, load_tickets  # noqa: E402
from routing import (PRIORITIES, abstention_efficiency, cap_count, expected_costs,  # noqa: E402
                     load_costs, realized_cost, top_k_mask)
from runs import RUNS_DIR, load_run_frame, proba_matrix  # noqa: E402

EXPERIMENTS = ROOT / "outputs" / "experiments.csv"
N_BOOT, SEED = 1000, 42


# ---------------------------------------------------------------- metrics helpers
def ece(p, y_idx, bins=10):
    conf, correct = p.max(1), p.argmax(1) == y_idx
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[b == i].mean() - conf[b == i].mean()) * (b == i).mean()
                     for i in range(bins) if (b == i).any()))


def reliability_table(p, y_idx, bins=10):
    conf, correct = p.max(1), p.argmax(1) == y_idx
    b = np.minimum((conf * bins).astype(int), bins - 1)
    rows = [{"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": int((b == i).sum()),
             "mean_conf": conf[b == i].mean() if (b == i).any() else np.nan,
             "accuracy": correct[b == i].mean() if (b == i).any() else np.nan} for i in range(bins)]
    return pd.DataFrame(rows)


def brier(p, y_idx):
    onehot = np.eye(p.shape[1])[y_idx]
    return float(((p - onehot) ** 2).sum(1).mean())


def fast_macro_f1(y, yhat, k):
    cm = np.bincount(y * k + yhat, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    present = denom > 0
    return float(np.mean(2 * tp[present] / denom[present]))


def escalation(priority, sentiment):
    priority, sentiment = np.asarray(priority), np.asarray(sentiment)
    return (sentiment == "Angry") | ((sentiment == "Frustrated") & np.isin(priority, ["P1", "P2"]))


# ---------------------------------------------------------------- core analysis
def _target_probs(run: str, t: str, val: pd.DataFrame) -> dict:
    if not (RUNS_DIR / run / "val_calibrated.csv").exists():
        calibrate_run(run)
    raw, cal = load_run_frame(run, "val.csv"), load_run_frame(run, "val_calibrated.csv")
    assert list(raw["ticket_id"]) == list(val["ticket_id"]) == list(cal["ticket_id"])
    classes, p_raw = proba_matrix(raw, t)
    _, p_cal = proba_matrix(cal, t)
    y = val[t].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return dict(classes=classes, p_raw=p_raw, p_cal=p_cal, y=y, pred=np.array(classes)[p_cal.argmax(1)])


def analyse(run: str, val: pd.DataFrame) -> dict:
    """Label metrics use config['label_sources'][target] when present (e.g. priority label from one run);
    routing and the correctness model always use this run's own probabilities."""
    run_dir = RUNS_DIR / run
    res = {"targets": {t: _target_probs(run, t, val) for t in TARGETS}}
    res["routing_prio"] = res["targets"]["priority"]
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    for t, src in cfg.get("label_sources", {}).items():
        res["targets"][t] = _target_probs(src, t, val)
    if not (run_dir / "correctness.csv").exists():
        run_correctness(run)
    corr = load_run_frame(run, "correctness.csv")
    assert list(corr["ticket_id"]) == list(val["ticket_id"])
    res["p_correct"] = corr["p_correct"].to_numpy()
    res["routing"] = routing_policies(res, val)
    return res


def routing_policies(res, val):
    W, A, cap = load_costs()
    team, prio = res["targets"]["assigned_team"], res["routing_prio"]
    true_prio = val["priority"].to_numpy()
    correct = team["pred"] == val["assigned_team"].to_numpy()
    n, k = len(val), cap_count(cap, len(val))
    ec_raw = expected_costs(team["p_raw"], prio["p_raw"], prio["classes"], W, A)["savings"].to_numpy()
    ec_cal = expected_costs(team["p_cal"], prio["p_cal"], prio["classes"], W, A)["savings"].to_numpy()
    naive_score = -team["p_cal"].max(1)
    oracle_score = np.where(correct, -np.inf, np.array([W[p] for p in true_prio]))
    p_correct = res["p_correct"]
    ec_corr = expected_costs(team["p_cal"], prio["p_cal"], prio["classes"], W, A, p_ok=p_correct)["savings"].to_numpy()

    masks = {
        "a_no_abstain": np.zeros(n, bool),
        "b_naive_low_conf": top_k_mask(naive_score, k),
        "c_expcost_raw": top_k_mask(ec_raw, k, positive_only=True),
        "d_expcost_cal": top_k_mask(ec_cal, k, positive_only=True),
        "e_oracle_floor": top_k_mask(oracle_score, k) & ~correct,
        "f_expcost_pcorrect": top_k_mask(ec_corr, k, positive_only=True),
        "g_naive_low_pcorrect": top_k_mask(-p_correct, k),
    }
    table, per_ticket = [], {}
    efficiency = {name: abstention_efficiency(m, correct, true_prio, W, A) for name, m in masks.items()}
    for name, m in masks.items():
        cost = realized_cost(correct, m, true_prio, W, A)
        per_ticket[name] = cost
        row = {"policy": name, "mean_cost": cost.mean(), "abstain_rate": m.mean(), "n_abstain": int(m.sum())}
        for p in PRIORITIES:
            row[f"abst_{p}"] = int((m & (true_prio == p)).sum())
        for p in PRIORITIES:
            row[f"misr_{p}"] = int((~m & ~correct & (true_prio == p)).sum())
        table.append(row)

    curve = []
    for r in np.round(np.arange(0, cap + 1e-9, 0.005), 3):
        kk = cap_count(r, n)
        row = {"abstain_rate": r, "n_abstain": kk}
        for name, score in (("b_naive_low_conf", naive_score), ("c_expcost_raw", ec_raw), ("d_expcost_cal", ec_cal)):
            row[name] = realized_cost(correct, top_k_mask(score, kk), true_prio, W, A).mean()
        curve.append(row)
    return {"table": pd.DataFrame(table), "per_ticket": per_ticket, "curve": pd.DataFrame(curve),
            "efficiency": efficiency, "correct": correct, "cap": cap, "W": W, "A": A}


def policy_bootstrap(per_ticket: dict, base="b_naive_low_conf") -> pd.DataFrame:
    """Paired bootstrap of mean-cost difference (policy - base); decisions fixed on the full VAL set."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_ticket[base]), size=(N_BOOT, len(per_ticket[base])))
    rows = []
    for name, cost in per_ticket.items():
        if name in (base, "a_no_abstain", "c_expcost_raw", "e_oracle_floor"):
            continue
        d = cost - per_ticket[base]
        boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"policy": name, "vs": base, "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "p_worse_or_equal": (boots >= 0).mean()})
    return pd.DataFrame(rows)


def conditional_reliability(res, val):
    team, prio = res["targets"]["assigned_team"], res["targets"]["priority"]
    conf = team["p_cal"].max(1)
    correct = res["routing"]["correct"]
    frame = pd.DataFrame({"true_priority": val["priority"].to_numpy(), "pred_priority": prio["pred"],
                          "conf": conf, "correct": correct})
    out = {}
    for key in ("true_priority", "pred_priority"):
        g = frame.groupby(key).agg(n=("correct", "size"), team_acc=("correct", "mean"), mean_conf=("conf", "mean"))
        g["acc_minus_conf"] = g["team_acc"] - g["mean_conf"]
        out[key] = g
    for p in ("P1", "P2"):
        sel = (frame["pred_priority"] == p).to_numpy()
        out[f"bins_pred_{p}"] = reliability_table(team["p_cal"][sel], team["y"][sel], bins=5)
    return out


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref")
    args = ap.parse_args()

    df = load_tickets()
    val = load_splits(df)["val"]
    res = analyse(args.run, val)
    run_dir = RUNS_DIR / args.run
    L, metrics = [], {"run": args.run, "targets": {}}
    L.append(f"RUN {args.run}  (VAL n={len(val)}; calibrated probabilities unless stated)")
    T = json.loads((run_dir / "calibration.json").read_text())

    for t in TARGETS:
        r = res["targets"][t]
        y_true = val[t].to_numpy()
        labels = list(range(len(r["classes"])))
        m = dict(accuracy=accuracy_score(y_true, r["pred"]),
                 macro_f1=f1_score(y_true, r["pred"], average="macro", zero_division=0),
                 log_loss_raw=log_loss(r["y"], r["p_raw"], labels=labels),
                 log_loss_cal=log_loss(r["y"], r["p_cal"], labels=labels),
                 brier_raw=brier(r["p_raw"], r["y"]), brier_cal=brier(r["p_cal"], r["y"]),
                 ece_raw=ece(r["p_raw"], r["y"]), ece_cal=ece(r["p_cal"], r["y"]), T=T[t]["T"])
        metrics["targets"][t] = m
        L.append(f"\n===== {t} =====")
        L.append("  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        direction = ("raw model UNDERconfident; T<1 sharpens probabilities" if m["T"] < 1
                     else "raw model OVERconfident; T>1 softens probabilities")
        L.append(f"temperature T={m['T']:.3f}: {direction}")
        L.append(classification_report(y_true, r["pred"], digits=4, zero_division=0))
        cm = confusion_matrix(y_true, r["pred"], labels=r["classes"])
        L.append("confusion (rows=true, cols=pred):\n" + pd.DataFrame(cm, r["classes"], r["classes"]).to_string())

    prio, sent = res["targets"]["priority"], res["targets"]["sentiment"]
    p, rcl, f, _ = precision_recall_fscore_support(val["priority"], prio["pred"], labels=["P1"], zero_division=0)
    metrics["P1"] = dict(precision=p[0], recall=rcl[0], f1=f[0])
    ep, er, ef, _ = precision_recall_fscore_support(escalation(val["priority"], val["sentiment"]),
                                                    escalation(prio["pred"], sent["pred"]),
                                                    average="binary", zero_division=0)
    metrics["escalation"] = dict(precision=ep, recall=er, f1=ef)
    L.append(f"\nP1 precision={p[0]:.4f} recall={rcl[0]:.4f} F1={f[0]:.4f}")
    L.append(f"Escalation risk precision={ep:.4f} recall={er:.4f} F1={ef:.4f}")

    for t in ("assigned_team", "priority"):
        r = res["targets"][t]
        for kind in ("p_raw", "p_cal"):
            L.append(f"\nReliability {t} ({kind}):\n" + reliability_table(r[kind], r["y"]).round(4).to_string(index=False))

    rt = res["routing"]
    L.append(f"\n===== Routing policies (cap={rt['cap']}, max abstains={cap_count(rt['cap'], len(val))}) =====")
    L.append(rt["table"].round(4).to_string(index=False))
    metrics["routing"] = rt["table"].set_index("policy").to_dict(orient="index")
    rt["curve"].to_csv(run_dir / "cost_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in ("b_naive_low_conf", "c_expcost_raw", "d_expcost_cal"):
        ax.plot(rt["curve"]["abstain_rate"] * 100, rt["curve"][c], marker=".", label=c)
    ax.axhline(rt["table"].set_index("policy").at["e_oracle_floor", "mean_cost"], ls="--", c="grey", label="e_oracle_floor (at cap)")
    ax.set(xlabel="abstain rate (%)", ylabel="mean routing cost", title=f"{args.run}: cost vs abstain rate (VAL)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(run_dir / "cost_curve.png", dpi=120)
    L.append("\nCost curve (forced top-k by each score; see cost_curve.csv/png):")
    L.append(rt["curve"].round(4).to_string(index=False))

    L.append("\n===== Abstention efficiency by TRUE priority "
             "(caught = misroutes turned into abstains; net_gain = caught*W - abstains*A) =====")
    for name, eff in rt["efficiency"].items():
        L.append(f"\n{name}:\n" + eff.round(4).to_string(index=False))
    metrics["efficiency"] = {k: v.to_dict(orient="records") for k, v in rt["efficiency"].items()}

    pb = policy_bootstrap(rt["per_ticket"])
    L.append(f"\n===== Policy bootstrap ({N_BOOT} paired resamples, seed {SEED}; decisions fixed on full VAL) =====")
    L.append(pb.round(4).to_string(index=False))
    metrics["policy_bootstrap"] = pb.to_dict(orient="records")

    is_err = ~rt["correct"]
    auroc = {"p_ok": roc_auc_score(is_err, -res["targets"]["assigned_team"]["p_cal"].max(1)),
             "p_correct": roc_auc_score(is_err, -res["p_correct"])}
    L.append(f"\nAUROC for detecting team errors on VAL: p_ok={auroc['p_ok']:.4f}  p_correct={auroc['p_correct']:.4f}")
    metrics["team_error_auroc"] = auroc

    cr = conditional_reliability(res, val)
    L.append("\n===== Conditional reliability of calibrated team confidence =====")
    for key, tbl_ in cr.items():
        L.append(f"\n{key}:\n" + tbl_.round(4).to_string())
    metrics["conditional_reliability"] = {k: v.reset_index().to_dict(orient="records") for k, v in cr.items()}

    if args.ref:
        ref = analyse(args.ref, val)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(val), size=(N_BOOT, len(val)))
        boot = {}
        for pol, key in (("d_expcost_cal", "policy_d_cost"), ("f_expcost_pcorrect", "policy_f_cost"),
                         ("b_naive_low_conf", "policy_b_cost")):
            d_run, d_ref = rt["per_ticket"][pol], ref["routing"]["per_ticket"][pol]
            diffs = d_run[idx].mean(1) - d_ref[idx].mean(1)
            boot[key] = (d_run.mean() - d_ref.mean(), *np.percentile(diffs, [2.5, 97.5]))
        for t in TARGETS:
            a, b = res["targets"][t], ref["targets"][t]
            k = len(a["classes"])
            ya, pred_a, pred_b = a["y"], a["p_cal"].argmax(1), b["p_cal"].argmax(1)
            diffs = [fast_macro_f1(ya[i], pred_a[i], k) - fast_macro_f1(ya[i], pred_b[i], k) for i in idx]
            boot[f"{t}_macro_f1"] = (fast_macro_f1(ya, pred_a, k) - fast_macro_f1(ya, pred_b, k),
                                     *np.percentile(diffs, [2.5, 97.5]))
        L.append(f"\n===== Paired bootstrap vs {args.ref} ({N_BOOT} resamples; run - ref, 95% CI) =====")
        for name, (d, lo, hi) in boot.items():
            L.append(f"{name:24s} diff={d:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]")
        metrics["bootstrap_vs_ref"] = {"ref": args.ref, **{k: dict(diff=v[0], lo=v[1], hi=v[2]) for k, v in boot.items()}}

    (run_dir / "report.txt").write_text("\n".join(L), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    tbl = rt["table"].set_index("policy")
    row = {"run": args.run, "ref": args.ref or "", "timestamp": datetime.now().isoformat(timespec="seconds"),
           **{f"{t}_f1": round(metrics["targets"][t]["macro_f1"], 4) for t in TARGETS},
           "p1_recall": round(metrics["P1"]["recall"], 4), "esc_f1": round(ef, 4),
           "ece_team_cal": round(metrics["targets"]["assigned_team"]["ece_cal"], 4),
           "cost_a": round(tbl.at["a_no_abstain", "mean_cost"], 4),
           "cost_d": round(tbl.at["d_expcost_cal", "mean_cost"], 4),
           "abstain_d": round(tbl.at["d_expcost_cal", "abstain_rate"], 4),
           "cost_e_floor": round(tbl.at["e_oracle_floor", "mean_cost"], 4),
           "cost_b": round(tbl.at["b_naive_low_conf", "mean_cost"], 4),
           "cost_f": round(tbl.at["f_expcost_pcorrect", "mean_cost"], 4),
           "cost_g": round(tbl.at["g_naive_low_pcorrect", "mean_cost"], 4)}
    # Union of columns so older rows (fewer columns) are kept intact; lock for parallel runs
    lock = EXPERIMENTS.with_suffix(".lock")
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        raise TimeoutError(f"could not acquire {lock}")
    try:
        exp = pd.concat([pd.read_csv(EXPERIMENTS), pd.DataFrame([row])], ignore_index=True) \
            if EXPERIMENTS.exists() else pd.DataFrame([row])
        exp.to_csv(EXPERIMENTS, index=False)
    finally:
        os.close(fd)
        os.remove(lock)

    # terminal summary
    print(f"RUN {args.run}")
    for t in TARGETS:
        m = metrics["targets"][t]
        print(f"  {t:14s} F1={m['macro_f1']:.4f}  T={m['T']:.3f}  ECE raw={m['ece_raw']:.4f} cal={m['ece_cal']:.4f}  "
              f"logloss {m['log_loss_raw']:.4f}->{m['log_loss_cal']:.4f}")
    print(f"  P1 P/R/F1={p[0]:.4f}/{rcl[0]:.4f}/{f[0]:.4f}  escalation F1={ef:.4f}")
    print(rt["table"].round(4).to_string(index=False))
    for name in ("b_naive_low_conf", "d_expcost_cal", "f_expcost_pcorrect"):
        print(f"\nefficiency {name}:\n" + rt["efficiency"][name].round(4).to_string(index=False))
    print("\n" + pb.round(4).to_string(index=False))
    print(f"\nteam-error AUROC: p_ok={auroc['p_ok']:.4f} p_correct={auroc['p_correct']:.4f}")
    for key, tbl_ in cr.items():
        print(f"\n{key}:\n" + tbl_.round(4).to_string())
    if args.ref:
        for name, (d, lo, hi) in boot.items():
            print(f"  vs {args.ref}: {name} diff={d:+.4f} CI=[{lo:+.4f}, {hi:+.4f}]")
    print(f"Full report: {run_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
