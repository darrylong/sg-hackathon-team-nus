"""Round 2C: priority candidate ranking, P1 probability check, combined_v2 / E8b / combined_v3.

python src/round2c.py   (needs E1_char_tfidf, E5b_ordinal_char, E11_p1_weighted, E11b_p1_weighted_char, E9_tree_selected)
Every new run goes through run_experiment.run_one (calibration -> correctness -> evaluate --ref baseline_lr).
Writes outputs/round2c/round2c_report.txt and decisions.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import experiments as ex
import selection as sel
from data import ROOT, TARGETS
from run_experiment import run_one
from runs import RUNS_DIR, assemble_run, make_run

OUT = ROOT / "outputs" / "round2c"
CURRENT_PRIORITY = "E11_p1_weighted"
CANDIDATES = ["E1_char_tfidf", "E5b_ordinal_char", "E11_p1_weighted", "E11b_p1_weighted_char"]
UNWEIGHTED = ["baseline_lr", "E1_char_tfidf", "E5_ordinal_priority", "E5b_ordinal_char"]
EXCESS_PP = 0.02


def fmt(ci):
    return f"{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]"


def finish(name):
    sel.stamp_config(name)
    run_one(name)
    sel._cache.pop(name, None)


def p1_stats(run):
    from sklearn.metrics import average_precision_score, roc_auc_score
    r = sel.get(run)
    pr = r["targets"]["priority"]
    y = sel.val_frame()["priority"].to_numpy()
    p1 = pr["p_cal"][:, pr["classes"].index("P1")]
    p, rc, f, _ = precision_recall_fscore_support(y, pr["pred"], labels=["P1"], zero_division=0)
    return {"P1_AUROC": roc_auc_score(y == "P1", p1), "P1_PR_AUC": average_precision_score(y == "P1", p1),
            "P1_P": p[0], "P1_R": rc[0], "P1_F1": f[0],
            "prio_macro_F1": json.loads((RUNS_DIR / run / "metrics.json").read_text())["targets"]["priority"]["macro_f1"],
            "cost_f": sel.cost_f(run)}


def summary_metrics(run):
    m = json.loads((RUNS_DIR / run / "metrics.json").read_text())
    return {"run": run, **{f"{t}_F1": m["targets"][t]["macro_f1"] for t in TARGETS},
            "P1_P": m["P1"]["precision"], "P1_R": m["P1"]["recall"], "P1_F1": m["P1"]["f1"],
            "escalation_F1": m["escalation"]["f1"],
            "cost_b": m["routing"]["b_naive_low_conf"]["mean_cost"],
            "cost_f": m["routing"]["f_expcost_pcorrect"]["mean_cost"],
            "abstain_f": m["routing"]["f_expcost_pcorrect"]["abstain_rate"],
            "cost_f_vs_baseline": fmt(tuple(m["bootstrap_vs_ref"]["policy_f_cost"][k] for k in ("diff", "lo", "hi")))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel.write_rules_md()
    sel.stamp_all()
    L, decisions = [], {}
    base_cost = sel.cost_f(sel.BASELINE)
    L.append(f"baseline_lr policy-f cost = {base_cost:.4f}; rule (a) limit = {base_cost + sel.COST_SLACK:.4f}")

    # ---- step 2: priority candidates vs current choice
    rows, passing = [], []
    for run in CANDIDATES:
        st = p1_stats(run)
        if run == CURRENT_PRIORITY:
            rows.append({"run": run, **st, "verdict": "current choice (reference)"})
            continue
        ok, checks, c = sel.check_priority(run, CURRENT_PRIORITY)
        rows.append({"run": run, **st, "PR_AUC_vs_E11": fmt(c["P1_PR_AUC"]), "AUROC_vs_E11": fmt(c["P1_AUROC"]),
                     "prioF1_vs_E11": fmt(c["priority_macro_f1"]), "cost_f_vs_E11": fmt(c["policy_f_cost"]),
                     "verdict": ("PASS" if ok else "fail ") + " " + str({k: bool(v) for k, v in checks.items()})})
        if ok:
            passing.append((c["P1_PR_AUC"][0], run))
    chosen = max(passing)[1] if passing else CURRENT_PRIORITY
    decisions["priority_label_run"] = chosen
    cand = pd.DataFrame(rows)
    L.append("\n===== Step 2: priority candidates (VAL; CIs = run - E11_p1_weighted) =====")
    L.append(cand.round(4).to_string(index=False))
    L.append(f"Chosen priority run: {chosen}" + ("" if passing else " (no candidate passed all of a/b/c; keep current)"))

    # ---- step 3: P1 probability check
    r = sel.get(chosen)
    pr = r["targets"]["priority"]
    p1 = pr["p_cal"][:, pr["classes"].index("P1")]
    y1 = (sel.val_frame()["priority"].to_numpy() == "P1")
    excess = p1.mean() - y1.mean()
    b = np.minimum((p1 * 10).astype(int), 9)
    rel = pd.DataFrame([{"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "n": int((b == i).sum()),
                         "mean_P(P1)": p1[b == i].mean() if (b == i).any() else np.nan,
                         "observed_P1_rate": y1[b == i].mean() if (b == i).any() else np.nan} for i in range(10)])
    L.append(f"\n===== Step 3: P1 probability check for {chosen} =====")
    L.append(f"mean P(P1) = {p1.mean():.4f}; true P1 share = {y1.mean():.4f}; excess = {excess * 100:+.2f} pp")
    L.append(rel.round(4).to_string(index=False))
    routing_run = chosen
    if excess > EXCESS_PP:
        unw = {u: p1_stats(u)["P1_PR_AUC"] for u in UNWEIGHTED}
        routing_run = max(unw, key=unw.get)
        L.append(f"Excess > {EXCESS_PP * 100:.0f} pp -> routing probabilities from best UNWEIGHTED run by P1 PR-AUC: "
                 f"{routing_run} ({ {k: round(v, 4) for k, v in unw.items()} })")
        name = "P1check_routing_unweighted"
        sources = {t: chosen for t in TARGETS}
        sources["priority"] = routing_run
        assemble_run(name, sources, {"change": f"{chosen} with priority routing probabilities from {routing_run}; "
                                               f"priority label from {chosen}"}, label_sources={"priority": chosen})
        finish(name)
        L.append(f"policy-f cost, routing from {chosen} (weighted): {sel.cost_f(chosen):.4f}")
        L.append(f"policy-f cost, routing from {routing_run} (unweighted): {sel.cost_f(name):.4f}")
    else:
        L.append("Excess <= 2 pp -> routing uses the chosen run's own probabilities.")
    decisions["priority_routing_run"] = routing_run

    # ---- step 4: combined_v2
    ok_cat, ch_cat, c_cat = sel.check_f1("E1_char_tfidf", sel.BASELINE, "category")
    ok_team, ch_team, c_team = sel.check_team("E1_char_tfidf", sel.BASELINE)
    f1 = {run: json.loads((RUNS_DIR / run / "metrics.json").read_text())["targets"]["sentiment"]["macro_f1"]
          for run in ("E1_char_tfidf", "E9_tree_selected")}
    sources = {"category": "E1_char_tfidf" if ok_cat else sel.BASELINE,
               "assigned_team": "E1_char_tfidf" if ok_team else sel.BASELINE,
               "sentiment": max(f1, key=f1.get), "priority": routing_run}
    label_sources = {"priority": chosen} if routing_run != chosen else None
    decisions["combined_v2"] = {"sources": sources, "label_sources": label_sources}
    L.append("\n===== Step 4: combined_v2 =====")
    L.append(f"category: E1 vs baseline F1 {fmt(c_cat['category_macro_f1'])} -> {'E1' if ok_cat else 'baseline'}")
    L.append(f"team: E1 cost_f {sel.cost_f('E1_char_tfidf'):.4f} (limit {base_cost + sel.COST_SLACK:.4f}), "
             f"team F1 vs baseline {fmt(c_team['assigned_team_macro_f1'])} -> {'E1' if ok_team else 'baseline'}")
    L.append(f"sentiment: {f1} -> {sources['sentiment']}")
    L.append(f"priority: label {chosen}, routing probabilities {routing_run}")
    assemble_run("combined_v2", sources, {"change": f"round 2C combined: {sources}; priority label {chosen}"},
                 label_sources=label_sources)
    finish("combined_v2")

    # ---- E8b stacker on combined_v2
    cfg = {"change": "HGB team stacker on combined_v2", "source_run": "combined_v2"}
    if label_sources:
        cfg["label_sources"] = label_sources
    make_run("E8b_team_stacker", lambda t: None, cfg, base_run="combined_v2",
             custom={"assigned_team": ex.team_stacker_producer("combined_v2")})
    finish("E8b_team_stacker")
    ok8, ch8, c8 = sel.check_team("E8b_team_stacker", "combined_v2")
    L.append(f"\nE8b_team_stacker vs combined_v2: cost_f {sel.cost_f('E8b_team_stacker'):.4f}, team F1 "
             f"{fmt(c8['assigned_team_macro_f1'])}, cost_f diff {fmt(c8['policy_f_cost'])} -> "
             f"{'PASS' if ok8 else 'fail'} {ch8}")
    decisions["E8b_passes_team_rule"] = bool(ok8)
    finals = [sel.BASELINE, "combined_v1", "combined_v2"]
    if ok8:
        v3 = {**sources, "assigned_team": "E8b_team_stacker"}
        assemble_run("combined_v3", v3, {"change": f"combined_v2 with team from E8b_team_stacker"},
                     label_sources=label_sources)
        finish("combined_v3")
        decisions["combined_v3"] = {"sources": v3, "label_sources": label_sources}
        finals.append("combined_v3")
    else:
        L.append("combined_v3 not built (E8b fails the team rule).")
    finals.append("E8b_team_stacker")

    L.append("\n===== Final comparison vs baseline_lr (VAL) =====")
    L.append(pd.DataFrame([summary_metrics(r) for r in finals]).round(4).to_string(index=False))
    (OUT / "round2c_report.txt").write_text("\n".join(L), encoding="utf-8")
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
