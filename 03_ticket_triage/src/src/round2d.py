"""Round 2D: factor coverage, interaction check, priority stackers (E12 / E12b), selection vs E5b,
P1 label threshold and combined_v4.  python src/round2d.py  -> outputs/round2d/

Existing src modules are imported read-only. New runs: E12_priority_stacker, E12b_priority_stacker_ctx,
combined_v4 (never overwritten if they already exist).
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures

import experiments as ex
import factors as F
import selection as sel
from data import ROOT, TARGETS, load_splits, load_tickets
from evaluate import fast_macro_f1
from p1_threshold import THRESHOLDS, label, scores
from round2c import fmt, p1_stats
from run_experiment import run_one
from runs import RUNS_DIR, assemble_run, get_folds, load_run_frame, make_run, proba_matrix

OUT = ROOT / "outputs" / "round2d"
PRIO_SRC, CTX_SRC, CURRENT = "E5b_ordinal_char", "combined_v3", "E5b_ordinal_char"
CURRENT_T = 0.225  # E5b option C from round 2C
LEAK_NOTE = ("Stacker trained on OOF probabilities that were produced with the same folds and calibrated with a "
             "temperature fit on all TRAIN OOF rows: a small, accepted leak.")
SEED = 42


def tags_for(df):
    path = OUT / "factor_tags_train_val.csv"
    if path.exists():
        cached = pd.read_csv(path).set_index("ticket_id")
        if set(df["ticket_id"]) <= set(cached.index):
            return cached.loc[df["ticket_id"], F.FACTORS].reset_index(drop=True)
    parts = load_splits(load_tickets())
    both = pd.concat([parts["train"], parts["val"]])
    tags = F.tag_frame(both["body"])
    tags.insert(0, "ticket_id", both["ticket_id"].to_numpy())
    tags.to_csv(path, index=False)
    return tags.set_index("ticket_id").loc[df["ticket_id"], F.FACTORS].reset_index(drop=True)


# ---------------------------------------------------------------- 1. coverage
def coverage(train, L):
    tags = tags_for(train)
    L.append("===== 1. Factor coverage on TRAIN (share of tickets per level) =====")
    for f in F.FACTORS:
        vc = tags[f].value_counts(normalize=True).reindex(F.LEVELS[f]).fillna(0)
        L.append(f"{f:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in vc.items()))
    L.append("\n10 random TRAIN tickets (seed 42):")
    s = train.sample(10, random_state=SEED)
    st = tags_for(s)
    for (_, r), (_, t) in zip(s.iterrows(), st.iterrows()):
        L.append(f"[{r.ticket_id} {r.priority}] {dict(t)}\n    {r.body[:260]}")
    return tags


# ---------------------------------------------------------------- 2. interactions
def interactions(train, tags, L):
    d = tags.assign(P1=(train["priority"].to_numpy() == "P1"),
                    P12=np.isin(train["priority"].to_numpy(), ["P1", "P2"]))
    L.append("\n===== 2. Interaction check (TRAIN) =====")
    t2 = d.groupby(["scope", "environment"]).agg(n=("P1", "size"), P1_rate=("P1", "mean"), P1orP2=("P12", "mean"))
    L.append("scope x environment:\n" + t2.round(3).to_string())
    t3 = d.groupby(["scope", "environment", "workaround"]).agg(n=("P1", "size"), P1_rate=("P1", "mean"),
                                                               P1orP2=("P12", "mean"))
    L.append("\nscope x environment x workaround (n >= 30):\n" + t3[t3["n"] >= 30].round(3).to_string())

    X = F.one_hot(tags)
    Xi = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(X)
    y = d["P1"].to_numpy().astype(int)
    cv = PredefinedSplit(get_folds(train).to_numpy())
    res = {}
    for name, M in (("main", X), ("main+pairwise", Xi)):
        p = cross_val_predict(LogisticRegression(max_iter=2000), M, y, cv=cv, method="predict_proba")[:, 1]
        res[name] = (log_loss(y, p), roc_auc_score(y, p), M.shape[1])
    gain_ll = res["main"][0] - res["main+pairwise"][0]
    gain_auc = res["main+pairwise"][1] - res["main"][1]
    for k, (ll, auc, nf) in res.items():
        L.append(f"LR is_P1 {k:14s} features={nf:4d}  CV log-loss={ll:.4f}  AUROC={auc:.4f}")
    verdict = "YES" if (gain_ll >= 0.005 or gain_auc >= 0.005) else "NO"
    L.append(f"Interaction gain: log-loss {gain_ll:+.4f}, AUROC {gain_auc:+.4f} -> combinations add clear signal: {verdict}")
    return {"cv": res, "gain_logloss": gain_ll, "gain_auroc": gain_auc, "clear_signal": verdict}


# ---------------------------------------------------------------- 3. stackers
def stack_X(prio_frame, ctx_frame, tickets, with_ctx):
    _, pp = proba_matrix(prio_frame, "priority")
    parts = [pp, F.one_hot(tags_for(tickets)),
             np.eye(len(ex.PRODUCTS))[tickets["product"].map(ex.PRODUCTS.index).to_numpy()],
             np.eye(len(ex.CHANNELS))[tickets["channel"].map(ex.CHANNELS.index).to_numpy()]]
    if with_ctx:
        parts += [proba_matrix(ctx_frame, "category")[1], proba_matrix(ctx_frame, "sentiment")[1]]
    return np.hstack(parts)


def hgb():
    return HistGradientBoostingClassifier(random_state=SEED)


def stacker_producer(with_ctx):
    def produce(train, val, folds, classes):
        po, pv = load_run_frame(PRIO_SRC, "oof_calibrated.csv"), load_run_frame(PRIO_SRC, "val_calibrated.csv")
        co, cv_ = load_run_frame(CTX_SRC, "oof_calibrated.csv"), load_run_frame(CTX_SRC, "val_calibrated.csv")
        for a, b in ((po, train), (co, train), (pv, val), (cv_, val)):
            assert list(a["ticket_id"]) == list(b["ticket_id"])
        X_tr, X_val = stack_X(po, co, train, with_ctx), stack_X(pv, cv_, val, with_ctx)
        y = train["priority"].to_numpy()
        p_oof = cross_val_predict(hgb(), X_tr, y, cv=PredefinedSplit(folds), method="predict_proba")
        model = hgb().fit(X_tr, y)
        assert list(model.classes_) == list(classes)
        return p_oof, model.predict_proba(X_val)
    return produce


def build_stacker(name, with_ctx):
    if (RUNS_DIR / name / "val.csv").exists():
        print(f"{name} exists -> reusing (not overwritten)")
    else:
        make_run(name, lambda t: None,
                 {"change": f"HGB priority stacker on {PRIO_SRC} OOF calibrated priority probs + factor/product/"
                            f"channel one-hots" + (f" + {CTX_SRC} category & sentiment probs" if with_ctx else ""),
                  "priority_source": PRIO_SRC, "context_source": CTX_SRC if with_ctx else None,
                  "other_targets_from": CTX_SRC, "leak_note": LEAK_NOTE},
                 base_run=CTX_SRC, custom={"priority": stacker_producer(with_ctx)})
        sel.stamp_config(name)
    run_one(name)
    sel._cache.pop(name, None)


# ---------------------------------------------------------------- 5. threshold (option C)
def option_c(run, val, sent_pred):
    cls, p_oof = proba_matrix(load_run_frame(run, "oof_calibrated.csv"), "priority")
    _, p_val = proba_matrix(load_run_frame(run, "val_calibrated.csv"), "priority")
    y_tr = load_splits(load_tickets())["train"]["priority"].to_numpy()
    arg = scores(y_tr, label(cls, p_oof))["prio_macro_F1"]
    ok = [t for t in THRESHOLDS if scores(y_tr, label(cls, p_oof, t))["prio_macro_F1"] >= arg - 0.005]
    t = float(min(ok)) if ok else None
    y_val, s_true = val["priority"].to_numpy(), val["sentiment"].to_numpy()
    return t, scores(y_val, label(cls, p_val, t), s_true, sent_pred), cls, p_val


# ---------------------------------------------------------------- 6. extra bootstrap for combined_v4 vs v3
def boot_extra(run, ref, t_run, t_ref, val):
    a, b = sel.get(run), sel.get(ref)
    y = val["priority"].to_numpy()
    pa = label(a["targets"]["priority"]["classes"], a["targets"]["priority"]["p_cal"], t_run)
    pb = label(b["targets"]["priority"]["classes"], b["targets"]["priority"]["p_cal"], t_ref)
    idx = np.random.default_rng(SEED).integers(0, len(y), size=(sel.N_BOOT, len(y)))
    err_a, err_b = ~a["routing"]["correct"], ~b["routing"]["correct"]
    sa, sb = -a["p_correct"], -b["p_correct"]
    cb_a, cb_b = a["routing"]["per_ticket"]["b_naive_low_conf"], b["routing"]["per_ticket"]["b_naive_low_conf"]

    def p1f1(yy, pp):
        return precision_recall_fscore_support(yy, pp, labels=["P1"], zero_division=0)[2][0]

    out = {}
    for name, fn in (("P1_F1_at_own_t", lambda i: p1f1(y[i], pa[i]) - p1f1(y[i], pb[i])),
                     ("policy_b_cost", lambda i: cb_a[i].mean() - cb_b[i].mean()),
                     ("team_error_AUROC_p_correct", lambda i: roc_auc_score(err_a[i], sa[i]) - roc_auc_score(err_b[i], sb[i]))):
        full = fn(np.arange(len(y)))
        d = np.array([fn(i) for i in idx])
        out[name] = (float(full), *map(float, np.percentile(d, [2.5, 97.5])))
    pr = {k: precision_recall_fscore_support(y, p, labels=["P1"], zero_division=0) for k, p in (("run", pa), ("ref", pb))}
    return out, {k: (v[0][0], v[1][0], v[2][0]) for k, v in pr.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = load_splits(load_tickets())
    train, val = parts["train"], parts["val"]
    L, summary = [], {}

    tags = coverage(train, L)
    summary["interactions"] = interactions(train, tags, L)
    print("\n".join(L[-4:]), flush=True)

    for name, ctx in (("E12_priority_stacker", False), ("E12b_priority_stacker_ctx", True)):
        build_stacker(name, ctx)
    L.append(f"\n===== 3. Priority stackers =====\n{LEAK_NOTE}")

    # ---- 4. selection vs E5b
    rows, passing = [], []
    for run in (CURRENT, "E12_priority_stacker", "E12b_priority_stacker_ctx"):
        st = p1_stats(run)
        row = {"run": run, **st}
        if run != CURRENT:
            ok, checks, c = sel.check_priority(run, CURRENT)
            c3 = sel.compare(run, CTX_SRC)
            row.update({"PR_AUC_vs_E5b": fmt(c["P1_PR_AUC"]), "AUROC_vs_E5b": fmt(c["P1_AUROC"]),
                        "prioF1_vs_E5b": fmt(c["priority_macro_f1"]), "cost_f_vs_E5b": fmt(c["policy_f_cost"]),
                        "cost_f_vs_combined_v3": fmt(c3["policy_f_cost"]),
                        "verdict": ("PASS " if ok else "fail ") + str({k: bool(v) for k, v in checks.items()})})
            if ok:
                passing.append((st["P1_PR_AUC"], run))
        else:
            row["verdict"] = "current choice"
        rows.append(row)
    table = pd.DataFrame(rows)
    winner = max(passing)[1] if passing else None
    summary["winner"] = winner
    L.append(f"\n===== 4. Selection vs {CURRENT} (rule a/b/c; a limit = {sel.cost_f(sel.BASELINE) + sel.COST_SLACK:.4f}) =====")
    L.append(table.round(4).to_string(index=False))
    L.append("Note: E12/E12b copy category/team/sentiment from combined_v3 (E5b run has baseline team), so their "
             "absolute policy-f cost includes the E8b team gain; cost_f_vs_combined_v3 isolates the priority change.")
    L.append(f"Winner: {winner or 'none -> keep ' + CURRENT}")
    print("\n".join(L[-4:]), flush=True)

    if winner:
        sent_pred = sel.get(CTX_SRC)["targets"]["sentiment"]["pred"]
        t_w, s_w, _, _ = option_c(winner, val, sent_pred)
        _, s_cur, _, _ = option_c(CURRENT, val, sent_pred)
        cls, p_cur = proba_matrix(load_run_frame(CURRENT, "val_calibrated.csv"), "priority")
        s_cur = scores(val["priority"].to_numpy(), label(cls, p_cur, CURRENT_T), val["sentiment"].to_numpy(), sent_pred)
        (RUNS_DIR / winner / "priority_threshold.json").write_text(json.dumps(
            {"rule": "P1 if P(P1) >= t else argmax over P2..P4 (calibrated)", "t": t_w,
             "selection": "option C: lowest t in 0.10..0.50 step 0.025 with TRAIN OOF macro F1 >= argmax - 0.005"}, indent=2))
        thr = pd.DataFrame([{"run": winner, "t": t_w, **s_w}, {"run": CURRENT, "t": CURRENT_T, **s_cur}])
        L.append("\n===== 5. P1 label threshold (option C; escalation uses combined_v3 sentiment) =====")
        L.append(thr.round(4).to_string(index=False))
        summary["threshold"] = t_w

        d = json.loads((ROOT / "outputs" / "round2c" / "decisions.json").read_text())["combined_v3"]
        if (RUNS_DIR / "combined_v4" / "val.csv").exists():
            print("combined_v4 exists -> reusing (not overwritten)")
        else:
            assemble_run("combined_v4", {**d["sources"], "priority": winner},
                         {"change": f"combined_v3 with priority from {winner}", "priority_threshold": t_w})
            sel.stamp_config("combined_v4")
        run_one("combined_v4")
        sel._cache.pop("combined_v4", None)
        c = sel.compare("combined_v4", CTX_SRC)
        extra, p1 = boot_extra("combined_v4", CTX_SRC, t_w, CURRENT_T, val)
        m4, m3 = (json.loads((RUNS_DIR / r / "metrics.json").read_text()) for r in ("combined_v4", CTX_SRC))
        L.append("\n===== 6. combined_v4 vs combined_v3 (CIs = v4 - v3) =====")
        for t in TARGETS:
            L.append(f"{t:14s} macro F1 v4={m4['targets'][t]['macro_f1']:.4f} v3={m3['targets'][t]['macro_f1']:.4f}  "
                     f"diff {fmt(c[f'{t}_macro_f1'])}")
        L.append(f"P1 P/R/F1 at own t: v4 (t={t_w}) {'/'.join(f'{x:.3f}' for x in p1['run'])}  "
                 f"v3 (t={CURRENT_T}) {'/'.join(f'{x:.3f}' for x in p1['ref'])}  F1 diff {fmt(extra['P1_F1_at_own_t'])}")
        for pol, key in (("b_naive_low_conf", "policy_b_cost"), ("f_expcost_pcorrect", "policy_f_cost")):
            ci = extra[key] if key in extra else c[key]
            L.append(f"{pol:20s} v4={m4['routing'][pol]['mean_cost']:.4f} v3={m3['routing'][pol]['mean_cost']:.4f}  "
                     f"diff {fmt(ci)}  abstain v4={m4['routing'][pol]['abstain_rate']:.4f} v3={m3['routing'][pol]['abstain_rate']:.4f}")
        L.append(f"team-error AUROC p_correct v4={m4['team_error_auroc']['p_correct']:.4f} "
                 f"v3={m3['team_error_auroc']['p_correct']:.4f} diff {fmt(extra['team_error_AUROC_p_correct'])}")
        print("\n".join(L[-8:]), flush=True)

    (OUT / "round2d_report.txt").write_text("\n".join(L), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
