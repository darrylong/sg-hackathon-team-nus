"""Read-only probe of the priority model: top weights + counterfactual template edits on VAL.

python src/probe.py   -> outputs/probe/probe_report.txt, probe.csv
Uses models/baseline/ (same settings as baseline_lr, fit on all TRAIN) + baseline_lr calibration temperature.
combined_v1 copies its priority probabilities from baseline_lr, so it shares this model.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd

from calibration import apply_temperature
from data import ROOT, build_text, load_splits, load_tickets
from runs import RUNS_DIR, load_run_frame, proba_matrix

OUT = ROOT / "outputs" / "probe"
SEED, N_PER_CLASS = 42, 75

# Exact template sentences found in TRAIN (see report header for counts)
LOW_SCOPE = ["Just this one server.", "Single device, nothing else.", "Only one user is affected.",
             "It's only me as far as I can tell."]
HIGH_SCOPE = ["All customers on the platform see this.", "Every branch is affected.", "The whole site is affected."]
PROD = ["This box runs our production ERP.", "It hosts our core banking VMs.", "Production workloads run on it.",
        "It is part of the live payments cluster.", "It serves the live e-commerce site.",
        "Customer-facing services run on this system.", "this is prod", r"This is the primary system at [^.]+\."]
TAG_RE = re.compile(r"\[[^\]]*sev\w*\s*:\s*(\w+)\]", re.I)


def _pattern(variants):
    parts = [v if v.startswith("This is the primary") else re.escape(v).rstrip(r"\.") + r"\.?" for v in variants]
    return re.compile("|".join(f"(?:{p})" for p in parts), re.I)


def _match_case(src, repl):
    return repl.lower() if src == src.lower() else repl


def replace_first(variants, repl):
    pat = _pattern(variants)

    def edit(body):
        m = pat.search(body)
        return None if not m else body[:m.start()] + _match_case(m.group(0), repl) + body[m.end():]
    return edit


def add_if_absent(sentence, absent_re):
    def edit(body):
        return None if re.search(absent_re, body, re.I) else body.rstrip() + " " + _match_case(body, sentence)
    return edit


def tag_edit(kind):
    def edit(body):
        m = TAG_RE.search(body)
        level = m.group(1).lower() if m else None
        if kind == "low_to_critical" and level and level.startswith("lo"):
            return body[:m.start()] + "[Customer-selected severity: Critical]" + body[m.end():]
        if kind == "critical_to_low" and level and level.startswith("cri"):
            return body[:m.start()] + "[Customer-selected severity: Low]" + body[m.end():]
        if kind == "add_critical" and not m:
            return "[Customer-selected severity: Critical] " + body
        return None
    return edit


EDITS = {
    "E1_scope_up": replace_first(LOW_SCOPE, "All customers on the platform see this."),
    "E2_scope_down": replace_first(HIGH_SCOPE, "Only one user is affected."),
    "E3_env_down": replace_first(PROD, "It is in our test environment."),
    "E4_workaround": add_if_absent("There is a manual workaround but it is slow.", r"workaround|limping"),
    "E5_tone_only": add_if_absent("This is unacceptable.", r"unacceptable"),
    "E6a_tag_low_to_critical": tag_edit("low_to_critical"),
    "E6b_tag_critical_to_low": tag_edit("critical_to_low"),
    "E6c_tag_add_critical": tag_edit("add_critical"),
    "E7_urgency_claim": add_if_absent("Marking this urgent.", r"urgent"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_tickets()
    parts = load_splits(df)
    train, val = parts["train"], parts["val"]
    vec = joblib.load(ROOT / "models" / "baseline" / "tfidf.joblib")
    clf = joblib.load(ROOT / "models" / "baseline" / "priority_lr.joblib")
    T = json.loads((RUNS_DIR / "baseline_lr" / "calibration.json").read_text())["priority"]["T"]
    classes = list(clf.classes_)

    def predict(frame):
        return apply_temperature(clf.predict_proba(vec.transform(build_text(frame))), T)

    L = []
    # sanity: saved model == baseline_lr val probabilities; combined_v1 shares them
    _, p_run = proba_matrix(load_run_frame("baseline_lr", "val.csv"), "priority")
    raw_val = clf.predict_proba(vec.transform(build_text(val)))
    L.append(f"Sanity: max |saved model - baseline_lr val.csv| priority proba = {np.abs(raw_val - p_run).max():.2e}")
    if (RUNS_DIR / "combined_v1" / "val.csv").exists():
        _, p_comb = proba_matrix(load_run_frame("combined_v1", "val.csv"), "priority")
        L.append(f"combined_v1 priority proba identical to baseline_lr: {np.allclose(p_comb, p_run)} "
                 "(priority copied from baseline_lr -> same model, results below apply to both)")
    L.append(f"Calibration temperature (priority) T={T:.3f}")

    # 1. weights
    names = vec.get_feature_names_out()
    L.append("\n===== 1. Top 25 positive features =====")
    for c in ("P1", "P4"):
        w = clf.coef_[classes.index(c)]
        top = np.argsort(-w)[:25]
        L.append(f"\n{c}:\n" + "\n".join(f"  {w[i]:+.3f}  {names[i]}" for i in top))

    # template variants used (TRAIN counts)
    L.append("\n===== Template variants used (ticket counts in TRAIN, case-insensitive) =====")
    for label, variants in (("low scope", LOW_SCOPE), ("high scope", HIGH_SCOPE), ("production", PROD)):
        for v in variants:
            n = train["body"].str.contains(_pattern([v]), regex=True).sum()
            L.append(f"  {label:10s} {n:5d}  {v}")
    for s in ("There is a manual workaround but it is slow.", "This is unacceptable.", "Marking this urgent.",
              "It is in our test environment."):
        L.append(f"  {'added':10s} {train['body'].str.contains(re.escape(s.rstrip('.')), case=False).sum():5d}  {s}")
    sev = train["body"].map(lambda b: (m := TAG_RE.search(b)) and m.group(1).lower())
    L.append(f"  severity tags in TRAIN: {sev.value_counts().head(6).to_dict()}")

    # 2. counterfactual edits on a stratified VAL sample
    sample = (val.groupby("priority", group_keys=False)
                 .apply(lambda g: g.sample(N_PER_CLASS, random_state=SEED)).reset_index(drop=True))
    base = predict(sample)
    i1, i2 = classes.index("P1"), classes.index("P2")
    rows, detail = [], []
    for name, edit in EDITS.items():
        new_bodies = sample["body"].map(edit)
        mask = new_bodies.notna().to_numpy()
        if not mask.any():
            rows.append({"edit": name, "n_edited": 0})
            continue
        edited = sample[mask].copy()
        edited["body"] = new_bodies[mask]
        p_new, p_old = predict(edited), base[mask]
        r_old, r_new = p_old.argmax(1), p_new.argmax(1)  # class index: lower = more urgent
        rows.append({
            "edit": name, "n_edited": int(mask.sum()),
            "mean_dP1": (p_new[:, i1] - p_old[:, i1]).mean(),
            "mean_dP1P2": (p_new[:, [i1, i2]].sum(1) - p_old[:, [i1, i2]].sum(1)).mean(),
            "share_changed": (r_new != r_old).mean(),
            "share_up": (r_new < r_old).mean(), "share_down": (r_new > r_old).mean(),
        })
        for tid, tp, a, b, po, pn in zip(edited["ticket_id"], edited["priority"], r_old, r_new,
                                          p_old[:, i1], p_new[:, i1]):
            detail.append({"edit": name, "ticket_id": tid, "true_priority": tp, "pred_before": classes[a],
                           "pred_after": classes[b], "P1_before": po, "P1_after": pn})
    table = pd.DataFrame(rows)
    L.append(f"\n===== 2. Counterfactual edits (VAL sample n={len(sample)}, {N_PER_CLASS} per true priority; "
             "calibrated probabilities; up = more urgent) =====")
    L.append(table.round(4).to_string(index=False))

    (OUT / "probe_report.txt").write_text("\n".join(L), encoding="utf-8")
    table.to_csv(OUT / "probe.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "probe_detail.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
