"""Write submission.csv and submission_curveball.csv with the final bundle (combined_v3, models/bundle_v3).

python src/make_submission.py   -> outputs/final/submission.csv, submission_curveball.csv, submission_log.txt

- Priority label: P1 if calibrated P(P1) >= bundle p1_threshold (0.225), else argmax over P2..P4.
- Decision: inference.Triage.decide (policy f + guards), then a STRICT cap: never more than floor(cap*N)
  abstains. If guard-flagged tickets alone exceed the slots, abstain on the guarded tickets with the highest
  expected route cost and log the rest (they are routed).
- eval_tickets.csv uses cap 0.15. README: the 15% capacity is measured on submission.csv only and the
  curveball file is scored per ticket, so curveball uses per-ticket decisions (cap=None).
- Abstain rows leave category / assigned_team blank, as in artifacts/sample_submission*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from inference import SUBMISSION_COLS, Triage  # noqa: E402
from routing import cap_count  # noqa: E402

ROOT = SRC.parent
BUNDLE = ROOT / "models" / "bundle_v3"
OUT = ROOT / "outputs" / "final"
CAP = 0.15
JOBS = [("eval/eval_tickets.csv", "submission.csv", CAP),
        ("eval/curveball_tickets.csv", "submission_curveball.csv", None)]
VALID = {
    "category": {"Hardware", "Networking", "Storage", "Billing", "Access", "Firmware", "Software", "Security"},
    "priority": {"P1", "P2", "P3", "P4"},
    "assigned_team": {"Server-HW", "Network-Support", "Storage-Support", "Software-Support", "Account-Admin",
                      "Security-Ops", "Billing-Ops"},
    "sentiment": {"Angry", "Frustrated", "Neutral", "Positive"},
}


def decide_capped(tri: Triage, scored: pd.DataFrame, cap):
    """tri.decide, then enforce at most floor(cap*N) abstains (guarded tickets ranked by expected route cost)."""
    pred = tri.decide(scored, cap)
    log = list(pred.attrs.get("notes", []))
    if cap is None:
        return pred, log
    slots = cap_count(cap, len(pred))
    abstain = pred["abstain"].to_numpy().astype(bool)
    if abstain.sum() > slots:
        guarded = pred["guarded"].to_numpy()
        keep = np.zeros(len(pred), bool)
        order = np.argsort(-np.where(guarded, pred["expected_route_cost"].to_numpy(), -np.inf), kind="stable")
        keep[order[:slots]] = True
        keep &= guarded
        dropped = pred.loc[abstain & ~keep, "ticket_id"].tolist()
        pred.loc[abstain & ~keep, ["abstain", "decision"]] = [0, "route"]
        pred.loc[abstain & ~keep, "reason"] = "Route: guard-flagged but the hand-off desk cap is full"
        log.append(f"cap enforcement: {len(dropped)} guard-flagged tickets routed instead of abstained: {dropped}")
    assert pred["abstain"].sum() <= slots, "abstains exceed cap"
    return pred, log


def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    sub = pred[SUBMISSION_COLS].copy()
    sub.loc[sub["abstain"] == 1, ["category", "assigned_team"]] = ""
    sub["abstain"] = sub["abstain"].astype(int)
    return sub


def validate(sub: pd.DataFrame, ids: pd.Series, sample: Path, cap) -> list:
    """Raises AssertionError on any rule violation; returns a list of check descriptions."""
    want_cols = list(pd.read_csv(sample, nrows=0).columns)
    assert list(sub.columns) == want_cols, f"columns {list(sub.columns)} != sample {want_cols}"
    assert sub["ticket_id"].is_unique, "duplicate ticket_id"
    assert list(sub["ticket_id"]) == list(ids), "ticket_ids differ from the input file"
    assert set(sub["abstain"].unique()) <= {0, 1}, "abstain must be 0/1"
    for col in ("priority", "sentiment"):
        assert sub[col].isin(VALID[col]).all(), f"{col} invalid or empty"
    routed = sub["abstain"] == 0
    for col in ("category", "assigned_team"):
        assert sub.loc[routed, col].isin(VALID[col]).all(), f"{col} invalid/empty on routed rows"
        assert sub.loc[~routed, col].fillna("").isin(VALID[col] | {""}).all(), f"{col} invalid on abstain rows"
    if cap is not None:
        assert sub["abstain"].sum() <= cap_count(cap, len(sub)), "abstain rate over cap"
    return [f"columns == {sample.name}", f"{len(sub)} rows, ids match input 1:1", "labels valid",
            "priority/sentiment filled", "category/team filled on routed rows", "abstain in {0,1}",
            f"abstains {int(sub['abstain'].sum())} <= cap slots {cap_count(cap, len(sub)) if cap else 'n/a'}"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tri = Triage.load(BUNDLE)
    assert tri.threshold == 0.225, f"unexpected P1 threshold {tri.threshold}"
    log = [f"bundle {BUNDLE.name}: run {tri.manifest['run']}, P1 threshold {tri.threshold}, policy f, guards on"]
    for src, name, cap in JOBS:
        df = pd.read_csv(ROOT / src)
        scored = tri.score(df)
        pred, notes = decide_capped(tri, scored, cap)
        sub = to_submission(pred)
        checks = validate(sub, df["ticket_id"].astype(str),
                          ROOT / "artifacts" / ("sample_submission.csv" if cap else "sample_submission_curveball.csv"), cap)
        sub.to_csv(OUT / name, index=False, encoding="utf-8")
        guarded = pred[pred["guarded"]]
        log += [f"\n== {src} -> {name} (cap={cap}) ==", f"n={len(sub)} abstain={int(sub['abstain'].sum())} "
                f"({sub['abstain'].mean():.2%}); guarded={len(guarded)}", *notes,
                "validation: " + "; ".join(checks)]
        if cap is None:
            alt = int(decide_capped(tri, scored, CAP)[0]["abstain"].sum())
            log.append(f"(for reference: with cap {CAP} this file would have {alt} abstains)")
        for r in guarded.itertuples():
            log.append(f"  guarded {r.ticket_id}: {list(r.guard_flags)} -> abstain={r.abstain}")
        if cap is None:
            cols = ["ticket_id", "category", "priority", "assigned_team", "sentiment", "abstain", "p_correct",
                    "savings", "guard_flags"]
            log.append(pred[cols].round(3).to_string(index=False))
        log.append("label counts: " + "; ".join(f"{c}={sub[c].value_counts().to_dict()}" for c in
                                                  ("priority", "sentiment")))
    (OUT / "submission_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
