"""Expected-cost abstention policy. Works on any batch size (reused by the MVP).

Costs and capacity come from artifacts/. For a single ticket (batch of 1) floor(cap*N)=0,
so pass cap=None to abstain whenever expected savings > 0.
"""
import json

import numpy as np
import pandas as pd

from data import ROOT

PRIORITIES = ["P1", "P2", "P3", "P4"]


def load_costs():
    """Returns (W, A, cap): misroute and abstain cost per priority, and max abstain rate."""
    m = pd.read_csv(ROOT / "artifacts" / "routing_cost_matrix.csv").pivot(
        index="true_priority", columns="outcome", values="cost")
    cap = json.loads((ROOT / "artifacts" / "triage_capacity.json").read_text())["max_abstain_rate"]
    return m["misroute"].to_dict(), m["abstain"].to_dict(), float(cap)


def cap_count(cap: float, n: int) -> int:
    return int(np.floor(cap * n + 1e-9))


def expected_costs(team_proba, prio_proba, prio_classes, W, A, p_ok=None) -> pd.DataFrame:
    """p_ok defaults to the max team probability; pass e.g. a correctness-model estimate instead."""
    team_proba, prio_proba = np.atleast_2d(team_proba), np.atleast_2d(prio_proba)
    w = np.array([W[k] for k in prio_classes])
    a = np.array([A[k] for k in prio_classes])
    p_ok = team_proba.max(1) if p_ok is None else np.atleast_1d(p_ok)
    e_route = (1 - p_ok) * (prio_proba @ w)
    e_abstain = prio_proba @ a
    return pd.DataFrame({"p_ok": p_ok, "e_route": e_route, "e_abstain": e_abstain,
                         "savings": e_route - e_abstain})


def top_k_mask(score: np.ndarray, k: int, positive_only: bool = False) -> np.ndarray:
    """Boolean mask of the k highest scores (ties broken by position)."""
    mask = np.zeros(len(score), dtype=bool)
    if k > 0:
        idx = np.argsort(-score, kind="stable")[:k]
        mask[idx] = True
    if positive_only:
        mask &= score > 0
    return mask


def abstain_decision(savings: np.ndarray, cap: float | None) -> np.ndarray:
    savings = np.asarray(savings)
    if cap is None:
        return savings > 0
    return top_k_mask(savings, cap_count(cap, len(savings)), positive_only=True)


def route(team_proba, team_classes, prio_proba, prio_classes, cap="default") -> pd.DataFrame:
    """Per-ticket routing decision. cap='default' uses the artifact cap; None disables it."""
    W, A, artifact_cap = load_costs()
    cap = artifact_cap if cap == "default" else cap
    out = expected_costs(team_proba, prio_proba, prio_classes, W, A)
    out.insert(0, "pred_team", np.asarray(team_classes)[np.atleast_2d(team_proba).argmax(1)])
    out["abstain"] = abstain_decision(out["savings"].to_numpy(), cap).astype(int)
    return out


def realized_cost(team_correct, abstain, true_priority, W, A) -> np.ndarray:
    """Per-ticket penalty scored with TRUE priority."""
    w = np.array([W[k] for k in true_priority])
    a = np.array([A[k] for k in true_priority])
    return np.where(abstain, a, np.where(team_correct, 0.0, w))


def abstention_efficiency(abstain, team_correct, true_priority, W, A) -> pd.DataFrame:
    """Per TRUE priority: abstains, errors caught, hit rate, net gain = caught*W - abstains*A."""
    abstain, team_correct, true_priority = map(np.asarray, (abstain, team_correct, true_priority))
    rows = []
    for k in PRIORITIES + ["ALL"]:
        sel = np.ones(len(abstain), bool) if k == "ALL" else true_priority == k
        n_abs = int((abstain & sel).sum())
        caught_mask = abstain & ~team_correct & sel
        gain = sum(W[p] for p in true_priority[caught_mask]) - sum(A[p] for p in true_priority[abstain & sel])
        rows.append({"priority": k, "abstains": n_abs, "caught": int(caught_mask.sum()),
                     "hit_rate": caught_mask.sum() / n_abs if n_abs else np.nan, "net_gain": gain})
    return pd.DataFrame(rows)
