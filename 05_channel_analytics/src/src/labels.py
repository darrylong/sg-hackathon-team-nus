"""Proxy churn labels for historical snapshots.

The judges' definition: a leaver places no orders, or only a token order, in the next quarter.
For a cutoff quarter Q we look one quarter ahead (Q+1):

    left_zero        no order lines at all in Q+1
    left_token       no orders, or Q+1 revenue < TOKEN_SHARE x the partner's average quarterly
                     revenue over the 4 quarters up to Q   <- primary label (matches the definition)
    left_persistent  left_token in both Q+1 and Q+2 (stricter; NaN when Q+2 is not in the data)

The baseline uses quarters since onboarding only, so a partner onboarded mid-window is compared
with its own short history. Partners with no revenue in the baseline window are leavers only if
they also place no orders in Q+1.
"""
from __future__ import annotations

import pandas as pd

from .features import quarter_end, shift_quarter

TOKEN_SHARE = 0.10


def build_labels(tables: dict[str, pd.DataFrame], cutoff: str, token_share: float = TOKEN_SHARE) -> pd.DataFrame:
    pq = tables["partner_quarter"]
    known = set(pq["quarter"])
    nq, nq2 = shift_quarter(cutoff, 1), shift_quarter(cutoff, 2)
    if nq not in known:
        raise ValueError(f"No data for {nq}: cannot label cutoff {cutoff}")

    end = quarter_end(cutoff)
    idx = tables["master"].loc[tables["master"]["onboarded_date"] <= end, "partner_id"]
    base_qs = [shift_quarter(cutoff, -i) for i in range(3, -1, -1)]
    cols = base_qs + [nq] + ([nq2] if nq2 in known else [])
    rev = pq.pivot(index="partner_id", columns="quarter", values="revenue_usd").reindex(index=idx, columns=cols)
    lines = pq.pivot(index="partner_id", columns="quarter", values="order_lines").reindex(index=idx, columns=cols)
    baseline = rev[base_qs].mean(axis=1)

    def token(quarter: str) -> pd.Series:
        return (lines[quarter] == 0) | (rev[quarter] < token_share * baseline)

    out = pd.DataFrame(index=idx)
    out["baseline_4q_revenue"] = baseline
    out["next_q_revenue"] = rev[nq]
    out["left_zero"] = (lines[nq] == 0).astype(int)
    out["left_token"] = token(nq).astype(int)
    out["left_persistent"] = (token(nq) & token(nq2)).astype(float) if nq2 in known else float("nan")
    out.insert(0, "cutoff", cutoff)
    return out.reset_index()
