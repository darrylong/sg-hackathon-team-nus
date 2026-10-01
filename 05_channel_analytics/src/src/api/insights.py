"""Insight cards for the dashboard. Every number in a takeaway is computed here from the data.

Card shape:
    {id, title, takeaway, metric: {label, value, format},
     chart: {type: line|bar|stacked_bar|area, x, x_format, y_format, series: [{key, label, color}], data},
     table: {columns: [{key, label, format}], rows} | None}
Colors are tokens resolved by the frontend: "region:AMS", "status:atRisk", "graph:0" ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_prep import COHORT_EXISTING, COHORT_NEW, PRODUCTS, REGIONS, TIERS
from ..eda import fm
from ..features import shift_quarter
from .analytics import clean

TOKEN_SHARE, RECOVER_SHARE = 0.10, 0.30


def _card(id_, title, takeaway, metric, chart, table=None) -> dict:
    return {"id": id_, "title": title, "takeaway": takeaway, "metric": metric, "chart": chart, "table": table}


def outlier(state):
    allsales = state.tables["sales"]
    out = allsales[allsales["is_outlier"]]
    if out.empty:
        return None
    o = out.iloc[0]
    region = str(o["region"])
    reported = allsales[allsales["region"] == region].groupby("quarter")["revenue_usd"].sum()
    cleaned = state.sales[state.sales["region"] == region].groupby("quarter")["revenue_usd"].sum()
    q = o["quarter"]
    pq_ = shift_quarter(q, -1)
    nxt = state.sales["revenue_usd"].max()
    data = [{"quarter": k, "reported": reported[k], "cleaned": cleaned.get(k, np.nan)} for k in reported.index]
    take = (f"One {o['product_family']} order line from {o['partner_id']} (qty {int(o['quantity']):,}, {fm(o['revenue_usd'])}) "
            f"is {o['revenue_usd'] / nxt:.0f}x the next-largest line. It makes {region} {q} look like "
            f"{reported[q] / reported[pq_] - 1:+.1%} QoQ; without it the quarter is {cleaned[q] / cleaned[pq_] - 1:+.1%}. "
            f"It is excluded from every number in this dashboard.")
    return _card("outlier", f"One order line distorts {region}", take,
                 {"label": f"{region} {q} inflated by", "value": o["revenue_usd"], "format": "money"},
                 {"type": "line", "x": "quarter", "y_format": "money", "data": data,
                  "series": [{"key": "reported", "label": "As reported", "color": "status:atRisk"},
                             {"key": "cleaned", "label": "Outlier removed", "color": "region:AMS"}]})


def growth_vs_shrinking(state):
    ps = state.tables["partner_summary"]
    act = ps[ps["revenue_h1_2025"] > 0]
    rows = []
    for r in REGIONS:
        g = act[act["region"].astype(str) == r]
        rows.append({"region": r, "total_growth": g["revenue_h1_2026"].sum() / g["revenue_h1_2025"].sum() - 1,
                     "pct_declining": (g["growth_h1_yoy"] < 0).mean(), "partners": len(g)})
    d = pd.DataFrame(rows).set_index("region")
    take = (f"Like-for-like H1 2026 vs H1 2025: AMS {d.loc['AMS', 'total_growth']:+.0%}, APJ {d.loc['APJ', 'total_growth']:+.0%}, "
            f"EMEA {d.loc['EMEA', 'total_growth']:+.0%}. Behind the positive APJ/EMEA totals, "
            f"{d.loc[['APJ', 'EMEA'], 'pct_declining'].min():.0%}-{d.loc[['APJ', 'EMEA'], 'pct_declining'].max():.0%} of partners are shrinking.")
    return _card("growth_vs_shrinking", "Totals grow while many partners shrink", take,
                 {"label": "APJ + EMEA partners shrinking", "value": float(
                     (act[act["region"].astype(str).isin(["APJ", "EMEA"])]["growth_h1_yoy"] < 0).mean()), "format": "pct"},
                 {"type": "bar", "x": "region", "y_format": "pct", "data": d.reset_index().to_dict("records"),
                  "series": [{"key": "total_growth", "label": "Revenue growth (like for like)", "color": "graph:0"},
                             {"key": "pct_declining", "label": "Share of partners shrinking", "color": "status:atRisk"}]})


def new_partners(state):
    pq = state.pq.merge(state.tables["partner_summary"][["partner_id", "cohort"]], on="partner_id")
    c = pq.groupby(["region", "cohort", "quarter"])["revenue_usd"].sum()
    apj_emea = pq[pq["region"].isin(["APJ", "EMEA"])].groupby(["quarter", "cohort"])["revenue_usd"].sum().unstack()
    data = [{"quarter": q, "existing": r.get(COHORT_EXISTING, 0), "new": r.get(COHORT_NEW, 0)} for q, r in apj_emea.iterrows()]
    lq = sorted(pq["quarter"].unique())[-1]
    ly = shift_quarter(lq, -4)
    trows = []
    for r in REGIONS:
        ex_now, ex_ly = c.get((r, COHORT_EXISTING, lq), 0), c.get((r, COHORT_EXISTING, ly), 0)
        trows.append({"region": r, "existing_yoy": ex_now / ex_ly - 1 if ex_ly else None,
                      "new_cohort_revenue": c.get((r, COHORT_NEW, lq), 0)})
    t = pd.DataFrame(trows).set_index("region")
    take = (f"Partners onboarded before 2024 are almost flat in APJ ({t.loc['APJ', 'existing_yoy']:+.1%} YoY in {lq}) and "
            f"EMEA ({t.loc['EMEA', 'existing_yoy']:+.1%}); their growth comes from partners onboarded since 2024 "
            f"({fm(t.loc['APJ', 'new_cohort_revenue'])} and {fm(t.loc['EMEA', 'new_cohort_revenue'])} in {lq}). "
            f"In AMS existing partners still grow {t.loc['AMS', 'existing_yoy']:+.0%}.")
    return _card("new_partners", "APJ and EMEA growth comes from new partners", take,
                 {"label": f"New-partner revenue, APJ + EMEA, {lq}", "value": data[-1]["new"], "format": "money"},
                 {"type": "stacked_bar", "x": "quarter", "y_format": "money", "data": data,
                  "series": [{"key": "existing", "label": "Onboarded before 2024", "color": "graph:0"},
                             {"key": "new", "label": "Onboarded 2024+", "color": "graph:2"}]},
                 {"columns": [{"key": "region", "label": "Region"},
                              {"key": "existing_yoy", "label": f"Existing partners YoY ({lq})", "format": "pct_signed"},
                              {"key": "new_cohort_revenue", "label": f"New partners' revenue ({lq})", "format": "money"}],
                  "rows": t.reset_index().to_dict("records")})


def concentration(state):
    rev = state.tables["partner_summary"]["revenue_total"].fillna(0).sort_values(ascending=False).to_numpy()
    cum = np.cumsum(rev) / rev.sum()
    n = len(rev)
    data = [{"partner_share": i / 50, "revenue_share": float(cum[max(int(n * i / 50) - 1, 0)])} for i in range(1, 51)]
    data.insert(0, {"partner_share": 0.0, "revenue_share": 0.0})
    top = {p: float(cum[int(n * p) - 1]) for p in (0.01, 0.10, 0.20)}
    take = (f"The top 1% of partners ({int(n * 0.01)}) bring {top[0.01]:.0%} of revenue, the top 10% bring {top[0.10]:.0%} "
            f"and the top 20% bring {top[0.20]:.0%}. A handful of large accounts set the trend.")
    return _card("concentration", "Revenue is concentrated", take,
                 {"label": "Revenue from the top 10% of partners", "value": top[0.10], "format": "pct"},
                 {"type": "area", "x": "partner_share", "x_format": "pct", "y_format": "pct", "data": data,
                  "series": [{"key": "revenue_share", "label": "Cumulative share of revenue", "color": "graph:0"}]})


def volume_not_price(state):
    g = state.sales.groupby(["quarter", "product_family"]).agg(rev=("revenue_usd", "sum"), qty=("quantity", "sum"))
    price = (g["rev"] / g["qty"]).unstack()[PRODUCTS]
    idx = price / price.iloc[0] * 100
    qty = g["qty"].unstack()[PRODUCTS]
    first, last = price.index[0], price.index[-1]
    move = (price.loc[last] / price.loc[first] - 1).abs().max()
    qgrowth = qty.loc[last].sum() / qty.loc[first].sum() - 1
    data = [{"quarter": q, **r.to_dict()} for q, r in idx.iterrows()]
    take = (f"Unit prices are flat: no product moved more than {move:.1%} between {first} and {last}, while units grew "
            f"{qgrowth:+.0%}. All revenue growth is volume, so growth depends on partner activity, not pricing.")
    return _card("volume_not_price", "Growth is volume, not price", take,
                 {"label": f"Largest unit-price move, {first} to {last}", "value": float(move), "format": "pct"},
                 {"type": "line", "x": "quarter", "y_format": "index", "data": data,
                  "series": [{"key": p, "label": p, "color": f"graph:{i}"} for i, p in enumerate(PRODUCTS)]})


def margin_mix(state):
    s = state.sales
    by_p = s.groupby("product_family").agg(rev=("revenue_usd", "sum"), m=("margin_usd", "sum"))
    by_p = (by_p["m"] / by_p["rev"] * 100).reindex(PRODUCTS).sort_values(ascending=False)
    by_t = s.groupby("tier").agg(rev=("revenue_usd", "sum"), m=("margin_usd", "sum"))
    by_t = (by_t["m"] / by_t["rev"] * 100).reindex(TIERS)
    q = s.groupby("quarter").agg(rev=("revenue_usd", "sum"), m=("margin_usd", "sum"))
    blended = q["m"] / q["rev"] * 100
    take = (f"Margin depends on what is sold and who sells it: GreenLake earns {by_p['GreenLake']:.1f}% vs ProLiant "
            f"{by_p['ProLiant']:.1f}%, and Platinum partners {by_t['Platinum']:.1f}% vs Silver {by_t['Silver']:.1f}%. "
            f"The blended margin barely moves ({blended.min():.1f}-{blended.max():.1f}%), so watch the mix, not the headline.")
    return _card("margin_mix", "Margin is a mix story", take,
                 {"label": "GreenLake vs ProLiant margin gap", "value": float(by_p["GreenLake"] - by_p["ProLiant"]), "format": "pp"},
                 {"type": "bar", "x": "product", "y_format": "pct_points",
                  "data": [{"product": k, "margin_pct": v} for k, v in by_p.items()],
                  "series": [{"key": "margin_pct", "label": "Partner margin %", "color": "graph:1"}]},
                 {"columns": [{"key": "tier", "label": "Tier"}, {"key": "margin_pct", "label": "Margin %", "format": "pct_points"}],
                  "rows": [{"tier": k, "margin_pct": v} for k, v in by_t.items()]})


def attainment(state):
    pq = state.pq[state.pq["target_usd"].notna()]
    g = pq.groupby("quarter")
    d = pd.DataFrame({"revenue_vs_target": g["revenue_usd"].sum() / g["target_usd"].sum(),
                      "pct_partners_hit": g["attainment"].apply(lambda a: (a >= 1).mean())})
    take = (f"In aggregate the channel delivers {d['revenue_vs_target'].min():.0%}-{d['revenue_vs_target'].max():.0%} of target "
            f"every quarter, yet only about {d['pct_partners_hit'].mean():.0%} of partners hit their own target. "
            f"The total hides a wide spread.")
    return _card("attainment", "Near target overall, most partners miss", take,
                 {"label": "Partners hitting target (avg quarter)", "value": float(d["pct_partners_hit"].mean()), "format": "pct"},
                 {"type": "line", "x": "quarter", "y_format": "pct", "data": d.reset_index().to_dict("records"),
                  "series": [{"key": "revenue_vs_target", "label": "Channel revenue / target", "color": "graph:0"},
                             {"key": "pct_partners_hit", "label": "Share of partners at or above target", "color": "graph:3"}]})


def seasonality(state):
    m = state.sales.groupby("month")["revenue_usd"].sum()
    miq = (m.index.month - 1) % 3 + 1
    spike = m[miq == 3].mean() / m[miq != 3].mean()
    q = state.sales.groupby(["quarter", "region"])["revenue_usd"].sum().unstack()[REGIONS]
    ratios = [{"year": y, **{r: q.loc[f"{y}-Q3", r] / q.loc[f"{y}-Q2", r] for r in REGIONS}}
              for y in (2024, 2025) if f"{y}-Q3" in q.index]
    data = [{"month": k.strftime("%Y-%m"), "revenue": v, "quarter_end": bool(i == 3)} for k, v, i in zip(m.index, m.values, miq)]
    take = (f"The last month of every quarter brings {spike:.1f}x the revenue of the other two, Q4 is the peak, and Q3 dips "
            f"below Q2 every year. Compare quarters year on year, not quarter on quarter.")
    return _card("seasonality", "Quarter-end spikes and a Q3 dip", take,
                 {"label": "Quarter-end month vs other months", "value": float(spike), "format": "ratio"},
                 {"type": "bar", "x": "month", "y_format": "money", "data": data, "highlight": "quarter_end",
                  "series": [{"key": "revenue", "label": "Monthly revenue", "color": "graph:0"}]},
                 {"columns": [{"key": "year", "label": "Year"}] + [{"key": r, "label": f"{r} Q3 / Q2", "format": "ratio"} for r in REGIONS],
                  "rows": ratios})


def churn_noise(state):
    rev = state.pq.pivot_table(index="partner_id", columns="quarter", values="revenue_usd", aggfunc="sum")
    qs = sorted(rev.columns)
    data = []
    for i in range(4, len(qs)):
        prev = rev[qs[i - 4:i]]
        base = prev.mean(axis=1)
        elig = (prev.notna().sum(axis=1) >= 2) & (base > 0) & rev[qs[i]].notna()
        leaver = elig & (rev[qs[i]] < TOKEN_SHARE * base)
        if i + 1 < len(qs):
            rec = int((leaver & (rev[qs[i + 1]] >= RECOVER_SHARE * base)).sum())
            data.append({"quarter": qs[i], "recovered": rec, "stayed_low": int(leaver.sum()) - rec, "unknown": 0})
        else:
            data.append({"quarter": qs[i], "recovered": 0, "stayed_low": 0, "unknown": int(leaver.sum())})
    known = [d for d in data if d["unknown"] == 0]
    rates = [d["recovered"] / (d["recovered"] + d["stayed_low"]) for d in known if d["recovered"] + d["stayed_low"]]
    take = (f"Each quarter roughly {min(d['recovered'] + d['stayed_low'] + d['unknown'] for d in data)}-"
            f"{max(d['recovered'] + d['stayed_low'] + d['unknown'] for d in data)} partners see revenue collapse below 10% of normal, "
            f"but {min(rates):.0%}-{max(rates):.0%} of them recover the next quarter. They are mostly small, irregular buyers. "
            f"That is why the at-risk model scores ordering frequency and recency, not just revenue drops.")
    return _card("churn_noise", "Most revenue collapses recover", take,
                 {"label": "Recover the next quarter (typical)", "value": float(np.median(rates)), "format": "pct"},
                 {"type": "stacked_bar", "x": "quarter", "y_format": "count", "data": data,
                  "series": [{"key": "recovered", "label": "Recovered next quarter", "color": "status:healthy"},
                             {"key": "stayed_low", "label": "Stayed low", "color": "status:atRisk"},
                             {"key": "unknown", "label": "Outcome not known yet", "color": "neutral"}]})


def declining_not_leaving(state):
    if state.at_risk is None:
        return None
    P = state.at_risk.panel.copy()
    P["size_decile"] = P.groupby("cutoff")["rev_avg_4q_log"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    groups = {"YoY down 2+ quarters": P["yoy_decline_streak"] >= 2, "YoY down 3+ quarters": P["yoy_decline_streak"] >= 3,
              "YoY down 4+ quarters": P["yoy_decline_streak"] >= 4, "Revenue < 50% of peak": P["rev_drawdown"] < 0.5}
    rows = []
    for name, mask in groups.items():
        g, rest = P[mask], P[~mask]
        ref = rest.groupby(["cutoff", "tier", "size_decile"], observed=True)["left_token"].mean()
        exp = g.set_index(["cutoff", "tier", "size_decile"]).index.map(ref).to_series().fillna(rest["left_token"].mean()).sum()
        rows.append({"group": name, "snapshots": len(g), "leavers": int(g["left_token"].sum()), "expected": float(exp),
                     "ratio": float(g["left_token"].sum() / exp) if exp else None})
    n_decl = int((state.at_risk.scores["segment"] == "Quietly declining").sum())
    take = (f"{n_decl} partners are quietly declining, but declining is not leaving: in history, partners with YoY revenue "
            f"down 2-4 quarters in a row left no more often than partners of the same size and tier "
            f"(ratio {min(r['ratio'] for r in rows[:3]):.2f}-{max(r['ratio'] for r in rows[:3]):.2f}). Only distance from "
            f"peak revenue carried extra risk ({rows[3]['ratio']:.2f}x), and the model already uses it. Decliners need a "
            f"growth conversation, not a retention alarm.")
    return _card("declining_not_leaving", "Declining is not the same as leaving", take,
                 {"label": "Partners quietly declining", "value": n_decl, "format": "count"},
                 {"type": "bar", "x": "group", "y_format": "ratio", "data": rows, "reference": 1.0,
                  "series": [{"key": "ratio", "label": "Leavers vs expected at same size and tier", "color": "status:declining"}]},
                 {"columns": [{"key": "group", "label": "Group"}, {"key": "snapshots", "label": "Partner snapshots", "format": "count"},
                              {"key": "leavers", "label": "Left next quarter", "format": "count"},
                              {"key": "expected", "label": "Expected", "format": "decimal"},
                              {"key": "ratio", "label": "Ratio", "format": "ratio"}],
                  "rows": rows})


BUILDERS = [growth_vs_shrinking, new_partners, concentration, margin_mix, attainment, seasonality, volume_not_price,
            outlier, churn_noise, declining_not_leaving]


def build_insights(state) -> list[dict]:
    cards = []
    for fn in BUILDERS:
        try:
            card = fn(state)
        except Exception as exc:  # noqa: BLE001 - one broken card must not hide the others
            state.errors[f"insight:{fn.__name__}"] = f"{type(exc).__name__}: {exc}"
            continue
        if card:
            cards.append(card)
    return clean(cards)
