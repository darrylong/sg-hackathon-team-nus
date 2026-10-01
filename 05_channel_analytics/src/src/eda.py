"""Exploratory data analysis of the partner channel data.

    python -m src.eda [--data-dir PATH]

Prints a summary to the console and writes:
    outputs/eda/eda_report.md    findings with tables and charts
    outputs/eda/charts/*.png     one chart per finding
    outputs/eda/tables/*.csv     the table behind each finding
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import plot_style as ps
from .data_prep import (COHORT_EXISTING, COHORT_NEW, DATA_DIR, PRODUCTS, REGIONS, ROOT, TIERS,
                        aggregate, build_all)

OUT_DIR = ROOT / "outputs" / "eda"
CHARTS = OUT_DIR / "charts"
TABLES = OUT_DIR / "tables"

# Naive "left the channel" label used to test how noisy a revenue-drop definition is.
TOKEN_SHARE = 0.10    # quarter revenue below 10% of the partner's trailing 4-quarter average
RECOVER_SHARE = 0.30  # back above 30% of that average the following quarter = recovered


# ---------------------------------------------------------------- formatting + report

def fm(v: float) -> str:
    return "-" if pd.isna(v) else f"${v / 1e6:,.1f}M"


def fk(v: float) -> str:
    return "-" if pd.isna(v) else f"${v / 1e3:,.0f}K"


def fp(v: float, signed: bool = False) -> str:
    return "-" if pd.isna(v) else (f"{v:+.1%}" if signed else f"{v:.1%}")


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.takeaways: list[str] = []

    def section(self, title: str) -> None:
        self.lines += [f"## {title}", ""]
        print(f"\n=== {title} ===")

    def text(self, s: str) -> None:
        self.lines += [s, ""]
        print(s)

    def table(self, df: pd.DataFrame, name: str, raw: pd.DataFrame | None = None) -> None:
        """Show `df` (display-formatted) and save `raw` (or df) as CSV."""
        (raw if raw is not None else df).to_csv(TABLES / f"{name}.csv", index=False)
        self.lines += [md_table(df), ""]
        print(df.to_string(index=False))

    def chart(self, path: Path, alt: str) -> None:
        self.lines += [f"![{alt}](charts/{path.name})", ""]

    def takeaway(self, s: str) -> None:
        self.takeaways.append(s)

    def write(self, path: Path, header: list[str]) -> None:
        body = header + ["## Key takeaways", ""] + [f"- {t}" for t in self.takeaways] + [""] + self.lines
        path.write_text("\n".join(body), encoding="utf-8")


# ---------------------------------------------------------------- helpers

def period_revenue(base: pd.DataFrame, by: str, start: str, end: str) -> pd.Series:
    mask = (base["order_date"] >= start) & (base["order_date"] <= end)
    return base[mask].groupby(by, observed=True)["revenue_usd"].sum()


def quarter_pivot(pq: pd.DataFrame, value: str = "revenue_usd") -> pd.DataFrame:
    return pq.pivot_table(index="partner_id", columns="quarter", values=value, aggfunc="sum")


# ---------------------------------------------------------------- sections

def data_quality(t: dict, r: Report) -> None:
    r.section("1. Data quality")
    sales, master, targets = t["sales"], t["master"], t["targets"]
    raw_cols = ["order_date", "partner_id", "partner_name", "region", "tier", "product_family",
                "quantity", "revenue_usd", "margin_pct"]

    dims = sales[["partner_id", "region", "tier"]].drop_duplicates().merge(
        master[["partner_id", "region", "tier"]], on="partner_id", suffixes=("", "_master"))
    first_order = sales.groupby("partner_id")["order_date"].min()
    onboarded = master.set_index("partner_id")["onboarded_date"]
    sold_ids, master_ids, target_ids = set(sales["partner_id"]), set(master["partner_id"]), set(targets["partner_id"])

    checks = [
        ("Sales lines", f"{len(sales):,}"),
        ("Partners in master", f"{len(master):,}"),
        ("Target rows", f"{len(targets):,}"),
        ("Order date range", f"{sales['order_date'].min():%Y-%m-%d} .. {sales['order_date'].max():%Y-%m-%d}"),
        ("Target quarters", f"{targets['quarter'].min()} .. {targets['quarter'].max()}"),
        ("Null cells (sales / master / targets)",
         f"{sales[raw_cols].isna().sum().sum()} / {master.isna().sum().sum()} / {targets.isna().sum().sum()}"),
        ("Duplicate sales lines", f"{sales.duplicated(subset=raw_cols).sum()}"),
        ("Lines with revenue or quantity <= 0", f"{((sales['revenue_usd'] <= 0) | (sales['quantity'] <= 0)).sum()}"),
        ("Sales partner ids missing from master", f"{len(sold_ids - master_ids)}"),
        ("Master partners with no sales", f"{len(master_ids - sold_ids)}"),
        ("Master partners with no targets", f"{len(master_ids - target_ids)}"),
        ("Region / tier differs between sales and master",
         f"{(dims['region'] != dims['region_master']).sum()} / {(dims['tier'] != dims['tier_master']).sum()}"),
        ("Partners with an order before onboarding", f"{(first_order < onboarded.reindex(first_order.index)).sum()}"),
        ("Partners onboarded during the data window", f"{(master['onboarded_date'] >= '2024-01-01').sum()}"),
        ("Outlier lines (revenue > 20x product-family p99)", f"{sales['is_outlier'].sum()}"),
    ]
    r.table(pd.DataFrame(checks, columns=["Check", "Result"]), "data_quality")

    out = sales[sales["is_outlier"]]
    if len(out):
        show = out[["order_date", "partner_id", "region", "tier", "product_family", "quantity", "revenue_usd"]].copy()
        show["order_date"] = show["order_date"].dt.strftime("%Y-%m-%d")
        next_biggest = sales.loc[~sales["is_outlier"], "revenue_usd"].max()
        r.text(f"Outlier lines (the next-largest normal line is {fm(next_biggest)}). "
               "They are kept in `sales` with `is_outlier=True` and excluded from every aggregate:")
        r.table(show.assign(revenue_usd=show["revenue_usd"].map(fm)), "outlier_lines", raw=out)
        for row in out.itertuples():
            r.takeaway(f"One corrupt-looking line: {row.partner_id} ({row.region}) {row.product_family}, "
                       f"qty {row.quantity:,}, {fm(row.revenue_usd)} on {row.order_date:%Y-%m-%d}, "
                       f"{row.revenue_usd / next_biggest:.0f}x the next-largest line. Excluded from all aggregates.")


def headline_trend(t: dict, r: Report) -> None:
    r.section("2. Revenue trend by region")
    rq = t["region_quarter"].pivot(index="quarter", columns="region", values="revenue_usd")
    rq["ALL"] = rq.sum(axis=1)
    yoy = rq / rq.shift(4) - 1
    show = rq.map(fm)
    for c in ["ALL"]:
        show[f"{c} YoY"] = yoy[c].map(lambda v: fp(v, signed=True))
    r.table(show.reset_index(), "revenue_region_quarter", raw=rq.reset_index())

    rm = t["region_month"].pivot(index="month", columns="region", values="revenue_usd")
    fig, ax = ps.plt.subplots(figsize=(10, 4.2))
    for region in REGIONS:
        ax.plot(rm.index, rm[region], color=ps.REGION_COLORS[region], label=region)
        ps.label_line_end(ax, rm.index[-1], rm[region].iloc[-1], region)
    ax.yaxis.set_major_formatter(ps.money_m)
    ax.set_title("Monthly revenue by region (outlier excluded) - quarter-end months spike")
    ax.legend(loc="upper left", ncol=3)
    r.chart(ps.save(fig, CHARTS / "01_monthly_revenue_by_region.png"), "Monthly revenue by region")

    month_of_q = t["region_month"].assign(m=lambda d: (d["month"].dt.month - 1) % 3 + 1).groupby("m")["revenue_usd"].sum()
    spike = month_of_q[3] / month_of_q[[1, 2]].mean()
    r.text(f"Third month of each quarter carries {spike:.2f}x the revenue of the first two months on average.")
    r.takeaway(f"Strong quarter-end pattern: the last month of a quarter is {spike:.1f}x the other two; "
               f"Q4 is the peak and Q3 dips below Q2 every year. Compare quarters YoY, not QoQ.")


def outlier_effect(t: dict, r: Report) -> None:
    out = t["sales"][t["sales"]["is_outlier"]]
    if out.empty:
        return
    r.section("3. How one outlier line distorts a region")
    region = out["region"].iloc[0]
    reported = t["sales"][t["sales"]["region"] == region].groupby("quarter")["revenue_usd"].sum()
    cleaned = t["region_quarter"].query("region == @region").set_index("quarter")["revenue_usd"]

    fig, ax = ps.plt.subplots(figsize=(9, 3.8))
    ax.plot(reported.index, reported.values, color=ps.SERIES[1], label="As reported", marker="o", markersize=5)
    ax.plot(cleaned.index, cleaned.values, color=ps.SERIES[0], label="Outlier line removed", marker="o", markersize=5)
    ax.yaxis.set_major_formatter(ps.money_m)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{region} quarterly revenue: as reported vs outlier removed")
    ax.legend(loc="upper left")
    r.chart(ps.save(fig, CHARTS / "02_outlier_effect.png"), "Outlier effect")

    q = out["quarter"].iloc[0]
    prev_q = reported.index[list(reported.index).index(q) - 1]
    r.text(f"{region} {q}: reported {fm(reported[q])} ({fp(reported[q] / reported[prev_q] - 1, True)} QoQ) "
           f"vs {fm(cleaned[q])} ({fp(cleaned[q] / cleaned[prev_q] - 1, True)} QoQ) once the line is removed.")


def dimension_breakdowns(t: dict, r: Report) -> None:
    r.section("4. Revenue, margin, growth and attainment by dimension")
    base = t["sales"][~t["sales"]["is_outlier"]]
    pq = t["partner_quarter"]
    last4 = sorted(pq["quarter"].unique())[-4:]
    total = base["revenue_usd"].sum()

    for dim in ["region", "tier", "product_family", "partner_type"]:
        agg = aggregate(base, dim).set_index(dim)
        h25 = period_revenue(base, dim, "2025-01-01", "2025-06-30")
        h26 = period_revenue(base, dim, "2026-01-01", "2026-06-30")
        tbl = pd.DataFrame({
            "revenue_usd": agg["revenue_usd"],
            "share": agg["revenue_usd"] / total,
            "margin_pct": agg["margin_pct"],
            "growth_h1_yoy": h26 / h25 - 1,
            "partners": agg["partners"],
        })
        if dim in pq.columns:  # targets are per partner, so no attainment by product
            recent = pq[pq["quarter"].isin(last4) & pq["target_usd"].notna()]
            g = recent.groupby(dim, observed=True)
            tbl["attainment_last4q"] = g["revenue_usd"].sum() / g["target_usd"].sum()
            tbl["pct_partner_q_hit"] = g["attainment"].apply(lambda a: (a >= 1).mean())
        raw = tbl.reset_index()
        show = raw.copy()
        show["revenue_usd"] = show["revenue_usd"].map(fm)
        show["share"] = show["share"].map(fp)
        show["margin_pct"] = show["margin_pct"].map(lambda v: f"{v:.1f}%")
        show["growth_h1_yoy"] = show["growth_h1_yoy"].map(lambda v: fp(v, True))
        for c in ["attainment_last4q", "pct_partner_q_hit"]:
            if c in show:
                show[c] = show[c].map(fp)
        r.text(f"**By {dim}** (30 months; growth = H1 2026 vs H1 2025; attainment over {last4[0]}..{last4[-1]})")
        r.table(show, f"breakdown_{dim}", raw=raw)

    # margin by product and tier
    fig, axes = ps.plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, dim, order in [(axes[0], "product_family", PRODUCTS), (axes[1], "tier", TIERS)]:
        m = aggregate(base, dim).set_index(dim).reindex(order)["margin_pct"].sort_values()
        ax.barh(m.index.astype(str), m.values, color=ps.SERIES[0], height=0.6, edgecolor=ps.SURFACE, linewidth=2)
        for y, v in enumerate(m.values):
            ax.annotate(f"{v:.1f}%", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=ps.INK_2)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.xaxis.set_major_formatter(ps.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title(f"Partner margin % by {dim.replace('_', ' ')}")
    r.chart(ps.save(fig, CHARTS / "03_margin_by_product_and_tier.png"), "Margin by product and tier")

    pm = aggregate(base, "product_family").set_index("product_family")["margin_pct"]
    tm = aggregate(base, "tier").set_index("tier")["margin_pct"]
    r.takeaway(f"Margin is driven by mix: GreenLake {pm['GreenLake']:.1f}% vs ProLiant {pm['ProLiant']:.1f}%; "
               f"Platinum partners {tm['Platinum']:.1f}% vs Silver {tm['Silver']:.1f}%. "
               f"Blended margin is nearly flat over time, so watch mix rather than the headline.")


def growth_illusion(t: dict, r: Report) -> None:
    r.section("5. Total growth vs the typical partner")
    ps_ = t["partner_summary"]
    act = ps_[ps_["revenue_h1_2025"] > 0]
    g = act.groupby("region", observed=True)
    tbl = pd.DataFrame({
        "total_growth": g["revenue_h1_2026"].sum() / g["revenue_h1_2025"].sum() - 1,
        "median_partner_growth": g["growth_h1_yoy"].median(),
        "pct_partners_declining": g["growth_h1_yoy"].apply(lambda x: (x < 0).mean()),
        "partners": g.size(),
    }).reindex(REGIONS)
    tbl.loc["ALL"] = [act["revenue_h1_2026"].sum() / act["revenue_h1_2025"].sum() - 1,
                      act["growth_h1_yoy"].median(), (act["growth_h1_yoy"] < 0).mean(), len(act)]
    raw = tbl.reset_index().rename(columns={"index": "region"})
    show = raw.copy()
    for c in ["total_growth", "median_partner_growth"]:
        show[c] = show[c].map(lambda v: fp(v, True))
    show["pct_partners_declining"] = show["pct_partners_declining"].map(fp)
    show["partners"] = show["partners"].astype(int)
    r.text("H1 2026 vs H1 2025, partners that sold in H1 2025 (new partners excluded, so this is like-for-like).")
    r.table(show, "growth_total_vs_partner", raw=raw)

    fig, ax = ps.plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(REGIONS))
    w = 0.38
    for i, (col, label) in enumerate([("total_growth", "Total revenue growth"),
                                      ("pct_partners_declining", "Share of partners declining")]):
        vals = tbl.loc[REGIONS, col].values
        ax.bar(x + (i - 0.5) * w, vals, w, color=ps.SERIES[i], label=label, edgecolor=ps.SURFACE, linewidth=2)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9, color=ps.INK_2)
    ax.set_xticks(x, REGIONS)
    ax.yaxis.set_major_formatter(ps.pct)
    ax.set_title("Like-for-like growth vs share of partners shrinking (H1 26 vs H1 25)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    r.chart(ps.save(fig, CHARTS / "04_growth_vs_declining.png"), "Growth vs declining partners")

    worst = tbl.loc[["APJ", "EMEA"], "pct_partners_declining"]
    r.takeaway(f"AMS like-for-like revenue grew {fp(tbl.loc['AMS', 'total_growth'], True)}, APJ "
               f"{fp(tbl.loc['APJ', 'total_growth'], True)}, EMEA {fp(tbl.loc['EMEA', 'total_growth'], True)}. "
               f"Yet {worst.min():.0%}-{worst.max():.0%} of APJ/EMEA partners are shrinking.")


def cohorts(t: dict, r: Report) -> None:
    r.section("6. New vs existing partners")
    pq = t["partner_quarter"].merge(t["partner_summary"][["partner_id", "cohort"]], on="partner_id")
    c = pq.groupby(["region", "cohort", "quarter"], observed=True)["revenue_usd"].sum().unstack("quarter")
    r.table(c.map(fm).reset_index(), "revenue_region_cohort_quarter", raw=c.reset_index())

    quarters = list(c.columns)
    fig, axes = ps.plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, region in zip(axes, REGIONS):
        for i, cohort in enumerate([COHORT_EXISTING, COHORT_NEW]):
            vals = c.loc[(region, cohort)] if (region, cohort) in c.index else pd.Series(0, index=quarters)
            ax.plot(quarters, vals.values, color=ps.SERIES[i], label=cohort)
        ax.yaxis.set_major_formatter(ps.money_m)
        ax.set_ylim(bottom=0)
        ax.set_xticks(quarters[::3])
        ax.set_title(region)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Quarterly revenue by partner cohort", x=0.01, ha="left", fontweight="semibold")
    r.chart(ps.save(fig, CHARTS / "05_cohort_revenue.png"), "Cohort revenue")

    msgs = []
    for region in REGIONS:
        ex = c.loc[(region, COHORT_EXISTING)]
        nw = c.loc[(region, COHORT_NEW)]
        msgs.append(f"{region}: existing {fp(ex['2026-Q2'] / ex['2025-Q2'] - 1, True)} YoY, "
                    f"new cohort {fm(nw['2026-Q2'])} in 2026-Q2")
    r.text("2026-Q2 vs 2025-Q2 - " + "; ".join(msgs) + ".")
    r.takeaway("Existing (pre-2024) partners are close to flat in APJ and EMEA; most growth there comes from "
               "partners onboarded since 2024. " + "; ".join(msgs) + ".")


def concentration(t: dict, r: Report) -> None:
    r.section("7. Revenue concentration")
    ps_ = t["partner_summary"].sort_values("revenue_total", ascending=False)
    share = ps_["revenue_total"].cumsum() / ps_["revenue_total"].sum()
    n = len(ps_)
    pts = {p: share.iloc[max(int(n * p) - 1, 0)] for p in [0.01, 0.05, 0.10, 0.20, 0.50]}
    r.table(pd.DataFrame({"top partners": [f"{p:.0%} ({int(n * p)})" for p in pts],
                          "share of revenue": [fp(v) for v in pts.values()]}), "concentration",
            raw=pd.DataFrame({"top_share": list(pts), "revenue_share": list(pts.values())}))
    top = ps_.head(10)[["partner_id", "partner_name", "region", "tier", "partner_type", "revenue_total", "growth_h1_yoy"]]
    r.table(top.assign(revenue_total=top["revenue_total"].map(fm),
                       growth_h1_yoy=top["growth_h1_yoy"].map(lambda v: fp(v, True))), "top10_partners", raw=top)

    fig, ax = ps.plt.subplots(figsize=(7, 4))
    xs = np.arange(1, n + 1) / n
    ax.plot(xs, share.values, color=ps.SERIES[0])
    for p in [0.10, 0.20]:
        ax.annotate(f"top {p:.0%} = {pts[p]:.0%} of revenue", (p, pts[p]), xytext=(10, -14),
                    textcoords="offset points", fontsize=9, color=ps.INK_2)
        ax.plot([p], [pts[p]], "o", color=ps.SERIES[0], markersize=6, markeredgecolor=ps.SURFACE, markeredgewidth=2)
    ax.xaxis.set_major_formatter(ps.pct)
    ax.yaxis.set_major_formatter(ps.pct)
    ax.set_xlabel("Share of partners (largest first)")
    ax.set_title("Cumulative share of revenue")
    r.chart(ps.save(fig, CHARTS / "06_concentration.png"), "Revenue concentration")
    r.takeaway(f"Top 10% of partners bring {pts[0.10]:.0%} of revenue and the top 20% bring {pts[0.20]:.0%}.")


def price_vs_volume(t: dict, r: Report) -> None:
    r.section("8. Price vs volume")
    pq = t["product_quarter"].copy()
    pq["unit_price"] = pq["revenue_usd"] / pq["quantity"]
    first, last = pq["quarter"].min(), pq["quarter"].max()
    p = pq.pivot(index="product_family", columns="quarter", values="unit_price")
    q = pq.pivot(index="product_family", columns="quarter", values="quantity")
    tbl = pd.DataFrame({
        f"unit price {first}": p[first], f"unit price {last}": p[last],
        "price change": p[last] / p[first] - 1, "quantity change": q[last] / q[first] - 1,
    }).reset_index()
    show = tbl.copy()
    for c in [f"unit price {first}", f"unit price {last}"]:
        show[c] = show[c].map(lambda v: f"${v:,.0f}")
    for c in ["price change", "quantity change"]:
        show[c] = show[c].map(lambda v: fp(v, True))
    r.table(show, "price_vs_volume", raw=tbl)
    r.takeaway(f"Unit prices are flat ({fp(tbl['price change'].abs().max())} max move); all revenue growth is volume.")


def attainment(t: dict, r: Report) -> None:
    r.section("9. Target attainment")
    pq = t["partner_quarter"][t["partner_quarter"]["target_usd"].notna()]
    g = pq.groupby("quarter")
    tbl = pd.DataFrame({
        "revenue_vs_target": g["revenue_usd"].sum() / g["target_usd"].sum(),
        "pct_partners_hit": g["attainment"].apply(lambda a: (a >= 1).mean()),
        "median_attainment": g["attainment"].median(),
        "partners_with_target": g.size(),
    }).reset_index()
    show = tbl.copy()
    for c in ["revenue_vs_target", "pct_partners_hit", "median_attainment"]:
        show[c] = show[c].map(fp)
    r.table(show, "attainment_by_quarter", raw=tbl)

    last_q = pq["quarter"].max()
    a = pq.loc[pq["quarter"] == last_q, "attainment"].clip(upper=3)
    fig, ax = ps.plt.subplots(figsize=(8, 3.6))
    ax.hist(a, bins=np.arange(0, 3.05, 0.1), color=ps.SERIES[0], edgecolor=ps.SURFACE, linewidth=1)
    ax.axvline(1, color=ps.INK_2, linewidth=1, linestyle="--")
    ax.annotate("target", (1, ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points",
                fontsize=9, color=ps.INK_2)
    ax.xaxis.set_major_formatter(ps.pct)
    ax.set_xlabel("Attainment (revenue / target, capped at 300%)")
    ax.set_title(f"Partner target attainment, {last_q}")
    r.chart(ps.save(fig, CHARTS / "07_attainment_distribution.png"), "Attainment distribution")
    r.takeaway(f"Aggregate attainment runs {tbl['revenue_vs_target'].min():.0%}-{tbl['revenue_vs_target'].max():.0%}, "
               f"but only about {tbl['pct_partners_hit'].mean():.0%} of partners hit target in a typical quarter.")


def churn_signals(t: dict, r: Report) -> None:
    r.section("10. Churn signals (no labels given)")
    pq = t["partner_quarter"]
    rev = quarter_pivot(pq)
    qs = sorted(rev.columns)
    tiers = t["master"].set_index("partner_id")["tier"]

    rows = []
    for i in range(4, len(qs)):
        prev = rev[qs[i - 4:i]]
        trailing = prev.mean(axis=1)
        eligible = (prev.notna().sum(axis=1) >= 2) & (trailing > 0) & rev[qs[i]].notna()
        leaver = eligible & (rev[qs[i]] < TOKEN_SHARE * trailing)
        recovered = (leaver & (rev[qs[i + 1]] >= RECOVER_SHARE * trailing)).sum() if i + 1 < len(qs) else np.nan
        rows.append({
            "quarter": qs[i], "eligible_partners": int(eligible.sum()), "naive_leavers": int(leaver.sum()),
            "rate": leaver.sum() / eligible.sum(), "recovered_next_q": recovered,
            "business_tier_share": (tiers.reindex(leaver[leaver].index) == "Business").mean(),
        })
    tbl = pd.DataFrame(rows)
    tbl["recovery_rate"] = tbl["recovered_next_q"] / tbl["naive_leavers"]
    show = tbl.copy()
    for c in ["rate", "recovery_rate", "business_tier_share"]:
        show[c] = show[c].map(fp)
    show["recovered_next_q"] = show["recovered_next_q"].map(lambda v: "-" if pd.isna(v) else int(v))
    r.text(f"Naive label: quarter revenue < {TOKEN_SHARE:.0%} of the trailing 4-quarter average. "
           f"Recovered = back above {RECOVER_SHARE:.0%} of that average the next quarter.")
    r.table(show, "naive_churn_label", raw=tbl)

    fig, ax = ps.plt.subplots(figsize=(8, 3.8))
    known = tbl["recovered_next_q"].notna()
    stayed = tbl["naive_leavers"] - tbl["recovered_next_q"].fillna(0)
    ax.bar(tbl.loc[known, "quarter"], tbl.loc[known, "recovered_next_q"], color=ps.SERIES[0],
           label="Recovered next quarter", edgecolor=ps.SURFACE, linewidth=2)
    ax.bar(tbl.loc[known, "quarter"], stayed[known], bottom=tbl.loc[known, "recovered_next_q"],
           color=ps.SERIES[1], label="Stayed low", edgecolor=ps.SURFACE, linewidth=2)
    ax.bar(tbl.loc[~known, "quarter"], tbl.loc[~known, "naive_leavers"], color=ps.NEUTRAL,
           label="Outcome not known yet", edgecolor=ps.SURFACE, linewidth=2)
    ax.set_title("Partners whose revenue collapsed in a quarter: most come back")
    ax.legend(loc="upper left")
    r.chart(ps.save(fig, CHARTS / "08_naive_churn_recovery.png"), "Naive churn recovery")

    summ = t["partner_summary"]
    last_q = qs[-1]
    dormant = summ[(summ["lines_last_q"] == 0) & (summ["onboarded_date"] < pd.Period(last_q.replace("-", ""), "Q").start_time)]
    last_month = summ["last_order"].dt.to_period("M")
    before_june = last_month[last_month < last_month.max()].value_counts().sort_index()
    r.text(f"- Partners with no orders in {last_q} (onboarded before it): **{len(dormant)}**")
    r.text(f"- Partners whose last order was before June 2026: **{before_june.sum()}**")
    r.text(f"- Partners still ramping (onboarded < 12 months ago): **{int(summ['is_ramping'].sum())}**")
    r.text(f"- Partners silent for more than 3x their usual gap between orders: "
           f"**{int((summ['recency_vs_typical_gap'] > 3).sum())}**")
    lm = before_june.rename_axis("last_order_month").reset_index(name="partners")
    lm["last_order_month"] = lm["last_order_month"].astype(str)
    r.table(lm, "last_order_month")

    fig, ax = ps.plt.subplots(figsize=(8, 3.4))
    ax.bar(lm["last_order_month"], lm["partners"], color=ps.SERIES[0], edgecolor=ps.SURFACE, linewidth=2)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Partners by month of last order (excluding June 2026)")
    r.chart(ps.save(fig, CHARTS / "09_last_order_month.png"), "Last order month")

    rr = tbl["recovery_rate"].dropna()
    r.takeaway(f"A naive revenue-drop label is noisy: {rr.min():.0%}-{rr.max():.0%} of partners whose quarter "
               f"collapsed recovered the next quarter, and they are mostly small Business-tier resellers. "
               f"Use a stricter label (low for 2 quarters, or months of silence after a regular rhythm).")
    r.takeaway(f"{len(dormant)} partners already had zero orders in {last_q}; {before_june.sum()} last ordered "
               f"before June 2026. These are the strongest at-risk candidates.")


def seasonality_baselines(t: dict, r: Report) -> None:
    r.section("11. Seasonality and naive Q3 2026 baselines")
    rq = t["region_quarter"].pivot(index="quarter", columns="region", values="revenue_usd")[REGIONS]
    rq["ALL"] = rq.sum(axis=1)
    cols = REGIONS + ["ALL"]
    ratio24 = rq.loc["2024-Q3"] / rq.loc["2024-Q2"]
    ratio25 = rq.loc["2025-Q3"] / rq.loc["2025-Q2"]
    yoy_h1 = (rq.loc["2026-Q1"] + rq.loc["2026-Q2"]) / (rq.loc["2025-Q1"] + rq.loc["2025-Q2"])
    a = rq.loc["2026-Q2"] * ratio25
    b = rq.loc["2025-Q3"] * yoy_h1
    for est in (a, b):  # ALL as the sum of regions so the baselines stay coherent
        est["ALL"] = est[REGIONS].sum()
    tbl = pd.DataFrame({"Q3/Q2 2024": ratio24, "Q3/Q2 2025": ratio25, "H1 YoY": yoy_h1,
                        "A: Q2'26 x Q3/Q2'25": a, "B: Q3'25 x H1 YoY": b, "mean of A, B": (a + b) / 2}).loc[cols]
    raw = tbl.reset_index().rename(columns={"index": "region"})
    show = raw.copy()
    for c in ["Q3/Q2 2024", "Q3/Q2 2025", "H1 YoY"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    for c in ["A: Q2'26 x Q3/Q2'25", "B: Q3'25 x H1 YoY", "mean of A, B"]:
        show[c] = show[c].map(fm)
    r.text("Sanity anchors for the forecast model (outlier excluded) - not the forecast itself.")
    r.table(show, "q3_baselines", raw=raw)
    r.takeaway(f"Two naive Q3 2026 baselines give {fm(a['ALL'])} and {fm(b['ALL'])} in total; any model "
               f"far from about {fm((a['ALL'] + b['ALL']) / 2)} needs a reason.")


# ---------------------------------------------------------------- main

def run(data_dir: Path | str = DATA_DIR) -> Path:
    for d in (CHARTS, TABLES):
        d.mkdir(parents=True, exist_ok=True)
    tables = build_all(data_dir)
    r = Report()
    for step in (data_quality, headline_trend, outlier_effect, dimension_breakdowns, growth_illusion,
                 cohorts, concentration, price_vs_volume, attainment, churn_signals, seasonality_baselines):
        step(tables, r)

    header = ["# Partner channel EDA", "",
              f"Data: `{Path(data_dir).resolve()}` - generated by `python -m src.eda`.", ""]
    report = OUT_DIR / "eda_report.md"
    r.write(report, header)
    print("\n=== Key takeaways ===")
    for tk in r.takeaways:
        print(f"- {tk}")
    print(f"\nReport: {report}\nCharts: {CHARTS}\nTables: {TABLES}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"folder with the 3 CSVs (default: {DATA_DIR})")
    run(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
