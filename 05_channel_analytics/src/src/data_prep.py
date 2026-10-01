"""Load, clean, merge and aggregate the partner channel data.

Run as a script to write the processed tables to outputs/processed/*.parquet:

    python -m src.data_prep [--data-dir PATH] [--keep-outliers]

Or import from other modules (EDA, models, the API backend):

    from src.data_prep import build_all
    tables = build_all()
    tables["partner_quarter"].head()
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CHANNEL_DATA_DIR", ROOT.parent / "05_channel_analytics" / "data"))
PROCESSED_DIR = ROOT / "outputs" / "processed"

DATA_START = pd.Timestamp("2024-01-01")
DATA_END = pd.Timestamp("2026-06-30")
AS_OF = DATA_END  # snapshot date for partner-level features

REGIONS = ["AMS", "APJ", "EMEA"]
TIERS = ["Platinum", "Gold", "Silver", "Business"]
PRODUCTS = ["ProLiant", "Synergy", "Storage", "Aruba", "GreenLake"]
PARTNER_TYPES = ["Distributor", "Reseller", "Service Provider"]
CATEGORIES = {
    "region": REGIONS,
    "tier": TIERS,
    "product_family": PRODUCTS,
    "partner_type": PARTNER_TYPES,
}

COHORT_EXISTING = "Onboarded pre-2024"
COHORT_NEW = "Onboarded 2024+"

# A line is an outlier if its revenue exceeds this multiple of the 99th
# percentile line for its product family. Catches the $37.8M GreenLake line
# (qty 12,800) without touching normal large Platinum deals.
OUTLIER_MULTIPLE = 20
# Partners onboarded within this many months of AS_OF are still ramping.
RAMPING_MONTHS = 12


# ---------------------------------------------------------------- helpers

def quarter_label(dates: pd.Series) -> pd.Series:
    """Timestamps -> 'YYYY-Qn', the format used in targets.csv."""
    return dates.dt.year.astype(str) + "-Q" + dates.dt.quarter.astype(str)


def month_start(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp()


def _categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known dimension columns to ordered categoricals, failing loudly on unknown values."""
    for col, cats in CATEGORIES.items():
        if col not in df.columns:
            continue
        unknown = set(df[col].dropna().unique()) - set(cats)
        if unknown:
            raise ValueError(f"Unexpected values in {col}: {sorted(unknown)}")
        df[col] = pd.Categorical(df[col], categories=cats, ordered=True)
    return df


# ---------------------------------------------------------------- load

def load_raw(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read the three CSVs with parsed dates and categorical dimensions."""
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "partner_sales.csv", parse_dates=["order_date"])
    master = pd.read_csv(data_dir / "partner_master.csv", parse_dates=["onboarded_date"])
    targets = pd.read_csv(data_dir / "targets.csv")
    return {
        "sales": _categorize(sales),
        "master": _categorize(master),
        "targets": targets,
    }


# ---------------------------------------------------------------- clean + merge

def flag_outlier_lines(sales: pd.DataFrame, multiple: float = OUTLIER_MULTIPLE) -> pd.Series:
    p99 = sales.groupby("product_family", observed=True)["revenue_usd"].transform(lambda x: x.quantile(0.99))
    return sales["revenue_usd"] > multiple * p99


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Add calendar columns, margin dollars, unit price and an outlier flag."""
    s = sales.copy()
    s["month"] = month_start(s["order_date"])
    s["quarter"] = quarter_label(s["order_date"])
    s["year"] = s["order_date"].dt.year
    s["margin_usd"] = s["revenue_usd"] * s["margin_pct"] / 100
    s["unit_price"] = s["revenue_usd"] / s["quantity"]
    s["is_outlier"] = flag_outlier_lines(s)
    return s


def clean_targets(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.copy()
    period = pd.PeriodIndex(t["quarter"].str.replace("-", ""), freq="Q")
    t["quarter_start"] = period.start_time
    t["quarter_end"] = period.end_time.normalize()
    return t


def merge_sales_master(sales: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Attach partner attributes not already on the sales lines."""
    attrs = master[["partner_id", "partner_type", "country", "onboarded_date"]]
    out = sales.merge(attrs, on="partner_id", how="left", validate="many_to_one")
    out["cohort"] = np.where(out["onboarded_date"] < DATA_START, COHORT_EXISTING, COHORT_NEW)
    out["tenure_months"] = ((out["order_date"] - out["onboarded_date"]).dt.days / 30.44).round(1)
    return out


# ---------------------------------------------------------------- aggregate

def aggregate(sales: pd.DataFrame, by: list[str] | str) -> pd.DataFrame:
    """Revenue, margin, quantity, line count and active partners for any grouping."""
    out = (
        sales.groupby(by, observed=True)
        .agg(
            revenue_usd=("revenue_usd", "sum"),
            margin_usd=("margin_usd", "sum"),
            quantity=("quantity", "sum"),
            order_lines=("revenue_usd", "size"),
            partners=("partner_id", "nunique"),
        )
        .reset_index()
    )
    out["margin_pct"] = out["margin_usd"] / out["revenue_usd"] * 100
    return out


def build_partner_month(sales: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """One row per partner per month from onboarding (or Jan 2024) to Jun 2026, zero-filled."""
    months = pd.DataFrame({"month": pd.date_range(DATA_START, DATA_END, freq="MS")})
    grid = master[["partner_id", "region", "tier", "partner_type", "onboarded_date"]].merge(months, how="cross")
    grid = grid[grid["month"] >= month_start(grid["onboarded_date"])]

    agg = (
        sales.groupby(["partner_id", "month"])
        .agg(
            revenue_usd=("revenue_usd", "sum"),
            margin_usd=("margin_usd", "sum"),
            quantity=("quantity", "sum"),
            order_lines=("revenue_usd", "size"),
            n_products=("product_family", "nunique"),
        )
        .reset_index()
    )
    pm = grid.merge(agg, on=["partner_id", "month"], how="left")
    num_cols = ["revenue_usd", "margin_usd", "quantity", "order_lines", "n_products"]
    pm[num_cols] = pm[num_cols].fillna(0)
    pm[["quantity", "order_lines", "n_products"]] = pm[["quantity", "order_lines", "n_products"]].astype(int)
    pm["quarter"] = quarter_label(pm["month"])
    pm["active"] = pm["order_lines"] > 0
    return pm.drop(columns="onboarded_date").reset_index(drop=True)


def build_partner_quarter(sales: pd.DataFrame, partner_month: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """One row per partner per quarter with revenue, margin, breadth, target and attainment."""
    pq = (
        partner_month.groupby(["partner_id", "region", "tier", "partner_type", "quarter"], observed=True)
        .agg(
            revenue_usd=("revenue_usd", "sum"),
            margin_usd=("margin_usd", "sum"),
            quantity=("quantity", "sum"),
            order_lines=("order_lines", "sum"),
            active_months=("active", "sum"),
        )
        .reset_index()
    )
    breadth = (
        sales.groupby(["partner_id", "quarter"])["product_family"].nunique().rename("n_products").reset_index()
    )
    pq = pq.merge(breadth, on=["partner_id", "quarter"], how="left")
    pq["n_products"] = pq["n_products"].fillna(0).astype(int)
    pq = pq.merge(targets[["partner_id", "quarter", "target_usd"]], on=["partner_id", "quarter"], how="left")
    pq["margin_pct"] = np.where(pq["revenue_usd"] > 0, pq["margin_usd"] / pq["revenue_usd"].where(pq["revenue_usd"] > 0) * 100, np.nan)
    pq["attainment"] = pq["revenue_usd"] / pq["target_usd"]
    return pq.sort_values(["partner_id", "quarter"]).reset_index(drop=True)


def build_partner_summary(sales: pd.DataFrame, partner_quarter: pd.DataFrame, master: pd.DataFrame,
                          as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    """Partner-level snapshot as of `as_of`: recency, order rhythm, momentum and ramping status."""
    order_days = sales[["partner_id", "order_date"]].drop_duplicates().sort_values(["partner_id", "order_date"])
    order_days["gap_days"] = order_days.groupby("partner_id")["order_date"].diff().dt.days
    rhythm = order_days.groupby("partner_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        order_days=("order_date", "size"),
        typical_gap_days=("gap_days", "median"),
    )

    rev = partner_quarter.pivot_table(index="partner_id", columns="quarter", values="revenue_usd", aggfunc="sum")
    lines = partner_quarter.pivot_table(index="partner_id", columns="quarter", values="order_lines", aggfunc="sum")
    quarters = sorted(rev.columns)
    last_q, prev4 = quarters[-1], quarters[-5:-1]

    def half(year: int, h: int) -> pd.Series:
        qs = [f"{year}-Q{q}" for q in ((1, 2) if h == 1 else (3, 4))]
        return rev.reindex(columns=qs).sum(axis=1, min_count=1)

    summary = master.set_index("partner_id")[["partner_name", "partner_type", "tier", "region", "country", "onboarded_date"]].join(rhythm)
    summary["revenue_total"] = rev.sum(axis=1)
    summary["revenue_h1_2025"] = half(2025, 1)
    summary["revenue_h1_2026"] = half(2026, 1)
    summary["growth_h1_yoy"] = summary["revenue_h1_2026"] / summary["revenue_h1_2025"].where(summary["revenue_h1_2025"] > 0) - 1
    summary["revenue_last_q"] = rev[last_q]
    summary["revenue_prev4q_avg"] = rev[prev4].mean(axis=1)  # NaN quarters (pre-onboarding) are skipped
    summary["last_q_vs_prev4q"] = summary["revenue_last_q"] / summary["revenue_prev4q_avg"].where(summary["revenue_prev4q_avg"] > 0)
    summary["lines_last_q"] = lines[last_q]
    summary["lines_prev4q_avg"] = lines[prev4].mean(axis=1)
    summary["days_since_last_order"] = (as_of - summary["last_order"]).dt.days
    summary["recency_vs_typical_gap"] = summary["days_since_last_order"] / summary["typical_gap_days"]
    summary["tenure_months"] = ((as_of - summary["onboarded_date"]).dt.days / 30.44).round(1)
    summary["is_ramping"] = summary["tenure_months"] < RAMPING_MONTHS
    summary["cohort"] = np.where(summary["onboarded_date"] < DATA_START, COHORT_EXISTING, COHORT_NEW)
    return summary.reset_index()


# ---------------------------------------------------------------- pipeline

def build_all(data_dir: Path | str = DATA_DIR, exclude_outliers: bool = True) -> dict[str, pd.DataFrame]:
    """Full pipeline. `sales` keeps every line (with is_outlier); every aggregate drops outliers by default."""
    raw = load_raw(data_dir)
    sales = merge_sales_master(clean_sales(raw["sales"]), raw["master"])
    targets = clean_targets(raw["targets"])
    master = raw["master"]

    base = sales[~sales["is_outlier"]] if exclude_outliers else sales
    partner_month = build_partner_month(base, master)
    partner_quarter = build_partner_quarter(base, partner_month, targets)

    return {
        "sales": sales,
        "master": master,
        "targets": targets,
        "partner_month": partner_month,
        "partner_quarter": partner_quarter,
        "partner_summary": build_partner_summary(base, partner_quarter, master),
        "region_month": aggregate(base, ["region", "month"]),
        "region_quarter": aggregate(base, ["region", "quarter"]),
        "tier_quarter": aggregate(base, ["tier", "quarter"]),
        "product_quarter": aggregate(base, ["product_family", "quarter"]),
        "region_tier_product_quarter": aggregate(base, ["region", "tier", "product_family", "quarter"]),
    }


def save_all(tables: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"  {name:<30} {len(df):>8,} rows -> {out_dir / (name + '.parquet')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DATA_DIR, help=f"folder with the 3 CSVs (default: {DATA_DIR})")
    parser.add_argument("--keep-outliers", action="store_true", help="keep outlier lines in the aggregates")
    args = parser.parse_args()

    tables = build_all(args.data_dir, exclude_outliers=not args.keep_outliers)
    flagged = tables["sales"]["is_outlier"].sum()
    print(f"Loaded {len(tables['sales']):,} sales lines; {flagged} flagged as outlier "
          f"({'kept' if args.keep_outliers else 'excluded from aggregates'}).")
    save_all(tables)


if __name__ == "__main__":
    main()
