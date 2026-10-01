"""Data loading, persistent train/val/test split, and text construction."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TICKETS = ROOT / "data" / "tickets.csv"
SPLIT_DIR = ROOT / "splits"
SPLITS = ("train", "val", "test")
SIZES = {"train": 21000, "val": 4500, "test": 4500}
TARGETS = ["category", "priority", "assigned_team", "sentiment"]
CHECK_COLS = TARGETS + ["product", "channel"]
SEED = 42
RARE_STRATUM = 10  # category|priority strata smaller than this fall back to category only


def load_tickets() -> pd.DataFrame:
    return pd.read_csv(TICKETS)


def _strat_key(df: pd.DataFrame) -> pd.Series:
    # Rare strata (e.g. Billing|P1, n=3) are merged into their category's largest stratum.
    key = df["category"] + "|" + df["priority"]
    counts = key.value_counts()
    largest = {k.split("|")[0]: k for k in counts.index[::-1]}  # last write = largest per category
    rare = key.map(counts) < RARE_STRATUM
    return key.where(~rare, df["category"].map(largest))


def create_splits(df: pd.DataFrame) -> None:
    key = _strat_key(df)
    train, rest = train_test_split(df, test_size=0.30, stratify=key, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=key.loc[rest.index], random_state=SEED)
    SPLIT_DIR.mkdir(exist_ok=True)
    for name, part in zip(SPLITS, (train, val, test)):
        part[["ticket_id"]].to_csv(SPLIT_DIR / f"{name}_ids.csv", index=False)


def load_splits(df: pd.DataFrame) -> dict:
    """Return {split_name: DataFrame}; creates split files only if none exist."""
    files = {s: SPLIT_DIR / f"{s}_ids.csv" for s in SPLITS}
    if not any(f.exists() for f in files.values()):
        print("Split files not found -> creating them (random_state=42).")
        create_splits(df)
    elif not all(f.exists() for f in files.values()):
        raise RuntimeError("Some split files are missing; refusing to regenerate. Investigate splits/.")
    ids = {s: pd.read_csv(f)["ticket_id"] for s, f in files.items()}

    all_ids = pd.concat(ids.values())
    assert all_ids.is_unique, "overlap / duplicate IDs across splits"
    assert set(all_ids) == set(df["ticket_id"]), "split IDs do not match tickets.csv"
    assert len(all_ids) == len(df) == 30000
    for s in SPLITS:
        assert len(ids[s]) == SIZES[s], f"{s} has {len(ids[s])} rows, expected {SIZES[s]}"

    indexed = df.set_index("ticket_id")
    return {s: indexed.loc[ids[s]].reset_index() for s in SPLITS}


def representativeness_report(df: pd.DataFrame, parts: dict, flag_pp: float = 2.0) -> None:
    for col in CHECK_COLS:
        table = pd.DataFrame({"full": df[col].value_counts(normalize=True) * 100})
        for s in SPLITS:
            table[s] = parts[s][col].value_counts(normalize=True) * 100
        missing = table[table[list(SPLITS)].isna().any(axis=1)].index.tolist()
        table = table.fillna(0)
        print(f"\n{col}:")
        print(table.round(2).to_string(float_format=lambda v: f"{v:.2f}%"))
        print("max deviation (pp):", ", ".join(
            f"{s}: {(table[s] - table['full']).abs().max():.2f}" for s in SPLITS))
        flagged = [(c, s) for s in SPLITS for c in table.index if abs(table.at[c, s] - table.at[c, "full"]) > flag_pp]
        if flagged:
            print(f"  FLAG >{flag_pp}pp:", flagged)
        if missing:
            print("  FLAG classes missing from a split:", missing)
        else:
            print("  all classes present in all splits")


def build_text(df: pd.DataFrame) -> pd.Series:
    product = df["product"].str.replace(" ", "_", regex=False)
    return ("CHANNEL_" + df["channel"] + " PRODUCT_" + product + " "
            + df["subject"].fillna("") + " " + df["body"].fillna(""))
