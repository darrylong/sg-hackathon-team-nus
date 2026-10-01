"""Input guards: rules that force a ticket to human triage regardless of model confidence.

guards(df, known_products) -> DataFrame with one boolean column per guard (True = flagged).
Add a guard by appending to GUARDS: (name, description, fn(df, ctx) -> bool array).
"""
import re

import numpy as np
import pandas as pd

MIN_BODY_CHARS = 20
# Portal bodies start with the customer's own severity; it carries no information about the issue itself
SEVERITY_PREFIX = re.compile(r"^\s*\[Customer-selected severity:[^\]]*\]\s*", re.I)


def body_content(body: pd.Series) -> pd.Series:
    return body.fillna("").astype(str).str.replace(SEVERITY_PREFIX, "", regex=True).str.strip()


def unknown_product(df: pd.DataFrame, ctx: dict) -> np.ndarray:
    return ~df["product"].fillna("").astype(str).str.strip().isin(ctx["known_products"]).to_numpy()


def short_body(df: pd.DataFrame, ctx: dict) -> np.ndarray:
    return (body_content(df["body"]).str.len() < ctx.get("min_body_chars", MIN_BODY_CHARS)).to_numpy()


def novel_topic(df: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Novelty (1 - max cosine similarity to any TRAIN ticket, src/novelty.py) at or above the bundle threshold.
    Inactive (all False) when no novelty scores are supplied."""
    nov, t = ctx.get("novelty"), ctx.get("novelty_threshold")
    if nov is None or t is None:
        return np.zeros(len(df), bool)
    return np.asarray(nov) >= t


GUARDS = [
    ("unknown_product", "product not seen in training", unknown_product),
    ("short_body", f"empty or very short body (< {MIN_BODY_CHARS} chars)", short_body),
    ("novel_topic", "topic unlike any training ticket", novel_topic),
]
DESCRIPTIONS = {name: desc for name, desc, _ in GUARDS}


def guards(df: pd.DataFrame, known_products, min_body_chars: int = MIN_BODY_CHARS,
           novelty=None, novelty_threshold=None) -> pd.DataFrame:
    ctx = {"known_products": set(known_products), "min_body_chars": min_body_chars,
           "novelty": novelty, "novelty_threshold": novelty_threshold}
    return pd.DataFrame({name: fn(df, ctx) for name, _, fn in GUARDS}, index=df.index)


def flag_lists(flags: pd.DataFrame) -> list:
    """Per-row list of the guard names that fired."""
    names = np.array(flags.columns)
    return [list(names[row]) for row in flags.to_numpy()]
