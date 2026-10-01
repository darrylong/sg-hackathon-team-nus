"""Deterministic impact-factor tagger for ticket bodies (round 2D).

Factors: scope, environment, workaround, severity_tag, urgency_claim, deescalation.
Matching: body is lower-cased with chat slang expanded; template regexes are searched first (exact,
case-insensitive); factors with no exact hit fall back to rapidfuzz partial_ratio >= 90 of each plain
template against each sentence (only sentences at least 85% of the template's length, so short
fragments cannot match inside a long template). Templates come from the TRAIN template search used by
src/probe.py (see outputs/probe/probe_report.txt).
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from experiments import parse_severity

FUZZ_CUTOFF = 90
MIN_FUZZY_LEN = 15

# Each level: list of (regex, plain_text_or_None). plain text enables the fuzzy fallback.
_N = r"\d+"
TEMPLATES = {
    "scope": {
        "wide": ["all customers on the platform see this", "every branch is affected", "the whole site is affected",
                 rf"all {_N} users in the company are impacted"],
        "few": [rf"about {_N} users are affected", "a few teams are impacted", "several vms on it are affected",
                rf"\b{_N} hosts see this"],
        "single": ["only one user is affected", "just this one server", "single device, nothing else",
                   "it's only me as far as i can tell", r"\bonly [a-z0-9-]*\d+ is affected"],
        "none": ["no impact on users at the moment", "only the staging cluster is affected"],
    },
    "environment": {
        "production": ["this box runs our production erp", "it hosts our core banking vms", "production workloads run on it",
                       "it is part of the live payments cluster", "it serves the live e-commerce site",
                       "customer-facing services run on this system", r"\bthis is prod\b",
                       r"this is the primary system at [^.]+"],
        "golive": ["go-live is tomorrow morning"],
        "nonprod": ["it is in our test environment", "this is a lab unit", "spare unit in the lab, not in service yet",
                    "this is the dr standby, not active right now", "only the staging cluster is affected"],
    },
    "workaround": {
        "failover": ["traffic failed over to the secondary, so we're limping along"],
        "slow": ["there is a manual workaround but it is slow", "still no fix, and the workaround is painful"],
        "have": ["we have a workaround for now"],
    },
    "deescalation": {
        "yes": [r"(?<![a-z])not urgent(?: at all)?(?![a-z])", "low priority, just curious",
                "not a big deal, just flagging it", "our manager flagged it as urgent, but honestly it can wait"],
    },
    "urgency_claim": {
        "yes": [r"(?<!not )(?<![a-z])urgent\s*[.!]", "need this fixed asap", "marking this urgent",
                "please treat as urgent"],
    },
}
# precedence when several levels of one factor match (first wins)
DEFAULTS = {"scope": "unknown", "environment": "unknown", "workaround": "none_mentioned",
            "deescalation": "no", "urgency_claim": "no"}
LEVELS = {
    "scope": ["wide", "few", "single", "none", "unknown"],
    "environment": ["production", "golive", "nonprod", "unknown"],
    "workaround": ["failover", "slow", "have", "none_mentioned"],
    "severity_tag": ["none", "low", "medium", "high", "critical"],
    "urgency_claim": ["yes", "no"],
    "deescalation": ["yes", "no"],
}
FACTORS = list(LEVELS)

_SLANG = [(r"\br\b", "are"), (r"\btmrw\b", "tomorrow"), (r"\bu\b", "you"), (r"\bw/", "with "), (r"\bpls\b", "please")]
_SPLIT = re.compile(r"(?<=[.!?])\s+|\]\s*")


def _is_regex(t):
    return any(ch in t for ch in "\\[(?")


def _compile(t):
    return re.compile(t if _is_regex(t) else re.escape(t).replace(r"\ ", r"\s+"))


_COMPILED = {f: {lvl: [(_compile(t), None if _is_regex(t) else t) for t in ts] for lvl, ts in levels.items()}
             for f, levels in TEMPLATES.items()}


def normalise(body: str) -> str:
    s = body.lower()
    for pat, rep in _SLANG:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s)


def tag(body: str) -> dict:
    s = normalise(body)
    out = {}
    for f, levels in _COMPILED.items():
        hit = next((lvl for lvl, pats in levels.items() if any(p.search(s) for p, _ in pats)), None)
        if hit is None:
            sents = [x.strip(" .!?") for x in _SPLIT.split(s) if x.strip()]
            for lvl, pats in levels.items():
                plains = [t for _, t in pats if t and len(t) >= MIN_FUZZY_LEN]
                if any(len(x) >= 0.85 * len(t) and fuzz.partial_ratio(t, x) >= FUZZ_CUTOFF
                       for t in plains for x in sents):
                    hit = lvl
                    break
        out[f] = hit or DEFAULTS[f]
    sev = parse_severity(body)
    out["severity_tag"] = sev.lower() if sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL") else "none"
    return {f: out[f] for f in FACTORS}


def tag_frame(bodies) -> pd.DataFrame:
    return pd.DataFrame([tag(b) for b in bodies])


def one_hot(tags: pd.DataFrame) -> np.ndarray:
    return np.hstack([(tags[f].to_numpy()[:, None] == np.array(LEVELS[f])[None, :]).astype(np.float32)
                      for f in FACTORS])


def one_hot_names():
    return [f"{f}={lvl}" for f in FACTORS for lvl in LEVELS[f]]
