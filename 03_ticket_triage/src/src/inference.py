"""Inference on a bundle written by src/bundle.py.

    from inference import Triage
    tri = Triage.load("models/bundle_v3")
    pred = tri.predict(df)              # single tickets / interactive: cap=None (abstain iff savings > 0)
    pred = tri.predict(df, cap=0.15)    # batch: at most floor(0.15 * N) abstains, guards included
    sub = to_submission(pred)

df needs subject, body, product, channel (ticket_id optional). Output: one row per ticket with labels,
calibrated probabilities for all four targets, p_correct, expected route / abstain cost, savings,
escalation flag, guard flags, abstain (0/1), decision and a reason string.
"""
import json
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:  # pickled pipelines reference top-level modules (data, experiments)
    sys.path.insert(0, str(SRC))

from calibration import apply_temperature  # noqa: E402
from correctness import CAT_COLS, build_features  # noqa: E402
from guards import DESCRIPTIONS, flag_lists, guards  # noqa: E402
from novelty import novelty, vectorize  # noqa: E402
from routing import cap_count, expected_costs, top_k_mask  # noqa: E402

TARGETS = ["category", "priority", "assigned_team", "sentiment"]
REQUIRED = ["subject", "body", "product", "channel"]
SUBMISSION_COLS = ["ticket_id", "category", "priority", "assigned_team", "sentiment", "abstain"]
CHECK_PACKAGES = {"scikit-learn": "scikit-learn", "numpy": "numpy", "joblib": "joblib"}


def escalation(priority, sentiment):
    priority, sentiment = np.asarray(priority), np.asarray(sentiment)
    return (sentiment == "Angry") | ((sentiment == "Frustrated") & np.isin(priority, ["P1", "P2"]))


def priority_label(classes, p, threshold):
    """P1 if P(P1) >= threshold, else argmax over the other classes (same rule as p1_threshold.label)."""
    i1 = classes.index("P1")
    rest = p.copy()
    rest[:, i1] = -1
    pred = np.array(classes, dtype=object)[rest.argmax(1)]
    pred[p[:, i1] >= threshold] = "P1"
    return pred


def onehot(values, vocab) -> np.ndarray:
    """One-hot rows in vocab order; unseen values get an all-zero row."""
    idx = {v: i for i, v in enumerate(vocab)}
    m = np.zeros((len(values), len(vocab)))
    for r, v in enumerate(values):
        if v in idx:
            m[r, idx[v]] = 1
    return m


class Triage:
    def __init__(self, bundle_dir: Path, manifest: dict, models: dict):
        self.bundle_dir, self.manifest, self.models = bundle_dir, manifest, models
        self.classes = manifest["classes"]
        self.T = manifest["temperatures"]
        self.W, self.A = manifest["cost_matrix"]["misroute"], manifest["cost_matrix"]["abstain"]
        self.cap = manifest["max_abstain_rate"]
        self.threshold = manifest["p1_threshold"]
        self.known_products = manifest["known_products"]
        self.min_body_chars = manifest.get("guards", {}).get("min_body_chars", 20)
        self._product_lookup = {self._norm(p): p for p in self.known_products}
        nov = manifest.get("novelty")  # added by Triage.load when novelty.npz/json exist
        self.novelty_threshold = nov["threshold"] if nov else None

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, bundle_dir) -> "Triage":
        bundle_dir = Path(bundle_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        for pkg, dist in CHECK_PACKAGES.items():
            built = manifest.get("package_versions", {}).get(pkg)
            try:
                here = version(dist)
            except PackageNotFoundError:
                here = None
            if built and here != built:
                warnings.warn(f"bundle built with {pkg}=={built}, running {here}; predictions may differ")
        c = manifest["components"]
        inputs = c["assigned_team"]["inputs"]
        models = {"category": joblib.load(bundle_dir / c["category"]["file"]),
                  "priority": joblib.load(bundle_dir / c["priority"]["file"]),
                  "sentiment": joblib.load(bundle_dir / c["sentiment"]["file"]),
                  "team_stacker": joblib.load(bundle_dir / c["assigned_team"]["file"]),
                  "team_base_team": joblib.load(bundle_dir / inputs["assigned_team"]["file"]),
                  "correctness": joblib.load(bundle_dir / c["correctness"]["file"])}
        # the stacker's category input is usually the category model itself (same run): load it once
        models["team_base_category"] = (models["category"] if inputs["category"]["run"] == c["category"]["run"]
                                        else joblib.load(bundle_dir / inputs["category"]["file"]))
        if (bundle_dir / "novelty.npz").exists() and (bundle_dir / "novelty.json").exists():
            import scipy.sparse as sp
            models["novelty_train_T"] = sp.load_npz(bundle_dir / "novelty.npz").T.tocsc()
            manifest = {**manifest, "novelty": json.loads((bundle_dir / "novelty.json").read_text())}
        return cls(bundle_dir, manifest, models)

    # ---------------------------------------------------------------- input handling
    @staticmethod
    def _norm(s) -> str:
        return " ".join(str(s).split()).casefold()

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(missing)}")
        out = df.reset_index(drop=True).copy()
        if "ticket_id" not in out.columns:
            out["ticket_id"] = [f"T-{i + 1:06d}" for i in range(len(out))]
        out["ticket_id"] = out["ticket_id"].astype(str)
        for col in ("subject", "body"):
            out[col] = out[col].fillna("").astype(str)
        # canonical product spelling for known products ("greenlake " -> "GreenLake"); unknown kept as typed
        prod = out["product"].fillna("").astype(str)
        out["product"] = [self._product_lookup.get(self._norm(p), p.strip()) for p in prod]
        out["channel"] = out["channel"].fillna("").astype(str).str.strip().str.lower()
        return out

    def _proba(self, model, X, target):
        p = model.predict_proba(X)
        return p[:, [list(model.classes_).index(c) for c in self.classes[target]]]

    # ---------------------------------------------------------------- scoring
    def stack_features(self, p_cat, p_team, df) -> np.ndarray:
        """experiments.stack_features layout; unknown product/channel -> all-zero one-hot."""
        return np.hstack([p_cat, p_team, onehot(df["product"], self.manifest["products"]),
                          onehot(df["channel"], self.manifest["channels"])])

    def raw_probabilities(self, df: pd.DataFrame) -> dict:
        m = self.models
        base_cat = self._proba(m["team_base_category"], df, "category")
        base_team = self._proba(m["team_base_team"], df, "assigned_team")
        X_stack = self.stack_features(base_cat, base_team, df)
        cat = base_cat if m["category"] is m["team_base_category"] else self._proba(m["category"], df, "category")
        return {"category": cat,
                "priority": self._proba(m["priority"], df, "priority"),
                "assigned_team": self._proba(m["team_stacker"], X_stack, "assigned_team"),
                "sentiment": self._proba(m["sentiment"], df, "sentiment")}

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Labels, calibrated probabilities, p_correct, expected costs and guard flags (no decision yet)."""
        df = self.prepare(df)
        raw = self.raw_probabilities(df)
        cal = {t: apply_temperature(raw[t], self.T[t]) for t in TARGETS}
        C = self.classes
        out = pd.DataFrame({"ticket_id": df["ticket_id"]})
        out["category"] = np.array(C["category"])[cal["category"].argmax(1)]
        out["priority"] = priority_label(C["priority"], cal["priority"], self.threshold)
        out["assigned_team"] = np.array(C["assigned_team"])[cal["assigned_team"].argmax(1)]
        out["sentiment"] = np.array(C["sentiment"])[cal["sentiment"].argmax(1)]
        for t in TARGETS:
            for i, c in enumerate(C[t]):
                out[f"proba_{t}_{c}"] = cal[t][:, i]

        frame = out[["ticket_id"] + [c for c in out.columns if c.startswith("proba_")]].copy()
        frame["ticket_id"] = [f"row{i}" for i in range(len(df))]  # unique keys for build_features' lookup
        meta = pd.DataFrame({"ticket_id": frame["ticket_id"], **{c: df[c].to_numpy() for c in CAT_COLS}})
        X_corr, _ = build_features(frame, meta)
        out["p_correct"] = self.models["correctness"].predict_proba(X_corr)[:, 1]

        ec = expected_costs(cal["assigned_team"], cal["priority"], C["priority"], self.W, self.A,
                            p_ok=out["p_correct"].to_numpy())
        out["expected_route_cost"] = ec["e_route"].to_numpy()
        out["expected_abstain_cost"] = ec["e_abstain"].to_numpy()
        out["savings"] = ec["savings"].to_numpy()
        out["escalation"] = escalation(out["priority"], out["sentiment"])

        nov = None
        if "novelty_train_T" in self.models:
            nov = novelty(vectorize(self.models["category"], df), self.models["novelty_train_T"])
            out["novelty"] = nov
        flags = guards(df, self.known_products, self.min_body_chars, nov, self.novelty_threshold)
        out["guard_flags"] = flag_lists(flags)
        out["guarded"] = flags.any(axis=1).to_numpy()
        return out

    # ---------------------------------------------------------------- decision
    def decide(self, scored: pd.DataFrame, cap: float | None = None, use_guards: bool = True) -> pd.DataFrame:
        """Policy f (routing.py): abstain on positive expected savings; with a cap, only the top floor(cap*N)
        by savings. Guard-flagged tickets always abstain and use up cap slots first.
        Accepts a raw ticket frame (it is scored first) or the output of score()."""
        if "savings" not in scored.columns:
            scored = self.score(scored)
        out = scored.copy()
        n = len(out)
        savings = out["savings"].to_numpy()
        guarded = out["guarded"].to_numpy() if use_guards else np.zeros(n, bool)
        notes = []
        if cap is None:
            policy = savings > 0
            slots = None
        else:
            slots = cap_count(cap, n)
            remaining = max(slots - int(guarded.sum()), 0)
            if guarded.sum() > slots:
                notes.append(f"{int(guarded.sum())} guard-flagged tickets exceed the {slots} hand-off slots "
                             f"({cap:.0%} cap); all of them still abstain")
            policy = top_k_mask(np.where(guarded, -np.inf, savings), remaining, positive_only=True)
        abstain = guarded | policy
        out["abstain"] = abstain.astype(int)
        out["decision"] = np.where(abstain, "abstain", "route")
        out["reason"] = [self._reason(r, g, a, cap) for r, g, a in
                         zip(out.itertuples(index=False), guarded, abstain)]
        out.attrs.update({"cap": cap, "slots": slots, "n_abstain": int(abstain.sum()),
                          "n_guarded": int(guarded.sum()), "notes": notes})
        return out

    def predict(self, df: pd.DataFrame, cap: float | None = None) -> pd.DataFrame:
        return self.decide(self.score(df), cap)

    # ---------------------------------------------------------------- explanations
    def _priority_phrase(self, r) -> str:
        p1, p2 = r.proba_priority_P1, r.proba_priority_P2
        if p1 >= self.threshold:
            return f"likely P1 (P(P1)={p1:.2f})"
        return "likely P1/P2" if p1 + p2 >= 0.5 else "likely P3/P4"

    def _reason(self, r, guarded, abstain, cap) -> str:
        costs = (f"expected misroute cost {r.expected_route_cost:.2f} "
                 f"{'>' if r.savings > 0 else '<='} hand-off cost {r.expected_abstain_cost:.2f}")
        ctx = f"{self._priority_phrase(r)}, team confidence {r.p_correct:.2f}"
        if guarded:
            names = "; ".join(f"{g} ({DESCRIPTIONS.get(g, g)})" for g in r.guard_flags)
            return f"Abstain: guard {names}. Model would have routed to {r.assigned_team} ({ctx})"
        if abstain:
            return f"Abstain: {ctx}; {costs}"
        if r.savings > 0:  # positive savings but outside the batch cap
            return (f"Route to {r.assigned_team}: {ctx}; {costs}, but the hand-off desk is full "
                    f"({cap:.0%} cap) and other tickets save more")
        return f"Route to {r.assigned_team}: {ctx}; {costs}"


# ---------------------------------------------------------------- output helpers
def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    """Submission column format (category/team kept filled on abstains, which the format allows)."""
    return pred[SUBMISSION_COLS].copy()


def to_records(pred: pd.DataFrame, classes: dict) -> list:
    """JSON-friendly per-ticket dicts with probabilities grouped by target."""
    recs = []
    for r in pred.to_dict(orient="records"):
        rec = {k: r[k] for k in ("ticket_id", "category", "priority", "assigned_team", "sentiment")}
        rec["probabilities"] = {t: {c: round(float(r[f"proba_{t}_{c}"]), 6) for c in classes[t]} for t in TARGETS}
        rec.update({"p_correct": round(float(r["p_correct"]), 6),
                    "expected_route_cost": round(float(r["expected_route_cost"]), 6),
                    "expected_abstain_cost": round(float(r["expected_abstain_cost"]), 6),
                    "savings": round(float(r["savings"]), 6),
                    "escalation": bool(r["escalation"]),
                    "guard_flags": list(r["guard_flags"]),
                    "abstain": int(r["abstain"]), "decision": r["decision"], "reason": r["reason"]})
        recs.append(rec)
    return recs
