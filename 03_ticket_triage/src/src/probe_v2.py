"""Round 2D re-probe: counterfactual template edits on ALL applicable VAL tickets, comparing
E5b_ordinal_char with the winning priority stacker (outputs/round2d/summary.json), if any.

python src/probe_v2.py [--fit-only]  -> outputs/round2d/probe_v2_report.txt, probe_v2.csv
Models are refit on TRAIN (cached under outputs/round2d/probe_models/) and must reproduce each run's VAL
probabilities (max abs diff < 1e-6) before probing. Stacker inputs (E5b probabilities, factors) are
recomputed from the EDITED text.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

import experiments as ex
import factors as F
from calibration import apply_temperature
from data import ROOT, build_text, load_splits, load_tickets
from probe import EDITS
from runs import RUNS_DIR, load_run_frame, proba_matrix

OUT = ROOT / "outputs" / "round2d"
CACHE = OUT / "probe_models"
E5B = "E5b_ordinal_char"
TOL = 1e-6


def T(run, target):
    return json.loads((RUNS_DIR / run / "calibration.json").read_text())[target]["T"]


def cached(name, fit):
    path = CACHE / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    model = fit()
    joblib.dump(model, path)
    return model


def verify(name, got, run, target, frame="val.csv"):
    _, want = proba_matrix(load_run_frame(run, frame), target)
    diff = float(np.abs(got - want).max())
    assert diff < TOL, f"{name} does not reproduce {run} {target}: max abs diff {diff:.2e}"
    return diff


class E5bModel:
    def __init__(self, train):
        self.m = cached("e5b_ordinal_char", lambda: ex.OrdinalLR("word_char").fit(train, train["priority"]))
        self.T = T(E5B, "priority")

    def raw(self, df):
        return self.m.predict_proba(df)

    def cal(self, df):
        return apply_temperature(self.raw(df), self.T)


class StackerModel:
    """Rebuilds round2d.stack_X from text: calibrated E5b priority probs, factors, product, channel [+ ctx]."""

    def __init__(self, run, train, e5b, with_ctx):
        from round2d import hgb
        self.run, self.e5b, self.with_ctx = run, e5b, with_ctx
        if with_ctx:
            self.cat_vec = joblib.load(ROOT / "models" / "baseline" / "tfidf.joblib")
            self.cat_clf = joblib.load(ROOT / "models" / "baseline" / "category_lr.joblib")
            self.sent = cached("e9_sentiment", lambda: ex.tree_selected("sentiment").fit(train, train["sentiment"]))
            self.T_cat, self.T_sent = T("combined_v3", "category"), T("combined_v3", "sentiment")
        po = load_run_frame(E5B, "oof_calibrated.csv")
        co = load_run_frame("combined_v3", "oof_calibrated.csv")
        tags = F.tag_frame(train["body"])
        X_tr = self._X(proba_matrix(po, "priority")[1], tags, train,
                       (proba_matrix(co, "category")[1], proba_matrix(co, "sentiment")[1]) if with_ctx else None)
        self.m = hgb().fit(X_tr, train["priority"].to_numpy())
        self.T = T(run, "priority")

    def _X(self, prio, tags, df, ctx):
        parts = [prio, F.one_hot(tags), np.eye(len(ex.PRODUCTS))[df["product"].map(ex.PRODUCTS.index).to_numpy()],
                 np.eye(len(ex.CHANNELS))[df["channel"].map(ex.CHANNELS.index).to_numpy()]]
        if ctx is not None:
            parts += list(ctx)
        return np.hstack(parts)

    def raw(self, df):
        ctx = None
        if self.with_ctx:
            pc = apply_temperature(self.cat_clf.predict_proba(self.cat_vec.transform(build_text(df))), self.T_cat)
            ps = apply_temperature(self.sent.predict_proba(df), self.T_sent)
            ctx = (pc, ps)
        return self.m.predict_proba(self._X(self.e5b.cal(df), F.tag_frame(df["body"]), df, ctx))

    def cal(self, df):
        return apply_temperature(self.raw(df), self.T)


def effect(model, orig, edited, classes):
    p0, p1 = model.cal(orig), model.cal(edited)
    i1, i2 = classes.index("P1"), classes.index("P2")
    r0, r1 = p0.argmax(1), p1.argmax(1)
    return {"n": len(orig), "mean_dP1": (p1[:, i1] - p0[:, i1]).mean(),
            "mean_dP1P2": (p1[:, [i1, i2]].sum(1) - p0[:, [i1, i2]].sum(1)).mean(),
            "share_up": (r1 < r0).mean(), "share_down": (r1 > r0).mean()}


def main(fit_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    parts = load_splits(load_tickets())
    train, val = parts["train"], parts["val"]
    L = []
    e5b = E5bModel(train)
    L.append(f"E5b refit reproduces {E5B} VAL priority: max abs diff "
             f"{verify('E5b', e5b.raw(val), E5B, 'priority'):.2e}")
    if fit_only:
        print(L[-1])
        return
    models = {E5B: e5b}
    winner = json.loads((OUT / "summary.json").read_text()).get("winner")
    if winner:
        st = StackerModel(winner, train, e5b, with_ctx="ctx" in winner)
        L.append(f"{winner} refit reproduces VAL priority: max abs diff {verify(winner, st.raw(val), winner, 'priority'):.2e}")
        models[winner] = st
    classes = ["P1", "P2", "P3", "P4"]
    val_tags = F.tag_frame(val["body"])

    rows = []
    for edit_name, edit in EDITS.items():
        new = val["body"].map(edit)
        mask = new.notna().to_numpy()
        if not mask.any():
            continue
        orig = val[mask].reset_index(drop=True)
        edited = orig.assign(body=new[mask].to_numpy())
        groups = {"all": np.ones(len(orig), bool)}
        if edit_name == "E1_scope_up":
            t = val_tags[mask].reset_index(drop=True)
            groups.update({f"env={e}": (t["environment"] == e).to_numpy() for e in ("production", "nonprod", "unknown", "golive")})
            groups.update({"workaround=none": (t["workaround"] == "none_mentioned").to_numpy(),
                           "workaround=any": (t["workaround"] != "none_mentioned").to_numpy()})
        for g, gm in groups.items():
            if gm.sum() == 0:
                continue
            for run, model in models.items():
                rows.append({"edit": edit_name, "subset": g, "run": run,
                             **effect(model, orig[gm], edited[gm], classes)})
    table = pd.DataFrame(rows)
    L.append("\nCounterfactual edits on ALL applicable VAL tickets (calibrated; up = more urgent)")
    L.append(table.round(4).to_string(index=False))
    (OUT / "probe_v2_report.txt").write_text("\n".join(L), encoding="utf-8")
    table.to_csv(OUT / "probe_v2.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main(fit_only="--fit-only" in sys.argv)
