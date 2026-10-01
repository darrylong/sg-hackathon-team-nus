"""Export a deployable model bundle that reproduces a selected combined run.

python src/bundle.py --run combined_v3 --rows train --out models/bundle_v3 [--jobs 2]
python src/bundle.py --run combined_v3 --rows all --out models/bundle_v3_all --i-have-locked

Reads outputs/round2c/decisions.json and each source run's config.json to work out how every target was
produced, refits those components on the chosen rows and joblib-saves them with a manifest.json:
  category.joblib      category model (e.g. baseline_lr word TF-IDF + LR)
  priority.joblib      priority model (E5b_ordinal_char)
  team_base_category.joblib / team_base_team.joblib   inputs of the team stacker (raw probabilities)
  team_stacker.joblib  HGB team stacker (experiments.stack_features layout)
  sentiment.joblib     sentiment model (E9_tree_selected)
  correctness.joblib   P(team correct) model, refit exactly as correctness.py does from OOF calibrated features
Temperatures come from each source run's calibration.json. --rows train reuses splits/train_folds.csv and
checks every recomputed OOF matrix against the stored oof_train.csv. --rows all (train+val+test) builds new
5-fold folds inside the bundle and is refused unless --i-have-locked is passed.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, cross_val_predict

import experiments as ex
import run_baseline_lr
from calibration import apply_temperature, fit_temperature
from correctness import CAT_COLS, build_features
from correctness import make_model as make_correctness_model
from data import ROOT, SEED, TARGETS, _strat_key, load_splits, load_tickets
from routing import load_costs
from runs import N_FOLDS, RUNS_DIR, class_lists, get_folds, load_run_frame, proba_columns

DECISIONS = ROOT / "outputs" / "round2c" / "decisions.json"
P1_OPTIONS = ROOT / "outputs" / "round2c" / "p1_threshold" / "options.csv"
P1_THRESHOLD = 0.225
OOF_ATOL = 1e-6
PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "rapidfuzz"]


# ---------------------------------------------------------------- resolving how each target was built
@dataclass(frozen=True)
class Leaf:
    """An estimator taking a ticket DataFrame, trained by runs.make_run for (run, target)."""
    run: str
    target: str

    def make(self):
        if self.run == "baseline_lr":
            return run_baseline_lr.make_model(self.target)
        return ex.EXPERIMENTS[self.run]["make_model"](self.target)


@dataclass(frozen=True)
class Stacker:
    """experiments.team_stacker_producer(source_run): HGB on [P(category), P(team)] of source_run + product/channel."""
    run: str
    source_run: str
    category: Leaf
    team: Leaf


def run_config(run: str) -> dict:
    return json.loads((RUNS_DIR / run / "config.json").read_text())


def resolve(run: str, target: str):
    """Follow config.json target_sources until a trainable component is reached."""
    if run == "baseline_lr":
        return Leaf(run, target)
    cfg = run_config(run)
    src = cfg.get("target_sources", {}).get(target)
    if run in ex.EXPERIMENTS and src == "trained":
        assert ex.EXPERIMENTS[run]["make_model"](target) is not None, (run, target)
        return Leaf(run, target)
    if src == "custom":
        source = cfg["source_run"]
        assert "stacker" in cfg.get("change", "").lower(), f"{run}/{target}: unknown custom component"
        return Stacker(run, source, resolve(source, "category"), resolve(source, "assigned_team"))
    if src and src.startswith("copied from "):
        return resolve(src[len("copied from "):], target)
    if src and (RUNS_DIR / src).is_dir():  # assembled run
        return resolve(src, target)
    raise ValueError(f"cannot resolve how {run}/{target} was built (target_sources={src!r})")


# ---------------------------------------------------------------- fitting
def _aligned(model, X, classes):
    p = model.predict_proba(X)
    return p[:, [list(model.classes_).index(c) for c in classes]]


def _fit_task(leaf: Leaf, rows: pd.DataFrame, fit_mask, pred_mask, classes):
    model = leaf.make().fit(rows[fit_mask], rows.loc[fit_mask, leaf.target])
    if pred_mask is None:
        return model
    return _aligned(model, rows[pred_mask], classes)


def fit_leaves(leaves, need_oof, rows, folds, classes, jobs):
    """Full fit for every leaf, plus 5-fold OOF probabilities for leaves in need_oof. Runs folds in parallel."""
    all_rows = np.ones(len(rows), bool)
    tasks = [(leaf, None) for leaf in leaves] + [(leaf, f) for leaf in leaves if leaf in need_oof
                                                 for f in range(N_FOLDS)]
    # longest first so the pool stays busy
    weight = {"E9_tree_selected": 3, "E1_char_tfidf": 2, "E5b_ordinal_char": 2}
    tasks.sort(key=lambda t: -weight.get(t[0].run, 1))
    out = Parallel(n_jobs=jobs, verbose=10)(
        delayed(_fit_task)(leaf, rows,
                           all_rows if f is None else folds != f,
                           None if f is None else folds == f,
                           classes[leaf.target])
        for leaf, f in tasks)
    models, oof = {}, {leaf: np.zeros((len(rows), len(classes[leaf.target]))) for leaf in need_oof}
    for (leaf, f), res in zip(tasks, out):
        if f is None:
            models[leaf] = res
        else:
            oof[leaf][folds == f] = res
    return models, oof


def stack_frame(p_cat, p_team, classes) -> pd.DataFrame:
    """Prediction frame with the column layout experiments.stack_features expects."""
    return pd.DataFrame(np.hstack([p_cat, p_team]),
                        columns=proba_columns("category", classes["category"])
                        + proba_columns("assigned_team", classes["assigned_team"]))


def check_oof(name, ours, run, target, classes, rows, report):
    """--rows train only: recomputed OOF must equal the stored source-run OOF."""
    stored = load_run_frame(run, "oof_train.csv")
    assert list(stored["ticket_id"]) == list(rows["ticket_id"])
    diff = float(np.abs(stored[proba_columns(target, classes[target])].to_numpy() - ours).max())
    report[name] = {"source": f"{run}/oof_train.csv", "max_abs_diff": diff, "ok": diff <= OOF_ATOL}
    print(f"  OOF check {name:28s} vs {run}: max|diff|={diff:.2e} {'OK' if diff <= OOF_ATOL else 'MISMATCH'}")


# ---------------------------------------------------------------- manifest helpers
def package_versions():
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def headline_metrics(run: str) -> dict:
    m = json.loads((RUNS_DIR / run / "metrics.json").read_text())
    f = m["routing"]["f_expcost_pcorrect"]
    out = {"split": "VAL", "n": 4500, "source": f"outputs/runs/{run}/metrics.json",
           "macro_f1": {t: m["targets"][t]["macro_f1"] for t in TARGETS},
           "P1_argmax": m["P1"], "escalation_argmax": m["escalation"],
           "policy_f": {k: f[k] for k in ("mean_cost", "abstain_rate", "n_abstain")},
           "policy_no_abstain_cost": m["routing"]["a_no_abstain"]["mean_cost"],
           "team_error_auroc": m["team_error_auroc"]}
    if P1_OPTIONS.exists():
        opt = pd.read_csv(P1_OPTIONS)
        row = opt[np.isclose(opt["t"].astype(float), P1_THRESHOLD)]
        if len(row):
            r = row.iloc[0]
            out["priority_with_P1_threshold"] = {"t": P1_THRESHOLD, "priority_macro_f1": r["prio_macro_F1"],
                                                 "P1_precision": r["P1_P"], "P1_recall": r["P1_R"],
                                                 "P1_f1": r["P1_F1"], "escalation_f1": r["escalation_F1"],
                                                 "source": str(P1_OPTIONS.relative_to(ROOT))}
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- main
def build(run: str, rows_mode: str, out: Path, jobs: int) -> dict:
    t0 = time.time()
    decisions = json.loads(DECISIONS.read_text())
    if run not in decisions:
        raise SystemExit(f"{run} not in {DECISIONS}")
    sources = decisions[run]["sources"]
    if decisions[run].get("label_sources"):
        raise SystemExit(f"{run} has label_sources; bundle export does not support split label/routing models")
    cfg = run_config(run)
    assert cfg["target_sources"] == sources, "decisions.json and run config disagree on sources"

    tickets = load_tickets()
    parts = load_splits(tickets)
    train = parts["train"]
    classes = class_lists(train)
    assert classes == cfg["classes"]
    if rows_mode == "train":
        rows = train
        folds = get_folds(train).to_numpy()
    else:
        rows = pd.concat([parts[s] for s in ("train", "val", "test")], ignore_index=True)
        assert class_lists(rows) == classes, "class lists change when adding val/test rows"
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = np.empty(len(rows), dtype=int)
        for f, (_, idx) in enumerate(skf.split(rows, _strat_key(rows))):
            folds[idx] = f

    comps = {t: resolve(run, t) for t in TARGETS}
    print("Components:", {t: (c.run if isinstance(c, Leaf) else f"stacker on {c.source_run}") for t, c in comps.items()})
    stacker = comps["assigned_team"]
    assert isinstance(stacker, Stacker), "this exporter expects an HGB team stacker (combined_v3 layout)"
    for t in ("category", "priority", "sentiment"):
        assert isinstance(comps[t], Leaf), t
    leaves = list(dict.fromkeys([comps["category"], comps["priority"], comps["sentiment"],
                                 stacker.category, stacker.team]))
    # OOF is needed for the stacker inputs and for the correctness features (category, priority, team);
    # sentiment OOF is not used anywhere (temperatures come from calibration.json).
    need_oof = {comps["category"], comps["priority"], stacker.category, stacker.team}

    print(f"Fitting {len(leaves)} components on {len(rows)} {rows_mode} rows ({jobs} jobs)...", flush=True)
    models, oof = fit_leaves(leaves, need_oof, rows, folds, classes, jobs)
    oof_report = {}
    if rows_mode == "train":
        for leaf in need_oof:
            check_oof(f"{leaf.run}/{leaf.target}", oof[leaf], leaf.run, leaf.target, classes, rows, oof_report)

    # ---- team stacker, exactly as experiments.team_stacker_producer
    X_stack = ex.stack_features(stack_frame(oof[stacker.category], oof[stacker.team], classes), rows)
    y_team = rows["assigned_team"].to_numpy()
    team_oof = cross_val_predict(ex.hgb(), X_stack, y_team, cv=PredefinedSplit(folds), method="predict_proba")
    stack_model = ex.hgb().fit(X_stack, y_team)
    assert list(stack_model.classes_) == classes["assigned_team"]
    if rows_mode == "train":
        check_oof("team_stacker/assigned_team", team_oof, stacker.run, "assigned_team", classes, rows, oof_report)

    # ---- temperatures from each source run's calibration.json
    temps, t_refit = {}, {}
    run_cal = json.loads((RUNS_DIR / run / "calibration.json").read_text())
    for t in TARGETS:
        src_T = json.loads((RUNS_DIR / sources[t] / "calibration.json").read_text())[t]["T"]
        assert abs(src_T - run_cal[t]["T"]) < 1e-12, f"{t}: {sources[t]} T != {run} T"
        temps[t] = src_T
    raw_oof = {"category": oof[comps["category"]], "priority": oof[comps["priority"]], "assigned_team": team_oof}
    for t, p in raw_oof.items():
        y = rows[t].map({c: i for i, c in enumerate(classes[t])}).to_numpy()
        t_refit[t] = fit_temperature(p, y)

    # ---- correctness model, refit exactly as correctness.run_correctness (final fit on all chosen rows)
    cal = pd.DataFrame({"ticket_id": rows["ticket_id"]})
    for t, p in raw_oof.items():
        cal[proba_columns(t, classes[t])] = apply_temperature(p, temps[t])
    X_corr, team_pred = build_features(cal, rows[["ticket_id"] + CAT_COLS])
    y_corr = (team_pred == y_team).astype(int)
    num_cols = [c for c in X_corr.columns if c not in CAT_COLS]
    corr_model = make_correctness_model(num_cols).fit(X_corr, y_corr)

    # ---- save
    out.mkdir(parents=True, exist_ok=True)
    files = {"category": comps["category"], "priority": comps["priority"], "sentiment": comps["sentiment"],
             "team_base_category": stacker.category, "team_base_team": stacker.team}
    for name, leaf in files.items():
        joblib.dump(models[leaf], out / f"{name}.joblib", compress=3)
    joblib.dump(stack_model, out / "team_stacker.joblib", compress=3)
    joblib.dump(corr_model, out / "correctness.joblib", compress=3)
    if rows_mode == "all":
        pd.DataFrame({"ticket_id": rows["ticket_id"], "fold": folds}).to_csv(out / "folds.csv", index=False)

    W, A, cap = load_costs()
    manifest = {
        "bundle_format": 1,
        "run": run,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows_mode,
        "n_rows": int(len(rows)),
        "build_seconds": None,
        "source_runs": sources,
        "components": {
            "category": {"file": "category.joblib", "run": comps["category"].run},
            "priority": {"file": "priority.joblib", "run": comps["priority"].run},
            "sentiment": {"file": "sentiment.joblib", "run": comps["sentiment"].run},
            "assigned_team": {"file": "team_stacker.joblib", "run": stacker.run, "kind": "hgb_stacker",
                              "stack_source_run": stacker.source_run,
                              "inputs": {"category": {"file": "team_base_category.joblib", "run": stacker.category.run},
                                         "assigned_team": {"file": "team_base_team.joblib", "run": stacker.team.run}},
                              "features": "[raw P(category), raw P(team), one-hot product (PRODUCTS), one-hot channel (CHANNELS)]"},
            "correctness": {"file": "correctness.joblib", "features": "correctness.build_features on calibrated "
                            "category/priority/team probabilities + product/channel"},
        },
        "temperatures": temps,
        "temperature_sources": {t: f"outputs/runs/{sources[t]}/calibration.json" for t in TARGETS},
        "temperature_refit_on_bundle_oof": t_refit,
        "classes": classes,
        "products": ex.PRODUCTS,
        "known_products": sorted(rows["product"].unique()),
        "channels": ex.CHANNELS,
        "p1_threshold": P1_THRESHOLD,
        "priority_rule": "P1 if calibrated P(P1) >= p1_threshold else argmax over P2..P4",
        "cost_matrix": {"misroute": W, "abstain": A},
        "max_abstain_rate": cap,
        "policy": "f_expcost_pcorrect: savings = (1-p_correct)*E[misroute|prio] - E[abstain|prio]; abstain top "
                  "floor(cap*N) by savings with savings > 0 (cap=None: abstain iff savings > 0)",
        "guards": {"min_body_chars": 20},
        "oof_checks": oof_report,
        "val_metrics": headline_metrics(run),
        "package_versions": package_versions(),
        "selection": decisions[run],
    }
    manifest["build_seconds"] = round(time.time() - t0, 1)
    manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(out.glob("*.joblib"))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="combined_v3")
    ap.add_argument("--rows", choices=["train", "all"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2, help="parallel fits (each word+char fit needs ~1-2 GB RAM)")
    ap.add_argument("--i-have-locked", action="store_true",
                    help="required for --rows all: confirms model selection is final and TEST may be trained on")
    args = ap.parse_args()
    if args.rows == "all" and not args.i_have_locked:
        sys.exit("Refusing --rows all (trains on VAL and TEST). Pass --i-have-locked once model selection is final.")
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    m = build(args.run, args.rows, out, args.jobs)
    bad = [k for k, v in m["oof_checks"].items() if not v["ok"]]
    print(f"\nBundle written to {out} in {m['build_seconds']:.0f}s")
    print("Temperatures:", {t: round(v, 4) for t, v in m["temperatures"].items()},
          "| refit on bundle OOF:", {t: round(v, 4) for t, v in m["temperature_refit_on_bundle_oof"].items()})
    if bad:
        print("WARNING: OOF mismatches vs stored runs:", bad)


if __name__ == "__main__":
    main()
