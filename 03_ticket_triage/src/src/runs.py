"""Standard, model-agnostic prediction format for experiment runs.

outputs/runs/<run_name>/
  oof_train.csv  5-fold out-of-fold probabilities on TRAIN
  val.csv        probabilities on VAL from a model fit on all of TRAIN
  config.json    model description and settings
Columns: ticket_id, proba_<target>_<class> for all targets (classes = sorted TRAIN labels).
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data import ROOT, SEED, SPLIT_DIR, TARGETS, _strat_key, load_splits, load_tickets

RUNS_DIR = ROOT / "outputs" / "runs"
FOLDS_FILE = SPLIT_DIR / "train_folds.csv"
N_FOLDS = 5


def class_lists(train: pd.DataFrame) -> dict:
    return {t: sorted(train[t].unique()) for t in TARGETS}


def get_folds(train: pd.DataFrame) -> pd.Series:
    """Fold id per TRAIN row (aligned to train's index). Created once, never overwritten."""
    if not FOLDS_FILE.exists():
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold = np.empty(len(train), dtype=int)
        for f, (_, idx) in enumerate(skf.split(train, _strat_key(train))):
            fold[idx] = f
        pd.DataFrame({"ticket_id": train["ticket_id"], "fold": fold}).to_csv(FOLDS_FILE, index=False)
        print(f"Created {FOLDS_FILE.name}")
    folds = pd.read_csv(FOLDS_FILE)
    assert folds["ticket_id"].is_unique and set(folds["ticket_id"]) == set(train["ticket_id"]), \
        "train_folds.csv does not match train_ids.csv"
    assert set(folds["fold"]) == set(range(N_FOLDS))
    return train["ticket_id"].map(folds.set_index("ticket_id")["fold"])


def proba_columns(target: str, classes: list) -> list:
    return [f"proba_{target}_{c}" for c in classes]


def proba_matrix(df: pd.DataFrame, target: str):
    """(classes, n x k array) for one target from a standard prediction frame."""
    prefix = f"proba_{target}_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    return [c[len(prefix):] for c in cols], df[cols].to_numpy()


def validate_pred_frame(df: pd.DataFrame, expected_ids, classes: dict) -> None:
    assert list(df["ticket_id"]) == list(expected_ids), "ticket_id mismatch / order"
    for t in TARGETS:
        cols = proba_columns(t, classes[t])
        missing = set(cols) - set(df.columns)
        assert not missing, f"missing columns {missing}"
        p = df[cols].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all(), f"{t}: invalid probabilities"
        assert np.allclose(p.sum(1), 1, atol=1e-6), f"{t}: rows do not sum to 1"


def write_pred_frame(path, df, expected_ids, classes) -> None:
    validate_pred_frame(df, expected_ids, classes)
    df.to_csv(path, index=False)


def _aligned_proba(model, X, classes):
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in classes]
    return p[:, order]


def make_run(run_name: str, make_model, config: dict, base_run: str | None = None,
             custom: dict | None = None, verbose: bool = True) -> None:
    """make_model(target) -> unfitted estimator taking a ticket DataFrame (e.g. sklearn Pipeline),
    or None to copy that target's probabilities unchanged from `base_run`.
    custom: optional {target: fn(train, val, folds, classes) -> (oof_proba, val_proba)} for models
    that do not fit the estimator interface (e.g. stackers)."""
    df = load_tickets()
    parts = load_splits(df)
    train, val = parts["train"], parts["val"]
    classes = class_lists(train)
    folds = get_folds(train).to_numpy()
    custom = custom or {}
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = (load_run_frame(base_run, "oof_train.csv"), load_run_frame(base_run, "val.csv")) if base_run else None

    oof = pd.DataFrame({"ticket_id": train["ticket_id"]})
    vdf = pd.DataFrame({"ticket_id": val["ticket_id"]})
    sources = {}
    for t in TARGETS:
        cols = proba_columns(t, classes[t])
        if t in custom:
            oof[cols], vdf[cols] = custom[t](train, val, folds, classes[t])
            sources[t] = "custom"
            continue
        if make_model(t) is None:
            assert base is not None, f"no model for {t} and no base_run"
            oof[cols], vdf[cols] = base[0][cols].to_numpy(), base[1][cols].to_numpy()
            sources[t] = f"copied from {base_run}"
            continue
        p = np.zeros((len(train), len(classes[t])))
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            model = make_model(t).fit(train[tr], train.loc[tr, t])
            p[te] = _aligned_proba(model, train[te], classes[t])
        oof[cols] = p
        model = make_model(t).fit(train, train[t])
        vdf[cols] = _aligned_proba(model, val, classes[t])
        sources[t] = "trained"
        if verbose:
            print(f"  [{run_name}] {t} done ({time.time() - t0:.0f}s)", flush=True)
    write_pred_frame(out / "oof_train.csv", oof, train["ticket_id"], classes)
    write_pred_frame(out / "val.csv", vdf, val["ticket_id"], classes)

    config = {**config, "run_name": run_name, "n_folds": N_FOLDS, "classes": classes,
              "target_sources": sources, "fit_seconds": round(time.time() - t0, 1)}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))


def assemble_run(run_name: str, sources: dict, config: dict, label_sources: dict | None = None) -> None:
    """Build a run by copying each target's probabilities from {target: source_run}.
    label_sources {target: run}: evaluate.py takes that target's LABEL metrics from another run,
    while routing / correctness keep using the copied probabilities."""
    if label_sources:
        config = {**config, "label_sources": label_sources}
    df = load_tickets()
    parts = load_splits(df)
    classes = class_lists(parts["train"])
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    for fname, split in (("oof_train.csv", "train"), ("val.csv", "val")):
        frame = pd.DataFrame({"ticket_id": parts[split]["ticket_id"]})
        for t in TARGETS:
            src = load_run_frame(sources[t], fname)
            assert list(src["ticket_id"]) == list(frame["ticket_id"])
            cols = proba_columns(t, classes[t])
            frame[cols] = src[cols].to_numpy()
        write_pred_frame(out / fname, frame, parts[split]["ticket_id"], classes)
    (out / "config.json").write_text(json.dumps({**config, "run_name": run_name, "classes": classes,
                                                 "target_sources": sources}, indent=2))


def load_run_frame(run_name: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_DIR / run_name / name)
