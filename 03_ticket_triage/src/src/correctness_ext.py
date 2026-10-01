"""Extended team-correctness model (round 2C+): correctness.py features plus cross-model / lookup features.

python src/correctness_ext.py --run <name> [--e1 E1_char_tfidf]
New features (all TRAIN rows out-of-fold, VAL from full-TRAIN models / tables, same folds as runs.get_folds):
  e1_agree      E1 calibrated team argmax == this run's team argmax
  e1_p_pred     E1 calibrated probability of this run's predicted team
  lookup_agree  most common TRAIN team for (argmax predicted category, product) == this run's team
  cat_entropy   entropy of this run's calibrated category probabilities (max_cat is already in the base set)
Writes correctness.csv (ticket_id, p_correct) and correctness.json (incl. coefficients) into the run folder.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from calibration import calibrate_run
from correctness import CAT_COLS, EPS, build_features, make_model
from data import load_splits, load_tickets
from runs import N_FOLDS, RUNS_DIR, get_folds, load_run_frame, proba_matrix

E1_RUN = "E1_char_tfidf"
NEW_FEATURES = ["e1_agree", "e1_p_pred", "lookup_agree", "cat_entropy"]


def fit_lookup(rows: pd.DataFrame) -> dict:
    """Most common team per (category, product), per category, and overall, from true TRAIN labels.
    Ties broken alphabetically by team so the table is deterministic."""
    def mode(g):
        c = g.value_counts()
        return sorted(c.index[c == c.max()])[0]
    return {"pair": rows.groupby(["category", "product"])["assigned_team"].agg(mode).to_dict(),
            "cat": rows.groupby("category")["assigned_team"].agg(mode).to_dict(),
            "all": mode(rows["assigned_team"])}


def apply_lookup(table: dict, category, product) -> np.ndarray:
    return np.array([table["pair"].get((c, p), table["cat"].get(c, table["all"]))
                     for c, p in zip(category, product)])


def build_features_ext(cal: pd.DataFrame, e1_cal: pd.DataFrame, lookup_team: np.ndarray, tickets: pd.DataFrame):
    """Base correctness features + NEW_FEATURES. Returns (X, team_pred)."""
    assert list(cal["ticket_id"]) == list(e1_cal["ticket_id"])
    X, team_pred = build_features(cal, tickets)
    team_classes, _ = proba_matrix(cal, "assigned_team")
    e1_classes, e1_team = proba_matrix(e1_cal, "assigned_team")
    assert e1_classes == team_classes
    pred_idx = pd.Index(team_classes).get_indexer(team_pred)
    _, cat = proba_matrix(cal, "category")
    new = pd.DataFrame({
        "e1_agree": (e1_team.argmax(1) == pred_idx).astype(float),
        "e1_p_pred": e1_team[np.arange(len(pred_idx)), pred_idx],
        "lookup_agree": (lookup_team == team_pred).astype(float),
        "cat_entropy": -(cat * np.log(np.clip(cat, EPS, 1))).sum(1),
    })
    num = X.drop(columns=CAT_COLS)
    return pd.concat([num, new, X[CAT_COLS]], axis=1), team_pred


def pred_category(cal: pd.DataFrame) -> np.ndarray:
    classes, cat = proba_matrix(cal, "category")
    return np.array(classes)[cat.argmax(1)]


def coefficients(model) -> pd.DataFrame:
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
    clf = model.named_steps["clf"]
    return pd.DataFrame({"feature": ["(intercept)"] + names,
                         "coef": np.r_[clf.intercept_, clf.coef_[0]]})


def run_correctness_ext(run: str, e1_run: str = E1_RUN) -> dict:
    run_dir = RUNS_DIR / run
    for r in (run, e1_run):
        if not (RUNS_DIR / r / "oof_calibrated.csv").exists():
            calibrate_run(r)
    tickets = load_tickets()
    parts = load_splits(tickets)
    train, val = parts["train"], parts["val"]
    oof_cal, val_cal = load_run_frame(run, "oof_calibrated.csv"), load_run_frame(run, "val_calibrated.csv")
    e1_oof, e1_val = load_run_frame(e1_run, "oof_calibrated.csv"), load_run_frame(e1_run, "val_calibrated.csv")
    assert list(oof_cal["ticket_id"]) == list(train["ticket_id"])
    assert list(val_cal["ticket_id"]) == list(val["ticket_id"])
    folds = get_folds(train).to_numpy()

    # lookup team: TRAIN out-of-fold tables, VAL from the full-TRAIN table
    cat_tr, cat_val = pred_category(oof_cal), pred_category(val_cal)
    lk_tr = np.empty(len(train), dtype=object)
    for f in range(N_FOLDS):
        table = fit_lookup(train[folds != f])
        lk_tr[folds == f] = apply_lookup(table, cat_tr[folds == f], train["product"].to_numpy()[folds == f])
    lk_val = apply_lookup(fit_lookup(train), cat_val, val["product"].to_numpy())

    X_tr, team_pred = build_features_ext(oof_cal, e1_oof, lk_tr, tickets)
    y_tr = (team_pred == train["assigned_team"].to_numpy()).astype(int)
    num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
    oof = np.zeros(len(train))
    for f in range(N_FOLDS):
        m = make_model(num_cols).fit(X_tr[folds != f], y_tr[folds != f])
        oof[folds == f] = m.predict_proba(X_tr[folds == f])[:, 1]
    p_ok_tr = np.exp(X_tr["log_p_ok"].to_numpy())

    model = make_model(num_cols).fit(X_tr, y_tr)
    X_val, val_pred = build_features_ext(val_cal, e1_val, lk_val, tickets)
    p_correct = model.predict_proba(X_val)[:, 1]
    pd.DataFrame({"ticket_id": val_cal["ticket_id"], "p_correct": p_correct}).to_csv(
        run_dir / "correctness.csv", index=False)

    y_val = (val_pred == val["assigned_team"].to_numpy()).astype(int)
    p_ok_val = np.exp(X_val["log_p_ok"].to_numpy())
    groups = {}
    for split, X, y in (("train_oof", X_tr, y_tr), ("val", X_val, y_val)):
        for feat in ("e1_agree", "lookup_agree"):
            g = pd.Series(1 - y).groupby(X[feat].to_numpy()).agg(["size", "mean"])
            groups[f"{split}/{feat}"] = {f"{int(k)}": {"n": int(r["size"]), "team_error_rate": float(r["mean"])}
                                         for k, r in g.iterrows()}
    info = {
        "model": "correctness_ext (base correctness.py features + " + ", ".join(NEW_FEATURES) + f"; E1 run {e1_run})",
        "train_oof": {"n": len(y_tr), "error_rate": 1 - y_tr.mean(),
                      "logloss_p_ok": log_loss(y_tr, np.clip(p_ok_tr, EPS, 1 - EPS)),
                      "logloss_p_correct": log_loss(y_tr, oof),
                      "auroc_p_ok": roc_auc_score(y_tr, p_ok_tr), "auroc_p_correct": roc_auc_score(y_tr, oof)},
        "val": {"n": len(y_val), "error_rate": 1 - y_val.mean(),
                "logloss_p_ok": log_loss(y_val, np.clip(p_ok_val, EPS, 1 - EPS)),
                "logloss_p_correct": log_loss(y_val, p_correct),
                "auroc_p_ok": roc_auc_score(y_val, p_ok_val), "auroc_p_correct": roc_auc_score(y_val, p_correct)},
        "feature_groups": groups,
        "coefficients_standardized": coefficients(model).set_index("feature")["coef"].to_dict(),
    }
    (run_dir / "correctness.json").write_text(json.dumps(info, indent=2, default=float))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--e1", default=E1_RUN)
    a = ap.parse_args()
    info = run_correctness_ext(a.run, a.e1)
    for split in ("train_oof", "val"):
        v = info[split]
        print(f"{split:9s} err={v['error_rate']:.4f}  logloss p_ok={v['logloss_p_ok']:.4f} p_correct={v['logloss_p_correct']:.4f}"
              f"  AUROC p_ok={v['auroc_p_ok']:.4f} p_correct={v['auroc_p_correct']:.4f}")
