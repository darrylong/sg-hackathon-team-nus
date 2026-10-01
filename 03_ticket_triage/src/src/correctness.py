"""Team-correctness model (model-agnostic): predicts P(argmax team is right) from calibrated probabilities.

python src/correctness.py --run <name>
Trains on TRAIN out-of-fold calibrated probabilities, reports its own OOF log-loss / AUROC vs p_ok,
fits on all TRAIN rows, applies to VAL and writes correctness.csv (ticket_id, p_correct).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from calibration import calibrate_run
from data import load_splits, load_tickets
from runs import N_FOLDS, RUNS_DIR, get_folds, load_run_frame, proba_matrix

EPS = 1e-12
CAT_COLS = ["product", "channel"]


def build_features(cal: pd.DataFrame, tickets: pd.DataFrame):
    """Features from a calibrated prediction frame + ticket metadata. Returns (X, team_pred)."""
    meta = tickets.set_index("ticket_id").loc[cal["ticket_id"], CAT_COLS].reset_index(drop=True)
    team_classes, team = proba_matrix(cal, "assigned_team")
    prio_classes, prio = proba_matrix(cal, "priority")
    _, cat = proba_matrix(cal, "category")
    s = np.sort(team, 1)
    p_ok = s[:, -1]
    X = pd.DataFrame({
        "log_p_ok": np.log(np.clip(p_ok, EPS, 1)),
        "log_1m_p_ok": np.log(np.clip(1 - p_ok, EPS, 1)),
        "margin": s[:, -1] - s[:, -2],
        "entropy": -(team * np.log(np.clip(team, EPS, 1))).sum(1),
        **{f"p_{c}": prio[:, i] for i, c in enumerate(prio_classes)},
        "max_cat": cat.max(1),
        "p_ok_x_P1": p_ok * prio[:, prio_classes.index("P1")],
        "p_ok_x_P2": p_ok * prio[:, prio_classes.index("P2")],
    })
    X[CAT_COLS] = meta.to_numpy()
    return X, np.array(team_classes)[team.argmax(1)]


def make_model(num_cols):
    return Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), num_cols),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def run_correctness(run: str) -> dict:
    run_dir = RUNS_DIR / run
    if not (run_dir / "oof_calibrated.csv").exists():
        calibrate_run(run)
    tickets = load_tickets()
    parts = load_splits(tickets)
    train, val = parts["train"], parts["val"]
    oof_cal, val_cal = load_run_frame(run, "oof_calibrated.csv"), load_run_frame(run, "val_calibrated.csv")
    assert list(oof_cal["ticket_id"]) == list(train["ticket_id"])

    X_tr, team_pred = build_features(oof_cal, tickets)
    y_tr = (team_pred == train["assigned_team"].to_numpy()).astype(int)
    num_cols = [c for c in X_tr.columns if c not in CAT_COLS]
    folds = get_folds(train).to_numpy()
    oof = np.zeros(len(train))
    for f in range(N_FOLDS):
        m = make_model(num_cols).fit(X_tr[folds != f], y_tr[folds != f])
        oof[folds == f] = m.predict_proba(X_tr[folds == f])[:, 1]
    p_ok_tr = np.exp(X_tr["log_p_ok"].to_numpy())

    model = make_model(num_cols).fit(X_tr, y_tr)
    X_val, val_pred = build_features(val_cal, tickets)
    p_correct = model.predict_proba(X_val)[:, 1]
    pd.DataFrame({"ticket_id": val_cal["ticket_id"], "p_correct": p_correct}).to_csv(
        run_dir / "correctness.csv", index=False)

    y_val = (val_pred == val["assigned_team"].to_numpy()).astype(int)
    p_ok_val = np.exp(X_val["log_p_ok"].to_numpy())
    info = {
        "train_oof": {"n": len(y_tr), "error_rate": 1 - y_tr.mean(),
                      "logloss_p_ok": log_loss(y_tr, np.clip(p_ok_tr, EPS, 1 - EPS)),
                      "logloss_p_correct": log_loss(y_tr, oof),
                      "auroc_p_ok": roc_auc_score(y_tr, p_ok_tr), "auroc_p_correct": roc_auc_score(y_tr, oof)},
        "val": {"n": len(y_val), "error_rate": 1 - y_val.mean(),
                "logloss_p_ok": log_loss(y_val, np.clip(p_ok_val, EPS, 1 - EPS)),
                "logloss_p_correct": log_loss(y_val, p_correct),
                "auroc_p_ok": roc_auc_score(y_val, p_ok_val), "auroc_p_correct": roc_auc_score(y_val, p_correct)},
    }
    (run_dir / "correctness.json").write_text(json.dumps(info, indent=2, default=float))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    info = run_correctness(ap.parse_args().run)
    for split, v in info.items():
        print(f"{split:9s} err={v['error_rate']:.4f}  logloss p_ok={v['logloss_p_ok']:.4f} p_correct={v['logloss_p_correct']:.4f}"
              f"  AUROC p_ok={v['auroc_p_ok']:.4f} p_correct={v['auroc_p_correct']:.4f}")
