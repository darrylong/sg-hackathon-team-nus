"""Per-target temperature scaling fit on OOF train probabilities.

python src/calibration.py --run <name>
Writes oof_calibrated.csv, val_calibrated.csv and calibration.json into the run folder.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from data import TARGETS, load_splits, load_tickets
from runs import RUNS_DIR, load_run_frame, proba_matrix, write_pred_frame

EPS = 1e-12


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, EPS, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], EPS, 1))))


def fit_temperature(p: np.ndarray, y_idx: np.ndarray) -> float:
    res = minimize_scalar(lambda logT: nll(apply_temperature(p, np.exp(logT)), y_idx),
                          bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(res.x))


def calibrate_run(run: str) -> dict:
    df = load_tickets()
    parts = load_splits(df)
    oof, val = load_run_frame(run, "oof_train.csv"), load_run_frame(run, "val.csv")
    truth = df.set_index("ticket_id")
    oof_cal, val_cal, info, classes_all = oof.copy(), val.copy(), {}, {}
    for t in TARGETS:
        classes, p_oof = proba_matrix(oof, t)
        _, p_val = proba_matrix(val, t)
        classes_all[t] = classes
        y = truth.loc[oof["ticket_id"], t].map({c: i for i, c in enumerate(classes)}).to_numpy()
        T = fit_temperature(p_oof, y)
        cols = [f"proba_{t}_{c}" for c in classes]
        oof_cal[cols] = apply_temperature(p_oof, T)
        val_cal[cols] = apply_temperature(p_val, T)
        info[t] = {"T": T, "oof_nll_raw": nll(p_oof, y), "oof_nll_cal": nll(oof_cal[cols].to_numpy(), y)}
    out = RUNS_DIR / run
    write_pred_frame(out / "oof_calibrated.csv", oof_cal, parts["train"]["ticket_id"], classes_all)
    write_pred_frame(out / "val_calibrated.csv", val_cal, parts["val"]["ticket_id"], classes_all)
    (out / "calibration.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    info = calibrate_run(ap.parse_args().run)
    for t, v in info.items():
        print(f"{t:14s} T={v['T']:.3f}  OOF NLL {v['oof_nll_raw']:.4f} -> {v['oof_nll_cal']:.4f}")
