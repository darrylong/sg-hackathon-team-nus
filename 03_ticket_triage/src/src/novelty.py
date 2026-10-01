"""Topic novelty: 1 - max cosine similarity to any TRAIN ticket in the bundle's category TF-IDF space.

The bundle's category pipeline (text -> word TF-IDF, L2-normalised, fit on TRAIN) gives the vectors, so cosine
similarity is a sparse dot product. build() stores the TRAIN matrix and the threshold (99.5th percentile of VAL
novelty) as models/<bundle>/novelty.npz and novelty.json without touching the rest of the bundle.

python src/novelty.py   -> builds models/bundle_v3/novelty.{npz,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CHUNK = 500
PERCENTILE = 99.5


def vectorize(category_pipeline, df):
    steps = category_pipeline.named_steps
    return steps["tfidf"].transform(steps["text"].transform(df)).tocsr()


def novelty(X_query, X_train_T, chunk: int = CHUNK) -> np.ndarray:
    """1 - max cosine similarity of each query row to any TRAIN row (rows are L2-normalised)."""
    out = np.empty(X_query.shape[0])
    for s in range(0, X_query.shape[0], chunk):
        sims = (X_query[s:s + chunk] @ X_train_T).max(axis=1).toarray().ravel()
        out[s:s + chunk] = 1 - sims
    return np.clip(out, 0, 1)


def build(bundle_dir=None):
    import joblib
    from data import load_splits, load_tickets
    bundle_dir = Path(bundle_dir or SRC.parent / "models" / "bundle_v3")
    pipe = joblib.load(bundle_dir / "category.joblib")
    parts = load_splits(load_tickets())
    X_train = vectorize(pipe, parts["train"])
    X_train_T = X_train.T.tocsc()
    val_nov = novelty(vectorize(pipe, parts["val"]), X_train_T)
    t = float(np.percentile(val_nov, PERCENTILE))
    sp.save_npz(bundle_dir / "novelty.npz", X_train, compressed=True)
    meta = {"guard": "novel_topic", "description": "topic unlike any training ticket",
            "score": "1 - max cosine similarity to any TRAIN ticket (bundle category TF-IDF, word 1-2 grams)",
            "threshold": t, "threshold_rule": f"{PERCENTILE}th percentile of VAL novelty",
            "train_rows": int(X_train.shape[0]), "val_flag_rate": float((val_nov >= t).mean())}
    (bundle_dir / "novelty.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(build())
