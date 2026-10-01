"""Baseline (word TF-IDF + LogisticRegression) in the standard run format. run_name = baseline_lr."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data import ROOT, TARGETS, build_text
from runs import load_run_frame, make_run, proba_matrix

RUN = "baseline_lr"
TFIDF = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)
LR = dict(max_iter=3000, random_state=42)


def make_model(target):
    return Pipeline([
        ("text", FunctionTransformer(build_text)),
        ("tfidf", TfidfVectorizer(**TFIDF)),
        ("clf", LogisticRegression(**LR)),
    ])


def main():
    make_run(RUN, make_model, {"description": "word TF-IDF (1,2) + LogisticRegression, one model per target",
                               "tfidf": {**TFIDF, "ngram_range": list(TFIDF["ngram_range"])}, "lr": LR})
    # Consistency check against the original baseline run
    old = pd.read_csv(ROOT / "outputs" / "baseline" / "val_predictions.csv")
    new = load_run_frame(RUN, "val.csv")
    assert list(old["ticket_id"]) == list(new["ticket_id"])
    for t in TARGETS:
        classes, p = proba_matrix(new, t)
        pred = [classes[i] for i in p.argmax(1)]
        f_old = f1_score(old[f"true_{t}"], old[f"pred_{t}"], average="macro")
        f_new = f1_score(old[f"true_{t}"], pred, average="macro")
        print(f"{t:14s} macro_F1 old={f_old:.4f} new={f_new:.4f} {'OK' if abs(f_old - f_new) < 1e-9 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
