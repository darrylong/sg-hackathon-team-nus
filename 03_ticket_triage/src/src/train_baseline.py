"""Baseline: word TF-IDF + 4 independent LogisticRegression models.

Trains on TRAIN, reports on VALIDATION only. TEST and eval/ are not touched.
Run from 03_ticket_triage/:  python src/train_baseline.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from data import ROOT, TARGETS, build_text, load_splits, load_tickets, representativeness_report

MODEL_DIR = ROOT / "models" / "baseline"
OUT_DIR = ROOT / "outputs" / "baseline"
CONF_COL = {"category": "category_confidence", "priority": "priority_confidence",
            "assigned_team": "team_confidence", "sentiment": "sentiment_confidence"}


def escalation(priority, sentiment):
    priority, sentiment = np.asarray(priority), np.asarray(sentiment)
    return (sentiment == "Angry") | ((sentiment == "Frustrated") & np.isin(priority, ["P1", "P2"]))


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def main():
    df = load_tickets()
    parts = load_splits(df)
    train, val = parts["train"], parts["val"]
    print("Split sizes:", {k: len(v) for k, v in parts.items()})
    representativeness_report(df, parts)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)
    X_train = vec.fit_transform(build_text(train))
    X_val = vec.transform(build_text(val))
    print(f"\nTF-IDF vocabulary: {len(vec.vocabulary_)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, MODEL_DIR / "tfidf.joblib")

    preds = pd.DataFrame({"ticket_id": val["ticket_id"]})
    summary = {}
    for target in TARGETS:
        clf = LogisticRegression(max_iter=3000, random_state=SEED_LR)
        clf.fit(X_train, train[target])
        joblib.dump(clf, MODEL_DIR / f"{target}_lr.joblib")

        proba = clf.predict_proba(X_val)
        y_pred = clf.classes_[proba.argmax(1)]
        conf = proba.max(1)
        y_true = val[target].to_numpy()
        preds[f"true_{target}"] = y_true
        preds[f"pred_{target}"] = y_pred
        preds[CONF_COL[target]] = conf
        for i, c in enumerate(clf.classes_):
            preds[f"proba_{target}_{c}"] = proba[:, i]

        correct = y_pred == y_true
        summary[target] = dict(acc=accuracy_score(y_true, y_pred), macro_f1=macro_f1(y_true, y_pred),
                               conf_all=conf.mean(), conf_right=conf[correct].mean(),
                               conf_wrong=conf[~correct].mean())
        print(f"\n===== {target} =====")
        print(f"accuracy={summary[target]['acc']:.4f}  macro_F1={summary[target]['macro_f1']:.4f}")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
        print("confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_).to_string())
        if target == "priority":
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=["P1"], zero_division=0)
            summary["P1"] = (p[0], r[0], f[0])
            print(f"P1 precision={p[0]:.4f}  recall={r[0]:.4f}  F1={f[0]:.4f}")

    # Escalation risk derived from predicted priority + sentiment
    esc_t = escalation(preds["true_priority"], preds["true_sentiment"])
    esc_p = escalation(preds["pred_priority"], preds["pred_sentiment"])
    ep, er, ef, _ = precision_recall_fscore_support(esc_t, esc_p, average="binary", zero_division=0)
    print(f"\n===== escalation risk (derived) =====\nprecision={ep:.4f} recall={er:.4f} F1={ef:.4f} "
          f"(true positives rate in val: {esc_t.mean():.3f})")

    # Oracle-style team lookups (use TRUE validation category) -- diagnostic only
    cat_map = train.groupby("category")["assigned_team"].agg(lambda s: s.value_counts().index[0])
    pair_map = train.groupby(["category", "product"])["assigned_team"].agg(lambda s: s.value_counts().index[0])
    lk_cat = val["category"].map(cat_map)
    lk_pair = pd.Series([pair_map.get((c, p), cat_map[c]) for c, p in zip(val["category"], val["product"])])
    n_fallback = sum((c, p) not in pair_map.index for c, p in zip(val["category"], val["product"]))
    lookups = {}
    print("\n===== ORACLE team lookups (use TRUE val category; diagnostic, not deployable) =====")
    for name, pred in (("category lookup", lk_cat), ("category+product lookup", lk_pair)):
        lookups[name] = (accuracy_score(val["assigned_team"], pred), macro_f1(val["assigned_team"], pred))
        print(f"{name}: accuracy={lookups[name][0]:.4f} macro_F1={lookups[name][1]:.4f}")
    print(f"(category+product: {n_fallback} val tickets fell back to category-only)")

    # Routing cost, no abstention
    costs = pd.read_csv(ROOT / "artifacts" / "routing_cost_matrix.csv")
    cost = costs.set_index(["true_priority", "outcome"])["cost"]
    right = preds["pred_assigned_team"] == preds["true_assigned_team"]
    outcome = np.where(right, "correct", "misroute")
    preds["routing_cost"] = [cost[(p, o)] for p, o in zip(preds["true_priority"], outcome)]
    print("\n===== routing cost (LR team, no abstention) =====")
    print(f"correct={right.sum()} wrong={(~right).sum()} accuracy={right.mean():.4f} "
          f"total_penalty={preds['routing_cost'].sum():.1f} mean_cost={preds['routing_cost'].mean():.4f}")
    by_p = preds.assign(wrong=~right).groupby("true_priority").agg(
        n=("wrong", "size"), wrong=("wrong", "sum"), total_cost=("routing_cost", "sum"))
    by_p["misroute_rate"] = (by_p["wrong"] / by_p["n"]).round(4)
    by_p["share_of_mean_cost"] = (by_p["total_cost"] / len(preds)).round(4)
    print(by_p.to_string())

    print("\n===== confidence (max predict_proba) on val =====")
    for t in TARGETS:
        s = summary[t]
        print(f"{t:14s} mean={s['conf_all']:.3f} correct={s['conf_right']:.3f} wrong={s['conf_wrong']:.3f}")

    preds.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    print("\n===== SUMMARY (validation) =====")
    for t in TARGETS:
        print(f"{t:14s} acc={summary[t]['acc']:.4f} macro_F1={summary[t]['macro_f1']:.4f}")
    print("P1 P/R/F1 = %.4f / %.4f / %.4f" % summary["P1"])
    print(f"escalation F1 = {ef:.4f}")
    print(f"mean routing cost (no abstain) = {preds['routing_cost'].mean():.4f}")


SEED_LR = 42

if __name__ == "__main__":
    main()
