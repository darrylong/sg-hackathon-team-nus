"""Round 2B model builders. Each experiment changes ONE thing relative to baseline_lr;
targets an experiment does not touch are copied from baseline_lr (identical, deterministic)."""
import re
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from data import build_text
from runs import load_run_frame, proba_matrix

SEED = 42
PRODUCTS = ["GreenLake", "ProLiant DL380 Gen11", "ProLiant DL360 Gen10", "Synergy 480 Gen11", "Alletra 6010",
            "Nimble HF40", "Aruba CX 6300", "Aruba CX 8360", "iLO 6"]
CHANNELS = ["email", "portal", "chat"]


# ---------------------------------------------------------------- text / features
def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)


def word_char():
    return FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])


FEATURES = {"word": word_tfidf, "word_char": word_char}

SEV_RE = re.compile(r"sev\w*\s*:\s*([a-z]+)", re.I)
SEV_LEVELS = ["low", "medium", "high", "critical"]


def parse_severity(body: str) -> str:
    m = SEV_RE.search(body)
    if not m:
        return "NONE"
    hit = process.extractOne(m.group(1).lower(), SEV_LEVELS, scorer=fuzz.ratio, score_cutoff=60)
    return hit[0].upper() if hit else "UNK"


def build_text_sev(df: pd.DataFrame) -> pd.Series:
    return "SEV_" + df["body"].map(parse_severity) + " " + build_text(df)


def to_dense(X):
    return X.toarray()


def lr(C=1.0, **kw):
    return LogisticRegression(max_iter=3000, random_state=SEED, C=C, **kw)


def text_lr(text_fn=build_text, features=word_tfidf, C=1.0, **kw):
    return Pipeline([("text", FunctionTransformer(text_fn)), ("tfidf", features()), ("clf", lr(C, **kw))])


# ---------------------------------------------------------------- custom estimators
class SoftmaxLinearSVC(LinearSVC):
    """LinearSVC whose decision scores are turned into probabilities by softmax."""

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class OrdinalLR(ClassifierMixin, BaseEstimator):
    """Cumulative binary LRs for P(y <= k); classes must be ordered when sorted (P1..P4)."""

    def __init__(self, features="word"):
        self.features = features

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        self.vec_ = FEATURES[self.features]()
        Z = self.vec_.fit_transform(build_text(X))
        yi = np.searchsorted(self.classes_, y)
        self.models_ = [lr().fit(Z, (yi <= k).astype(int)) for k in range(len(self.classes_) - 1)]
        return self

    def predict_proba(self, X):
        Z = self.vec_.transform(build_text(X))
        cum = np.column_stack([m.predict_proba(Z)[:, 1] for m in self.models_] + [np.ones(Z.shape[0])])
        p = np.clip(np.diff(np.column_stack([np.zeros(Z.shape[0]), cum]), axis=1), 0, None)
        s = p.sum(1, keepdims=True)
        return np.where(s > 0, p / np.where(s > 0, s, 1), 1 / p.shape[1])


class TeamCascade(ClassifierMixin, BaseEstimator):
    """P(team) = sum_c P(category=c | text) * P(team | c, product), add-1 smoothed counts,
    backing off to P(team | c) for unseen (category, product) pairs. Uses X['category'] only in fit."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        self.cat_model_ = text_lr().fit(X, X["category"])
        k = len(self.classes_)
        pair = pd.crosstab([X["category"], X["product"]], y).reindex(columns=self.classes_, fill_value=0)
        cat = pd.crosstab(X["category"], y).reindex(columns=self.classes_, fill_value=0)
        self.pair_ = (pair + 1).div(pair.sum(1) + k, axis=0)
        self.cat_ = (cat + 1).div(cat.sum(1) + k, axis=0)
        return self

    def predict_proba(self, X):
        pc = self.cat_model_.predict_proba(X)
        out = np.zeros((len(X), len(self.classes_)))
        products = X["product"].to_numpy()
        for j, c in enumerate(self.cat_model_.classes_):
            table = {p: (self.pair_.loc[(c, p)] if (c, p) in self.pair_.index else self.cat_.loc[c]).to_numpy()
                     for p in set(products)}
            out += pc[:, [j]] * np.vstack([table[p] for p in products])
        return out / out.sum(1, keepdims=True)


class P1WeightedLR(ClassifierMixin, BaseEstimator):
    """TF-IDF + LR with sample_weight = p1_weight for P1 tickets, 1 otherwise."""

    def __init__(self, p1_weight=3.0, features="word"):
        self.p1_weight = p1_weight
        self.features = features

    def fit(self, X, y):
        y = np.asarray(y)
        w = np.where(y == "P1", self.p1_weight, 1.0)
        self.pipe_ = text_lr(features=FEATURES[self.features]).fit(X, y, clf__sample_weight=w)
        self.classes_ = self.pipe_.classes_
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(X)


# ---------------------------------------------------------------- template sentence factors (E10)
_ID_RE = re.compile(r"\b[a-z]+[-_][a-z0-9-]*\d[a-z0-9-]*\b")
_EMAIL_RE = re.compile(r"\S+@\S+")
_USER_RE = re.compile(r"\b[a-z]+\.[a-z]+\b")
_NUM_RE = re.compile(r"\d+(?:[.:,/x-]\d+)*")
_PRODUCT_RES = [re.compile(re.escape(p.lower())) for p in sorted(PRODUCTS, key=len, reverse=True)]
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\]\s*")


def normalise_sentence(s: str) -> str:
    s = s.lower()
    for r in _PRODUCT_RES:
        s = r.sub("<product>", s)
    s = _EMAIL_RE.sub("<email>", s)
    s = _ID_RE.sub("<id>", s)
    s = _USER_RE.sub("<user>", s)
    s = _NUM_RE.sub("<num>", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;:!?\"'[]()")


def ticket_sentences(body: str) -> list:
    return [n for n in (normalise_sentence(s) for s in _SPLIT_RE.split(body)) if n]


class TemplateFactors(TransformerMixin, BaseEstimator):
    """Binary features for normalised sentences seen in >= min_tickets training tickets,
    with typo'd sentences fuzzy-mapped (rapidfuzz ratio >= threshold); plus product/channel one-hot."""

    def __init__(self, min_tickets=20, threshold=90):
        self.min_tickets = min_tickets
        self.threshold = threshold

    def fit(self, X, y=None):
        counts = Counter(s for b in X["body"] for s in set(ticket_sentences(b)))
        self.vocab_ = sorted(s for s, n in counts.items() if n >= self.min_tickets)
        self.index_ = {s: i for i, s in enumerate(self.vocab_)}
        self.counts_ = {s: counts[s] for s in self.vocab_}
        self.cache_ = {}
        return self

    def _lookup(self, s):
        if s in self.index_:
            return self.index_[s]
        if s not in self.cache_:
            hit = process.extractOne(s, self.vocab_, scorer=fuzz.ratio, score_cutoff=self.threshold)
            self.cache_[s] = self.index_[hit[0]] if hit else None
        return self.cache_[s]

    def transform(self, X):
        M = np.zeros((len(X), len(self.vocab_) + len(PRODUCTS) + len(CHANNELS)), dtype=np.float32)
        for r, b in enumerate(X["body"]):
            for s in ticket_sentences(b):
                i = self._lookup(s)
                if i is not None:
                    M[r, i] = 1
        off = len(self.vocab_)
        M[np.arange(len(X)), off + X["product"].map(PRODUCTS.index).to_numpy()] = 1
        M[np.arange(len(X)), off + len(PRODUCTS) + X["channel"].map(CHANNELS.index).to_numpy()] = 1
        return M


def hgb():
    return HistGradientBoostingClassifier(random_state=SEED)


# ---------------------------------------------------------------- stacker (E8)
def stack_features(frame: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    _, pc = proba_matrix(frame, "category")
    _, pt = proba_matrix(frame, "assigned_team")
    prod = np.eye(len(PRODUCTS))[meta["product"].map(PRODUCTS.index).to_numpy()]
    chan = np.eye(len(CHANNELS))[meta["channel"].map(CHANNELS.index).to_numpy()]
    return np.hstack([pc, pt, prod, chan])


def team_stacker_producer(source_run: str):
    def produce(train, val, folds, classes):
        oof, vdf = load_run_frame(source_run, "oof_train.csv"), load_run_frame(source_run, "val.csv")
        assert list(oof["ticket_id"]) == list(train["ticket_id"]) and list(vdf["ticket_id"]) == list(val["ticket_id"])
        X_tr, X_val = stack_features(oof, train), stack_features(vdf, val)
        y = train["assigned_team"].to_numpy()
        p_oof = cross_val_predict(hgb(), X_tr, y, cv=PredefinedSplit(folds), method="predict_proba")
        model = hgb().fit(X_tr, y)
        assert list(model.classes_) == list(classes)
        return p_oof, model.predict_proba(X_val)
    return produce


# ---------------------------------------------------------------- experiment registry
def only(targets, factory):
    """make_model that trains `factory(target)` for the given targets and copies the rest."""
    return lambda t: factory(t) if t in targets else None


def tree_selected(t):
    return Pipeline([("text", FunctionTransformer(build_text)), ("tfidf", word_char()),
                     ("select", SelectKBest(chi2, k=2000)),
                     ("dense", FunctionTransformer(to_dense, accept_sparse=True)), ("clf", hgb())])


def template_hgb(t):
    return Pipeline([("factors", TemplateFactors()), ("clf", hgb())])


EXPERIMENTS = {
    "E1_char_tfidf": dict(change="word + char_wb(3,5) TF-IDF, all targets",
                          make_model=lambda t: text_lr(features=word_char)),
    "E2_severity_token": dict(change="SEV_<level> token from customer severity, all targets",
                              make_model=lambda t: text_lr(text_fn=build_text_sev)),
    "E4_sentiment_balanced": dict(change="class_weight=balanced for sentiment",
                                  make_model=only({"sentiment"}, lambda t: text_lr(class_weight="balanced"))),
    "E5_ordinal_priority": dict(change="ordinal cumulative LRs for priority",
                                make_model=only({"priority"}, lambda t: OrdinalLR())),
    "E6_linear_svc": dict(change="LinearSVC + softmax, all targets",
                          make_model=lambda t: Pipeline([("text", FunctionTransformer(build_text)),
                                                         ("tfidf", word_tfidf()),
                                                         ("clf", SoftmaxLinearSVC(random_state=SEED))])),
    "E7_team_cascade": dict(change="team = sum_c P(cat|text) P(team|cat,product)",
                            make_model=only({"assigned_team"}, lambda t: TeamCascade())),
    "E9_tree_selected": dict(change="chi2 top-2000 word+char -> HGB for priority & sentiment",
                             make_model=only({"priority", "sentiment"}, tree_selected)),
    "E10_template_factors": dict(change="template-sentence factors + product/channel -> HGB for priority & sentiment",
                                 make_model=only({"priority", "sentiment"}, template_hgb)),
}
EXPERIMENTS["E11_p1_weighted"] = dict(change="priority: sample_weight 3 for P1 (final priority features = baseline word TF-IDF + LR)",
                                     make_model=only({"priority"}, lambda t: P1WeightedLR(3.0)))
EXPERIMENTS["E11b_p1_weighted_char"] = dict(change="priority: word+char TF-IDF + LR, sample_weight 3 for P1",
                                           make_model=only({"priority"}, lambda t: P1WeightedLR(3.0, "word_char")))
EXPERIMENTS["E5b_ordinal_char"] = dict(change="priority: ordinal cumulative LRs on word+char TF-IDF",
                                       make_model=only({"priority"}, lambda t: OrdinalLR("word_char")))
TUNE_C = [0.3, 3, 10, 30]  # C=1 is baseline_lr
for _t in ("category", "priority", "assigned_team", "sentiment"):
    for _c in TUNE_C:
        EXPERIMENTS[f"E3_C_{_t}_{_c}"] = dict(change=f"C={_c} for {_t}",
                                              make_model=only({_t}, lambda t, c=_c: text_lr(C=c)))

# ---------------------------------------------------------------- combined_v1 (chosen by round-2B selection rules)
# team/priority: no run improved VAL policy-f cost with a CI excluding 0 -> baseline_lr.
# category: no significant macro-F1 gain -> baseline_lr. sentiment: E9 (+0.0052 F1, CI excludes 0).
COMBINED_SOURCES = {"category": "baseline_lr", "priority": "baseline_lr",
                    "assigned_team": "baseline_lr", "sentiment": "E9_tree_selected"}
