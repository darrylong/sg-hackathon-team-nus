"""Run with: .venv/bin/python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.data_prep import build_all
from src.features import (FEATURES, SCORE_CUTOFF, TRAIN_CUTOFFS, build_features, feature_dictionary,
                          quarter_end, shift_quarter)
from src.labels import build_labels


@pytest.fixture(scope="module")
def tables():
    return build_all()


def truncate(tables, cutoff):
    """Drop every row after the cutoff date, as if the future had not happened yet."""
    end = quarter_end(cutoff)
    t = dict(tables)
    t["sales"] = tables["sales"][tables["sales"]["order_date"] <= end]
    t["partner_month"] = tables["partner_month"][tables["partner_month"]["month"] <= end]
    t["partner_quarter"] = tables["partner_quarter"][tables["partner_quarter"]["quarter"] <= cutoff]
    t["targets"] = tables["targets"][tables["targets"]["quarter"] <= cutoff]
    return t


@pytest.mark.parametrize("cutoff", ["2024-Q4", "2025-Q3", "2026-Q1"])
def test_no_future_leakage(tables, cutoff):
    full = build_features(tables, cutoff)
    past_only = build_features(truncate(tables, cutoff), cutoff)
    pd.testing.assert_frame_equal(full, past_only)


def test_score_snapshot_covers_every_partner(tables):
    X = build_features(tables, SCORE_CUTOFF)
    assert len(X) == len(tables["master"]) == X["partner_id"].nunique()
    assert set(X["partner_id"]) == set(tables["master"]["partner_id"])


def test_population_is_onboarded_partners(tables):
    for c in TRAIN_CUTOFFS:
        X = build_features(tables, c)
        onboarded = tables["master"].loc[tables["master"]["onboarded_date"] <= quarter_end(c), "partner_id"]
        assert set(X["partner_id"]) == set(onboarded)


def test_no_infinite_values(tables):
    X = build_features(tables, SCORE_CUTOFF)
    num = X[FEATURES].select_dtypes("number")
    assert not np.isinf(num.to_numpy()).any()


def test_bounded_features(tables):
    X = build_features(tables, SCORE_CUTOFF)
    for c in ["rev_growth_qoq", "rev_growth_2q", "yoy_q1", "yoy_2q", "lines_yoy_2q"]:
        assert X[c].dropna().between(-1, 1).all(), c
    for c in [c for c in FEATURES if c.startswith("share_")] + ["product_hhi_2q", "active_month_share_6m"]:
        assert X[c].dropna().between(0, 1).all(), c


def test_feature_dictionary_matches_columns(tables):
    X = build_features(tables, SCORE_CUTOFF)
    assert list(X.columns) == ["partner_id", "cutoff"] + feature_dictionary()["feature"].tolist()


def test_labels_require_next_quarter(tables):
    with pytest.raises(ValueError):
        build_labels(tables, SCORE_CUTOFF)


def test_label_definition(tables):
    c = "2025-Q2"
    y = build_labels(tables, c).set_index("partner_id")
    nq = shift_quarter(c, 1)
    pq = tables["partner_quarter"].set_index(["partner_id", "quarter"])
    zero = y.index[y["left_zero"] == 1]
    assert (pq.loc[[(p, nq) for p in zero], "order_lines"] == 0).all()
    assert (y["left_zero"] <= y["left_token"]).all()
    assert (y["left_persistent"] <= y["left_token"]).all()
