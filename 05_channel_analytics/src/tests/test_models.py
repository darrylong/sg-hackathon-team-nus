"""Run with: .venv/bin/python -m pytest -q   (expects `python -m src.run_all` to have been run once)"""
import numpy as np
import pandas as pd
import pytest

from src.at_risk import MODEL_DIR, SUBMISSION_DIR, scale_odds, sym_to_pct, validate_at_risk
from src.data_prep import REGIONS, build_all
from src import churn_adjustment as churn
from src.compare_versions import V1_DIR, consistency
from src.forecast import ENSEMBLES, backtest, forecast_all, load_inputs, project_target, validate_forecast


@pytest.fixture(scope="module")
def tables():
    return build_all()


def test_submission_at_risk_valid(tables):
    sub = pd.read_csv(SUBMISSION_DIR / "submission_at_risk.csv")
    validate_at_risk(sub, tables["master"])
    assert 0 < sub["at_risk"].sum() < len(sub)


def test_flagged_partners_have_reasons():
    scores = pd.read_csv(MODEL_DIR / "partner_scores.csv")
    flagged = scores[scores["at_risk"] == 1]
    assert flagged["reason_1"].notna().all() and (flagged["reason_1"].str.len() > 10).all()
    # flags are the highest-risk partners
    assert flagged["p_leave"].min() >= scores.loc[scores["at_risk"] == 0, "p_leave"].max()


def test_health_score_orders_like_risk():
    scores = pd.read_csv(MODEL_DIR / "partner_scores.csv")
    assert np.allclose(scores["health_score"], 100 * (1 - scores["p_leave"]), atol=1e-3)


def test_submission_forecast_valid():
    sub = pd.read_csv(SUBMISSION_DIR / "submission_forecast.csv")
    validate_forecast(sub)


@pytest.fixture(scope="module")
def inputs(tables):
    return load_inputs(tables)


def test_forecast_uses_only_past_data(inputs):
    quarterly, monthly, targets_q = inputs
    full = forecast_all(quarterly, monthly, targets_q, "2025-Q4")
    past = forecast_all(quarterly[quarterly.index < "2025-Q4"], monthly[monthly.index < "2025-10-01"],
                        targets_q[targets_q.index < "2025-Q4"], "2025-Q4")
    pd.testing.assert_frame_equal(full, past)


def test_project_target_ignores_target_quarter(inputs):
    quarterly, monthly, targets_q = inputs
    tampered = targets_q.copy()
    tampered.loc["2025-Q4"] *= 10  # the real target of the forecast quarter must not matter
    pd.testing.assert_frame_equal(forecast_all(quarterly, monthly, targets_q, "2025-Q4"),
                                  forecast_all(quarterly, monthly, tampered, "2025-Q4"))
    assert np.isfinite(project_target(targets_q["AMS"], "2026-Q3"))


def test_all_methods_produce_forecasts(inputs):
    quarterly, monthly, targets_q = inputs
    fc = forecast_all(quarterly, monthly, targets_q, "2026-Q3")
    assert fc.notna().all().all()
    for e in ENSEMBLES:
        assert np.isclose(fc.loc["ALL", e], fc.loc[REGIONS, e].sum())


def test_churn_adjustment_zero_when_normal():
    share = pd.Series(0.004, index=REGIONS)
    f = pd.Series([700e6, 190e6, 225e6], index=REGIONS)
    adj = churn.adjustment(share, share, f)
    assert np.allclose(adj.to_numpy(), 0)
    adj2 = churn.adjustment(share * 2, share, f)
    assert (adj2[REGIONS] < 0).all() and np.isclose(adj2["ALL"], adj2[REGIONS].sum())


@pytest.mark.skipif(not (V1_DIR / "outputs" / "models" / "forecast_backtest.csv").exists(), reason="v1 outputs absent")
def test_core3_matches_v1(inputs):
    bt1 = pd.read_csv(V1_DIR / "outputs" / "models" / "forecast_backtest.csv")
    ok, tbl = consistency(bt1, backtest(*inputs))
    assert ok, tbl


def test_helpers():
    assert sym_to_pct(0) == 0
    assert np.isclose(sym_to_pct(1 / 3), 1.0)  # (2 - 1) / (2 + 1) -> +100%
    p = np.array([0.01, 0.1, 0.5])
    assert np.all(scale_odds(p, 1.0) == p) and np.all(scale_odds(p, 2.0) > p)


def test_in_memory_compute_reproduces_submissions(tables):
    """The API computes in memory; it must give exactly the submitted files."""
    from src import at_risk, forecast
    ar = at_risk.compute(tables)
    sub_risk = pd.read_csv(SUBMISSION_DIR / "submission_at_risk.csv")
    pd.testing.assert_frame_equal(at_risk.submission_frame(ar).reset_index(drop=True), sub_risk, check_dtype=False)
    fc = forecast.compute(tables, ar)
    sub_fc = pd.read_csv(SUBMISSION_DIR / "submission_forecast.csv")
    pd.testing.assert_frame_equal(fc.sub.reset_index(drop=True), sub_fc, check_dtype=False)
