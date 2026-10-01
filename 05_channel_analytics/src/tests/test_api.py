"""API tests. Run with: .venv/bin/python -m pytest -q tests/test_api.py"""
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.at_risk import SUBMISSION_DIR

ENDPOINTS = {
    "/api/meta": ["regions", "quarters", "generated_at", "sections"],
    "/api/kpis": ["revenue", "yoy", "margin_pct", "attainment", "at_risk", "forecast"],
    "/api/trend?by=product_family&grain=month": ["keys", "rows"],
    "/api/breakdown?dim=tier": ["rows", "growth_label"],
    "/api/partners": ["total", "rows"],
    "/api/partners/PT-00001": ["profile", "score", "reasons", "quarterly", "monthly"],
    "/api/at-risk": ["summary", "flagged", "drivers", "segments"],
    "/api/forecast": ["submission", "history", "methods", "backtest_mape", "churn"],
    "/api/insights": ["cards"],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:  # lifespan builds the state once (~10 s)
        yield c


@pytest.mark.parametrize("path,keys", ENDPOINTS.items())
def test_endpoints_ok_and_json_safe(client, path, keys):
    r = client.get(path)
    assert r.status_code == 200, r.text
    assert "NaN" not in r.text and "Infinity" not in r.text
    body = r.json()
    for k in keys:
        assert k in body, f"{k} missing from {path}"


def test_region_subtotals_add_up(client):
    total = client.get("/api/kpis").json()["revenue"]
    parts = sum(client.get(f"/api/kpis?region={r}").json()["revenue"] for r in ["AMS", "APJ", "EMEA"])
    assert abs(total - parts) < 1


def test_empty_filters_return_empty_lists(client):
    p = client.get("/api/partners?tier=Business&partner_type=Distributor")
    assert p.status_code == 200 and p.json()["total"] == 0 and p.json()["rows"] == []
    t = client.get("/api/trend?q_from=2026-Q2&q_to=2025-Q1")
    assert t.status_code == 200 and t.json()["rows"] == []
    b = client.get("/api/breakdown?q_from=2026-Q2&q_to=2025-Q1")
    assert b.status_code == 200 and b.json()["rows"] == []


def test_at_risk_count_matches_submission(client):
    sub = pd.read_csv(SUBMISSION_DIR / "submission_at_risk.csv")
    body = client.get("/api/at-risk").json()
    assert len(body["flagged"]) == body["summary"]["flagged"] == int(sub["at_risk"].sum())
    assert client.get("/api/kpis").json()["at_risk"]["flagged"] == int(sub["at_risk"].sum())


def test_forecast_matches_submission(client):
    sub = pd.read_csv(SUBMISSION_DIR / "submission_forecast.csv").set_index("region")
    for row in client.get("/api/forecast").json()["submission"]:
        s = sub.loc[row["region"]]
        assert (row["forecast"], row["lo80"], row["hi80"]) == (s["forecast_revenue_usd"], s["lo80_usd"], s["hi80_usd"])


def test_flagged_partners_have_reasons(client):
    for row in client.get("/api/at-risk").json()["flagged"]:
        assert row["reason_1"], row["partner_id"]


def test_unknown_partner_404(client):
    assert client.get("/api/partners/PT-99999").status_code == 404
    assert client.get("/api/does-not-exist").status_code == 404


def test_spa_fallback_serves_index(client):
    if not (app_module.DIST / "index.html").exists():
        pytest.skip("frontend not built")
    r = client.get("/partners/PT-00001")
    assert r.status_code == 200 and '<div id="root">' in r.text


def _wait_until_idle(client, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get("/api/pipeline/status").json()
        if not s["running"]:
            return s
        time.sleep(0.1)
    raise AssertionError("pipeline did not finish")


def test_pipeline_run_success_swaps_state(client, monkeypatch):
    old = app_module.holder.state

    def fake_build():
        time.sleep(0.5)
        return old  # same data, but goes through the async path + swap

    monkeypatch.setattr(app_module.holder, "build", fake_build)
    assert client.post("/api/pipeline/run").status_code == 202
    assert client.post("/api/pipeline/run").status_code == 409  # already running
    s = _wait_until_idle(client)
    assert s["error"] is None and s["progress"] == "done"
    assert app_module.holder.state is old


def test_pipeline_failure_keeps_previous_state(client, monkeypatch):
    old = app_module.holder.state

    def broken_build():
        raise RuntimeError("Could not load data from /nowhere")

    monkeypatch.setattr(app_module.holder, "build", broken_build)
    assert client.post("/api/pipeline/run").status_code == 202
    s = _wait_until_idle(client)
    assert "Could not load data" in s["error"]
    assert app_module.holder.state is old
    assert client.get("/api/kpis").status_code == 200  # still serving the old analysis
