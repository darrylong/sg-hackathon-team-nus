"""API smoke tests (needs the bundle).  python -m pytest tests/test_api.py -v"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api import app  # noqa: E402

TICKET = {"subject": "Array offline", "body": "Our Alletra 6010 went offline after the firmware update and "
          "the payments cluster is down.", "product": "Alletra 6010", "channel": "email"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        if not c.get("/health").json()["bundle_loaded"]:
            pytest.fail(c.get("/health").json()["error"])
        yield c


def test_health_and_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model_info").json()
    assert info["manifest"]["run"] == "combined_v3"
    assert "policy_f" in info["headline_val_metrics"]


def test_predict(client):
    r = client.post("/predict", json=TICKET)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in ("route", "abstain") and body["reason"]
    assert set(body["probabilities"]) == {"category", "priority", "assigned_team", "sentiment"}


def test_triage_alias(client):
    r = client.post("/triage", json=TICKET)
    assert r.status_code == 200, r.text
    assert r.json() == client.post("/predict", json=TICKET).json()
    assert client.post("/triage", json={**TICKET, "channel": "fax"}).status_code == 422
    paths = client.get("/openapi.json").json()["paths"]
    assert "/triage" in paths and paths["/triage"]["post"]["responses"]["200"] == paths["/predict"]["post"]["responses"]["200"]


def test_predict_validation(client):
    assert client.post("/predict", json={**TICKET, "channel": "fax"}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in TICKET.items() if k != "body"}).status_code == 422
    r = client.post("/predict", json={**TICKET, "body": "", "channel": "CHAT"})
    assert r.status_code == 200 and "short_body" in r.json()["guard_flags"] and r.json()["abstain"] == 1


def test_predict_batch(client):
    rows = [{"ticket_id": f"X-{i}", **TICKET} for i in range(20)]
    rows[3]["body"] = "hi"
    csv = pd.DataFrame(rows).to_csv(index=False).encode()
    r = client.post("/predict_batch", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["n_tickets"] == 20 and out["max_abstains"] == 3 and out["n_abstain"] <= 3
    sub = pd.read_csv(io.StringIO(client.get(out["download_url"]).text))
    assert list(sub.columns) == ["ticket_id", "category", "priority", "assigned_team", "sentiment", "abstain"]
    assert sub.loc[3, "abstain"] == 1
    r = client.post("/predict_batch?format=csv", files={"file": ("t.csv", csv, "text/csv")})
    assert r.headers["content-type"].startswith("text/csv")


def test_predict_batch_missing_columns(client):
    csv = b"ticket_id,subject\nX-1,hello\n"
    r = client.post("/predict_batch", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 422 and "channel, body, product" in r.json()["detail"]
