"""Ticket triage API.

Run from 03_ticket_triage/:   uvicorn app.api:app --port 8000      (or: python app/api.py)
Bundle: env TRIAGE_BUNDLE, else "bundle_dir" in app/config.json, else models/bundle_v3 (relative to 03_ticket_triage/).

GET  /health          liveness + whether the bundle loaded
GET  /model_info      bundle manifest + headline VAL metrics
POST /predict         one ticket (JSON); no capacity cap: abstain iff expected savings > 0
POST /triage          alias of /predict
POST /predict_batch   CSV upload (ticket_id, channel, subject, body, product); 15% abstain cap;
                      JSON with a download_url for the submission-format CSV (or ?format=csv for the CSV itself)
"""
import io
import json
import os
import sys
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference import Triage, to_records, to_submission  # noqa: E402

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
DEFAULT_BUNDLE = "models/bundle_v3"
BATCH_COLUMNS = ["ticket_id", "channel", "subject", "body", "product"]
CHANNELS = ("email", "portal", "chat")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_STORED_BATCHES = 20


def bundle_path() -> Path:
    p = os.environ.get("TRIAGE_BUNDLE")
    if not p and CONFIG_FILE.exists():
        p = json.loads(CONFIG_FILE.read_text()).get("bundle_dir")
    p = Path(p or DEFAULT_BUNDLE)
    return p if p.is_absolute() else ROOT / p


STATE = {"triage": None, "error": None, "bundle": None}
BATCHES: "OrderedDict[str, str]" = OrderedDict()


@asynccontextmanager
async def lifespan(_app):
    STATE["bundle"] = bundle_path()
    try:
        STATE["triage"] = Triage.load(STATE["bundle"])
    except Exception as e:  # keep serving /health so the failure is visible
        STATE["error"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="Ticket Triage API", version="1.0", lifespan=lifespan)


def triage() -> Triage:
    if STATE["triage"] is None:
        raise HTTPException(503, f"model bundle not loaded from {STATE['bundle']}: {STATE['error']}")
    return STATE["triage"]


# ---------------------------------------------------------------- schemas
class TicketIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "subject": "Array offline",
        "body": "Our Alletra 6010 went offline after the firmware update and the payments cluster is down.",
        "product": "Alletra 6010", "channel": "email"}})
    ticket_id: str | None = Field(None, max_length=100)
    subject: str = Field("", max_length=2000)
    body: str = Field(..., max_length=50_000, description="may be empty; very short bodies are sent to a human")
    product: str = Field(..., max_length=200)
    channel: Literal["email", "portal", "chat"]

    @field_validator("channel", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class TicketOut(BaseModel):
    ticket_id: str
    category: str
    priority: str
    assigned_team: str
    sentiment: str
    probabilities: dict[str, dict[str, float]]
    p_correct: float
    expected_route_cost: float
    expected_abstain_cost: float
    savings: float
    escalation: bool
    guard_flags: list[str]
    abstain: int
    decision: Literal["route", "abstain"]
    reason: str


class BatchOut(BaseModel):
    batch_id: str
    n_tickets: int
    n_abstain: int
    abstain_rate: float
    cap: float
    max_abstains: int
    n_guarded: int
    notes: list[str]
    download_url: str
    predictions: list[TicketOut]


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    ok = STATE["triage"] is not None
    return {"status": "ok" if ok else "error", "bundle_loaded": ok, "bundle_dir": str(STATE["bundle"]),
            "error": STATE["error"]}


@app.get("/model_info")
def model_info():
    m = triage().manifest
    return {"bundle_dir": str(STATE["bundle"]), "headline_val_metrics": m.get("val_metrics"),
            "manifest": {k: v for k, v in m.items() if k != "val_metrics"}}


@app.post("/predict", response_model=TicketOut)
@app.post("/triage", response_model=TicketOut, summary="Triage (alias of /predict)")
def predict(ticket: TicketIn):
    tri = triage()
    df = pd.DataFrame([ticket.model_dump()])
    if df.at[0, "ticket_id"] is None:
        df = df.drop(columns="ticket_id")
    pred = tri.predict(df, cap=None)
    return to_records(pred, tri.classes)[0]


def _bad(msg: str):
    raise HTTPException(422, msg)


def read_batch_csv(raw: bytes) -> pd.DataFrame:
    if not raw.strip():
        _bad("uploaded file is empty")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        _bad("file is not valid UTF-8")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as e:
        _bad(f"could not parse CSV: {e}")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in BATCH_COLUMNS if c not in df.columns]
    if missing:
        _bad(f"CSV is missing required column(s): {', '.join(missing)}. "
             f"Expected columns: {', '.join(BATCH_COLUMNS)}; found: {', '.join(df.columns)}")
    if df.empty:
        _bad("CSV has a header but no rows")
    df = df[BATCH_COLUMNS].copy()
    df["ticket_id"] = df["ticket_id"].str.strip()
    blank = df.index[df["ticket_id"] == ""]
    if len(blank):
        _bad(f"blank ticket_id on data row(s) {', '.join(str(i + 1) for i in blank[:10])}")
    dup = df.loc[df["ticket_id"].duplicated(), "ticket_id"].unique()
    if len(dup):
        _bad(f"duplicate ticket_id(s): {', '.join(dup[:10])}{' ...' if len(dup) > 10 else ''}")
    ch = df["channel"].str.strip().str.lower()
    bad = df.loc[~ch.isin(CHANNELS), ["ticket_id", "channel"]]
    if len(bad):
        ex = ", ".join(f"{r.ticket_id}={r.channel!r}" for r in bad.head(10).itertuples())
        _bad(f"{len(bad)} row(s) with invalid channel (expected {', '.join(CHANNELS)}): {ex}")
    df["channel"] = ch
    return df


@app.post("/predict_batch", response_model=BatchOut,
          responses={200: {"content": {"text/csv": {}}, "description": "JSON, or the CSV with ?format=csv"}})
async def predict_batch(file: UploadFile = File(..., description="CSV with ticket_id, channel, subject, body, product"),
                        format: Literal["json", "csv"] = Query("json")):
    tri = triage()
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        _bad(f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    df = read_batch_csv(raw)
    pred = tri.predict(df, cap=tri.cap)
    csv_text = to_submission(pred).to_csv(index=False)
    batch_id = uuid.uuid4().hex[:12]
    BATCHES[batch_id] = csv_text
    while len(BATCHES) > MAX_STORED_BATCHES:
        BATCHES.popitem(last=False)
    if format == "csv":
        return _csv_response(csv_text, batch_id)
    n = len(pred)
    return {"batch_id": batch_id, "n_tickets": n, "n_abstain": int(pred["abstain"].sum()),
            "abstain_rate": float(pred["abstain"].mean()), "cap": tri.cap,
            "max_abstains": pred.attrs["slots"], "n_guarded": pred.attrs["n_guarded"],
            "notes": pred.attrs["notes"], "download_url": f"/predict_batch/{batch_id}/submission.csv",
            "predictions": to_records(pred, tri.classes)}


def _csv_response(text: str, batch_id: str) -> Response:
    return Response(text, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="submission_{batch_id}.csv"'})


@app.get("/predict_batch/{batch_id}/submission.csv")
def download_submission(batch_id: str):
    if batch_id not in BATCHES:
        raise HTTPException(404, f"unknown or expired batch_id {batch_id!r} (last {MAX_STORED_BATCHES} kept)")
    return _csv_response(BATCHES[batch_id], batch_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
