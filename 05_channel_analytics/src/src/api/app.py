"""FastAPI app: JSON API under /api plus the built React dashboard (frontend/dist).

    python -m src.serve            -> http://localhost:8000
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..data_prep import ROOT
from . import analytics as A
from .state import AppState

log = logging.getLogger("channel-api")
DIST = ROOT / "frontend" / "dist"


class Holder:
    """Current state plus pipeline status. The state is swapped atomically after a successful rebuild."""

    def __init__(self) -> None:
        self.state: AppState | None = None
        self.lock = threading.Lock()
        self.status = {"running": False, "error": None, "progress": None, "started_at": None, "finished_at": None}

    def set_progress(self, step: str) -> None:
        self.status["progress"] = step
        log.info("analysis: %s", step)

    def build(self) -> AppState:
        return AppState(progress=self.set_progress)


holder = Holder()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if holder.state is None:
        t0 = time.time()
        holder.state = holder.build()
        log.info("analysis ready in %.0fs", time.time() - t0)
    yield


app = FastAPI(title="Partner Channel Analytics API", version="2.0", lifespan=lifespan)


def ok(payload, status_code: int = 200) -> JSONResponse:
    return JSONResponse(A.clean(payload), status_code=status_code)


def state() -> AppState:
    if holder.state is None:
        raise HTTPException(503, "analysis not ready")
    return holder.state


def filters(region, tier, product_family, partner_type, q_from, q_to) -> A.Filters:
    return A.Filters.parse(region, tier, product_family, partner_type, q_from, q_to)


# ---------------------------------------------------------------- data endpoints

@app.get("/api/meta")
def get_meta():
    return ok(A.meta(state()))


@app.get("/api/kpis")
def get_kpis(region: Optional[str] = None, tier: Optional[str] = None, product_family: Optional[str] = None,
             partner_type: Optional[str] = None, q_from: Optional[str] = None, q_to: Optional[str] = None):
    return ok(A.kpis(state(), filters(region, tier, product_family, partner_type, q_from, q_to)))


@app.get("/api/trend")
def get_trend(grain: str = Query("quarter", pattern="^(quarter|month)$"),
              by: str = Query("region", pattern="^(region|tier|product_family|partner_type)$"),
              region: Optional[str] = None, tier: Optional[str] = None, product_family: Optional[str] = None,
              partner_type: Optional[str] = None, q_from: Optional[str] = None, q_to: Optional[str] = None):
    return ok(A.trend(state(), filters(region, tier, product_family, partner_type, q_from, q_to), grain, by))


@app.get("/api/breakdown")
def get_breakdown(dim: str = Query("region", pattern="^(region|tier|product_family|partner_type)$"),
                  region: Optional[str] = None, tier: Optional[str] = None, product_family: Optional[str] = None,
                  partner_type: Optional[str] = None, q_from: Optional[str] = None, q_to: Optional[str] = None):
    return ok(A.breakdown(state(), filters(region, tier, product_family, partner_type, q_from, q_to), dim))


@app.get("/api/partners")
def get_partners(search: Optional[str] = None, segment: Optional[str] = None, sort: str = "revenue_range", desc: bool = True,
                 page: int = 1, page_size: int = 25,
                 region: Optional[str] = None, tier: Optional[str] = None, product_family: Optional[str] = None,
                 partner_type: Optional[str] = None, q_from: Optional[str] = None, q_to: Optional[str] = None):
    f = filters(region, tier, product_family, partner_type, q_from, q_to)
    return ok(A.partners(state(), f, search, segment, sort, desc, page, page_size))


@app.get("/api/partners/{partner_id}")
def get_partner(partner_id: str):
    detail = A.partner_detail(state(), partner_id)
    if detail is None:
        raise HTTPException(404, f"unknown partner {partner_id}")
    return ok(detail)


@app.get("/api/at-risk")
def get_at_risk():
    return ok(A.at_risk_summary(state()))


@app.get("/api/forecast")
def get_forecast():
    return ok(A.forecast_payload(state()))


@app.get("/api/insights")
def get_insights():
    s = state()
    return ok({"available": s.insights is not None, "cards": s.insights or [], "error": s.errors.get("insights")})


# ---------------------------------------------------------------- pipeline

def _run_pipeline() -> None:
    try:
        new_state = holder.build()
        with holder.lock:
            holder.state = new_state  # atomic swap; the old state served requests until now
        holder.status.update(error=None, progress="done")
    except Exception as exc:  # noqa: BLE001 - keep serving the previous state
        log.exception("analysis rebuild failed")
        holder.status.update(error=f"{type(exc).__name__}: {exc}", progress="failed")
    finally:
        holder.status.update(running=False, finished_at=time.time())


@app.post("/api/pipeline/run", status_code=202)
def run_pipeline():
    with holder.lock:
        if holder.status["running"]:
            raise HTTPException(409, "analysis already running")
        holder.status.update(running=True, error=None, progress="starting", started_at=time.time(), finished_at=None)
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return ok({"accepted": True}, status_code=202)


@app.get("/api/pipeline/status")
def pipeline_status():
    s = holder.state
    return ok({**holder.status, "generated_at": s.generated_at if s else None})


# ---------------------------------------------------------------- frontend (SPA)

@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(404, "unknown API route")
    index = DIST / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Frontend not built</h1><p>Run <code>./start.sh</code> or "
                            "<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>.</p>", status_code=503)
    target = (DIST / full_path).resolve()
    if full_path and target.is_file() and DIST.resolve() in target.parents:
        return FileResponse(target)
    return FileResponse(index)  # client-side routes such as /partners/PT-00001
