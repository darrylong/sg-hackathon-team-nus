"""In-memory application state: data, health scores, forecast and insights, computed once at startup."""
from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import at_risk, forecast
from ..data_prep import DATA_DIR, build_all

log = logging.getLogger("channel-api")


class AppState:
    """Built in this order; only step 1 is mandatory. A failed later step leaves that section as None."""

    def __init__(self, data_dir: Path | str = DATA_DIR, progress: Callable[[str], None] | None = None) -> None:
        progress = progress or (lambda step: None)
        self.errors: dict[str, str] = {}

        progress("loading data")
        try:
            self.tables = build_all(data_dir)
        except Exception as exc:  # noqa: BLE001 - surface a clear message for the one mandatory step
            raise RuntimeError(f"Could not load data from {Path(data_dir).resolve()}: {exc}") from exc
        sales = self.tables["sales"]
        self.sales = sales[~sales["is_outlier"]].copy()
        for c in ["region", "tier", "product_family", "partner_type"]:
            self.sales[c] = self.sales[c].astype(str)
        self.pq = self.tables["partner_quarter"].copy()
        for c in ["region", "tier", "partner_type"]:
            self.pq[c] = self.pq[c].astype(str)

        progress("training the at-risk model")
        self.at_risk = self._safe("at_risk", lambda: at_risk.compute(self.tables))

        progress("forecasting 2026-Q3")
        self.forecast = self._safe("forecast", lambda: forecast.compute(self.tables, self.at_risk))

        progress("finding declining partners")
        self.declining = self._safe("declining", self._declining)

        progress("building insights")
        from .insights import build_insights  # local import: insights reads this state
        self.insights = self._safe("insights", lambda: build_insights(self))

        self.generated_at = datetime.now()
        progress("done")

    def _safe(self, name: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one broken section must not take the dashboard down
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            log.error("section %s failed:\n%s", name, traceback.format_exc())
            return None

    def _declining(self) -> pd.DataFrame | None:
        if self.at_risk is None:
            return None
        X = self.at_risk.X.set_index("partner_id")
        ids = X.index[X["yoy_decline_streak"] >= 4]
        s = self.at_risk.scores.set_index("partner_id").loc[ids]
        out = s.assign(yoy_decline_streak=X.loc[ids, "yoy_decline_streak"], rev_drawdown=X.loc[ids, "rev_drawdown"])
        return out.reset_index().sort_values("yoy_h1_pct")
