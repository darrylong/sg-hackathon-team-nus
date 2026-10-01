"""Regenerate every output and both submission files in one go.

    python -m src.run_all          (about 20 s)
"""
from __future__ import annotations

import time
from typing import Callable

from . import at_risk, compare_versions, data_prep, eda, features, forecast


def main(progress: Callable[[str], None] | None = None) -> None:
    state: dict = {}
    steps = [
        ("data prep", lambda: (state.__setitem__("tables", data_prep.build_all()), data_prep.save_all(state["tables"]))),
        ("EDA", eda.run),
        ("features", features.main),
        ("at-risk model", lambda: state.__setitem__("at_risk", at_risk.run(state["tables"]))),
        ("forecast", lambda: forecast.run(state["tables"], state["at_risk"])),
        ("v1 vs v2 comparison", compare_versions.run),
    ]
    for name, step in steps:
        if progress:
            progress(name)
        t0 = time.time()
        print(f"\n##### {name}")
        step()
        print(f"##### {name} done in {time.time() - t0:.0f}s")
    print(f"\nSubmissions: {at_risk.SUBMISSION_DIR}")


if __name__ == "__main__":
    main()
