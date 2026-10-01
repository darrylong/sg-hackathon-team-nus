"""Start the dashboard: API + built frontend on one port.

    python -m src.serve [--host 127.0.0.1] [--port 8000]
"""
from __future__ import annotations

import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    print(f"Starting on http://localhost:{args.port} (the analysis takes ~15 s before the first page loads)")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
