"""Start the triage API (uvicorn app.api:app) and the Streamlit UI together.

python start.py [--api-port 8000] [--ui-port 8501] [--host 127.0.0.1]
Waits for the API's /health, then starts the UI, prints both URLs, and stops both on Ctrl+C
(or when either process exits). Works on Windows, macOS and Linux.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_START_TIMEOUT = 300  # loading the model bundle can take a while on a cold disk
UI_START_TIMEOUT = 60


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=2) as r:
        return json.loads(r.read())


def get_status(url: str) -> int:
    with urllib.request.urlopen(url, timeout=2) as r:
        return r.status


def spawn(cmd, env=None):
    kw = {}
    if os.name == "nt":  # own process group: Ctrl+C reaches only us, and we stop the children
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, cwd=ROOT, env=env, **kw)


def wait_until(check, proc, timeout, what):
    """Poll check() until it returns a truthy value; fail if proc dies or the timeout passes."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited with code {proc.returncode} during startup (see output above)")
        try:
            result = check()
            if result:
                return result
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{what} did not become ready within {timeout} s")


def stop(procs):
    for name, p in procs:
        if p.poll() is None:
            print(f"Stopping {name} (pid {p.pid})...", flush=True)
            p.terminate()
    for name, p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"{name} did not stop; killing it.", flush=True)
            p.kill()
            p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    a = ap.parse_args()
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break / console close -> same clean shutdown as Ctrl+C
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    api_url, ui_url = f"http://{a.host}:{a.api_port}", f"http://{a.host}:{a.ui_port}"
    for port, what in ((a.api_port, "API"), (a.ui_port, "UI")):
        if not port_free(a.host, port):
            sys.exit(f"Port {port} for the {what} is already in use. Stop whatever is using it, "
                     f"or pass --{what.lower()}-port <other port>.")

    procs = []
    try:
        print(f"Starting API on {api_url} ...", flush=True)
        api = spawn([sys.executable, "-m", "uvicorn", "app.api:app", "--host", a.host, "--port", str(a.api_port)])
        procs.append(("API", api))
        health = wait_until(lambda: get_json(f"{api_url}/health"), api, API_START_TIMEOUT, "API")
        if not health.get("bundle_loaded"):
            print(f"WARNING: API is up but the model bundle did not load: {health.get('error')}", flush=True)

        print(f"Starting UI on {ui_url} ...", flush=True)
        env = {**os.environ, "TRIAGE_API_URL": api_url}
        ui = spawn([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "frontend.py"),
                    "--server.address", a.host, "--server.port", str(a.ui_port),
                    "--server.headless", "true", "--browser.gatherUsageStats", "false"], env=env)
        procs.append(("UI", ui))
        wait_until(lambda: get_status(f"{ui_url}/_stcore/health") == 200, ui, UI_START_TIMEOUT, "UI")

        print("\n" + "=" * 60)
        print(f"  UI : {ui_url}")
        print(f"  API: {api_url}   (docs: {api_url}/docs)")
        print("  Press Ctrl+C to stop both.")
        print("=" * 60 + "\n", flush=True)
        while True:
            for name, p in procs:
                if p.poll() is not None:
                    raise RuntimeError(f"{name} exited unexpectedly with code {p.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C received.", flush=True)
    except RuntimeError as e:
        print(f"ERROR: {e}", flush=True)
        stop(procs)
        sys.exit(1)
    stop(procs)
    print("Stopped.")


if __name__ == "__main__":
    main()
