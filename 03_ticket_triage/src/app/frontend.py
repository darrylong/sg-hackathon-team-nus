"""Streamlit front end for the ticket triage API.

Run from 03_ticket_triage/:   python start.py      (API + UI together)
or, with the API already up:   streamlit run app/frontend.py
API address: env TRIAGE_API_URL (default http://127.0.0.1:8000).
"""
import html
import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("TRIAGE_API_URL", "http://127.0.0.1:8000").rstrip("/")
BATCH_COLUMNS = ["ticket_id", "channel", "subject", "body", "product"]
CHANNELS = ["email", "portal", "chat"]
OTHER_PRODUCT = "Other (not listed)…"
TIMEOUT = 120

# HPE palette (Problem 05 dashboard look)
BRAND, HEALTHY, DECLINING, AT_RISK, ACCENT = "#0072CE", "#01A982", "#FFB81C", "#FF5042", "#7630EA"
TEXT, MUTED, CARD_BG, GRID = "#1A1A2E", "#6B6B7B", "#FFFFFF", "#EEEEEE"
DECISION_STYLE = {"ROUTE": (HEALTHY, "#FFFFFF"), "SEND TO HUMAN": (DECLINING, TEXT)}  # (background, text)


class ApiError(Exception):
    """Error with a message that is safe to show to the user."""


# ---------------------------------------------------------------- API helpers
def api(method: str, path: str, **kw):
    """Call the API; returns the requests.Response or raises ApiError with a friendly message."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kw)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the triage API at {API_URL}. Start it with `python start.py` "
                       f"(or `uvicorn app.api:app --port 8000`) and retry.") from None
    except requests.Timeout:
        raise ApiError(f"The triage API at {API_URL} did not answer within {TIMEOUT} s.") from None
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):  # pydantic validation errors
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
        raise ApiError(f"API error {r.status_code}: {detail}")
    return r


@st.cache_data(ttl=60, show_spinner=False)
def model_info() -> dict:
    return api("GET", "/model_info").json()  # exceptions are not cached, so a later retry works


def check_batch_columns(raw: bytes) -> pd.DataFrame:
    """Parse the upload locally so missing columns get a clear message before anything is sent."""
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ApiError("The file is not valid UTF-8 text. Save it as a UTF-8 CSV and upload again.") from None
    except Exception as e:
        raise ApiError(f"Could not read the file as CSV: {e}") from None
    cols = [c.strip() for c in df.columns]
    missing = [c for c in BATCH_COLUMNS if c not in cols]
    if missing:
        raise ApiError(f"The CSV is missing required column(s): {', '.join(missing)}. "
                       f"Required: {', '.join(BATCH_COLUMNS)}. Found: {', '.join(cols) or '(none)'}.")
    if df.empty:
        raise ApiError("The CSV has a header row but no tickets.")
    return df


def predict_one(subject: str, body: str, product: str, channel: str) -> dict:
    return api("POST", "/predict", json={"subject": subject, "body": body, "product": product,
                                         "channel": channel}).json()


def predict_batch(raw: bytes, filename: str = "batch.csv") -> tuple[dict, bytes]:
    """Returns (batch JSON, submission CSV bytes)."""
    check_batch_columns(raw)
    res = api("POST", "/predict_batch", files={"file": (filename, raw, "text/csv")}).json()
    csv = api("GET", res["download_url"]).content
    return res, csv


def batch_table(res: dict) -> pd.DataFrame:
    rows = []
    for p in res["predictions"]:
        rows.append({"ticket_id": p["ticket_id"],
                     "decision": "SEND TO HUMAN" if p["abstain"] else "ROUTE",
                     "assigned_team": p["assigned_team"], "category": p["category"], "priority": p["priority"],
                     "sentiment": p["sentiment"], "team_confidence": p["p_correct"],
                     "unhappy_customer": p["escalation"], "guard_flags": ", ".join(p["guard_flags"]),
                     "reason": p["reason"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- UI pieces
SHADOW = "0 2px 8px rgba(0,0,0,0.08)"
CSS = f"""<style>
html, body, .stApp, .stApp button, .stApp input, .stApp textarea {{font-family: 'Metric', 'Segoe UI', Arial, sans-serif;}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {{display: none !important;}}
header[data-testid="stHeader"] {{background: transparent; height: 64px;}}
.hpe-header {{position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 999989; background: {CARD_BG};
  border-bottom: 3px solid {BRAND}; display: flex; align-items: baseline; gap: 14px; padding: 0 28px;
  box-sizing: border-box; line-height: 61px;}}
.hpe-header .title {{font-weight: 700; font-size: 1.3rem; color: {TEXT};}}
.hpe-header .sub {{font-size: .9rem; color: {MUTED};}}
.block-container {{padding-top: 88px !important; max-width: 1400px;}}
section[data-testid="stSidebar"] {{background: {CARD_BG}; border-right: 1px solid {GRID};
  width: 220px !important; min-width: 220px !important; max-width: 220px !important;}}
[data-testid="stSidebarContent"] {{padding-top: 56px;}}
[data-testid="stSidebarUserContent"] {{padding: 0 14px 16px;}}
.side-h {{color: {MUTED}; text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 600;
  margin: 14px 0 8px;}}
.side-note {{color: {MUTED}; font-size: .75rem; line-height: 1.35;}}
[class*="st-key-card"] {{background: {CARD_BG}; border-radius: 8px; padding: 20px; box-shadow: {SHADOW}; border: none;}}
[data-testid="stMetric"] {{background: {CARD_BG}; border-radius: 8px; padding: 14px 16px; box-shadow: {SHADOW};}}
[data-testid="stMetricLabel"] p {{color: {MUTED}; font-size: .8rem;}}
[data-testid="stMetricValue"] {{color: {TEXT}; font-weight: 700; font-size: 1.55rem;}}
[data-testid="stMetricDelta"] {{font-size: .8rem;}}
[data-testid="stMetricDelta"] svg {{display: none;}}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{padding: 8px 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);}}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{font-size: 1.05rem;}}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{font-size: .7rem;}}
.st-key-kpi_p1 [data-testid="stMetric"] {{border-top: 4px solid {AT_RISK};}}
.st-key-kpi_p1 [data-testid="stMetricValue"] {{color: {AT_RISK};}}
[data-testid^="stBaseButton-primary"] {{background: {BRAND}; color: #FFFFFF; border: none; border-radius: 6px;}}
[data-testid^="stBaseButton-primary"]:hover {{background: #005DA8; color: #FFFFFF;}}
.stTabs [data-baseweb="tab-highlight"] {{background-color: {BRAND};}}
.stTabs [aria-selected="true"] p {{color: {BRAND}; font-weight: 600;}}
[data-testid="stProgress"] [role="progressbar"] > div > div > div {{background-color: {BRAND};}}
.hpe-badge {{border-radius: 8px; padding: 18px 22px; margin: 4px 0 16px;}}
.hpe-badge .label {{font-size: 2rem; font-weight: 800; letter-spacing: .04em; line-height: 1.1;}}
.hpe-badge .reason {{font-size: .9rem; margin-top: 6px;}}
.hpe-callout {{background: {CARD_BG}; border-left: 4px solid; border-radius: 8px; padding: 12px 16px;
  margin: 10px 0; box-shadow: {SHADOW}; color: {TEXT}; font-size: .92rem;}}
</style>"""
HEADER = ('<div class="hpe-header"><span class="title">HPE Ticket Triage</span>'
          '<span class="sub">Problem 03 - Intelligent Support Ticket Triage</span></div>')


def badge(abstain: bool, reason: str):
    label = "SEND TO HUMAN" if abstain else "ROUTE"
    bg, fg = DECISION_STYLE[label]
    st.markdown(f'<div class="hpe-badge" style="background:{bg};color:{fg}"><div class="label">{label}</div>'
                f'<div class="reason">{html.escape(reason)}</div></div>', unsafe_allow_html=True)


def callout(color: str, title: str, text: str):
    st.markdown(f'<div class="hpe-callout" style="border-left-color:{color}"><b>{html.escape(title)}</b> '
                f'&nbsp;{html.escape(text)}</div>', unsafe_allow_html=True)


def side_heading(text: str):
    st.sidebar.markdown(f'<div class="side-h">{html.escape(text)}</div>', unsafe_allow_html=True)


def sidebar():
    side_heading("Model")
    try:
        info = model_info()
    except ApiError as e:
        st.sidebar.error(str(e))
        if st.sidebar.button("Retry"):
            st.rerun()
        return None
    m, man = info.get("headline_val_metrics") or {}, info["manifest"]
    st.sidebar.markdown(f'<div class="side-note">Bundle <b>{man.get("run")}</b> · built '
                        f'{str(man.get("created", ""))[:10]} · VAL n={m.get("n", "?")}</div>', unsafe_allow_html=True)
    f1 = m.get("macro_f1", {})
    p1 = m.get("priority_with_P1_threshold") or {}
    side_heading("Validation Metrics")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Team F1", f"{f1.get('assigned_team', float('nan')):.3f}")
    c2.metric("Category F1", f"{f1.get('category', float('nan')):.3f}")
    c1.metric("Priority F1", f"{p1.get('priority_macro_f1', f1.get('priority', float('nan'))):.3f}")
    c2.metric("Sentiment F1", f"{f1.get('sentiment', float('nan')):.3f}")
    if p1:
        c1.metric("P1 Recall", f"{p1['P1_recall']:.3f}")
        c2.metric("P1 Precision", f"{p1['P1_precision']:.3f}")
    esc = (m.get("escalation_argmax") or {}).get("f1")
    if esc is not None:
        c1.metric("Unhappy F1", f"{esc:.3f}")
    pf = m.get("policy_f") or {}
    if pf:
        side_heading("Routing Cost (VAL)")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Mean Cost", f"{pf['mean_cost']:.2f}",
                  f"{pf['mean_cost'] - m['policy_no_abstain_cost']:+.2f} vs none", delta_color="inverse")
        c2.metric("Hand-Off Rate", f"{pf['abstain_rate']:.1%}", f"cap {man.get('max_abstain_rate', 0.15):.1%}",
                  delta_color="off")
    if p1:
        st.sidebar.markdown(f'<div class="side-note">Priority uses P1 threshold {man.get("p1_threshold")}: '
                            f'{html.escape(str(man.get("priority_rule")))}</div>', unsafe_allow_html=True)
    return info


def single_tab(info):
    if info is None:
        st.info("The ticket form needs the API (product list comes from the model bundle).")
        return
    products = info["manifest"].get("products") or info["manifest"].get("known_products") or []
    with st.container(key="card_ticket_form"):
        with st.form("ticket", border=False):
            subject = st.text_input("Subject", placeholder="e.g. Array offline after firmware update")
            body = st.text_area("Body", height=160, placeholder="Describe the issue…")
            c1, c2 = st.columns(2)
            product = c1.selectbox("Product", products + [OTHER_PRODUCT])
            channel = c2.selectbox("Channel", CHANNELS)
            other = st.text_input("Product Name (Only If Other)")
            submitted = st.form_submit_button("Triage Ticket", type="primary")
    if submitted:
        if product == OTHER_PRODUCT:
            product = other.strip()
            if not product:
                st.error("Enter a product name, or choose one from the list.")
                return
        try:
            with st.spinner("Scoring…"):
                st.session_state["single"] = predict_one(subject, body, product, channel)
        except ApiError as e:
            st.error(str(e))
            return
    res = st.session_state.get("single")
    if res:
        show_single(res)


def show_single(r: dict):
    probs = r["probabilities"]
    with st.container(key="card_single_result"):
        badge(bool(r["abstain"]), r["reason"])
        c = st.columns(4)
        c[0].metric("Assigned Team", r["assigned_team"], f"confidence {r['p_correct']:.1%}", delta_color="off")
        for col, t in zip(c[1:], ("category", "priority", "sentiment")):
            tile = col.container(key="kpi_p1" if t == "priority" and r[t] == "P1" else f"kpi_{t}")
            tile.metric(t.title(), r[t], f"confidence {probs[t][r[t]]:.1%}", delta_color="off")
        if r["escalation"]:
            callout(DECLINING, "Unhappy Customer", "Escalation risk: angry, or frustrated with a P1/P2 issue.")
        for g in r["guard_flags"]:
            callout(AT_RISK, f"Guard Flag: {g}", "This ticket always goes to a human.")
        with st.expander("Probabilities and Costs"):
            cols = st.columns(4)
            for col, (t, p) in zip(cols, probs.items()):
                col.caption(t.replace("_", " ").title())
                col.dataframe(pd.Series(p, name="Probability").sort_values(ascending=False).map("{:.1%}".format))
            st.caption(f"Expected misroute cost {r['expected_route_cost']:.2f} · hand-off cost "
                       f"{r['expected_abstain_cost']:.2f} · savings from hand-off {r['savings']:+.2f}")


def styled_batch(res: dict):
    df = batch_table(res)
    df["team_confidence"] *= 100
    df.columns = [c.replace("_", " ").title().replace("Id", "ID") for c in df.columns]
    css = {k: f"background-color: {b}; color: {f}; font-weight: 600" for k, (b, f) in DECISION_STYLE.items()}
    return df.style.map(lambda v: css.get(v, ""), subset=["Decision"])


def batch_tab():
    with st.container(key="card_batch_upload"):
        st.caption(f"CSV columns: {', '.join(BATCH_COLUMNS)}. At most 15.0% of a batch is sent to a human; "
                   "guard-flagged tickets always are.")
        up = st.file_uploader("Upload Tickets CSV", type=["csv"])
        run = up is not None and st.button("Run Batch", type="primary")
    if run:
        try:
            with st.spinner(f"Scoring {up.name}…"):
                res, csv = predict_batch(up.getvalue(), up.name)
            st.session_state["batch"] = (up.name, res, csv)
        except ApiError as e:
            st.session_state.pop("batch", None)
            st.error(str(e))
    if "batch" not in st.session_state:
        return
    name, res, csv = st.session_state["batch"]
    c = st.columns(4)
    c[0].metric("Tickets", res["n_tickets"])
    c[1].metric("Sent to Human", res["n_abstain"], f"max {res['max_abstains']}", delta_color="off")
    c[2].metric("Hand-Off Rate", f"{res['abstain_rate']:.1%}", f"cap {res['cap']:.1%}", delta_color="off")
    c[3].metric("Guard-Flagged", res["n_guarded"])
    with st.container(key="card_batch_results"):
        st.progress(min(res["abstain_rate"] / res["cap"], 1.0) if res["cap"] else 0.0,
                    text=f"{res['abstain_rate']:.1%} of {res['cap']:.1%} hand-off capacity used")
        for n in res["notes"]:
            callout(DECLINING, "Note", n)
        st.dataframe(styled_batch(res), hide_index=True,
                     column_config={"Team Confidence": st.column_config.NumberColumn(format="%.1f%%")})
        st.download_button("Download Submission CSV", csv, file_name=f"submission_{res['batch_id']}.csv",
                           mime="text/csv", type="primary")
        st.caption(f"From {name}; format: ticket_id, category, priority, assigned_team, sentiment, abstain.")


def main():
    st.set_page_config(page_title="HPE Ticket Triage", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS + HEADER, unsafe_allow_html=True)
    info = sidebar()
    single, batch = st.tabs(["Single Ticket", "Batch CSV"])
    with single:
        single_tab(info)
    with batch:
        batch_tab()


if __name__ == "__main__":
    main()
