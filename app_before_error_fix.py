# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
API_BASE = os.getenv("SENTRA_API_URL", "http://127.0.0.1:8002").rstrip("/")

st.set_page_config(
    page_title="SENTRA AI | Intelligence Platform",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DESIGN SYSTEM --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --navy:#142443; --navy2:#1b3156; --ink:#172544;
  --muted:#7183a5; --line:#e1e8f4; --bg:#f3f6fc;
  --blue:#5e78f4; --teal:#37d6bd;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(243,246,252,.96); }
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#12213d 0%,#1b3156 100%);
  border-right:1px solid #263c60;
  min-width:280px !important;
}
[data-testid="stSidebar"] > div:first-child { padding:28px 20px 24px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:0.55rem; }
[data-testid="stSidebar"] * { color:#eef4ff; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d7e1f3; }
[data-testid="stSidebar"] .stRadio > div { gap:13px; }
[data-testid="stSidebar"] .stRadio label {
  border:1px solid transparent; border-radius:12px;
  padding:9px 10px; min-height:34px; font-size:14px;
  font-weight:600; transition:all .15s ease;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background:#20385d; border-color:#35537e;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
  background:#20385d; border-color:#35537e;
}
[data-testid="stSidebar"] input[type="radio"] { accent-color:#e84d65; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
  color:#f4f7ff; font-size:14px; letter-spacing:.2px;
}
.block-container { max-width:1500px; padding-top:1.1rem; padding-bottom:2.5rem; }
h1,h2,h3,h4 { color:var(--ink); font-family:'Space Grotesk',sans-serif; }
.eyebrow {
  color:#6684cf; font-size:10px; font-weight:800;
  letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;
}
.topbar {
  display:flex; justify-content:space-between; align-items:center;
  padding:0 0 18px; margin:0 0 28px; border-bottom:1px solid var(--line);
  color:#8291ad; font-size:12px;
}
.page-title { font-family:'Space Grotesk',sans-serif; color:#172544;
  font-size:29px; font-weight:800; letter-spacing:-.8px; margin:0 0 5px; }
.page-subtitle { color:#7b8eae; font-size:13px; margin:0 0 24px; }
.date-chip { border:1px solid var(--line); background:white; color:#7183a5;
  border-radius:10px; padding:10px 13px; font-size:11px; white-space:nowrap; }
.metric-card {
  background:white; border:1px solid var(--line); border-radius:17px;
  padding:19px 20px; min-height:126px;
  box-shadow:0 8px 22px rgba(31,55,96,.045);
  position:relative; overflow:hidden;
}
.metric-card:after {
  content:''; position:absolute; right:-9px; top:-18px;
  width:72px; height:72px; background:#e7efff;
  border-radius:0 0 0 30px; opacity:.9;
}
.metric-label {
  color:#8191af; font-size:10px; font-weight:800;
  letter-spacing:1.2px; text-transform:uppercase;
}
.metric-value {
  color:#172544; font-family:'Space Grotesk',sans-serif;
  font-size:29px; font-weight:800; margin-top:10px;
}
.metric-note { color:#7b8eae; font-size:11px; margin-top:4px; }
.section-head {
  display:flex; justify-content:space-between; align-items:center;
  margin:25px 0 13px;
}
.section-head h3 { font-size:17px; margin:0; }
.section-head span {
  color:#8293b1; font-size:10px; letter-spacing:1.5px; font-weight:700;
}
.model-card {
  background:white; border:1px solid var(--line); border-radius:17px;
  padding:19px; min-height:205px; margin-bottom:2px;
  box-shadow:0 7px 20px rgba(31,55,96,.04);
}
.model-icon {
  display:flex; width:43px; height:43px; align-items:center;
  justify-content:center; background:#eaf0ff; color:#5277ee;
  border-radius:13px; font-weight:800; font-size:13px; margin-bottom:13px;
}
.model-name { color:#172544; font-weight:750; font-size:16px; margin-bottom:7px; }
.model-type { color:#6684cf; font-size:11px; margin-bottom:12px; }
.model-desc { color:#7b8eae; font-size:12px; min-height:34px; }
.model-footer {
  border-top:1px solid var(--line); margin-top:15px;
  padding-top:12px; color:#7b8eae; font-size:11px;
}
.panel {
  background:white; border:1px solid var(--line); border-radius:17px;
  padding:20px; box-shadow:0 8px 22px rgba(31,55,96,.04);
}
.status-pill {
  display:inline-block; padding:6px 10px; border-radius:20px;
  background:#e8f8f3; color:#159b79; font-size:11px; font-weight:800;
}
div[data-testid="stDataFrame"] {
  border:1px solid var(--line); border-radius:12px; overflow:hidden;
}
.stButton button, .stDownloadButton button {
  border-radius:10px; font-weight:700; min-height:40px;
}
div.stButton > button[kind="primary"] {
  background:linear-gradient(90deg,#526cf0,#8062e9); border:0;
}
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
  border-radius:11px; border-color:#dce5f3;
}
hr { border-color:var(--line); }
footer { visibility:hidden; }
@media (max-width: 900px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .page-title { font-size:24px; }
}

/* SENTRA_NAV_FINAL */
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    min-height: 52px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {
    font-size: 18px !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #20385d !important;
    border-left: 4px solid #45dec6 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #263f66 !important;
}
/* END_SENTRA_NAV_FINAL */

/* SENTRA_CUSTOM_NAV */
[data-testid="stSidebar"] .stRadio {
    display: none !important;
}
[data-testid="stSidebar"] div.stButton {
    margin: 0 0 7px 0 !important;
}
[data-testid="stSidebar"] div.stButton > button {
    min-height: 43px !important;
    height: 43px !important;
    border-radius: 11px !important;
    padding: 0 13px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: background .15s ease, border .15s ease !important;
}
[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #e7efff !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: #20385d !important;
    border-color: #35537e !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: #20385d !important;
    color: #ffffff !important;
    border: 1px solid #35537e !important;
    border-left: 4px solid #45dec6 !important;
}
[data-testid="stSidebar"] div.stButton > button p {
    font-size: 14px !important;
    font-weight: 500 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] div.stButton > button:focus {
    box-shadow: none !important;
}
/* END SENTRA_CUSTOM_NAV */

/* SENTRA_BLACK_SIDEBAR_V2 */

/* Full-height black sidebar */
[data-testid="stSidebar"] {
    background: #08090D !important;
    min-width: 290px !important;
    width: 290px !important;
    border-right: 1px solid #252631 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #08090D !important;
    height: 100vh !important;
}

/* Sidebar content */
[data-testid="stSidebar"] section {
    background: #08090D !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem 2rem 1rem !important;
}

/* Brand and headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] p {
    color: #D0D0DA;
}

/* Workspace label */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    letter-spacing: .2px;
}

/* Navigation spacing */
[data-testid="stSidebar"] div.stButton {
    margin: 0 0 9px 0 !important;
}

/* Navigation buttons */
[data-testid="stSidebar"] div.stButton > button {
    width: 100% !important;
    min-height: 46px !important;
    height: 46px !important;
    padding: 0 15px !important;
    border-radius: 9px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    color: #D9D9E2 !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all .18s ease !important;
}

/* Navigation text */
[data-testid="stSidebar"] div.stButton > button p {
    color: inherit !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

/* Active navigation item */
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: #241779 !important;
    color: #FFFFFF !important;
    border: 1px solid #40339B !important;
    box-shadow: 0 4px 16px rgba(36,23,121,.25) !important;
}

/* Hover effect */
[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: #191923 !important;
    color: #FFFFFF !important;
    border: 1px solid #333344 !important;
}

/* Engine information */
[data-testid="stSidebar"] hr {
    border-color: #30303B !important;
    margin-top: 25px !important;
    margin-bottom: 25px !important;
}

/* Remove unnecessary radio navigation */
[data-testid="stSidebar"] .stRadio {
    display: none !important;
}

/* Clean sidebar scrollbar */
[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 5px;
}

[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #343442;
    border-radius: 10px;
}

/* END_SENTRA_BLACK_SIDEBAR_V2 */


/* SENTRA_TWO_PANEL_NAV */
[data-testid="stSidebar"] {
    background: #08090D !important;
    min-width: 355px !important;
    width: 355px !important;
    border-right: 1px solid #272730 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #08090D !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem 2rem 1rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: #30303A !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_rail_"] button {
    height: 44px !important;
    min-height: 44px !important;
    width: 44px !important;
    padding: 0 !important;
    justify-content: center !important;
    background: transparent !important;
    color: #D7D7E0 !important;
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_rail_"] button[kind="primary"] {
    background: #281C78 !important;
    color: #FFFFFF !important;
    border-color: #493BA7 !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_rail_"] button:hover {
    background: #211B3D !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button {
    height: 44px !important;
    min-height: 44px !important;
    padding: 0 13px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: transparent !important;
    color: #D7D7E0 !important;
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    box-shadow: none !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button[kind="primary"] {
    background: #281C78 !important;
    color: #FFFFFF !important;
    border-color: #493BA7 !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button:hover {
    background: #211B3D !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button p {
    font-size: 14px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D0D0DA;
}
/* END SENTRA_TWO_PANEL_NAV */


/* SENTRA_SIDEBAR_POLISH */
[data-testid="stSidebar"] {
    background: #08090D !important;
    min-width: 355px !important;
    width: 355px !important;
    border-right: 1px solid #282832 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #08090D !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem 2rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: #30303A !important;
    margin-top: 30px !important;
    margin-bottom: 22px !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 9px !important;
}
[data-testid="stSidebar"] [data-testid="column"] {
    min-width: 0 !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_rail_"] button {
    width: 44px !important;
    height: 44px !important;
    min-height: 44px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 10px !important;
    background: transparent !important;
    color: #D7D7E0 !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_rail_"] button[kind="primary"] {
    background: #281C78 !important;
    border-color: #5544B8 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_rail_"] button svg {
    width: 22px !important;
    height: 22px !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button {
    width: 100% !important;
    height: 44px !important;
    min-height: 44px !important;
    padding: 0 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: 10px !important;
    background: transparent !important;
    color: #D7D7E0 !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button[kind="primary"] {
    background: #281C78 !important;
    border-color: #5544B8 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button > div {
    width: 100% !important;
    justify-content: flex-start !important;
}
[data-testid="stSidebar"] div[class*="st-key-sentra_label_"] button p {
    width: 100% !important;
    text-align: left !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] button:hover {
    background: #1B1B27 !important;
    border-color: #3A365E !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D0D0DA;
}
@media (max-width: 700px) {
    [data-testid="stSidebar"] {
        min-width: 290px !important;
        width: 290px !important;
    }
}
/* END_SENTRA_SIDEBAR_POLISH */

</style>
""", unsafe_allow_html=True)


# -------------------- DATA HELPERS --------------------
def find_samples():
    candidates = [
        DATA_DIR / "processed" / "mvsa_single" / "samples.csv",
        DATA_DIR / "processed" / "samples.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = list(DATA_DIR.glob("**/samples.csv")) if DATA_DIR.exists() else []
    return found[0] if found else None


@st.cache_data(show_spinner=False)
def load_dataset(path_string, modified):
    if not path_string:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path_string)
        # Keep mixed columns consistent so Streamlit/Arrow can render them.
        for column in frame.columns:
            if frame[column].dtype == "object":
                values = frame[column]
                numeric = pd.to_numeric(values, errors="coerce")
                non_null = values.notna().sum()
                if non_null and numeric.notna().sum() >= max(1, int(non_null * 0.85)):
                    frame[column] = numeric
                else:
                    frame[column] = values.fillna("").astype(str)
        return frame
    except Exception as exc:
        st.warning(f"Could not read dataset: {exc}")
        return pd.DataFrame()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_runs():
    runs = []
    base = ARTIFACTS / "runs"
    if base.exists():
        folders = sorted(
            base.iterdir(),
            key=lambda item: item.stat().st_mtime if item.is_dir() else 0,
            reverse=True,
        )
        for folder in folders:
            if not folder.is_dir():
                continue
            config = read_json(folder / "config.json")
            status = read_json(folder / "status.json")
            runs.append({
                "Run": str(folder.name),
                "Model": str(config.get("model", folder.name.split("_")[0])),
                "Dataset": str(config.get("dataset", "—")),
                "Status": str(status.get("status", "unknown")),
                "Epochs": str(config.get("epochs", status.get("total_epochs", "—"))),
                "Device": str(config.get("device", status.get("device", "—"))),
                "Updated": datetime.fromtimestamp(folder.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return runs


def api_get(path, timeout=2):
    try:
        response = requests.get(API_BASE + path, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def safe_label_counts(frame):
    for column in ["label", "sentiment", "sentiment_label", "class"]:
        if column in frame.columns:
            return (
                frame[column].astype(str)
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )
    return pd.DataFrame()


def section(title, tag):
    st.markdown(
        f"<div class='section-head'><h3>{title}</h3><span>{tag}</span></div>",
        unsafe_allow_html=True,
    )


def metric(column, label, value, note):
    column.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# -------------------- LOAD PROJECT DATA --------------------
sample_path = find_samples()
df = load_dataset(
    str(sample_path) if sample_path else "",
    sample_path.stat().st_mtime if sample_path else 0,
)
runs = get_runs()
health, health_error = api_get("/api/health")
models, models_error = api_get("/api/models")
comparison = read_json(ARTIFACTS / "comparison.json")
comp_rows = comparison.get("models", comparison.get("results", [])) if isinstance(comparison, dict) else comparison
if not isinstance(comp_rows, list):
    comp_rows = []

architecture = [
    ("CNN", "C", "IMAGE", "CNN image-only classifier", sum("cnn" in run["Run"].lower() for run in runs)),
    ("LSTM", "L", "TEXT", "BiLSTM text-only classifier", sum("lstm" in run["Run"].lower() for run in runs)),
    ("BERT", "B", "TEXT", "BERT text-only classifier", sum("bert" in run["Run"].lower() and "cnn_bert" not in run["Run"].lower() for run in runs)),
    ("CNN_LSTM", "CL", "MULTIMODAL", "CNN + BiLSTM multimodal fusion", sum("cnn_lstm" in run["Run"].lower() for run in runs)),
    ("CNN_BERT", "CB", "MULTIMODAL", "CNN + BERT multimodal fusion", sum("cnn_bert" in run["Run"].lower() for run in runs)),
]


# -------------------- SIDEBAR --------------------
pages = [
    "Overview",
    "Architectures",
    "Datasets",
    "Prediction Lab",
    "Training Monitor",
    "Diagnostics",
    "Model Comparison",
    "Explainability",
    "Experiment History",
    "Data Explorer",
    "Settings",
]
page_icons = {
    "Overview": "dashboard",
    "Architectures": "account_tree",
    "Datasets": "database",
    "Prediction Lab": "psychology",
    "Training Monitor": "monitoring",
    "Diagnostics": "settings",
    "Model Comparison": "compare_arrows",
    "Explainability": "lightbulb",
    "Experiment History": "history",
    "Data Explorer": "table_chart",
    "Settings": "tune",
}

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:13px;
                    padding:20px 5px 34px;white-space:nowrap;">
          <svg width="42" height="46" viewBox="0 0 64 72"
               xmlns="http://www.w3.org/2000/svg"
               style="flex-shrink:0;filter:drop-shadow(0 0 7px #17d9c455)">
            <defs>
              <linearGradient id="sentraBrand" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#45F5D0"/>
                <stop offset="100%" stop-color="#477BFF"/>
              </linearGradient>
            </defs>
            <path d="M32 4 L57 19 L57 49 L32 66 L7 49 L7 19 Z"
                  fill="#111827" stroke="url(#sentraBrand)" stroke-width="3.5"/>
            <path d="M32 14 L48 24 L48 44 L32 55 L16 44 L16 24 Z"
                  fill="none" stroke="#45DEC6" stroke-width="2"/>
            <path d="M23 27 H41 M23 35 H37 M23 43 H41"
                  stroke="#45DEC6" stroke-width="3.5" stroke-linecap="round"/>
          </svg>
          <div style="font-family:'Space Grotesk',sans-serif;
                      font-size:22px;font-weight:800;letter-spacing:.5px;">
            <span style="color:#FFFFFF">SENTRA</span>
            <span style="color:#45DEC6"> AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='color:#858596;font-size:10px;font-weight:700;"
        "letter-spacing:2px;padding:0 5px 15px'>WORKSPACE</div>",
        unsafe_allow_html=True,
    )

    if "sentra_page" not in st.session_state:
        st.session_state["sentra_page"] = "Overview"

    # Two-panel navigation: icon rail + page labels
    rail, navigation = st.columns([0.27, 1.0], gap="small")

    material_icons = {
        "Overview": "dashboard",
        "Architectures": "account_tree",
        "Datasets": "database",
        "Prediction Lab": "psychology",
        "Training Monitor": "monitoring",
        "Diagnostics": "settings",
    }

    for item in pages:
        active = st.session_state["sentra_page"] == item
        with rail:
            if st.button(
                " ",
                icon=f":material/{material_icons[item]}:",
                key=f"sentra_rail_{item}",
                type="primary" if active else "secondary",
                help=item,
                use_container_width=True,
            ):
                st.session_state["sentra_page"] = item
                st.rerun()

        with navigation:
            if st.button(
                item,
                key=f"sentra_label_{item}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state["sentra_page"] = item
                st.rerun()

    page = st.session_state["sentra_page"]

    st.markdown("<div style='height:45px'></div><hr>", unsafe_allow_html=True)
    st.markdown("**SENTRA ENGINE**")
    st.caption("Local AI environment")

    if health:
        st.markdown(
            "<span style='color:#45dec6;font-weight:700'>? API Connected</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='color:#ff647c;font-weight:700'>? API Offline</span>",
            unsafe_allow_html=True,
        )

    st.caption("Multimodal Intelligence Platform")
    st.markdown("<div style='height:58px'></div><hr>", unsafe_allow_html=True)
    st.caption("SENTRA AI  |  DEMO DASHBOARD")


# -------------------- PAGE HEADER --------------------
header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
    st.markdown(
        f"<div class='topbar'><span>Workspace / <b style='color:#172544'>{page}</b></span>"
        f"<span style='color:{'#159b79' if health else '#d65b70'};font-weight:800'>● API {'ONLINE' if health else 'OFFLINE'}</span></div>",
        unsafe_allow_html=True,
    )
st.markdown("<div class='eyebrow'>MULTIMODAL INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
st.markdown(f"<div class='page-title'>{'Intelligence Overview' if page == 'Overview' else page}</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Your neural systems, experiments and inference environment at a glance.</div>", unsafe_allow_html=True)


# -------------------- OVERVIEW --------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    metric(c1, "SYSTEM STATUS", "ONLINE" if health else "OFFLINE", "API connection")
    metric(c2, "ARCHITECTURES", str(len(architecture)), "Registered neural models")
    metric(c3, "TRAINING RUNS", str(len(runs)), "Recorded experiments")
    metric(c4, "COMPUTE DEVICE", str((health or {}).get("device", "—")).upper(), "Inference environment")

    section("Neural Architecture Registry", "01 / MODEL CATALOG")
    cards = st.columns(3, gap="medium")
    for index, (name, icon_text, modality, description, count) in enumerate(architecture):
        with cards[index % 3]:
            st.markdown(
                f"""<div class="model-card">
                <div class="model-icon">{icon_text}</div>
                <div class="model-name">{name}</div>
                <div class="model-type">{modality}</div>
                <div class="model-desc">{description}</div>
                <div class="model-footer">{count} registered runs</div>
                </div>""",
                unsafe_allow_html=True,
            )

    section("Dataset Intelligence", "02 / DATA OVERVIEW")
    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**REGISTERED DATASETS**")
        if not df.empty:
            st.markdown(f"<div class='metric-value'>{len(df):,}</div>", unsafe_allow_html=True)
            st.caption(f"Processed samples · {sample_path.relative_to(ROOT) if sample_path else 'MVSA Single'}")
            st.caption(f"{len(df.columns)} available features")
            st.dataframe(df.head(4), width="stretch", hide_index=True)
        else:
            st.markdown("<div class='metric-value'>—</div>", unsafe_allow_html=True)
            st.info("Processed dataset not found. Expected data/processed/mvsa_single/samples.csv.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**SENTIMENT DISTRIBUTION**")
        counts = safe_label_counts(df)
        if not counts.empty:
            fig = px.bar(
                counts, x="Count", y="Sentiment", orientation="h", text="Count",
                color="Sentiment", color_discrete_sequence=["#5e78f4", "#9875e9", "#31c9ae"],
            )
            fig.update_layout(
                height=245, margin=dict(l=5, r=10, t=8, b=5), showlegend=False,
                paper_bgcolor="white", plot_bgcolor="white", font_color="#263858",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Sentiment labels will appear here when the processed dataset is available.")
        st.markdown("</div>", unsafe_allow_html=True)

    section("AI Prediction Lab", "03 / LIVE INFERENCE")
    prediction_left, prediction_right = st.columns(2, gap="medium")
    with prediction_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**TEXT SENTIMENT ANALYSIS**")
        st.caption("TEXT INPUT")
        overview_text = st.text_area(
            "Enter text to analyze",
            placeholder="Enter a sentence, review or social media post to analyze sentiment...",
            height=110, label_visibility="collapsed", key="overview_prediction_text",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with prediction_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**PREDICTION OUTPUT**")
        st.caption("MODEL RESPONSE")
        st.markdown(
            "<div style='background:#111e3b;color:#9db0d2;border-radius:12px;padding:22px 18px;font-size:12px;min-height:128px'>"
            "SENTIMENT CLASSIFICATION<br><br><b style='color:white;font-size:18px'>Awaiting input</b><br>"
            "Run an analysis from the Prediction Lab page to view model output.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- ARCHITECTURES --------------------
elif page == "Architectures":
    section("Neural Architecture Registry", "MODEL CATALOG")
    arch_df = pd.DataFrame([
        {"Architecture": item[0], "Modality": item[2], "Description": item[3], "Registered runs": item[4]}
        for item in architecture
    ])
    st.dataframe(arch_df, width="stretch", hide_index=True)
    st.info("SENTRA supports CNN, LSTM, BERT and multimodal CNN-LSTM / CNN-BERT pipelines.")
    if models:
        st.markdown("#### API model registry")
        st.json(models)


# -------------------- DATASETS --------------------
elif page == "Datasets":
    section("Dataset Explorer", "DATA MANAGEMENT")
    if df.empty:
        st.error("Processed dataset not found.")
    else:
        st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        search = st.text_input("Filter rows", placeholder="Search across dataset text...")
        view = df
        if search:
            mask = view.astype(str).apply(
                lambda series: series.str.contains(search, case=False, na=False)
            ).any(axis=1)
            view = view[mask]
        st.dataframe(view, width="stretch", height=430, hide_index=True)
        st.download_button(
            "Download filtered CSV",
            view.to_csv(index=False).encode("utf-8"),
            "sentra_dataset.csv",
            "text/csv",
        )
        st.markdown("#### Column profile")
        profile = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data type": [str(value) for value in df.dtypes],
            "Missing values": df.isna().sum().values,
            "Unique values": df.nunique(dropna=False).values,
        })
        st.dataframe(profile, width="stretch", hide_index=True)
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            selected = st.selectbox("Numeric column", numeric.columns)
            fig = px.histogram(df, x=selected, nbins=30, color_discrete_sequence=["#5e78f4"])
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch")


# -------------------- PREDICTION LAB --------------------
elif page == "Prediction Lab":
    section("AI Prediction Lab", "LIVE INFERENCE")
    st.caption("Text sentiment inference through your running SENTRA API.")
    text_input = st.text_area(
        "TEXT SENTIMENT ANALYSIS",
        placeholder="Enter a sentence, review or social media post to analyze sentiment...",
        height=150,
    )
    endpoint = st.text_input(
        "Prediction endpoint",
        value="/api/predict/text",
        help="Change this if your FastAPI route uses a different path.",
    )
    if st.button("RUN SENTIMENT ANALYSIS", type="primary"):
        if not text_input.strip():
            st.warning("Enter text before running prediction.")
        else:
            result, error = None, None
            for payload in [{"text": text_input}, {"text_raw": text_input}, {"input": text_input}]:
                try:
                    response = requests.post(API_BASE + endpoint, json=payload, timeout=45)
                    if response.ok:
                        result = response.json()
                        break
                    error = f"HTTP {response.status_code}: {response.text[:500]}"
                except Exception as exc:
                    error = str(exc)
            if result:
                st.success("Prediction received")
                label = result.get("label", result.get("sentiment", result.get("prediction", "Result")))
                st.markdown(
                    f"""<div class="panel">
                    <div class="metric-label">SENTIMENT CLASSIFICATION</div>
                    <div class="metric-value">{str(label).upper()}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.json(result)
                probabilities = result.get("probabilities")
                if isinstance(probabilities, dict):
                    probability_df = pd.DataFrame({
                        "Sentiment": list(probabilities.keys()),
                        "Probability": list(probabilities.values()),
                    })
                    fig = px.bar(
                        probability_df, x="Sentiment", y="Probability",
                        color="Sentiment", text_auto=".2f",
                        color_discrete_sequence=["#5e78f4", "#9875e9", "#31c9ae"],
                    )
                    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.error("Prediction API is unavailable or the endpoint/payload does not match.")
                st.code(f"{API_BASE}{endpoint}\n{error or 'No response'}")
                st.info("Start your FastAPI backend, then verify the correct route in http://127.0.0.1:8002/docs.")


# -------------------- TRAINING MONITOR --------------------
elif page == "Training Monitor":
    section("Training Monitor", "EXPERIMENT TRACKING")
    if runs:
        run_df = pd.DataFrame(runs)
        # Explicit string conversion prevents ArrowTypeError for mixed Epochs values.
        for column in run_df.columns:
            run_df[column] = run_df[column].fillna("—").astype(str)
        st.dataframe(run_df, width="stretch", hide_index=True)
        status_counts = run_df["Status"].value_counts().rename_axis("Status").reset_index(name="Runs")
        fig = px.bar(
            status_counts, x="Status", y="Runs", color="Status", text="Runs",
            color_discrete_sequence=["#5e78f4", "#31c9ae", "#f29a64"],
        )
        fig.update_layout(height=300, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")
        selected = st.selectbox("Inspect run", run_df["Run"].tolist())
        run_path = ARTIFACTS / "runs" / selected
        st.markdown("#### Run configuration")
        st.json(read_json(run_path / "config.json"))
        st.markdown("#### Run status")
        st.json(read_json(run_path / "status.json"))
    else:
        st.info("No training run folders found in artifacts/runs.")
    if st.button("Refresh training data"):
        st.cache_data.clear()
        st.rerun()


# -------------------- DIAGNOSTICS --------------------
elif page == "Diagnostics":
    section("System Diagnostics", "HEALTH & FILES")
    st.markdown("#### API connection")
    if health:
        st.success("Backend API is responding.")
        st.json(health)
    else:
        st.error(f"Backend offline: {health_error}")
        st.code(f"API base: {API_BASE}\nExpected health route: {API_BASE}/api/health")
    st.markdown("#### Model registry response")
    if models:
        st.json(models)
    else:
        st.warning(f"Could not load /api/models: {models_error}")
    st.markdown("#### Local project assets")
    checks = {
        "Processed samples CSV": sample_path,
        "Artifacts directory": ARTIFACTS if ARTIFACTS.exists() else None,
        "Runs directory": ARTIFACTS / "runs" if (ARTIFACTS / "runs").exists() else None,
        "Comparison report": ARTIFACTS / "comparison.json" if (ARTIFACTS / "comparison.json").exists() else None,
    }
    for label, path in checks.items():
        available = bool(path and Path(path).exists())
        st.write("AVAILABLE" if available else "MISSING", label, str(path) if path else "Not found")
    st.markdown("#### Runtime")
    st.json({
        "Python": os.sys.version.split()[0],
        "Streamlit": st.__version__,
        "API URL": API_BASE,
        "Dataset rows": len(df),
        "Training runs": len(runs),
    })


st.markdown("---")
st.caption("SENTRA AI | Multimodal Intelligence Platform | Local dashboard")


# SENTRA_ADDITIONAL_WORKSPACE_PAGES
if page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Compare registered model experiments and their measured performance.")
    import json
    from pathlib import Path as _Path
    comparison_file = _Path("artifacts/comparison.json")
    if comparison_file.exists():
        try:
            comparison_data = json.loads(comparison_file.read_text(encoding="utf-8"))
            st.json(comparison_data)
        except Exception as exc:
            st.warning(f"Could not read comparison results: {exc}")
    else:
        st.info("No model comparison results found yet. Run model evaluation to populate this section.")

elif page == "Explainability":
    st.title("Model Explainability")
    st.caption("Understand model decisions and the signals behind predictions.")
    st.info("This workspace is ready for feature attribution, confidence analysis and prediction-level explanations.")
    st.markdown("""
    **Explainability tools**
    - Prediction confidence and class probabilities
    - Important input features
    - Text token contribution analysis
    - Image-region contribution analysis
    """)
    st.warning("Detailed explanations require compatible attribution outputs from the trained models.")

elif page == "Experiment History":
    st.title("Experiment History")
    st.caption("Browse saved training runs from your local SENTRA environment.")
    from pathlib import Path as _Path
    runs_dir = _Path("artifacts/runs")
    if runs_dir.exists():
        runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            st.metric("Saved experiments", len(runs))
            st.dataframe(
                [{"Experiment": p.name,
                  "Last modified": __import__("datetime").datetime.fromtimestamp(
                      p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
                 for p in runs],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No saved experiment folders found.")
    else:
        st.info("The artifacts/runs folder has not been created yet.")

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Inspect the processed MVSA Single dataset.")
    from pathlib import Path as _Path
    samples_file = _Path("data/processed/mvsa_single/samples.csv")
    if samples_file.exists():
        st.success(f"Dataset found: {samples_file}")
        try:
            import pandas as _pd
            data = _pd.read_csv(samples_file)
            st.metric("Total samples", f"{len(data):,}")
            st.write("Dataset columns")
            st.write(list(data.columns))
            st.dataframe(data.head(100), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not load dataset: {exc}")
    else:
        st.warning("Processed MVSA Single samples.csv was not found.")

elif page == "Settings":
    st.title("SENTRA Settings")
    st.caption("Local dashboard and environment information.")
    st.markdown("**Dashboard configuration**")
    st.write("Application: SENTRA AI")
    st.write("Interface: Streamlit")
    st.write("Workspace: Local environment")
    st.write("API endpoint: " + str(__import__("os").getenv(
        "SENTRA_API_URL", "http://127.0.0.1:8002")))
    st.info("Model and API configuration is managed by the backend environment.")

# END_SENTRA_ADDITIONAL_WORKSPACE_PAGES
