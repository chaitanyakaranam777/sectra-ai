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
  min-width:300px !important;
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
.block-container { max-width:1500px; padding-top:1.35rem; padding-bottom:3rem; }
h1,h2,h3,h4 { color:var(--ink); font-family:'Space Grotesk',sans-serif; }
.eyebrow {
  color:#6684cf; font-size:10px; font-weight:800;
  letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;
}
.hero {
  background:linear-gradient(115deg,#142443,#263f6d);
  color:white; padding:30px 34px; border-radius:20px;
  margin:8px 0 24px; box-shadow:0 14px 35px rgba(22,42,78,.13);
}
.hero h1 { color:white!important; font-size:30px; margin:0 0 8px; }
.hero p { color:#d6e2fa; margin:0; font-size:14px; }
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
  .hero { padding:23px; }
  .hero h1 { font-size:24px; }
}
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
]
page_icons = {
    "Overview": "◆",
    "Architectures": "⬡",
    "Datasets": "▤",
    "Prediction Lab": "✧",
    "Training Monitor": "⌁",
    "Diagnostics": "⌘",
}

with st.sidebar:
    st.markdown(
        """<div style="padding:12px 4px 25px;font-family:'Space Grotesk';
        font-size:23px;font-weight:800;letter-spacing:.5px">
        <span style="color:#ffffff">⬡ SENTRA</span>
        <span style="color:#45dec6"> AI</span></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#8fa6cc;font-size:10px;font-weight:800;letter-spacing:2px;padding:0 4px 10px'>WORKSPACE</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        pages,
        format_func=lambda item: f"{page_icons[item]}   {item}",
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:72px'></div><hr>", unsafe_allow_html=True)
    st.markdown("**SENTRA ENGINE**")
    st.caption("Local AI environment")
    if health:
        st.markdown("<span style='color:#45dec6;font-weight:700'>● API Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#ff647c;font-weight:700'>● API Offline</span>", unsafe_allow_html=True)
    st.caption("Multimodal Intelligence Platform")
    st.markdown("<div style='height:58px'></div><hr>", unsafe_allow_html=True)
    st.caption("SENTRA AI  |  DEMO DASHBOARD")


# -------------------- PAGE HEADER --------------------
st.markdown("<div class='eyebrow'>MULTIMODAL INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
st.markdown(
    """<div class="hero">
    <h1>SENTRA AI Intelligence Dashboard</h1>
    <p>Your neural systems, experiments and inference environment at a glance.</p>
    </div>""",
    unsafe_allow_html=True,
)


# -------------------- OVERVIEW --------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    metric(c1, "SYSTEM STATUS", "ONLINE" if health else "OFFLINE", "API connection")
    metric(c2, "ARCHITECTURES", "5", "Registered neural models")
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
        st.markdown("**REGISTERED DATASET**")
        st.markdown(f"<div class='metric-value'>{len(df):,} samples</div>", unsafe_allow_html=True)
        st.caption(f"Source: {sample_path.relative_to(ROOT) if sample_path else 'samples.csv not found'}")
        st.caption(f"{len(df.columns)} available features" if not df.empty else "Dataset unavailable")
        if not df.empty:
            st.dataframe(df.head(5), width="stretch", hide_index=True)
        else:
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
                height=260, margin=dict(l=5, r=10, t=10, b=5), showlegend=False,
                paper_bgcolor="white", plot_bgcolor="white", font_color="#263858",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No recognized sentiment label column found.")
        st.markdown("</div>", unsafe_allow_html=True)

    section("Dataset Analytics", "03 / VISUAL INSIGHTS")
    if not df.empty:
        a, b = st.columns(2, gap="medium")
        with a:
            fig = go.Figure(go.Bar(
                x=["Rows", "Columns"], y=[len(df), len(df.columns)],
                marker_color=["#80c5f5", "#6b7ff0"],
                text=[len(df), len(df.columns)], textposition="outside",
            ))
            fig.update_layout(
                title="Dataset dimensions", height=320,
                margin=dict(l=15, r=15, t=45, b=25),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch")
        with b:
            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                selected_column = st.selectbox("Explore numeric feature", numeric.columns, key="overview_numeric")
                fig = px.histogram(
                    df, x=selected_column, nbins=25,
                    color_discrete_sequence=["#5e78f4"],
                    title=f"Distribution: {selected_column}",
                )
                fig.update_layout(
                    height=320, margin=dict(l=15, r=15, t=45, b=25),
                    paper_bgcolor="white", plot_bgcolor="white",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numeric features available for distribution plots.")

    section("Model Performance", "04 / BENCHMARK")
    if comp_rows:
        comparison_df = pd.json_normalize(comp_rows)
        for column in comparison_df.columns:
            if comparison_df[column].dtype == "object":
                comparison_df[column] = comparison_df[column].fillna("").astype(str)
        st.dataframe(comparison_df, width="stretch", hide_index=True)
    else:
        st.info("No comparison results found at artifacts/comparison.json.")


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
