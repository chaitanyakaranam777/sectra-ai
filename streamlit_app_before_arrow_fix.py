import os
import json
import glob
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"

API = os.getenv("SENTRA_API_URL", "http://127.0.0.1:8001")

st.set_page_config(
    page_title="SENTRA AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #142342;
    --navy2: #1b2c4e;
    --teal: #38d9c2;
    --bg: #f2f5fc;
    --muted: #8190ad;
    --border: #e1e8f3;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
}

header[data-testid="stHeader"] {
    background: rgba(242,245,252,.96);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#14213d,#192d50);
    border-right: 1px solid #263b5e;
}

[data-testid="stSidebar"] * {
    color: #eef4ff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 12px;
    padding: 9px 12px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #263e65;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 13px;
    font-weight: 600;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.brand {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
    color: white;
    padding: 5px 0 25px 0;
}

.brand span {
    color: var(--teal);
}

.workspace {
    color: #8190ad;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
    margin-top: 15px;
}

.hero-label {
    color: #647fce;
    letter-spacing: 2px;
    font-size: 10px;
    font-weight: 800;
    margin-bottom: 10px;
}

.main-title {
    font-size: 29px;
    font-weight: 800;
    color: #142342;
    margin-bottom: 3px;
}

.subtitle {
    font-size: 13px;
    color: #8190ad;
}

.section-heading {
    color: #142342;
    font-size: 16px;
    font-weight: 800;
    margin: 24px 0 13px 0;
}

.section-number {
    color: #8292b0;
    font-size: 10px;
    letter-spacing: 1.4px;
    text-align: right;
}

.metric {
    background: white;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 22px 19px;
    min-height: 137px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(31,52,88,.045);
}

.metric:after {
    content: "";
    position: absolute;
    right: -15px;
    top: -27px;
    width: 78px;
    height: 78px;
    border-radius: 22px;
    opacity: .18;
    background: #38d9c2;
}

.metric.blue:after {background:#7895ff;}
.metric.purple:after {background:#b78bff;}
.metric.orange:after {background:#ffbf70;}

.metric-label {
    color: #8292b0;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 800;
}

.metric-value {
    color: #172440;
    font-size: 27px;
    font-weight: 800;
    margin-top: 20px;
}

.metric-note {
    color: #8292b0;
    font-size: 10px;
    margin-top: 5px;
}

.panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 21px;
    box-shadow: 0 10px 28px rgba(31,52,88,.045);
}

.architecture {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px;
    min-height: 170px;
    box-shadow: 0 8px 24px rgba(31,52,88,.04);
}

.architecture h3 {
    font-size: 15px;
    color: #172440;
    margin-top: 12px;
}

.architecture p {
    color: #8090ad;
    font-size: 11px;
}

.architecture-icon {
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: #e8efff;
    color: #5878e8;
    font-weight: 800;
    font-size: 13px;
}

.small-label {
    color: #8493ae;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.output-box {
    background: #14213d;
    color: #e5edff;
    border-radius: 13px;
    padding: 19px;
    min-height: 200px;
}

.output-box h2 {
    color: #43ddc5;
}

.stButton button {
    background: linear-gradient(110deg,#536ff0,#8064e8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 19px;
    font-weight: 700;
}

.stButton button:hover {
    background: #405cd9;
    color: white;
    border: none;
}

div[data-testid="stTextArea"] textarea {
    background: #f9fbff;
    border: 1px solid #dce5f3;
    border-radius: 12px;
    color: #172440;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

hr {
    border-color: #dfe6f2;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data(ttl=30)
def get_csv_files():
    return sorted(set(
        glob.glob(str(DATA / "**" / "*.csv"), recursive=True)
    ))

@st.cache_data(ttl=30)
def read_csv(path):
    return pd.read_csv(path, low_memory=False)

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_runs():
    result = []
    if RUNS.exists():
        for folder in sorted(RUNS.iterdir()):
            if not folder.is_dir():
                continue

            config = read_json(folder / "config.json")
            status = read_json(folder / "status.json")

            result.append({
                "Run": folder.name,
                "Model": config.get("model", folder.name.split("_")[0]),
                "Dataset": config.get("dataset", "MVSA Single"),
                "Status": status.get("status", "Unknown"),
                "Epochs": str(status.get(
                    "total_epochs",
                    status.get("epochs", "—")
                )),
                "Device": str(status.get(
                    "device",
                    config.get("device", "—")
                ))
            })
    return result

def get_api_health():
    try:
        r = requests.get(f"{API}/api/health", timeout=3)
        if r.ok:
            return True, r.json()
        return False, {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def safe_json_records(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["models", "results", "comparisons", "runs", "data"]:
            if isinstance(obj.get(key), list):
                return obj[key]
        return [obj]
    return []

def find_label(df):
    candidates = [
        "sentiment", "label", "sentiment_label",
        "polarity", "target", "class"
    ]
    cols = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None

def card(label, value, note, color=""):
    st.markdown(f"""
    <div class="metric {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def heading(title, number):
    a, b = st.columns([4, 1])
    with a:
        st.markdown(f'<div class="section-heading">{title}</div>',
                    unsafe_allow_html=True)
    with b:
        st.markdown(
            f'<div class="section-number">{number}</div>',
            unsafe_allow_html=True
        )

def plot_style(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter", color="#34415c", size=11),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", y=-0.18),
    )
    return fig

# ============================================================
# LOAD PROJECT DATA
# ============================================================

csv_files = get_csv_files()
runs = get_runs()
api_online, health = get_api_health()

df = None
preferred = next(
    (x for x in csv_files if Path(x).name.lower() == "samples.csv"),
    csv_files[0] if csv_files else None
)

if preferred:
    try:
        df = read_csv(preferred)
    except Exception:
        df = None

comparison = read_json(ARTIFACTS / "comparison.json")

architectures = [
    ("CNN", "C", "image", "CNN image-only classifier", "#e8efff"),
    ("LSTM", "L", "text", "BiLSTM text-only classifier", "#dcf8f0"),
    ("BERT", "B", "text", "BERT text-only classifier", "#f0e8ff"),
    ("CNN_LSTM", "CL", "multimodal", "CNN + BiLSTM multimodal fusion", "#fff0df"),
    ("CNN_BERT", "CB", "multimodal", "CNN + BERT multimodal fusion", "#ffe7ef")
]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">⬡ SENTRA <span>AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="workspace">WORKSPACE</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Architectures",
            "Datasets",
            "Prediction Lab",
            "Training Monitor",
            "Diagnostics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("**SENTRA ENGINE**")
    st.caption("Local AI environment")
    st.markdown(
        "🟢 API Connected" if api_online else "🔴 API Offline"
    )
    st.caption("Multimodal Intelligence Platform")

# ============================================================
# HEADER
# ============================================================

left, right = st.columns([5, 1])

with left:
    st.caption(f"Workspace / {page}")

with right:
    if st.button("↻ Refresh"):
        st.cache_data.clear()
        st.rerun()

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown('<div class="hero-label">MULTIMODAL INTELLIGENCE PLATFORM</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">Intelligence Overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Your neural systems, experiments and inference environment at a glance.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    total_rows = len(df) if df is not None else 0
    architecture_count = len(architectures)
    run_count = len(runs)

    device = health.get("device", "—") if api_online else "—"

    cols = st.columns(4)
    with cols[0]:
        card("SYSTEM STATUS", "ONLINE" if api_online else "OFFLINE",
             "API connection", "")
    with cols[1]:
        card("ARCHITECTURES", "5", "Registered neural models", "blue")
    with cols[2]:
        card("TRAINING RUNS", run_count, "Recorded experiments", "purple")
    with cols[3]:
        card("COMPUTE DEVICE", str(device).upper(),
             "Inference environment", "orange")

    # ARCHITECTURE REGISTRY

    heading("Neural Architecture Registry", "01 / MODEL CATALOG")

    arch_cols = st.columns(3)

    for i, (name, symbol, modality, description, color) in enumerate(architectures):
        with arch_cols[i % 3]:
            count = sum(
                1 for r in runs
                if name.lower() in r["Run"].lower()
                or name.lower() in r["Model"].lower()
            )

            st.markdown(f"""
            <div class="architecture">
                <div class="architecture-icon">{symbol}</div>
                <h3>{name}</h3>
                <p>{modality}</p>
                <p>{description}</p>
                <hr>
                <p>{count} registered runs</p>
            </div>
            """, unsafe_allow_html=True)

    # DATASET INTELLIGENCE

    heading("Dataset Intelligence", "02 / DATA OVERVIEW")

    if df is not None:
        label_col = find_label(df)

        with st.container(border=True):
            left, right = st.columns(2)

            with left:
                st.markdown('<div class="small-label">REGISTERED DATASETS</div>',
                            unsafe_allow_html=True)
                st.markdown(f"## {len(df):,} samples")
                st.caption(f"Processed dataset: {Path(preferred).name}")
                st.caption(f"{len(df.columns)} available features")

                st.dataframe(df.head(5), width="stretch", hide_index=True)

            with right:
                st.markdown('<div class="small-label">SENTIMENT DISTRIBUTION</div>',
                            unsafe_allow_html=True)

                if label_col:
                    counts = df[label_col].fillna("Missing").astype(str).value_counts()

                    fig = go.Figure()

                    palette = ["#5d7cf4", "#9878ed", "#35c9a8", "#f29b6d"]

                    for i, (label, count) in enumerate(counts.items()):
                        percentage = count / max(len(df), 1) * 100

                        fig.add_trace(go.Bar(
                            x=[percentage],
                            y=[str(label)],
                            orientation="h",
                            marker_color=palette[i % len(palette)],
                            text=[f"{percentage:.1f}%"],
                            textposition="outside",
                            name=str(label)
                        ))

                    fig.update_layout(
                        barmode="stack",
                        xaxis=dict(range=[0, 100], title="Percentage"),
                        showlegend=False,
                        height=260,
                        template="plotly_white",
                        paper_bgcolor="rgba(255,255,255,0)",
                        plot_bgcolor="rgba(255,255,255,0)"
                    )

                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No recognised sentiment label column found.")

    else:
        st.warning("Processed dataset was not found.")

    # QUICK ANALYTICS

    heading("Dataset Analytics", "03 / VISUAL INSIGHTS")

    if df is not None:
        left, right = st.columns(2)

        with left:
            fig = px.bar(
                x=["Rows", "Columns"],
                y=[len(df), len(df.columns)],
                title="Dataset Dimensions",
                text_auto=True
            )
            plot_style(fig)
            st.plotly_chart(fig, width="stretch")

        with right:
            numeric = df.select_dtypes(include="number")

            if not numeric.empty:
                fig = px.histogram(
                    numeric,
                    x=numeric.columns[0],
                    title=f"Distribution: {numeric.columns[0]}",
                    color_discrete_sequence=["#5c7cf4"]
                )
                plot_style(fig)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numeric columns available for distribution analysis.")

    # MODEL COMPARISON

    heading("Model Performance", "04 / BENCHMARK")

    if comparison:
        records = safe_json_records(comparison)
        comparison_df = pd.DataFrame(records)

        if not comparison_df.empty:
            st.dataframe(comparison_df, width="stretch", hide_index=True)

            metric = next(
                (m for m in ["macro_f1", "f1_macro", "accuracy", "roc_auc"]
                 if m in comparison_df.columns),
                None
            )

            if metric:
                comparison_df[metric] = pd.to_numeric(
                    comparison_df[metric], errors="coerce"
                )

                comparison_df = comparison_df.dropna(subset=[metric])

                if not comparison_df.empty:
                    fig = px.bar(
                        comparison_df,
                        x="Model" if "Model" in comparison_df else comparison_df.columns[0],
                        y=metric,
                        title=f"Model Comparison — {metric}",
                        color_discrete_sequence=["#5d7cf4"],
                        text_auto=".3f"
                    )
                    plot_style(fig)
                    st.plotly_chart(fig, width="stretch")
    else:
        st.info("No model comparison report found.")

# ============================================================
# ARCHITECTURES
# ============================================================

elif page == "Architectures":

    st.markdown('<div class="hero-label">MODEL CATALOG</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">Neural Architectures</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore the five registered SENTRA AI architectures.</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)

    for i, (name, symbol, modality, description, color) in enumerate(architectures):
        with cols[i % 3]:
            count = sum(
                1 for r in runs
                if name.lower() in r["Run"].lower()
                or name.lower() in r["Model"].lower()
            )

            st.markdown(f"""
            <div class="architecture">
                <div class="architecture-icon">{symbol}</div>
                <h3>{name}</h3>
                <p>{modality.upper()}</p>
                <p>{description}</p>
                <hr>
                <p>{count} registered runs</p>
            </div>
            """, unsafe_allow_html=True)

    st.info("Architecture descriptions are registry labels. Actual model availability depends on saved checkpoints.")

# ============================================================
# DATASETS
# ============================================================

elif page == "Datasets":

    st.markdown('<div class="hero-label">DATA INTELLIGENCE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">Dataset Explorer</div>',
                unsafe_allow_html=True)

    if csv_files:
        options = [
            str(Path(p).relative_to(ROOT))
            for p in csv_files
        ]

        chosen = st.selectbox("Select dataset", options)
        selected = read_csv(str(ROOT / chosen))

        a, b, c = st.columns(3)
        with a:
            card("TOTAL RECORDS", f"{len(selected):,}", "Dataset rows")
        with b:
            card("FEATURES", len(selected.columns), "Columns", "blue")
        with c:
            card("MISSING VALUES", int(selected.isna().sum().sum()),
                 "Across all columns", "purple")

        label = find_label(selected)

        if label:
            counts = selected[label].fillna("Missing").astype(str).value_counts()

            fig = px.pie(
                names=counts.index,
                values=counts.values,
                hole=.55,
                title="Sentiment Distribution",
                color_discrete_sequence=[
                    "#5d7cf4", "#35c9a8", "#9878ed", "#f29b6d"
                ]
            )
            plot_style(fig)
            st.plotly_chart(fig, width="stretch")

        st.markdown("### Dataset preview")
        st.dataframe(selected, width="stretch", hide_index=True)

        st.download_button(
            "Download CSV",
            selected.to_csv(index=False).encode("utf-8"),
            file_name=Path(chosen).name,
            mime="text/csv"
        )

        st.markdown("### Column information")
        info = pd.DataFrame({
            "Column": selected.columns.astype(str),
            "Data Type": selected.dtypes.astype(str).values,
            "Missing": selected.isna().sum().values,
            "Unique": selected.nunique().values
        })

        st.dataframe(info, width="stretch", hide_index=True)

    else:
        st.warning("No processed CSV datasets found.")

# ============================================================
# PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":

    st.markdown('<div class="hero-label">LIVE INFERENCE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">AI Prediction Lab</div>',
                unsafe_allow_html=True)
    st.caption("Submit text to your existing SENTRA AI inference API.")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### TEXT SENTIMENT ANALYSIS")

            model = st.selectbox(
                "Select architecture",
                ["lstm", "bert", "cnn_bert", "cnn_lstm"]
            )

            user_text = st.text_area(
                "Text input",
                placeholder="Enter a sentence, review or social media post...",
                height=180
            )

            run_prediction = st.button(
                "RUN SENTIMENT ANALYSIS →",
                type="primary"
            )

    with right:
        with st.container(border=True):
            st.markdown("### PREDICTION OUTPUT")

            if "prediction_result" not in st.session_state:
                st.session_state.prediction_result = None

            if run_prediction:
                if not user_text.strip():
                    st.warning("Enter some text first.")
                else:
                    try:
                        # Discover prediction endpoints from FastAPI OpenAPI.
                        schema = requests.get(
                            f"{API}/openapi.json", timeout=8
                        ).json()

                        candidates = []

                        for path, methods in schema.get("paths", {}).items():
                            if any(x in path.lower() for x in ["predict", "inference"]):
                                if "post" in methods:
                                    candidates.append(path)

                        if not candidates:
                            st.error("No prediction POST endpoint was found in the backend OpenAPI schema.")
                        else:
                            selected_path = st.selectbox(
                                "Detected API endpoint",
                                candidates,
                                key="prediction_endpoint"
                            )

                            payload_options = [
                                {"text": user_text, "model": model},
                                {"input": user_text, "model": model},
                                {"text": user_text},
                                {"input": user_text}
                            ]

                            result = None
                            last_error = None

                            for payload in payload_options:
                                response = requests.post(
                                    f"{API}{selected_path}",
                                    json=payload,
                                    timeout=60
                                )

                                if response.ok:
                                    result = response.json()
                                    break

                                last_error = response.text

                            if result is not None:
                                st.session_state.prediction_result = result
                            else:
                                st.error("Prediction request failed.")
                                st.code(str(last_error))

                    except Exception as e:
                        st.error(f"Prediction error: {e}")

            result = st.session_state.prediction_result

            if result:
                label = (
                    result.get("label")
                    or result.get("sentiment")
                    or result.get("prediction")
                    or "RESULT"
                )

                st.markdown(f"""
                <div class="output-box">
                    <div class="small-label">SENTIMENT CLASSIFICATION</div>
                    <h2>{str(label).upper()}</h2>
                    <p>Model response received from SENTRA AI.</p>
                </div>
                """, unsafe_allow_html=True)

                st.json(result)

                probabilities = result.get("probabilities")

                if isinstance(probabilities, dict):
                    p_df = pd.DataFrame({
                        "Sentiment": list(probabilities.keys()),
                        "Probability": list(probabilities.values())
                    })

                    fig = px.bar(
                        p_df,
                        x="Sentiment",
                        y="Probability",
                        title="Prediction Probabilities",
                        color_discrete_sequence=["#35c9a8"]
                    )
                    plot_style(fig)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.markdown("""
                <div class="output-box">
                    <div class="small-label">SENTIMENT CLASSIFICATION</div>
                    <h2>AWAITING INPUT</h2>
                    <p>Your model prediction will appear here.</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TRAINING MONITOR
# ============================================================

elif page == "Training Monitor":

    st.markdown('<div class="hero-label">EXPERIMENT TRACKING</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">Training Monitor</div>',
                unsafe_allow_html=True)

    if runs:
        run_df = pd.DataFrame(runs)

        a, b, c = st.columns(3)
        with a:
            card("TOTAL RUNS", len(runs), "Registered experiments")
        with b:
            card("MODELS", run_df["Model"].nunique(), "Model families", "blue")
        with c:
            card("DATASETS", run_df["Dataset"].nunique(), "Used in experiments", "purple")

        st.markdown("### Training run registry")
        st.dataframe(run_df, width="stretch", hide_index=True)

        status_counts = run_df["Status"].astype(str).value_counts()

        fig = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Training Status",
            labels={"x": "Status", "y": "Runs"},
            color_discrete_sequence=["#5d7cf4"]
        )
        plot_style(fig)
        st.plotly_chart(fig, width="stretch")

        chosen = st.selectbox("Inspect training run", run_df["Run"].tolist())

        run_path = RUNS / chosen

        st.markdown("### Experiment configuration")
        st.json({
            "run": chosen,
            "config": read_json(run_path / "config.json"),
            "status": read_json(run_path / "status.json")
        })

    else:
        st.warning("No saved training runs found.")

# ============================================================
# DIAGNOSTICS
# ============================================================

elif page == "Diagnostics":

    st.markdown('<div class="hero-label">SYSTEM DIAGNOSTICS</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">Diagnostics</div>',
                unsafe_allow_html=True)

    if st.button("Refresh diagnostics"):
        st.cache_data.clear()
        api_online, health = get_api_health()

    a, b = st.columns(2)

    with a:
        card("API STATUS", "ONLINE" if api_online else "OFFLINE",
             "FastAPI health endpoint")

    with b:
        card("COMPUTE DEVICE",
             str(health.get("device", "Unknown")).upper(),
             "Backend-reported device")

    st.markdown("### Backend health response")
    st.json(health)

    st.markdown("### API documentation")
    st.link_button("OPEN FASTAPI DOCS", f"{API}/docs")

    st.markdown("### Project environment")
    st.json({
        "Project": str(ROOT),
        "Python": os.sys.version.split()[0],
        "Dataset files": len(csv_files),
        "Training run folders": len(runs),
        "Model artifact directory": str(RUNS),
        "API URL": API
    })

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("SENTRA AI • MULTIMODAL INTELLIGENCE PLATFORM • DEMO ENVIRONMENT")
