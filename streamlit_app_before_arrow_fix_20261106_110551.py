import os
import json
import glob
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
API_URL = os.getenv("SENTRA_API_URL", "http://127.0.0.1:8001")

st.set_page_config(
    page_title="SENTRA AI | Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #f5f7fb;
}
[data-testid="stSidebar"] {
    background: #101a32;
}
[data-testid="stSidebar"] * {
    color: #f5f7ff !important;
}
.hero {
    background: linear-gradient(120deg,#111c38,#243b68);
    padding: 28px 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 34px;
    margin: 0;
    font-weight: 800;
}
.hero p {
    color: #cbd5e8;
    margin-bottom: 0;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e8edf5;
    box-shadow: 0 3px 14px rgba(20,40,80,.04);
}
.metric-label {
    color: #738096;
    font-size: 13px;
    font-weight: 600;
}
.metric-value {
    color: #17233d;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
}
.section-title {
    color: #17233d;
    font-weight: 750;
    font-size: 21px;
    margin-top: 15px;
}
.small-note {
    color: #77839a;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def discover_csv_files():
    patterns = [
        str(DATA_DIR / "processed" / "**" / "*.csv"),
        str(DATA_DIR / "**" / "*.csv")
    ]
    files = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    return sorted(
        [Path(x) for x in files if ".venv" not in x and "node_modules" not in x],
        key=lambda p: (0 if p.name.lower() == "samples.csv" else 1, str(p))
    )


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_runs():
    result = []
    run_dir = ARTIFACTS / "runs"
    if not run_dir.exists():
        return result

    for folder in sorted(run_dir.iterdir()):
        if not folder.is_dir():
            continue
        status = load_json(folder / "status.json")
        config = load_json(folder / "config.json")
        result.append({
            "Run": folder.name,
            "Model": config.get("model", folder.name.split("_")[0]),
            "Dataset": config.get("dataset", "Not specified"),
            "Status": status.get("status", "Unknown"),
            "Epochs": status.get("total_epochs", status.get("epochs", "—")),
            "Device": status.get("device", config.get("device", "—")),
            "Path": str(folder)
        })
    return result


def find_records(obj):
    records = []
    if isinstance(obj, list):
        for item in obj:
            records.extend(find_records(item))
    elif isinstance(obj, dict):
        metric_keys = [
            "accuracy", "macro_f1", "f1_macro",
            "roc_auc", "auc", "precision", "recall"
        ]
        if any(k in obj for k in metric_keys):
            row = dict(obj)
            row["Model"] = (
                obj.get("model")
                or obj.get("run_id")
                or obj.get("name")
                or obj.get("run")
                or "Model"
            )
            records.append(row)
        else:
            for value in obj.values():
                records.extend(find_records(value))
    return records


def find_label_column(df):
    candidates = [
        "sentiment", "label", "sentiment_label",
        "polarity", "target", "class", "category"
    ]
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def metric_card(label, value, subtitle=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="small-note">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True
    )


csv_files = discover_csv_files()
runs = get_runs()

comparison_path = ARTIFACTS / "comparison.json"
comparison_data = load_json(comparison_path)
comparison_records = find_records(comparison_data)

df = None
selected_csv = None
if csv_files:
    preferred = next(
        (p for p in csv_files if p.name.lower() == "samples.csv"),
        csv_files[0]
    )
    selected_csv = preferred
    try:
        df = load_csv(str(preferred))
    except Exception as e:
        st.error(f"Could not read dataset: {e}")

label_col = find_label_column(df) if df is not None else None

with st.sidebar:
    st.markdown("## 🧠 SENTRA AI")
    st.caption("Multimodal Intelligence Platform")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Dataset Explorer",
            "Model Benchmark",
            "Training Runs",
            "API Monitor",
            "About SENTRA"
        ]
    )

    st.divider()
    st.caption("PROJECT STATUS")

    st.write("Dataset")
    st.write("🟢 Available" if df is not None else "🔴 Not found")

    st.write("Model artifacts")
    st.write("🟢 Available" if runs else "🟡 No run folders found")

    st.write("Comparison report")
    st.write("🟢 Available" if comparison_records else "🟡 Not found")

    st.divider()
    st.caption("SENTRA AI • DEMO DASHBOARD")


if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>SENTRA AI Intelligence Dashboard</h1>
        <p>Multimodal sentiment analysis, model evaluation and intelligent insights.</p>
    </div>
    """, unsafe_allow_html=True)

    total_rows = len(df) if df is not None else 0
    total_columns = len(df.columns) if df is not None else 0
    total_runs = len(runs)
    completed_runs = sum(
        1 for r in runs if str(r["Status"]).lower() in ["completed", "success", "finished"]
    )

    a, b, c, d = st.columns(4)
    with a:
        metric_card("DATASET RECORDS", f"{total_rows:,}", "Loaded from processed CSV")
    with b:
        metric_card("FEATURE COLUMNS", total_columns, "Available dataset columns")
    with c:
        metric_card("TRAINING RUNS", total_runs, "Discovered artifact folders")
    with d:
        metric_card("COMPLETED RUNS", completed_runs, "Based on saved run status")

    st.markdown('<div class="section-title">Analytics Overview</div>', unsafe_allow_html=True)

    if df is not None:
        left, right = st.columns(2)

        with left:
            st.markdown("**Dataset structure**")
            fig = px.bar(
                x=["Rows", "Columns"],
                y=[total_rows, total_columns],
                labels={"x": "Dataset component", "y": "Count"},
                title="Dataset dimensions"
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("**Sentiment distribution**")
            if label_col:
                counts = df[label_col].fillna("Missing").astype(str).value_counts()
                fig = px.pie(
                    names=counts.index,
                    values=counts.values,
                    hole=.55,
                    title=f"Distribution by {label_col}"
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recognised sentiment/label column was found. Check Dataset Explorer.")

        st.markdown("**Dataset preview**")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("No processed CSV dataset was detected. Check the data/processed folder.")

    st.markdown("**Model benchmark snapshot**")
    if comparison_records:
        st.dataframe(pd.DataFrame(comparison_records), use_container_width=True)
    else:
        st.info("No readable model comparison records found in artifacts/comparison.json.")


elif page == "Dataset Explorer":
    st.markdown('<div class="hero"><h1>Dataset Explorer</h1><p>Explore, filter and inspect your processed data.</p></div>', unsafe_allow_html=True)

    if not csv_files:
        st.warning("No CSV files were found inside the data folder.")
    else:
        options = [str(p.relative_to(ROOT)) for p in csv_files]
        chosen = st.selectbox("Select dataset", options)
        current = load_csv(str(ROOT / chosen))

        st.write(f"**Selected file:** `{chosen}`")
        x1, x2, x3 = st.columns(3)
        with x1:
            metric_card("ROWS", f"{len(current):,}")
        with x2:
            metric_card("COLUMNS", len(current.columns))
        with x3:
            metric_card("MISSING VALUES", f"{int(current.isna().sum().sum()):,}")

        st.markdown("### Data preview")
        st.dataframe(current.head(100), use_container_width=True)

        st.markdown("### Column information")
        info = pd.DataFrame({
            "Column": current.columns,
            "Data type": [str(x) for x in current.dtypes],
            "Missing values": current.isna().sum().values,
            "Unique values": current.nunique(dropna=True).values
        })
        st.dataframe(info, use_container_width=True)

        st.markdown("### Missing-value analysis")
        missing = current.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing):
            fig = px.bar(
                x=missing.index.astype(str),
                y=missing.values,
                labels={"x": "Column", "y": "Missing values"},
                title="Missing values by column"
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected.")

        st.download_button(
            "Download selected dataset",
            current.to_csv(index=False).encode("utf-8"),
            file_name=Path(chosen).name,
            mime="text/csv"
        )


elif page == "Model Benchmark":
    st.markdown('<div class="hero"><h1>Model Benchmark</h1><p>Compare model evaluation results and identify the strongest experiment.</p></div>', unsafe_allow_html=True)

    if comparison_records:
        metrics_df = pd.DataFrame(comparison_records)
        st.dataframe(metrics_df, use_container_width=True)

        metric_candidates = [
            "macro_f1", "f1_macro", "accuracy",
            "roc_auc", "auc", "precision", "recall"
        ]
        available = [m for m in metric_candidates if m in metrics_df.columns]

        if available:
            metric = st.selectbox("Choose evaluation metric", available)
            plot_df = metrics_df.copy()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna(subset=[metric])

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x="Model",
                    y=metric,
                    color="Model",
                    title=f"Model comparison — {metric}",
                    text_auto=".3f"
                )
                fig.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                best = plot_df.loc[plot_df[metric].idxmax()]
                st.success(f"Highest recorded {metric}: {best['Model']} ({best[metric]:.4f})")
            else:
                st.info("Metric values could not be converted into numbers.")
        else:
            st.info("The comparison file was found, but its metric columns were not recognised.")
    else:
        st.warning("Model comparison file not found or no metrics could be extracted.")
        st.caption("Expected location: artifacts/comparison.json")

    st.markdown("### Known experiment families")
    st.write("CNN • LSTM • BERT • CNN-LSTM • CNN-BERT")


elif page == "Training Runs":
    st.markdown('<div class="hero"><h1>Training Runs</h1><p>Inspect experiment configurations and saved training status.</p></div>', unsafe_allow_html=True)

    if runs:
        run_df = pd.DataFrame(runs).drop(columns=["Path"])
        st.dataframe(run_df, use_container_width=True)

        status_counts = run_df["Status"].astype(str).value_counts()
        fig = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            labels={"x": "Training status", "y": "Number of runs"},
            title="Training run status"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        chosen_run = st.selectbox("Inspect run", [r["Run"] for r in runs])
        selected_run = next(r for r in runs if r["Run"] == chosen_run)
        st.json({
            "run": selected_run["Run"],
            "model": selected_run["Model"],
            "dataset": selected_run["Dataset"],
            "status": selected_run["Status"],
            "epochs": selected_run["Epochs"],
            "device": selected_run["Device"]
        })
    else:
        st.warning("No model run folders found under artifacts/runs.")

    st.caption("Run status is read from saved status.json files; it is not a live training process monitor.")


elif page == "API Monitor":
    st.markdown('<div class="hero"><h1>API Monitor</h1><p>Check the connection to your SENTRA AI backend.</p></div>', unsafe_allow_html=True)

    st.write("Backend URL:", API_URL)

    if st.button("Check backend health", type="primary"):
        try:
            response = requests.get(
                f"{API_URL.rstrip('/')}/api/health",
                timeout=8
            )
            st.write("HTTP status:", response.status_code)
            if response.ok:
                st.success("Backend health endpoint responded successfully.")
                st.json(response.json())
            else:
                st.error("Backend returned an unsuccessful HTTP status.")
                st.code(response.text[:2000])
        except requests.RequestException as e:
            st.error("Backend is not reachable.")
            st.code(str(e))

    st.info("The dashboard can run independently. Live predictions require a reachable backend and its actual prediction endpoint.")


else:
    st.markdown('<div class="hero"><h1>About SENTRA AI</h1><p>Intelligent multimodal sentiment analysis.</p></div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project overview

    SENTRA AI is designed to analyse sentiment using machine-learning and
    deep-learning approaches across text and image inputs.

    ### Model families
    - **CNN:** image-based feature learning.
    - **LSTM:** sequential text modelling.
    - **BERT:** contextual language understanding.
    - **CNN-LSTM:** multimodal architecture.
    - **CNN-BERT:** multimodal architecture.

    ### Dashboard modules
    - Dataset inspection and exploratory analytics.
    - Model benchmark comparison.
    - Training experiment inspection.
    - Backend health monitoring.

    ### Technical stack
    Python · PyTorch · Pandas · Plotly · Streamlit · FastAPI
    """)

    st.caption("SENTRA AI — Project demonstration interface")


st.divider()
st.caption("SENTRA AI | Streamlit Demo | Results shown are based on locally available project artifacts.")
