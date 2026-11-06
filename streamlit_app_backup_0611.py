import os, json, glob, time
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
API_BASE = os.getenv("SENTRA_API_URL", "http://127.0.0.1:8001").rstrip("/")

st.set_page_config(page_title="SENTRA AI | Intelligence Platform", page_icon="ðŸ§ ",
                   layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --navy:#142443; --navy2:#1b3156; --ink:#172544; --muted:#7183a5; --line:#e1e8f4; --bg:#f3f6fc; --teal:#37d6bd; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(243,246,252,.92); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111e38 0%,#1b3156 100%); border-right:1px solid #263c60; }
[data-testid="stSidebar"] * { color:#eef4ff !important; }
[data-testid="stSidebar"] .stRadio label { padding:9px 12px; border-radius:11px; font-weight:600; }
[data-testid="stSidebar"] .stRadio label:hover { background:#29436b; }
.block-container { max-width:1500px; padding-top:1.8rem; padding-bottom:3rem; }
h1,h2,h3 { color:#142443; font-family:'Space Grotesk',sans-serif; }
.hero { background:linear-gradient(115deg,#142443,#263f6d); color:white; padding:28px 32px; border-radius:20px; margin:4px 0 22px; box-shadow:0 14px 35px rgba(22,42,78,.13); }
.hero h1 { color:white !important; font-size:30px; margin:0 0 7px; }
.hero p { color:#d6e2fa; margin:0; }
.eyebrow { color:#6684cf; font-size:10px; font-weight:800; letter-spacing:2px; text-transform:uppercase; margin-bottom:7px; }
.metric-card { background:white; border:1px solid var(--line); border-radius:17px; padding:18px 19px; min-height:122px; box-shadow:0 8px 22px rgba(31,55,96,.045); position:relative; overflow:hidden; }
.metric-card:after { content:''; position:absolute; right:-9px; top:-18px; width:72px; height:72px; background:#e7efff; border-radius:0 0 0 30px; opacity:.9; }
.metric-label { color:#8191af; font-size:10px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
.metric-value { color:#172544; font-family:'Space Grotesk'; font-size:29px; font-weight:800; margin-top:10px; }
.metric-note { color:#7b8eae; font-size:11px; margin-top:4px; }
.panel { background:white; border:1px solid var(--line); border-radius:17px; padding:19px 20px; box-shadow:0 8px 22px rgba(31,55,96,.04); }
.model-card { background:white; border:1px solid var(--line); border-radius:16px; padding:18px; min-height:185px; box-shadow:0 7px 20px rgba(31,55,96,.04); }
.model-icon { display:inline-flex; width:42px; height:42px; align-items:center; justify-content:center; background:#eaf0ff; color:#5277ee; border-radius:13px; font-weight:800; margin-bottom:12px; }
.small-muted { color:#7b8eae; font-size:12px; }
.section-head { display:flex; justify-content:space-between; align-items:center; margin:22px 0 12px; }
.section-head span { color:#8293b1; font-size:10px; letter-spacing:1.5px; font-weight:700; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:700; }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#526cf0,#8062e9); border:0; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

# ---------- Data helpers ----------
def find_samples():
    candidates = [
        DATA_DIR / "processed" / "mvsa_single" / "samples.csv",
        DATA_DIR / "processed" / "samples.csv",
    ]
    for p in candidates:
        if p.exists(): return p
    found = list(DATA_DIR.glob("**/samples.csv")) if DATA_DIR.exists() else []
    return found[0] if found else None

@st.cache_data(show_spinner=False)
def load_dataset(path_string, modified):
    if not path_string: return pd.DataFrame()
    try:
        df = pd.read_csv(path_string)
        # Normalize mixed-type columns before Streamlit/Arrow renders them.
        for col in df.columns:
            if df[col].dtype == "object":
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null = df[col].notna().sum()
                if non_null and numeric.notna().sum() >= max(1, int(non_null * .85)):
                    df[col] = numeric
                else:
                    df[col] = df[col].fillna("").astype(str)
        return df
    except Exception as e:
        st.warning(f"Could not read dataset: {e}")
        return pd.DataFrame()

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

def get_runs():
    runs=[]
    base=ARTIFACTS/"runs"
    if base.exists():
        for folder in sorted(base.iterdir(), key=lambda p:p.stat().st_mtime if p.is_dir() else 0, reverse=True):
            if not folder.is_dir(): continue
            cfg=read_json(folder/"config.json")
            status=read_json(folder/"status.json")
            runs.append({"Run":folder.name,"Model":cfg.get("model",folder.name.split("_")[0]),
                         "Dataset":cfg.get("dataset","â€”"),"Status":status.get("status","unknown"),
                         "Epochs":cfg.get("epochs",status.get("total_epochs","â€”")),
                         "Device":cfg.get("device",status.get("device","â€”")),
                         "Updated":datetime.fromtimestamp(folder.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    return runs

def api_get(path, timeout=2):
    try:
        r=requests.get(API_BASE+path, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def safe_label_counts(df):
    for c in ["label","sentiment","sentiment_label","class"]:
        if c in df.columns:
            return df[c].astype(str).value_counts().rename_axis("Sentiment").reset_index(name="Count")
    return pd.DataFrame()

# ---------- Load data ----------
sample_path=find_samples()
df=load_dataset(str(sample_path) if sample_path else "", sample_path.stat().st_mtime if sample_path else 0)
runs=get_runs()
health, health_error=api_get("/api/health")
models, models_error=api_get("/api/models")
comparison=read_json(ARTIFACTS/"comparison.json")
if isinstance(comparison,dict):
    comp_rows=comparison.get("models",comparison.get("results",[]))
else: comp_rows=comparison
if not isinstance(comp_rows,list): comp_rows=[]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div style='padding:12px 4px 20px;font-family:Space Grotesk;font-size:23px;font-weight:800'>â¬¡ SENTRA <span style='color:#45dec6'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow' style='color:#8fa6cc'>WORKSPACE</div>", unsafe_allow_html=True)
    page=st.radio("Navigation",["â—ˆ  Overview","â¬¡  Architectures","â–¤  Datasets","âœ¦  Prediction Lab","âŒ  Training Monitor","âŒ˜  Diagnostics"],label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**SENTRA ENGINE**")
    st.caption("Local AI environment")
    if health:
        st.markdown("ðŸŸ¢ API Connected")
    else:
        st.markdown("ðŸ”´ API Offline")
    st.caption("Multimodal Intelligence Platform")
    st.markdown("---")
    st.caption("SENTRA AI â€¢ DEMO DASHBOARD")

# ---------- Header ----------
st.markdown("<div class='eyebrow'>MULTIMODAL INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>ðŸ§  SENTRA AI Intelligence Dashboard</h1><p>Your neural systems, experiments and inference environment at a glance.</p></div>", unsafe_allow_html=True)

def metric(col, label, value, note):
    col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",unsafe_allow_html=True)

def section(title, tag):
    st.markdown(f"<div class='section-head'><h3 style='font-size:17px;margin:0'>{title}</h3><span>{tag}</span></div>",unsafe_allow_html=True)

architecture=[
    ("CNN","C","image","CNN image-only classifier",sum("cnn" in x["Run"].lower() for x in runs)),
    ("LSTM","L","text","BiLSTM text-only classifier",sum("lstm" in x["Run"].lower() for x in runs)),
    ("BERT","B","text","BERT text-only classifier",sum("bert" in x["Run"].lower() and "cnn_bert" not in x["Run"].lower() for x in runs)),
    ("CNN_LSTM","CL","multimodal","CNN + BiLSTM multimodal fusion",sum("cnn_lstm" in x["Run"].lower() for x in runs)),
    ("CNN_BERT","CB","multimodal","CNN + BERT multimodal fusion",sum("cnn_bert" in x["Run"].lower() for x in runs)),
]

# ---------- Overview ----------
if page.startswith("â—ˆ"):
    c1,c2,c3,c4=st.columns(4)
    metric(c1,"SYSTEM STATUS","ONLINE" if health else "OFFLINE","API connection")
    metric(c2,"ARCHITECTURES","5","Registered neural models")
    metric(c3,"TRAINING RUNS",str(len(runs)),"Discovered experiment folders")
    metric(c4,"COMPUTE DEVICE",str((health or {}).get("device","â€”")).upper(),"Inference environment")
    section("Neural Architecture Registry","01 / MODEL CATALOG")
    cols=st.columns(3)
    for i,(name,icon,modality,desc,count) in enumerate(architecture):
        with cols[i%3]:
            st.markdown(f"<div class='model-card'><div class='model-icon'>{icon}</div><h4 style='margin:0;color:#172544'>{name}</h4><p class='small-muted'>{modality}</p><p class='small-muted'>{desc}</p><hr><span class='small-muted'>{count} registered runs</span></div>",unsafe_allow_html=True)
    section("Dataset Intelligence","02 / DATA OVERVIEW")
    left,right=st.columns([1.1,1])
    with left:
        st.markdown("<div class='panel'>",unsafe_allow_html=True)
        st.markdown("**REGISTERED DATASET**")
        st.markdown(f"<div class='metric-value'>{len(df):,} samples</div>",unsafe_allow_html=True)
        st.caption(f"Source: {sample_path.relative_to(ROOT) if sample_path else 'samples.csv not found'}")
        st.caption(f"{len(df.columns)} available features" if not df.empty else "Dataset unavailable")
        if not df.empty:
            st.dataframe(df.head(5),width="stretch",hide_index=True)
        else:
            st.info("Processed dataset not found. Expected data/processed/mvsa_single/samples.csv.")
        st.markdown("</div>",unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'>",unsafe_allow_html=True)
        st.markdown("**SENTIMENT DISTRIBUTION**")
        counts=safe_label_counts(df)
        if not counts.empty:
            fig=px.bar(counts,x="Count",y="Sentiment",orientation="h",text="Count",color="Sentiment",color_discrete_sequence=["#5e78f4","#9875e9","#31c9ae"])
            fig.update_layout(height=260,margin=dict(l=5,r=10,t=10,b=5),showlegend=False,paper_bgcolor="white",plot_bgcolor="white",font_color="#263858")
            st.plotly_chart(fig,width="stretch")
        else:
            st.info("No recognized sentiment label column found.")
        st.markdown("</div>",unsafe_allow_html=True)
    section("Dataset Analytics","03 / VISUAL INSIGHTS")
    if not df.empty:
        a,b=st.columns(2)
        with a:
            fig=go.Figure(go.Bar(x=["Rows","Columns"],y=[len(df),len(df.columns)],marker_color=["#80c5f5","#6b7ff0"],text=[len(df),len(df.columns)],textposition="outside"))
            fig.update_layout(title="Dataset dimensions",height=320,margin=dict(l=15,r=15,t=45,b=25),paper_bgcolor="white",plot_bgcolor="white")
            st.plotly_chart(fig,width="stretch")
        with b:
            numeric=df.select_dtypes(include=np.number)
            if not numeric.empty:
                col=st.selectbox("Explore numeric feature",numeric.columns,key="overview_numeric")
                fig=px.histogram(df,x=col,nbins=25,color_discrete_sequence=["#5e78f4"],title=f"Distribution: {col}")
                fig.update_layout(height=320,margin=dict(l=15,r=15,t=45,b=25),paper_bgcolor="white",plot_bgcolor="white")
                st.plotly_chart(fig,width="stretch")
            else:
                st.info("No numeric features available for distribution plots.")
    section("Model Performance","04 / BENCHMARK")
    if comp_rows:
        st.dataframe(pd.json_normalize(comp_rows),width="stretch",hide_index=True)
    else:
        st.info("No comparison results found at artifacts/comparison.json.")

# ---------- Architectures ----------
elif page.startswith("â¬¡"):
    section("Neural Architecture Registry","MODEL CATALOG")
    for row in architecture:
        pass
    arch_df=pd.DataFrame([{"Architecture":x[0],"Modality":x[2],"Description":x[3],"Registered runs":x[4]} for x in architecture])
    st.dataframe(arch_df,width="stretch",hide_index=True)
    st.info("SENTRA supports CNN, LSTM, BERT and multimodal CNN-LSTM / CNN-BERT pipelines.")
    if models:
        st.json(models)

# ---------- Datasets ----------
elif page.startswith("â–¤"):
    section("Dataset Explorer","DATA MANAGEMENT")
    if df.empty:
        st.error("Processed dataset not found.")
    else:
        st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        search=st.text_input("Filter rows (search across visible text)",placeholder="Type a keyword...")
        view=df
        if search:
            mask=view.astype(str).apply(lambda s:s.str.contains(search,case=False,na=False)).any(axis=1)
            view=view[mask]
        st.dataframe(view,width="stretch",height=430,hide_index=True)
        st.download_button("â¬‡ Download filtered CSV",view.to_csv(index=False).encode("utf-8"),"sentra_dataset.csv","text/csv")
        st.markdown("#### Column profile")
        profile=pd.DataFrame({"Column":df.columns,"Data type":[str(x) for x in df.dtypes],"Missing values":df.isna().sum().values,"Unique values":df.nunique(dropna=False).values})
        st.dataframe(profile,width="stretch",hide_index=True)
        numeric=df.select_dtypes(include=np.number)
        if not numeric.empty:
            selected=st.selectbox("Numeric column",numeric.columns)
            fig=px.histogram(df,x=selected,nbins=30,color_discrete_sequence=["#5e78f4"])
            st.plotly_chart(fig,width="stretch")

# ---------- Prediction ----------
elif page.startswith("âœ¦"):
    section("AI Prediction Lab","LIVE INFERENCE")
    st.caption("Text sentiment inference through your running SENTRA API.")
    text_input=st.text_area("TEXT SENTIMENT ANALYSIS",placeholder="Enter a sentence, review or social media post to analyze sentiment...",height=150)
    endpoint=st.text_input("Prediction endpoint",value="/api/predict/text",help="Change this if your FastAPI route uses a different path.")
    if st.button("âš¡ RUN SENTIMENT ANALYSIS",type="primary"):
        if not text_input.strip():
            st.warning("Enter text before running prediction.")
        else:
            result=None; err=None
            for payload in [{"text":text_input},{"text_raw":text_input},{"input":text_input}]:
                try:
                    response=requests.post(API_BASE+endpoint,json=payload,timeout=45)
                    if response.ok:
                        result=response.json(); break
                    err=f"HTTP {response.status_code}: {response.text[:500]}"
                except Exception as e: err=str(e)
            if result:
                st.success("Prediction received")
                label=result.get("label",result.get("sentiment",result.get("prediction","Result")))
                st.markdown(f"<div class='panel'><div class='metric-label'>SENTIMENT CLASSIFICATION</div><div class='metric-value'>{str(label).upper()}</div></div>",unsafe_allow_html=True)
                st.json(result)
                probs=result.get("probabilities")
                if isinstance(probs,dict):
                    p=pd.DataFrame({"Sentiment":list(probs.keys()),"Probability":list(probs.values())})
                    fig=px.bar(p,x="Sentiment",y="Probability",color="Sentiment",text_auto=".2f",color_discrete_sequence=["#5e78f4","#9875e9","#31c9ae"])
                    st.plotly_chart(fig,width="stretch")
            else:
                st.error("Prediction API is unavailable or the endpoint/payload does not match.")
                st.code(f"{API_BASE}{endpoint}\n{err or 'No response'}")
                st.info("Start your FastAPI backend, then verify the correct route in http://127.0.0.1:8001/docs.")

# ---------- Training ----------
elif page.startswith("âŒ"):
    section("Training Monitor","EXPERIMENT TRACKING")
    if runs:
        run_df=pd.DataFrame(runs)
        # Ensure mixed values such as Epochs are consistently represented.
        for col in run_df.columns:
            run_df[col]=run_df[col].fillna("â€”").astype(str)
        st.dataframe(run_df,width="stretch",hide_index=True)
        status_counts=run_df["Status"].value_counts().rename_axis("Status").reset_index(name="Runs")
        fig=px.bar(status_counts,x="Status",y="Runs",color="Status",text="Runs",color_discrete_sequence=["#5e78f4","#31c9ae","#f29a64"])
        fig.update_layout(height=300,paper_bgcolor="white",plot_bgcolor="white")
        st.plotly_chart(fig,width="stretch")
        selected=st.selectbox("Inspect run",run_df["Run"].tolist())
        run_path=ARTIFACTS/"runs"/selected
        st.markdown("#### Run configuration")
        st.json(read_json(run_path/"config.json"))
        st.markdown("#### Run status")
        st.json(read_json(run_path/"status.json"))
    else:
        st.info("No training run folders found in artifacts/runs.")
    if st.button("â†» Refresh training data"):
        st.cache_data.clear(); st.rerun()

# ---------- Diagnostics ----------
elif page.startswith("âŒ˜"):
    section("System Diagnostics","HEALTH & FILES")
    st.markdown("#### API connection")
    if health:
        st.success("Backend API is responding.")
        st.json(health)
    else:
        st.error(f"Backend offline: {health_error}")
        st.code(f"API base: {API_BASE}\nExpected health route: {API_BASE}/api/health")
    st.markdown("#### Model registry response")
    if models: st.json(models)
    else: st.warning(f"Could not load /api/models: {models_error}")
    st.markdown("#### Local project assets")
    checks={
        "Processed samples CSV":sample_path,
        "Artifacts directory":ARTIFACTS if ARTIFACTS.exists() else None,
        "Runs directory":ARTIFACTS/"runs" if (ARTIFACTS/"runs").exists() else None,
        "Comparison report":ARTIFACTS/"comparison.json" if (ARTIFACTS/"comparison.json").exists() else None,
    }
    for label,path in checks.items():
        st.write(("ðŸŸ¢" if path and Path(path).exists() else "ðŸ”´"),label, str(path) if path else "Not found")
    st.markdown("#### Runtime")
    st.write({"Python":os.sys.version.split()[0],"Streamlit":st.__version__,"API URL":API_BASE,"Dataset rows":len(df),"Training runs":len(runs)})

st.markdown("---")
st.caption("SENTRA AI â€¢ Multimodal Intelligence Platform â€¢ Local demo dashboard")
