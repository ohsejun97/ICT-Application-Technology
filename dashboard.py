"""
dashboard.py
============
Streamlit real-time dashboard — Bio-AI DTI Pipeline monitoring

Run:
  conda run -n bioinfo streamlit run dashboard.py
"""

import json
import time
import hashlib
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Bio-AI DTI Pipeline Dashboard",
    page_icon="💊",
    layout="wide",
)

ROOT = Path(__file__).parent

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

log_mode = st.sidebar.radio(
    "Log Source",
    ["demo (demo_log.jsonl)", "pipeline (pipeline_log.jsonl)"],
    index=0,
)
LOG_STEM = "demo_log" if log_mode.startswith("demo") else "pipeline_log"
LOG_PATH = ROOT / "results" / f"{LOG_STEM}.jsonl"
SUM_PATH = ROOT / "results" / (LOG_STEM.replace("log", "summary") + ".json")

pkd_high = st.sidebar.slider("HIGH threshold (pKd ≥)", 5.0, 10.0, 7.0, 0.1)
pkd_mod  = st.sidebar.slider("MODERATE threshold (pKd ≥)", 3.0, 7.0, 5.0, 0.1)
auto_ref = st.sidebar.checkbox("Auto-refresh (2s)", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🗑 Clear Logs (click before recording)", use_container_width=True):
    for p in [LOG_PATH, SUM_PATH]:
        if p.exists():
            p.unlink()
    st.sidebar.success("Cleared — run demo.py to start")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** SaProt-650M + ft-ChemBERTa  \n"
    "**Pretrain:** BindingDB 80K (r=0.89)  \n"
    "**Transfer:** DAVIS r=0.87 / KIBA r=0.86"
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("💊 Bio-AI DTI Query Pipeline")
st.caption("Real-time Drug-Target Interaction Prediction Dashboard | SaProt-650M + ft-ChemBERTa")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    records = []
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(records) if records else pd.DataFrame()

def load_summary() -> dict:
    if not SUM_PATH.exists():
        return {}
    try:
        with open(SUM_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

df      = load_log()
summary = load_summary()

if not df.empty:
    def reclassify(pkd):
        if pkd >= pkd_high:  return "HIGH"
        if pkd >= pkd_mod:   return "MODERATE"
        return "LOW"
    df["decision"] = df["pKd"].apply(reclassify)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_seqs = st.tabs(["📡 Live Monitor", "🧬 DAVIS Protein Sequences"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1 — Live Monitor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_live:

    # ── Network stats ─────────────────────────────────────────────────────────
    total    = summary.get("total_queries",  summary.get("network", {}).get("total_queries", 0))
    received = summary.get("received",       summary.get("network", {}).get("received", len(df) if not df.empty else 0))
    dropped  = summary.get("dropped",        summary.get("network", {}).get("dropped", 0))
    loss_pct = summary.get("loss_rate",      summary.get("network", {}).get("loss_rate", 0.0)) * 100
    imputed  = summary.get("imputed",        summary.get("network", {}).get("inference_imputed", 0))
    net_alert= summary.get("network_alert",  summary.get("network", {}).get("network_alert", False))

    mq        = summary.get("model_quality", {})
    n_3di     = mq.get("used_3di",      int(df["used_3di"].sum()) if not df.empty and "used_3di" in df.columns else 0)
    n_inferred= mq.get("total_inferred", received)
    rate_3di  = mq.get("3di_rate",      n_3di / n_inferred if n_inferred else 0)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Queries",   total or received)
    col2.metric("Received",        received)
    col3.metric("Dropped",         dropped, delta=f"-{loss_pct:.1f}%", delta_color="inverse")
    col4.metric("Imputed",         imputed)
    col5.metric("Packet Loss",     f"{loss_pct:.1f}%")
    col6.metric("3Di Hit Rate",    f"{rate_3di:.0%}",
                delta="✅ Structural tokens used" if rate_3di > 0.5 else "⚠️ fallback (#)",
                delta_color="normal" if rate_3di > 0.5 else "inverse")

    if net_alert or (total > 0 and loss_pct > 30):
        st.error("⚠️  ALERT: Packet loss rate exceeds 30% — Network Degraded!")

    st.markdown("---")

    if df.empty:
        st.info(
            f"`{LOG_PATH.name}` not found. Run the pipeline first.\n\n"
            "```bash\npython demo.py\n```"
        )
    else:
        # ── Binding decision summary ──────────────────────────────────────────
        n_high  = (df["decision"] == "HIGH").sum()
        n_mod   = (df["decision"] == "MODERATE").sum()
        n_low   = (df["decision"] == "LOW").sum()
        avg_pkd = df["pKd"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 HIGH",     n_high)
        c2.metric("🟡 MODERATE", n_mod)
        c3.metric("🔴 LOW",      n_low)
        c4.metric("Avg pKd",     f"{avg_pkd:.4f}")

        st.markdown("---")

        # ── pKd time series ───────────────────────────────────────────────────
        st.subheader("📈 pKd Time Series")
        colors = df["decision"].map({"HIGH": "#22c55e", "MODERATE": "#eab308", "LOW": "#ef4444"})
        fig = go.Figure()
        fig.add_hline(y=pkd_high, line_dash="dash", line_color="#22c55e",
                      annotation_text=f"HIGH ≥{pkd_high}", annotation_position="top left")
        fig.add_hline(y=pkd_mod,  line_dash="dash", line_color="#eab308",
                      annotation_text=f"MODERATE ≥{pkd_mod}", annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df)+1)),
            y=df["pKd"],
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2),
            marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
            text=df["drug_name"] + "<br>" + df["protein_name"] + "<br>path: " + df["path"],
            hovertemplate="<b>Query #%{x}</b><br>pKd: %{y:.4f}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Query Index", yaxis_title="pKd",
            height=350, margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa",
            xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Marker color — 🟢 HIGH  🟡 MODERATE  🔴 LOW  |  Dashed lines: thresholds")

        st.markdown("---")

        # ── Recovery path & 3Di usage ─────────────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.subheader("🔄 Recovery Path Distribution")
            path_counts = df["path"].value_counts().reset_index()
            path_counts.columns = ["Path", "Count"]
            st.bar_chart(path_counts.set_index("Path"))

        with right:
            st.subheader("🧬 3Di Structural Token Usage")
            if "used_3di" in df.columns:
                di_counts = df["used_3di"].map({True: "3Di ✅ Cache Hit", False: "⚠️ Fallback (#)"}).value_counts()
                st.bar_chart(di_counts)
                st.caption(f"3Di hits: {n_3di} / Total inferred: {n_inferred} ({rate_3di:.0%})")

        st.markdown("---")

        # ── Per-query result table ────────────────────────────────────────────
        st.subheader("📋 Per-Query Results")

        def _badge(dec):
            return {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(dec, "⬜")

        cols = ["query_id", "drug_name", "protein_name", "pKd",
                "decision", "path", "corrupt", "latency_ms", "used_3di"]
        display = df[[c for c in cols if c in df.columns]].copy()
        display.columns = [{"query_id": "Query ID", "drug_name": "Drug",
                            "protein_name": "Target", "pKd": "pKd",
                            "decision": "Decision", "path": "Path",
                            "corrupt": "Corrupted", "latency_ms": "Latency (ms)",
                            "used_3di": "3Di"}.get(c, c)
                           for c in display.columns]
        if "Decision" in display.columns:
            display["Decision"] = display["Decision"].apply(lambda d: f"{_badge(d)} {d}")
        if "Corrupted" in display.columns:
            display["Corrupted"] = display["Corrupted"].apply(lambda x: "⚡" if x else "")
        if "3Di" in display.columns:
            display["3Di"] = display["3Di"].apply(lambda x: "✅" if x else "⚠️")
        st.dataframe(display, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2 — DAVIS Protein Sequence Browser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_seqs:
    st.subheader("🧬 DAVIS Representative Protein Sequences (3Di Cache Verified)")
    st.caption(
        "Full-length canonical sequences used in demo.py — "
        "same distribution as DAVIS training data → 3Di structural tokens fully utilized."
    )

    seqs_path = ROOT / "davis_seqs_for_demo.json"
    if not seqs_path.exists():
        st.warning("`davis_seqs_for_demo.json` not found. Run `python prepare_sequences.py` first.")
    else:
        with open(seqs_path, encoding="utf-8") as f:
            davis_data = json.load(f)

        cache_path   = ROOT / "cache" / "3di_tokens_davis.json"
        raw_cache    = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
        cache_hashes = {v["seq_hash"] for v in raw_cache.values() if v.get("status") == "ok"}

        rows = []
        for name, entry in davis_data.items():
            seq = entry["seq"] if isinstance(entry, dict) else entry
            h   = hashlib.md5(seq.encode()).hexdigest()
            hit = h in cache_hashes
            rows.append({
                "Protein":      name,
                "Length (aa)":  len(seq),
                "UniProt":      entry.get("uniprot", "-") if isinstance(entry, dict) else "-",
                "3Di Cache":    "✅ Hit" if hit else "⚠️ Miss",
                "MD5 (first16)": h[:16],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("---")

        selected = st.selectbox("View sequence", list(davis_data.keys()))
        if selected:
            entry = davis_data[selected]
            seq   = entry["seq"] if isinstance(entry, dict) else entry
            h     = hashlib.md5(seq.encode()).hexdigest()
            hit   = h in cache_hashes

            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("Length", f"{len(seq)} aa")
                st.metric("3Di Cache", "✅ Hit" if hit else "⚠️ Miss")
                if isinstance(entry, dict):
                    st.metric("UniProt", entry.get("uniprot", "-"))
            with col_b:
                wrapped = "\n".join(seq[i:i+60] for i in range(0, len(seq), 60))
                st.text_area(f"{selected} — Full Sequence", wrapped, height=200)

        st.markdown("---")
        st.caption(
            "**Why full-length sequences matter:**  \n"
            "SaProt-650M was validated on DAVIS canonical sequences (600–1600 aa). "
            "The 3Di token cache was also built from the same sequences.  \n"
            "The original demo used ~150 aa N-terminal fragments → no kinase domain → "
            "0% 3Di cache hit → ~5% performance loss (per SaProt paper)."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Source: {LOG_PATH.name}")

if auto_ref:
    time.sleep(2)
    st.rerun()
