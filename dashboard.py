"""
dashboard.py
============
Streamlit 실시간 대시보드 — Bio-AI DTI Pipeline 모니터링

실행:
  conda run -n bioinfo streamlit run dashboard.py
  (또는 streamlit이 설치된 환경에서)

pipeline.py 가 results/pipeline_log.jsonl 을 실시간으로 기록하며,
이 대시보드는 2초마다 자동 갱신됩니다.
"""

import json
import time
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

ROOT     = Path(__file__).parent
LOG_PATH = ROOT / "results" / "pipeline_log.jsonl"
SUM_PATH = ROOT / "results" / "pipeline_summary.json"

# ── 사이드바 컨트롤 ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
pkd_high = st.sidebar.slider("HIGH 임계값 (pKd ≥)", 5.0, 10.0, 7.0, 0.1)
pkd_mod  = st.sidebar.slider("MODERATE 임계값 (pKd ≥)", 3.0, 7.0, 5.0, 0.1)
auto_ref = st.sidebar.checkbox("자동 새로고침 (2초)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**모델:** SaProt-650M + ft-ChemBERTa  \n"
    "**학습:** BindingDB 80K (r=0.89)  \n"
    "**전이:** DAVIS r=0.87 / KIBA r=0.86"
)

# ── 메인 타이틀 ─────────────────────────────────────────────────────────────────
st.title("💊 Bio-AI DTI Query Pipeline")
st.caption("Drug-Target Interaction 실시간 예측 대시보드 | SaProt-650M + ft-ChemBERTa")

# ── 데이터 로드 ─────────────────────────────────────────────────────────────────
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

# ── 임계값 재적용 (슬라이더 연동) ───────────────────────────────────────────────
if not df.empty:
    def reclassify(pkd):
        if pkd >= pkd_high:  return "HIGH"
        if pkd >= pkd_mod:   return "MODERATE"
        return "LOW"
    df["decision"] = df["pKd"].apply(reclassify)

# ── 패킷 통계 카드 ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total    = summary.get("total_queries", 0)
received = summary.get("received",      len(df) if not df.empty else 0)
dropped  = summary.get("dropped",       0)
loss_pct = summary.get("loss_rate",     0.0) * 100
imputed  = summary.get("imputed",       0)
net_alert= summary.get("network_alert", False)

col1.metric("총 쿼리",      total or received)
col2.metric("수신 완료",    received)
col3.metric("드롭",         dropped, delta=f"-{loss_pct:.1f}%", delta_color="inverse")
col4.metric("대체 추론",    imputed)
col5.metric("패킷 손실률",  f"{loss_pct:.1f}%")

if net_alert or (total > 0 and loss_pct > 30):
    st.error("⚠️  ALERT: 패킷 손실률이 30%를 초과했습니다 — Network Degraded!")

st.markdown("---")

if df.empty:
    st.info("pipeline.py 를 실행하면 결과가 여기에 표시됩니다.\n\n"
            "`conda run -n bioinfo python pipeline.py`")
else:
    # ── 결합 결정 요약 ─────────────────────────────────────────────────────────
    n_high = (df["decision"] == "HIGH").sum()
    n_mod  = (df["decision"] == "MODERATE").sum()
    n_low  = (df["decision"] == "LOW").sum()
    avg_pkd = df["pKd"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 HIGH",     n_high)
    c2.metric("🟡 MODERATE", n_mod)
    c3.metric("🔴 LOW",      n_low)
    c4.metric("평균 pKd",    f"{avg_pkd:.4f}")

    st.markdown("---")

    # ── pKd 시계열 그래프 ──────────────────────────────────────────────────────
    st.subheader("📈 pKd 시계열")
    chart_df = df[["query_id","pKd","decision"]].copy()
    chart_df.index = range(len(chart_df))

    # 결정별 색상 (Streamlit line_chart는 단색이므로 threshold 라인 추가)
    st.line_chart(chart_df.set_index("query_id")["pKd"])

    # threshold 기준선 표시용 테이블
    st.caption(f"기준선 — HIGH: pKd ≥ {pkd_high}  |  MODERATE: pKd ≥ {pkd_mod}  |  LOW: pKd < {pkd_mod}")

    st.markdown("---")

    # ── 결과 테이블 ────────────────────────────────────────────────────────────
    st.subheader("📋 쿼리별 결과")

    def _badge(dec):
        return {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(dec, "⬜")

    display = df[["query_id","drug_name","protein_name","pKd","decision","path","corrupt","latency_ms","used_3di"]].copy()
    display.columns = ["쿼리ID","약물","표적단백질","pKd","결합판정","추론경로","변조","지연(ms)","3Di"]
    display["결합판정"] = display["결합판정"].apply(lambda d: f"{_badge(d)} {d}")
    display["변조"]     = display["변조"].apply(lambda x: "⚡" if x else "")
    display["3Di"]      = display["3Di"].apply(lambda x: "✅" if x else "⚠️")
    st.dataframe(display, use_container_width=True)

    # ── 경로 분포 ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 추론 경로 분포")
    path_counts = df["path"].value_counts().reset_index()
    path_counts.columns = ["경로", "건수"]
    st.bar_chart(path_counts.set_index("경로"))

# ── 마지막 갱신 시각 ───────────────────────────────────────────────────────────
st.caption(f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── 자동 새로고침 ─────────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(2)
    st.rerun()
