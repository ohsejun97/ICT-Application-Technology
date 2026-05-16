"""
dashboard.py
============
Streamlit 실시간 대시보드 — Bio-AI DTI Pipeline 모니터링

실행:
  conda run -n bioinfo streamlit run dashboard.py
"""

import json
import time
import hashlib
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

# ── 사이드바 ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")

log_mode = st.sidebar.radio(
    "로그 소스",
    ["demo (demo_log.jsonl)", "pipeline (pipeline_log.jsonl)"],
    index=0,
)
LOG_STEM = "demo_log" if log_mode.startswith("demo") else "pipeline_log"
LOG_PATH = ROOT / "results" / f"{LOG_STEM}.jsonl"
SUM_PATH = ROOT / "results" / (LOG_STEM.replace("log", "summary") + ".json")

pkd_high = st.sidebar.slider("HIGH 임계값 (pKd ≥)", 5.0, 10.0, 7.0, 0.1)
pkd_mod  = st.sidebar.slider("MODERATE 임계값 (pKd ≥)", 3.0, 7.0, 5.0, 0.1)
auto_ref = st.sidebar.checkbox("자동 새로고침 (2초)", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🗑 로그 초기화 (녹화 전 클릭)", use_container_width=True):
    for p in [LOG_PATH, SUM_PATH]:
        if p.exists():
            p.unlink()
    st.sidebar.success("초기화 완료 — demo.py를 실행하세요")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**모델:** SaProt-650M + ft-ChemBERTa  \n"
    "**학습:** BindingDB 80K (r=0.89)  \n"
    "**전이:** DAVIS r=0.87 / KIBA r=0.86"
)

# ── 메인 타이틀 ───────────────────────────────────────────────────────────────
st.title("💊 Bio-AI DTI Query Pipeline")
st.caption("Drug-Target Interaction 실시간 예측 대시보드 | SaProt-650M + ft-ChemBERTa")

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 구성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_live, tab_seqs = st.tabs(["📡 실시간 모니터링", "🧬 DAVIS 단백질 서열"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1 — 실시간 모니터링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_live:

    # ── 네트워크 통계 카드 ────────────────────────────────────────────────────
    total    = summary.get("total_queries",  summary.get("network", {}).get("total_queries", 0))
    received = summary.get("received",       summary.get("network", {}).get("received", len(df) if not df.empty else 0))
    dropped  = summary.get("dropped",        summary.get("network", {}).get("dropped", 0))
    loss_pct = summary.get("loss_rate",      summary.get("network", {}).get("loss_rate", 0.0)) * 100
    imputed  = summary.get("imputed",        summary.get("network", {}).get("inference_imputed", 0))
    net_alert= summary.get("network_alert",  summary.get("network", {}).get("network_alert", False))

    # 3Di 히트율
    mq = summary.get("model_quality", {})
    n_3di     = mq.get("used_3di",      int(df["used_3di"].sum()) if not df.empty and "used_3di" in df.columns else 0)
    n_inferred= mq.get("total_inferred", received)
    rate_3di  = mq.get("3di_rate",      n_3di / n_inferred if n_inferred else 0)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("총 쿼리",       total or received)
    col2.metric("수신 완료",     received)
    col3.metric("드롭",          dropped, delta=f"-{loss_pct:.1f}%", delta_color="inverse")
    col4.metric("대체 추론",     imputed)
    col5.metric("패킷 손실률",   f"{loss_pct:.1f}%")
    col6.metric("3Di 히트율",    f"{rate_3di:.0%}",
                delta="✅ 구조정보 활용" if rate_3di > 0.5 else "⚠️ fallback",
                delta_color="normal" if rate_3di > 0.5 else "inverse")

    if net_alert or (total > 0 and loss_pct > 30):
        st.error("⚠️  ALERT: 패킷 손실률이 30%를 초과했습니다 — Network Degraded!")

    st.markdown("---")

    if df.empty:
        st.info(
            f"`{LOG_PATH.name}` 이 없습니다. 파이프라인을 먼저 실행하세요.\n\n"
            "```bash\nconda run -n bioinfo python demo.py\n```"
        )
    else:
        # ── 결합 결정 요약 ────────────────────────────────────────────────────
        n_high  = (df["decision"] == "HIGH").sum()
        n_mod   = (df["decision"] == "MODERATE").sum()
        n_low   = (df["decision"] == "LOW").sum()
        avg_pkd = df["pKd"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 HIGH",     n_high)
        c2.metric("🟡 MODERATE", n_mod)
        c3.metric("🔴 LOW",      n_low)
        c4.metric("평균 pKd",    f"{avg_pkd:.4f}")

        st.markdown("---")

        # ── pKd 시계열 그래프 ─────────────────────────────────────────────────
        st.subheader("📈 pKd 시계열")
        chart_df = df[["query_id", "pKd"]].copy()
        chart_df.index = range(len(chart_df))
        st.line_chart(chart_df.set_index("query_id")["pKd"])
        st.caption(f"기준선 — HIGH: pKd ≥ {pkd_high}  |  MODERATE: pKd ≥ {pkd_mod}  |  LOW: pKd < {pkd_mod}")

        st.markdown("---")

        # ── 추론 경로 분포 ────────────────────────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.subheader("🔄 추론 경로 분포")
            path_counts = df["path"].value_counts().reset_index()
            path_counts.columns = ["경로", "건수"]
            st.bar_chart(path_counts.set_index("경로"))

        with right:
            st.subheader("🧬 3Di 토큰 사용 현황")
            if "used_3di" in df.columns:
                di_counts = df["used_3di"].map({True: "3Di ✅ 캐시 히트", False: "⚠️ fallback (#)"}).value_counts()
                st.bar_chart(di_counts)
                st.caption(f"3Di 히트: {n_3di}건 / 전체 추론: {n_inferred}건 ({rate_3di:.0%})")

        st.markdown("---")

        # ── 쿼리별 결과 테이블 ────────────────────────────────────────────────
        st.subheader("📋 쿼리별 결과")

        def _badge(dec):
            return {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(dec, "⬜")

        cols = ["query_id", "drug_name", "protein_name", "pKd",
                "decision", "path", "corrupt", "latency_ms", "used_3di"]
        display = df[[c for c in cols if c in df.columns]].copy()
        display.columns = [{"query_id":"쿼리ID","drug_name":"약물",
                            "protein_name":"표적단백질","pKd":"pKd",
                            "decision":"결합판정","path":"추론경로",
                            "corrupt":"변조","latency_ms":"지연(ms)",
                            "used_3di":"3Di"}.get(c, c)
                           for c in display.columns]
        if "결합판정" in display.columns:
            display["결합판정"] = display["결합판정"].apply(lambda d: f"{_badge(d)} {d}")
        if "변조" in display.columns:
            display["변조"] = display["변조"].apply(lambda x: "⚡" if x else "")
        if "3Di" in display.columns:
            display["3Di"] = display["3Di"].apply(lambda x: "✅" if x else "⚠️")
        st.dataframe(display, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2 — DAVIS 단백질 서열 브라우저
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_seqs:
    st.subheader("🧬 DAVIS 대표 단백질 서열 (3Di 캐시 히트)")
    st.caption("demo.py 에서 실제로 사용하는 full-length 서열. "
               "학습 데이터(DAVIS)와 동일 분포 → 3Di 구조 토큰 정상 활용.")

    seqs_path = ROOT / "davis_seqs_for_demo.json"
    if not seqs_path.exists():
        st.warning("`davis_seqs_for_demo.json` 없음. `python extract_davis_seqs.py` 를 먼저 실행하세요.")
    else:
        with open(seqs_path, encoding="utf-8") as f:
            davis_data = json.load(f)

        # 3Di 캐시에서 히트 여부 재확인
        cache_path = ROOT / "cache" / "3di_tokens_davis.json"
        raw_cache  = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
        cache_hashes = {v["seq_hash"] for v in raw_cache.values() if v.get("status") == "ok"}

        rows = []
        for name, entry in davis_data.items():
            seq  = entry["seq"] if isinstance(entry, dict) else entry
            h    = hashlib.md5(seq.encode()).hexdigest()
            hit  = h in cache_hashes
            rows.append({
                "단백질":    name,
                "길이(aa)":  len(seq),
                "UniProt":   entry.get("uniprot", "-") if isinstance(entry, dict) else "-",
                "3Di 캐시":  "✅ 히트" if hit else "⚠️ 미스",
                "MD5 (앞16)": h[:16],
            })

        meta_df = pd.DataFrame(rows)
        st.dataframe(meta_df, use_container_width=True)

        st.markdown("---")

        # 서열 뷰어
        selected = st.selectbox("서열 보기", list(davis_data.keys()))
        if selected:
            entry = davis_data[selected]
            seq   = entry["seq"] if isinstance(entry, dict) else entry
            h     = hashlib.md5(seq.encode()).hexdigest()
            hit   = h in cache_hashes

            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("길이", f"{len(seq)} aa")
                st.metric("3Di 캐시", "✅ 히트" if hit else "⚠️ 미스")
                if isinstance(entry, dict):
                    st.metric("UniProt", entry.get("uniprot", "-"))
            with col_b:
                # 60aa 줄바꿈
                wrapped = "\n".join(seq[i:i+60] for i in range(0, len(seq), 60))
                st.text_area(f"{selected} 전체 서열", wrapped, height=200)

        st.markdown("---")
        st.caption(
            "**왜 full-length 서열이 필요한가?**  \n"
            "SaProt-650M은 DAVIS canonical 서열(600~1600aa)로 검증되었으며, "
            "3Di 토큰 캐시도 동일 서열 기준으로 구축됨.  \n"
            "이전 demo는 ~150aa 단편(N-말단)을 사용 → 키나아제 도메인 없음 → 3Di 캐시 0% 히트 → 성능 약 5% 손실."
        )

# ── 마지막 갱신 ───────────────────────────────────────────────────────────────
st.caption(f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  소스: {LOG_PATH.name}")

if auto_ref:
    time.sleep(2)
    st.rerun()
