# System Architecture — Bio-AI DTI Query Pipeline

## 6-Step Pipeline Overview

```mermaid
flowchart TD
    subgraph GEN["① Data Generation — Edge Node"]
        A1["📋 DTI Query 생성\n(Drug SMILES + Protein AA Seq)"]
        A2["약물: Imatinib / Gefitinib\n / Dasatinib 등 10종\n표적: ABL1 / EGFR / BRAF 등"]
        A1 --> A2
    end

    subgraph TX["② Transmission — WAN Simulation"]
        B1["⏱ 지연 주입\nsleep(0.4 ~ 1.2 s)\nrandom.uniform()"]
        B2["▼ 패킷 드롭\n20% 확률 손실\nrandom.random() < drop_rate"]
        B3["⚡ 페이로드 변조\n15% 확률 SMILES 문자 치환\nnoise_sigma = 0.05"]
        B1 --> B2 --> B3
    end

    subgraph COL["③ Collection — Server Node"]
        C1["📥 Queue 수신\nmp.Queue (IPC)\nsentinel = None 로 종료"]
        C2["🔄 Rolling Buffer\n최근 5개 pKd 유지\n(imputation용)"]
        C1 --> C2
    end

    subgraph AI["④ AI / Recovery"]
        D1{"패킷 상태?"}
        D2["정상 추론\nSaProt-650M FP16\n+ ft-ChemBERTa\n→ pKd 예측"]
        D3["변조 복구\ncorrupt_recovered:\n노이즈 SMILES로 추론\n(모델 자체 복원력 활용)"]
        D4["드롭 대체\ndrop_imputed:\nrolling mean(최근 5개)"]
        D1 -->|"normal"| D2
        D1 -->|"corrupt"| D3
        D1 -->|"dropped"| D4
    end

    subgraph DEC["⑤ Decision — 결합 친화력 판정"]
        E1{"pKd 임계값 비교"}
        E2["🟢 HIGH\npKd ≥ 7.0\n강한 결합 후보"]
        E3["🟡 MODERATE\n5.0 ≤ pKd < 7.0\n추가 검토 필요"]
        E4["🔴 LOW\npKd < 5.0\n결합력 부족"]
        E5["⚠️ ALERT\n손실률 > 30%\nNetwork Degraded"]
        E1 -->|"pKd ≥ 7.0"| E2
        E1 -->|"5.0 ≤ pKd < 7.0"| E3
        E1 -->|"pKd < 5.0"| E4
    end

    subgraph DASH["⑥ Dashboard — Streamlit"]
        F1["📊 실시간 차트\npKd 분포 / 판정 현황\n2초 자동 갱신"]
        F2["📁 JSONL 로그\npipeline_log.jsonl\ndemo_log.jsonl"]
        F3["📋 Summary JSON\n총 쿼리 / 손실률\n약물별 평균 pKd"]
        F1 --- F2 --- F3
    end

    GEN -->|"쿼리 생성"| TX
    TX -->|"전송 / 손실 / 변조"| COL
    COL -->|"수신 버퍼"| AI
    AI -->|"pKd 값"| DEC
    DEC -->|"결과 기록"| DASH
    DEC -->|"손실률 감시"| E5

    style GEN fill:#1a1a2e,color:#e0e0e0,stroke:#4a90d9
    style TX  fill:#16213e,color:#e0e0e0,stroke:#e94560
    style COL fill:#0f3460,color:#e0e0e0,stroke:#4a90d9
    style AI  fill:#1a1a2e,color:#e0e0e0,stroke:#53d8fb
    style DEC fill:#16213e,color:#e0e0e0,stroke:#f5a623
    style DASH fill:#0f3460,color:#e0e0e0,stroke:#7ed321
    style E2  fill:#1a5c2a,color:#ffffff
    style E3  fill:#5c4a00,color:#ffffff
    style E4  fill:#5c1a1a,color:#ffffff
    style E5  fill:#5c1a1a,color:#ffffff,stroke:#ff4444
```

---

## Intentional Network Constraints

| 제약 유형 | 구현 방식 | 파라미터 |
|-----------|-----------|----------|
| **전송 지연** | `time.sleep(random.uniform(lat_min, lat_max))` | 0.4 ~ 1.2 s |
| **패킷 드롭** | `if random.random() < drop_rate: continue` | 20% 기본값 |
| **페이로드 변조** | SMILES 문자 무작위 치환 (C↔N↔O↔S) | 15% / σ=0.05 |

## Recovery Strategy

| 상황 | 복구 방법 |
|------|-----------|
| 정상 패킷 | SaProt-650M + ChemBERTa 직접 추론 |
| 변조 패킷 | 노이즈 SMILES 그대로 추론 (`corrupt_recovered`) |
| 드롭 패킷 | Rolling mean of last 5 pKd values (`drop_imputed`) |
| 추론 실패 | Rolling mean fallback (`imputed`) |

## Tech Stack

| 레이어 | 기술 |
|--------|------|
| 언어 | Python 3.10 |
| AI 모델 | SaProt-650M (FP16), ft-ChemBERTa |
| 파이프라인 | `multiprocessing.Queue` (Edge ↔ Server IPC) |
| 대시보드 | Streamlit |
| 데이터 | BindingDB 80K (pre-train), DAVIS / KIBA (fine-tune) |
| 환경 | conda `bioinfo` |
