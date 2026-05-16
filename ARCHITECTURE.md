# System Architecture — Bio-AI DTI Query Pipeline

## 6-Step Pipeline Overview

```mermaid
flowchart TD
    subgraph GEN["① Data Generation — Edge Node"]
        A1["DTI Query Builder\nDrug SMILES + Protein AA Seq"]
        A2["10 Anti-cancer drugs\nABL1 / EGFR / BRAF / BTK\nPDGFRA / ALK targets"]
        A1 --> A2
    end

    subgraph TX["② Transmission — WAN Simulation"]
        B1["Latency Injection\nsleep(0.6 ~ 1.4 s)\nrandom.uniform()"]
        B2["Packet Drop\n30% loss probability\nrandom.random() < drop_rate"]
        B3["Payload Corruption\n15% SMILES char substitution\nnoise_sigma = 0.05"]
        B1 --> B2 --> B3
    end

    subgraph COL["③ Collection — Server Node"]
        C1["Queue Receiver\nmp.Queue IPC\nsentinel = None"]
        C2["Rolling Buffer\nlast 5 pKd values\nfor imputation"]
        C1 --> C2
    end

    subgraph AI["④ AI Inference & Recovery"]
        D1{"Packet\nStatus?"}
        D2["Normal Inference\nSaProt-650M FP16\n+ ft-ChemBERTa\n→ pKd prediction"]
        D3["Corrupt Recovery\ncorrupt_recovered\nnoisy SMILES → infer\nmodel robustness"]
        D4["Drop Imputation\ndrop_imputed\nrolling mean\nlast 5 values"]
        D1 -->|"normal"| D2
        D1 -->|"corrupt"| D3
        D1 -->|"dropped"| D4
    end

    subgraph DEC["⑤ Decision — Binding Affinity"]
        E1{"pKd\nThreshold"}
        E2["HIGH\npKd ≥ 7.0\nPromising Candidate"]
        E3["MODERATE\n5.0 ≤ pKd < 7.0\nFurther Review"]
        E4["LOW\npKd < 5.0\nWeak Binding"]
        E5["ALERT\nLoss Rate > 30%\nNetwork Degraded"]
        E1 -->|"pKd ≥ 7.0"| E2
        E1 -->|"5.0 ≤ pKd < 7.0"| E3
        E1 -->|"pKd < 5.0"| E4
    end

    subgraph DASH["⑥ Dashboard — Streamlit"]
        F1["Real-time Charts\npKd time-series\nAuto-refresh 2s"]
        F2["JSONL Log\ndemo_log.jsonl"]
        F3["Summary JSON\nLoss rate / 3Di hit rate\nPer-drug avg pKd"]
        F1 --- F2 --- F3
    end

    GEN -->|"generate queries"| TX
    TX -->|"send / drop / corrupt"| COL
    COL -->|"buffer"| AI
    AI -->|"pKd value"| DEC
    DEC -->|"log results"| DASH
    DEC -->|"monitor loss rate"| E5

    style GEN  fill:#FFF0F3,color:#2D0A0A,stroke:#DC143C,stroke-width:2px
    style TX   fill:#FFF0F3,color:#2D0A0A,stroke:#DC143C,stroke-width:2px
    style COL  fill:#FFF0F3,color:#2D0A0A,stroke:#DC143C,stroke-width:2px
    style AI   fill:#FFF0F3,color:#2D0A0A,stroke:#DC143C,stroke-width:2px
    style DEC  fill:#FFF0F3,color:#2D0A0A,stroke:#DC143C,stroke-width:2px
    style DASH fill:#FFF0F3,color:#2D0A0A,stroke:#DC143C,stroke-width:2px

    style A1 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style A2 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style B1 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style B2 fill:#DC143C,color:#FFFFFF,stroke:#8B0000
    style B3 fill:#DC143C,color:#FFFFFF,stroke:#8B0000
    style C1 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style C2 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style D1 fill:#8B0000,color:#FFFFFF,stroke:#5C0000
    style D2 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style D3 fill:#DC143C,color:#FFFFFF,stroke:#8B0000
    style D4 fill:#DC143C,color:#FFFFFF,stroke:#8B0000
    style E1 fill:#8B0000,color:#FFFFFF,stroke:#5C0000
    style E2 fill:#5C0000,color:#FFFFFF,stroke:#2D0000
    style E3 fill:#DC143C,color:#FFFFFF,stroke:#8B0000
    style E4 fill:#F4A0A0,color:#2D0A0A,stroke:#DC143C
    style E5 fill:#2D0A0A,color:#FFFFFF,stroke:#DC143C
    style F1 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style F2 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
    style F3 fill:#FFFFFF,color:#2D0A0A,stroke:#DC143C
```

---

## Intentional Network Constraints

| Constraint | Implementation | Parameter |
|------------|---------------|-----------|
| **Latency** | `time.sleep(random.uniform(lat_min, lat_max))` | 0.6 ~ 1.4 s |
| **Packet Drop** | `if random.random() < drop_rate: continue` | 30% (demo) |
| **Payload Corruption** | Random SMILES char substitution (C↔N↔O↔S) | 15% / σ=0.05 |

## Recovery Strategy

| Situation | Recovery Method | Log Path |
|-----------|----------------|----------|
| Normal packet | SaProt-650M + ChemBERTa direct inference | `normal` |
| Corrupted payload | Infer with noisy SMILES — model robustness | `corrupt_recovered` |
| Dropped packet | Rolling mean of last 5 pKd values | `drop_imputed` |
| Inference failure | Rolling mean fallback | `imputed` |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| Protein Encoder | SaProt-650M-AF2 (FP16, frozen) + FoldSeek 3Di tokens |
| Drug Encoder | ft-ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, layers 4~5) |
| Regression Head | MLP [1280+768 → 512 → 256 → 64 → 1] |
| Pipeline IPC | `multiprocessing.Queue` + `mp.Event` |
| Dashboard | Streamlit (auto-refresh 2s) |
| Training Data | BindingDB 80K → DAVIS 30K → KIBA 118K |
| Environment | conda `bioinfo` |

## Experiment Results (Final)

| Metric | Value |
|--------|-------|
| Zero Silent Drop | **50 / 50 (100%)** |
| Corrupt Recovery Accuracy | **3 / 3 (100%)** |
| Drop Recovery Accuracy | **9 / 17 (52.9%)** |
| 3Di Token Hit Rate | **33 / 33 (100%)** |
| Network Alert | **Triggered (34% > 30%)** |
| AI Inference Failure | **0 / 33 (0%)** |
| Avg pKd | **6.5469** |
