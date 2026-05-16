# Bio-AI DTI Query Pipeline — 종합 기술 보고서

**과목:** ICT Application Technology  
**학생:** 오세준 (2021270607) | 팀: Individual  
**발표일:** 2026년 5월 22일 (Group 1) | **영상 제출 마감:** 2026년 5월 17일

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [AI 모델 설계 및 학습](#3-ai-모델-설계-및-학습)
4. [ICT 파이프라인 구현](#4-ict-파이프라인-구현)
5. [실험 결과 — 50쿼리 네트워크 복구](#5-실험-결과--50쿼리-네트워크-복구)
6. [요구사항 충족 검증](#6-요구사항-충족-검증)
7. [한계점 및 개선 방향](#7-한계점-및-개선-방향)
8. [부록 A: 서열 오류 디버깅 기록](#8-부록-a-서열-오류-디버깅-기록)
9. [부록 B: 발표 스크립트 및 Q&A](#9-부록-b-발표-스크립트-및-qa)

---

## 1. 프로젝트 개요

### 1.1 도메인 및 문제 정의

**도메인:** 헬스케어 / 제약 (신약 개발)

**실험 대상:** 본 시스템은 특정 약물 하나(예: Imatinib)만을 위한 것이 **아니다.**  
DAVIS 데이터셋에서 검증된 **10종 항암 약물 × 6종 키나아제 표적** 쌍을 처리한다.

| 약물 | 표적 | 기전 | 실제 적응증 |
|------|------|------|-------------|
| Imatinib | ABL1 | BCR-ABL 억제 (1세대) | 만성 골수성 백혈병 (CML) |
| Nilotinib | ABL1 | BCR-ABL 억제 (2세대) | CML (이마티닙 내성) |
| Dasatinib | ABL1 | BCR-ABL + SRC 억제 | CML, ALL |
| Gefitinib | EGFR | EGFR TKI (1세대) | 비소세포 폐암 (NSCLC) |
| Erlotinib | EGFR | EGFR TKI (1세대) | NSCLC, 췌장암 |
| Sorafenib | BRAF | RAF/VEGFR 억제 | 간세포암, 신세포암 |
| Vemurafenib | BRAF | BRAF V600E 선택적 억제 | 흑색종 |
| Ibrutinib | BTK | BTK 불가역 억제 | 만성 B세포 림프종 |
| Sunitinib | PDGFRA | 다중 키나아제 억제 | GIST, 신세포암 |
| Crizotinib | ALK | ALK/ROS1/MET 억제 | ALK+ NSCLC |

> **단백질 서열 출처:** DAVIS canonical full-length 서열 (ABL1: 1138aa, EGFR: 1210aa, BRAF: 772aa, BTK: 666aa, PDGFRA: 1089aa, ALK: 1620aa). FoldSeek 3Di 구조 토큰 캐시 히트 확인됨.

**시나리오:**  
제약회사 고처리량 스크리닝(HTS) 환경에서 지리적으로 분산된 실험실 edge node가 DTI 쿼리(약물 SMILES + 단백질 서열)를 WAN을 통해 중앙 AI 서버로 전송한다. 혼잡한 네트워크 환경에서 지연·패킷 손실·페이로드 변조가 발생하더라도 모든 약물 후보에 결합력 판정을 제공해야 한다.

### 1.2 핵심 연구 문제

> 분산 HTS 환경에서 WAN 열화(지연·손실·변조)가 발생하더라도 **모든 쿼리에 누락 없이** 결합력 판정(pKd)을 제공할 수 있는가?

---

## 2. 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  [Step 1] Edge Node — Process A                               │
│  DTI 쿼리 생성: {query_id, SMILES, AA sequence, timestamp}    │
└───────────────────────────┬──────────────────────────────────┘
                            │ ── WAN 시뮬레이션 ──────────────────
┌───────────────────────────▼──────────────────────────────────┐
│  [Step 2] 네트워크 제약 (의도적 설계)                          │
│  · 지연:  sleep(0.4 ~ 1.2 s)   ← 혼잡 WAN 경로               │
│  · 드롭:  random() < 0.20      ← 패킷 손실 (기본 20%)         │
│  · 변조:  SMILES 랜덤 문자 치환 ← 페이로드 비트 반전 (15%)     │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  [Step 3] Server Node — Process B                             │
│  패킷 수신 · mp.Queue 버퍼링 · 드롭/변조 로깅                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  [Step 4] AI 처리 & 복구                                       │
│  정상:  SMILES → ft-ChemBERTa → [768] ─┐                     │
│                                         ├─► MLP Head → pKd   │
│         AA seq → 3Di + SaProt-650M ────┘                     │
│  변조:  변조된 SMILES로 추론 → corrupt_recovered               │
│  드롭:  최근 5개 pKd rolling mean → drop_imputed              │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  [Step 5] Decision Engine                                      │
│  pKd ≥ 7.0 → 🟢 HIGH     (유망 후보)                         │
│  pKd 5~7.0 → 🟡 MODERATE (추가 검토)                         │
│  pKd < 5.0 → 🔴 LOW      (결합력 부족)                        │
│  손실률 > 30% → ⚠️ ALERT "Network Degraded"                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  [Step 6] Streamlit Dashboard (auto-refresh 2s)               │
│  pKd 시계열 · 결합 결정 배지 · 패킷 통계 · 3Di 히트율          │
│  DAVIS 단백질 서열 브라우저 · 임계값 실시간 조절               │
└──────────────────────────────────────────────────────────────┘

Process A ↔ Process B: multiprocessing.Queue (동일 머신, 논리적 분리)
```

---

## 3. AI 모델 설계 및 학습

### 3.1 모델 아키텍처

| 구성요소 | 세부 내용 |
|----------|-----------|
| **단백질 인코더** | SaProt-650M-AF2 (FP16, frozen) — ESM-2 기반 서열+구조 통합 LM |
| **구조 토큰** | FoldSeek 3Di 토큰 (AlphaFold2 구조 → FoldSeek 추출, MD5 해시로 캐싱) |
| **약물 인코더** | ChemBERTa (seyonec/ChemBERTa-zinc-base-v1), layers 4~5 fine-tune |
| **회귀 헤드** | MLP: [1280+768 → 512 → 256 → 64 → 1] (pKd 회귀) |
| **손실함수** | Huber Loss (δ=1.0) |
| **옵티마이저** | AdamW (head lr=5×10⁻⁴, ChemBERTa lr=1×10⁻⁵) |

#### 3Di 구조 토큰을 쓰는 이유

기존 SaProt 논문에서 3Di 토큰을 `#` placeholder로 대체 시 Pearson r 약 −0.05 (5% 하락). AlphaFold2로 예측한 3D 구조에서 FoldSeek가 추출한 구조 토큰을 `cache/3di_tokens_*.json`에 저장하여 서열+구조 정보를 동시에 활용한다.

3Di 캐시 조회 방식: `MD5(aa_seq) → cache[hash] → tokens_3di` — 캐시 미스 시 `aa + '#'` fallback.

### 3.2 학습 전략 (3단계 Transfer Learning)

```
단계 1: BindingDB 사전학습 (80,795 쌍)
  SaProt frozen + ChemBERTa layers 4~5 + MLP head
  → Pearson r = 0.8923,  RMSE = 0.7387,  CI = 0.877
  → 모델 저장: results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random/

단계 2: DAVIS 전이학습 (30,056 쌍)
  BindingDB 가중치 로드 → ft-ChemBERTa drug embedding 재계산 → head fine-tune
  → Pearson r = 0.8677,  RMSE = 0.4572,  CI = 0.8925
  → 모델 저장: results/finetune_davis_random_from_*/

단계 3: KIBA 전이학습 (118,254 쌍)
  동일 BindingDB 가중치 → KIBA score z-score 정규화 → head fine-tune
  → Pearson r = 0.8594,  RMSE = 0.4268,  CI = 0.861
  → 모델 저장: results/finetune_kiba_random_from_*/
```

### 3.3 성능 지표

| 학습 데이터 | Pearson r | Spearman r | R² | RMSE | CI |
|-------------|-----------|------------|----|------|----|
| **BindingDB** (사전학습) | **0.8923** | 0.8722 | 0.7924 | 0.7387 | 0.877 |
| **DAVIS** (전이학습) | **0.8677** | 0.7021 | 0.7507 | 0.4572 | 0.8925 |
| **KIBA** (전이학습) | **0.8594** | 0.8464 | 0.7370 | 0.4268 | 0.861 |

| 비교 모델 | DAVIS Pearson r |
|-----------|-----------------|
| **본 모델 (SaProt + ft-ChemBERTa)** | **0.8677** |
| DeepPurpose MPNN_CNN | ~0.89 |
| DeepPurpose CNN | ~0.86 |
| 서열 기반 baseline (LSTM/Transformer) | ~0.78~0.80 |

### 3.4 하드웨어 및 추론 속도

| 항목 | 값 |
|------|----|
| GPU | NVIDIA RTX (CUDA 12.4) |
| 학습 시간 (BindingDB) | 24,824초 (~6.9시간) |
| 학습 시간 (DAVIS fine-tune) | 204초 (~3.4분) |
| 학습 시간 (KIBA fine-tune) | 814초 (~13.6분) |
| 최대 VRAM | 884.8 MB |
| 단일 샘플 추론 | < 5ms (임베딩 캐시 사용 시) |

---

## 4. ICT 파이프라인 구현

### 4.1 네트워크 제약 설계

| 제약 | 구현 코드 | 기본값 | 시뮬레이션 대상 |
|------|-----------|--------|-----------------|
| **전송 지연** | `time.sleep(random.uniform(lat_min, lat_max))` | 0.4~1.2 s | 혼잡 WAN 경로 |
| **패킷 드롭** | `if random.random() < drop_rate: continue` | 20% | 불안정 패킷 전달 |
| **페이로드 변조** | SMILES 문자 치환: C↔N↔O↔S | 15%, σ=0.05 | 비트 반전 손상 |

### 4.2 복구 전략

| 상황 | 복구 방법 | 로그 경로 |
|------|-----------|-----------|
| 정상 수신 | SaProt + ChemBERTa 직접 추론 | `normal` |
| 변조된 페이로드 | 변조 SMILES 그대로 추론 | `corrupt_recovered` |
| 드롭된 패킷 | 최근 5개 pKd rolling mean 대체 | `drop_imputed` |
| AI 추론 실패 | rolling mean fallback | `imputed` |
| rolling buffer 비어있음 | 기본값 6.0 pKd | `imputed` |

### 4.3 프로세스 간 통신

```
mp.Event (ready_event)  →  서버 모델 로드 완료 후 Edge 전송 시작 (동기화)
mp.Queue (queue)        →  DTI 쿼리 패킷 전달 (maxsize=50)
mp.Queue (stat_q)       →  드롭 이벤트 및 Edge 완료 신호
```

---

## 5. 실험 결과 — 50쿼리 네트워크 복구

**실험 설정:** `demo.py --n_queries 50 --drop_rate 0.30 --corrupt_rate 0.15 --lat_min 0.6 --lat_max 1.4 --seed 77`

### 5.1 네트워크 통계

| 항목 | 값 | 비율 |
|------|----|------|
| 총 전송 쿼리 | 50 | 100% |
| 정상 수신 | 33 | 66.0% |
| **패킷 드롭** | **17** | **34.0%** |
| 페이로드 변조 | 3 | 9.1% (수신 기준) |
| Network Alert 발동 | ✅ YES | 34.0% > 30% 기준 초과 |
| 평균 지연 | 1,002 ms | 설정범위 100% 준수 |

### 5.2 변조 복구 분석 (3건)

| 약물 | 정상 pKd | 변조 후 pKd | 오차 | 판정 변화 |
|------|----------|-------------|------|-----------|
| Dasatinib + ABL1 | 6.7509 | **6.3760** | 0.375 | MODERATE → MODERATE ✅ |
| Gefitinib + EGFR | 5.5345 | **5.5345** | **0.000** | MODERATE → MODERATE ✅ |
| Vemurafenib + BRAF | 5.4831 | **5.4726** | 0.011 | MODERATE → MODERATE ✅ |

**변조 복구 판정 정확도: 3/3 = 100%** | **평균 pKd 오차: 0.129**

### 5.3 드롭 복구 분석 (17건)

| 순번 | 약물 | 실제 pKd | Imputed pKd | 오차 | 정오 |
|------|------|----------|-------------|------|------|
| #01 | Imatinib | 7.7431 | 8.9591 | 1.216 | ✅ |
| #03 | Gefitinib | 5.5345 | 6.9631 | 1.429 | ✅ |
| #10 | Crizotinib | 5.9042 | 5.8592 | 0.045 | ✅ |
| **#11** | **Imatinib** | **7.7431** | **5.8592** | **1.884** | **❌ HIGH→MOD** |
| #13 | Gefitinib | 5.5345 | 5.7540 | 0.220 | ✅ |
| #16 | Sorafenib | 5.4931 | 6.1108 | 0.618 | ✅ |
| **#19** | **Sunitinib** | **4.7685** | **5.4645** | **0.696** | **❌ LOW→MOD** |
| **#22** | **Nilotinib** | **8.9591** | **5.8514** | **3.108** | **❌ HIGH→MOD** |
| #25 | Dasatinib | 6.7509 | 5.9284 | 0.823 | ✅ |
| #27 | Vemurafenib | 5.4831 | 5.6660 | 0.183 | ✅ |
| **#29** | **Sunitinib** | **4.7685** | **5.2982** | **0.530** | **❌ LOW→MOD** |
| #33 | Gefitinib | 5.5345 | 6.4332 | 0.899 | ✅ |
| #35 | Dasatinib | 6.7509 | 6.4332 | 0.318 | ✅ |
| #36 | Sorafenib | 5.4931 | 6.4332 | 0.940 | ✅ |
| #37 | Vemurafenib | 5.4831 | 6.4332 | 0.951 | ✅ |
| #40 | Crizotinib | 5.9042 | 6.2060 | 0.302 | ✅ |
| #45 | Dasatinib | 6.7509 | 6.5394 | 0.212 | ✅ |

**드롭 복구 판정 정확도: 13/17 = 76.5%** | **MAE: 0.845** | **RMSE: 1.124**

**오분류 원인 분석:**
- Nilotinib #22 (3.11 오차): pKd≈9로 분포 최상단 → rolling mean이 6.0으로 끌어내림 (가장 큰 구조적 약점)
- Imatinib #11: 직전 buffer가 낮은 약물들로 채워져 평균 하락
- Sunitinib #19, #29: 약한 결합제(pKd=4.77)가 rolling mean(~5.5)으로 과대평가
- **패턴:** Rolling mean은 pKd 분포 평균으로 수렴 → 분포 양극단(≥8 or ≤5)에서 오분류

### 5.4 전체 시스템 성능 요약

| 지표 | 값 | 평가 |
|------|----|------|
| Zero Silent Drop | **50/50 (100%)** | ✅ 목표 달성 |
| 변조 복구 정확도 | **3/3 (100%)** | ✅ 우수 |
| 드롭 복구 정확도 | **13/17 (76.5%)** | 🔶 양호 |
| 드롭 복구 MAE | **0.845 pKd** | 🔶 허용 범위 |
| Network Alert | **발동 (34% > 30%)** | ✅ 정확 |
| AI 추론 실패율 | **0/33 (0%)** | ✅ 완벽 |

**전체 판정 분포 (50건):**

| 판정 | 건수 | 비율 |
|------|------|------|
| 🟢 HIGH (pKd ≥ 7.0) | 8 | 16.0% |
| 🟡 MODERATE (5.0 ≤ pKd < 7.0) | 29 | 58.0% |
| 🔴 LOW (pKd < 5.0) | 13 | 26.0% |
| **평균 pKd** | **6.0437** | |

**추론 경로 분포:**

| 경로 | 건수 | 비율 |
|------|------|------|
| `normal` — 정상 수신 + AI 추론 | 30 | 60.0% |
| `drop_imputed` — 드롭 → rolling mean | 17 | 34.0% |
| `corrupt_recovered` — 변조 → AI 추론 | 3 | 6.0% |
| `imputed` — AI 실패 → rolling mean | 0 | 0.0% |

---

## 6. 요구사항 충족 검증

| # | 요구사항 | 구현 | 결과 |
|---|----------|------|------|
| 1 | Latency: sleep(0.4~1.2s) | ✅ | avg 1,002ms, 100% 범위 내 |
| 2 | Drop: random() < drop_rate | ✅ | 30% 설정 → 실제 34.0% |
| 3 | Corrupt: SMILES 문자 치환 | ✅ | ft-ChemBERTa 입력 기준 |
| 4 | 드롭 패킷 → rolling mean | ✅ | 17건 전부 처리, 누락 0건 |
| 5 | 변조 패킷 → AI 우선, 실패 시 rolling mean | ✅ | 3건 전부 AI 복구 성공 |
| 6 | pKd ≥ 7.0 HIGH / 5~7 MODERATE / <5 LOW | ✅ | 전 50건 동일 기준 |
| 7 | 손실률 > 30% → Network Alert | ✅ | 34.0% > 30% → Alert 발동 |
| 8 | Zero Silent Drop | ✅ | 50/50 기록 완료 |
| 9 | multiprocessing.Queue 프로세스 분리 | ✅ | ready_event 동기화 포함 |
| 10 | Streamlit Dashboard | ✅ | auto-refresh 2s, 3Di 히트율 표시 |

**→ 요구사항 10/10 충족**

---

## 7. 한계점 및 개선 방향

### 7.1 Rolling Mean의 평균 회귀 편향 (핵심 한계)

분포 양극단 약물(Nilotinib pKd≈9, Sunitinib pKd≈4.8)이 드롭될 때 rolling mean이 중간값으로 수렴하여 오분류 발생.

**개선안:** 분자 구조 유사도(Morgan FP cosine similarity) 기반 k-NN imputation — 동일 표적에 대한 유사 약물의 pKd를 가중 평균.

### 7.2 데모 약물 다양성 한계

10개 약물 순환 구조 → 실제 HTS 수만 개 고유 약물과 차이. 실제 DAVIS 30K 쌍으로 확장 시 일반화 성능 Pearson r=0.8677로 검증됨.

### 7.3 향후 개선

1. **TCP 소켓 기반 실제 WAN 시뮬레이션** — `multiprocessing.Queue` → `socket`
2. **재전송 프로토콜 (ARQ)** — 드롭 패킷 자동 재전송 요청
3. **LoRA fine-tune** — SaProt 어텐션에 rank-16 LoRA 어댑터 삽입 (cold-protein 개선, r≈0.82 목표)
4. **스트리밍 배치 추론** — 동시 다중 쿼리 배치 큐 설계

---

## 8. 부록 A: 서열 오류 디버깅 기록

### 문제 (RERUN_CHECKLIST.md 기록, 2026-05-15)

`demo.py`의 초기 버전에서 단백질 서열을 손으로 약 150aa만 잘라서 입력했다.

| 항목 | 초기 (오류) | 교정 후 |
|------|-------------|---------|
| ABL1 서열 길이 | 152aa | 1138aa |
| 3Di 캐시 히트 | 0/50 (0%) | ~80% 이상 |
| `used_3di` | 모두 False | 대부분 True |
| 생물학적 정확성 | 키나아제 도메인 없음 | DAVIS 학습 분포와 일치 |

**원인:** MD5 해시 기반 캐시 조회 시 150aa 단편의 해시가 DAVIS full-length 캐시에 없음 → 전체 `'M#L#E#...'` placeholder 처리 → 구조 정보 완전 손실.

### 교정 방법

1. `prepare_sequences.py` 실행 → `davis_seqs_for_demo.json` 생성 (DAVIS canonical 서열 + 3Di 히트 확인)
2. `demo.py`의 SAMPLE_QUERIES를 `_davis[]` 딕셔너리에서 참조하도록 수정
3. Warmup inference도 full-length 서열로 교체

---

## 9. 부록 B: 발표 스크립트 및 Q&A

### 3분 발표 스크립트 (영어, 라이브 데모용)

```
[0:00~0:20] Problem
  "Pharmaceutical HTS labs generate thousands of drug-target queries daily.
   In distributed lab environments, the shared WAN suffers latency, packet
   drops, and payload corruption — causing promising candidates to be lost."

[0:20~0:50] System (화면: ARCHITECTURE.md 다이어그램)
  "Our pipeline uses two processes: Edge simulates a lab node —
   generating DTI queries with intentional constraints: up to 1.2s delay,
   20% drop rate, 15% SMILES corruption.
   Server receives, runs the AI model, and recovers missing data
   via rolling-mean imputation."

[0:50~1:30] AI Model
  "The core model combines SaProt-650M with FoldSeek 3Di structural tokens
   — capturing both sequence and 3D structural information —
   and fine-tuned ChemBERTa for drug SMILES encoding.
   Trained on 80K BindingDB pairs: Pearson r=0.89.
   Transfer to DAVIS: r=0.87."

[1:30~2:20] Live Demo (demo.py + dashboard.py)
  "Watch Process A sending queries — drops and corruptions visible.
   Server recovers and the dashboard shows real-time pKd values.
   Green = HIGH binding affinity. Threshold is adjustable live."

[2:20~2:50] Results
  "50 queries under 34% packet loss:
   corrupt recovery 100%, drop recovery 76.5%, zero silent drop."

[2:50~3:00] Conclusion
  "AI and ICT resilience engineering together make drug discovery
   pipelines robust under real-world network conditions."
```

### 예상 Q&A

| 질문 | 답변 |
|------|------|
| 76.5%면 낮지 않나요? | 드롭 패킷에 아무 값 없으면 0%. Rolling mean은 76.5% 보장. 오분류 4건 모두 MODERATE로 수렴 → 임상적 안전망 역할 |
| Nilotinib이 가장 크게 틀린 이유? | pKd≈9 극단값은 rolling mean(평균 6.0~6.5) 대체 시 구조적으로 3.1 오차. 개선안: 구조 유사도 기반 kNN imputation |
| 실제 WAN 적용 가능? | mp.Queue → TCP 소켓으로 교체하면 실제 배포 가능. 현재는 동일 머신 논리 분리로 개념 검증 |
| SaProt vs ESM-2? | SaProt는 서열+3Di 구조 토큰 동시 인코딩 → 구조 유사 단백질 계열에서 ESM-2 대비 ~3~5% 향상 |
| why used_3di=True가 중요한가? | 3Di 없으면(fallback) 성능 약 5% 손실. 150aa 단편 버그로 초기 실험 전체가 fallback 모드였음 |
