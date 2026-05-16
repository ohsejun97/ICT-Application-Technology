# PPT Prototype Guide
## Bio-AI DTI Query Pipeline — ICT Application Technology

> AI 툴(Gemini/Claude)에 붙여넣어 PPT 초안을 생성하기 위한 슬라이드 구성 가이드.

---

## 디자인 가이드

- **컬러:** Crimson(`#DC143C`) 포인트 + 흰 배경 + 다크(`#1a1a1a`) 텍스트
- **폰트:** 제목 Bold 28pt / 본문 18pt / 캡션 14pt
- **레이아웃:** 16:9 와이드
- **총 슬라이드:** 8장 + 타이틀

---

## Slide 1 — Title

**제목:** Bio-AI DTI Query Pipeline  
**부제:** Real-time Drug-Target Interaction Prediction Under Network Constraints

| 항목 | 내용 |
|------|------|
| 과목 | ICT Application Technology |
| 이름 / 학번 | 오세준 / 2021270607 |
| 발표일 | 2026-05-22 |

**비주얼:** 약물 분자 구조 이미지 or 단순 crimson 배경 타이포

---

## Slide 2 — Problem & Motivation

**제목:** The Problem: Drug Queries Getting Lost in the Network

**핵심 메시지 (크게):**
> *"A promising drug candidate silently dropped = a potential treatment never evaluated."*

**3단 레이아웃:**

```
[Left]                    [Center]                   [Right]
Distributed HTS Labs  →   WAN Network         →   Central AI Server
(Edge Node)               · Latency 0.6~1.4s       (DTI Inference)
Drug SMILES +             · 30% Packet Loss
Protein Sequence          · 15% Payload Corrupt
```

**바텀 포인트:**
- 10 anti-cancer drugs × 6 kinase targets = **60 unique drug-target queries**
- Network degradation → **silent data loss** without recovery

---

## Slide 3 — System Architecture

**제목:** 6-Step Resilient Pipeline

**비주얼:** `ARCHITECTURE.md`의 Mermaid 다이어그램 스크린샷 사용  
(GitHub에서 `ARCHITECTURE.md` 열면 렌더링됨 → 스크린샷)

```
① Data Generation → ② Transmission → ③ Collection
      ↓ (WAN: delay / drop / corrupt)
④ AI Inference & Recovery → ⑤ Decision → ⑥ Dashboard
```

**하단 설명 박스 3개:**
- `multiprocessing.Queue` — Process A ↔ Process B IPC
- `mp.Event` — Server waits for model load before Edge transmits
- Auto-refresh Streamlit — 2s real-time monitoring

---

## Slide 4 — AI Model

**제목:** Dual-Encoder Architecture with 3D Structural Tokens

**모델 다이어그램 (텍스트로 표현):**

```
Drug SMILES  →  ft-ChemBERTa  →  [768-dim]  ──┐
                (layers 4~5 fine-tuned)          ├──▶  MLP Head  ──▶  pKd
Protein AA   →  3Di tokens    →  SaProt-650M  ──┘
               (FoldSeek /         (FP16,          [1280+768 → 512 → 256 → 1]
               AlphaFold2)          frozen)
```

**성능 테이블:**

| Dataset | Pearson r | RMSE |
|---------|-----------|------|
| BindingDB (pretrain) | **0.8923** | 0.7387 |
| DAVIS (transfer) | **0.8677** | 0.4572 |
| KIBA (transfer) | **0.8594** | 0.4268 |

**포인트:** 3Di 토큰 제거 시 r ≈ −0.05 (SaProt 논문) → 구조 정보가 성능에 직접 기여

---

## Slide 5 — Network Constraints & Recovery

**제목:** Intentional Constraints + 3-Path Recovery

**왼쪽 — 3종 제약 (코드 라인으로):**

```python
# ① Latency
time.sleep(random.uniform(0.6, 1.4))

# ② Packet Drop
if random.random() < 0.30:
    continue          # silently lost

# ③ Payload Corruption
smiles = corrupt_smiles(smiles, sigma=0.05)
```

**오른쪽 — 복구 경로 플로우:**

```
패킷 도착?
  ├── YES, 정상    →  SaProt + ChemBERTa  →  pKd  [normal]
  ├── YES, 변조    →  노이즈 SMILES로 추론  →  pKd  [corrupt_recovered]
  └── NO (드롭)   →  Rolling Mean (last 5) →  pKd  [drop_imputed]
```

**핵심:** 어떤 경우에도 **모든 쿼리에 pKd 값 제공** (Zero Silent Drop)

---

## Slide 6 — Experiment Results: Network

**제목:** 60 Queries, 36.7% Packet Loss → 0 Missed Decisions

**크게 표시할 숫자 (카드 형식 4개):**

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  60 / 60     │  │  36.7%       │  │  3 / 3       │  │  38 / 38     │
│  Zero Silent │  │  Packet Loss │  │  Corrupt     │  │  3Di Token   │
│  Drop ✅     │  │  → ALERT ✅  │  │  Recovery ✅ │  │  Hit Rate ✅ │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

**파이/바 차트 (경로 분포):**

| Path | Count | % |
|------|-------|---|
| normal | 35 | 58.3% |
| drop_imputed | 22 | 36.7% |
| corrupt_recovered | 3 | 5.0% |

**데이터 파일:** `results/demo_summary.json` 참조

---

## Slide 7 — Experiment Results: Selectivity

**제목:** Drug Selectivity Profiling — 10 × 6 Matrix

**테이블 (히트맵 스타일, HIGH=크림슨, MODERATE=연한 빨강, DROP=회색):**

```
              ABL1   EGFR   BRAF   BTK   PDGFRA   ALK
Imatinib      DROP   6.25   DROP   5.77   8.62★   6.56
Nilotinib     7.53   7.69   7.57   DROP   DROP    8.35★
Dasatinib     DROP   6.40   5.93   DROP   8.14★   5.92
Gefitinib     DROP   7.15★  6.42   DROP   8.08    6.41
Erlotinib     DROP   7.28★  DROP   6.99   DROP    5.92
Sorafenib     6.58   6.05   DROP   5.58   DROP    DROP
Vemurafenib   DROP   6.08   6.03   DROP   7.96★   6.36
Ibrutinib     6.03   5.67   DROP   5.00★  7.31    4.82
Sunitinib     6.32   5.92   DROP   5.25   DROP    5.26
Crizotinib    5.84   5.83   DROP   DROP   7.66★   DROP
```

**하단 인사이트 3줄:**
- Imatinib → PDGFRA **8.62** (highest): matches clinical use in GIST treatment
- Ibrutinib → BTK **5.00** (lowest among its row): confirms BTK selectivity
- DROP cells = network loss in action — pipeline still delivered 60/60 decisions

---

## Slide 8 — Conclusion & Future Work

**제목:** Key Takeaways

**왼쪽 — 달성:**
- ✅ Zero Silent Drop: 60/60 (100%)
- ✅ Corrupt Recovery: 3/3 (100%)
- ✅ 3Di Token Usage: 38/38 (100%)
- ✅ Network Alert: 36.7% > 30% triggered
- ✅ Selectivity matrix: biologically meaningful results

**오른쪽 — 한계 & 개선:**
- 🔶 Rolling mean: regresses to mean → misclassifies extreme pKd
- → **Fix:** Morgan FP cosine similarity k-NN imputation
- 🔶 Unique pair design: drop accuracy unverifiable
- → **Fix:** ARQ (Automatic Repeat reQuest) retransmission

**발표 마무리 문장:**
> *"AI and ICT resilience engineering together make drug discovery pipelines  
> robust under real-world network degradation."*

---

## AI 프롬프트 템플릿

아래를 Gemini / Claude 웹에 붙여넣으세요:

```
I need to create a PowerPoint presentation for a university project.
Topic: Bio-AI DTI Query Pipeline — Real-time Drug-Target Interaction Prediction Under Network Constraints
Course: ICT Application Technology
Student: Oh Sejun (2021270607)
Presentation: 3 minutes, English, live demo

Please create 8 slides with this structure:

Slide 1 - Title slide
Slide 2 - Problem: pharmaceutical HTS labs generate drug-target queries over WAN; network degradation (30% packet loss, latency, corruption) causes silent data loss
Slide 3 - 6-step pipeline: Data Generation → Transmission (with intentional constraints) → Collection → AI Inference & Recovery → Decision → Dashboard
Slide 4 - AI Model: SaProt-650M (protein, FP16) + ft-ChemBERTa (drug) + MLP Head. Performance: BindingDB r=0.89, DAVIS r=0.87, KIBA r=0.86
Slide 5 - Network constraints code (sleep for latency, random drop, SMILES corruption) + 3-path recovery (normal / corrupt_recovered / drop_imputed via rolling mean)
Slide 6 - Results: 60 queries, 36.7% packet loss, Zero Silent Drop 60/60, 3Di hit 100%, Network Alert triggered
Slide 7 - Drug selectivity matrix (10 drugs × 6 kinase targets). Key finding: Imatinib → PDGFRA 8.62 (matches GIST clinical use); Ibrutinib → BTK most selective
Slide 8 - Conclusion: achieved zero silent drop with AI+ICT resilience; limitation: rolling mean bias; future: k-NN imputation + ARQ retransmission

Design: crimson (#DC143C) accent color, white background, clean modern style.
Each slide should have a clear title, key visual or table, and 2-3 bullet points max.
```

---

## 참고 파일 위치 (repo pull 후)

| 용도 | 파일 |
|------|------|
| 아키텍처 다이어그램 | `ARCHITECTURE.md` → GitHub에서 열어 스크린샷 |
| 실험 결과 수치 | `results/demo_summary.json` |
| 상세 분석 | `docs/REPORT.md` Section 5 |
| 선택성 매트릭스 | `docs/REPORT.md` Section 5.2 |
| 모델 성능 수치 | `results/SaProt-650M-.../result.json` |

> **노트북에서 데모 실행 필요 시:** `.pt` 모델 가중치 파일이 gitignore라 없음.  
> `dti_head.pt`, `chemberta_ft.pt`를 USB/클라우드로 별도 복사 후  
> `results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random/` 에 넣을 것.
