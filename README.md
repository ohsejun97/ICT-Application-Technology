# Bio-AI DTI Query Pipeline

**과목:** ICT Application Technology | **학생:** 오세준 (2021270607)  
**발표일:** 2026-05-22 | **GitHub:** [ohsejun97/ICT-Application-Technology](https://github.com/ohsejun97/ICT-Application-Technology)

WAN 네트워크 열화(지연·패킷 손실·페이로드 변조) 환경에서도 **모든 약물-표적 쌍에 누락 없이** 결합 친화도(pKd)를 예측하는 end-to-end ICT 파이프라인.

---

## 이 프로젝트는 무엇을 실험하는가?

### 연구 대상 범위

**Imatinib + BCR-ABL 하나만을 위한 시스템이 아니다.**  
DAVIS 데이터셋에서 검증된 **10종 항암 약물 × 6종 키나아제 표적**을 처리한다.

| 약물 | 표적 단백질 | 적응증 |
|------|-------------|--------|
| Imatinib, Nilotinib, Dasatinib | ABL1 (1138aa) | 만성 골수성 백혈병 (CML) |
| Gefitinib, Erlotinib | EGFR (1210aa) | 비소세포 폐암 (NSCLC) |
| Sorafenib, Vemurafenib | BRAF (772aa) | 흑색종·간암 |
| Ibrutinib | BTK (666aa) | B세포 림프종 |
| Sunitinib | PDGFRA (1089aa) | GIST·신세포암 |
| Crizotinib | ALK (1620aa) | ALK+ NSCLC |

모든 단백질 서열은 DAVIS canonical full-length 서열 (`davis_seqs_for_demo.json`)을 사용하며, FoldSeek 3Di 구조 토큰 캐시 히트가 확인된 서열이다.

### AI 모델 요약

```
SMILES  → ft-ChemBERTa (768-dim)  ─┐
                                     ├→ MLP Head → pKd (회귀)
AA seq  → 3Di tokens → SaProt-650M  ─┘
```

- BindingDB 사전학습: **Pearson r = 0.8923**
- DAVIS 전이학습: **Pearson r = 0.8677**
- KIBA 전이학습: **Pearson r = 0.8594**

---

## 빠른 시작

### 환경 설정

```bash
# conda 환경 생성 (처음 한 번만)
bash setup_env.sh

# 또는 직접 설치
conda create -n bioinfo python=3.10 -y
conda activate bioinfo
pip install -r requirements.txt
```

> **주의:** `torchvision`, `torchaudio`는 transformers와 충돌 — 설치하지 말 것.  
> DeepPurpose는 `prepare_sequences.py` 실행 시에만 필요: `pip install DeepPurpose`

### 최초 1회: DAVIS 서열 준비

```bash
# DAVIS canonical 서열 추출 (3Di 캐시 히트 확인 포함)
conda run -n bioinfo python scripts/prepare_sequences.py
# → davis_seqs_for_demo.json 생성 (이미 repo에 포함되어 있어 재실행 불필요)
```

### 발표 데모 실행 (메인)

```bash
# 터미널 1: Streamlit 대시보드 먼저 실행
conda run -n bioinfo streamlit run dashboard.py
# → http://localhost:8501 브라우저에서 열기

# 터미널 2: 파이프라인 데모 실행
conda run -n bioinfo python demo.py

# 영상 촬영용 권장 설정 (50쿼리, 30% 드롭)
conda run -n bioinfo python demo.py \
    --n_queries 50 \
    --drop_rate 0.30 \
    --corrupt_rate 0.15 \
    --lat_min 0.6 --lat_max 1.4 \
    --seed 77 \
    --output results/demo50_log.jsonl
```

대시보드 사이드바에서 "로그 소스"를 `demo (demo_log.jsonl)`으로 선택 후 자동 새로고침 ON.

### 기타 실행 명령

```bash
# 단순 파이프라인 (UI 없는 버전)
conda run -n bioinfo python scripts/pipeline.py
conda run -n bioinfo python scripts/pipeline.py --n_queries 50 --drop_rate 0.20

# 모델 재학습 (GPU 필요, ~7시간)
conda run -n bioinfo python models/train.py --dataset bindingdb
conda run -n bioinfo python models/train.py --dataset davis

# 시각화 생성
conda run -n bioinfo python models/plot_poster_figures.py
```

---

## 파일 구조

```
ICT_2026/
│
├── demo.py                    ← 발표용 메인 데모 (ANSI 색상, 영상 촬영 최적화)
├── dashboard.py               ← Streamlit 실시간 대시보드 (auto-refresh 2s)
│
├── davis_seqs_for_demo.json   ← 10개 단백질 full-length 서열 + 3Di 히트 정보
├── requirements.txt           ← Python 의존성
├── setup_env.sh               ← conda 환경 초기 설정 스크립트
│
├── tools/                     ← 추론 엔진 (demo.py가 런타임에 임포트)
│   ├── dti_tool.py            ← DTI 추론 API — SaProt + ChemBERTa + MLP Head (핵심)
│   ├── chemberta_drug_encoder.py  ← ChemBERTa 약물 인코더 (학습 시 사용)
│   ├── foldseek_tool.py       ← FoldSeek 3Di 토큰 추출 (캐시 구축 시 사용)
│   ├── alphafold_tool.py      ← AlphaFold2 구조 조회 (캐시 구축 시 사용)
│   └── gnn_drug_encoder.py    ← GNN 약물 인코더 (학습 실험용)
│
├── models/                    ← 모델 학습 · 전이학습 · 평가 스크립트
│   ├── train.py               ← 메인 학습 (BindingDB/DAVIS/KIBA, LoRA 지원)
│   ├── finetune_head.py       ← 헤드 전이학습 (기본)
│   ├── finetune_head_ft.py    ← ft-ChemBERTa + 헤드 전이학습
│   ├── train_chemberta_unfreeze.py ← ChemBERTa layers 4~5 unfreeze 학습
│   ├── cross_eval.py          ← 교차 데이터셋 평가 (DAVIS↔KIBA)
│   ├── build_3di_cache.py     ← FoldSeek 3Di 토큰 캐시 구축
│   ├── preprocess_bindingdb.py ← BindingDB 전처리
│   ├── plot_poster_figures.py ← 발표 시각화 생성
│   └── README.md              ← 각 스크립트 상세 설명
│
├── scripts/                   ← 환경 준비 및 유틸리티
│   ├── prepare_sequences.py   ← DAVIS full-length 서열 추출 → davis_seqs_for_demo.json
│   ├── pipeline.py            ← 기본 파이프라인 (demo.py 단순화 버전)
│   └── README.md              ← 각 스크립트 상세 설명
│
├── cache/                     ← 3Di 토큰 캐시 (MD5 해시 → 구조 토큰)
│   ├── 3di_tokens_davis.json  ← DAVIS 379개 단백질 3Di 토큰
│   ├── 3di_tokens_kiba.json   ← KIBA 단백질 3Di 토큰
│   └── 3di_tokens_bindingdb.json ← BindingDB 단백질 3Di 토큰
│
├── results/                   ← 학습 모델 가중치 및 실험 결과
│   ├── SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random/
│   │   ├── result.json        ← 학습 결과 (r=0.8923)
│   │   ├── dti_head.pt        ← MLP head 가중치 (demo.py 로드)
│   │   └── chemberta_ft.pt    ← fine-tuned ChemBERTa 가중치
│   ├── finetune_davis_random_.../result.json   ← DAVIS 전이 결과 (r=0.8677)
│   ├── finetune_kiba_random_.../result.json    ← KIBA 전이 결과 (r=0.8594)
│   ├── demo_log.jsonl         ← 최종 실험 결과 로그
│   └── demo_summary.json      ← 최종 실험 요약
│
├── docs/                      ← 기술 문서
│   └── REPORT.md              ← 종합 기술 보고서 (모델·실험·Q&A 포함)
│
├── ARCHITECTURE.md            ← 시스템 아키텍처 다이어그램 (Mermaid)
└── README.md                  ← 이 파일
```

---

## 주요 파라미터

`demo.py` / `pipeline.py` 공통 CLI 옵션:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--n_queries` | 20 | 전송할 쿼리 수 |
| `--drop_rate` | 0.20 | 패킷 드롭 확률 (0.0~1.0) |
| `--corrupt_rate` | 0.15 | 페이로드 변조 확률 |
| `--lat_min` / `--lat_max` | 0.4 / 1.2 | 전송 지연 범위 (초) |
| `--pkd_high` | 7.0 | HIGH 결합 임계값 |
| `--pkd_mod` | 5.0 | MODERATE 결합 임계값 |
| `--seed` | 7 | 재현성 시드 |
| `--output` | results/demo_log.jsonl | 결과 로그 경로 |

---

## 출력 형식

결과 로그 (`*.jsonl`) 각 행은 다음 형식:

```json
{
  "query_id": "Q01",
  "drug_name": "Imatinib",
  "protein_name": "ABL1",
  "pKd": 7.7431,
  "decision": "HIGH",
  "path": "normal",
  "corrupt": false,
  "used_3di": true,
  "latency_ms": 842.3,
  "timestamp": "2026-05-16T..."
}
```

`path` 값: `normal` | `corrupt_recovered` | `drop_imputed` | `imputed`

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 언어 | Python 3.10 (`bioinfo` conda 환경) |
| 단백질 인코더 | SaProt-650M-AF2 (EsmModel, FP16, frozen) |
| 구조 토큰 | FoldSeek 3Di (AlphaFold2 구조 기반, MD5 캐시) |
| 약물 인코더 | ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, layers 4~5 fine-tuned) |
| 회귀 헤드 | MLP [1280+768 → 512 → 256 → 64 → 1] |
| 프로세스 통신 | `multiprocessing.Queue` + `mp.Event` |
| 대시보드 | Streamlit (auto-refresh 2s) |
| 데이터셋 | BindingDB (80K), DAVIS (30K), KIBA (118K) |
| 화학정보학 | RDKit |

---

## 실험 핵심 결과 (50쿼리, 드롭률 34%)

```
Zero Silent Drop:   50/50  (100%)
변조 복구 정확도:    3/3   (100%)
드롭 복구 정확도:   13/17  (76.5%)
Network Alert:      정확 발동 (34% > 30% 기준)
AI 추론 실패:        0/33  (0%)
```

자세한 분석은 [`docs/REPORT.md`](docs/REPORT.md) 참조.
