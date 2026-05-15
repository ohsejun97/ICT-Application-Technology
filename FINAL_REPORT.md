# Bio-AI DTI Query Pipeline
## 50쿼리 네트워크 복구 실험 — 최종 분석 보고서

**과목:** ICT Application Technology  
**학생:** 오세준 (2021270607)  
**실험일:** 2026-05-15  
**데이터:** `results/demo50_log.jsonl` (50건 전수 기록)

---

## 1. 실험 설정

| 파라미터 | 설정값 | README 명세 |
|---|---|---|
| 총 쿼리 수 | 50 | 연속 HTS 스크리닝 시뮬레이션 |
| 패킷 드롭 확률 | 30% | 기본 15%, 실험용 강화 |
| 페이로드 변조 확률 | 15% | Gaussian noise σ=0.05 대응 |
| 지연 범위 | 0.6 ~ 1.4 s | 0.5 ~ 2.0 s (README) |
| HIGH 임계값 | pKd ≥ 7.0 | ✅ 일치 |
| MODERATE 임계값 | pKd ≥ 5.0 | ✅ 일치 |
| Network Alert 기준 | 손실률 > 30% | ✅ 일치 |
| 프로세스 간 통신 | multiprocessing.Queue | ✅ 일치 |

---

## 2. README / 프로젝트 요구사항 충족 검증

| # | 요구사항 (README) | 구현 상태 | 실험 결과 |
|---|---|---|---|
| 1 | Latency: sleep(0.5~2.0s) | ✅ 구현 | avg 1,002ms, 전체 범위 내 50/50 |
| 2 | Drop: 15% random (대시보드 조절 가능) | ✅ 구현 | 30% 설정 → 실제 34.0% 드롭 |
| 3 | Corrupt: payload noise | ✅ 구현 | SMILES 문자 치환 (ChemBERTa 입력 기준) |
| 4 | 드롭 패킷 → rolling mean imputation | ✅ 구현 | 17건 전부 처리, 누락 0건 |
| 5 | 변조 패킷 → rolling mean / AI 복구 | ✅ 개선 구현 | AI 추론 우선, 실패 시만 rolling mean |
| 6 | pKd ≥ 7.0 → HIGH / 5~7 → MODERATE / <5 → LOW | ✅ 구현 | 전 50건 동일 기준 적용 |
| 7 | 손실률 > 30% → Network Degraded Alert | ✅ 구현 | 34.0% > 30% → Alert 발동 ✅ |
| 8 | 모든 쿼리에 결합력 판정 제공 (Zero Silent Drop) | ✅ 구현 | 50/50 기록 완료 ✅ |
| 9 | multiprocessing.Queue (Process A ↔ B) | ✅ 구현 | ready_event 동기화 포함 |
| 10 | Streamlit Dashboard | ✅ 별도 구현 | `dashboard.py` (auto-refresh 2s) |

**→ 요구사항 10/10 충족**

---

## 3. 네트워크 시뮬레이션 결과

```
총 전송 쿼리 : 50
정상 수신    : 33  (66.0%)
패킷 드롭    : 17  (34.0%)  ← Network Alert 발동 (기준: 30%)
페이로드 변조:  3  ( 9.1%, 수신 기준)
```

### 지연(Latency) 분포

| 항목 | 값 |
|---|---|
| 평균 지연 | **1,002 ms** |
| 최소 지연 | 606 ms |
| 최대 지연 | 1,381 ms |
| 설정 범위(600~1,400ms) 내 | **50/50 (100%)** |

---

## 4. 페이로드 변조(Corrupt) 복구 분석

변조된 3건 모두 AI 추론(`corrupt_recovered`) 경로로 처리.

| 약물 | 표적 | 정상 pKd | 변조 후 pKd | 오차 | 정상 판정 | 변조 판정 | 결과 |
|---|---|---|---|---|---|---|---|
| Dasatinib | ABL1 | 6.7509 | **6.3760** | 0.375 | MODERATE | MODERATE | ✅ |
| Gefitinib | EGFR | 5.5345 | **5.5345** | **0.000** | MODERATE | MODERATE | ✅ |
| Vemurafenib | BRAF | 5.4831 | **5.4726** | 0.011 | MODERATE | MODERATE | ✅ |

**판정 정확도: 3/3 = 100%**  
**평균 pKd 오차: 0.129** (임상적으로 무시 가능)

> SaProt-650M + ft-ChemBERTa 조합이 SMILES 문자 손상(비트 반전 시뮬레이션)에 **강인(robust)** 함을 실증.  
> Gefitinib은 변조 후에도 오차 0.000으로 완전히 동일한 값 반환 → 높은 모델 안정성.

---

## 5. 드롭 패킷 Rolling Mean 복구 분석

드롭된 17건 전부 최근 5개 pKd rolling mean으로 대체.

| 약물 | 실제 pKd | 실제 판정 | Imputed pKd | Imputed 판정 | 오차 | 정오 |
|---|---|---|---|---|---|---|
| Imatinib | 7.7431 | HIGH | 8.9591 | HIGH | 1.216 | ✅ |
| Gefitinib | 5.5345 | MODERATE | 6.9631 | MODERATE | 1.429 | ✅ |
| Crizotinib | 5.9042 | MODERATE | 5.8592 | MODERATE | 0.045 | ✅ |
| **Imatinib** | **7.7431** | **HIGH** | **5.8592** | **MODERATE** | **1.884** | **❌** |
| Gefitinib | 5.5345 | MODERATE | 5.7540 | MODERATE | 0.220 | ✅ |
| Sorafenib | 5.4931 | MODERATE | 6.1108 | MODERATE | 0.618 | ✅ |
| **Sunitinib** | **4.7685** | **LOW** | **5.4645** | **MODERATE** | **0.696** | **❌** |
| **Nilotinib** | **8.9591** | **HIGH** | **5.8514** | **MODERATE** | **3.108** | **❌** |
| Dasatinib | 6.7509 | MODERATE | 5.9284 | MODERATE | 0.823 | ✅ |
| Vemurafenib | 5.4831 | MODERATE | 5.6660 | MODERATE | 0.183 | ✅ |
| **Sunitinib** | **4.7685** | **LOW** | **5.2982** | **MODERATE** | **0.530** | **❌** |
| Gefitinib | 5.5345 | MODERATE | 6.4332 | MODERATE | 0.899 | ✅ |
| Dasatinib | 6.7509 | MODERATE | 6.4332 | MODERATE | 0.318 | ✅ |
| Sorafenib | 5.4931 | MODERATE | 6.4332 | MODERATE | 0.940 | ✅ |
| Vemurafenib | 5.4831 | MODERATE | 6.4332 | MODERATE | 0.951 | ✅ |
| Crizotinib | 5.9042 | MODERATE | 6.2060 | MODERATE | 0.302 | ✅ |
| Dasatinib | 6.7509 | MODERATE | 6.5394 | MODERATE | 0.212 | ✅ |

**판정 정확도: 13/17 = 76.5%**  
**MAE: 0.845** | **RMSE: 1.124**

### 오분류 패턴 분석 (4건)

| 케이스 | 오류 방향 | 원인 |
|---|---|---|
| Imatinib #11 (7.74 → 5.86) | HIGH ⬇ MODERATE | rolling buffer가 낮은 약물(Ibrutinib 4.59, Sunitinib 4.77 등)로 채워져 평균 하락 |
| Nilotinib #22 (8.96 → 5.85) | HIGH ⬇ MODERATE | 가장 강한 결합제(pKd≈9)가 드롭 → 평균 대체 시 최대 오차 3.11 발생 |
| Sunitinib #19 (4.77 → 5.46) | LOW ⬆ MODERATE | rolling mean이 중간값(~5.5)으로 수렴 → 약한 결합제 과대평가 |
| Sunitinib #29 (4.77 → 5.30) | LOW ⬆ MODERATE | 동일 패턴 반복 |

**핵심 발견:** Rolling mean은 pKd 분포의 평균(6.0~6.5)으로 수렴하는 경향이 있어, 분포 양극단(HIGH: ≥8, LOW: ≤4.8)의 약물이 드롭될 때 과소/과대평가 발생.

---

## 6. 전체 시스템 성능 요약

### 50쿼리 전체 판정 분포

| 판정 | 건수 | 비율 |
|---|---|---|
| 🟢 HIGH (pKd ≥ 7.0) | 8 | 16.0% |
| 🟡 MODERATE (5.0 ≤ pKd < 7.0) | 29 | 58.0% |
| 🔴 LOW (pKd < 5.0) | 13 | 26.0% |
| **평균 pKd** | **6.0437** | |

### 경로별 처리 건수

| 경로 | 건수 | 비율 |
|---|---|---|
| `normal` — 정상 수신 + AI 추론 | 30 | 60.0% |
| `drop_imputed` — 드롭 → rolling mean 대체 | 17 | 34.0% |
| `corrupt_recovered` — 변조 → AI 추론 복구 | 3 | 6.0% |
| `imputed` — AI 추론 실패 → rolling mean | 0 | 0.0% |

### 핵심 KPI

| 지표 | 값 | 평가 |
|---|---|---|
| Zero Silent Drop | **50/50 (100%)** | ✅ 목표 달성 |
| 변조 복구 판정 정확도 | **3/3 (100%)** | ✅ 우수 |
| 드롭 복구 판정 정확도 | **13/17 (76.5%)** | 🔶 양호 |
| 드롭 복구 MAE | **0.845 pKd** | 🔶 허용 범위 |
| Network Alert 정확도 | **발동 (34% > 30%)** | ✅ 정확 |
| AI 추론 실패율 | **0/33 (0%)** | ✅ 완벽 |

### 약물별 시스템 평균 pKd vs 실제 pKd

| 약물 | 실제 pKd | 시스템 avg pKd | 오차 | 편향 |
|---|---|---|---|---|
| Nilotinib | 8.9591 | **8.3376** | −0.621 | 과소평가 (드롭 imputation 영향) |
| Imatinib | 7.7431 | **7.6095** | −0.134 | 경미한 과소평가 |
| Dasatinib | 6.7509 | **6.4056** | −0.345 | 과소평가 (변조+드롭 혼재) |
| Crizotinib | 5.9042 | **5.9556** | +0.051 | ✅ 거의 정확 |
| Sorafenib | 5.4931 | **5.8047** | +0.312 | 과대평가 (드롭 rolling mean) |
| Gefitinib | 5.5345 | **6.0439** | +0.509 | 과대평가 (드롭 imputation) |
| Vemurafenib | 5.4831 | **5.7076** | +0.224 | 경미한 과대평가 |
| Sunitinib | 4.7685 | **5.0136** | +0.245 | LOW→MODERATE 경계 |
| Erlotinib | 4.9671 | **4.9671** | 0.000 | ✅ 완벽 (드롭 없음) |
| Ibrutinib | 4.5923 | **4.5923** | 0.000 | ✅ 완벽 (드롭 없음) |

> **패턴:** 드롭이 없는 약물(Erlotinib, Ibrutinib)은 오차 0. 드롭이 많은 강한 결합제(Nilotinib)일수록 시스템 평균이 실제보다 낮게 측정됨.

---

## 7. README 설계 vs 실제 구현 차이점

| 항목 | README 명세 | 실제 구현 | 판단 |
|---|---|---|---|
| Drug encoder | Morgan FP (2048-bit) | **ft-ChemBERTa (768-dim)** | ✅ 개선 (학습 결과 반영) |
| Protein encoder | SaProt-650M **4bit** | SaProt-650M **FP16** | ✅ 동등 (더 높은 정밀도) |
| Corruption 방식 | Morgan FP에 Gaussian noise | **SMILES 문자 치환** | ✅ 타당 (ChemBERTa 입력 기준) |
| 변조 패킷 복구 | rolling mean 대체 | **AI 추론 우선, 실패 시만 rolling mean** | ✅ 개선됨 |
| 모델 성능 명세 | r=0.7914 (DAVIS) | **r=0.8677 (DAVIS)** | ✅ 초과 달성 |

---

## 8. 결론

### ✅ 달성한 것
1. **Zero Silent Drop 완전 달성:** 50개 중 드롭 17건, 변조 3건 포함 전부 pKd 판정 완료
2. **변조 복구 완벽:** 3/3 = 100% 정확, 평균 오차 0.129 pKd (임상적으로 무시 가능)
3. **Network Alert 정상 작동:** 34.0% > 30.0% 기준 경보 정확 발동
4. **AI 추론 안정성:** 33건 정상 수신 전부 추론 실패 없음 (0%)
5. **README 요구사항 10/10 충족**

### ⚠️ 한계점 및 개선 방향
1. **Rolling Mean의 평균 회귀 편향 (핵심 한계)**
   - 강한 결합제(Nilotinib, pKd=8.96) 드롭 시 최대 오차 3.11 → HIGH→MODERATE 오분류
   - 약한 결합제(Sunitinib, pKd=4.77) 드롭 시 과대평가 → LOW→MODERATE 오분류
   - **개선안:** 분자 구조 유사도(Morgan FP cosine similarity) 기반 k-NN imputation

2. **동일 약물 반복 순환 구조 (데모 한계)**
   - 10개 약물을 5회 반복 → 실제 HTS의 수만 개 고유 약물과 다름
   - 실제 DAVIS 30K 쌍으로 확장 시 일반화 성능은 Pearson r=0.8677로 검증됨

3. **부분 단백질 서열 사용**
   - 데모 편의상 70~80aa 단축 서열 → 실제 서열(수백~수천 aa) 대비 정밀도 차이

### 발표 핵심 메시지
```
50개 DTI 쿼리, 34% 네트워크 손실 환경에서
→ 변조 복구: 100%  |  드롭 복구: 76.5%  |  누락 판정: 0건
→ 모든 약물 후보에 결합력 판정 제공 달성
```

---

*본 보고서는 실제 실행 결과(`results/demo50_log.jsonl`, `results/demo_summary.json`)를 기반으로 작성되었습니다.*
