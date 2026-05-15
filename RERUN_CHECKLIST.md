# 재실험 체크리스트
## 현재 데모의 오류 목록 + 연구실 GPU 환경 교정 방법

---

## 오류 1 — 단백질 서열 불일치 (핵심 문제)

### 무슨 일이 일어났나

`demo.py`의 SAMPLE_QUERIES에 단백질 서열을 손으로 앞 ~150aa만 잘라서 넣었다.

```
현재 demo ABL1 서열: 152aa (첫 N-말단 150aa만)
DAVIS 학습 데이터 ABL1: 1130aa (UniProt canonical)
DAVIS 캐시 서열 길이: min=288aa, median=651aa, max=2549aa
```

### 왜 문제인가

| 항목 | 현재 상태 | 올바른 상태 |
|---|---|---|
| 서열 길이 | 150aa | 600~1130aa |
| 학습 분포와 일치 | ❌ 분포 밖 | ✅ 동일 분포 |
| 3Di 캐시 히트 | 0/50 (0%) | ~30~40/50 예상 |
| used_3di | False (전체) | True (대부분) |
| 구조 정보 활용 | 0% | ~5% 성능 향상분 회복 |

### 생물학적 오류

- 150aa 짜리 N-말단 단편은 실제 **키나아제 도메인**이 아님
- ABL1 키나아제 도메인은 약 245~531번 잔기 (287aa) — 현재 서열엔 없음
- SaProt 모델이 "키나아제 도메인이 없는 단백질"을 보고 pKd를 예측하는 상황
- 그래서 Dasatinib(강결합제)가 Imatinib보다 낮게 나온 것

---

## 오류 2 — 3Di 구조 토큰 미사용

### 무슨 일이 일어났나

`dti_tool.py`의 `_seq_to_sa()` 함수는 MD5 해시로 캐시를 조회한다.

```python
seq_hash = hashlib.md5(aa_seq.encode()).hexdigest()
tokens_3di = _3di_cache.get(seq_hash)
# 캐시 미스 시 → '#' placeholder fallback
return "".join(aa + "#" for aa in aa_seq)
```

demo 서열(150aa 단편)의 MD5 해시는 DAVIS 캐시에 없으므로  
**50쿼리 전부 `'M#L#E#I#...'` 형태로 처리** — 구조 정보 완전 손실.

### 영향

SaProt 논문에 따르면 3Di 토큰 제거 시 Pearson r 약 −0.05 (~5%) 하락.  
즉, 현재 데모는 모델 성능 풀스펙이 아닌 약화된 버전으로 추론 중.

---

## 교정 방법 (연구실에서 순서대로)

### Step 1 — 실제 DAVIS 서열 추출

```bash
conda activate bioinfo
```

```python
# extract_davis_seqs.py 로 저장 후 실행
from DeepPurpose import dataset
import hashlib, json

X_drug, X_target, y = dataset.load_process_DAVIS('./data', binary=False)

# 길이로 단백질 식별
target_lengths = {
    1130: 'ABL1',
    1210: 'EGFR',
    766:  'BRAF',
    659:  'BTK',
    1089: 'PDGFRA',
}

cache = json.load(open('cache/3di_tokens_davis.json'))
found = {}

for seq in set(X_target):
    h = hashlib.md5(seq.encode()).hexdigest()
    slen = len(seq)
    if slen in target_lengths:
        name = target_lengths[slen]
        if h in cache and cache[h].get('status') == 'ok':
            found[name] = seq
            print(f"✅ {name} ({slen}aa) — 3Di 캐시 히트, 토큰: {cache[h]['tokens_3di'][:10]}...")
        else:
            print(f"⚠️  {name} ({slen}aa) — 캐시 미스")
    # ALK는 길이 불명 — 500aa 이상 후보 출력
    elif slen > 1400 and h in cache:
        print(f"  ALK 후보? ({slen}aa) hash={h[:16]}...")

with open('davis_seqs_for_demo.json', 'w') as f:
    json.dump(found, f, indent=2)
print(f"\n저장 완료: {list(found.keys())}")
```

### Step 2 — demo.py 서열 교체

`demo.py` 파일 상단 import 바로 아래에 추가:

```python
import json as _json, pathlib as _pl
_davis_seqs = _json.load(open(_pl.Path(__file__).parent / 'davis_seqs_for_demo.json'))
```

`SAMPLE_QUERIES`의 AA 서열 부분을 전부 `_davis_seqs.get("ABL1")` 형태로 교체:

```python
SAMPLE_QUERIES = [
    ("Q01", "Imatinib",
     "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
     "ABL1", _davis_seqs["ABL1"]),

    ("Q02", "Nilotinib",
     "CC1=CN=C(C(=C1)NC(=O)C2=CC(=CC=N2)C(F)(F)F)NC3=CC(=C(C=C3)CN4CCN(CC4)C)C(F)(F)F",
     "ABL1", _davis_seqs["ABL1"]),

    ("Q03", "Gefitinib",
     "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
     "EGFR", _davis_seqs["EGFR"]),

    ("Q04", "Erlotinib",
     "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
     "EGFR", _davis_seqs["EGFR"]),

    ("Q05", "Dasatinib",
     "CC1=NC(=CC(=C1)NC(=O)C2=CC(=CC=C2)Cl)NC3=NC=C(C=N3)C4=CN=CC=C4",
     "ABL1", _davis_seqs["ABL1"]),

    ("Q06", "Sorafenib",
     "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
     "BRAF", _davis_seqs["BRAF"]),

    ("Q07", "Vemurafenib",
     "CCCS(=O)(=O)NC1=CC(=C(C=C1F)NC(=O)C2=CNC3=CC(=C(C=C23)Cl)F)F",
     "BRAF", _davis_seqs["BRAF"]),

    ("Q08", "Ibrutinib",
     "C=CC(=O)N1CCCC(C1)N2C=NC3=C(N=CN=C23)NCC4=CC=CC=C4",
     "BTK", _davis_seqs["BTK"]),

    ("Q09", "Sunitinib",
     "CCN(CC)CCNC(=O)C1=C(NC2=CC=CC3=CC=CC=C23)C(=O)C1=O",
     "PDGFRA", _davis_seqs["PDGFRA"]),

    # ALK가 캐시에 없으면 ABL1로 대체 (Crizotinib은 ABL1에도 결합)
    ("Q10", "Crizotinib",
     "CCCS(=O)(=O)NC1=C2C=C(NC(=O)C3=CN=C4C=CC=CC4=C3)C=CC2=NC(=N1)N",
     "ABL1", _davis_seqs.get("ALK", _davis_seqs["ABL1"])),
]
```

### Step 3 — 재실험

```bash
conda run -n bioinfo python demo.py \
    --n_queries 50 \
    --drop_rate 0.30 \
    --corrupt_rate 0.15 \
    --lat_min 0.6 --lat_max 1.4 \
    --seed 77 \
    --output results/demo50_log.jsonl
```

### Step 4 — 3Di 히트율 확인

```bash
python -c "
import json
r = [json.loads(l) for l in open('results/demo50_log.jsonl')]
used = sum(1 for x in r if x.get('used_3di'))
print(f'used_3di=True: {used}/50')
print(f'목표: 30건 이상이면 정상')
"
```

### Step 5 — 시각화 재생성

```bash
python scripts/plot_presentation.py
```

---

## 교정 후 기대 변화

| 지표 | 현재 | 교정 후 예상 |
|---|---|---|
| used_3di | 0/50 | 30~45/50 |
| Dasatinib vs Imatinib 순서 | 역전 (오류) | 정상화 가능성 높음 |
| 변조 복구 정확도 | 100% | 100% 유지 |
| 드롭 복구 정확도 | 76.5% | 소폭 변동 |
| Zero Silent Drop | 50/50 | 50/50 유지 |
| Network Alert | 정상 | 정상 유지 |
| 발표 신뢰도 | 3Di 미사용 → 약점 | 3Di 정상 활용 → 강점 |

---

## 교정 불가 시 발표 대응 (현재 결과 그대로 쓸 경우)

Q&A에서 "왜 used_3di가 전부 False냐"고 물으면:

> *"The demo sequences were manually truncated to ~150aa for convenience,
> which didn't match the 3Di cache built from full DAVIS sequences.
> The model fell back to '#' placeholder — still functional,
> with approximately 5% performance degradation per the SaProt paper.
> The correct fix is using the actual DAVIS canonical sequences,
> which I plan to address in the next iteration."*

이 답변이 가능하려면 이 문서의 내용을 숙지하고 있어야 합니다.

---

## 변경 불필요한 것들 (건드리지 말 것)

- `train_dti_saprot.py`, `scripts/finetune_head_ft.py` — 학습 결과 정상
- `pipeline.py` ICT 로직 (드롭/변조/복구/Alert) — 완전 정상
- `dashboard.py` — 정상
- 학습 결과 수치 (r=0.8923, 0.8677, 0.8594) — DAVIS 실제 서열로 검증된 값
- `results/SaProt-650M-bindingdb-*/result.json` — 건드리지 말 것
