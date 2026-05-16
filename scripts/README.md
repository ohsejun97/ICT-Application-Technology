# scripts/

데모 실행 전 **환경 준비 및 유틸리티** 스크립트 모음.

> 모든 스크립트는 **프로젝트 루트에서** 실행해야 한다.

---

## 파일별 설명

### `prepare_sequences.py` — DAVIS 단백질 서열 준비

DAVIS 데이터셋에서 3Di 캐시 히트가 확인된 full-length 단백질 서열을 추출하여  
`davis_seqs_for_demo.json`으로 저장.

**데모 실행 전 최초 1회 실행 필요.** (이미 repo에 포함되어 있으면 생략 가능)

```bash
python scripts/prepare_sequences.py
```

**출력:** `davis_seqs_for_demo.json` — ABL1, EGFR, BRAF, BTK, PDGFRA, ALK full-length 서열

**왜 필요한가?**  
SaProt의 3Di 구조 토큰 캐시는 DAVIS canonical 서열(600~1600aa) 기준으로 구축되어 있다.  
이 스크립트는 DAVIS 데이터에서 해당 서열을 찾아 MD5 캐시 히트를 검증한다.

---

### `pipeline.py` — 기본 파이프라인 (텍스트 출력)

`demo.py`의 단순화 버전. ANSI 색상 없이 기본 텍스트로 파이프라인 실행.  
대시보드 없이 빠르게 동작 확인할 때 사용.

```bash
python scripts/pipeline.py
python scripts/pipeline.py --n_queries 30 --drop_rate 0.20
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--n_queries` | 30 | 전송 쿼리 수 |
| `--drop_rate` | 0.15 | 패킷 드롭 확률 |
| `--noise_sigma` | 0.05 | SMILES 변조 강도 |
| `--lat_min/max` | 0.5/2.0 | 지연 범위 (초) |

**출력:** `results/pipeline_log.jsonl`

> 발표용 데모는 루트의 `demo.py`를 사용할 것.
