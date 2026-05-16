# models/

SaProt + ChemBERTa DTI 모델의 **학습 · 전이학습 · 평가 · 데이터 전처리** 스크립트 모음.  
실험 재현이나 모델 재학습이 필요할 때 사용한다. 데모 실행(`demo.py`)에는 불필요.

> 모든 스크립트는 **프로젝트 루트에서** 실행해야 한다.  
> 예: `conda run -n bioinfo python models/train.py --dataset davis`

---

## 파일별 설명

### `train.py` — 메인 학습 스크립트

SaProt-650M + ft-ChemBERTa + MLP Head를 DAVIS / KIBA / BindingDB로 학습.

```bash
# BindingDB 사전학습 (약 7시간, GPU 필요)
python models/train.py --dataset bindingdb --encoder 650M

# DAVIS 학습
python models/train.py --dataset davis --encoder 650M

# LoRA fine-tune (SaProt 어텐션에 rank-16 어댑터)
python models/train.py --dataset davis --encoder 650M --lora
```

**출력:** `results/<run_name>/dti_head.pt`, `chemberta_ft.pt`, `result.json`

---

### `finetune_head.py` — 헤드 전이학습 (기본)

BindingDB 사전학습 가중치를 불러와 DAVIS / KIBA에 MLP Head만 fine-tune.  
임베딩이 캐시된 경우 ~1~2분 내 완료.

```bash
python models/finetune_head.py \
    --source_model results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random \
    --target_dataset davis
```

---

### `finetune_head_ft.py` — ft-ChemBERTa + 헤드 전이학습

ChemBERTa drug embedding도 함께 재계산하여 fine-tune.  
`finetune_head.py`보다 느리지만 더 높은 성능.

```bash
python models/finetune_head_ft.py \
    --source_model results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random \
    --target_dataset davis
```

---

### `train_chemberta_unfreeze.py` — ChemBERTa layers 4~5 unfreeze 학습

ChemBERTa의 상위 레이어(4~5)를 unfreeze하고 BindingDB로 end-to-end 학습.  
현재 배포 모델의 `chemberta_ft.pt`를 생성한 스크립트.

```bash
python models/train_chemberta_unfreeze.py
```

---

### `cross_eval.py` — 교차 데이터셋 평가

학습된 모델을 다른 데이터셋으로 교차 평가 (DAVIS→KIBA, KIBA→DAVIS).

```bash
python models/cross_eval.py \
    --model results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random \
    --eval_dataset kiba
```

---

### `build_3di_cache.py` — FoldSeek 3Di 토큰 캐시 구축

UniProt ID 목록을 입력받아 AlphaFold2 구조를 가져오고 FoldSeek로 3Di 토큰 추출.  
결과를 `cache/3di_tokens_*.json`에 저장. **foldseek 바이너리 필요.**

```bash
python models/build_3di_cache.py --dataset davis
```

---

### `preprocess_bindingdb.py` — BindingDB 전처리

공개 BindingDB 데이터셋을 학습용 CSV (`data/BindingDB/bindingdb_kd.csv`)로 변환.

```bash
python models/preprocess_bindingdb.py
```

---

### `plot_poster_figures.py` — 발표 시각화 생성

실험 결과(`results/`)를 읽어 발표 포스터용 그래프 생성.

```bash
python models/plot_poster_figures.py
```

---

## 학습 결과 요약

| 모델 | 데이터 | Pearson r | RMSE | 저장 위치 |
|------|--------|-----------|------|-----------|
| SaProt-650M + ft-ChemBERTa | BindingDB 80K | **0.8923** | 0.7387 | `results/SaProt-650M-bindingdb-.../` |
| 전이학습 | DAVIS 30K | **0.8677** | 0.4572 | `results/finetune_davis_.../` |
| 전이학습 | KIBA 118K | **0.8594** | 0.4268 | `results/finetune_kiba_.../` |
