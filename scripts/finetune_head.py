"""
finetune_head.py
================
BindingDB로 학습된 DTI MLP Head를 DAVIS / KIBA 데이터셋에 Transfer Learning.

SaProt + ChemBERTa 임베딩은 이미 캐시되어 있으므로 추가 GPU 로드 없이
MLP Head만 빠르게 fine-tune합니다. (~1–2분)

사용법:
  python scripts/finetune_head.py \\
      --source_model results/SaProt-650M-bindingdb-3di-chemberta \\
      --target_dataset davis

  python scripts/finetune_head.py \\
      --source_model results/SaProt-650M-bindingdb-3di-chemberta \\
      --target_dataset kiba

  # cold split 평가
  python scripts/finetune_head.py \\
      --source_model results/SaProt-650M-bindingdb-3di-chemberta \\
      --target_dataset davis --split cold_drug

출력:
  results/finetune_{target}_{split}_from_{source}/
    ├── dti_head.pt        ← fine-tuned head 가중치
    └── result.json        ← 평가 결과 (Pearson r, RMSE, CI, Spearman r)
"""

import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── stdout + 파일 동시 출력 ────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self._files:
            f.flush()
    def fileno(self):
        # wget 등 내부에서 fileno() 호출 시 원본 stdout 기준으로 반환
        return self._files[0].fileno()

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source_model", default="results/SaProt-650M-bindingdb-3di-chemberta",
                    help="BindingDB로 학습된 모델 디렉토리 (Head 가중치 warm-start용)")
parser.add_argument("--target_dataset", required=True, choices=["davis", "kiba"],
                    help="Fine-tune 대상 데이터셋")
parser.add_argument("--split", default="random",
                    choices=["random", "cold_drug", "cold_protein"],
                    help="Data split 전략 (default: random)")
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch_size", type=int,   default=128)
parser.add_argument("--lr",         type=float, default=3e-4,
                    help="Learning rate (warm-start이므로 1e-3보다 낮게)")
parser.add_argument("--patience",   type=int,   default=10)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--no_warmstart", action="store_true",
                    help="BindingDB head 가중치 사용 안 함 (scratch부터 학습)")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).parent.parent

# ── 로그 파일 설정 ────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_name = f"finetune_{args.target_dataset}_{args.split}.log"
_log_file = open(LOG_DIR / _log_name, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)

# ── source model 설정 로드 ─────────────────────────────────────────────────────
source_dir = Path(args.source_model)
if not source_dir.is_absolute() and not str(args.source_model).startswith("results"):
    source_dir = ROOT / "results" / args.source_model

if not source_dir.exists():
    sys.exit(f"❌ source_model 디렉토리 없음: {source_dir}")

with open(source_dir / "result.json") as f:
    src_cfg = json.load(f)

encoder      = src_cfg["encoder"]       # "650M"
quant        = src_cfg["quant"]         # "none"
prot_dim     = src_cfg["prot_dim"]      # 1280
drug_encoder = src_cfg["drug_encoder"]  # "chemberta"
use_3di      = src_cfg["use_3di"]       # True

DRUG_DIM = 768 if drug_encoder == "chemberta" else 2048

run_name = (f"finetune_{args.target_dataset}_{args.split}"
            f"_from_{source_dir.name}")
out_dir  = ROOT / "results" / run_name
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 62)
print(f"  DTI Head Fine-tuning (Transfer Learning)")
print(f"  Source : {source_dir.name}")
print(f"  Target : {args.target_dataset.upper()} | Split: {args.split}")
print(f"  Device : {DEVICE}")
print(f"  LR: {args.lr}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
print(f"  Warm-start: {'OFF (scratch)' if args.no_warmstart else 'ON (BindingDB head)'}")
print("=" * 62, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
print(f"[1] {args.target_dataset.upper()} 데이터 로드...")

try:
    import DeepPurpose.dataset as dp_dataset
except ImportError:
    sys.exit("❌ pip install DeepPurpose")

if args.target_dataset == "davis":
    X_drugs, X_targets, y = dp_dataset.load_process_DAVIS(
        path=str(ROOT / "data"), binary=False, convert_to_log=True)
elif args.target_dataset == "kiba":
    X_drugs, X_targets, y = dp_dataset.load_process_KIBA(
        path=str(ROOT / "data"), binary=False, threshold=9)

y = np.array(y, dtype=np.float32)
print(f"    총 {len(y):,}쌍  |  레이블 범위: {y.min():.2f}~{y.max():.2f}  "
      f"mean={y.mean():.2f}  std={y.std():.2f}")

# KIBA 레이블 정규화 (선택적): KIBA score는 스케일이 달라 정규화 후 학습
# 학습/추론 후 역정규화로 실제 KIBA score로 복원
if args.target_dataset == "kiba":
    y_mean = y.mean()
    y_std  = y.std()
    y_norm = (y - y_mean) / y_std
    print(f"    KIBA 정규화: (x - {y_mean:.4f}) / {y_std:.4f}")
else:
    y_mean, y_std = None, None
    y_norm = y

# ── 데이터 분할 ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)

if args.split == "random":
    idx   = rng.permutation(len(y_norm))
    n_tr  = int(len(y_norm) * 0.70)
    n_val = int(len(y_norm) * 0.10)
    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr:n_tr + n_val]
    te_idx  = idx[n_tr + n_val:]

elif args.split == "cold_drug":
    unique_drugs = np.array(list(dict.fromkeys(X_drugs)))
    rng.shuffle(unique_drugs)
    n_te   = int(len(unique_drugs) * 0.20)
    n_val_ = int(len(unique_drugs) * 0.10)
    test_drugs = set(unique_drugs[:n_te])
    val_drugs  = set(unique_drugs[n_te:n_te + n_val_])
    X_drugs_arr = np.array(X_drugs)
    te_idx  = np.where(np.isin(X_drugs_arr, list(test_drugs)))[0]
    val_idx = np.where(np.isin(X_drugs_arr, list(val_drugs)))[0]
    tr_idx  = np.where(~np.isin(X_drugs_arr, list(test_drugs | val_drugs)))[0]
    print(f"    Cold-drug: test={len(test_drugs)} drugs, val={len(val_drugs)} drugs")

elif args.split == "cold_protein":
    unique_prots = np.array(list(dict.fromkeys(X_targets)))
    rng.shuffle(unique_prots)
    n_te   = int(len(unique_prots) * 0.20)
    n_val_ = int(len(unique_prots) * 0.10)
    test_prots = set(unique_prots[:n_te])
    val_prots  = set(unique_prots[n_te:n_te + n_val_])
    X_targets_arr = np.array(X_targets)
    te_idx  = np.where(np.isin(X_targets_arr, list(test_prots)))[0]
    val_idx = np.where(np.isin(X_targets_arr, list(val_prots)))[0]
    tr_idx  = np.where(~np.isin(X_targets_arr, list(test_prots | val_prots)))[0]
    print(f"    Cold-protein: test={len(test_prots)} prots, val={len(val_prots)} prots")

print(f"    Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] 캐시된 임베딩 로드 (SaProt / ChemBERTa 재로드 불필요)
# ══════════════════════════════════════════════════════════════════════════════
print("[2] 캐시된 임베딩 로드...")

_3di_tag     = "_3di" if use_3di else ""
prot_cache   = ROOT / "cache" / f"prot_embs_{args.target_dataset}_{encoder}_{quant}{_3di_tag}.pt"
drug_cache   = ROOT / "cache" / f"drug_embs_{args.target_dataset}_{drug_encoder}.pt"

if not prot_cache.exists():
    sys.exit(f"❌ 단백질 임베딩 캐시 없음: {prot_cache}\n"
             f"   먼저 cross_eval.py를 실행해 캐시를 생성하세요.")
if not drug_cache.exists():
    sys.exit(f"❌ 약물 임베딩 캐시 없음: {drug_cache}\n"
             f"   먼저 cross_eval.py를 실행해 캐시를 생성하세요.")

prot_embs_all = torch.load(prot_cache, weights_only=True)  # [n_unique_prot, prot_dim]
drug_embs_all = torch.load(drug_cache, weights_only=True)  # [n_unique_drug, drug_dim]
print(f"    단백질 임베딩: {prot_embs_all.shape}")
print(f"    약물   임베딩: {drug_embs_all.shape}")

# unique 순서 맞추기 (cross_eval.py와 동일한 방식)
unique_targets = list(dict.fromkeys(X_targets))
unique_drugs   = list(dict.fromkeys(X_drugs))
tgt2idx  = {t: i for i, t in enumerate(unique_targets)}
drug2idx = {d: i for i, d in enumerate(unique_drugs)}
tgt_indices  = np.array([tgt2idx[t]  for t in X_targets])
drug_indices = np.array([drug2idx[d] for d in X_drugs])

print(f"    ✅ 임베딩 로드 완료  (단백질 {len(unique_targets)}개, 약물 {len(unique_drugs)}개)\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] DTI Head 정의 및 가중치 초기화
# ══════════════════════════════════════════════════════════════════════════════
class DTIHead(nn.Module):
    def __init__(self, prot_dim, drug_dim=768, hidden=512):
        super().__init__()
        self.prot_enc = nn.Sequential(
            nn.Linear(prot_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 256), nn.GELU(),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, 256), nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1),
        )
    def forward(self, prot_emb, drug_fp):
        return self.regressor(
            torch.cat([self.prot_enc(prot_emb), self.drug_enc(drug_fp)], dim=-1)
        ).squeeze(-1)

head = DTIHead(prot_dim, drug_dim=DRUG_DIM).to(DEVICE)

source_ckpt = source_dir / "dti_head.pt"
if not args.no_warmstart and source_ckpt.exists():
    head.load_state_dict(
        torch.load(source_ckpt, map_location=DEVICE, weights_only=True)
    )
    print(f"[3] Head 가중치 warm-start: {source_ckpt}")
elif args.no_warmstart:
    print(f"[3] Head scratch 초기화 (--no_warmstart)")
else:
    print(f"[3] ⚠️  source head 없음 → scratch 초기화: {source_ckpt}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# [4] Dataset & DataLoader
# ══════════════════════════════════════════════════════════════════════════════
class EmbDataset(Dataset):
    def __init__(self, indices):
        self.p_idx  = tgt_indices[indices]
        self.d_idx  = drug_indices[indices]
        self.labels = y_norm[indices]

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return (prot_embs_all[self.p_idx[i]],
                drug_embs_all[self.d_idx[i]],
                torch.tensor(self.labels[i], dtype=torch.float32))

train_loader = DataLoader(EmbDataset(tr_idx),  batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(EmbDataset(val_idx), batch_size=512,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(EmbDataset(te_idx),  batch_size=512,
                          shuffle=False, num_workers=2, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════════
# [5] 학습 설정
# ══════════════════════════════════════════════════════════════════════════════
optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6)
criterion = nn.HuberLoss(delta=1.0)

def _concordance_index(y_true, y_pred, sample=5000, seed=42):
    rng_ = np.random.default_rng(seed)
    idx  = rng_.choice(len(y_true), min(sample, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]
    concordant = total = 0
    for i in range(len(yt)):
        for j in range(i + 1, len(yt)):
            if yt[i] == yt[j]: continue
            total += 1
            if (yt[i] > yt[j]) == (yp[i] > yp[j]): concordant += 1
    return concordant / total if total > 0 else 0.0

def evaluate(loader):
    head.eval()
    preds, labels = [], []
    with torch.no_grad():
        for prot, drug, label in loader:
            prot, drug = prot.to(DEVICE), drug.to(DEVICE)
            pred = head(prot, drug).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())
    preds  = np.array(preds,  dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    # 역정규화 (KIBA)
    if y_std is not None:
        preds_raw  = preds  * y_std + y_mean
        labels_raw = labels * y_std + y_mean
    else:
        preds_raw  = preds
        labels_raw = labels
    r, p_val = pearsonr(preds_raw, labels_raw)
    sp_r, _  = spearmanr(preds_raw, labels_raw)
    rmse     = float(np.sqrt(np.mean((preds_raw - labels_raw) ** 2)))
    mae      = float(np.mean(np.abs(preds_raw - labels_raw)))
    ci       = _concordance_index(labels_raw, preds_raw)
    ss_res   = float(np.sum((labels_raw - preds_raw) ** 2))
    ss_tot   = float(np.sum((labels_raw - labels_raw.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r), float(sp_r), rmse, mae, ci, r2, float(p_val)

# ══════════════════════════════════════════════════════════════════════════════
# [6] 학습 루프
# ══════════════════════════════════════════════════════════════════════════════
import time

best_val_r       = -1.0
best_head_state  = None
patience_cnt     = 0
history          = []

print(f"[4] 학습 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")
print(f"    {'Epoch':>5} | {'Train Loss':>10} | {'Val r':>7} | {'Best':>7} | {'RMSE':>7} | {'CI':>6}")
print("    " + "-" * 58)

t_start = time.time()

for epoch in range(1, args.epochs + 1):
    head.train()
    train_loss = 0.0
    for prot, drug, label in train_loader:
        prot, drug, label = prot.to(DEVICE), drug.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = head(prot, drug)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(label)
    train_loss /= len(tr_idx)
    scheduler.step()

    val_r, val_sp_r, val_rmse, val_mae, val_ci, val_r2, _ = evaluate(val_loader)
    history.append({"epoch": epoch, "train_loss": train_loss,
                    "val_r": val_r, "val_rmse": val_rmse})

    is_best = val_r > best_val_r
    if is_best:
        best_val_r      = val_r
        best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        patience_cnt    = 0
    else:
        patience_cnt += 1

    marker = " ★" if is_best else ""
    print(f"    {epoch:>5} | {train_loss:>10.4f} | {val_r:>7.4f} | "
          f"{best_val_r:>7.4f} | {val_rmse:>7.4f} | {val_ci:>6.4f}{marker}",
          flush=True)

    if patience_cnt >= args.patience:
        print(f"\n    Early stopping at epoch {epoch} (patience={args.patience})")
        break

# ══════════════════════════════════════════════════════════════════════════════
# [7] 테스트 평가
# ══════════════════════════════════════════════════════════════════════════════
head.load_state_dict(best_head_state)
torch.save(best_head_state, out_dir / "dti_head.pt")
print(f"\n[5] Best head 저장: {out_dir}/dti_head.pt")

test_r, test_sp_r, test_rmse, test_mae, test_ci, test_r2, test_pval = evaluate(test_loader)
elapsed = time.time() - t_start

print(f"\n{'='*62}")
print(f"  Fine-tune 결과 — {args.target_dataset.upper()} ({args.split})")
print(f"{'='*62}")
print(f"  Pearson r  : {test_r:.4f}   (선형 상관계수, 1.0이 완벽 예측)")
print(f"  p-value    : {test_pval:.2e}  (통계적 유의성, <0.05면 유의)")
print(f"  Spearman r : {test_sp_r:.4f}   (순위 기반 상관계수, 스케일 무관)")
print(f"  R²         : {test_r2:.4f}   (설명된 분산 비율, 1.0이 완벽)")
print(f"  RMSE       : {test_rmse:.4f}   (평균 예측 오차 pKd 단위)")
print(f"  MAE        : {test_mae:.4f}   (평균 절대 오차 pKd 단위)")
print(f"  CI         : {test_ci:.4f}   (결합력 순위 일치도, 0.5=랜덤)")
print(f"  학습 시간  : {elapsed:.1f}s")
print(f"{'='*62}\n")

# ── 결과 저장 ──────────────────────────────────────────────────────────────────
result = {
    "run_name":           run_name,
    "source_model":       str(source_dir),
    "target_dataset":     args.target_dataset,
    "split":              args.split,
    "warm_start":         not args.no_warmstart,
    "encoder":            encoder,
    "quant":              quant,
    "drug_encoder":       drug_encoder,
    "use_3di":            use_3di,
    "prot_dim":           prot_dim,
    "lr":                 args.lr,
    "epochs_trained":     epoch,
    "best_val_r":         round(best_val_r,  4),
    "test_pearson_r":     round(test_r,      4),
    "test_p_value":       float(test_pval),
    "test_spearman_r":    round(test_sp_r,   4),
    "test_r2":            round(test_r2,     4),
    "test_rmse":          round(test_rmse,   4),
    "test_mae":           round(test_mae,    4),
    "test_ci":            round(test_ci,     4),
    "train_time_sec":     round(elapsed,     1),
    "log_file":           str(LOG_DIR / _log_name),
    "n_train":            int(len(tr_idx)),
    "n_val":              int(len(val_idx)),
    "n_test":             int(len(te_idx)),
    "timestamp":          datetime.now().isoformat(),
}

if args.target_dataset == "kiba":
    result["kiba_label_mean"] = float(y_mean)
    result["kiba_label_std"]  = float(y_std)

with open(out_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"결과 저장: {out_dir}/result.json")
print(f"로그 저장: {LOG_DIR / _log_name}")
sys.stdout = sys.__stdout__   # 원본 복원 후 파일 닫기
_log_file.close()
