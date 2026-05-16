"""
train_chemberta_unfreeze.py
===========================
ChemBERTa 마지막 N개 레이어를 unfreeze하여 BindingDB cold_drug split으로 학습.

기존 frozen 방식과 달리 ChemBERTa 임베딩을 배치마다 on-the-fly 계산하여
약물 표현이 pKd 예측 태스크에 맞게 fine-tune됨.

SaProt은 여전히 frozen + 캐시 재사용.

사용법:
  python scripts/train_chemberta_unfreeze.py --unfreeze 2 --split cold_drug
  python scripts/train_chemberta_unfreeze.py --unfreeze 2 --split cold_protein
  python scripts/train_chemberta_unfreeze.py --unfreeze 4 --split cold_drug

출력:
  results/SaProt-650M-bindingdb-3di-chemberta-unfreeze{N}-{split}/
    ├── dti_head.pt
    ├── chemberta_ft.pt      ← fine-tuned ChemBERTa 가중치 (last N layers)
    ├── result.json
    └── training_history.csv

  logs/chemberta_unfreeze{N}_{split}.log
"""

import sys
import json
import math
import time
import argparse
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# ── Tee 로거 ──────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files: f.write(data); f.flush()
    def flush(self):
        for f in self._files:
            try: f.flush()
            except: pass
    def fileno(self):
        return self._files[0].fileno()
    def isatty(self):
        return False

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--unfreeze",    type=int,   default=2,
                    help="ChemBERTa 마지막 몇 개 레이어 unfreeze (default=2)")
parser.add_argument("--split",       default="cold_drug",
                    choices=["random", "cold_drug", "cold_protein"])
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--batch_size",  type=int,   default=64,
                    help="on-the-fly ChemBERTa 인코딩이므로 frozen 128보다 작게")
parser.add_argument("--lr_head",     type=float, default=5e-4,
                    help="MLP Head learning rate")
parser.add_argument("--lr_cb",       type=float, default=1e-5,
                    help="ChemBERTa fine-tune learning rate (작게 유지)")
parser.add_argument("--patience",    type=int,   default=10)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--max_length",  type=int,   default=128,
                    help="ChemBERTa tokenizer max_length (default=128, SMILES 대부분 <100 토큰)")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = Path(__file__).parent.parent

# ── 로그 설정 ──────────────────────────────────────────────────────────────────
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_name = f"chemberta_unfreeze{args.unfreeze}_{args.split}.log"
_log_fp   = open(LOG_DIR / _log_name, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fp)

run_name = (f"SaProt-650M-bindingdb-3di-chemberta"
            f"-unfreeze{args.unfreeze}-{args.split}")
out_dir  = ROOT / "results" / run_name
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print(f"  ChemBERTa Unfreeze Training")
print(f"  Unfreeze last {args.unfreeze} layers | Split: {args.split}")
print(f"  LR head={args.lr_head}  LR ChemBERTa={args.lr_cb}")
print(f"  Batch={args.batch_size}  Epochs={args.epochs}")
print(f"  Device: {DEVICE}")
print("=" * 64, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
print("[1] BindingDB 로드...")
bdb = pd.read_csv(ROOT / "data/BindingDB/bindingdb_kd.csv")
X_drugs   = bdb["smiles"].tolist()
X_targets = bdb["sequence"].tolist()
y         = np.array(bdb["pkd"].tolist(), dtype=np.float32)
print(f"    {len(y):,}쌍  |  pKd {y.min():.2f}~{y.max():.2f}")

rng = np.random.default_rng(args.seed)

if args.split == "random":
    idx   = rng.permutation(len(y))
    n_tr  = int(len(y) * 0.70); n_val = int(len(y) * 0.10)
    tr_idx  = idx[:n_tr]; val_idx = idx[n_tr:n_tr+n_val]; te_idx = idx[n_tr+n_val:]

elif args.split == "cold_drug":
    ud = np.array(list(dict.fromkeys(X_drugs))); rng.shuffle(ud)
    n_te = int(len(ud)*0.20); n_val_ = int(len(ud)*0.10)
    test_d = set(ud[:n_te]); val_d = set(ud[n_te:n_te+n_val_])
    Xa = np.array(X_drugs)
    te_idx  = np.where(np.isin(Xa, list(test_d)))[0]
    val_idx = np.where(np.isin(Xa, list(val_d)))[0]
    tr_idx  = np.where(~np.isin(Xa, list(test_d | val_d)))[0]
    print(f"    Cold-drug: test={len(test_d)} drugs, val={len(val_d)} drugs")

elif args.split == "cold_protein":
    up = np.array(list(dict.fromkeys(X_targets))); rng.shuffle(up)
    n_te = int(len(up)*0.20); n_val_ = int(len(up)*0.10)
    test_p = set(up[:n_te]); val_p = set(up[n_te:n_te+n_val_])
    Xa = np.array(X_targets)
    te_idx  = np.where(np.isin(Xa, list(test_p)))[0]
    val_idx = np.where(np.isin(Xa, list(val_p)))[0]
    tr_idx  = np.where(~np.isin(Xa, list(test_p | val_p)))[0]
    print(f"    Cold-protein: test={len(test_p)} prots, val={len(val_p)} prots")

print(f"    Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] SaProt 단백질 임베딩 (캐시 재사용 — frozen 유지)
# ══════════════════════════════════════════════════════════════════════════════
print("[2] SaProt 단백질 임베딩 캐시 로드...")
prot_cache = ROOT / "cache" / "prot_embs_bindingdb_650M_none_3di.pt"
if not prot_cache.exists():
    sys.exit(f"단백질 임베딩 캐시 없음: {prot_cache}\n"
             "먼저 cross_eval.py 또는 train_dti_saprot.py를 실행해 캐시 생성")

prot_embs_all = torch.load(prot_cache, weights_only=True)   # [n_prot, 1280]
unique_targets = list(dict.fromkeys(X_targets))
tgt2idx        = {t: i for i, t in enumerate(unique_targets)}
tgt_indices    = np.array([tgt2idx[t] for t in X_targets])
PROT_DIM       = prot_embs_all.shape[1]
print(f"    단백질 임베딩: {prot_embs_all.shape}  ✅\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] ChemBERTa — 마지막 N 레이어 unfreeze
# ══════════════════════════════════════════════════════════════════════════════
CHEMBERTA_ID = "seyonec/ChemBERTa-zinc-base-v1"
print(f"[3] ChemBERTa 로드 + 마지막 {args.unfreeze}레이어 unfreeze...")

cb_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
cb_model     = AutoModel.from_pretrained(CHEMBERTA_ID).to(DEVICE)

# 전체 freeze 후 마지막 N개 레이어 + pooler만 unfreeze
for p in cb_model.parameters():
    p.requires_grad_(False)

encoder_layers = cb_model.encoder.layer          # 12개 레이어 리스트
n_layers       = len(encoder_layers)             # 12
unfreeze_from  = n_layers - args.unfreeze        # 10 (last 2)

for i in range(unfreeze_from, n_layers):
    for p in encoder_layers[i].parameters():
        p.requires_grad_(True)

# pooler도 unfreeze (마지막 hidden → [CLS] projection)
if hasattr(cb_model, "pooler") and cb_model.pooler is not None:
    for p in cb_model.pooler.parameters():
        p.requires_grad_(True)

n_cb_total     = sum(p.numel() for p in cb_model.parameters()) / 1e6
n_cb_trainable = sum(p.numel() for p in cb_model.parameters() if p.requires_grad) / 1e6
print(f"    ChemBERTa: {n_cb_total:.1f}M params | trainable: {n_cb_trainable:.2f}M "
      f"(layers {unfreeze_from}~{n_layers-1})\n")

DRUG_DIM = 768

def encode_smiles_batch(smiles_list: list) -> torch.Tensor:
    """SMILES 리스트 → [B, 768] (gradient 흐름 유지)"""
    inputs = cb_tokenizer(
        smiles_list, padding=True, truncation=True,
        max_length=args.max_length, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out    = cb_model(**inputs)
    hidden = out.last_hidden_state                          # [B, L, 768]
    mask   = inputs["attention_mask"].unsqueeze(-1).float() # [B, L, 1]
    emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return emb  # [B, 768]

# ══════════════════════════════════════════════════════════════════════════════
# [4] Dataset — SMILES 문자열 인덱스 저장 (임베딩 미리 계산 X)
# ══════════════════════════════════════════════════════════════════════════════
unique_drugs  = list(dict.fromkeys(X_drugs))
drug2idx      = {d: i for i, d in enumerate(unique_drugs)}
drug_indices  = np.array([drug2idx[d] for d in X_drugs])
unique_drugs_arr = np.array(unique_drugs)

class SMILESDataset(Dataset):
    def __init__(self, indices):
        self.prot_idx  = tgt_indices[indices]
        self.drug_idx  = drug_indices[indices]
        self.labels    = y[indices]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.prot_idx[i],
                self.drug_idx[i],
                float(self.labels[i]))

def collate_fn(batch):
    prot_idxs, drug_idxs, labels = zip(*batch)
    prot_embs = prot_embs_all[list(prot_idxs)]           # [B, 1280]
    smiles    = [unique_drugs_arr[i] for i in drug_idxs] # B strings
    return prot_embs, smiles, torch.tensor(labels, dtype=torch.float32)

train_loader = DataLoader(SMILESDataset(tr_idx),  batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(SMILESDataset(val_idx), batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=collate_fn)
test_loader  = DataLoader(SMILESDataset(te_idx),  batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=collate_fn)

# ══════════════════════════════════════════════════════════════════════════════
# [5] DTI Head
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
    def forward(self, prot_emb, drug_emb):
        return self.regressor(
            torch.cat([self.prot_enc(prot_emb), self.drug_enc(drug_emb)], dim=-1)
        ).squeeze(-1)

head = DTIHead(PROT_DIM, DRUG_DIM).to(DEVICE)
n_head = sum(p.numel() for p in head.parameters()) / 1e6
print(f"[4] DTI Head: {n_head:.2f}M params\n")

# ── 옵티마이저: ChemBERTa 파라미터는 lr 1/50 ─────────────────────────────────
cb_params   = [p for p in cb_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([
    {"params": head.parameters(), "lr": args.lr_head},
    {"params": cb_params,         "lr": args.lr_cb},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-7)
criterion = nn.HuberLoss(delta=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# [6] 유틸
# ══════════════════════════════════════════════════════════════════════════════
def _concordance_index(y_true, y_pred, sample=5000, seed=0):
    rng_ = np.random.default_rng(seed)
    idx  = rng_.choice(len(y_true), min(sample, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]
    c = t = 0
    for i in range(len(yt)):
        for j in range(i+1, len(yt)):
            if yt[i] == yt[j]: continue
            t += 1
            if (yt[i] > yt[j]) == (yp[i] > yp[j]): c += 1
    return c / t if t > 0 else 0.0

def evaluate(loader):
    head.eval(); cb_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for prot, smiles, label in loader:
            prot     = prot.to(DEVICE)
            drug_emb = encode_smiles_batch(smiles)
            pred     = head(prot, drug_emb).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())
    preds  = np.array(preds,  dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    r,  pval = pearsonr(preds, labels)
    sp_r, _  = spearmanr(preds, labels)
    rmse     = float(np.sqrt(np.mean((preds - labels)**2)))
    mae      = float(np.mean(np.abs(preds - labels)))
    ci       = _concordance_index(labels, preds)
    ss_res   = float(np.sum((labels - preds)**2))
    ss_tot   = float(np.sum((labels - labels.mean())**2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r), float(sp_r), rmse, mae, ci, r2, float(pval)

# ══════════════════════════════════════════════════════════════════════════════
# [7] 학습 루프
# ══════════════════════════════════════════════════════════════════════════════
best_val_r      = -1.0
best_head_state = None
best_cb_state   = None
patience_cnt    = 0
history         = []

print(f"[5] 학습 시작")
print(f"    {'Epoch':>5} | {'Loss':>8} | {'Val r':>7} | {'Best':>7} | {'RMSE':>7} | {'CI':>6}")
print("    " + "-" * 54)

t_start = time.time()

for epoch in range(1, args.epochs + 1):
    head.train(); cb_model.train()
    train_loss = 0.0

    for prot, smiles, label in train_loader:
        prot  = prot.to(DEVICE)
        label = label.to(DEVICE)
        drug_emb = encode_smiles_batch(smiles)   # gradient 흐름
        pred  = head(prot, drug_emb)
        loss  = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + cb_params, 1.0)
        optimizer.step()
        train_loss += loss.item() * len(label)

    train_loss /= len(tr_idx)
    scheduler.step()

    val_r, val_sp, val_rmse, val_mae, val_ci, val_r2, _ = evaluate(val_loader)
    history.append({"epoch": epoch, "train_loss": train_loss,
                    "val_r": val_r, "val_rmse": val_rmse, "val_ci": val_ci})

    is_best = val_r > best_val_r
    if is_best:
        best_val_r    = val_r
        best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        best_cb_state   = {k: v.clone() for k, v in cb_model.state_dict().items()
                           if any(f"layer.{i}." in k for i in range(unfreeze_from, n_layers))
                           or "pooler" in k}
        patience_cnt  = 0
    else:
        patience_cnt += 1

    marker = " ★" if is_best else ""
    print(f"    {epoch:>5} | {train_loss:>8.4f} | {val_r:>7.4f} | "
          f"{best_val_r:>7.4f} | {val_rmse:>7.4f} | {val_ci:>6.4f}{marker}",
          flush=True)

    if patience_cnt >= args.patience:
        print(f"\n    Early stopping at epoch {epoch}")
        break

# ══════════════════════════════════════════════════════════════════════════════
# [8] 테스트
# ══════════════════════════════════════════════════════════════════════════════
head.load_state_dict(best_head_state)
# ChemBERTa best 복원
cur = cb_model.state_dict()
cur.update(best_cb_state)
cb_model.load_state_dict(cur)

test_r, test_sp_r, test_rmse, test_mae, test_ci, test_r2, test_pval = evaluate(test_loader)
elapsed = time.time() - t_start

print(f"\n{'='*64}")
print(f"  결과 — {run_name}")
print(f"{'='*64}")
print(f"  Pearson r  : {test_r:.4f}   (선형 상관계수, 1.0이 완벽)")
print(f"  p-value    : {test_pval:.2e}  (통계적 유의성)")
print(f"  Spearman r : {test_sp_r:.4f}   (순위 기반, 스케일 무관)")
print(f"  R²         : {test_r2:.4f}   (설명된 분산 비율)")
print(f"  RMSE       : {test_rmse:.4f}   (평균 예측 오차, pKd 단위)")
print(f"  MAE        : {test_mae:.4f}   (평균 절대 오차, pKd 단위)")
print(f"  CI         : {test_ci:.4f}   (순위 일치도, 0.5=랜덤)")
print(f"  학습 시간  : {elapsed:.0f}s ({elapsed/60:.1f}분)")
print(f"{'='*64}\n")

# ── 비교 기준선 출력 ──────────────────────────────────────────────────────────
baseline = {"cold_drug": 0.7083, "cold_protein": 0.6549, "random": 0.8737}
base_r   = baseline.get(args.split, 0.0)
delta    = test_r - base_r
print(f"  기준선 (frozen ChemBERTa, {args.split}): r={base_r:.4f}")
print(f"  개선량: {'+' if delta >= 0 else ''}{delta:.4f}\n")

# ── 저장 ─────────────────────────────────────────────────────────────────────
torch.save(best_head_state, out_dir / "dti_head.pt")
torch.save(best_cb_state,   out_dir / "chemberta_ft.pt")
pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

peak_vram = round(torch.cuda.max_memory_allocated() / 1024**2, 1) \
            if torch.cuda.is_available() else 0.0

result = {
    "run_name":           run_name,
    "dataset":            "bindingdb",
    "split":              args.split,
    "encoder":            "650M",
    "quant":              "none",
    "drug_encoder":       "chemberta",
    "use_3di":            True,
    "chemberta_unfreeze": args.unfreeze,
    "lr_head":            args.lr_head,
    "lr_chemberta":       args.lr_cb,
    "prot_dim":           PROT_DIM,
    "test_pearson_r":     round(test_r,     4),
    "test_p_value":       float(test_pval),
    "test_spearman_r":    round(test_sp_r,  4),
    "test_r2":            round(test_r2,    4),
    "test_rmse":          round(test_rmse,  4),
    "test_mae":           round(test_mae,   4),
    "test_ci":            round(test_ci,    4),
    "best_val_r":         round(best_val_r, 4),
    "baseline_r":         base_r,
    "delta_r":            round(delta,      4),
    "epochs_trained":     epoch,
    "train_time_sec":     round(elapsed,    1),
    "n_train":            int(len(tr_idx)),
    "n_val":              int(len(val_idx)),
    "n_test":             int(len(te_idx)),
    "peak_vram_mb":       peak_vram,
    "timestamp":          datetime.now().isoformat(),
    "log_file":           str(LOG_DIR / _log_name),
}
with open(out_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"결과 저장: {out_dir}/result.json")
print(f"로그 저장: {LOG_DIR / _log_name}")
sys.stdout = sys.__stdout__
_log_fp.close()
