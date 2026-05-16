"""
finetune_head_ft.py
===================
Phase 1h fine-tuned ChemBERTa로 DAVIS / KIBA drug embedding을 재계산한 뒤
MLP Head만 fine-tune하는 Transfer Learning 스크립트. (Phase 1i)

Phase 1g(finetune_head.py)와의 차이:
  - drug embedding을 frozen ChemBERTa 캐시가 아닌
    fine-tuned ChemBERTa(layers 4~5 업데이트) 가중치로 재계산
  - DAVIS: DeepPurpose 로더 + convert_to_log=True (Kd nM → pKd)
  - KIBA : KIBA score z-score 정규화 후 학습, 추론 시 역정규화

사용법:
  python scripts/finetune_head_ft.py --target_dataset davis
  python scripts/finetune_head_ft.py --target_dataset kiba

출력:
  results/finetune_{dataset}_{split}_from_SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random/
    ├── dti_head.pt   ← 최적 head 가중치
    └── result.json
  cache/drug_embs_{dataset}_chemberta_ft.pt  ← ft ChemBERTa drug embedding 캐시
"""

import sys, json, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# ── Tee 로거 ──────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files): self._files = files
    def write(self, data):
        for f in self._files: f.write(data); f.flush()
    def flush(self):
        for f in self._files: f.flush()
    def fileno(self): return self._files[0].fileno()
    def isatty(self): return False

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--target_dataset", required=True, choices=["davis", "kiba"])
parser.add_argument("--split",      default="random",
                    choices=["random", "cold_drug", "cold_protein"])
parser.add_argument("--ft_model_dir",
                    default="results/SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random",
                    help="Phase 1h fine-tuned 모델 디렉토리 (chemberta_ft.pt + dti_head.pt)")
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch_size", type=int,   default=128)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--patience",   type=int,   default=10)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--max_length", type=int,   default=128,
                    help="ChemBERTa tokenizer max_length")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = Path(__file__).parent.parent

# ── 로그 설정 ──────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_name = f"finetune_{args.target_dataset}_{args.split}_ft.log"
_log_fp   = open(LOG_DIR / _log_name, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fp)

ft_model_dir = ROOT / args.ft_model_dir if not Path(args.ft_model_dir).is_absolute() \
               else Path(args.ft_model_dir)

run_name = (f"finetune_{args.target_dataset}_{args.split}"
            f"_from_{ft_model_dir.name}_ft")
out_dir  = ROOT / "results" / run_name
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print(f"  Transfer Learning (Phase 1i) — ChemBERTa fine-tune 임베딩")
print(f"  Source : {ft_model_dir.name}")
print(f"  Target : {args.target_dataset.upper()} | Split: {args.split}")
print(f"  Device : {DEVICE}  |  LR: {args.lr}  |  Epochs: {args.epochs}")
print("=" * 64, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드 (DeepPurpose)
# ══════════════════════════════════════════════════════════════════════════════
print(f"[1] {args.target_dataset.upper()} 데이터 로드...")
try:
    import DeepPurpose.dataset as dp_dataset
except ImportError:
    sys.exit("❌ pip install DeepPurpose")

if args.target_dataset == "davis":
    # Kd(nM) → pKd: -log10(Kd/1e9) 자동 변환
    X_drugs, X_targets, y = dp_dataset.load_process_DAVIS(
        path=str(ROOT / "data"), binary=False, convert_to_log=True)
elif args.target_dataset == "kiba":
    X_drugs, X_targets, y = dp_dataset.load_process_KIBA(
        path=str(ROOT / "data"), binary=False, threshold=9)

y = np.array(y, dtype=np.float32)
print(f"    총 {len(y):,}쌍  |  레이블 {y.min():.2f}~{y.max():.2f}  "
      f"mean={y.mean():.2f}  std={y.std():.2f}")

# KIBA z-score 정규화 (스케일 mismatch 보정)
if args.target_dataset == "kiba":
    y_mean, y_std = float(y.mean()), float(y.std())
    y_norm = (y - y_mean) / y_std
    print(f"    KIBA z-score: mean={y_mean:.4f}  std={y_std:.4f}")
else:
    y_mean = y_std = None
    y_norm = y

# ── 데이터 분할 ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)

if args.split == "random":
    idx    = rng.permutation(len(y_norm))
    n_tr   = int(len(y_norm) * 0.70)
    n_val  = int(len(y_norm) * 0.10)
    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr:n_tr + n_val]
    te_idx  = idx[n_tr + n_val:]

elif args.split == "cold_drug":
    ud = np.array(list(dict.fromkeys(X_drugs))); rng.shuffle(ud)
    n_te = int(len(ud) * 0.20); n_v = int(len(ud) * 0.10)
    td, vd = set(ud[:n_te]), set(ud[n_te:n_te + n_v])
    Xa = np.array(X_drugs)
    te_idx  = np.where(np.isin(Xa, list(td)))[0]
    val_idx = np.where(np.isin(Xa, list(vd)))[0]
    tr_idx  = np.where(~np.isin(Xa, list(td | vd)))[0]
    print(f"    cold_drug: test={len(td)} drugs, val={len(vd)} drugs")

elif args.split == "cold_protein":
    up = np.array(list(dict.fromkeys(X_targets))); rng.shuffle(up)
    n_te = int(len(up) * 0.20); n_v = int(len(up) * 0.10)
    tp, vp = set(up[:n_te]), set(up[n_te:n_te + n_v])
    Xa = np.array(X_targets)
    te_idx  = np.where(np.isin(Xa, list(tp)))[0]
    val_idx = np.where(np.isin(Xa, list(vp)))[0]
    tr_idx  = np.where(~np.isin(Xa, list(tp | vp)))[0]
    print(f"    cold_protein: test={len(tp)} prots, val={len(vp)} prots")

print(f"    Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] SaProt 단백질 임베딩 캐시 로드 (기존 캐시 재사용)
# ══════════════════════════════════════════════════════════════════════════════
print("[2] SaProt 단백질 임베딩 캐시 로드...")
prot_cache = ROOT / "cache" / f"prot_embs_{args.target_dataset}_650M_none_3di.pt"
if not prot_cache.exists():
    sys.exit(f"❌ 단백질 임베딩 캐시 없음: {prot_cache}")

prot_embs_all = torch.load(prot_cache, weights_only=True)
unique_targets = list(dict.fromkeys(X_targets))
tgt2idx        = {t: i for i, t in enumerate(unique_targets)}
tgt_indices    = np.array([tgt2idx[t] for t in X_targets])
PROT_DIM       = prot_embs_all.shape[1]
print(f"    단백질 임베딩: {prot_embs_all.shape}  ✅\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] Fine-tuned ChemBERTa로 Drug Embedding 계산 (캐시 우선)
# ══════════════════════════════════════════════════════════════════════════════
drug_cache = ROOT / "cache" / f"drug_embs_{args.target_dataset}_chemberta_ft.pt"
unique_drugs = list(dict.fromkeys(X_drugs))
drug2idx     = {d: i for i, d in enumerate(unique_drugs)}
drug_indices = np.array([drug2idx[d] for d in X_drugs])
DRUG_DIM     = 768

if drug_cache.exists():
    print(f"[3] ft drug embedding 캐시 로드: {drug_cache.name}")
    drug_embs_all = torch.load(drug_cache, weights_only=True)
    print(f"    약물 임베딩: {drug_embs_all.shape}  ✅\n")
else:
    print(f"[3] Fine-tuned ChemBERTa로 drug embedding 계산 중...")
    CHEMBERTA_ID = "seyonec/ChemBERTa-zinc-base-v1"
    cb_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
    cb_model     = AutoModel.from_pretrained(CHEMBERTA_ID).to(DEVICE)

    # fine-tuned layers (4~5 + pooler) 가중치 병합
    ft_ckpt = ft_model_dir / "chemberta_ft.pt"
    if not ft_ckpt.exists():
        sys.exit(f"❌ chemberta_ft.pt 없음: {ft_ckpt}")
    ft_state = torch.load(ft_ckpt, map_location=DEVICE, weights_only=True)
    cur_state = cb_model.state_dict()
    cur_state.update(ft_state)
    cb_model.load_state_dict(cur_state)
    print(f"    ft 가중치 로드 완료: {len(ft_state)} tensors 업데이트")

    cb_model.eval()
    for p in cb_model.parameters():
        p.requires_grad_(False)

    embs = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(unique_drugs), batch_size):
            batch = unique_drugs[i:i + batch_size]
            inputs = cb_tokenizer(
                batch, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            out    = cb_model(**inputs)
            hidden = out.last_hidden_state
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(emb.cpu())
            if (i // batch_size + 1) % 5 == 0:
                print(f"    {i + len(batch)}/{len(unique_drugs)} 완료...", flush=True)

    drug_embs_all = torch.cat(embs, dim=0)
    torch.save(drug_embs_all, drug_cache)
    print(f"    약물 임베딩: {drug_embs_all.shape}  ✅")
    print(f"    캐시 저장: {drug_cache.name}\n")

    del cb_model
    torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# [4] DTI Head — Phase 1h head warm-start
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

head_ckpt = ft_model_dir / "dti_head.pt"
if head_ckpt.exists():
    head.load_state_dict(
        torch.load(head_ckpt, map_location=DEVICE, weights_only=True))
    print(f"[4] Head warm-start: {head_ckpt}")
else:
    print(f"[4] ⚠️  head 체크포인트 없음 → scratch")
n_params = sum(p.numel() for p in head.parameters()) / 1e6
print(f"    Head params: {n_params:.2f}M\n")

# ══════════════════════════════════════════════════════════════════════════════
# [5] Dataset / DataLoader
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
# [6] 학습 설정
# ══════════════════════════════════════════════════════════════════════════════
optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6)
criterion = nn.HuberLoss(delta=1.0)

def _concordance_index(y_true, y_pred, sample=5000, seed=42):
    rng_ = np.random.default_rng(seed)
    idx  = rng_.choice(len(y_true), min(sample, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]
    c = t = 0
    for i in range(len(yt)):
        for j in range(i + 1, len(yt)):
            if yt[i] == yt[j]: continue
            t += 1
            if (yt[i] > yt[j]) == (yp[i] > yp[j]): c += 1
    return c / t if t > 0 else 0.0

def evaluate(loader):
    head.eval()
    preds, labels = [], []
    with torch.no_grad():
        for prot, drug, label in loader:
            preds.extend(head(prot.to(DEVICE), drug.to(DEVICE)).cpu().numpy())
            labels.extend(label.numpy())
    preds  = np.array(preds,  dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    if y_std is not None:
        preds  = preds  * y_std + y_mean
        labels = labels * y_std + y_mean
    r, pval  = pearsonr(preds, labels)
    sp_r, _  = spearmanr(preds, labels)
    rmse     = float(np.sqrt(np.mean((preds - labels) ** 2)))
    mae      = float(np.mean(np.abs(preds - labels)))
    ci       = _concordance_index(labels, preds)
    ss_res   = float(np.sum((labels - preds) ** 2))
    ss_tot   = float(np.sum((labels - labels.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r), float(sp_r), rmse, mae, ci, r2, float(pval)

# ══════════════════════════════════════════════════════════════════════════════
# [7] 학습 루프
# ══════════════════════════════════════════════════════════════════════════════
best_val_r      = -1.0
best_head_state = None
patience_cnt    = 0
t_start         = time.time()

print(f"[5] 학습 시작  (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")
print(f"    {'Epoch':>5} | {'Loss':>8} | {'Val r':>7} | {'Best':>7} | {'RMSE':>7} | {'CI':>6}")
print("    " + "-" * 54)

for epoch in range(1, args.epochs + 1):
    head.train()
    train_loss = 0.0
    for prot, drug, label in train_loader:
        prot, drug, label = prot.to(DEVICE), drug.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(head(prot, drug), label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(label)
    train_loss /= len(tr_idx)
    scheduler.step()

    val_r, _, val_rmse, _, val_ci, _, _ = evaluate(val_loader)

    is_best = val_r > best_val_r
    if is_best:
        best_val_r      = val_r
        best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        patience_cnt    = 0
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
# [8] 테스트 평가 및 저장
# ══════════════════════════════════════════════════════════════════════════════
head.load_state_dict(best_head_state)
torch.save(best_head_state, out_dir / "dti_head.pt")

test_r, test_sp_r, test_rmse, test_mae, test_ci, test_r2, test_pval = evaluate(test_loader)
elapsed = time.time() - t_start

print(f"\n{'='*64}")
print(f"  결과 — {args.target_dataset.upper()} ({args.split}) [ft ChemBERTa]")
print(f"{'='*64}")
print(f"  Pearson r  : {test_r:.4f}")
print(f"  p-value    : {test_pval:.2e}")
print(f"  Spearman r : {test_sp_r:.4f}")
print(f"  R²         : {test_r2:.4f}")
print(f"  RMSE       : {test_rmse:.4f}")
print(f"  MAE        : {test_mae:.4f}")
print(f"  CI         : {test_ci:.4f}")
print(f"  학습 시간  : {elapsed:.1f}s ({elapsed/60:.1f}분)")
print(f"{'='*64}\n")

result = {
    "run_name":           run_name,
    "source_model":       str(ft_model_dir),
    "target_dataset":     args.target_dataset,
    "split":              args.split,
    "chemberta_ft":       True,
    "encoder":            "650M",
    "quant":              "none",
    "drug_encoder":       "chemberta",
    "use_3di":            True,
    "prot_dim":           PROT_DIM,
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
    "n_train":            int(len(tr_idx)),
    "n_val":              int(len(val_idx)),
    "n_test":             int(len(te_idx)),
    "timestamp":          datetime.now().isoformat(),
    "log_file":           str(LOG_DIR / _log_name),
}
if args.target_dataset == "kiba":
    result["kiba_label_mean"] = y_mean
    result["kiba_label_std"]  = y_std

with open(out_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"결과 저장: {out_dir}/result.json")
print(f"로그 저장: {LOG_DIR / _log_name}")
sys.stdout = sys.__stdout__
_log_fp.close()
