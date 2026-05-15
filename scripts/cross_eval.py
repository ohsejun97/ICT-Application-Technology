"""
cross_eval.py
=============
BindingDB로 학습한 DTI 모델을 DAVIS / KIBA 테스트셋에 적용하여 교차 검증.

사용법:
  python scripts/cross_eval.py \\
      --model_dir results/SaProt-650M-bindingdb-3di-chemberta-cold_drug \\
      --eval_datasets davis kiba

출력:
  results/<model_dir>/cross_eval_davis.json
  results/<model_dir>/cross_eval_kiba.json
"""

import sys
import json
import math
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True,
                    help="학습된 모델 디렉토리 (results/ 하위)")
parser.add_argument("--eval_datasets", nargs="+", default=["davis", "kiba"],
                    choices=["davis", "kiba"],
                    help="평가할 데이터셋 (default: davis kiba)")
args = parser.parse_args()

model_dir = Path(args.model_dir)
if not model_dir.is_absolute() and not str(args.model_dir).startswith("results"):
    model_dir = Path("results") / args.model_dir

# ── 학습 설정 로드 ──────────────────────────────────────────────────────────────
with open(model_dir / "result.json") as f:
    train_cfg = json.load(f)

encoder      = train_cfg["encoder"]          # "650M" or "35M"
quant        = train_cfg["quant"]            # "none" / "8bit" / "4bit"
drug_encoder = train_cfg["drug_encoder"]     # "morgan" / "chemberta"
use_3di      = train_cfg["use_3di"]
prot_dim     = train_cfg["prot_dim"]

print("=" * 60)
print(f"  Cross-Dataset Evaluation")
print(f"  Model : {model_dir.name}")
print(f"  Protein: SaProt-{encoder} ({quant if quant != 'none' else 'FP16'}) + {'3Di' if use_3di else '#'}")
print(f"  Drug  : {drug_encoder}")
print(f"  Eval  : {args.eval_datasets}")
print("=" * 60, "\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DeepPurpose (DAVIS/KIBA 로드용) ────────────────────────────────────────────
try:
    import DeepPurpose.dataset as dp_dataset
except ImportError:
    sys.exit("❌ pip install DeepPurpose")

# ── SaProt 로드 ────────────────────────────────────────────────────────────────
SAPROT_IDS  = {"650M": "westlake-repl/SaProt_650M_AF2",
               "35M":  "westlake-repl/SaProt_35M_AF2"}
from transformers import EsmModel, EsmTokenizer

print(f"[1] SaProt-{encoder} 로드...")
model_id  = SAPROT_IDS[encoder]
tokenizer = EsmTokenizer.from_pretrained(model_id)

if quant == "8bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    saprot = EsmModel.from_pretrained(model_id, quantization_config=bnb_cfg,
                                      device_map="auto")
elif quant == "4bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_quant_type="nf4")
    saprot = EsmModel.from_pretrained(model_id, quantization_config=bnb_cfg,
                                      device_map="auto")
else:
    saprot = EsmModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)

saprot.eval()
for p in saprot.parameters():
    p.requires_grad_(False)
print(f"    ✅ SaProt 로드 완료\n")

# ── Drug encoder 로드 ──────────────────────────────────────────────────────────
if drug_encoder == "chemberta":
    from tools.chemberta_drug_encoder import ChemBERTaDrugEncoder, CHEMBERTA_DIM
    drug_enc_model = ChemBERTaDrugEncoder(device=DEVICE)
    DRUG_DIM = CHEMBERTA_DIM
    print(f"[2] ChemBERTa 로드 완료 (dim={DRUG_DIM})\n")
elif drug_encoder == "morgan":
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    DRUG_DIM = 2048
    print(f"[2] Morgan FP (2048-bit)\n")

# ── DTI Head 정의 및 가중치 로드 ───────────────────────────────────────────────
class DTIHead(nn.Module):
    def __init__(self, prot_dim, drug_dim=2048, hidden=512):
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
head.load_state_dict(torch.load(model_dir / "dti_head.pt", map_location=DEVICE, weights_only=True))
head.eval()
print(f"[3] DTI Head 가중치 로드 완료: {model_dir}/dti_head.pt\n")


def _concordance_index(y_true, y_pred, sample=3000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), min(sample, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]
    concordant = total = 0
    for i in range(len(yt)):
        for j in range(i + 1, len(yt)):
            if yt[i] == yt[j]: continue
            total += 1
            if (yt[i] > yt[j]) == (yp[i] > yp[j]): concordant += 1
    return concordant / total if total > 0 else 0.0


def evaluate_on_dataset(eval_dataset: str):
    print(f"\n{'='*50}")
    print(f"  평가 데이터셋: {eval_dataset.upper()}")
    print(f"{'='*50}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    if eval_dataset == "davis":
        X_drugs, X_targets, y = dp_dataset.load_process_DAVIS(
            path="./data", binary=False, convert_to_log=True)
    elif eval_dataset == "kiba":
        X_drugs, X_targets, y = dp_dataset.load_process_KIBA(
            path="./data", binary=False, threshold=9)
    y = np.array(y, dtype=np.float32)
    print(f"  총 {len(y):,}쌍  |  pKd {y.min():.2f}~{y.max():.2f}")

    # ── 3Di 캐시 로드 ────────────────────────────────────────────────────────
    _tokens_3di_cache = {}
    if use_3di:
        from tools.foldseek_tool import aa_seq_to_sa_tokens
        cache_path = Path(f"./cache/3di_tokens_{eval_dataset}.json")
        if not cache_path.exists():
            print(f"  ⚠️  3Di 캐시 없음: {cache_path} → '#' placeholder 사용")
        else:
            with open(cache_path) as f:
                _tokens_3di_cache = json.load(f)
            n_ok = sum(1 for v in _tokens_3di_cache.values() if v.get("status") == "ok")
            print(f"  3Di 캐시: {len(_tokens_3di_cache)}개 단백질 (ok={n_ok})")

    def aa_to_sa(seq: str) -> str:
        if not use_3di:
            return "".join(aa + "#" for aa in seq)
        h = hashlib.md5(seq.encode()).hexdigest()
        entry = _tokens_3di_cache.get(h, {})
        tokens = entry.get("tokens_3di") if entry.get("status") == "ok" else None
        if tokens is None:
            return "".join(aa + "#" for aa in seq)
        from tools.foldseek_tool import aa_seq_to_sa_tokens
        return aa_seq_to_sa_tokens(seq, tokens)

    # ── 단백질 임베딩 계산 (캐시 우선) ─────────────────────────────────────
    unique_targets = list(dict.fromkeys(X_targets))
    tgt2idx = {t: i for i, t in enumerate(unique_targets)}
    _3di_tag = "_3di" if use_3di else ""
    prot_cache_path = Path(f"./cache/prot_embs_{eval_dataset}_{encoder}_{quant}{_3di_tag}.pt")

    if prot_cache_path.exists():
        print(f"  단백질 임베딩 캐시 사용: {prot_cache_path}")
        prot_embs = torch.load(prot_cache_path, weights_only=True)
    else:
        print(f"  단백질 임베딩 사전 계산 ({len(unique_targets)}개)...")
        import time
        prot_embs = torch.zeros(len(unique_targets), prot_dim, dtype=torch.float32)
        t0 = time.time()
        with torch.no_grad():
            for i, seq in enumerate(unique_targets):
                inputs = tokenizer(aa_to_sa(seq), return_tensors="pt",
                                   truncation=True, max_length=1024, padding=False)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                out = saprot(**inputs)
                prot_embs[i] = out.last_hidden_state[0, 1:-1, :].float().mean(0).cpu()
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (len(unique_targets) - i - 1)
                    print(f"    {i+1}/{len(unique_targets)} ({elapsed:.0f}s, ETA {eta:.0f}s)")
        torch.save(prot_embs, prot_cache_path)
        print(f"    ✅ 캐시 저장: {prot_cache_path}")

    # ── 약물 임베딩 계산 ─────────────────────────────────────────────────────
    unique_drugs = list(dict.fromkeys(X_drugs))
    drug2idx = {d: i for i, d in enumerate(unique_drugs)}

    if drug_encoder == "chemberta":
        drug_embs = drug_enc_model.encode(unique_drugs, show_progress=True)  # [N, 768] CPU tensor
    elif drug_encoder == "morgan":
        drug_embs = np.zeros((len(unique_drugs), 2048), dtype=np.float32)
        for i, smi in enumerate(unique_drugs):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                drug_embs[i] = np.array(list(fp), dtype=np.float32)
        drug_embs = torch.tensor(drug_embs)

    # ── 전체 추론 ────────────────────────────────────────────────────────────
    tgt_indices  = np.array([tgt2idx[t]  for t in X_targets])
    drug_indices = np.array([drug2idx[d] for d in X_drugs])

    preds, labels = [], []
    BATCH = 512
    with torch.no_grad():
        for start in range(0, len(y), BATCH):
            end  = min(start + BATCH, len(y))
            pidx = tgt_indices[start:end]
            didx = drug_indices[start:end]
            prot = prot_embs[pidx].to(DEVICE)
            drug = drug_embs[didx].to(DEVICE)
            pred = head(prot, drug).cpu().numpy()
            preds.extend(pred)
            labels.extend(y[start:end])

    preds  = np.array(preds,  dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    r, p_val = pearsonr(preds, labels)
    rmse     = float(math.sqrt(np.mean((preds - labels) ** 2)))
    mae      = float(np.mean(np.abs(preds - labels)))
    ci       = _concordance_index(labels, preds)

    print(f"\n  결과:")
    print(f"    Pearson r : {r:.4f}")
    print(f"    RMSE      : {rmse:.4f}")
    print(f"    MAE       : {mae:.4f}")
    print(f"    CI        : {ci:.4f}")

    cross_result = {
        "model_dir":       str(model_dir),
        "eval_dataset":    eval_dataset,
        "n_pairs":         int(len(y)),
        "cross_pearson_r": round(float(r),    4),
        "cross_rmse":      round(rmse,        4),
        "cross_mae":       round(mae,         4),
        "cross_ci":        round(ci,          4),
        "cross_p_value":   float(p_val),
    }
    out_path = model_dir / f"cross_eval_{eval_dataset}.json"
    with open(out_path, "w") as f:
        json.dump(cross_result, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")
    return cross_result


# ── 실행 ───────────────────────────────────────────────────────────────────────
all_results = {}
for ds in args.eval_datasets:
    all_results[ds] = evaluate_on_dataset(ds)

print(f"\n{'='*60}")
print(f"  Cross-Eval 요약  —  {model_dir.name}")
print(f"{'='*60}")
print(f"  {'Dataset':<12} {'Pearson r':>10} {'RMSE':>8} {'CI':>8}")
print(f"  {'-'*40}")
for ds, res in all_results.items():
    print(f"  {ds.upper():<12} {res['cross_pearson_r']:>10.4f} "
          f"{res['cross_rmse']:>8.4f} {res['cross_ci']:>8.4f}")
print("=" * 60)
