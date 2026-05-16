"""
train_dti_saprot.py  (v2 — LoRA 지원)
======================================
SaProt + DTI 회귀 헤드를 DAVIS 연속 pKd 데이터로 학습 및 평가

모드:
  frozen  : SaProt 완전 고정, 임베딩 캐시 사용 → DTI 헤드만 학습 (빠름, ~1분)
  LoRA    : SaProt 어텐션에 rank-16 어댑터 삽입 → SaProt + 헤드 함께 학습 (느림, 수 시간)

사용법:
  python train_dti_saprot.py --encoder 650M                       # frozen 기준
  python train_dti_saprot.py --encoder 35M                        # frozen 경량
  python train_dti_saprot.py --encoder 650M --quant 4bit          # frozen 4bit
  python train_dti_saprot.py --encoder 650M --lora                # LoRA 기준
  python train_dti_saprot.py --encoder 35M  --lora                # LoRA 경량  ← 핵심 실험
  python train_dti_saprot.py --encoder 650M --quant 4bit --lora   # LoRA 4bit
"""

import os, sys, time, json, argparse, math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("❌ pip install rdkit")

try:
    import DeepPurpose.dataset as dp_dataset
except ImportError:
    sys.exit("❌ pip install DeepPurpose")

# ══════════════════════════════════════════════════════════════════════════════
# 인자 파싱
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="SaProt DTI Trainer")
parser.add_argument("--dataset",    default="davis",
                    choices=["davis", "kiba", "bindingdb", "davis+bindingdb"],
                    help="Training dataset: davis, kiba, bindingdb, davis+bindingdb")
parser.add_argument("--encoder",    default="650M", choices=["650M", "35M"])
parser.add_argument("--quant",      default="none", choices=["none", "8bit", "4bit"])
parser.add_argument("--lora",       action="store_true")
parser.add_argument("--lora_r",     type=int,   default=16)
parser.add_argument("--lora_alpha", type=int,   default=32)
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch_size", type=int,   default=0,
                    help="0=auto (frozen:128, LoRA 650M:8, LoRA 35M:32)")
parser.add_argument("--lr",         type=float, default=0.0,
                    help="0=auto (frozen:1e-3, LoRA:5e-5)")
parser.add_argument("--patience",   type=int,   default=10)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--use_3di",      action="store_true",
                    help="Use FoldSeek 3Di structural tokens instead of '#' placeholder")
parser.add_argument("--drug_encoder", default="morgan", choices=["morgan", "gnn", "chemberta"],
                    help="Drug encoder: morgan=Morgan FP (fixed), gnn=MPNN+MorganFP (trainable), chemberta=ChemBERTa (frozen)")
parser.add_argument("--gnn_warmup_epochs", type=int, default=10,
                    help="GNN 2단계 학습: 이 에포크까지 GNN 동결 후 해동 (default=10)")
parser.add_argument("--split", default="random",
                    choices=["random", "cold_drug", "cold_protein"],
                    help="Data split strategy: random (default), cold_drug, cold_protein")
args = parser.parse_args()

# 자동 기본값
if args.batch_size == 0:
    if args.lora:
        args.batch_size = 8 if args.encoder == "650M" else 32
    elif args.drug_encoder == "gnn":
        args.batch_size = 32   # GNN: dense adj matrix → VRAM 절약
    else:
        args.batch_size = 128  # morgan / chemberta (캐시 기반, VRAM 여유)
if args.lr == 0.0:
    args.lr = 5e-5 if args.lora else 1e-3

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAPROT_IDS  = {"650M": "westlake-repl/SaProt_650M_AF2",
               "35M":  "westlake-repl/SaProt_35M_AF2"}
SAPROT_DIMS = {"650M": 1280, "35M": 480}

run_name = f"SaProt-{args.encoder}"
if args.quant != "none":        run_name += f"-{args.quant}"
if args.lora:                   run_name += "-lora"
run_name += f"-{args.dataset}"
if args.use_3di:                run_name += "-3di"
if args.drug_encoder == "gnn":        run_name += "-gnn"
if args.drug_encoder == "chemberta":  run_name += "-chemberta"
if args.split != "random":            run_name += f"-{args.split}"

print("=" * 60)
print(f"  DTI Training — {run_name}")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Device: {DEVICE} | Dataset: {args.dataset.upper()} | "
      f"Encoder: {args.encoder} | Quant: {args.quant} | 3Di: {args.use_3di}")
print(f"  batch={args.batch_size} | lr={args.lr}")
print("=" * 60, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
if args.dataset == "davis":
    print("[1] Loading DAVIS (continuous pKd)...")
    X_drugs, X_targets, y = dp_dataset.load_process_DAVIS(
        path="./data", binary=False, convert_to_log=True
    )
elif args.dataset == "kiba":
    print("[1] Loading KIBA (KIBA score, regression)...")
    X_drugs, X_targets, y = dp_dataset.load_process_KIBA(
        path="./data", binary=False, threshold=9
    )
elif args.dataset == "bindingdb":
    print("[1] Loading BindingDB from preprocessed CSV...")
    _bdb = pd.read_csv("./data/BindingDB/bindingdb_kd.csv")
    X_drugs   = _bdb["smiles"].tolist()
    X_targets = _bdb["sequence"].tolist()
    y         = _bdb["pkd"].tolist()
elif args.dataset == "davis+bindingdb":
    print("[1] Loading DAVIS + BindingDB (combined, pKd)...")
    X_d_davis, X_t_davis, y_davis = dp_dataset.load_process_DAVIS(
        path="./data", binary=False, convert_to_log=True
    )
    _bdb = pd.read_csv("./data/BindingDB/bindingdb_kd.csv")
    X_d_bdb   = _bdb["smiles"].tolist()
    X_t_bdb   = _bdb["sequence"].tolist()
    y_bdb     = _bdb["pkd"].tolist()
    X_drugs   = X_d_davis + X_d_bdb
    X_targets = X_t_davis + X_t_bdb
    y         = y_davis   + y_bdb
    print(f"    DAVIS: {len(y_davis):,} pairs  +  BindingDB: {len(y_bdb):,} pairs")

y = np.array(y, dtype=np.float32)
print(f"    Total: {len(y):,} pairs  |  target: {y.min():.2f} ~ {y.max():.2f}")

rng = np.random.default_rng(args.seed)

if args.split == "random":
    idx    = rng.permutation(len(y))
    n_tr   = int(len(y) * 0.70)
    n_val  = int(len(y) * 0.10)
    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr:n_tr + n_val]
    te_idx  = idx[n_tr + n_val:]

elif args.split == "cold_drug":
    unique_drugs = np.array(list(dict.fromkeys(X_drugs)))
    rng.shuffle(unique_drugs)
    n_te  = int(len(unique_drugs) * 0.20)
    n_val_ = int(len(unique_drugs) * 0.10)
    test_drugs = set(unique_drugs[:n_te])
    val_drugs  = set(unique_drugs[n_te:n_te + n_val_])
    X_drugs_arr = np.array(X_drugs)
    te_idx  = np.where(np.isin(X_drugs_arr, list(test_drugs)))[0]
    val_idx = np.where(np.isin(X_drugs_arr, list(val_drugs)))[0]
    tr_idx  = np.where(~np.isin(X_drugs_arr, list(test_drugs | val_drugs)))[0]
    print(f"    Cold-drug: {len(test_drugs)} test drugs / {len(val_drugs)} val drugs / "
          f"{len(unique_drugs)-len(test_drugs)-len(val_drugs)} train drugs")

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
    print(f"    Cold-protein: {len(test_prots)} test proteins / {len(val_prots)} val proteins / "
          f"{len(unique_prots)-len(test_prots)-len(val_prots)} train proteins")

print(f"    Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] SaProt 로드
# ══════════════════════════════════════════════════════════════════════════════
print(f"[2] SaProt-{args.encoder} 로드 (quant={args.quant}, lora={args.lora})...")
model_id  = SAPROT_IDS[args.encoder]
prot_dim  = SAPROT_DIMS[args.encoder]
tokenizer = EsmTokenizer.from_pretrained(model_id)

if args.quant == "4bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    saprot = EsmModel.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map="auto", low_cpu_mem_usage=True, add_pooling_layer=False,
    )
elif args.quant == "8bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    saprot = EsmModel.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map="auto", low_cpu_mem_usage=True, add_pooling_layer=False,
    )
else:
    saprot = EsmModel.from_pretrained(
        model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    ).to(DEVICE)

# ── LoRA 적용 ─────────────────────────────────────────────────────────────────
if args.lora:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        # FEATURE_EXTRACTION: 분류 헤드 없는 인코더 전용
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    saprot = get_peft_model(saprot, lora_cfg)
    saprot.print_trainable_parameters()
    saprot.train()
    # 650M에서 VRAM 절약을 위한 gradient checkpointing
    if args.encoder == "650M" and args.quant == "none":
        saprot.enable_input_require_grads()
        saprot.base_model.model.gradient_checkpointing_enable()
    saprot = saprot.to(DEVICE)
else:
    saprot.eval()
    for p in saprot.parameters():
        p.requires_grad_(False)

n_params     = sum(p.numel() for p in saprot.parameters()) / 1e6
n_trainable  = sum(p.numel() for p in saprot.parameters() if p.requires_grad) / 1e6
print(f"    ✅ {n_params:.0f}M params total | {n_trainable:.2f}M trainable\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] 단백질 처리 — frozen: 임베딩 캐시 / LoRA: 토큰 사전 계산
# ══════════════════════════════════════════════════════════════════════════════
# ── 3Di 토큰 캐시 로드 (--use_3di) ───────────────────────────────────────────
import hashlib as _hashlib

_tokens_3di_cache: dict = {}
if args.use_3di:
    from tools.foldseek_tool import aa_seq_to_sa_tokens, check_foldseek
    if not check_foldseek():
        sys.exit("❌ foldseek not found in PATH. Install from https://github.com/steineggerlab/foldseek/releases")
    _cache_3di_path = Path(f"./cache/3di_tokens_{args.dataset}.json")
    if not _cache_3di_path.exists():
        sys.exit(f"❌ 3Di 캐시 없음: {_cache_3di_path}\n"
                 f"   먼저 실행: python scripts/build_3di_cache.py --dataset {args.dataset}")
    with open(_cache_3di_path) as _f:
        _tokens_3di_cache = json.load(_f)
    _n_ok = sum(1 for v in _tokens_3di_cache.values() if v.get("status") == "ok")
    print(f"    3Di 캐시 로드: {len(_tokens_3di_cache)}개 단백질 (ok={_n_ok})\n")


def aa_to_sa(seq: str) -> str:
    if not args.use_3di:
        return "".join(aa + "#" for aa in seq)
    h = _hashlib.md5(seq.encode()).hexdigest()
    entry = _tokens_3di_cache.get(h, {})
    tokens_3di = entry.get("tokens_3di") if entry.get("status") == "ok" else None
    return aa_seq_to_sa_tokens(seq, tokens_3di)


unique_targets = list(dict.fromkeys(X_targets))
tgt2idx        = {t: i for i, t in enumerate(unique_targets)}
tgt_indices    = np.array([tgt2idx[t] for t in X_targets])

if not args.lora:
    # ── frozen 모드: 임베딩 캐시 ─────────────────────────────────────────────
    _3di_tag   = "_3di" if args.use_3di else ""
    cache_path = Path(f"./cache/prot_embs_{args.dataset}_{args.encoder}_{args.quant}{_3di_tag}.pt")
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        print(f"[3] 단백질 임베딩 캐시 로드: {cache_path}")
        prot_embs = torch.load(cache_path, weights_only=True)
    else:
        print(f"[3] 단백질 임베딩 사전 계산 ({len(unique_targets)}개)...")
        t0 = time.time()
        prot_embs = torch.zeros(len(unique_targets), prot_dim, dtype=torch.float32)
        with torch.no_grad():
            for i, seq in enumerate(unique_targets):
                inputs = tokenizer(aa_to_sa(seq), return_tensors="pt",
                                   truncation=True, max_length=1024, padding=False)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                out    = saprot(**inputs)
                hidden = out.last_hidden_state[0, 1:-1, :].float()
                prot_embs[i] = hidden.mean(0).cpu()
                if (i + 1) % 50 == 0 or i == len(unique_targets) - 1:
                    elapsed = time.time() - t0
                    eta     = elapsed / (i + 1) * (len(unique_targets) - i - 1)
                    print(f"    {i+1}/{len(unique_targets)}  "
                          f"({elapsed:.0f}s 경과, ETA {eta:.0f}s)")
        torch.save(prot_embs, cache_path)
        print(f"    ✅ 캐시 저장: {cache_path}\n")

else:
    # ── LoRA 모드: 토큰 사전 계산 (실제 임베딩은 학습 중 계산) ─────────────
    print(f"[3] 단백질 토큰 사전 계산 ({len(unique_targets)}개)...")
    all_input_ids  = []
    all_attn_masks = []
    MAX_LEN = 512   # 650M + LoRA VRAM 제약
    for seq in unique_targets:
        enc = tokenizer(aa_to_sa(seq), return_tensors="pt",
                        truncation=True, max_length=MAX_LEN, padding=False)
        all_input_ids.append(enc["input_ids"][0])
        all_attn_masks.append(enc["attention_mask"][0])
    print(f"    ✅ 토큰화 완료 (max_len={MAX_LEN})\n")

# ══════════════════════════════════════════════════════════════════════════════
# [4] 약물 인코딩 (Morgan FP 또는 GNN 그래프 변환)
# ══════════════════════════════════════════════════════════════════════════════
unique_drugs  = list(dict.fromkeys(X_drugs))
drug2idx      = {d: i for i, d in enumerate(unique_drugs)}
drug_indices  = np.array([drug2idx[d] for d in X_drugs])

if args.drug_encoder == "morgan":
    print("[4] 약물 Morgan FP 계산 (2048-bit, radius=2)...")

    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(list(fp), dtype=np.float32)

    drug_fps  = np.zeros((len(unique_drugs), 2048), dtype=np.float32)
    n_invalid = 0
    for i, smi in enumerate(unique_drugs):
        fp = smiles_to_fp(smi)
        if fp is not None: drug_fps[i] = fp
        else: n_invalid += 1
    drug_fps = torch.tensor(drug_fps)
    print(f"    ✅ {len(unique_drugs)}개 약물 | 유효하지 않은 SMILES: {n_invalid}개\n")
    DRUG_DIM  = 2048
    gnn_encoder = None

elif args.drug_encoder == "chemberta":
    from tools.chemberta_drug_encoder import ChemBERTaDrugEncoder, CHEMBERTA_DIM

    _cache_path = Path(f"./cache/drug_embs_{args.dataset}_chemberta.pt")
    if _cache_path.exists():
        print(f"[4] ChemBERTa 약물 임베딩 캐시 로드: {_cache_path}", flush=True)
        drug_fps = torch.load(_cache_path, weights_only=True)
    else:
        print(f"[4] ChemBERTa 약물 임베딩 계산 ({len(unique_drugs)}개 약물)...", flush=True)
        _cb_encoder = ChemBERTaDrugEncoder(device=str(DEVICE))
        drug_fps = _cb_encoder.encode(list(unique_drugs), batch_size=64)
        torch.save(drug_fps, _cache_path)
        del _cb_encoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"    ✅ 캐시 저장: {_cache_path}", flush=True)

    print(f"    ✅ {len(unique_drugs)}개 약물 | ChemBERTa 임베딩: {drug_fps.shape}\n", flush=True)
    DRUG_DIM    = CHEMBERTA_DIM  # 768
    gnn_encoder = None

else:  # gnn  →  Morgan FP + GNN concat
    print("[4] 약물 인코딩 (Morgan FP 2048-dim + GNN 256-dim concat)...")
    from tools.gnn_drug_encoder import (
        smiles_to_graph, collate_graphs, GNNDrugEncoder, GNN_OUT_DIM,
    )

    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(list(fp), dtype=np.float32)

    # Morgan FP (고정)
    drug_fps  = np.zeros((len(unique_drugs), 2048), dtype=np.float32)
    for i, smi in enumerate(unique_drugs):
        fp = smiles_to_fp(smi)
        if fp is not None: drug_fps[i] = fp
    drug_fps = torch.tensor(drug_fps)

    # GNN 그래프
    drug_graphs = []
    n_invalid   = 0
    for smi in unique_drugs:
        g = smiles_to_graph(smi)
        if g is None:
            drug_graphs.append(None)
            n_invalid += 1
        else:
            drug_graphs.append(g)
    _valid_g = next(g for g in drug_graphs if g is not None)
    drug_graphs = [g if g is not None else _valid_g for g in drug_graphs]

    print(f"    ✅ {len(unique_drugs)}개 약물 | 그래프 변환 실패: {n_invalid}개")
    DRUG_DIM    = 2048 + GNN_OUT_DIM   # 2048 + 256 = 2304
    gnn_encoder = GNNDrugEncoder().to(DEVICE)
    n_gnn = sum(p.numel() for p in gnn_encoder.parameters()) / 1e6
    print(f"    GNNDrugEncoder: {n_gnn:.2f}M params  |  drug_dim={DRUG_DIM} (Morgan+GNN)\n")

# ══════════════════════════════════════════════════════════════════════════════
# [5] 데이터셋 & 데이터로더
# ══════════════════════════════════════════════════════════════════════════════
class FrozenDataset(Dataset):
    """Morgan FP 모드: (prot_emb, drug_fp, label)"""
    def __init__(self, indices):
        self.prot_idx = tgt_indices[indices]
        self.drug_idx = drug_indices[indices]
        self.labels   = y[indices]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (prot_embs[self.prot_idx[i]],
                drug_fps[self.drug_idx[i]],
                torch.tensor(self.labels[i], dtype=torch.float32))

class FrozenDatasetGNN(Dataset):
    """GNN 모드: (prot_emb, (morgan_fp, graph_tuple), label)"""
    def __init__(self, indices):
        self.prot_idx = tgt_indices[indices]
        self.drug_idx = drug_indices[indices]
        self.labels   = y[indices]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        didx = self.drug_idx[i]
        return (prot_embs[self.prot_idx[i]],
                (drug_fps[didx], drug_graphs[didx]),  # (morgan_fp, graph_tuple)
                torch.tensor(self.labels[i], dtype=torch.float32))

class LoRADataset(Dataset):
    def __init__(self, indices):
        self.prot_idx = tgt_indices[indices]
        self.drug_idx = drug_indices[indices]
        self.labels   = y[indices]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.prot_idx[i],
                drug_fps[self.drug_idx[i]],
                torch.tensor(self.labels[i], dtype=torch.float32))

def lora_collate(batch):
    prot_idxs, drug_fps_b, labels = zip(*batch)
    ids   = [all_input_ids[j]  for j in prot_idxs]
    masks = [all_attn_masks[j] for j in prot_idxs]
    ids_pad   = torch.nn.utils.rnn.pad_sequence(ids,   batch_first=True, padding_value=tokenizer.pad_token_id)
    masks_pad = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return ids_pad, masks_pad, torch.stack(drug_fps_b), torch.tensor(labels)

def gnn_collate(batch):
    prots, drug_tuples, labels = zip(*batch)
    fps, graphs = zip(*drug_tuples)
    nf_pad, adj_pad, bf_pad, mask = collate_graphs(list(graphs))
    return (torch.stack(prots),
            (torch.stack(fps), (nf_pad, adj_pad, bf_pad, mask)),
            torch.stack(labels))

if args.lora:
    DS = LoRADataset
    train_loader = DataLoader(DS(tr_idx),  batch_size=args.batch_size,
                              shuffle=True,  collate_fn=lora_collate, num_workers=0)
    val_loader   = DataLoader(DS(val_idx), batch_size=args.batch_size,
                              shuffle=False, collate_fn=lora_collate, num_workers=0)
    test_loader  = DataLoader(DS(te_idx),  batch_size=args.batch_size,
                              shuffle=False, collate_fn=lora_collate, num_workers=0)
elif args.drug_encoder == "gnn":
    train_loader = DataLoader(FrozenDatasetGNN(tr_idx),  batch_size=args.batch_size,
                              shuffle=True,  collate_fn=gnn_collate, num_workers=0)
    val_loader   = DataLoader(FrozenDatasetGNN(val_idx), batch_size=args.batch_size,
                              shuffle=False, collate_fn=gnn_collate, num_workers=0)
    test_loader  = DataLoader(FrozenDatasetGNN(te_idx),  batch_size=args.batch_size,
                              shuffle=False, collate_fn=gnn_collate, num_workers=0)
else:
    train_loader = DataLoader(FrozenDataset(tr_idx),  batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(FrozenDataset(val_idx), batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(FrozenDataset(te_idx),  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════════
# [6] DTI 헤드
# ══════════════════════════════════════════════════════════════════════════════
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
n_head = sum(p.numel() for p in head.parameters()) / 1e6
print(f"[5] DTI 헤드: {n_head:.2f}M params  (drug_dim={DRUG_DIM})\n")

# ══════════════════════════════════════════════════════════════════════════════
# [7] 옵티마이저 — LoRA: SaProt 어댑터 + 헤드 / frozen: 헤드만
# ══════════════════════════════════════════════════════════════════════════════
if args.lora:
    lora_params = [p for p in saprot.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": head.parameters(), "lr": args.lr * 10}],
        weight_decay=1e-4,
    )
elif args.drug_encoder == "gnn":
    # GNN + head 함께 학습 (protein은 frozen)
    optimizer = torch.optim.Adam(
        [{"params": gnn_encoder.parameters(), "lr": args.lr},
         {"params": head.parameters(),        "lr": args.lr * 5}],
        weight_decay=1e-4,
    )
else:
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6
)
criterion = nn.HuberLoss(delta=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# [8] 학습 루프 공통 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def get_prot_emb(input_ids, attention_mask):
    """LoRA 모드: SaProt forward → mean pool"""
    out    = saprot(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state.float()    # [B, L, D]
    # CLS(0), EOS(-1) 제외 mean pool
    mask   = attention_mask[:, 1:-1].unsqueeze(-1).float()
    emb    = (hidden[:, 1:-1, :] * mask).sum(1) / mask.sum(1).clamp(min=1)
    return emb    # [B, D]

# ══════════════════════════════════════════════════════════════════════════════
# [9] 훈련
# ══════════════════════════════════════════════════════════════════════════════
best_val_r, best_head_state, patience_cnt = -1.0, None, 0
best_lora_state = None
history = []

print(f"[6] 훈련 시작 (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})", flush=True)
print(f"    {'Epoch':>5} | {'Train Loss':>10} | {'Val r':>7} | {'Best':>7} | {'RMSE':>7} | {'CI':>6}", flush=True)
print("    " + "-" * 58, flush=True)

t_start = time.time()

def _get_drug_emb(batch_drug):
    """GNN 모드: Morgan FP + GNN concat → drug embedding. Morgan 모드: 그대로 반환."""
    if args.drug_encoder == "gnn":
        morgan_fp, graph_tuple = batch_drug
        morgan_fp = morgan_fp.to(DEVICE)
        nf, adj, bf, mask = graph_tuple
        nf   = nf.to(DEVICE)
        adj  = adj.to(DEVICE)
        bf   = bf.to(DEVICE)
        mask = mask.to(DEVICE)
        gnn_emb = gnn_encoder(nf, adj, bf, mask)       # [B, 256]
        return torch.cat([morgan_fp, gnn_emb], dim=-1)  # [B, 2304]
    return batch_drug.to(DEVICE)

for epoch in range(1, args.epochs + 1):
    # ── GNN 2단계 학습: warmup_epochs까지 GNN 동결 ──────────────────────────
    if args.drug_encoder == "gnn":
        if epoch <= args.gnn_warmup_epochs:
            gnn_encoder.eval()
            for p in gnn_encoder.parameters():
                p.requires_grad_(False)
            if epoch == 1:
                print(f"    [2단계 학습] Stage 1: epoch {args.gnn_warmup_epochs}까지 GNN 동결 → Morgan FP로 Head 수렴", flush=True)
        elif epoch == args.gnn_warmup_epochs + 1:
            gnn_encoder.train()
            for p in gnn_encoder.parameters():
                p.requires_grad_(True)
            print(f"\n    [2단계 학습] Stage 2: GNN 해동 → Morgan FP + GNN 함께 학습\n", flush=True)

    # ── train ──────────────────────────────────────────────────────────────
    head.train()
    if args.lora:                                                   saprot.train()
    if args.drug_encoder == "gnn" and epoch > args.gnn_warmup_epochs: gnn_encoder.train()
    train_loss = 0.0

    for batch in train_loader:
        if args.lora:
            ids, masks, drug, label = batch
            ids, masks = ids.to(DEVICE), masks.to(DEVICE)
            drug, label = drug.to(DEVICE), label.to(DEVICE)
            prot = get_prot_emb(ids, masks)
        else:
            prot, drug, label = batch
            prot  = prot.to(DEVICE)
            label = label.to(DEVICE)
            drug  = _get_drug_emb(drug)

        pred  = head(prot, drug)
        loss  = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        clip_params = list(head.parameters())
        if args.lora:                   clip_params += list(saprot.parameters())
        if args.drug_encoder == "gnn" and epoch > args.gnn_warmup_epochs:
            clip_params += list(gnn_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optimizer.step()
        train_loss += loss.item() * len(label)

    train_loss /= len(tr_idx)
    scheduler.step()

    # ── validate ───────────────────────────────────────────────────────────
    head.eval()
    if args.lora:                   saprot.eval()
    if args.drug_encoder == "gnn":  gnn_encoder.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            if args.lora:
                ids, masks, drug, label = batch
                ids, masks = ids.to(DEVICE), masks.to(DEVICE)
                drug = drug.to(DEVICE)
                prot = get_prot_emb(ids, masks)
            else:
                prot, drug, label = batch
                prot = prot.to(DEVICE)
                drug = _get_drug_emb(drug)
            pred = head(prot, drug).cpu().numpy()
            val_preds.extend(pred)
            val_labels.extend(label.numpy())

    val_r, _ = pearsonr(val_preds, val_labels)
    _vp = np.array(val_preds, dtype=np.float32)
    _vl = np.array(val_labels, dtype=np.float32)
    val_rmse = float(math.sqrt(np.mean((_vp - _vl) ** 2)))
    # CI (빠른 근사: 샘플 500개)
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(_vp), min(500, len(_vp)), replace=False)
    _yt, _yp = _vl[_idx], _vp[_idx]
    _conc = _tot = 0
    for _i in range(len(_yt)):
        for _j in range(_i+1, len(_yt)):
            if _yt[_i] == _yt[_j]: continue
            _tot += 1
            if (_yt[_i] > _yt[_j]) == (_yp[_i] > _yp[_j]): _conc += 1
    val_ci = _conc / _tot if _tot > 0 else 0.0

    is_best = val_r > best_val_r

    if is_best:
        best_val_r    = val_r
        best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        if args.lora:
            best_lora_state = {k: v.clone() for k, v in saprot.state_dict().items()
                               if "lora" in k}
        patience_cnt = 0
    else:
        patience_cnt += 1

    history.append({"epoch": epoch, "train_loss": train_loss, "val_r": val_r,
                    "val_rmse": val_rmse, "val_ci": val_ci})
    marker = " ★" if is_best else ""
    print(f"    {epoch:>5} | {train_loss:>10.4f} | {val_r:>7.4f} | "
          f"{best_val_r:>7.4f}{marker} | {val_rmse:>7.4f} | {val_ci:>6.4f}", flush=True)

    if patience_cnt >= args.patience:
        print(f"\n    Early stopping (patience={args.patience})", flush=True)
        break

train_time = time.time() - t_start

# ══════════════════════════════════════════════════════════════════════════════
# [10] 테스트
# ══════════════════════════════════════════════════════════════════════════════
head.load_state_dict(best_head_state)
head.eval()
if args.lora:
    # best LoRA 가중치 복원
    cur = saprot.state_dict()
    cur.update(best_lora_state)
    saprot.load_state_dict(cur)
    saprot.eval()

test_preds, test_labels = [], []
if args.drug_encoder == "gnn":
    gnn_encoder.eval()
with torch.no_grad():
    for batch in test_loader:
        if args.lora:
            ids, masks, drug, label = batch
            ids, masks = ids.to(DEVICE), masks.to(DEVICE)
            drug = drug.to(DEVICE)
            prot = get_prot_emb(ids, masks)
        else:
            prot, drug, label = batch
            prot  = prot.to(DEVICE)
            drug  = _get_drug_emb(drug)
        pred = head(prot, drug).cpu().numpy()
        test_preds.extend(pred)
        test_labels.extend(label.numpy())

test_r, test_p = pearsonr(test_preds, test_labels)

# ── 추가 지표 계산 ─────────────────────────────────────────────────────────────
_pred = np.array(test_preds, dtype=np.float32)
_true = np.array(test_labels, dtype=np.float32)
test_rmse = float(math.sqrt(np.mean((_pred - _true) ** 2)))
test_mae  = float(np.mean(np.abs(_pred - _true)))

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

test_ci = _concordance_index(_true, _pred)

# ── 추론 속도 측정 (단일 샘플, warmup 제외) ────────────────────────────────────
head.eval()
_n_speed = min(200, len(te_idx))
_speed_times = []
with torch.no_grad():
    # warmup
    _b = next(iter(test_loader))
    if args.lora:
        _ids, _masks, _drug, _ = _b
        _prot_s = get_prot_emb(_ids[:1].to(DEVICE), _masks[:1].to(DEVICE))
        _drug_s  = _drug[:1].to(DEVICE)
    else:
        _prot_s, _drug_s_raw, _ = _b
        _prot_s = _prot_s[:1].to(DEVICE)
        if args.drug_encoder == "gnn":
            _fp_s, _graph_s = _drug_s_raw
            _drug_s = _get_drug_emb((_fp_s[:1], tuple(t[:1] for t in _graph_s)))
        else:
            _drug_s = _drug_s_raw[:1].to(DEVICE)
    head(_prot_s, _drug_s)
    # 실측
    for _ in range(_n_speed):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _t0 = time.perf_counter()
        head(_prot_s, _drug_s)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _speed_times.append((time.perf_counter() - _t0) * 1000)  # ms

infer_ms_mean = round(float(np.mean(_speed_times)), 3)
infer_ms_std  = round(float(np.std(_speed_times)),  3)

# ── VRAM 사용량 ────────────────────────────────────────────────────────────────
peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1) \
               if torch.cuda.is_available() else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# [11] 결과 저장
# ══════════════════════════════════════════════════════════════════════════════
out_dir = Path("./results") / run_name
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"y_pred": test_preds, "y_true": test_labels}).to_csv(
    out_dir / "test_predictions.csv", index=False)
pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
torch.save(best_head_state, out_dir / "dti_head.pt")
if args.lora:
    torch.save(best_lora_state, out_dir / "lora_adapter.pt")

result = {
    "run_name":          run_name,
    "dataset":           args.dataset,
    "encoder":           args.encoder,
    "quant":             args.quant,
    "lora":              args.lora,
    "use_3di":           args.use_3di,
    "drug_encoder":      args.drug_encoder,
    "split":             args.split,
    "lora_r":            args.lora_r if args.lora else None,
    "prot_dim":          prot_dim,
    "timestamp":         datetime.now().isoformat(),
    # ── 성능 지표
    "test_pearson_r":    round(float(test_r),   4),
    "test_rmse":         round(test_rmse,        4),
    "test_mae":          round(test_mae,         4),
    "test_ci":           round(test_ci,          4),
    "test_p_value":      float(test_p),
    "best_val_r":        round(float(best_val_r), 4),
    # ── 학습 정보
    "epochs_trained":    len(history),
    "train_time_sec":    round(train_time, 1),
    "n_train":           len(tr_idx),
    "n_val":             len(val_idx),
    "n_test":            len(te_idx),
    # ── 추론 속도 / 하드웨어
    "infer_ms_mean":     infer_ms_mean,
    "infer_ms_std":      infer_ms_std,
    "peak_vram_mb":      peak_vram_mb,
}
with open(out_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# ── 마크다운 실험 보고서 ────────────────────────────────────────────────────────
_perf_mark = ("✅✅ 목표 초과 (r ≥ 0.9)" if test_r >= 0.9 else
              "✅  목표 달성 (r ≥ 0.85)" if test_r >= 0.85 else
              "🔄  근접 (r ≥ 0.8)"       if test_r >= 0.8 else
              "△   양호 (r ≥ 0.6)"       if test_r >= 0.6 else
              "❌  성능 미달")

_report = f"""# 실험 보고서 — {run_name}

**생성일시:** {result['timestamp']}

---

## 실험 설정

| 항목 | 값 |
|---|---|
| 데이터셋 | {args.dataset.upper()} |
| Protein Encoder | SaProt-{args.encoder} ({args.quant if args.quant != 'none' else 'FP16'}) |
| Drug Encoder | {'Morgan FP (2048-bit) + GNN/MPNN (256-dim) concat → 2304-dim' if args.drug_encoder == 'gnn' else ('ChemBERTa (seyonec/ChemBERTa-zinc-base-v1, frozen, 768-dim)' if args.drug_encoder == 'chemberta' else 'Morgan FP (radius=2, 2048-bit)')} |
| FoldSeek 3Di | {'✅ 사용' if args.use_3di else '❌ Placeholder'} |
| LoRA | {'✅ rank=' + str(args.lora_r) if args.lora else '❌ Frozen'} |
| Split | Random 70 / 10 / 20 |
| Train / Val / Test | {len(tr_idx):,} / {len(val_idx):,} / {len(te_idx):,} |

---

## 성능 지표

| 지표 | 값 | 설명 |
|---|---|---|
| **Pearson r** | **{test_r:.4f}** | 예측-실측 선형 상관계수 (주 지표) |
| RMSE | {test_rmse:.4f} | 평균 예측 오차 (pKd 단위) |
| MAE | {test_mae:.4f} | 평균 절대 오차 (pKd 단위) |
| CI | {test_ci:.4f} | 결합력 순위 일치도 (0.5=랜덤, 1.0=완벽) |
| Val best r | {best_val_r:.4f} | 검증셋 최고 Pearson r |

**판정:** {_perf_mark}

### SOTA 비교

| 모델 | DAVIS Pearson r |
|---|---|
| 본 실험 | **{test_r:.4f}** |
| DeepPurpose MPNN_CNN | ~0.89 (SOTA) |
| DeepPurpose CNN | ~0.86 |
| 서열 기반 baseline | ~0.78~0.80 |

---

## 학습 정보

| 항목 | 값 |
|---|---|
| 학습 에포크 | {len(history)} |
| 총 학습 시간 | {train_time:.1f}초 ({train_time/60:.1f}분) |
| Early stopping | patience={args.patience} |

---

## 추론 속도 / 하드웨어

| 항목 | 값 |
|---|---|
| 단일 샘플 추론 시간 | **{infer_ms_mean:.3f} ms** (± {infer_ms_std:.3f} ms) |
| 추론 속도 | **{1000/infer_ms_mean:.0f} samples/sec** |
| 학습 중 최대 VRAM | {peak_vram_mb:.1f} MB |

---

## 결과 파일

| 파일 | 내용 |
|---|---|
| `result.json` | 전체 지표 요약 |
| `test_predictions.csv` | 예측값 vs 실측값 |
| `training_history.csv` | 에포크별 loss / val_r |
| `dti_head.pt` | 최적 모델 가중치 |
"""

with open(out_dir / "report.md", "w", encoding="utf-8") as f:
    f.write(_report)

print(f"\n{'='*56}")
print(f"  모델:          {run_name}")
print(f"  Pearson r:     {test_r:.4f}   RMSE: {test_rmse:.4f}   CI: {test_ci:.4f}")
print(f"  Val best r:    {best_val_r:.4f}")
print(f"  추론 속도:     {infer_ms_mean:.3f} ms/sample  ({1000/infer_ms_mean:.0f} samples/sec)")
print(f"  VRAM 최대:     {peak_vram_mb:.0f} MB")
print(f"  학습 시간:     {train_time:.0f}초")
print(f"  {_perf_mark}")
print(f"  결과 저장:     {out_dir}/")
print(f"  보고서:        {out_dir}/report.md")
print("=" * 56)
print("\n[완료]")
