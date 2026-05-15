"""
dti_tool.py
===========
DTI Prediction Tool — Agent Tool #1

SaProt-650M (FP16, frozen) + FoldSeek 3Di tokens (protein encoder)
ChemBERTa fine-tuned on BindingDB (drug encoder, layers 4~5 adapted)
MLP Head trained on BindingDB 80K pairs (Pearson r=0.8923)

Model pipeline:
  SMILES  → ft-ChemBERTa (mean pool) → [768-dim]  ─┐
                                                     ├→ MLP Head → pKd
  AA seq  → 3Di tokens → SaProt-650M (FP16 frozen)  ─┘

3Di tokens: looked up from cache by seq MD5 hash.
            Falls back to '#' placeholder if not cached
            (slight performance drop, still functional).

Usage (standalone test):
  python tools/dti_tool.py
"""

import sys
import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Model paths ───────────────────────────────────────────────────────────────
FT_MODEL_DIR  = ROOT / "results" / "SaProt-650M-bindingdb-3di-chemberta-unfreeze2-random"
HEAD_CKPT     = FT_MODEL_DIR / "dti_head.pt"
CHEMBERTA_FT  = FT_MODEL_DIR / "chemberta_ft.pt"
SAPROT_ID     = "westlake-repl/SaProt_650M_AF2"
CHEMBERTA_ID  = "seyonec/ChemBERTa-zinc-base-v1"

# 3Di 캐시 파일 목록 (모두 로드하여 통합 조회)
_3DI_CACHES = [
    ROOT / "cache" / "3di_tokens_davis.json",
    ROOT / "cache" / "3di_tokens_kiba.json",
    ROOT / "cache" / "3di_tokens_bindingdb.json",
]
CHEMBERTA_MAX_LEN = 128
PROT_DIM = 1280
DRUG_DIM = 768
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Singleton state ───────────────────────────────────────────────────────────
_saprot       = None
_sa_tokenizer = None
_cb_model     = None
_cb_tokenizer = None
_head         = None
_3di_cache    = {}   # seq_md5 → tokens_3di string


# ══════════════════════════════════════════════════════════════════════════════
# DTI Head (Phase 1h/1i 아키텍처)
# ══════════════════════════════════════════════════════════════════════════════
class DTIHead(nn.Module):
    def __init__(self, prot_dim=PROT_DIM, drug_dim=DRUG_DIM, hidden=512):
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


# ══════════════════════════════════════════════════════════════════════════════
# 3Di 캐시 관련 유틸
# ══════════════════════════════════════════════════════════════════════════════
def _load_3di_caches():
    global _3di_cache
    if _3di_cache:
        return
    for path in _3DI_CACHES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.values():
            if entry.get("status") == "ok" and entry.get("tokens_3di"):
                _3di_cache[entry["seq_hash"]] = entry["tokens_3di"]


def _seq_to_sa(aa_seq: str) -> str:
    """AA 서열 → SA 토큰 문자열 (3Di 캐시 있으면 구조 토큰, 없으면 '#' fallback)"""
    _load_3di_caches()
    seq_hash = hashlib.md5(aa_seq.encode()).hexdigest()
    tokens_3di = _3di_cache.get(seq_hash)
    if tokens_3di and len(tokens_3di) == len(aa_seq):
        return "".join(aa.upper() + di.lower()
                       for aa, di in zip(aa_seq, tokens_3di))
    # fallback: '#' placeholder (구조 정보 없음, 소폭 성능 하락)
    return "".join(aa + "#" for aa in aa_seq)


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드 (singleton)
# ══════════════════════════════════════════════════════════════════════════════
def _load_models():
    global _saprot, _sa_tokenizer, _cb_model, _cb_tokenizer, _head

    if _saprot is not None:
        return

    from transformers import EsmModel, EsmTokenizer, AutoModel, AutoTokenizer

    # ── SaProt-650M FP16 (frozen) ────────────────────────────────────────────
    print("  [DTI Tool] Loading SaProt-650M FP16 ...")
    _sa_tokenizer = EsmTokenizer.from_pretrained(SAPROT_ID)
    _saprot = EsmModel.from_pretrained(
        SAPROT_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        add_pooling_layer=False,
    )
    _saprot.eval()
    for p in _saprot.parameters():
        p.requires_grad_(False)

    # ── ChemBERTa + fine-tuned weights ──────────────────────────────────────
    print("  [DTI Tool] Loading ft-ChemBERTa ...")
    _cb_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
    _cb_model     = AutoModel.from_pretrained(CHEMBERTA_ID).to(DEVICE)

    if CHEMBERTA_FT.exists():
        ft_state  = torch.load(CHEMBERTA_FT, map_location=DEVICE, weights_only=True)
        cur_state = _cb_model.state_dict()
        cur_state.update(ft_state)
        _cb_model.load_state_dict(cur_state)
    else:
        print("  [DTI Tool] ⚠️  chemberta_ft.pt not found — using vanilla ChemBERTa")

    _cb_model.eval()
    for p in _cb_model.parameters():
        p.requires_grad_(False)

    # ── MLP Head ─────────────────────────────────────────────────────────────
    print("  [DTI Tool] Loading DTI head ...")
    _head = DTIHead().to(DEVICE)
    _head.load_state_dict(
        torch.load(HEAD_CKPT, map_location=DEVICE, weights_only=True))
    _head.eval()

    print("  [DTI Tool] Ready  (SaProt FP16 + ft-ChemBERTa + MLP Head)\n")


# ══════════════════════════════════════════════════════════════════════════════
# 인코딩 함수
# ══════════════════════════════════════════════════════════════════════════════
def _encode_protein(aa_seq: str) -> torch.Tensor:
    """AA 서열 → [1, 1280] SaProt 임베딩"""
    sa_seq = _seq_to_sa(aa_seq)
    inputs = _sa_tokenizer(
        sa_seq, return_tensors="pt",
        truncation=True, max_length=1024, padding=False,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out    = _saprot(**inputs)
        hidden = out.last_hidden_state[0, 1:-1, :].float()  # skip CLS, EOS
        emb    = hidden.mean(0).unsqueeze(0)                 # [1, 1280]
    return emb


def _encode_drug(smiles: str) -> torch.Tensor:
    """SMILES → [1, 768] ft-ChemBERTa 임베딩"""
    inputs = _cb_tokenizer(
        smiles, return_tensors="pt",
        padding=True, truncation=True, max_length=CHEMBERTA_MAX_LEN,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out    = _cb_model(**inputs)
        hidden = out.last_hidden_state                         # [1, L, 768]
        mask   = inputs["attention_mask"].unsqueeze(-1).float()
        emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # [1, 768]
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def predict_binding(smiles: str, aa_seq: str) -> dict:
    """
    약물-표적 결합 친화도(pKd) 예측.

    Args:
        smiles : 약물 SMILES 문자열
        aa_seq : 단백질 아미노산 서열 (1-letter code)

    Returns:
        dict with keys: pKd, interpretation, smiles, seq_length, used_3di
        dict with key 'error' on failure
    """
    _load_models()

    if not smiles or not smiles.strip():
        return {"error": "Empty SMILES string.", "smiles": smiles}
    if not aa_seq or len(aa_seq) < 5:
        return {"error": "Amino acid sequence too short (< 5 aa).", "smiles": smiles}

    # 3Di 캐시 사용 여부 기록
    _load_3di_caches()
    seq_hash  = hashlib.md5(aa_seq.encode()).hexdigest()
    used_3di  = seq_hash in _3di_cache

    try:
        drug_emb = _encode_drug(smiles)
        prot_emb = _encode_protein(aa_seq)

        with torch.no_grad():
            pKd = _head(prot_emb, drug_emb).item()

    except Exception as e:
        return {"error": str(e), "smiles": smiles}

    return {
        "smiles":         smiles,
        "seq_length":     len(aa_seq),
        "pKd":            round(pKd, 4),
        "interpretation": _interpret_pkd(pKd),
        "used_3di":       used_3di,
    }


def _interpret_pkd(pkd: float) -> str:
    if pkd >= 9.0:
        return "Very strong binding (pKd ≥ 9.0, Kd ≤ 1 nM)"
    elif pkd >= 7.0:
        return "Strong binding (pKd 7–9, Kd 1–100 nM)"
    elif pkd >= 5.0:
        return "Moderate binding (pKd 5–7, Kd 0.1–10 µM)"
    else:
        return "Weak / no significant binding (pKd < 5)"


def format_result(r: dict) -> str:
    if "error" in r:
        return f"[DTI Tool] Error: {r['error']}"
    return (
        f"[DTI Tool]\n"
        f"  pKd          : {r['pKd']}\n"
        f"  Interpretation: {r['interpretation']}\n"
        f"  Drug SMILES  : {r['smiles'][:60]}{'...' if len(r['smiles']) > 60 else ''}\n"
        f"  Protein length: {r['seq_length']} aa\n"
        f"  3Di tokens   : {'✅ cached' if r.get('used_3di') else '⚠️  fallback (#)'}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Imatinib + ABL1 (DAVIS에 있는 쌍, 3Di 캐시 있음)
    TEST_SMILES = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    TEST_SEQ    = (
        "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGD"
        "NTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVR"
    )

    print("=" * 60)
    print("  DTI Tool — Standalone Test")
    print("  Model: SaProt-650M FP16 + ft-ChemBERTa + MLP (r=0.8923)")
    print("  Drug : Imatinib")
    print("  Target: ABL1 (partial)")
    print("=" * 60)
    result = predict_binding(TEST_SMILES, TEST_SEQ)
    print(format_result(result))
