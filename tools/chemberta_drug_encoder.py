"""
chemberta_drug_encoder.py
=========================
ChemBERTa 기반 약물 인코더.
PubChem 77M SMILES로 사전학습된 RoBERTa 기반 분자 Transformer.

모델: seyonec/ChemBERTa-zinc-base-v1 (~92M params, 768-dim output)
입력 : SMILES 문자열 리스트
출력 : [768]-dim 약물 임베딩 텐서 (frozen inference)

사용법:
    encoder = ChemBERTaDrugEncoder(device="cuda")
    embs = encoder.encode(smiles_list)  # [N, 768]
"""

import torch
from transformers import AutoModel, AutoTokenizer

CHEMBERTA_MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"
CHEMBERTA_DIM = 768


class ChemBERTaDrugEncoder:
    """
    ChemBERTa frozen drug encoder.

    Parameters
    ----------
    model_name : HuggingFace 모델 ID
    device     : "cuda" or "cpu"
    """

    def __init__(
        self,
        model_name: str = CHEMBERTA_MODEL_ID,
        device: str = "cpu",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(
        self,
        smiles_list: list,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        SMILES 리스트 → [N, 768] embedding tensor (CPU).

        invalid SMILES는 zero vector로 처리.
        """
        all_embs = []
        n = len(smiles_list)

        for start in range(0, n, batch_size):
            batch = smiles_list[start : start + batch_size]

            # 빈 문자열 / None 방어
            batch = [s if isinstance(s, str) and s.strip() else "C" for s in batch]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # mean pooling over all token embeddings (attention_mask 기준)
                hidden = outputs.last_hidden_state          # [B, L, 768]
                mask   = inputs["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
                emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, 768]

            all_embs.append(emb.cpu())

            if show_progress:
                done = min(start + batch_size, n)
                print(f"    {done}/{n} 약물 임베딩 완료", flush=True)

        return torch.cat(all_embs, dim=0)  # [N, 768]


# ── 빠른 테스트 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    smiles_list = [
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccc2ccccc2c1",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    encoder = ChemBERTaDrugEncoder(device=device)
    embs = encoder.encode(smiles_list, show_progress=True)
    print(f"Output shape: {embs.shape}  (expected [{len(smiles_list)}, {CHEMBERTA_DIM}])")
    print("✅ ChemBERTa Drug Encoder 테스트 완료")
