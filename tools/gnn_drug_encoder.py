"""
gnn_drug_encoder.py
===================
MPNN(Message Passing Neural Network) 기반 약물 인코더.
torch_geometric 없이 순수 PyTorch + RDKit으로 구현.

입력 : SMILES 문자열
출력 : [out_dim]-dim 약물 임베딩 벡터
"""

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from typing import Optional, Tuple, List

# ── 원자 / 결합 특성 정의 ────────────────────────────────────────────────────────

ATOM_SYMBOLS = [
    'C','N','O','S','F','P','Cl','Br','I','B','Si','Se','other'
]  # 13
HYBRIDIZATIONS = ['SP','SP2','SP3','SP3D','SP3D2','other']  # 6
N_ATOM_FEAT = 13 + 11 + 1 + 5 + 1 + 1 + 6  # = 38
N_BOND_FEAT = 4 + 1 + 1                      # = 6
MAX_ATOMS   = 50   # DAVIS max=46, KIBA도 대부분 50 이하


def _one_hot(val, choices: list) -> list:
    vec = [0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1
    return vec


def get_atom_features(atom) -> list:
    """원자 특성 벡터 (38-dim)"""
    sym  = _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)                              # 13
    deg  = _one_hot(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10])                # 11
    fchg = [max(-2, min(2, atom.GetFormalCharge())) / 2.0]                      #  1
    numh = _one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])                          #  5
    arom = [float(atom.GetIsAromatic())]                                          #  1
    ring = [float(atom.IsInRing())]                                               #  1
    hyb  = _one_hot(str(atom.GetHybridization()).split('.')[-1], HYBRIDIZATIONS) #  6
    return sym + deg + fchg + numh + arom + ring + hyb


def get_bond_features(bond) -> list:
    """결합 특성 벡터 (6-dim)"""
    btype = _one_hot(str(bond.GetBondTypeAsDouble()), ['1.0','1.5','2.0','3.0']) # 4
    conj  = [float(bond.GetIsConjugated())]                                       # 1
    ring  = [float(bond.IsInRing())]                                              # 1
    return btype + conj + ring


def smiles_to_graph(
    smiles: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    SMILES → (node_feats, adj, bond_feats).

    Returns
    -------
    node_feats : FloatTensor [n, N_ATOM_FEAT]
    adj        : FloatTensor [n, n]
    bond_feats : FloatTensor [n, n, N_BOND_FEAT]
    None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = min(mol.GetNumAtoms(), MAX_ATOMS)

    nf  = torch.zeros(n, N_ATOM_FEAT)
    adj = torch.zeros(n, n)
    bf  = torch.zeros(n, n, N_BOND_FEAT)

    for i, atom in enumerate(mol.GetAtoms()):
        if i >= n:
            break
        nf[i] = torch.tensor(get_atom_features(atom), dtype=torch.float32)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i >= n or j >= n:
            continue
        feat = torch.tensor(get_bond_features(bond), dtype=torch.float32)
        adj[i, j] = adj[j, i] = 1.0
        bf[i, j]  = bf[j, i]  = feat

    return nf, adj, bf


def collate_graphs(
    graph_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    List[(nf, adj, bf)] → 패딩된 배치 텐서 + mask.

    Returns
    -------
    nf_pad   : [B, N_max, N_ATOM_FEAT]
    adj_pad  : [B, N_max, N_max]
    bf_pad   : [B, N_max, N_max, N_BOND_FEAT]
    mask     : [B, N_max]  float (1=실제 원자, 0=패딩)
    """
    ns = [g[0].shape[0] for g in graph_list]
    N  = max(ns)
    B  = len(graph_list)

    nf_pad  = torch.zeros(B, N, N_ATOM_FEAT)
    adj_pad = torch.zeros(B, N, N)
    bf_pad  = torch.zeros(B, N, N, N_BOND_FEAT)
    mask    = torch.zeros(B, N)

    for i, (nf, adj, bf) in enumerate(graph_list):
        n = nf.shape[0]
        nf_pad[i, :n]       = nf
        adj_pad[i, :n, :n]  = adj
        bf_pad[i, :n, :n]   = bf
        mask[i, :n]         = 1.0

    return nf_pad, adj_pad, bf_pad, mask


# ── MPNN 레이어 ──────────────────────────────────────────────────────────────────

class MPNNLayer(nn.Module):
    """
    단일 MPNN 메시지 패싱 레이어.
    이웃 원자 특성 + 결합 특성을 합쳐 메시지 생성 → GRU로 노드 상태 갱신.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.msg_fn = nn.Sequential(
            nn.Linear(hidden + N_BOND_FEAT, hidden),
            nn.ReLU(),
        )
        self.gru  = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        h:    torch.Tensor,  # [B, N, hidden]
        adj:  torch.Tensor,  # [B, N, N]
        bf:   torch.Tensor,  # [B, N, N, N_BOND_FEAT]
        mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        B, N, D = h.shape

        # 이웃 → 현재 노드 방향으로 메시지 수집
        h_nbr   = h.unsqueeze(2).expand(B, N, N, D)         # [B, N, N, D]
        msg_in  = torch.cat([h_nbr, bf], dim=-1)             # [B, N, N, D+6]
        msg     = self.msg_fn(msg_in)                         # [B, N, N, D]
        msg     = msg * adj.unsqueeze(-1)                     # 실제 결합만
        agg     = msg.sum(dim=2)                              # [B, N, D]

        # GRU 업데이트
        h_new = self.gru(
            agg.reshape(B * N, D),
            h.reshape(B * N, D),
        ).reshape(B, N, D)

        h_new = self.norm(h_new)
        h_new = h_new * mask.unsqueeze(-1)  # 패딩 마스킹
        return h_new


# ── GNN Drug Encoder ─────────────────────────────────────────────────────────────

GNN_OUT_DIM = 256  # drug embedding 출력 차원 (DTIHead drug_dim 과 일치)


class GNNDrugEncoder(nn.Module):
    """
    MPNN 기반 약물 인코더.

    Parameters
    ----------
    hidden   : 내부 hidden 차원 (기본 256)
    out_dim  : 출력 임베딩 차원 (기본 GNN_OUT_DIM=256)
    n_layers : 메시지 패싱 레이어 수 (기본 4)
    """
    def __init__(
        self,
        hidden:   int = 256,
        out_dim:  int = GNN_OUT_DIM,
        n_layers: int = 4,
    ):
        super().__init__()
        self.embed  = nn.Linear(N_ATOM_FEAT, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden) for _ in range(n_layers)])
        self.proj   = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(
        self,
        node_feats: torch.Tensor,  # [B, N, N_ATOM_FEAT]
        adj:        torch.Tensor,  # [B, N, N]
        bond_feats: torch.Tensor,  # [B, N, N, N_BOND_FEAT]
        mask:       torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        h = torch.relu(self.embed(node_feats))  # [B, N, hidden]
        h = h * mask.unsqueeze(-1)

        for layer in self.layers:
            h = layer(h, adj, bond_feats, mask)

        # Global Readout: mean + max pooling → concat
        mask_exp = mask.unsqueeze(-1)                              # [B, N, 1]
        h_mean = (h * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-6)
        h_max  = h.masked_fill(mask_exp == 0, -1e9).max(1).values

        out = torch.cat([h_mean, h_max], dim=-1)                  # [B, hidden*2]
        return self.proj(out)                                      # [B, out_dim]


# ── 빠른 테스트 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    smiles_list = [
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # Imatinib
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "c1ccc2ccccc2c1",          # Naphthalene
    ]
    graphs = [smiles_to_graph(s) for s in smiles_list]
    valid  = [g for g in graphs if g is not None]
    print(f"Graphs built: {len(valid)}/{len(smiles_list)}")

    nf, adj, bf, mask = collate_graphs(valid)
    print(f"Batch shapes → nf:{nf.shape}  adj:{adj.shape}  bf:{bf.shape}  mask:{mask.shape}")

    model = GNNDrugEncoder()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"GNNDrugEncoder: {n_params:.2f}M params")

    emb = model(nf, adj, bf, mask)
    print(f"Output embedding: {emb.shape}  (expected [3, {GNN_OUT_DIM}])")
    print("✅ GNN Drug Encoder 테스트 완료")
