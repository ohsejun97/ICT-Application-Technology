"""
preprocess_bindingdb.py
=======================
서버에서 실행: BindingDB_All.tsv (8GB) → bindingdb_kd.csv (작은 CSV)

사용법:
  python preprocess_bindingdb.py --input ./BindingDB_All.tsv --output ./bindingdb_kd.csv

출력 컬럼: smiles, sequence, pkd, uniprot_id
  - uniprot_id: 3Di 캐시 빌드 시 BLAST 스킵용 (없으면 빈 문자열)
"""

import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True, help="BindingDB_All.tsv 경로")
parser.add_argument("--output", required=True, help="출력 CSV 경로")
args = parser.parse_args()

print(f"[1] Reading {args.input} ...")
df = pd.read_csv(args.input, sep="\t", on_bad_lines="skip", low_memory=False)
print(f"    Raw rows: {len(df):,}")

# 단일 체인 단백질만
df = df[df["Number of Protein Chains in Target (>1 implies a multichain complex)"] == 1.0]

# SMILES / InChI null 제거
df = df[df["Ligand SMILES"].notnull()]
df = df[df["Ligand InChI"].notnull()]

# Kd 컬럼 필터링
df = df[df["Kd (nM)"].notnull()]

# UniProt or PubChem ID 중 하나는 있어야
df = df[df["PubChem CID"].notnull() | df["UniProt (SwissProt) Primary ID of Target Chain 1"].notnull()]

# 타깃 서열 null 제거
df = df[df["BindingDB Target Chain Sequence 1"].notnull()]

# 슬라이스 복사 — 이후 컬럼 수정 시 SettingWithCopyWarning 방지
df = df.copy()

print(f"[2] After filtering: {len(df):,} rows")

# > < 기호 제거 후 float 변환
df["Kd (nM)"] = df["Kd (nM)"].astype(str).str.replace(">", "").str.replace("<", "").str.strip()
df["Kd (nM)"] = pd.to_numeric(df["Kd (nM)"], errors="coerce")
df = df[df["Kd (nM)"].notnull()]

# 이상값 제거: 0 이하(log 불가) 및 10,000,000 nM 초과
df = df[df["Kd (nM)"] > 0]
df = df[df["Kd (nM)"] <= 10_000_000.0]

# Kd(nM) → pKd = -log10(Kd * 1e-9)
pkd = -np.log10(df["Kd (nM)"].values * 1e-9)

out = pd.DataFrame({
    "smiles":     df["Ligand SMILES"].values,
    "sequence":   df["BindingDB Target Chain Sequence 1"].values,
    "pkd":        pkd,
    "uniprot_id": df["UniProt (SwissProt) Primary ID of Target Chain 1"].values,
})

# inf / nan 최종 제거
out = out[np.isfinite(out["pkd"])]

# 중복 (smiles, sequence) 쌍 → pKd 평균, uniprot_id는 첫 번째 non-null 값
before = len(out)
out = out.groupby(["smiles", "sequence"], as_index=False).agg(
    pkd=("pkd", "mean"),
    uniprot_id=("uniprot_id", "first"),
)
# uniprot_id null → 빈 문자열 (BLAST fallback 신호)
out["uniprot_id"] = out["uniprot_id"].fillna("")

print(f"    Dedup: {before:,} → {len(out):,} (중복 제거 {before - len(out):,}행)")

n_with_uid = (out["uniprot_id"] != "").sum()
print(f"    UniProt ID 보유: {n_with_uid:,} / {out['sequence'].nunique():,} 고유 타겟")

print(f"[3] Final pairs: {len(out):,}")
print(f"    Unique drugs:   {out['smiles'].nunique():,}")
print(f"    Unique targets: {out['sequence'].nunique():,}")
print(f"    pKd range: {out['pkd'].min():.2f} ~ {out['pkd'].max():.2f}, mean={out['pkd'].mean():.2f}")

out.to_csv(args.output, index=False)
print(f"[4] Saved → {args.output}")
