"""
foldseek_tool.py
================
FoldSeek 3Di Token Extractor — Agent Tool helper

PDB 파일 → FoldSeek 3Di 구조 토큰 추출.
SaProt SA 시퀀스 포맷으로 변환하는 유틸도 포함.

Requires: foldseek binary in PATH (https://github.com/steineggerlab/foldseek)

Usage (standalone test):
  python tools/foldseek_tool.py cache/alphafold/P00533.pdb
"""

import subprocess
import tempfile
from pathlib import Path


FOLDSEEK_BIN = "foldseek"


def check_foldseek() -> bool:
    """FoldSeek 바이너리가 PATH에 있는지 확인."""
    result = subprocess.run(
        [FOLDSEEK_BIN, "version"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def extract_3di_tokens(pdb_path: str) -> str | None:
    """
    PDB 파일에서 FoldSeek 3Di 구조 토큰을 추출.

    Args:
        pdb_path: PDB 파일 경로

    Returns:
        3Di 토큰 문자열 (소문자, e.g. "daklmsvtcq...") 또는 실패 시 None.
        반환된 문자열 길이 == PDB 내 아미노산 잔기 수.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        print(f"  [FoldSeek] PDB not found: {pdb_path}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_prefix = tmp / "db"
        fasta_out = tmp / "3di.fasta"

        # Step 1: PDB → FoldSeek DB (AA + 3Di 동시 생성)
        result = subprocess.run(
            [FOLDSEEK_BIN, "createdb", str(pdb_path), str(db_prefix),
             "--threads", "1", "--chain-name-mode", "0"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [FoldSeek] createdb failed: {result.stderr[:200]}")
            return None

        # Step 2: _ss 헤더 연결 (convert2fasta가 헤더 필요)
        ss_path = str(db_prefix) + "_ss"
        result = subprocess.run(
            [FOLDSEEK_BIN, "lndb", str(db_prefix) + "_h", ss_path + "_h"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [FoldSeek] lndb failed: {result.stderr[:200]}")
            return None

        # Step 3: _ss → FASTA 변환 (3Di 시퀀스)
        result = subprocess.run(
            [FOLDSEEK_BIN, "convert2fasta", ss_path, str(fasta_out)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [FoldSeek] convert2fasta failed: {result.stderr[:200]}")
            return None

        if not fasta_out.exists():
            print("  [FoldSeek] Output FASTA not created.")
            return None

        # Step 3: FASTA 파싱 → 3Di 시퀀스 추출
        tokens = ""
        for line in fasta_out.read_text().strip().splitlines():
            if not line.startswith(">"):
                tokens += line.strip()

        if not tokens:
            print("  [FoldSeek] Empty 3Di sequence in output.")
            return None

        return tokens.lower()


def aa_seq_to_sa_tokens(aa_seq: str, tokens_3di: str | None) -> str:
    """
    아미노산 시퀀스 + 3Di 토큰 → SaProt SA 시퀀스 포맷 변환.

    SaProt 입력 포맷:
      - 구조 있음: "MaEvKc..."  (AA upper + 3Di lower, interleaved)
      - 구조 없음: "M#E#T#..."  (AA upper + '#' placeholder)

    Args:
        aa_seq:     아미노산 시퀀스 (e.g. "MEVK...")
        tokens_3di: FoldSeek 3Di 토큰 (e.g. "adkm...") 또는 None

    Returns:
        SaProt SA 시퀀스 문자열
    """
    if tokens_3di is None:
        # 구조 정보 없음 → 플레이스홀더 사용
        return "".join(aa + "#" for aa in aa_seq)

    # 길이 mismatch 처리 (AlphaFold가 일부 잔기 누락할 수 있음)
    min_len = min(len(aa_seq), len(tokens_3di))
    if len(aa_seq) != len(tokens_3di):
        print(f"  [FoldSeek] Length mismatch: AA={len(aa_seq)}, 3Di={len(tokens_3di)} "
              f"→ truncating to {min_len}")

    sa_seq = "".join(
        aa.upper() + di.lower()
        for aa, di in zip(aa_seq[:min_len], tokens_3di[:min_len])
    )

    # 남은 잔기는 플레이스홀더
    if len(aa_seq) > min_len:
        sa_seq += "".join(aa + "#" for aa in aa_seq[min_len:])

    return sa_seq


if __name__ == "__main__":
    import sys

    if not check_foldseek():
        print("❌ foldseek not found in PATH.")
        print("   Install: https://github.com/steineggerlab/foldseek/releases")
        sys.exit(1)

    pdb = sys.argv[1] if len(sys.argv) > 1 else "cache/alphafold/P00533.pdb"
    print(f"Input PDB: {pdb}")

    tokens = extract_3di_tokens(pdb)
    if tokens:
        print(f"3Di tokens ({len(tokens)} residues): {tokens[:40]}...")
        # 예시 AA 시퀀스로 SA 변환 테스트
        dummy_aa = "M" * len(tokens)
        sa = aa_seq_to_sa_tokens(dummy_aa, tokens)
        print(f"SA format  ({len(sa)//2} residues): {sa[:40]}...")
    else:
        print("❌ Failed to extract 3Di tokens.")
