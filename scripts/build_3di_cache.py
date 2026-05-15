"""
build_3di_cache.py
==================
DAVIS/KIBA 단백질 시퀀스에 대한 FoldSeek 3Di 토큰 캐시 빌더.

Pipeline per protein:
  AA 시퀀스
    → UniProt ID 조회 (EBI BLAST API)
    → AlphaFold DB PDB 다운로드
    → FoldSeek 3Di 토큰 추출
    → 캐시 저장 (cache/3di_tokens_{dataset}.json)

실패한 단백질은 'status': 'failed' 로 기록 → 학습 시 '#' 플레이스홀더 사용.

Usage:
  conda run -n bioinfo python scripts/build_3di_cache.py --dataset davis
  conda run -n bioinfo python scripts/build_3di_cache.py --dataset kiba
  conda run -n bioinfo python scripts/build_3di_cache.py --dataset davis --resume
"""

import sys
import json
import time
import hashlib
import argparse
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.alphafold_tool import fetch_alphafold_structure
from tools.foldseek_tool import extract_3di_tokens, check_foldseek

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Build FoldSeek 3Di token cache")
parser.add_argument("--dataset",  default="davis",
                    choices=["davis", "kiba", "bindingdb", "davis+bindingdb"])
parser.add_argument("--resume",   action="store_true",
                    help="Resume from existing cache (skip already-done entries)")
parser.add_argument("--max_workers", type=int, default=1,
                    help="Concurrent BLAST jobs (default: 1, EBI 권장 최대 3)")
args = parser.parse_args()

CACHE_PATH = Path(f"./cache/3di_tokens_{args.dataset}.json")
CACHE_PATH.parent.mkdir(exist_ok=True)

# ── EBI BLAST API 설정 ─────────────────────────────────────────────────────────
EBI_BLAST_RUN    = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run"
EBI_BLAST_STATUS = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/{}"
EBI_BLAST_RESULT = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/{}/json"


def seq_hash(seq: str) -> str:
    """시퀀스의 MD5 해시 (캐시 키)."""
    return hashlib.md5(seq.encode()).hexdigest()


def blast_sequence_to_uniprot(seq: str, retries: int = 3) -> str | None:
    """
    EBI BLAST API로 아미노산 시퀀스 → UniProt accession 조회.

    Returns:
        UniProt accession (e.g. "P00533") 또는 None.
    """
    # 너무 짧은 시퀀스는 BLAST 불가
    if len(seq) < 20:
        return None

    for attempt in range(retries):
        try:
            # Job 제출
            resp = requests.post(
                EBI_BLAST_RUN,
                data={
                    "email":    "bioai.capstone@gmail.com",
                    "program":  "blastp",
                    "database": "uniprotkb_swissprot",
                    "sequence": ">protein\n" + seq,
                    "stype":    "protein",
                    "matrix":   "BLOSUM62",
                    "exp":      "1e-10",
                    "filter":   "F",
                    "gapalign": "true",
                    "alignments": 5,
                    "scores":   5,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                time.sleep(5 * (attempt + 1))
                continue

            job_id = resp.text.strip()
            if not job_id:
                continue

            # 완료 대기 (최대 120초)
            for _ in range(40):
                time.sleep(3)
                status_resp = requests.get(
                    EBI_BLAST_STATUS.format(job_id), timeout=15
                )
                status = status_resp.text.strip()
                if status == "FINISHED":
                    break
                if status in ("FAILED", "ERROR", "DELETED"):
                    return None
            else:
                return None  # timeout

            # 결과 파싱
            result_resp = requests.get(
                EBI_BLAST_RESULT.format(job_id), timeout=30
            )
            if result_resp.status_code != 200:
                return None

            data = result_resp.json()
            hits = data.get("hits", [])
            if not hits:
                return None

            # 첫 번째 hit에서 UniProt accession 추출 (hit_acc 필드)
            acc = hits[0].get("hit_acc", "")
            if acc:
                return acc
            # fallback: hit_def 파싱 "SP:Q2M2I8 ..."
            hit_def = hits[0].get("hit_def", "")
            parts = hit_def.split()
            if parts and ":" in parts[0]:
                return parts[0].split(":")[1]
            return None

        except Exception as e:
            print(f"    BLAST attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))

    return None


def process_protein(seq: str, idx: int, total: int, uniprot_id: str = None) -> dict:
    """
    단일 단백질 처리: BLAST → AlphaFold → FoldSeek.

    Returns:
        {
          "seq_hash": ...,
          "seq_len": ...,
          "uniprot_id": ...,      # None if not found
          "tokens_3di": ...,      # None if failed
          "coverage": ...,        # 3Di 커버리지 (0~1)
          "status": "ok" | "no_uniprot" | "no_structure" | "no_3di" | "failed"
        }
    """
    h = seq_hash(seq)
    prefix = f"  [{idx}/{total}]"

    # ── 1. UniProt ID 조회 (직접 제공된 경우 BLAST 스킵) ──────────────────
    if uniprot_id:
        print(f"{prefix} UniProt ID 직접 사용: {uniprot_id} (BLAST 스킵)", flush=True)
    else:
        print(f"{prefix} BLAST ({len(seq)} aa)...", flush=True)
        uniprot_id = blast_sequence_to_uniprot(seq)

    if uniprot_id is None:
        print(f"{prefix} ❌ UniProt ID 조회 실패 → '#' 플레이스홀더 사용")
        return {"seq_hash": h, "seq_len": len(seq), "uniprot_id": None,
                "tokens_3di": None, "coverage": 0.0, "status": "no_uniprot"}

    print(f"{prefix} UniProt: {uniprot_id}", flush=True)

    # ── 2. AlphaFold PDB 다운로드 ──────────────────────────────────────────
    af_result = fetch_alphafold_structure(uniprot_id)
    if "error" in af_result:
        print(f"{prefix} ❌ AlphaFold 없음: {af_result['error']}")
        return {"seq_hash": h, "seq_len": len(seq), "uniprot_id": uniprot_id,
                "tokens_3di": None, "coverage": 0.0, "status": "no_structure"}

    pdb_path = af_result["pdb_path"]
    print(f"{prefix} PDB: {pdb_path}  pLDDT={af_result.get('plddt_global','?')}", flush=True)

    # ── 3. FoldSeek 3Di 토큰 추출 ─────────────────────────────────────────
    tokens_3di = extract_3di_tokens(pdb_path)
    if tokens_3di is None:
        print(f"{prefix} ❌ FoldSeek 실패")
        return {"seq_hash": h, "seq_len": len(seq), "uniprot_id": uniprot_id,
                "tokens_3di": None, "coverage": 0.0, "status": "no_3di"}

    # 커버리지: 얼마나 많은 잔기에 3Di 토큰이 있는지
    coverage = min(len(tokens_3di), len(seq)) / len(seq)
    print(f"{prefix} ✅ 3Di: {len(tokens_3di)} tokens  coverage={coverage:.1%}", flush=True)

    return {
        "seq_hash":  h,
        "seq_len":   len(seq),
        "uniprot_id": uniprot_id,
        "tokens_3di": tokens_3di,
        "coverage":  round(coverage, 4),
        "status":    "ok",
    }


def main():
    # ── FoldSeek 확인 ──────────────────────────────────────────────────────
    if not check_foldseek():
        print("❌ foldseek not found in PATH.")
        print("   설치: https://github.com/steineggerlab/foldseek/releases")
        sys.exit(1)

    # ── 데이터셋 로드 ──────────────────────────────────────────────────────
    print(f"[1] {args.dataset.upper()} 데이터셋 로드...")
    if args.dataset in ("davis", "kiba", "davis+bindingdb"):
        try:
            import DeepPurpose.dataset as dp_dataset
        except ImportError:
            sys.exit("❌ pip install DeepPurpose")

    if args.dataset == "davis":
        X_drugs, X_targets, _ = dp_dataset.load_process_DAVIS(
            path="./data", binary=False, convert_to_log=True
        )
    elif args.dataset == "kiba":
        X_drugs, X_targets, _ = dp_dataset.load_process_KIBA(
            path="./data", binary=False, threshold=9
        )
    elif args.dataset in ("bindingdb", "davis+bindingdb"):
        import pandas as pd
        _bdb = pd.read_csv("./data/BindingDB/bindingdb_kd.csv")
        # sequence → uniprot_id 매핑 (BLAST 스킵용)
        _uid_map = dict(zip(_bdb["sequence"], _bdb["uniprot_id"]))
        _uid_map = {k: v for k, v in _uid_map.items() if isinstance(v, str) and v.strip()}
        if args.dataset == "bindingdb":
            X_targets = _bdb["sequence"].tolist()
        else:
            _, X_t_davis, _ = dp_dataset.load_process_DAVIS(
                path="./data", binary=False, convert_to_log=True
            )
            X_targets = X_t_davis + _bdb["sequence"].tolist()

    unique_seqs = list(dict.fromkeys(X_targets))
    print(f"    고유 단백질: {len(unique_seqs)}개\n")

    # ── 기존 캐시 로드 (--resume) ──────────────────────────────────────────
    cache: dict[str, dict] = {}
    if args.resume and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        done = sum(1 for v in cache.values() if v["status"] == "ok")
        print(f"[2] 기존 캐시 로드: {len(cache)}개 항목 (ok={done})\n")
    else:
        print("[2] 새 캐시 생성\n")

    # ── 처리 ──────────────────────────────────────────────────────────────
    total = len(unique_seqs)
    n_ok = n_fail = 0

    # BindingDB: sequence → uniprot_id 매핑 (없으면 빈 dict)
    uid_map = _uid_map if args.dataset in ("bindingdb", "davis+bindingdb") else {}

    for i, seq in enumerate(unique_seqs, 1):
        h = seq_hash(seq)

        # 이미 처리된 항목 스킵 (resume 모드)
        if args.resume and h in cache and cache[h]["status"] == "ok":
            n_ok += 1
            continue

        result = process_protein(seq, i, total, uniprot_id=uid_map.get(seq))
        cache[h] = result

        if result["status"] == "ok":
            n_ok += 1
        else:
            n_fail += 1

        # 중간 저장 (10개마다)
        if i % 10 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            print(f"  → 중간 저장 완료 ({i}/{total})", flush=True)

        # EBI API 부하 방지
        time.sleep(1)

    # ── 최종 저장 ──────────────────────────────────────────────────────────
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    # ── 요약 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  데이터셋:  {args.dataset.upper()}")
    print(f"  총 단백질: {total}")
    print(f"  ✅ 성공:   {n_ok} ({n_ok/total:.1%})")
    print(f"  ❌ 실패:   {n_fail} ({n_fail/total:.1%})")
    by_status = {}
    for v in cache.values():
        by_status[v["status"]] = by_status.get(v["status"], 0) + 1
    for k, cnt in sorted(by_status.items()):
        print(f"    {k}: {cnt}")
    print(f"  캐시 저장: {CACHE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
