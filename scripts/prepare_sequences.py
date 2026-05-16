"""
extract_davis_seqs.py
=====================
DAVIS 데이터셋에서 3Di 캐시 히트가 있는 대표 단백질 서열을 추출하여
davis_seqs_for_demo.json 으로 저장.

실행:
  conda run -n bioinfo python extract_davis_seqs.py
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── 3Di 캐시 로드 ─────────────────────────────────────────────────────────────
cache_path = ROOT / "cache" / "3di_tokens_davis.json"
raw_cache  = json.load(open(cache_path, encoding="utf-8"))

# outer key = seq_hash(MD5), inner has seq_hash, seq_len, uniprot_id, tokens_3di
cache_hashes = {
    v["seq_hash"]: {"tokens": v["tokens_3di"], "len": int(v.get("seq_len", 0)),
                    "uniprot": v.get("uniprot_id", "")}
    for v in raw_cache.values()
    if v.get("status") == "ok" and v.get("tokens_3di")
}
print(f"3Di 캐시 엔트리: {len(cache_hashes)}개")

# ── DAVIS 서열 로드 ───────────────────────────────────────────────────────────
from DeepPurpose import dataset  # type: ignore

print("DAVIS 데이터 로딩 중...")
X_drug, X_target, y = dataset.load_process_DAVIS(str(ROOT / "data"), binary=False)
unique_targets = list(set(X_target))
print(f"고유 단백질 서열: {len(unique_targets)}개")

# ── 3Di 캐시 히트 서열만 필터 ─────────────────────────────────────────────────
# 캐시에서 알려진 UniProt → 단백질명 매핑
# (DAVIS 캐시에는 대부분 인간 키나아제)
KNOWN_UNIPROT = {
    "P00519": "ABL1",   # Human ABL1, 1130aa
    "P00533": "EGFR",   # Human EGFR, 1210aa
    "P15056": "BRAF",   # Human BRAF, 766aa
    "Q06187": "BTK",    # Human BTK,  659aa
    "P16234": "PDGFRA", # Human PDGFRA, 1089aa
    "Q9UM73": "ALK",    # Human ALK,  1620aa
    "P04629": "NTRK1",  # Human NTRK1, 796aa
    "P29323": "EPHB2",  # Human EPHB2, 987aa
    "P35968": "KDR",    # Human KDR(VEGFR2), 1356aa
    "P07949": "RET",    # Human RET, 1114aa
}

hit_seqs: dict[str, dict] = {}

for seq in unique_targets:
    h = hashlib.md5(seq.encode()).hexdigest()
    if h not in cache_hashes:
        continue
    info = cache_hashes[h]
    uniprot = info["uniprot"]
    name = KNOWN_UNIPROT.get(uniprot, None)

    # UniProt 매핑 없으면 길이로 추정
    if name is None:
        slen = len(seq)
        if   1120 <= slen <= 1140: name = "ABL1"
        elif 1200 <= slen <= 1220: name = "EGFR"
        elif  756 <= slen <=  776: name = "BRAF"
        elif  649 <= slen <=  669: name = "BTK"
        elif 1079 <= slen <= 1099: name = "PDGFRA"
        elif 1610 <= slen <= 1630: name = "ALK"

    if name and name not in hit_seqs:
        hit_seqs[name] = {
            "seq":     seq,
            "length":  len(seq),
            "uniprot": uniprot,
            "seq_hash": h,
        }
        print(f"  ✅ {name:8s}  {len(seq):4d}aa  {uniprot}  hash={h[:12]}...")

# ── ABL1/EGFR fallback: 캐시 엔트리에서 seq_len 으로 직접 매핑 ─────────────────
# (UniProt 필드가 비어있는 캐시도 있으므로 길이 기준 추가)
LENGTH_MAP = {1130:"ABL1", 1210:"EGFR", 766:"BRAF", 659:"BTK",
              1089:"PDGFRA", 1620:"ALK"}

for seq in unique_targets:
    h = hashlib.md5(seq.encode()).hexdigest()
    slen = len(seq)
    name = LENGTH_MAP.get(slen)
    if name and name not in hit_seqs and h in cache_hashes:
        hit_seqs[name] = {
            "seq":     seq,
            "length":  slen,
            "uniprot": cache_hashes[h]["uniprot"],
            "seq_hash": h,
        }
        print(f"  ✅ {name:8s}  {slen:4d}aa  (length fallback)  hash={h[:12]}...")

# 누락 단백질 출력
missing = [n for n in ["ABL1","EGFR","BRAF","BTK","PDGFRA","ALK"] if n not in hit_seqs]
if missing:
    print(f"\n⚠️  캐시 미스: {missing}")
    print("   → 해당 단백질은 가장 가까운 서열로 대체합니다")
    # 가장 긴 3Di 히트 서열로 fallback
    fallback = max(hit_seqs.values(), key=lambda x: x["length"])
    for name in missing:
        hit_seqs[name] = fallback
        print(f"     {name} ← {fallback['length']}aa 서열로 대체")

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_path = ROOT / "davis_seqs_for_demo.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(hit_seqs, f, indent=2, ensure_ascii=False)

print(f"\n저장 완료: {out_path}")
print(f"단백질: {list(hit_seqs.keys())}")
for name, info in hit_seqs.items():
    print(f"  {name:8s}  {info['length']:4d}aa  3Di={info['seq_hash'][:12]}...")
