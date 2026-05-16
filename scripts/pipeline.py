"""
pipeline.py
===========
Bio-AI DTI Query Pipeline — ICT 네트워크 시뮬레이션 + 실시간 추론

Process A (Edge Node) → [WAN 시뮬레이션] → Process B (Server Node)
  - Latency : sleep(0.5~2.0 s)
  - Drop     : 15% 확률
  - Corrupt  : Gaussian noise σ=0.05 (drug embedding)

실행:
  conda run -n bioinfo python pipeline.py
  conda run -n bioinfo python pipeline.py --n_queries 50 --drop_rate 0.20
"""

import os, sys, time, json, random, argparse, math
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np

# ── 인자 ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_queries",   type=int,   default=30,   help="전송할 쿼리 수")
parser.add_argument("--drop_rate",   type=float, default=0.15, help="패킷 드롭 확률")
parser.add_argument("--noise_sigma", type=float, default=0.05, help="Gaussian 노이즈 σ")
parser.add_argument("--lat_min",     type=float, default=0.5,  help="최소 지연 (초)")
parser.add_argument("--lat_max",     type=float, default=2.0,  help="최대 지연 (초)")
parser.add_argument("--pkd_high",    type=float, default=7.0,  help="HIGH 결합 임계값")
parser.add_argument("--pkd_mod",     type=float, default=5.0,  help="MODERATE 결합 임계값")
parser.add_argument("--alert_loss",  type=float, default=0.30, help="Network Alert 임계 손실률")
parser.add_argument("--output",      default="results/pipeline_log.jsonl", help="결과 로그 파일")
args = parser.parse_args()

ROOT = Path(__file__).parent.parent
OUT  = ROOT / args.output
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 샘플 DTI 쿼리 (DAVIS 대표 쌍) ───────────────────────────────────────────────
SAMPLE_QUERIES = [
    # (query_id, drug_name, SMILES, protein_name, AA_sequence_partial)
    ("Q001","Imatinib",
     "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
     "ABL1",
     "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGD"
     "NTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVR"),
    ("Q002","Gefitinib",
     "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
     "EGFR",
     "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLS"
     "FLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNN"),
    ("Q003","Erlotinib",
     "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
     "EGFR",
     "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLS"
     "FLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNN"),
    ("Q004","Dasatinib",
     "CC1=NC(=CC(=C1)NC(=O)C2=CC(=CC=C2)Cl)NC3=NC=C(C=N3)C4=CN=CC=C4",
     "ABL1",
     "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGD"
     "NTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVR"),
    ("Q005","Sorafenib",
     "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
     "BRAF",
     "MAHHHHHHHHHHHSSGVDLGTENLYFQSNAMHTTVKTLRDLSRDAQLHSATPNLNALFGSSSSQFQSQNIPSSSSL"
     "SSSFERESQNRQHSEAQEQSLSRQRSSSSSMSSSSLASSSGSSSSSGSSSSVSHSSSSGSSSSSGSSSSDGSSSSS"),
    ("Q006","Sunitinib",
     "CCN(CC)CCNC(=O)C1=C(NC2=CC=CC3=CC=CC=C23)C(=O)C1=O",
     "PDGFRA",
     "MGSSHHHHHHHHHGSACEESVGPEAPQRSLEKAKLNFQTIPFVLTQKFNQLPIFSPFASSNRQPEQSPLRFQDIED"
     "GIDLNLEQPEVFLSQEISNLPYLDPVVVQSREALLSQPLKIEEGQKLADLFSQESGPKEKSFESLTLPAFKQRYE"),
    ("Q007","Vemurafenib",
     "CCCS(=O)(=O)NC1=CC(=C(C=C1F)NC(=O)C2=CNC3=CC(=C(C=C23)Cl)F)F",
     "BRAF",
     "MAHHHHHHHHHHHSSGVDLGTENLYFQSNAMHTTVKTLRDLSRDAQLHSATPNLNALFGSSSSQFQSQNIPSSSSL"
     "SSSFERESQNRQHSEAQEQSLSRQRSSSSSMSSSSLASSSGSSSSSGSSSSVSHSSSSGSSSSSGSSSSDGSSSSS"),
    ("Q008","Ibrutinib",
     "C=CC(=O)N1CCCC(C1)N2C=NC3=C(N=CN=C23)NCC4=CC=CC=C4",
     "BTK",
     "MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSYYEYDFERDMFMLNLNDRIEGMSEGKKLRMLLERIINYL"
     "QEEEALHKPINGEDILQKLDNGLYLNQRHSVDVKFRPFKQDIKETLKQNMTLHEQYEELIKQFEIFLQDNQKQTV"),
    ("Q009","Nilotinib",
     "CC1=CN=C(C(=C1)NC(=O)C2=CC(=CC=N2)C(F)(F)F)NC3=CC(=C(C=C3)CN4CCN(CC4)C)C(F)(F)F",
     "ABL1",
     "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGD"
     "NTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVR"),
    ("Q010","Crizotinib",
     "CC(=O)NC1=C(C=CC(=C1)OCC2=C(C=CN=C2Cl)F)F",
     "ALK",
     "MALREEEQLSAGPGQPRLLCSVQPPPARGGPAAGGKRPPAEAGESSRDPRSSQLPPAAAAGPSRPLEQPQQLSTPLP"
     "QPQQPPPPPPQNSSSSQPPLPQDNSSTAAASAQPLVLQRVAANLVTPPLSPVTQPPPQTQPFVTPPSKSPNQANHR"),
]


def _make_query(idx: int) -> dict:
    q = SAMPLE_QUERIES[idx % len(SAMPLE_QUERIES)]
    return {
        "query_id":    q[0] + f"_{idx:03d}",
        "drug_name":   q[1],
        "smiles":      q[2],
        "protein_name":q[3],
        "aa_seq":      q[4],
        "timestamp":   datetime.now().isoformat(),
        "seq_idx":     idx,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Process A — Edge Node
# ══════════════════════════════════════════════════════════════════════════════
def edge_node(queue: mp.Queue, n_queries: int, drop_rate: float,
              noise_sigma: float, lat_min: float, lat_max: float,
              stats_queue: mp.Queue):
    """DTI 쿼리를 생성하고 WAN을 통해 서버로 전송 (시뮬레이션)."""
    sent = dropped = corrupted = 0
    print(f"\n[Edge Node] 시작 — {n_queries}개 쿼리 전송 예정", flush=True)

    for i in range(n_queries):
        query = _make_query(i)

        # ── 지연 시뮬레이션 ─────────────────────────────────────────────────────
        latency = random.uniform(lat_min, lat_max)
        time.sleep(latency)

        # ── 패킷 드롭 ──────────────────────────────────────────────────────────
        if random.random() < drop_rate:
            dropped += 1
            stats_queue.put({"event": "drop", "query_id": query["query_id"],
                             "drug_name": query["drug_name"]})
            print(f"  [Edge→DROP]  #{i+1:03d} {query['query_id']} "
                  f"({query['drug_name']}) dropped", flush=True)
            continue

        # ── 페이로드 변조 ──────────────────────────────────────────────────────
        is_corrupt = random.random() < 0.10   # 10% 변조
        query["corrupt"] = is_corrupt
        query["latency_ms"] = round(latency * 1000, 1)
        if is_corrupt:
            query["noise_sigma"] = noise_sigma
            corrupted += 1

        queue.put(query)
        sent += 1
        flag = "⚡CORRUPT" if is_corrupt else "OK"
        print(f"  [Edge→TX]    #{i+1:03d} {query['query_id']} "
              f"({query['drug_name']}) lat={latency:.2f}s [{flag}]", flush=True)

    queue.put(None)   # sentinel
    stats_queue.put({
        "event": "edge_done",
        "sent": sent, "dropped": dropped, "corrupted": corrupted,
        "total": n_queries,
    })
    print(f"\n[Edge Node] 완료 — 전송:{sent} 드롭:{dropped} 변조:{corrupted}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI Inference (Server Side)
# ══════════════════════════════════════════════════════════════════════════════
def _load_dti_tool():
    sys.path.insert(0, str(ROOT))
    from tools.dti_tool import predict_binding
    return predict_binding


def _apply_corruption(smiles: str, sigma: float) -> str:
    """SMILES 문자열에 랜덤 문자 치환으로 노이즈 시뮬레이션 (구조 정보 손상)."""
    chars = list(smiles)
    replacements = {"C": "N", "N": "O", "O": "S", "S": "C", "c": "n", "n": "o"}
    n_corrupt = max(1, int(len(chars) * sigma))
    positions = random.sample(range(len(chars)), min(n_corrupt, len(chars)))
    for pos in positions:
        c = chars[pos]
        if c in replacements:
            chars[pos] = replacements[c]
    return "".join(chars)


# ══════════════════════════════════════════════════════════════════════════════
# Process B — Server Node
# ══════════════════════════════════════════════════════════════════════════════
def server_node(queue: mp.Queue, stats_queue: mp.Queue,
                pkd_high: float, pkd_mod: float,
                alert_loss: float, output_path: Path,
                noise_sigma: float):
    """수신된 쿼리를 버퍼링하고 AI 추론 + 결과 판정."""

    print("\n[Server Node] 시작 — DTI 모델 로드 중...", flush=True)
    try:
        predict_binding = _load_dti_tool()
        model_ready = True
        print("[Server Node] 모델 로드 완료\n", flush=True)
    except Exception as e:
        print(f"[Server Node] 모델 로드 실패: {e}\n  → 시뮬레이션 모드로 진행", flush=True)
        model_ready = False

    rolling_pkd: deque = deque(maxlen=5)   # imputation용 rolling buffer
    results = []
    n_received = n_imputed = n_corrupt_handled = 0
    log_fh = open(output_path, "w", encoding="utf-8")

    def _write_log(record: dict):
        log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_fh.flush()

    def _decide(pkd: float) -> str:
        if pkd >= pkd_high: return "HIGH"
        if pkd >= pkd_mod:  return "MODERATE"
        return "LOW"

    def _impute() -> float | None:
        if not rolling_pkd:
            return None
        return round(float(np.mean(rolling_pkd)), 4)

    while True:
        try:
            query = queue.get(timeout=60)
        except Exception:
            break
        if query is None:
            break

        n_received += 1
        qid       = query["query_id"]
        drug_name = query["drug_name"]
        smiles    = query["smiles"]
        aa_seq    = query["aa_seq"]
        is_corrupt= query.get("corrupt", False)

        print(f"  [Server←RX]  {qid} ({drug_name})"
              f"{' [CORRUPT]' if is_corrupt else ''}", flush=True)

        # ── AI 추론 ────────────────────────────────────────────────────────────
        pkd = None
        path = "normal"
        used_3di = False

        if is_corrupt:
            n_corrupt_handled += 1
            smiles_noisy = _apply_corruption(smiles, noise_sigma)
            if model_ready:
                try:
                    res = predict_binding(smiles_noisy, aa_seq)
                    if "error" not in res:
                        pkd = res["pKd"]
                        used_3di = res.get("used_3di", False)
                        path = "corrupt_recovered"
                    else:
                        pkd = _impute()
                        path = "imputed"
                        n_imputed += 1
                except Exception:
                    pkd = _impute()
                    path = "imputed"
                    n_imputed += 1
            else:
                pkd = _impute() or round(random.uniform(4.5, 8.5), 4)
                path = "imputed"
                n_imputed += 1
        else:
            if model_ready:
                try:
                    res = predict_binding(smiles, aa_seq)
                    if "error" not in res:
                        pkd = res["pKd"]
                        used_3di = res.get("used_3di", False)
                    else:
                        pkd = _impute()
                        path = "imputed"
                        n_imputed += 1
                except Exception:
                    pkd = _impute()
                    path = "imputed"
                    n_imputed += 1
            else:
                # 시뮬레이션 모드: 약물별 pKd 근사
                sim_map = {
                    "Imatinib": 8.2, "Gefitinib": 7.8, "Erlotinib": 7.6,
                    "Dasatinib": 8.5, "Sorafenib": 6.9, "Sunitinib": 7.1,
                    "Vemurafenib": 8.0, "Ibrutinib": 7.4, "Nilotinib": 8.1,
                    "Crizotinib": 7.7,
                }
                base = sim_map.get(drug_name, 7.0)
                pkd  = round(base + random.uniform(-0.3, 0.3), 4)

        if pkd is None:
            pkd = 6.0
            path = "imputed"
            n_imputed += 1

        rolling_pkd.append(pkd)
        decision = _decide(pkd)

        record = {
            "query_id":    qid,
            "drug_name":   drug_name,
            "protein_name":query["protein_name"],
            "pKd":         pkd,
            "decision":    decision,
            "path":        path,
            "corrupt":     is_corrupt,
            "used_3di":    used_3di,
            "latency_ms":  query.get("latency_ms", 0),
            "timestamp":   datetime.now().isoformat(),
        }
        results.append(record)
        _write_log(record)

        badge = {"HIGH": "🟢 HIGH", "MODERATE": "🟡 MODERATE", "LOW": "🔴 LOW"}[decision]
        print(f"  [Server→AI]  {qid}  pKd={pkd:.4f}  {badge}"
              f"  [{path}]{'  3Di✅' if used_3di else ''}", flush=True)

    log_fh.close()

    # ── 최종 통계 ──────────────────────────────────────────────────────────────
    try:
        edge_info = stats_queue.get_nowait()
    except Exception:
        edge_info = {}

    total_sent  = edge_info.get("total", n_received)
    dropped     = edge_info.get("dropped", 0)
    loss_rate   = dropped / total_sent if total_sent > 0 else 0.0
    net_alert   = loss_rate > alert_loss

    high   = sum(1 for r in results if r["decision"] == "HIGH")
    mod    = sum(1 for r in results if r["decision"] == "MODERATE")
    low    = sum(1 for r in results if r["decision"] == "LOW")
    avg_pkd = float(np.mean([r["pKd"] for r in results])) if results else 0.0

    summary = {
        "total_queries":   total_sent,
        "received":        n_received,
        "dropped":         dropped,
        "loss_rate":       round(loss_rate, 4),
        "corrupt_handled": n_corrupt_handled,
        "imputed":         n_imputed,
        "HIGH":  high, "MODERATE": mod, "LOW": low,
        "avg_pKd":         round(avg_pkd, 4),
        "network_alert":   net_alert,
        "timestamp":       datetime.now().isoformat(),
    }

    summary_path = OUT.parent / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  [Pipeline] 완료")
    print(f"  총 쿼리:   {total_sent}")
    print(f"  수신:      {n_received}")
    print(f"  드롭:      {dropped}  (손실률 {loss_rate:.1%})")
    print(f"  변조 복구: {n_corrupt_handled}")
    print(f"  대체 추론: {n_imputed}")
    print(f"  HIGH:      {high}  MODERATE: {mod}  LOW: {low}")
    print(f"  평균 pKd:  {avg_pkd:.4f}")
    if net_alert:
        print(f"  ⚠️  ALERT: 패킷 손실률 {loss_rate:.1%} > {alert_loss:.0%} — Network Degraded!")
    print(f"  결과 저장: {OUT}")
    print(f"  요약 저장: {summary_path}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    queue       = mp.Queue(maxsize=100)
    stats_queue = mp.Queue()

    p_edge = mp.Process(
        target=edge_node,
        args=(queue, args.n_queries, args.drop_rate, args.noise_sigma,
              args.lat_min, args.lat_max, stats_queue),
    )
    p_server = mp.Process(
        target=server_node,
        args=(queue, stats_queue, args.pkd_high, args.pkd_mod,
              args.alert_loss, OUT, args.noise_sigma),
    )

    print("=" * 60)
    print("  Bio-AI DTI Query Pipeline")
    print(f"  쿼리 수:    {args.n_queries}")
    print(f"  드롭 확률:  {args.drop_rate:.0%}")
    print(f"  노이즈 σ:   {args.noise_sigma}")
    print(f"  지연:       {args.lat_min}~{args.lat_max}s")
    print(f"  HIGH임계:   pKd ≥ {args.pkd_high}")
    print(f"  출력:       {OUT}")
    print("=" * 60)

    p_server.start()
    p_edge.start()

    p_edge.join()
    p_server.join()
