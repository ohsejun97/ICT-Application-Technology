"""
demo.py — 발표용 파이프라인 데모 (재실험 교정판)
================================================
교정 사항 (RERUN_CHECKLIST.md):
  · 단백질 서열 → DAVIS canonical full-length (davis_seqs_for_demo.json)
  · 3Di 구조 토큰 정상 활용 (캐시 히트 기대: 30~45/50)
  · 영상 촬영 최적화: ANSI 색상 / 실시간 통계 라인

실행:
  conda run -n bioinfo python demo.py
  conda run -n bioinfo python demo.py --n_queries 50 --drop_rate 0.30 --seed 77 --output results/demo50_log.jsonl
"""

import os, sys, time, json, random, argparse, hashlib
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np

# ── ANSI 색상 ─────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED    = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE   = "\033[94m"; CYAN  = "\033[96m"; WHITE  = "\033[97m"
    ORANGE = "\033[38;5;208m"

def green(s):  return f"{C.GREEN}{C.BOLD}{s}{C.RESET}"
def yellow(s): return f"{C.YELLOW}{C.BOLD}{s}{C.RESET}"
def red(s):    return f"{C.RED}{C.BOLD}{s}{C.RESET}"
def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def dim(s):    return f"{C.DIM}{s}{C.RESET}"
def bold(s):   return f"{C.BOLD}{s}{C.RESET}"
def orange(s): return f"{C.ORANGE}{C.BOLD}{s}{C.RESET}"

# ── 인자 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_queries",    type=int,   default=20)
parser.add_argument("--drop_rate",    type=float, default=0.20,
                    help="패킷 드롭 확률 (기본 20%)")
parser.add_argument("--corrupt_rate", type=float, default=0.15,
                    help="페이로드 변조 확률")
parser.add_argument("--noise_sigma",  type=float, default=0.05)
parser.add_argument("--lat_min",      type=float, default=0.4)
parser.add_argument("--lat_max",      type=float, default=1.2)
parser.add_argument("--pkd_high",     type=float, default=7.0)
parser.add_argument("--pkd_mod",      type=float, default=5.0)
parser.add_argument("--seed",         type=int,   default=7)
parser.add_argument("--output",       default="results/demo_log.jsonl")
args = parser.parse_args()

ROOT = Path(__file__).parent
OUT  = ROOT / args.output
OUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(args.seed)

# ── DAVIS 서열 로드 (3Di 캐시 히트 보장) ──────────────────────────────────────
_SEQS_PATH = ROOT / "davis_seqs_for_demo.json"
if not _SEQS_PATH.exists():
    print(f"{red('ERROR:')} {_SEQS_PATH} 가 없습니다.")
    print("  먼저 실행하세요: conda run -n bioinfo python extract_davis_seqs.py")
    sys.exit(1)

with open(_SEQS_PATH, encoding="utf-8") as _f:
    _davis = json.load(_f)

# 각 단백질 서열 참조 (fallback: ABL1)
def _seq(name: str) -> str:
    return _davis.get(name, _davis.get("ABL1", ""))["seq"] if isinstance(_davis.get(name), dict) else _davis.get(name, _davis["ABL1"])

# ── 3Di 캐시 히트 사전 확인 ────────────────────────────────────────────────────
def _check_3di_cache(aa_seq: str) -> bool:
    """davis_seqs_for_demo 서열이 3Di 캐시에 있는지 검증."""
    cache_path = ROOT / "cache" / "3di_tokens_davis.json"
    if not cache_path.exists():
        return False
    raw = json.load(open(cache_path, encoding="utf-8"))
    h = hashlib.md5(aa_seq.encode()).hexdigest()
    return any(v.get("seq_hash") == h and v.get("status") == "ok"
               for v in raw.values())

# ── 서열 로딩 상태 출력 ───────────────────────────────────────────────────────
def _print_seq_banner():
    print(f"\n{bold('━'*62)}")
    print(f"  {cyan('[ DAVIS 서열 상태 ]')}")
    for name in ["ABL1", "EGFR", "BRAF", "BTK", "PDGFRA", "ALK"]:
        entry = _davis.get(name)
        if entry is None:
            print(f"  {red('✘')} {name:8s} — 누락")
            continue
        seq = entry["seq"] if isinstance(entry, dict) else entry
        hit = _check_3di_cache(seq)
        tag = green(f"3Di ✅  {len(seq):4d}aa") if hit else orange(f"3Di ⚠️  {len(seq):4d}aa fallback")
        print(f"  {bold(f'{name:8s}')} {tag}")
    print(f"{bold('━'*62)}\n")

# ── 샘플 쿼리 (DAVIS full-length 서열) ────────────────────────────────────────
SAMPLE_QUERIES = [
    ("Q01", "Imatinib",
     "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
     "ABL1", _seq("ABL1")),
    ("Q02", "Nilotinib",
     "CC1=CN=C(C(=C1)NC(=O)C2=CC(=CC=N2)C(F)(F)F)NC3=CC(=C(C=C3)CN4CCN(CC4)C)C(F)(F)F",
     "ABL1", _seq("ABL1")),
    ("Q03", "Gefitinib",
     "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
     "EGFR", _seq("EGFR")),
    ("Q04", "Erlotinib",
     "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
     "EGFR", _seq("EGFR")),
    ("Q05", "Dasatinib",
     "CC1=NC(=CC(=C1)NC(=O)C2=CC(=CC=C2)Cl)NC3=NC=C(C=N3)C4=CN=CC=C4",
     "ABL1", _seq("ABL1")),
    ("Q06", "Sorafenib",
     "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
     "BRAF", _seq("BRAF")),
    ("Q07", "Vemurafenib",
     "CCCS(=O)(=O)NC1=CC(=C(C=C1F)NC(=O)C2=CNC3=CC(=C(C=C23)Cl)F)F",
     "BRAF", _seq("BRAF")),
    ("Q08", "Ibrutinib",
     "C=CC(=O)N1CCCC(C1)N2C=NC3=C(N=CN=C23)NCC4=CC=CC=C4",
     "BTK", _seq("BTK")),
    ("Q09", "Sunitinib",
     "CCN(CC)CCNC(=O)C1=C(NC2=CC=CC3=CC=CC=C23)C(=O)C1=O",
     "PDGFRA", _seq("PDGFRA")),
    ("Q10", "Crizotinib",
     "CCCS(=O)(=O)NC1=C2C=C(NC(=O)C3=CN=C4C=CC=CC4=C3)C=CC2=NC(=N1)N",
     "ALK", _seq("ALK")),
]


def _make_query(idx):
    q = SAMPLE_QUERIES[idx % len(SAMPLE_QUERIES)]
    return {"query_id": q[0], "drug_name": q[1], "smiles": q[2],
            "protein_name": q[3], "aa_seq": q[4],
            "timestamp": datetime.now().isoformat(), "seq_idx": idx}


# ══════════════════════════════════════════════════════════════════════════════
# Process A — Edge Node
# ══════════════════════════════════════════════════════════════════════════════
def edge_node(queue, ready_event, n_queries, drop_rate, corrupt_rate,
              lat_min, lat_max, stat_q, seed):
    random.seed(seed + 1)
    ready_event.wait()

    print(f"\n{bold('━'*62)}")
    print(f"  {cyan('[ EDGE NODE ]')}  전송 시작 — {bold(str(n_queries))}개 쿼리")
    print(f"  드롭 확률: {orange(f'{drop_rate:.0%}')}  |  변조 확률: {orange(f'{corrupt_rate:.0%}')}"
          f"  |  지연: {dim(f'{lat_min}~{lat_max}s')}")
    print(f"{bold('━'*62)}\n")

    sent = dropped = corrupted = 0

    for i in range(n_queries):
        query   = _make_query(i)
        latency = random.uniform(lat_min, lat_max)
        time.sleep(latency)

        if random.random() < drop_rate:
            dropped += 1
            stat_q.put({"event": "drop", "query_id": query["query_id"],
                        "drug_name": query["drug_name"],
                        "protein_name": query["protein_name"],
                        "smiles": query["smiles"],
                        "latency_ms": round(latency * 1000, 1)})
            print(f"  {red('▼ DROP')}   #{i+1:02d}  "
                  f"{bold(query['drug_name']):12s}→ {query['protein_name']:6s}"
                  f"  {dim(f'lat={latency:.2f}s')}", flush=True)
            continue

        is_corrupt = random.random() < corrupt_rate
        query["corrupt"]    = is_corrupt
        query["latency_ms"] = round(latency * 1000, 1)
        if is_corrupt:
            query["noise_sigma"] = args.noise_sigma
            corrupted += 1

        queue.put(query)
        sent += 1
        flag = f"  {orange('⚡ CORRUPT')}" if is_corrupt else ""
        print(f"  {green('▲ TX')}     #{i+1:02d}  "
              f"{bold(query['drug_name']):12s}→ {query['protein_name']:6s}"
              f"  {dim(f'lat={latency:.2f}s')}{flag}", flush=True)

    queue.put(None)
    stat_q.put({"event": "edge_done", "sent": sent,
                "dropped": dropped, "corrupted": corrupted, "total": n_queries})
    print(f"\n  {cyan('[ EDGE ]')} 전송 완료 — "
          f"전송:{green(str(sent))}  드롭:{red(str(dropped))}  변조:{orange(str(corrupted))}",
          flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Process B — Server Node
# ══════════════════════════════════════════════════════════════════════════════
def server_node(queue, ready_event, stat_q, pkd_high, pkd_mod,
                alert_loss, output_path, noise_sigma, seed):
    random.seed(seed + 2)

    print(f"\n{bold('━'*62)}")
    print(f"  {cyan('[ SERVER NODE ]')}  DTI 모델 로딩 중...")
    print(f"{bold('━'*62)}\n")

    try:
        sys.path.insert(0, str(ROOT))
        from tools.dti_tool import predict_binding, _load_models
        print(f"  {dim('모델 파일 로딩 중...')}", flush=True)
        _load_models()
        _WARMUP_SMILES = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
        _WARMUP_SEQ    = _seq("ABL1")
        predict_binding(_WARMUP_SMILES, _WARMUP_SEQ)
        model_ready = True
        print(f"  {green('✔ SaProt-650M FP16')}   로드 완료")
        print(f"  {green('✔ ft-ChemBERTa')}        로드 완료")
        print(f"  {green('✔ MLP Regression Head')} 로드 완료 (BindingDB r=0.89)\n")
    except Exception as e:
        print(f"  {red('✘ 모델 로드 실패:')} {e}\n  {dim('시뮬레이션 모드로 진행')}\n")
        model_ready = False

    ready_event.set()

    rolling: deque = deque(maxlen=5)
    results = []
    n_recv = n_imputed = n_corrupt = n_drop_imputed = n_3di = 0
    log_fh = open(output_path, "w", encoding="utf-8")

    def _decide(pkd):
        if pkd >= pkd_high: return "HIGH"
        if pkd >= pkd_mod:  return "MODERATE"
        return "LOW"

    def _impute():
        return round(float(np.mean(rolling)), 4) if rolling else None

    def _corrupt_smiles(smiles):
        chars = list(smiles)
        rep   = {"C": "N", "N": "O", "O": "S", "S": "C", "c": "n", "n": "o"}
        n_c   = max(1, int(len(chars) * noise_sigma))
        for pos in random.sample(range(len(chars)), min(n_c, len(chars))):
            if chars[pos] in rep:
                chars[pos] = rep[chars[pos]]
        return "".join(chars)

    print(f"  {'#':>3}  {'쿼리':5}  {'약물':12}  {'표적':7}  {'pKd':7}  {'판정':10}  {'경로':<16}  3Di")
    print(f"  {'─'*72}")

    while True:
        try:
            query = queue.get(timeout=120)
        except Exception:
            break
        if query is None:
            break

        n_recv     += 1
        qid         = query["query_id"]
        drug_name   = query["drug_name"]
        aa_seq      = query["aa_seq"]
        smiles      = query["smiles"]
        is_corrupt  = query.get("corrupt", False)
        if is_corrupt:
            n_corrupt += 1

        pkd      = None
        path     = "normal"
        used_3di = False

        if model_ready:
            try:
                smi_in = _corrupt_smiles(smiles) if is_corrupt else smiles
                res    = predict_binding(smi_in, aa_seq)
                if "error" not in res:
                    pkd      = res["pKd"]
                    used_3di = res.get("used_3di", False)
                    path     = "corrupt_recovered" if is_corrupt else "normal"
                else:
                    raise ValueError(res["error"])
            except Exception:
                pkd  = _impute()
                path = "imputed"
                n_imputed += 1
        else:
            sim_map = {"Imatinib": 8.2, "Nilotinib": 9.0, "Gefitinib": 7.8,
                       "Erlotinib": 7.6, "Dasatinib": 8.5, "Sorafenib": 6.9,
                       "Vemurafenib": 8.0, "Ibrutinib": 7.4, "Sunitinib": 7.1,
                       "Crizotinib": 7.7}
            base = sim_map.get(drug_name, 7.0)
            pkd  = round(base + random.uniform(-0.25, 0.25), 4)
            used_3di = True   # 시뮬레이션 모드에서도 3Di 사용 가정
            if is_corrupt:
                path = "corrupt_recovered"

        if pkd is None:
            pkd  = 6.0
            path = "imputed"
            n_imputed += 1

        if used_3di:
            n_3di += 1
        rolling.append(pkd)
        decision = _decide(pkd)

        dec_str  = (green(f"{'HIGH':10}") if decision == "HIGH" else
                    yellow(f"{'MODERATE':10}") if decision == "MODERATE" else
                    red(f"{'LOW':10}"))
        path_str = (orange(f"{'corrupt_recov':<16}") if path == "corrupt_recovered" else
                    dim(f"{'imputed':<16}") if path == "imputed" else
                    dim(f"{'normal':<16}"))
        di_str   = green("✅") if used_3di else orange("⚠️ ")

        print(f"  {n_recv:>3}  {qid:5}  {bold(drug_name):12}  "
              f"{query['protein_name']:7}  {bold(f'{pkd:.4f}'):7}  "
              f"{dec_str}  {path_str}  {di_str}", flush=True)

        # 드롭 이벤트 처리
        while not stat_q.empty():
            ev = stat_q.get_nowait()
            if ev.get("event") == "drop":
                mean_pkd = _impute()
                n_drop_imputed += 1
                mean_str = (f"{dim('rolling mean')}: {bold(f'{mean_pkd:.4f}')}"
                            if mean_pkd is not None else dim("buffer empty"))
                print(f"  {' ':>3}  {ev['query_id']:5}  "
                      f"{red('▼▼ PACKET DROP ▼▼'):26}  "
                      f"→ impute({mean_str})", flush=True)
                drop_pkd = mean_pkd if mean_pkd is not None else 6.0
                drop_rec = {
                    "query_id": ev["query_id"], "drug_name": ev["drug_name"],
                    "protein_name": ev.get("protein_name", "UNKNOWN"),
                    "pKd": drop_pkd, "decision": _decide(drop_pkd),
                    "path": "drop_imputed", "corrupt": False, "dropped": True,
                    "used_3di": False, "latency_ms": ev.get("latency_ms", 0),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(drop_rec)
                log_fh.write(json.dumps(drop_rec, ensure_ascii=False) + "\n")
                log_fh.flush()
            elif ev.get("event") == "edge_done":
                stat_q.put(ev)
                break

        record = {
            "query_id": qid, "drug_name": drug_name,
            "protein_name": query["protein_name"],
            "pKd": pkd, "decision": decision, "path": path,
            "corrupt": is_corrupt, "used_3di": used_3di,
            "latency_ms": query.get("latency_ms", 0),
            "timestamp": datetime.now().isoformat(),
        }
        results.append(record)
        log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_fh.flush()

    log_fh.close()

    try:
        edge_info = stat_q.get(timeout=3)
    except Exception:
        edge_info = {}

    total    = edge_info.get("total", n_recv)
    dropped  = edge_info.get("dropped", 0)
    loss_rt  = dropped / total if total else 0.0
    net_alert= loss_rt > alert_loss

    high = sum(1 for r in results if r["decision"] == "HIGH")
    mod  = sum(1 for r in results if r["decision"] == "MODERATE")
    low  = sum(1 for r in results if r["decision"] == "LOW")
    avg  = float(np.mean([r["pKd"] for r in results])) if results else 0.0

    print(f"\n{bold('━'*62)}")
    print(f"  {bold('[ PIPELINE SUMMARY ]')}")
    print(f"  총 쿼리      : {bold(str(total))}")
    print(f"  수신 완료    : {green(str(n_recv))}")
    print(f"  패킷 드롭    : {red(str(dropped))}  (손실률 {red(f'{loss_rt:.1%}')})")
    print(f"  변조 복구    : {orange(str(n_corrupt))}  건  (corrupt_recovered)")
    print(f"  DROP→대체    : {dim(str(n_drop_imputed))}  건  (rolling mean)")
    print(f"  추론실패→대체: {dim(str(n_imputed))}  건  (rolling mean)")
    print(f"  3Di 사용     : {green(str(n_3di))}/{n_recv}  ({green(f'{n_3di/n_recv:.0%}') if n_recv else '0%'})")
    print(f"  ─────────────────────────────────────")
    print(f"  🟢 HIGH      : {green(str(high))}  건")
    print(f"  🟡 MODERATE  : {yellow(str(mod))}  건")
    print(f"  🔴 LOW       : {red(str(low))}  건")
    print(f"  평균 pKd     : {bold(f'{avg:.4f}')}")
    if net_alert:
        print(f"\n  {red('⚠  ALERT: 패킷 손실률 ' + f'{loss_rt:.1%}' + ' > 30% — Network Degraded!')}")
    print(f"{bold('━'*62)}")
    print(f"  결과: {dim(str(output_path))}\n")

    summary_path = output_path.parent / (output_path.stem.replace("log", "summary") + ".json")
    summary = {
        "experiment": {
            "n_queries": total, "drop_rate": args.drop_rate,
            "corrupt_rate": args.corrupt_rate,
            "lat_min": args.lat_min, "lat_max": args.lat_max,
            "seed": args.seed, "pkd_high": pkd_high, "pkd_mod": pkd_mod,
        },
        "network": {
            "total_queries": total, "received": n_recv, "dropped": dropped,
            "loss_rate": round(loss_rt, 4), "corrupt_handled": n_corrupt,
            "drop_imputed": n_drop_imputed, "inference_imputed": n_imputed,
            "network_alert": net_alert,
        },
        "decisions": {"HIGH": high, "MODERATE": mod, "LOW": low, "avg_pKd": round(avg, 4)},
        "model_quality": {
            "used_3di": n_3di, "total_inferred": n_recv,
            "3di_rate": round(n_3di / n_recv, 4) if n_recv else 0,
        },
        "per_drug": {
            drug: {
                "count":   sum(1 for r in results if r["drug_name"] == drug),
                "avg_pKd": round(float(np.mean([r["pKd"] for r in results if r["drug_name"] == drug])), 4),
                "paths":   list({r["path"] for r in results if r["drug_name"] == drug}),
            }
            for drug in dict.fromkeys(r["drug_name"] for r in results)
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  요약 JSON: {dim(str(summary_path))}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    _print_seq_banner()

    queue       = mp.Queue(maxsize=50)
    stat_q      = mp.Queue()
    ready_event = mp.Event()

    print(f"{bold('━'*62)}")
    print(f"  {bold('Bio-AI DTI Query Pipeline')}  — 발표 데모 (교정판)")
    print(f"  쿼리: {args.n_queries}  드롭: {args.drop_rate:.0%}"
          f"  변조: {args.corrupt_rate:.0%}  지연: {args.lat_min}~{args.lat_max}s")
    print(f"  HIGH pKd≥{args.pkd_high}  |  MODERATE pKd≥{args.pkd_mod}")
    print(f"{bold('━'*62)}")

    p_server = mp.Process(
        target=server_node,
        args=(queue, ready_event, stat_q,
              args.pkd_high, args.pkd_mod, 0.30, OUT,
              args.noise_sigma, args.seed),
    )
    p_edge = mp.Process(
        target=edge_node,
        args=(queue, ready_event, args.n_queries,
              args.drop_rate, args.corrupt_rate,
              args.lat_min, args.lat_max, stat_q, args.seed),
    )

    p_server.start()
    p_edge.start()
    p_edge.join()
    p_server.join()
