"""
alphafold_tool.py
=================
AlphaFold DB API Tool — Agent Tool #2

Given a UniProt accession ID, fetches the predicted 3D protein structure
from the AlphaFold Protein Structure Database (EBI).

Returns:
  - Protein metadata (name, gene, organism, sequence length, pLDDT)
  - PDB file path (downloaded and cached locally)

Usage (standalone test):
  python tools/alphafold_tool.py P00533
"""

import os
import sys
import requests
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache" / "alphafold"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AF_API = "https://alphafold.ebi.ac.uk/api/prediction"


def fetch_alphafold_structure(uniprot_id: str) -> dict:
    """
    Query AlphaFold DB for a protein structure by UniProt accession ID.

    Args:
        uniprot_id: UniProt accession (e.g., "P00533" for EGFR)

    Returns:
        dict with keys:
          uniprot_id, gene, name, organism, seq_length,
          plddt_global, plddt_very_high_frac,
          pdb_path, pdb_url, entry_id
    """
    uniprot_id = uniprot_id.strip().upper()

    # ── Check local cache ────────────────────────────────────────
    pdb_path = CACHE_DIR / f"{uniprot_id}.pdb"
    meta_path = CACHE_DIR / f"{uniprot_id}_meta.txt"

    if pdb_path.exists() and meta_path.exists():
        print(f"  [AlphaFold] Cache hit: {uniprot_id}")
        meta = {}
        for line in meta_path.read_text().splitlines():
            k, _, v = line.partition("=")
            meta[k] = v
        meta["pdb_path"] = str(pdb_path)
        meta["cached"] = True
        return meta

    # ── Query AlphaFold DB API ───────────────────────────────────
    print(f"  [AlphaFold] Querying API: {uniprot_id} ...")
    try:
        resp = requests.get(f"{AF_API}/{uniprot_id}", timeout=15)
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}", "uniprot_id": uniprot_id}

    if resp.status_code == 404:
        return {"error": f"UniProt ID '{uniprot_id}' not found in AlphaFold DB.",
                "uniprot_id": uniprot_id}
    if resp.status_code != 200:
        return {"error": f"API returned status {resp.status_code}",
                "uniprot_id": uniprot_id}

    data = resp.json()[0]

    # ── Download PDB file ────────────────────────────────────────
    pdb_url = data["pdbUrl"]
    print(f"  [AlphaFold] Downloading PDB: {pdb_url} ...")
    try:
        pdb_resp = requests.get(pdb_url, timeout=30)
        pdb_resp.raise_for_status()
        pdb_path.write_bytes(pdb_resp.content)
    except requests.exceptions.RequestException as e:
        return {"error": f"PDB download failed: {e}", "uniprot_id": uniprot_id}

    # ── Build result ─────────────────────────────────────────────
    seq_len = data.get("uniprotEnd", len(data.get("uniprotSequence", "")))
    result = {
        "uniprot_id":          uniprot_id,
        "entry_id":            data.get("entryId", ""),
        "gene":                data.get("gene", "N/A"),
        "name":                data.get("uniprotDescription", "N/A"),
        "organism":            data.get("organismScientificName", "N/A"),
        "seq_length":          str(seq_len),
        "plddt_global":        str(round(data.get("globalMetricValue", 0), 2)),
        "plddt_very_high_frac":str(round(data.get("fractionPlddtVeryHigh", 0), 3)),
        "pdb_url":             pdb_url,
        "pdb_path":            str(pdb_path),
        "cached":              False,
    }

    # ── Save metadata cache ──────────────────────────────────────
    meta_path.write_text(
        "\n".join(f"{k}={v}" for k, v in result.items() if k != "pdb_path")
    )

    return result


def format_result(r: dict) -> str:
    """Human-readable summary for Agent output."""
    if "error" in r:
        return f"[AlphaFold Tool] Error: {r['error']}"

    cached_tag = " (cached)" if r.get("cached") else ""
    return (
        f"[AlphaFold Tool]{cached_tag}\n"
        f"  Protein  : {r['name']} ({r['gene']})\n"
        f"  UniProt  : {r['uniprot_id']}  |  Entry: {r['entry_id']}\n"
        f"  Organism : {r['organism']}\n"
        f"  Length   : {r['seq_length']} aa\n"
        f"  pLDDT    : {r['plddt_global']} (global)  |  "
        f"Very-high conf: {float(r['plddt_very_high_frac'])*100:.1f}%\n"
        f"  PDB      : {r['pdb_path']}"
    )


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "P00533"
    result = fetch_alphafold_structure(uid)
    print(format_result(result))
