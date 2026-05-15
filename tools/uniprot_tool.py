"""
uniprot_tool.py
===============
Protein Name Resolver — Agent Tool #5

Given a gene name or protein name, resolves it to a UniProt accession ID
and retrieves the amino acid sequence via the UniProt Search API.

This enables general users to query by gene/protein name instead of
requiring a UniProt accession ID.

Examples:
  "EGFR"   → P00533, 1210 aa
  "ABL1"   → P00519, 1130 aa
  "COX-1"  → P23219 (searches as PTGS1)
  "HMGCR"  → P04035 (HMG-CoA reductase, Lipitor target)
  "PDE5A"  → O76074

Note:
  - Searches human proteins by default (organism: Homo sapiens, taxon 9606)
  - For viral/bacterial proteins, set organism_id=None (broader search)
  - Returns the top-ranked canonical entry from UniProt

Usage (standalone test):
  python tools/uniprot_tool.py EGFR
  python tools/uniprot_tool.py ABL1
  python tools/uniprot_tool.py HMGCR
"""

import sys
import json
import requests
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache" / "uniprot"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL  = "https://rest.uniprot.org/uniprotkb/{accession}"


def resolve_protein_name(name: str, organism_id: int = 9606) -> dict:
    """
    Resolve a gene/protein name to UniProt accession ID and amino acid sequence.

    Args:
        name:        Gene name or protein name (e.g., "EGFR", "ABL1", "HMGCR")
        organism_id: NCBI taxonomy ID (default: 9606 = Homo sapiens)
                     Set to None to search all organisms (e.g., for viral proteins)

    Returns:
        dict with keys:
          query_name, uniprot_id, gene, name, organism, seq_length,
          sequence, reviewed, cached
        or dict with key 'error' on failure
    """
    name = name.strip()
    org_tag = str(organism_id) if organism_id else "all"
    cache_key = f"{name.lower().replace(' ', '_')}_{org_tag}"
    cache_path = CACHE_DIR / f"{cache_key}.json"

    # ── Check cache ───────────────────────────────────────────────
    if cache_path.exists():
        print(f"  [UniProt] Cache hit: {name}")
        data = json.loads(cache_path.read_text())
        data["cached"] = True
        return data

    # ── Search UniProt ────────────────────────────────────────────
    print(f"  [UniProt] Searching: {name} ...")
    query = f"gene:{name}"
    if organism_id:
        query += f" AND organism_id:{organism_id}"
    query += " AND reviewed:true"   # Swiss-Prot only (high quality)

    params = {
        "query":  query,
        "fields": "accession,gene_names,protein_name,organism_name,sequence",
        "format": "json",
        "size":   1,
    }

    try:
        resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}", "query_name": name}

    results = resp.json().get("results", [])

    # ── Fallback: relax to TrEMBL if no Swiss-Prot hit ───────────
    if not results:
        print(f"  [UniProt] No reviewed entry — retrying without reviewed filter ...")
        params["query"] = params["query"].replace(" AND reviewed:true", "")
        try:
            resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error on retry: {e}", "query_name": name}

    if not results:
        hint = f" Try a different name or set organism_id=None for non-human proteins." \
               if organism_id else ""
        return {"error": f"Protein '{name}' not found in UniProt.{hint}", "query_name": name}

    entry = results[0]
    accession = entry["primaryAccession"]
    gene      = entry.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A")
    prot_name = (entry.get("proteinDescription", {})
                      .get("recommendedName", {})
                      .get("fullName", {})
                      .get("value", "N/A"))
    if prot_name == "N/A":
        prot_name = (entry.get("proteinDescription", {})
                          .get("submissionNames", [{}])[0]
                          .get("fullName", {})
                          .get("value", "N/A"))
    organism  = entry.get("organism", {}).get("scientificName", "N/A")
    sequence  = entry.get("sequence", {}).get("value", "")
    reviewed  = entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

    result = {
        "query_name":  name,
        "uniprot_id":  accession,
        "gene":        gene,
        "name":        prot_name,
        "organism":    organism,
        "seq_length":  str(len(sequence)),
        "sequence":    sequence,
        "reviewed":    reviewed,
        "cached":      False,
    }

    # ── Save cache ────────────────────────────────────────────────
    cache_path.write_text(json.dumps({k: v for k, v in result.items() if k != "cached"},
                                     ensure_ascii=False, indent=2))
    return result


def format_result(r: dict) -> str:
    """Human-readable summary for Agent output."""
    if "error" in r:
        return f"[UniProt Tool] Error: {r['error']}"

    cached_tag  = " (cached)" if r.get("cached") else ""
    reviewed_tag = " [Swiss-Prot]" if r.get("reviewed") else " [TrEMBL]"
    seq_preview = r["sequence"][:40] + "..." if len(r["sequence"]) > 40 else r["sequence"]
    return (
        f"[UniProt Tool]{cached_tag}\n"
        f"  Query    : {r['query_name']}\n"
        f"  UniProt  : {r['uniprot_id']}{reviewed_tag}\n"
        f"  Gene     : {r['gene']}  |  Protein: {r['name']}\n"
        f"  Organism : {r['organism']}\n"
        f"  Length   : {r['seq_length']} aa\n"
        f"  Sequence : {seq_preview}"
    )


if __name__ == "__main__":
    gene = sys.argv[1] if len(sys.argv) > 1 else "EGFR"
    org  = int(sys.argv[2]) if len(sys.argv) > 2 else 9606
    result = resolve_protein_name(gene, organism_id=org)
    print(format_result(result))
