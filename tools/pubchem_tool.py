"""
pubchem_tool.py
===============
Drug Name Resolver — Agent Tool #4

Given a drug name (common name, brand name, or synonym),
resolves it to a canonical SMILES string via the PubChem REST API.

This enables general users to query by drug name instead of SMILES.

Examples:
  "Imatinib"   → CC1=C(C=C(C=C1)NC(=O)c1ccc(cc1)CN1CCN(CC1)C)C(=O)Nc1ccc(cc1)N1CCN(CC1)C
  "아스피린"    → (LLM should translate to "Aspirin" first)
  "Aspirin"    → CC(=O)Oc1ccccc1C(=O)O
  "Viagra"     → (brand name — may fail; use "Sildenafil" instead)
  "Sildenafil" → CCCC1=NN(C2=CC(=C(C=C21)S(=O)(=O)N3CCN(CC3)C)OCC)C

Usage (standalone test):
  python tools/pubchem_tool.py Imatinib
  python tools/pubchem_tool.py Aspirin
  python tools/pubchem_tool.py Sildenafil
"""

import sys
import json
import requests
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache" / "pubchem"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON"


def resolve_drug_name(name: str) -> dict:
    """
    Resolve a drug name to SMILES and basic properties via PubChem.

    Args:
        name: Drug name (e.g., "Imatinib", "Aspirin", "Sildenafil")
              - Common names, generic names, and some brand names supported
              - Korean names should be translated to English first

    Returns:
        dict with keys:
          query_name, name, cid, smiles, formula, mol_weight, iupac_name,
          pubchem_url, cached
        or dict with key 'error' on failure
    """
    name = name.strip()
    cache_key = name.lower().replace(" ", "_")
    cache_path = CACHE_DIR / f"{cache_key}.json"

    # ── Check cache ───────────────────────────────────────────────
    if cache_path.exists():
        print(f"  [PubChem] Cache hit: {name}")
        data = json.loads(cache_path.read_text())
        data["cached"] = True
        return data

    # ── Query PubChem API ─────────────────────────────────────────
    print(f"  [PubChem] Querying: {name} ...")
    url = PUBCHEM_URL.format(name=requests.utils.quote(name))
    try:
        resp = requests.get(url, timeout=15)
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}", "query_name": name}

    if resp.status_code == 404:
        return {
            "error": f"Drug '{name}' not found in PubChem. "
                     f"Try the generic name (e.g., 'Sildenafil' instead of 'Viagra').",
            "query_name": name,
        }
    if resp.status_code != 200:
        return {"error": f"PubChem API returned status {resp.status_code}", "query_name": name}

    props = resp.json()["PropertyTable"]["Properties"][0]

    result = {
        "query_name":  name,
        "name":        name,
        "cid":         str(props.get("CID", "")),
        "smiles":      props.get("IsomericSMILES") or props.get("SMILES", ""),
        "formula":     props.get("MolecularFormula", ""),
        "mol_weight":  str(round(float(props.get("MolecularWeight", 0)), 3)),
        "iupac_name":  props.get("IUPACName", ""),
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{props.get('CID', '')}",
        "cached":      False,
    }

    # ── Save cache ────────────────────────────────────────────────
    cache_path.write_text(json.dumps({k: v for k, v in result.items() if k != "cached"},
                                     ensure_ascii=False, indent=2))
    return result


def format_result(r: dict) -> str:
    """Human-readable summary for Agent output."""
    if "error" in r:
        return f"[PubChem Tool] Error: {r['error']}"

    cached_tag = " (cached)" if r.get("cached") else ""
    return (
        f"[PubChem Tool]{cached_tag}\n"
        f"  Query    : {r['query_name']}\n"
        f"  CID      : {r['cid']}  |  {r['pubchem_url']}\n"
        f"  Formula  : {r['formula']}  |  MW: {r['mol_weight']} g/mol\n"
        f"  SMILES   : {r['smiles']}\n"
        f"  IUPAC    : {r['iupac_name']}"
    )


if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else "Aspirin"
    result = resolve_drug_name(drug)
    print(format_result(result))
