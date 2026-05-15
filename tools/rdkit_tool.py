"""
rdkit_tool.py
=============
RDKit 3D Ligand Tool — Agent Tool #3

Given a SMILES string, generates a 3D conformer using RDKit
(ETKDG algorithm + MMFF94 force field optimization).

Returns:
  - Molecule metadata (name, formula, mol_weight, n_atoms, n_bonds)
  - SDF file path (saved and cached locally)

Usage (standalone test):
  python tools/rdkit_tool.py "CC1=CC=CC=C1"              # toluene
  python tools/rdkit_tool.py "CC(=O)Oc1ccccc1C(=O)O"    # aspirin
"""

import sys
import hashlib
from pathlib import Path

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("❌ pip install rdkit")

CACHE_DIR = Path(__file__).parent.parent / "cache" / "ligands"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _smiles_to_key(smiles: str) -> str:
    """Canonical SMILES → short hash key for cache filename."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def generate_3d_ligand(smiles: str, name: str = None) -> dict:
    """
    Generate a 3D conformer from a SMILES string.

    Args:
        smiles: SMILES string (e.g., "CC(=O)Oc1ccccc1C(=O)O")
        name:   Optional molecule name for the SDF file

    Returns:
        dict with keys:
          smiles, canonical_smiles, name, formula, mol_weight,
          n_atoms, n_bonds, n_rotatable_bonds,
          sdf_path, status
    """
    smiles = smiles.strip()

    # ── Parse SMILES ─────────────────────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: '{smiles}'", "smiles": smiles}

    canonical_smiles = Chem.MolToSmiles(mol)
    cache_key = _smiles_to_key(smiles)
    sdf_path  = CACHE_DIR / f"{cache_key}.sdf"

    # ── Check cache ───────────────────────────────────────────────
    if sdf_path.exists():
        print(f"  [RDKit] Cache hit: {cache_key}")
        mol_name = name or cache_key
        mol3d = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
        return _build_result(smiles, canonical_smiles, mol3d, mol_name, sdf_path, cached=True)

    # ── Add hydrogens & generate 3D conformer ────────────────────
    print(f"  [RDKit] Generating 3D conformer ...")
    mol_h = Chem.AddHs(mol)

    result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if result == -1:
        return {"error": "3D embedding failed (ETKDG). Try a different SMILES.",
                "smiles": smiles}

    # ── MMFF94 force field optimization ──────────────────────────
    ff_result = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=2000)
    status = "optimized" if ff_result == 0 else "embedding_only"

    # ── Save SDF ──────────────────────────────────────────────────
    mol_name = name or cache_key
    mol_h.SetProp("_Name", mol_name)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol_h)
    writer.close()
    print(f"  [RDKit] Saved: {sdf_path}")

    return _build_result(smiles, canonical_smiles, mol_h, mol_name, sdf_path,
                         cached=False, status=status)


def _build_result(smiles, canonical_smiles, mol, name, sdf_path, cached=False, status="cached"):
    formula      = rdMolDescriptors.CalcMolFormula(mol)
    mol_weight   = round(Descriptors.MolWt(mol), 3)
    n_atoms      = mol.GetNumAtoms()
    n_bonds      = mol.GetNumBonds()
    n_rot        = rdMolDescriptors.CalcNumRotatableBonds(mol)

    return {
        "smiles":           smiles,
        "canonical_smiles": canonical_smiles,
        "name":             name,
        "formula":          formula,
        "mol_weight":       mol_weight,
        "n_atoms":          n_atoms,
        "n_bonds":          n_bonds,
        "n_rotatable_bonds":n_rot,
        "sdf_path":         str(sdf_path),
        "status":           status,
        "cached":           cached,
    }


def format_result(r: dict) -> str:
    """Human-readable summary for Agent output."""
    if "error" in r:
        return f"[RDKit Tool] Error: {r['error']}"

    cached_tag = " (cached)" if r.get("cached") else ""
    return (
        f"[RDKit Tool]{cached_tag}\n"
        f"  Name     : {r['name']}\n"
        f"  SMILES   : {r['smiles']}\n"
        f"  Formula  : {r['formula']}  |  MW: {r['mol_weight']} g/mol\n"
        f"  Atoms    : {r['n_atoms']}  |  Bonds: {r['n_bonds']}  "
        f"|  Rotatable: {r['n_rotatable_bonds']}\n"
        f"  Status   : {r['status']}\n"
        f"  SDF      : {r['sdf_path']}"
    )


if __name__ == "__main__":
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)Oc1ccccc1C(=O)O"
    name   = sys.argv[2] if len(sys.argv) > 2 else None
    result = generate_3d_ligand(smiles, name)
    print(format_result(result))
