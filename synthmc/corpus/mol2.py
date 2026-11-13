from __future__ import annotations

from rdkit import Chem, RDLogger

from synthmc.common.sanitize import SANITIZE_SANS_KEKULIZATION

RDLogger.DisableLog("rdApp.*")


def parse_mol2(mol2_text: str) -> Chem.Mol:
    """Parse + sanitize one mol2 block, with a kekulization fallback for
    Cp-type aromatic rings."""
    mol = Chem.MolFromMol2Block(mol2_text, sanitize=False)
    if mol is None:
        raise ValueError("MolFromMol2Block returned None")
    try:
        Chem.SanitizeMol(mol)
    except Chem.KekulizeException:
        mol = Chem.MolFromMol2Block(mol2_text, sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_SANS_KEKULIZATION)
    return mol


__all__ = ["parse_mol2"]
