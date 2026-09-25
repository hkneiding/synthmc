"""Shared mol2-block parsing for BOS-Lig data (gen_mol2 and spec_mol2 both
use the same format) - split out from ligand_corpus.py so donor_sites.py
can use it too without a circular import (ligand_corpus.py builds the
corpus using donor_sites.py's donor-group extraction)."""

from __future__ import annotations

from rdkit import Chem, RDLogger

from synthmc.common.sanitize import SANITIZE_SANS_KEKULIZATION

RDLogger.DisableLog("rdApp.*")


def parse_mol2(mol2_text: str) -> Chem.Mol:
    """Parse + sanitize one mol2 block (Hs stay explicit, no atoms removed
    - see ligand_corpus.py's module docstring for why explicit-H matters),
    with a targeted fallback for the dominant real failure mode found when
    this was first run against the full BOS-Lig set (~66,810 ligands):
    odd-membered all-carbon aromatic rings (Cp-type organometallic
    fragments - cyclopentadienyl etc.) fail RDKit's default Kekulization
    because the mol2 doesn't carry the formal negative charge that would
    make a Kekule structure exist (consistent with the "ZeroCharges" tag in
    every gen_mol2/spec_mol2 header). That's a genuine, understood RDKit
    limitation on this input, not noise, so on a KekulizeException
    specifically (and only that exception) this retries with
    kekulization/aromaticity-perception skipped, keeping the ring's
    declared aromaticity as-is rather than trying to recompute it."""
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
