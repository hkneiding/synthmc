"""Morgan/ECFP4 fingerprinting - shared by corpus/ (fingerprinting a
BOS-Lig ligand parsed from mol2), scoring/evaluate.py (fingerprinting a
caller-supplied SMILES directly), and xyz/ligand_chem.py (fingerprinting a
query fragment's OpenBabel-derived SMILES). Not xyz-specific despite
living next to xyz-detection code in an earlier version of this project -
moved here once scoring/evaluate.py and corpus/ligand_corpus.py both
needed it too, which was also creating a circular import (xyz/ depending
on corpus/ via evaluate_complex, corpus/ depending back on xyz/ for this)."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_MORGAN_RADIUS = 2  # ECFP4-equivalent
_MORGAN_NBITS = 2048
_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_NBITS)


def morgan_fingerprint(mol: Chem.Mol):
    return _morgan_generator.GetFingerprint(mol)


__all__ = ["morgan_fingerprint"]
