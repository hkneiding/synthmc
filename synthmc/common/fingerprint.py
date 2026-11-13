from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_MORGAN_RADIUS = 2  # ECFP4-equivalent
_MORGAN_NBITS = 2048
_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_NBITS)


def morgan_fingerprint(mol: Chem.Mol):
    """ECFP4-equivalent Morgan fingerprint of `mol`."""
    return _morgan_generator.GetFingerprint(mol)


__all__ = ["morgan_fingerprint"]
