from __future__ import annotations

from rdkit import Chem

# Fallback for BOS-Lig fragments that fail kekulization (e.g. Cp-type rings).
SANITIZE_SANS_KEKULIZATION = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY

# Further fallback: also skip valence/property checking.
SANITIZE_MINIMAL = SANITIZE_SANS_KEKULIZATION ^ Chem.SANITIZE_PROPERTIES

__all__ = ["SANITIZE_SANS_KEKULIZATION", "SANITIZE_MINIMAL"]
