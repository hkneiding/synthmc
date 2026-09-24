"""Element reference data used for bond perception.

Radii are pulled from ASE's data table (Cordero et al. 2008 covalent
radii) rather than hand-transcribed, to avoid silent transcription errors
in a table that directly controls bond perception correctness.
"""

from __future__ import annotations

from ase.data import atomic_numbers, covalent_radii

TRANSITION_METALS = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "La",  # tmQM includes lanthanum-containing Werner-type complexes
    }
)


def covalent_radius(symbol: str) -> float:
    """Single-bond covalent radius in Angstrom."""
    z = atomic_numbers[symbol]
    r = covalent_radii[z]
    if r <= 0:
        raise ValueError(f"No covalent radius tabulated for element {symbol!r}")
    return float(r)


def is_transition_metal(symbol: str) -> bool:
    return symbol in TRANSITION_METALS


__all__ = ["TRANSITION_METALS", "covalent_radius", "is_transition_metal"]
