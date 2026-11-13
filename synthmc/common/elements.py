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
    """Whether `symbol` is in TRANSITION_METALS."""
    return symbol in TRANSITION_METALS


__all__ = ["TRANSITION_METALS", "covalent_radius", "is_transition_metal"]
