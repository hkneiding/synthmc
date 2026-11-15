from __future__ import annotations

from ase.data import atomic_numbers, covalent_radii


def covalent_radius(symbol: str) -> float:
    """Single-bond covalent radius in Angstrom."""
    z = atomic_numbers[symbol]
    r = covalent_radii[z]
    if r <= 0:
        raise ValueError(f"No covalent radius tabulated for element {symbol!r}")
    return float(r)


__all__ = ["covalent_radius"]
