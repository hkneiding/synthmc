"""A ligand's donor-atom signature: what actually touches the metal,
independent of the rest of the ligand's structure - the thing precedent
matching should require to agree between a query ligand and a corpus
candidate, per-group.

Deliberately a plain function of (symbols, index groups) with no toolkit
dependency, so the exact same code computes it for a corpus ligand (donor
atom indices come from BOS-Lig's spec_mol2 dummy metal atom, see
corpus/donor_sites.py) and a query ligand (donor atom indices come from
which fragment atoms were bonded to the real metal before it was removed,
see xyz/fragments.py) - there's no other way to be sure both sides mean the
same thing by "same binding atoms".
"""

from __future__ import annotations

from collections import Counter


def site_signature(symbols: list[str], group: list[int]) -> str:
    """One donor site's signature: the element symbol for a plain
    single-atom donor ('N'), or an element histogram + hapticity for a
    multi-atom (hapto) site ('C5(h5)' for a Cp ring, 'C6(h6)' for an
    arene, ...)."""
    group_symbols = [symbols[i] for i in group]
    if len(group_symbols) == 1:
        return group_symbols[0]
    counts = Counter(group_symbols)
    hist = "".join(f"{el}{n}" for el, n in sorted(counts.items()))
    return f"{hist}(h{len(group_symbols)})"


def donor_signature(symbols: list[str], groups: list[list[int]]) -> tuple[str, ...]:
    """A ligand's full donor signature: sorted site signatures across all
    its metal-binding groups, e.g. ('N','N','N','N') for four separate
    monodentate N donors, or ('C5(h5)',) for one eta5 ring."""
    return tuple(sorted(site_signature(symbols, group) for group in groups))


__all__ = ["site_signature", "donor_signature"]
