from __future__ import annotations

from collections import Counter


def site_signature(symbols: list[str], group: list[int]) -> str:
    """One donor site's signature: an element symbol ('N'), or an element
    histogram + hapticity for a multi-atom site ('C5(h5)')."""
    group_symbols = [symbols[i] for i in group]
    if len(group_symbols) == 1:
        return group_symbols[0]
    counts = Counter(group_symbols)
    hist = "".join(f"{el}{n}" for el, n in sorted(counts.items()))
    return f"{hist}(h{len(group_symbols)})"


def donor_signature(symbols: list[str], groups: list[list[int]]) -> tuple[str, ...]:
    """A ligand's full donor signature: sorted site signatures across all
    its metal-binding groups."""
    return tuple(sorted(site_signature(symbols, group) for group in groups))


__all__ = ["site_signature", "donor_signature"]
