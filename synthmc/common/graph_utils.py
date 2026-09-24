"""Shared graph utility: group a subset of atom indices by mutual bonding.

Used on the corpus side (corpus/donor_sites.py, clustering a spec_mol2's
donor atoms into hapto sites via its dummy metal atom's bonds), the
xyz-detection side (xyz/fragments.py, clustering a fragment's
metal-distance-flagged atoms), and the core API (scoring/evaluate.py,
clustering a caller-supplied flat list of binding-atom indices) - same
rule everywhere: two atoms belong to the same site iff they're bonded to
each other, directly or transitively, through the ligand's own bonds.
"""

from __future__ import annotations


def group_by_mutual_bonding(nodes: set[int] | list[int], bonds: list[tuple[int, int]]) -> list[list[int]]:
    """Partition `nodes` into groups where two indices land in the same
    group iff they're bonded to each other (directly or transitively) via
    `bonds` - e.g. a cyclopentadienyl ring's 5 carbons collapse into one
    group; four separate, mutually-unbonded donor atoms stay four
    singletons."""
    nodes = set(nodes)
    adjacency: dict[int, set[int]] = {}
    for a, b in bonds:
        if a in nodes and b in nodes:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    remaining = set(nodes)
    groups = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        frontier = [seed]
        while frontier:
            cur = frontier.pop()
            for neighbor in adjacency.get(cur, ()):
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    group.add(neighbor)
                    frontier.append(neighbor)
        groups.append(sorted(group))
    return groups


__all__ = ["group_by_mutual_bonding"]
