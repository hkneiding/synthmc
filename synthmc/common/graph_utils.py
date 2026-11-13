from __future__ import annotations


def group_by_mutual_bonding(nodes: set[int] | list[int], bonds: list[tuple[int, int]]) -> list[list[int]]:
    """Partition `nodes` into groups of mutually (directly or
    transitively) bonded atoms, via `bonds`."""
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
