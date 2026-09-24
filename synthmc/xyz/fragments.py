"""Split a mononuclear TMC into its metal atom and individual ligand
fragments, using nothing but simple covalent-radius distance cutoffs - no
coordination-geometry classification, no hapticity-clustering logic.

The trick that makes ligand splitting itself simple: delete the metal atom
from the bond graph and take connected components of what's left. Each
component *is* a ligand, including hapto ones (a Cp ring stays together as
one component through its own ring bonds - no separate hapto-site
detection needed).

Each fragment's `donor_groups` (which of its own atoms actually touch the
metal, grouped by hapticity) does need one metal-distance check per
non-metal atom - the same covalent-radius rule as everything else here,
not a geometry/shape judgment. Two donor atoms land in the same group iff
they're bonded to each other through the ligand's own (non-metal) bonds -
identical rule to corpus/donor_sites.py's corpus-side grouping (there, via
BOS-Lig's dummy metal atom's bonds instead of a real one), so a query
ligand's donor_signature (see common/donor_signature.py) is directly
comparable to a corpus ligand's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthmc.common.elements import covalent_radius, is_transition_metal
from synthmc.common.graph_utils import group_by_mutual_bonding

# Simple, inspectable bonding rule: bonded if distance <= tolerance * (sum of
# covalent radii). 1.3 is a common, well-tested multiplier for organic
# connectivity at equilibrium geometries (similar order to what OpenBabel/
# xyz2mol-style perceivers use) - deliberately a single, easy-to-eyeball
# constant rather than anything fitted.
BOND_TOLERANCE = 1.3

# Metal-donor bonds need their own, much tighter tolerance - 1.3x on top of
# a metal's (typically large) covalent radius admits atoms that are merely
# nearby, not bonded.
#
# 1.08 is an empirical choice, not a clean derivation, and provably cannot
# be made to get every case right. There is a real, confirmed overlap
# between two populations: (a) a handful of genuinely long dative bonds
# (e.g. a THF ligand's real Y-O bond, ratio 1.123) and (b) a much more
# common and more damaging failure mode - atoms one real bond *away* from
# the true donor riding along into a spurious "hapto" group merge, because
# they independently clear the distance cutoff and are directly bonded to
# the true donor via the ligand's own chemistry (a silylamide's Si next to
# its donor N, ratio up to 1.147; an acac-type ligand's carbonyl C next to
# its donor O, ratio up to 1.147; a diselenophosphinate's central P inside
# the Se-M-Se chelate ring, ratio up to 1.098). Since some of (b) sits above
# 1.123, no cutoff gets both populations right - confirmed by directly
# measuring end-to-end corpus match rates (exact/knn/none) on a 200-
# structure sample across tolerance 1.0-1.15: 1.15 (chosen only against the
# single THF case) was in fact the *worst* value tested on every metric,
# because it let population (b) through essentially unfiltered. 1.08
# minimizes ligands with no corpus match at all on that same sample,
# accepting the rarer long-dative-bond case (like THF-Y) as a disclosed
# miss rather than chasing a single number that cannot satisfy both.
# Also avoids the much worse failure mode a looser tolerance (1.3, this
# module's ordinary organic-bond value) caused: an entire 13-atom span of
# one ligand collapsing into a single false "hapto" group because enough
# spurious bridging atoms got admitted to transitively connect otherwise-
# separate donor sites.
METAL_DONOR_BOND_TOLERANCE = 1.08


@dataclass
class LigandFragment:
    atom_indices: list[int]  # indices into the parent structure's symbols/coords
    symbols: list[str]
    coords: np.ndarray  # (n, 3)
    donor_groups: list[list[int]]  # fragment-local indices (into symbols/coords) of atoms bonded to the metal, grouped by hapticity


@dataclass
class SplitStructure:
    identifier: str
    metal_index: int
    metal_symbol: str
    ligands: list[LigandFragment]


def _pairwise_bonds(symbols: list[str], coords: np.ndarray) -> list[tuple[int, int]]:
    n = len(symbols)
    radii = np.array([covalent_radius(s) for s in symbols])
    bonds = []
    for i in range(n):
        d = np.linalg.norm(coords[i + 1 :] - coords[i], axis=1)
        cutoff = BOND_TOLERANCE * (radii[i + 1 :] + radii[i])
        for offset in np.nonzero(d <= cutoff)[0]:
            bonds.append((i, i + 1 + int(offset)))
    return bonds


def _connected_components(n: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _is_bonded_elsewhere(atom: int, dist_to_metal: float, other_coords: np.ndarray, bonds: list[tuple[int, int]]) -> bool:
    """True if `atom` already has a real covalent bond (from `bonds` - the
    ligand's own ordinary organic bonds, already computed via
    BOND_TOLERANCE) that's shorter than its distance to the metal - i.e.
    it's someone else's covalent partner, not a free candidate for the
    metal-fallback pick below.

    Same concept as coordgeo's _is_bonded_elsewhere (github.com/hkneiding/
    coordgeo, core.py) - excludes a hydrogen that's geometrically close to
    the metal only because it's caught up in an agostic-type C-H...M
    proximity, from being mistaken for a genuine M-H hydride - but reuses
    this module's own already-computed bond list rather than re-deriving
    bond-ness from a fresh radius-sum-plus-tolerance check, since that work
    is already done by the time the fallback runs."""
    for a, b in bonds:
        if a == atom:
            other = b
        elif b == atom:
            other = a
        else:
            continue
        if np.linalg.norm(other_coords[atom] - other_coords[other]) < dist_to_metal:
            return True
    return False


def split_structure(identifier: str, symbols: list[str], coords: np.ndarray) -> SplitStructure:
    """Find the (single) metal atom and split everything else into ligand
    fragments by connected components of ordinary covalent bonds.

    Raises ValueError if the structure doesn't have exactly one metal atom
    (mononuclear-only, matching the scope of the reference corpus).
    """
    metal_indices = [i for i, s in enumerate(symbols) if is_transition_metal(s)]
    if len(metal_indices) != 1:
        raise ValueError(
            f"{identifier}: expected exactly 1 transition-metal atom, found {len(metal_indices)}"
        )
    metal_index = metal_indices[0]

    other_indices = [i for i in range(len(symbols)) if i != metal_index]
    other_symbols = [symbols[i] for i in other_indices]
    other_coords = coords[other_indices]

    local_bonds = _pairwise_bonds(other_symbols, other_coords)
    components = _connected_components(len(other_indices), local_bonds)

    metal_pos = coords[metal_index]
    metal_radius = covalent_radius(symbols[metal_index])
    dists_to_metal = np.linalg.norm(other_coords - metal_pos, axis=1)
    cutoffs_to_metal = METAL_DONOR_BOND_TOLERANCE * (metal_radius + np.array([covalent_radius(s) for s in other_symbols]))
    donor_other_local = {i for i in range(len(other_indices)) if dists_to_metal[i] <= cutoffs_to_metal[i]}

    ligands = []
    for component in components:
        global_indices = [other_indices[i] for i in component]
        frag_local_of = {other_local: frag_local for frag_local, other_local in enumerate(component)}

        donor_in_component = donor_other_local & set(component)
        raw_groups = group_by_mutual_bonding(donor_in_component, local_bonds)
        donor_groups = [sorted(frag_local_of[i] for i in group) for group in raw_groups]

        if not donor_groups:
            # split_structure only ever creates a fragment from atoms left
            # over after removing the metal - it exists in this xyz frame
            # because it's part of the complex, so its closest atom to the
            # metal has *some* bond, even where METAL_DONOR_BOND_TOLERANCE's
            # single global ratio misses it (the systematically-longer-bond
            # metals, e.g. Y/La - see that constant's docstring). Falling
            # back to the nearest atom rather than leaving this fragment
            # with an empty (and therefore never-matchable) donor signature
            # - but skipping any hydrogen that's really bonded to something
            # else nearby first (see _is_bonded_elsewhere), so an agostic-
            # type C-H...M proximity doesn't get mistaken for a genuine M-H
            # hydride just because it happened to be the closest atom.
            by_distance = sorted(component, key=lambda i: dists_to_metal[i])
            nearest_other_local = next(
                (
                    i
                    for i in by_distance
                    if not (other_symbols[i] == "H" and _is_bonded_elsewhere(i, dists_to_metal[i], other_coords, local_bonds))
                ),
                by_distance[0],  # every candidate was excluded - fall back to the plain nearest anyway
            )
            donor_groups = [[frag_local_of[nearest_other_local]]]

        ligands.append(
            LigandFragment(
                atom_indices=global_indices,
                symbols=[symbols[i] for i in global_indices],
                coords=coords[global_indices],
                donor_groups=donor_groups,
            )
        )

    return SplitStructure(
        identifier=identifier,
        metal_index=metal_index,
        metal_symbol=symbols[metal_index],
        ligands=ligands,
    )


__all__ = ["LigandFragment", "SplitStructure", "split_structure", "BOND_TOLERANCE"]
