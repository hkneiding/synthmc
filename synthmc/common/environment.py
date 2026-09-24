"""A donor site's local structural environment, out to a given bond-graph
radius - the building block for corpus/environment_corpus.py's full
coordination-environment precedent (radius 0 = the site's own atoms and
their mutual bonds only; radius 1 = plus everything one bond further out;
and so on for radii 2 and 3).

Represented as canonical SMILES (not abstract SMARTS query syntax) for
consistency with the rest of this project - canonical SMILES already
encodes formal charge and explicit H count directly. Radius 0 does *not*
end up distinguishing hybridization/aromaticity (e.g. pyridine-type N vs.
amine-type N) - this is standard, expected behavior for any radius-based
circular environment (the same is true of ECFP0 vs. ECFP2 in ordinary
Morgan fingerprinting), not a gap specific to this implementation.

A bond that gets cut at the radius boundary is dropped entirely, with no
dummy-atom marker and no degree preservation - whatever's left of a
truncated atom gets RDKit's normal implicit-H filling, same as any other
atom. An earlier version marked cuts with a dummy atom specifically to
preserve degree (see project conversation/memory for the full argument),
on the reasoning that two chemically different environments - e.g. a
ring-embedded donor whose neighbor continues into the rest of the ring,
vs. an open-chain donor whose neighbor terminates there - would otherwise
canonicalize identically at whatever the largest computed radius happens
to be. That's still true; this is a deliberate choice to accept that
collision risk in exchange for a higher exact-match rate at each radius
(more structures collapse into the same bucket - more hits, but some hits
now mean less than a "real" match would). Measure the actual effect on
match rates and decide from there, rather than reason about it in the
abstract - see project conversation for the comparison.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# First fallback: same salvage used throughout this project (mol2.py,
# ligand_corpus.py) for the same underlying reason - a radius cut through
# an aromatic ring routinely leaves atoms that can't be kekulized as a
# standalone fragment, even though the original whole molecule was
# perfectly valid.
_SANITIZE_SANS_KEKULIZATION = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY

# Second, last-resort fallback: also skip valence/property checking. Found
# a real (if rare) source-data case this needs: an O atom with formal
# charge -2 bonded via an RDKit "unspecified" bond type (an unusual
# phosphazene-type fragment in one BOS-Lig entry) - valid in the original
# whole molecule's own sanitize context, but which a freshly-sanitized
# isolated fragment can reject on valence grounds alone, independent of
# kekulization. This needs to run over ~150k complexes without crashing on
# rare-but-real oddities in the source data, so the final fallback accepts
# the fragment as-is rather than raising - still a valid, comparable
# canonical string for this exact-match precedent-counting purpose even if
# not a chemically pristine independent molecule.
_SANITIZE_MINIMAL = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY ^ Chem.SANITIZE_PROPERTIES


def _atoms_within_radius(mol: Chem.Mol, group: list[int], radius: int) -> set[int]:
    """Multi-source BFS: every atom within `radius` bonds of *any* atom in
    `group`, group members themselves included at distance 0. Using a
    multi-source BFS (not per-atom radius queries unioned afterward)
    matters specifically at radius=0: a multi-atom (hapto) group's own
    internal bonds must always be included regardless of radius - they're
    zero bonds away from the group as a whole, even though a single-atom
    radius-0 query around any one member alone would see nothing."""
    distance = {i: 0 for i in group}
    frontier = list(group)
    while frontier:
        next_frontier = []
        for atom_idx in frontier:
            d = distance[atom_idx]
            if d >= radius:
                continue
            for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
                other = bond.GetOtherAtomIdx(atom_idx)
                if other not in distance:
                    distance[other] = d + 1
                    next_frontier.append(other)
        frontier = next_frontier
    return set(distance)


def site_bulk(mol: Chem.Mol, group: list[int], radius: int) -> int:
    """Heavy-atom count (explicit H excluded) within `radius` bonds of
    `group` - a purely topological steric-bulk proxy, computed straight off
    the same atom set site_environment uses, with no conformer generation
    or 3D geometry involved. See corpus/steric_corpus.py and
    scoring/matching.py's find_steric_plausibility for how this is
    aggregated and normalized into the steric-plausibility score term."""
    env_atoms = _atoms_within_radius(mol, group, radius)
    return sum(1 for i in env_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def site_environment(mol: Chem.Mol, group: list[int], radius: int) -> str:
    """Canonical SMILES of `group`'s local environment out to `radius`
    bonds. Cut bonds are simply dropped - see module docstring."""
    env_atoms = _atoms_within_radius(mol, group, radius)

    rw = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for i in env_atoms:
        atom = mol.GetAtomWithIdx(i)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        old_to_new[i] = rw.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in env_atoms and b in env_atoms:
            rw.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())

    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Chem.MolSanitizeException:
        try:
            Chem.SanitizeMol(frag, sanitizeOps=_SANITIZE_SANS_KEKULIZATION)
        except Chem.MolSanitizeException:
            Chem.SanitizeMol(frag, sanitizeOps=_SANITIZE_MINIMAL)
    return Chem.MolToSmiles(frag)


__all__ = ["site_environment", "site_bulk"]
