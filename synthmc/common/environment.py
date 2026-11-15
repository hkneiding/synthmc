from __future__ import annotations

from rdkit import Chem, RDLogger

from synthmc.common.sanitize import SANITIZE_MINIMAL, SANITIZE_SANS_KEKULIZATION

RDLogger.DisableLog("rdApp.*")


def _atoms_within_radius(mol: Chem.Mol, group: list[int], radius: int) -> set[int]:
    """Every atom within `radius` bonds of any atom in `group` (group
    members included at distance 0)."""
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


def ligand_bulk(mol: Chem.Mol, groups: list[list[int]], radius: int) -> int:
    """Heavy-atom count within `radius` bonds of any of a ligand's donor
    sites, de-duplicating atoms shared between sites."""
    env_atoms: set[int] = set()
    for group in groups:
        env_atoms |= _atoms_within_radius(mol, group, radius)
    return sum(1 for i in env_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def site_environment(mol: Chem.Mol, group: list[int], radius: int) -> str:
    """Canonical SMILES of `group`'s local environment out to `radius`
    bonds."""
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
            Chem.SanitizeMol(frag, sanitizeOps=SANITIZE_SANS_KEKULIZATION)
        except Chem.MolSanitizeException:
            Chem.SanitizeMol(frag, sanitizeOps=SANITIZE_MINIMAL)
    return Chem.MolToSmiles(frag)


__all__ = ["site_environment", "ligand_bulk"]
