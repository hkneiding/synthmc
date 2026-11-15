from __future__ import annotations

from rdkit import Chem, RDLogger

from synthmc.common.graph_utils import group_by_mutual_bonding
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")


def _strip_metal_dummy(spec_mol2_text: str) -> tuple[str, list[list[int]]]:
    """Strip the dummy metal ('X') atom from a spec_mol2 block. Returns
    the trimmed mol2 text plus the donor-atom hapto groups."""
    lines = spec_mol2_text.split("\n")
    atom_start = lines.index("@<TRIPOS>ATOM") + 1
    bond_start = lines.index("@<TRIPOS>BOND") + 1
    bond_end = next((i for i in range(bond_start, len(lines)) if lines[i].startswith("@<TRIPOS>")), len(lines))
    atom_lines = lines[atom_start : bond_start - 1]
    bond_lines = lines[bond_start:bond_end]

    serials = [int(line.split()[0]) for line in atom_lines]
    types = [line.split()[5] for line in atom_lines]
    try:
        x_serial = serials[types.index("X")]
    except ValueError:
        raise ValueError("no dummy metal ('X') atom found in spec_mol2 block") from None

    bonds = [(int(parts[1]), int(parts[2]), parts[3]) for parts in (line.split() for line in bond_lines)]

    donor_serials: set[int] = set()
    non_x_bonds: list[tuple[int, int]] = []
    for a, b, _btype in bonds:
        if a == x_serial:
            donor_serials.add(b)
            continue
        if b == x_serial:
            donor_serials.add(a)
            continue
        non_x_bonds.append((a, b))
    groups = group_by_mutual_bonding(donor_serials, non_x_bonds)

    remaining_serials = [s for s in serials if s != x_serial]
    old_to_new = {s: i + 1 for i, s in enumerate(remaining_serials)}  # 1-indexed, for the mol2 text itself

    new_atom_lines = []
    for line in atom_lines:
        parts = line.split()
        serial = int(parts[0])
        if serial == x_serial:
            continue
        parts[0] = str(old_to_new[serial])
        new_atom_lines.append(" ".join(parts))

    new_bond_lines = []
    for bond_id, (a, b, btype) in enumerate((b for b in bonds if x_serial not in b[:2]), start=1):
        new_bond_lines.append(f"{bond_id} {old_to_new[a]} {old_to_new[b]} {btype}")

    header = lines[: atom_start - 1]
    header[2] = f"{len(new_atom_lines)}\t{len(new_bond_lines)}\t1"  # natoms / nbonds / nsubst counts line
    trimmed = "\n".join(header + ["@<TRIPOS>ATOM"] + new_atom_lines + ["@<TRIPOS>BOND"] + new_bond_lines + [""])

    groups_0idx = [[old_to_new[s] - 1 for s in group] for group in groups]
    return trimmed, groups_0idx


def spec_mol2_to_smiles_and_donor_groups(spec_mol2_text: str) -> tuple[str, list[list[int]]]:
    """Canonical SMILES for one spec_mol2 block, plus its donor-atom groups
    mapped into that SMILES string's atom order."""
    trimmed_mol2, groups_0idx = _strip_metal_dummy(spec_mol2_text)
    mol = parse_mol2(trimmed_mol2)

    smiles = Chem.MolToSmiles(mol)
    output_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
    orig_to_smiles_pos = {orig_idx: pos for pos, orig_idx in enumerate(output_order)}

    smiles_groups = [[orig_to_smiles_pos[i] for i in group] for group in groups_0idx]
    return smiles, smiles_groups


__all__ = ["spec_mol2_to_smiles_and_donor_groups"]
