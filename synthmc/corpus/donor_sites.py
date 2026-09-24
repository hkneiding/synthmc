"""Metal-connecting atom groups for a BOS-Lig ligand, mapped into canonical
SMILES atom order.

Unlike a fresh query structure (only 3D coordinates, no known bonds -
xyz/ligand_chem.py perceives bonds via OpenBabel), BOS-Lig's spec_mol2 blocks
already have real bond orders *and* an explicit dummy metal atom (SYBYL
type "X", exactly one per block - confirmed across all 72,395 rows) whose
bonds point at the real donor atom(s) for that specific binding mode. So
getting "which atoms touch the metal, grouped by hapticity" doesn't need
any geometry/coordinate work at all here: X's neighbors *are* the donor
atoms, and two donors belong to the same hapto group iff they're bonded to
each other (directly or transitively) through the ligand's own (non-X)
bonds - a cyclopentadienyl ring's 5 carbons are all mutually ring-bonded
and collapse into one group; four separate pyridyl/amine donors on a
tetradentate ligand aren't bonded to each other and stay four singletons.
Confirmed both patterns concretely in BOS-Lig data before writing this.

RDKit can't parse mol2's dummy "X" atom type directly (its mol2 reader
looks "X" up in the periodic table and fails), so X and every bond touching
it are stripped from the mol2 text before parsing, renumbering the
remaining atoms contiguously. Since mol2.py's parse_mol2 keeps every atom
(Hs included, no RemoveHs step - see its docstring), that renumbered
position is exactly the parsed Mol's atom index, with no further remapping
needed.

Atom positions in the *SMILES string* (not just the RDKit Mol) come from
RDKit's own `_smilesAtomOutputOrder` property, set as a side effect of
MolToSmiles - an exact integer correspondence, not the coordinate-matching
approach needed when only OpenBabel's plain-text SMILES output is
available (see xyz/smiles.py's get_smiles_and_smiles_metal_connecting_index_group,
written for the query side where no such RDKit-native mapping exists).
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

from synthmc.common.graph_utils import group_by_mutual_bonding
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")


def _strip_metal_dummy(spec_mol2_text: str) -> tuple[str, list[list[int]]]:
    """Remove the dummy metal ('X') atom and its bonds from a spec_mol2
    block, renumbering the remaining atoms contiguously. Returns the
    trimmed mol2 text plus the donor-atom hapto groups, expressed as
    0-indexed positions into that trimmed block's atom order (which equals
    the parsed Mol's atom order - see module docstring)."""
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
    """Canonical SMILES (explicit-H, matching ligand_corpus.py's convention)
    for one spec_mol2 block, plus its metal-connecting atom groups mapped
    into that SMILES string's atom order - one sublist per hapto site, e.g.
    [[3], [7], [12]] for three separate monodentate donors or
    [[3,4,5,6,7]] for one eta5 ring."""
    trimmed_mol2, groups_0idx = _strip_metal_dummy(spec_mol2_text)
    mol = parse_mol2(trimmed_mol2)

    smiles = Chem.MolToSmiles(mol)
    output_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
    orig_to_smiles_pos = {orig_idx: pos for pos, orig_idx in enumerate(output_order)}

    smiles_groups = [[orig_to_smiles_pos[i] for i in group] for group in groups_0idx]
    return smiles, smiles_groups


__all__ = ["spec_mol2_to_smiles_and_donor_groups"]
