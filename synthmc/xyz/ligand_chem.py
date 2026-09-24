"""Turn a metal-free ligand fragment (symbols + 3D coordinates) into a
canonical SMILES + Morgan fingerprint - the two things the precedent score
needs.

Bond-order perception is OpenBabel's (via its xyz -> mdl bond-order
assignment, the user's own snippet), not RDKit's rdDetermineBonds. The
earlier RDKit-based version needed a total formal charge to resolve
ambiguous bond orders, which isn't well-defined for an isolated ligand
fragment cut from a coordinated structure - chasing that (charge sweeps,
then a "radical-first" convention match against BOS-Lig's own charge-naive
mol2 blocks) helped some cases and hurt others with no clean win either way
(see conversation/project memory). OpenBabel's perception needs no charge
argument at all, sidestepping the problem rather than resolving it.

OpenBabel's SMILES writer is re-parsed through RDKit rather than used
directly, for one concrete reason: it must land in the same canonical form
the corpus uses (see corpus/ligand_corpus.py's RDKit-based build), or an
identical ligand on both sides would produce two different strings and
exact-match lookups would never fire.

Both sides keep explicit hydrogens rather than stripping them
(Chem.RemoveHs) - BOS-Lig's mol2 blocks list every H as an explicit atom
already, so this is the natural convention for the corpus, and keeping it
on the query side too (via OpenBabel's "h" output option, forcing every H
into the SMILES) avoids two real problems a RemoveHs-normalized version
had: RemoveHs's default re-sanitize re-triggers the same Kekulization
failure the Cp-ring salvage path in corpus/mol2.py exists to route
around, and - more fundamentally - mapping a donor atom's position from an
explicit-H SMILES (needed for coordinate-matching against raw xyz
coordinates, see corpus/donor_sites.py-style extraction) into a *separately*
generated implicit-H canonical SMILES isn't guaranteed to be the same atom
position at all, since OpenBabel's traversal order isn't guaranteed to
agree between two independently-run conversions. Staying explicit-H
throughout removes that conversion step entirely rather than needing to
track an index mapping through it.
"""

from __future__ import annotations

from dataclasses import dataclass

from openbabel import openbabel
from rdkit import Chem, RDLogger

from synthmc.common.fingerprint import morgan_fingerprint
from synthmc.xyz.fragments import LigandFragment
from synthmc.xyz.smiles import get_smiles_and_smiles_metal_connecting_index_group

RDLogger.DisableLog("rdApp.*")


@dataclass
class LigandMol:
    smiles: str
    fingerprint: object  # rdkit.DataStructs.ExplicitBitVect


def _fragment_xyz_block(symbols: list[str], coords) -> str:
    lines = [str(len(symbols)), ""]
    lines.extend(f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, (x, y, z) in zip(symbols, coords))
    return "\n".join(lines) + "\n"


def _smiles_via_openbabel(xyz_block: str) -> str:
    """The user's own snippet for getting a SMILES from an xyz block via
    OpenBabel's bond-order perception, adapted to (a) take an already
    well-formed xyz string rather than stripping a pre-existing comment
    line out of one, and (b) force explicit hydrogens in the output (the
    "h" option) to match the corpus's convention - see module docstring."""
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mdl")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz_block)
    obConversion.SetInAndOutFormats("mdl", "smi")
    obConversion.AddOption("h", obConversion.OUTOPTIONS)
    return obConversion.WriteString(mol).strip()


_KEEP_EXPLICIT_HS = Chem.SmilesParserParams()
_KEEP_EXPLICIT_HS.removeHs = False  # MolFromSmiles defaults to stripping explicit [H] atoms right back out


def _canonicalize(smiles: str) -> Chem.Mol:
    """Re-parse an OpenBabel SMILES through RDKit, so the result lands in
    the same canonical form corpus/ligand_corpus.py produces (see module
    docstring) - both toolkits can write "a" valid SMILES for the same
    molecule, but only agreeing on which one is canonical makes exact-match
    lookups against the corpus possible at all."""
    ob_smiles = smiles.split()[0] if smiles else smiles  # OB writes "SMILES\tTitle"
    mol = Chem.MolFromSmiles(ob_smiles, _KEEP_EXPLICIT_HS)
    if mol is None:
        raise ValueError(f"RDKit could not parse OpenBabel SMILES {ob_smiles!r}")
    return mol


def ligand_to_mol(fragment: LigandFragment) -> LigandMol:
    """Perceive bond orders for one ligand fragment (OpenBabel) and return
    its canonical SMILES + Morgan fingerprint (both RDKit, matching the
    corpus's convention). Raises ValueError if either step fails."""
    if fragment.symbols == []:
        raise ValueError("empty fragment")

    if len(fragment.symbols) == 1:
        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(fragment.symbols[0]))
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return LigandMol(smiles=Chem.MolToSmiles(mol), fingerprint=morgan_fingerprint(mol))

    xyz_block = _fragment_xyz_block(fragment.symbols, fragment.coords)
    ob_smiles = _smiles_via_openbabel(xyz_block)
    if not ob_smiles:
        raise ValueError(f"OpenBabel produced no SMILES for fragment ({''.join(fragment.symbols)})")
    mol = _canonicalize(ob_smiles)

    return LigandMol(smiles=Chem.MolToSmiles(mol), fingerprint=morgan_fingerprint(mol))


def ligand_to_mol_with_binding_groups(fragment: LigandFragment) -> tuple[LigandMol, list[list[int]]]:
    """Like ligand_to_mol, but also returns fragment.donor_groups (given in
    the fragment's own xyz atom order) remapped into the returned canonical
    SMILES string's own atom-position order - the format xyz/adapter.py
    needs to call pipeline.evaluate_complex's core API, which takes SMILES-
    indexed binding atoms rather than xyz indices (see its module
    docstring for why detection and scoring are kept separate).

    Uses smiles.py's coordinate-matching (OpenBabel's "x"+"h" output
    options echo each atom's position alongside the SMILES, matched back
    against the original xyz coordinates) to go from xyz index -> position
    in OpenBabel's own SMILES text, then RDKit's `_smilesAtomOutputOrder`
    (an exact integer mapping, no further coordinate matching needed) to go
    from there to a position in the final *canonical* SMILES - the
    coordinate-matching step only has to run once, at the one point it's
    unavoidable (bridging "atom identified purely by position" to "atom
    identified by a parsed Mol object"), mirroring corpus/donor_sites.py's
    corpus-side approach after that bridge is crossed."""
    if len(fragment.symbols) == 1:
        # a single-atom SMILES has exactly one atom, at position 0 - same
        # as the fragment's own (trivial) donor_groups, so no remapping needed
        return ligand_to_mol(fragment), fragment.donor_groups

    xyz_block = _fragment_xyz_block(fragment.symbols, fragment.coords)
    ob_smiles, ob_groups = get_smiles_and_smiles_metal_connecting_index_group(xyz_block, fragment.donor_groups)
    mol = _canonicalize(ob_smiles)

    canonical_smiles = Chem.MolToSmiles(mol)
    output_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
    orig_to_canonical_pos = {orig_idx: pos for pos, orig_idx in enumerate(output_order)}
    canonical_groups = [[orig_to_canonical_pos[i] for i in group] for group in ob_groups]

    lig_mol = LigandMol(smiles=canonical_smiles, fingerprint=morgan_fingerprint(mol))
    return lig_mol, canonical_groups


__all__ = ["LigandMol", "ligand_to_mol", "ligand_to_mol_with_binding_groups", "morgan_fingerprint"]
