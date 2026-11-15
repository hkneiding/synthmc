from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

from synthmc.common.donor_signature import donor_signature
from synthmc.common.elements import covalent_radius
from synthmc.common.environment import ligand_bulk, site_environment
from synthmc.common.fingerprint import morgan_fingerprint
from synthmc.common.graph_utils import group_by_mutual_bonding
from synthmc.common.sanitize import SANITIZE_MINIMAL, SANITIZE_SANS_KEKULIZATION
from synthmc.corpus.environment_corpus import RADII, EnvironmentCorpus
from synthmc.corpus.ligand_corpus import LigandCorpus
from synthmc.corpus.steric_corpus import STERIC_RADIUS, StericCorpus
from synthmc.scoring.matching import (
    PrecedentMatch,
    environment_percentile,
    find_closest_precedent,
    find_environment_count,
    find_steric_plausibility,
)

RDLogger.DisableLog("rdApp.*")

_KEEP_EXPLICIT_HS = Chem.SmilesParserParams()
_KEEP_EXPLICIT_HS.removeHs = False  # MolFromSmiles defaults to stripping explicit [H] atoms right back out

_KEEP_EXPLICIT_HS_UNSANITIZED = Chem.SmilesParserParams()
_KEEP_EXPLICIT_HS_UNSANITIZED.removeHs = False
_KEEP_EXPLICIT_HS_UNSANITIZED.sanitize = False


def _parse_ligand_smiles(smiles_in: str) -> Chem.Mol:
    """Parse `smiles_in`, falling back to progressively more permissive
    sanitization on failure."""
    mol = Chem.MolFromSmiles(smiles_in, _KEEP_EXPLICIT_HS)
    if mol is not None:
        return mol

    mol = Chem.MolFromSmiles(smiles_in, _KEEP_EXPLICIT_HS_UNSANITIZED)
    if mol is None:
        raise ValueError(f"could not parse SMILES {smiles_in!r}")
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_SANS_KEKULIZATION)
        except Chem.MolSanitizeException:
            Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_MINIMAL)
    return mol


@dataclass
class LigandPrecedent:
    formula: str
    smiles: str | None  # canonicalized form of the input SMILES; None if that input couldn't be parsed
    donor_signature: tuple[str, ...] | None
    by_ligand: PrecedentMatch | None  # level 1: this ligand, any metal, any mode
    by_ligand_and_metal: PrecedentMatch | None  # level 2: this ligand with this structure's metal, any mode
    by_ligand_and_mode: PrecedentMatch | None  # level 3: this ligand in this coordination mode, any metal
    by_ligand_mode_and_metal: PrecedentMatch | None  # level 4: this ligand in this mode, with this metal
    # mean of the above 4 levels' .score, treating a None level (no corpus data at all for
    # that key - e.g. a metal BOS-TMC has zero entries for) as a 0 contribution rather than
    # excluding it - a documented-but-undiscovered combination is scored the same as a
    # confirmed-absent one, deliberately, rather than left out of the average. Always a real
    # float when evaluate_complex produces this (every ligand it returns parsed successfully -
    # see its docstring).
    precedent_score: float | None = None


@dataclass
class StericMatch:
    normalized_bulk: float  # total heavy-atom bulk across all donor sites, divided by the metal's covalent radius
    cn: int  # coordination number - total number of donor sites (hapto groups), counting repeated ligands
    plausibility: float  # find_steric_plausibility's one-sided survival-function value, in [0, 1]


@dataclass
class StructureResult:
    metal_symbol: str
    ligands: list[LigandPrecedent] = field(default_factory=list)
    identifier: str = ""  # optional caller-supplied label (e.g. a CSD refcode), not used internally
    # radius -> precedent count for this whole complex's joint donor-site environment.
    full_environment: dict[int, int | None] = field(default_factory=dict)
    full_environment_and_metal: dict[int, int | None] = field(default_factory=dict)
    # radius -> how common that joint environment is, relative to every joint environment on
    # record at that radius/metal-scope (0.0 if no match at all).
    full_environment_percentile: dict[int, float | None] = field(default_factory=dict)
    full_environment_and_metal_percentile: dict[int, float | None] = field(default_factory=dict)
    # None if the metal has no tabulated covalent radius, or no corpus complex shares this CN.
    steric: StericMatch | None = None


def _formula(symbols: list[str]) -> str:
    """Molecular formula string from a flat atom-symbol list, e.g.
    ['C', 'H', 'H', 'H', 'N'] -> 'CH3N'."""
    counts = Counter(symbols)
    return "".join(f"{el}{n}" for el, n in sorted(counts.items()))


def evaluate_complex(
    metal: str,
    ligands: list[tuple[str, list[int]]],
    ligand_corpus: LigandCorpus,
    environment_corpus: EnvironmentCorpus,
    steric_corpus: StericCorpus,
    identifier: str = "",
) -> StructureResult:
    """Evaluate a TMC's `metal` and `ligands` (SMILES, binding_atoms)
    pairs against the three corpora. Raises on an unparseable SMILES."""
    results = []
    envs: dict[int, list[str]] = {r: [] for r in RADII}
    total_bulk = 0
    cn = 0

    for smiles_in, binding_atoms in ligands:
        mol = _parse_ligand_smiles(smiles_in)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        smiles = Chem.MolToSmiles(mol)  # canonicalize - the caller's own SMILES needn't already be

        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        groups = group_by_mutual_bonding(binding_atoms, bonds)
        signature = donor_signature(symbols, groups)
        fp = morgan_fingerprint(mol)

        for group in groups:
            for r in RADII:
                envs[r].append(site_environment(mol, group, r))

        total_bulk += ligand_bulk(mol, groups, STERIC_RADIUS)
        cn += len(groups)

        by_ligand = find_closest_precedent(
            smiles, fp, ligand_corpus.by_ligand, ligand_corpus.count_distribution.get("by_ligand")
        )
        by_ligand_and_metal = find_closest_precedent(
            smiles,
            fp,
            ligand_corpus.by_ligand_and_metal.get(metal),
            ligand_corpus.count_distribution.get("by_ligand_and_metal"),
        )
        by_ligand_and_mode = find_closest_precedent(
            smiles,
            fp,
            ligand_corpus.by_ligand_and_mode.get(signature),
            ligand_corpus.count_distribution.get("by_ligand_and_mode"),
        )
        by_ligand_mode_and_metal = find_closest_precedent(
            smiles,
            fp,
            ligand_corpus.by_ligand_mode_and_metal.get((metal, signature)),
            ligand_corpus.count_distribution.get("by_ligand_mode_and_metal"),
        )
        levels = (by_ligand, by_ligand_and_metal, by_ligand_and_mode, by_ligand_mode_and_metal)
        precedent_score = sum(m.score if m is not None else 0.0 for m in levels) / len(levels)

        results.append(
            LigandPrecedent(
                formula=_formula(symbols),
                smiles=smiles,
                donor_signature=signature,
                by_ligand=by_ligand,
                by_ligand_and_metal=by_ligand_and_metal,
                by_ligand_and_mode=by_ligand_and_mode,
                by_ligand_mode_and_metal=by_ligand_mode_and_metal,
                precedent_score=precedent_score,
            )
        )

    full_environment: dict[int, int | None] = {}
    full_environment_and_metal: dict[int, int | None] = {}
    full_environment_percentile: dict[int, float | None] = {}
    full_environment_and_metal_percentile: dict[int, float | None] = {}
    for r in RADII:
        key = tuple(sorted(envs[r]))
        full_environment[r] = find_environment_count(key, environment_corpus.by_environment[r])
        full_environment_and_metal[r] = find_environment_count(
            (metal, key), environment_corpus.by_environment_and_metal[r]
        )
        full_environment_percentile[r] = environment_percentile(
            key, environment_corpus.by_environment[r], environment_corpus.count_distribution[r]
        )
        full_environment_and_metal_percentile[r] = environment_percentile(
            (metal, key), environment_corpus.by_environment_and_metal[r], environment_corpus.count_distribution_and_metal[r]
        )

    steric: StericMatch | None = None
    try:
        r_metal = covalent_radius(metal)
    except (KeyError, ValueError):
        r_metal = None
    if r_metal is not None:
        normalized_bulk = total_bulk / r_metal
        plausibility = find_steric_plausibility(normalized_bulk, cn, steric_corpus)
        if plausibility is not None:
            steric = StericMatch(normalized_bulk=normalized_bulk, cn=cn, plausibility=plausibility)

    return StructureResult(
        metal_symbol=metal,
        ligands=results,
        identifier=identifier,
        full_environment=full_environment,
        full_environment_and_metal=full_environment_and_metal,
        full_environment_percentile=full_environment_percentile,
        full_environment_and_metal_percentile=full_environment_and_metal_percentile,
        steric=steric,
    )


__all__ = ["LigandPrecedent", "StructureResult", "StericMatch", "evaluate_complex"]
