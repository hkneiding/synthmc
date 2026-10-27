"""Core entry point: given a metal and its ligands - already known, not
detected - precedent lookup against a LigandCorpus (see its module
docstring for the 4 per-ligand levels) and, optionally, an
EnvironmentCorpus (see its module docstring for the whole-complex
full-coordination-environment precedent).

This is the primary interface: the intended use case is evaluating
in-silico-built TMCs, where the ligands and their binding atoms are already
known by construction - you chose them to build the candidate.

Each of the 3 precedent terms (ligand, coordination environment, steric) is
its own 0-1-ish signal - see matching.py for how ligand precedent's 4 levels
each become a score, and LigandPrecedent.precedent_score for how those 4
combine per ligand. How multiple ligands within one complex fold into a
single whole-complex ligand-precedent term, and how the 3 top-level terms
themselves combine into one final score, are still open (see project
conversation)."""

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
    """Parse `smiles_in` with normal full sanitization; on failure, retry
    with the same cascading fallback common/environment.py already uses
    (full -> skip kekulization -> also skip valence/property checking - see
    common/sanitize.py for both tiers). Needed for the same reason there:
    structures corpus/mol2.py already tolerated on the way into the corpus
    (Cp-type fragments, unusual NHC/phosphazene-type valence cases - all
    from BOS-Lig's zero-formal-charge mol2 convention) have to be tolerated
    again here, since a plain Chem.MolFromSmiles with default sanitization
    has none of that leniency. Only a genuine syntax error, or a sanitize
    failure the last (most permissive) tier still can't get past, raises."""
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
    # radius -> precedent count for this whole complex's joint donor-site environment - see
    # environment_corpus.py. Empty dict if no EnvironmentCorpus was passed.
    full_environment: dict[int, int | None] = field(default_factory=dict)
    full_environment_and_metal: dict[int, int | None] = field(default_factory=dict)
    # radius -> how common that joint environment is, relative to every joint environment on
    # record at that radius/metal-scope (environment_percentile - count_percentile applied to
    # the match above, 0.0 if no match at all, since a non-match's count is 0 by definition).
    # Same emptiness rule as full_environment above - a non-match here is a real 0.0, not None;
    # only "never computed at all" (no EnvironmentCorpus passed) is None.
    full_environment_percentile: dict[int, float | None] = field(default_factory=dict)
    full_environment_and_metal_percentile: dict[int, float | None] = field(default_factory=dict)
    # None if no StericCorpus was passed, the metal has no tabulated covalent radius, or no
    # corpus complex shares this coordination number - see steric_corpus.py.
    steric: StericMatch | None = None


def _formula(symbols: list[str]) -> str:
    counts = Counter(symbols)
    return "".join(f"{el}{n}" for el, n in sorted(counts.items()))


def evaluate_complex(
    metal: str,
    ligands: list[tuple[str, list[int]]],
    corpus: LigandCorpus,
    identifier: str = "",
    environment_corpus: EnvironmentCorpus | None = None,
    steric_corpus: StericCorpus | None = None,
) -> StructureResult:
    """Evaluate a TMC whose chemistry you already know: `metal` (element
    symbol) and `ligands`, a list of (explicit-H SMILES, binding_atoms)
    pairs, where `binding_atoms` is a flat list of that SMILES string's own
    atom positions that touch the metal. Hapticity grouping (which of those
    positions form one multi-atom site vs. separate ones) is derived
    automatically from the SMILES's own bond graph - the caller doesn't
    need to work out denticity themselves, just which atoms bind.

    Pass `environment_corpus` to also get the whole-complex full
    coordination-environment precedent (see environment_corpus.py), and/or
    `steric_corpus` for the steric-plausibility term (see
    corpus/steric_corpus.py) - both left optional since they're separate
    corpora from `corpus` and a caller may only want the per-ligand
    levels.

    Raises if any ligand's SMILES can't be parsed - this function is for
    chemistry you already know, so a parse failure means the input is
    wrong, not that this ligand happened to be unusual; it shouldn't be
    silently swallowed and mistaken for a real (if weak) precedent score."""
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

        if environment_corpus is not None:
            for group in groups:
                for r in RADII:
                    envs[r].append(site_environment(mol, group, r))

        if steric_corpus is not None:
            total_bulk += ligand_bulk(mol, groups, STERIC_RADIUS)
            cn += len(groups)

        by_ligand = find_closest_precedent(smiles, fp, corpus.by_ligand, corpus.count_distribution.get("by_ligand"))
        by_ligand_and_metal = find_closest_precedent(
            smiles, fp, corpus.by_ligand_and_metal.get(metal), corpus.count_distribution.get("by_ligand_and_metal")
        )
        by_ligand_and_mode = find_closest_precedent(
            smiles, fp, corpus.by_ligand_and_mode.get(signature), corpus.count_distribution.get("by_ligand_and_mode")
        )
        by_ligand_mode_and_metal = find_closest_precedent(
            smiles,
            fp,
            corpus.by_ligand_mode_and_metal.get((metal, signature)),
            corpus.count_distribution.get("by_ligand_mode_and_metal"),
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
    if environment_corpus is not None:
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
    if steric_corpus is not None:
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
