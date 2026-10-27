"""The full coordination-environment precedent: how often has a metal been
observed with *this exact combination* of donor-site environments across
all of its ligands simultaneously - not per-ligand, not per-site
independently, but the whole complex's joint donor-site multiset as one
key. See complex_ligands.py for how a complex's complete ligand multiset
(respecting repeated ligands) gets reconstructed from BOS-TMC +  BOS-Lig,
and common/environment.py for what a "donor-site environment" is.

Radii 0-4 (see RADII), each in a metal-agnostic and metal-specific form
(10 levels total, mirroring the existing per-ligand corpus's 2x2 structure
at each radius):
  - by_environment[r]              - radius-r joint environment, any metal
  - by_environment_and_metal[r]    - radius-r joint environment, this metal

Exact-match only, no Tanimoto/similarity fallback - deliberately: the "unit"
being compared here is a whole complex's *set* of small environment
fragments, not one molecule, so there's no single well-defined fingerprint
to compute similarity against without inventing a whole new (and untested)
notion of what "two complexes' environments are similar" should mean. That
question is left open rather than answered with an ad hoc guess.

Once a match is found, how *common* it is still matters - count_distribution/
count_distribution_and_metal pool every occurrence count at a given radius
into one sorted reference array per radius, the same role
corpus/ligand_corpus.py's count_distribution plays, so
scoring/matching.py's environment_percentile can rank a query's matched
count the same way count_percentile ranks a ligand's."""

from __future__ import annotations

import ast
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from synthmc.common.environment import site_environment
from synthmc.corpus.bostmc import load_complexes
from synthmc.corpus.complex_ligands import reconstruct_ligand_multiset
from synthmc.corpus.donor_sites import _strip_metal_dummy
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")

RADII = (0, 1, 2, 3, 4)


@dataclass
class EnvironmentCorpus:
    by_environment: dict[int, dict[tuple[str, ...], int]] = field(default_factory=lambda: {r: {} for r in RADII})
    by_environment_and_metal: dict[int, dict[tuple[str, tuple[str, ...]], int]] = field(
        default_factory=lambda: {r: {} for r in RADII}
    )
    n_complexes_seen: int = 0
    n_complexes_used: int = 0  # reconstructed cleanly and had every candidate parse successfully
    # radius -> every occurrence count in by_environment[r]/by_environment_and_metal[r], sorted -
    # the reference population scoring/matching.py's environment_percentile ranks a query match's
    # count against, same role as LigandCorpus.count_distribution.
    count_distribution: dict[int, list[int]] = field(default_factory=lambda: {r: [] for r in RADII})
    count_distribution_and_metal: dict[int, list[int]] = field(default_factory=lambda: {r: [] for r in RADII})

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "EnvironmentCorpus":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


def build_environment_corpus(
    boslig_csv: str | Path,
    bostmc_dir: str | Path,
    progress_every: int = 20000,
    progress_callback=None,
    exclude_refcodes: set[str] | None = None,
) -> EnvironmentCorpus:
    """`exclude_refcodes` supports building a proper held-out corpus for
    validation (see scripts/holdout_environment_test.py) - complexes in
    this set are skipped entirely, as if they didn't exist in BOS-TMC, so
    a held-out query against them can't trivially match itself."""
    corpus = EnvironmentCorpus()

    complexes = load_complexes(bostmc_dir)  # base_refcode -> {metal, coordination_formula}
    if exclude_refcodes:
        complexes = {rc: info for rc, info in complexes.items() if rc not in exclude_refcodes}

    lig = pd.read_csv(boslig_csv)

    # Parse every spec_gh row once (Mol + donor groups + molecular formula),
    # and build refcode -> list of spec_gh indices touching it.
    parsed: dict[str, tuple] = {}  # spec_gh -> (mol, groups, formula_counter)
    refcode_to_spec_ghs: dict[str, list[str]] = {}
    n_seen = 0
    for _, row in lig.iterrows():
        n_seen += 1
        spec_gh = row["spec_gh"]
        try:
            trimmed_mol2, groups = _strip_metal_dummy(row["spec_mol2"])
            mol = parse_mol2(trimmed_mol2)
            formula = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
            parsed[spec_gh] = (mol, groups, formula)
        except Exception:
            pass  # this ligand just won't be a usable candidate for any refcode

        for refcode in ast.literal_eval(row["spec_refcode"]):
            refcode_to_spec_ghs.setdefault(refcode, []).append(spec_gh)

        if progress_callback is not None and n_seen % progress_every == 0:
            progress_callback("parse", n_seen, len(lig), corpus)

    n_done = 0
    for base_refcode, info in complexes.items():
        n_done += 1
        corpus.n_complexes_seen += 1

        candidate_ghs = [gh for gh in refcode_to_spec_ghs.get(base_refcode, []) if gh in parsed]
        candidate_formulas = [parsed[gh][2] for gh in candidate_ghs]

        resolved = reconstruct_ligand_multiset(info["coordination_formula"], info["metal"], candidate_formulas)
        if resolved is None:
            if progress_callback is not None and n_done % progress_every == 0:
                progress_callback("build", n_done, len(complexes), corpus)
            continue

        envs: dict[int, list[str]] = {r: [] for r in RADII}
        for complex_ligand in resolved:
            mol, groups, _ = parsed[candidate_ghs[complex_ligand.candidate_index]]
            for _ in range(complex_ligand.multiplicity):
                for group in groups:
                    for r in RADII:
                        envs[r].append(site_environment(mol, group, r))

        metal = info["metal"]
        for r in RADII:
            key = tuple(sorted(envs[r]))
            corpus.by_environment[r][key] = corpus.by_environment[r].get(key, 0) + 1
            metal_key = (metal, key)
            corpus.by_environment_and_metal[r][metal_key] = corpus.by_environment_and_metal[r].get(metal_key, 0) + 1

        corpus.n_complexes_used += 1

        if progress_callback is not None and n_done % progress_every == 0:
            progress_callback("build", n_done, len(complexes), corpus)

    for r in RADII:
        corpus.count_distribution[r] = sorted(corpus.by_environment[r].values())
        corpus.count_distribution_and_metal[r] = sorted(corpus.by_environment_and_metal[r].values())

    return corpus


__all__ = ["EnvironmentCorpus", "build_environment_corpus", "RADII"]
