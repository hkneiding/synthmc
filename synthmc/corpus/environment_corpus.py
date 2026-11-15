from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from synthmc.common.environment import site_environment
from synthmc.corpus._persistence import PicklePersistable
from synthmc.corpus._spec_parsing import parse_spec_ligands
from synthmc.corpus.bostmc import load_complexes
from synthmc.corpus.complex_ligands import reconstruct_ligand_multiset

RDLogger.DisableLog("rdApp.*")

RADII = (0, 1, 2, 3, 4)


@dataclass
class EnvironmentCorpus(PicklePersistable):
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


def build_environment_corpus(
    boslig_csv: str | Path,
    bostmc_dir: str | Path,
    progress_every: int = 20000,
    progress_callback=None,
    exclude_refcodes: set[str] | None = None,
) -> EnvironmentCorpus:
    """Build the coordination-environment corpus, optionally excluding
    `exclude_refcodes` for held-out validation."""
    corpus = EnvironmentCorpus()

    complexes = load_complexes(bostmc_dir)  # base_refcode -> {metal, coordination_formula}
    if exclude_refcodes:
        complexes = {rc: info for rc, info in complexes.items() if rc not in exclude_refcodes}

    lig = pd.read_csv(boslig_csv)

    parsed, refcode_to_spec_ghs = parse_spec_ligands(lig, corpus, progress_callback, progress_every)

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
