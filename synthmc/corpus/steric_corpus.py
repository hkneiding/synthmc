from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from synthmc.common.elements import covalent_radius
from synthmc.common.environment import ligand_bulk
from synthmc.corpus._persistence import PicklePersistable
from synthmc.corpus._spec_parsing import parse_spec_ligands
from synthmc.corpus.bostmc import load_complexes
from synthmc.corpus.complex_ligands import reconstruct_ligand_multiset

RDLogger.DisableLog("rdApp.*")

STERIC_RADIUS = 2  # bonds from a donor site; "out to the 3rd coordination sphere from the metal"


@dataclass
class StericCorpus(PicklePersistable):
    # coordination number -> sorted list of normalized total-bulk values
    # observed across real complexes with that CN.
    by_cn: dict[int, list[float]] = field(default_factory=dict)
    n_complexes_seen: int = 0
    n_complexes_used: int = 0  # reconstructed cleanly, every candidate parsed, and metal had a tabulated radius


def build_steric_corpus(
    boslig_csv: str | Path,
    bostmc_dir: str | Path,
    progress_every: int = 20000,
    progress_callback=None,
    exclude_refcodes: set[str] | None = None,
) -> StericCorpus:
    """Build the per-CN steric corpus, optionally excluding
    `exclude_refcodes` for held-out validation."""
    corpus = StericCorpus()

    complexes = load_complexes(bostmc_dir)  # base_refcode -> {metal, coordination_formula}
    if exclude_refcodes:
        complexes = {rc: info for rc, info in complexes.items() if rc not in exclude_refcodes}

    lig = pd.read_csv(boslig_csv)

    # Same source data, same reconstruction mechanism as environment_corpus.py's
    # build loop - only what's tallied per complex differs below.
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

        try:
            r_metal = covalent_radius(info["metal"])
        except (KeyError, ValueError):
            if progress_callback is not None and n_done % progress_every == 0:
                progress_callback("build", n_done, len(complexes), corpus)
            continue

        total_bulk = 0
        cn = 0
        for complex_ligand in resolved:
            mol, groups, _ = parsed[candidate_ghs[complex_ligand.candidate_index]]
            for _ in range(complex_ligand.multiplicity):
                total_bulk += ligand_bulk(mol, groups, STERIC_RADIUS)
                cn += len(groups)

        corpus.by_cn.setdefault(cn, []).append(total_bulk / r_metal)
        corpus.n_complexes_used += 1

        if progress_callback is not None and n_done % progress_every == 0:
            progress_callback("build", n_done, len(complexes), corpus)

    for values in corpus.by_cn.values():
        values.sort()

    return corpus


__all__ = ["StericCorpus", "build_steric_corpus", "STERIC_RADIUS"]
