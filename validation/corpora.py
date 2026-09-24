"""Build all three precedent corpora with a held-out refcode set excluded,
for validation.corpora.evaluate to query against."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthmc.corpus.environment_corpus import EnvironmentCorpus, build_environment_corpus
from synthmc.corpus.ligand_corpus import LigandCorpus, build_ligand_corpus
from synthmc.corpus.steric_corpus import StericCorpus, build_steric_corpus
from validation.reconstruct import ReconstructedComplex, ligand_corpus_exclusions


@dataclass
class HoldoutCorpora:
    ligand: LigandCorpus
    environment: EnvironmentCorpus
    steric: StericCorpus


def build_holdout_corpora(
    boslig_csv: str | Path,
    bostmc_dir: str | Path,
    holdout_refcodes: set[str],
    reconstructed_holdout: dict[str, ReconstructedComplex],
    progress_callback=None,
) -> HoldoutCorpora:
    """`reconstructed_holdout` (from validation.reconstruct.reconstruct_complexes,
    same `holdout_refcodes`) supplies the exact per-ligand-structure counts
    build_ligand_corpus needs to exclude precisely - see its docstring."""
    exclude_spec_counts, exclude_gen_counts = ligand_corpus_exclusions(reconstructed_holdout, boslig_csv)

    ligand = build_ligand_corpus(
        boslig_csv,
        bostmc_dir,
        progress_callback=progress_callback,
        exclude_refcodes=holdout_refcodes,
        exclude_gen_counts=exclude_gen_counts,
        exclude_spec_counts=exclude_spec_counts,
    )
    environment = build_environment_corpus(
        boslig_csv, bostmc_dir, progress_callback=progress_callback, exclude_refcodes=holdout_refcodes
    )
    steric = build_steric_corpus(
        boslig_csv, bostmc_dir, progress_callback=progress_callback, exclude_refcodes=holdout_refcodes
    )

    return HoldoutCorpora(ligand=ligand, environment=environment, steric=steric)


__all__ = ["HoldoutCorpora", "build_holdout_corpora"]
