"""Combine the three precedent terms - ligand, coordination environment,
steric plausibility - from a StructureResult into one SynthesizabilityResult
with a single overall score.

Each term is reduced to one 0-1(ish) number the same way throughout this
project: by simple averaging. Ligand precedent already averages its own 4
levels per ligand instance (LigandPrecedent.precedent_score - see
scoring/evaluate.py); this module averages those per-ligand scores across
every ligand in the complex to get one whole-complex ligand term, and
averages the coordination-environment term's 10 environment_percentile
values (5 radii x {any-metal, metal-specific} - see
StructureResult.full_environment_percentile/full_environment_and_metal_percentile)
into one whole-complex environment term, the same way LigandCorpus's 4
levels get averaged - not a hard match/no-match hit rate, a non-match
already scores 0.0 within environment_percentile itself (see
matching.py), so the only thing left to do here is average the 10 values.
Steric plausibility is already a single per-complex number (see
corpus/steric_corpus.py), so it's used as-is.

`None` is used strictly for "no data was available to compute this", never
for "checked and found nothing" - the same distinction the rest of this
project maintains (see scoring/matching.py's module docstring), with one
deliberate exception: LigandPrecedent.precedent_score itself already scores
"no corpus data for this specific key" as a real 0 rather than excluding it
(a documented-but-undiscovered combination is treated the same as a
confirmed-absent one - see evaluate.py). What still yields None here is
simply not passing an EnvironmentCorpus/StericCorpus to evaluate_complex -
evaluate_complex never produces a partial result otherwise; it raises
instead (see its docstring)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from synthmc.scoring.evaluate import StructureResult


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


@dataclass
class SynthesizabilityResult:
    structure: StructureResult  # full underlying detail - per-ligand precedent, raw environment/steric data
    ligand_score: float | None  # mean of LigandPrecedent.precedent_score across every ligand in the complex
    environment_score: float | None  # mean of environment_percentile across 5 radii x 2 metal-scope checks
    steric_score: float | None  # = structure.steric.plausibility, carried through unchanged
    overall_score: float | None  # mean of whichever of the 3 terms above are not None; None if all 3 are

    def summary(self) -> None:
        """Print a breakdown of each of the 3 terms, plus the combined score."""
        s = self.structure
        header = f"Synthesizability summary - metal: {s.metal_symbol}"
        if s.identifier:
            header += f" ({s.identifier})"
        print(header)
        print()

        print(f"[1] Ligand precedent: {_fmt(self.ligand_score)}")
        print(f"      {len(s.ligands)} ligand(s)")
        for lig in s.ligands:
            print(f"      {lig.formula:20s} precedent_score={_fmt(lig.precedent_score)}")
        print()

        print(f"[2] Coordination environment precedent: {_fmt(self.environment_score)}")
        for r in sorted(s.full_environment):
            any_metal = _fmt(s.full_environment_percentile.get(r))
            metal = _fmt(s.full_environment_and_metal_percentile.get(r))
            print(f"      radius {r}: any-metal={any_metal:8s} metal-specific={metal}")
        print()

        print(f"[3] Steric plausibility: {_fmt(self.steric_score)}")
        print()

        print(f"Overall score: {_fmt(self.overall_score)}")


def score_synthesizability(structure: StructureResult) -> SynthesizabilityResult:
    ligand_scores = [lig.precedent_score for lig in structure.ligands if lig.precedent_score is not None]
    ligand_score = statistics.mean(ligand_scores) if ligand_scores else None

    if structure.full_environment_percentile:
        env_values = list(structure.full_environment_percentile.values()) + list(
            structure.full_environment_and_metal_percentile.values()
        )
        environment_score = statistics.mean(env_values)
    else:
        environment_score = None

    steric_score = structure.steric.plausibility if structure.steric is not None else None

    term_scores = [t for t in (ligand_score, environment_score, steric_score) if t is not None]
    overall_score = statistics.mean(term_scores) if term_scores else None

    return SynthesizabilityResult(
        structure=structure,
        ligand_score=ligand_score,
        environment_score=environment_score,
        steric_score=steric_score,
        overall_score=overall_score,
    )


__all__ = ["SynthesizabilityResult", "score_synthesizability"]
