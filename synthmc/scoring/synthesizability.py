"""Combine the three precedent terms - ligand, coordination environment,
steric plausibility - from a StructureResult into one SynthesizabilityResult
with a single overall score.

Each term is reduced to one 0-1(ish) number the same way throughout this
project: by simple averaging. Ligand precedent already averages its own 4
levels per ligand instance (LigandPrecedent.precedent_score - see
scoring/evaluate.py); this module averages those per-ligand scores across
every ligand in the complex to get one whole-complex ligand term, and
averages the coordination-environment term's hard match/no-match outcomes
(5 radii x {any-metal, metal-specific} = 10 binary checks) into one
whole-complex environment term, the same way LigandCorpus's 4 levels get
averaged. Steric plausibility is already a single per-complex number (see
corpus/steric_corpus.py), so it's used as-is.

`None` is used strictly for "no data was available to compute this", never
for "checked and found nothing" - the same distinction the rest of this
project maintains (see scoring/matching.py's module docstring), with one
deliberate exception: LigandPrecedent.precedent_score itself already scores
"no corpus data for this specific key" as a real 0 rather than excluding it
(a documented-but-undiscovered combination is treated the same as a
confirmed-absent one - see evaluate.py). What still yields None here is a
genuinely unevaluated ligand: simply not passing an EnvironmentCorpus/
StericCorpus to evaluate_complex, or (xyz-derived results only) a fragment
that failed *detection* before evaluate_complex ever saw it - see
xyz/adapter.py. evaluate_complex itself never produces a partial
LigandPrecedent anymore; it raises instead (see its docstring)."""

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
    environment_score: float | None  # mean of hard 1/0 match outcomes across 5 radii x 2 metal-scope checks
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

        n_failed = sum(1 for lig in s.ligands if lig.error is not None)
        print(f"[1] Ligand precedent: {_fmt(self.ligand_score)}")
        print(f"      {len(s.ligands)} ligand(s), {n_failed} failed detection (xyz-derived results only)")
        for lig in s.ligands:
            label = lig.formula if lig.smiles is not None else f"? ({lig.error})"
            print(f"      {label:20s} precedent_score={_fmt(lig.precedent_score)}")
        print()

        print(f"[2] Coordination environment precedent: {_fmt(self.environment_score)}")
        for r in sorted(s.full_environment):
            any_metal = "match" if s.full_environment[r] is not None else "no match"
            metal = "match" if s.full_environment_and_metal[r] is not None else "no match"
            print(f"      radius {r}: any-metal={any_metal:8s} metal-specific={metal}")
        print()

        print(f"[3] Steric plausibility: {_fmt(self.steric_score)}")
        print()

        print(f"Overall score: {_fmt(self.overall_score)}")


def score_synthesizability(structure: StructureResult) -> SynthesizabilityResult:
    ligand_scores = [lig.precedent_score for lig in structure.ligands if lig.precedent_score is not None]
    ligand_score = statistics.mean(ligand_scores) if ligand_scores else None

    # A detection failure (xyz-derived results only - see xyz/adapter.py) fills
    # full_environment/full_environment_and_metal with None for every radius rather than
    # leaving the dict empty, which would otherwise look like "checked, no precedent" rather
    # than "couldn't check" - excluded here the same way xyz/adapter.py documents it.
    any_ligand_failed = any(lig.error is not None for lig in structure.ligands)
    if structure.full_environment and not any_ligand_failed:
        env_values = list(structure.full_environment.values()) + list(structure.full_environment_and_metal.values())
        environment_score = sum(1.0 if v is not None else 0.0 for v in env_values) / len(env_values)
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
