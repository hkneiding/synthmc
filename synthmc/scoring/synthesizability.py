from __future__ import annotations

import statistics
from dataclasses import dataclass

from synthmc.scoring.evaluate import StructureResult


def _fmt(x: float | None) -> str:
    """3-decimal string, or "n/a" for None - for summary()'s printout."""
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
        for r in sorted(s.environment):
            env = s.environment[r]
            any_metal = _fmt(env.percentile)
            metal = _fmt(env.percentile_and_metal)
            print(f"      radius {r}: any-metal={any_metal:8s} metal-specific={metal}")
        print()

        print(f"[3] Steric plausibility: {_fmt(self.steric_score)}")
        print()

        print(f"Overall score: {_fmt(self.overall_score)}")


def score_synthesizability(structure: StructureResult) -> SynthesizabilityResult:
    """Reduce an evaluate_complex result to the 3 term scores plus one
    overall score."""
    ligand_scores = [lig.precedent_score for lig in structure.ligands if lig.precedent_score is not None]
    ligand_score = statistics.mean(ligand_scores) if ligand_scores else None

    if structure.environment:
        env_values = [m.percentile for m in structure.environment.values()] + [
            m.percentile_and_metal for m in structure.environment.values()
        ]
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
