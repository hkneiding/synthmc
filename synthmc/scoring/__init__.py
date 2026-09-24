"""The core evaluation API: given a metal and its ligands (already known -
see evaluate.py's module docstring for why), look up precedent against a
LigandCorpus. evaluate_complex is the primary interface this whole project
exposes - xyz/ is a separate, optional detection layer that feeds into it,
not a dependency of it."""

from synthmc.scoring.evaluate import LigandPrecedent, StericMatch, StructureResult, evaluate_complex
from synthmc.scoring.matching import (
    PrecedentMatch,
    count_percentile,
    find_closest_precedent,
    find_environment_count,
    find_steric_plausibility,
)
from synthmc.scoring.synthesizability import SynthesizabilityResult, score_synthesizability

__all__ = [
    "LigandPrecedent",
    "StructureResult",
    "StericMatch",
    "evaluate_complex",
    "PrecedentMatch",
    "find_closest_precedent",
    "count_percentile",
    "find_environment_count",
    "find_steric_plausibility",
    "SynthesizabilityResult",
    "score_synthesizability",
]
