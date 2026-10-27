"""synthmc: precedent-based ligand evaluation for transition metal complexes.

Package layout:
  common/  - small shared helpers (element data, graph utilities, donor
             signatures, donor-site environments) used by both corpus/ and
             scoring/.
  corpus/  - builds/represents the precedent corpora from BOS-Lig + BOS-TMC:
             LigandCorpus (4 per-ligand levels) and EnvironmentCorpus (the
             whole-complex full coordination-environment precedent).
  scoring/ - the API: evaluate_complex(metal, ligands, corpus) against a
             corpus, given chemistry you already know.
"""

from synthmc.corpus import (
    EnvironmentCorpus,
    LigandCorpus,
    StericCorpus,
    build_environment_corpus,
    build_ligand_corpus,
    build_steric_corpus,
)
from synthmc.scoring import (
    LigandPrecedent,
    PrecedentMatch,
    StericMatch,
    StructureResult,
    SynthesizabilityResult,
    evaluate_complex,
    find_closest_precedent,
    score_synthesizability,
)

__all__ = [
    "LigandCorpus",
    "build_ligand_corpus",
    "EnvironmentCorpus",
    "build_environment_corpus",
    "StericCorpus",
    "build_steric_corpus",
    "LigandPrecedent",
    "StructureResult",
    "StericMatch",
    "SynthesizabilityResult",
    "score_synthesizability",  # combines the 3 precedent terms into one overall score
    "evaluate_complex",  # the interface - chemistry already known
    "PrecedentMatch",
    "find_closest_precedent",
]
