"""synthmc: precedent-based ligand evaluation for transition metal complexes.

Package layout:
  common/  - small shared helpers (element data, graph utilities, donor
             signatures, donor-site environments) used by both corpus/ and
             scoring/.
  corpus/  - builds/represents the precedent corpora from BOS-Lig + BOS-TMC:
             LigandCorpus (4 per-ligand levels) and EnvironmentCorpus (the
             whole-complex full coordination-environment precedent).
  scoring/ - the core API: evaluate_complex(metal, ligands, corpus) against
             a corpus, given chemistry you already know.
  xyz/     - optional helper layer: detects a metal + ligands + binding
             atoms from a raw xyz structure, then calls scoring.evaluate_complex.
             Not the primary interface - see xyz/adapter.py.
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
from synthmc.xyz import evaluate_xyz_file, evaluate_xyz_structure

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
    "evaluate_complex",  # primary interface - chemistry already known
    "evaluate_xyz_structure",  # helper: detect chemistry from a raw xyz structure, then evaluate_complex
    "evaluate_xyz_file",  # helper: evaluate_xyz_structure over every frame in a tmQM-format xyz file
    "PrecedentMatch",
    "find_closest_precedent",
]
