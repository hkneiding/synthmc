"""Building and representing the precedent corpora from BOS-Lig (+ BOS-TMC)
- ligand_corpus.py's 4 per-ligand levels, and environment_corpus.py's
whole-complex full-coordination-environment precedent. Nothing here does
precedent lookups or knows about a query structure; that's scoring/'s job."""

from synthmc.corpus.bostmc import load_complexes, load_refcode_to_metal
from synthmc.corpus.donor_sites import spec_mol2_to_smiles_and_donor_groups
from synthmc.corpus.environment_corpus import RADII, EnvironmentCorpus, build_environment_corpus
from synthmc.corpus.ligand_corpus import LigandCorpus, SignatureBucket, build_ligand_corpus
from synthmc.corpus.steric_corpus import STERIC_RADIUS, StericCorpus, build_steric_corpus

__all__ = [
    "LigandCorpus",
    "SignatureBucket",
    "build_ligand_corpus",
    "EnvironmentCorpus",
    "build_environment_corpus",
    "RADII",
    "StericCorpus",
    "build_steric_corpus",
    "STERIC_RADIUS",
    "load_refcode_to_metal",
    "load_complexes",
    "spec_mol2_to_smiles_and_donor_groups",
]
