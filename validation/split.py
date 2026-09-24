"""Pick a held-out set of BOS-TMC refcodes, shared across all three
precedent corpora - see package docstring for why a single shared split
matters (so "unseen" is the same population for ligand, environment, and
steric validation, rather than three independently-drawn samples that would
make cross-comparisons between them meaningless)."""

from __future__ import annotations

import random
from pathlib import Path

from synthmc.corpus.bostmc import load_complexes


def make_holdout_split(bostmc_dir: str | Path, holdout_size: int = 300, seed: int = 0) -> set[str]:
    """A reproducible random sample of `holdout_size` base_refcodes out of
    every BOS-TMC complex load_complexes can see (already restricted to
    unambiguous single-component refcodes - see bostmc.py)."""
    complexes = load_complexes(bostmc_dir)
    rng = random.Random(seed)
    return set(rng.sample(sorted(complexes.keys()), holdout_size))


__all__ = ["make_holdout_split"]
