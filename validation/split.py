"""Pick a held-out set of BOS-TMC refcodes, shared across all three
precedent corpora - see package docstring for why a single shared split
matters (so "unseen" is the same population for ligand, environment, and
steric validation, rather than three independently-drawn samples that would
make cross-comparisons between them meaningless)."""

from __future__ import annotations

import csv
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


def make_year_holdout_split(
    bostmc_dir: str | Path, years_csv: str | Path, holdout_size: int = 15287
) -> set[str]:
    """The `holdout_size` most-recently-deposited BOS-TMC refcodes (by CSD
    `deposition_date`, from `years_csv` - refcode,deposition_date,
    publication_year,error), restricted to load_complexes's unambiguous
    population the same way make_holdout_split is. Size-matched to the
    random split rather than a calendar-year cutoff, deliberately - CSD
    deposition rate isn't flat over time, so a fixed-year cutoff confounds
    "temporal" with "sample size" as the thing that changed; picking
    "most recent N" fixes N and lets the cutoff year fall out of that.

    Refcodes with no usable deposition_date (CCDC query error, or not in
    years_csv at all) are dropped from the eligible population entirely -
    the same "narrow accepted gap over a forced guess" principle used
    elsewhere in this project - and are never training data either, since
    we can't verify they predate the holdout cutoff."""
    complexes = load_complexes(bostmc_dir)

    dated: list[tuple[str, str]] = []  # (refcode, deposition_date), sortable as ISO strings
    with open(years_csv, newline="") as f:
        for row in csv.DictReader(f):
            rc = row["refcode"]
            if rc in complexes and row["deposition_date"] and not row["error"]:
                dated.append((rc, row["deposition_date"]))

    if len(dated) < holdout_size:
        raise ValueError(f"only {len(dated)} dated refcodes available, need {holdout_size}")

    dated.sort(key=lambda pair: (pair[1], pair[0]))  # oldest first, refcode as tiebreak
    return {rc for rc, _ in dated[-holdout_size:]}


__all__ = ["make_holdout_split", "make_year_holdout_split"]
