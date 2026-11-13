from __future__ import annotations

import csv
import random
from pathlib import Path

from synthmc.corpus.bostmc import load_complexes


def make_holdout_split(bostmc_dir: str | Path, holdout_size: int = 300, seed: int = 0) -> set[str]:
    """A reproducible random sample of `holdout_size` BOS-TMC refcodes."""
    complexes = load_complexes(bostmc_dir)
    rng = random.Random(seed)
    return set(rng.sample(sorted(complexes.keys()), holdout_size))


def make_year_holdout_split(
    bostmc_dir: str | Path, years_csv: str | Path, holdout_size: int = 15287
) -> set[str]:
    """The `holdout_size` most-recently-deposited BOS-TMC refcodes, by
    `deposition_date` in `years_csv`."""
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
