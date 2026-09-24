"""Minimal BOS-TMC reader.

load_refcode_to_metal: refcode -> metal, for cross-referencing BOS-Lig's
spec_refcode lists to attribute a specific ligand occurrence to a specific
metal (see ligand_corpus.py's metal-specific precedent buckets).

load_complexes: refcode -> {metal, coordination_formula}, for
complex_ligands.py's full-complex ligand-multiset reconstruction (needs
the per-ligand-formula multiplicities coordination_formula carries, e.g.
"Ir (BC29H29N)2 C5H7O2" - verified self-consistent with BOS-TMC's own
`natoms` field on a 3000-complex sample, zero mismatches).

Neither reads the full BOS-TMC x BOS-Lig join the retired first version of
this project's corpus did (see legacy/synthmc/corpus.py) - just the columns
each specific need actually requires.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

_COMP_SUFFIX_RE = re.compile(r"_comp_\d+$")


def _load_unambiguous(bostmc_dir: str | Path, usecols: list[str]) -> pd.DataFrame:
    """Shared loading step: concat all 6 spin-state CSVs, dedup by refcode,
    then drop multi-component base refcodes (ambiguous - which component
    does a given ligand/metal actually belong to - dropped rather than
    risk misattribution; same rule the original BOS-TMC x BOS-Lig join
    used, ~1.8% of entries)."""
    files = sorted(glob.glob(str(Path(bostmc_dir) / "dataset/properties/terachem/BOSTMC-*.csv")))
    if not files:
        raise FileNotFoundError(f"No BOSTMC-*.csv files found under {bostmc_dir}")
    dfs = [pd.read_csv(f, usecols=usecols) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="refcode")

    df["base_refcode"] = df["refcode"].apply(lambda r: _COMP_SUFFIX_RE.sub("", r))
    counts = df["base_refcode"].value_counts()
    unambiguous = set(counts[counts == 1].index)
    return df[df["base_refcode"].isin(unambiguous)]


def load_refcode_to_metal(bostmc_dir: str | Path) -> dict[str, str]:
    """base_refcode (BOS-TMC's refcode_comp_N with the suffix stripped, to
    match BOS-Lig's bare-refcode convention) -> metal symbol."""
    df = _load_unambiguous(bostmc_dir, ["refcode", "metal"])
    return dict(zip(df["base_refcode"], df["metal"]))


def load_complexes(bostmc_dir: str | Path) -> dict[str, dict]:
    """base_refcode -> {"metal": str, "coordination_formula": str}."""
    df = _load_unambiguous(bostmc_dir, ["refcode", "metal", "coordination formula"])
    return {
        base_refcode: {"metal": metal, "coordination_formula": cf}
        for base_refcode, metal, cf in zip(df["base_refcode"], df["metal"], df["coordination formula"])
    }


__all__ = ["load_refcode_to_metal", "load_complexes"]
