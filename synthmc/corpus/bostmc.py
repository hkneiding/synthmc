from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

_COMP_SUFFIX_RE = re.compile(r"_comp_\d+$")


def _load_unambiguous(bostmc_dir: str | Path, usecols: list[str]) -> pd.DataFrame:
    """Load all BOS-TMC CSVs, dedup by refcode, and drop ambiguous
    multi-component base refcodes."""
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
    """base_refcode -> metal symbol."""
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
