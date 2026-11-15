from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from synthmc.corpus.bostmc import load_refcode_to_metal
from synthmc.common.donor_signature import donor_signature
from synthmc.corpus._persistence import PicklePersistable
from synthmc.corpus.donor_sites import _strip_metal_dummy
from synthmc.common.fingerprint import morgan_fingerprint
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")

_COMP_SUFFIX_RE = re.compile(r"_comp_\d+$")


@dataclass
class SignatureBucket:
    """Everything precedented for one level-dependent key."""

    counts: dict[str, int] = field(default_factory=dict)  # canonical SMILES -> occurrence count
    smiles_list: list[str] = field(default_factory=list)  # parallel to fingerprints, for bulk similarity search
    fingerprints: list = field(default_factory=list)  # rdkit ExplicitBitVects, same order as smiles_list


@dataclass
class LigandCorpus(PicklePersistable):
    by_ligand: SignatureBucket = field(default_factory=SignatureBucket)  # level 1
    by_ligand_and_metal: dict[str, SignatureBucket] = field(default_factory=dict)  # level 2
    by_ligand_and_mode: dict[tuple[str, ...], SignatureBucket] = field(default_factory=dict)  # level 3
    by_ligand_mode_and_metal: dict[tuple[str, tuple[str, ...]], SignatureBucket] = field(default_factory=dict)  # level 4
    n_gen_seen: int = 0
    n_gen_parse_failures: int = 0
    n_spec_seen: int = 0
    n_spec_parse_failures: int = 0
    # level name -> every occurrence count at that level, pooled across *all*
    # of its sub-buckets (e.g. across every metal for by_ligand_and_metal)
    # and sorted - the reference population scoring/matching.py's
    # count_percentile ranks a query count against. Pooled rather than kept
    # per-sub-bucket so a rare metal/mode's small sub-bucket doesn't give a
    # coarse, near-meaningless percentile - same reasoning as
    # corpus/steric_corpus.py conditioning only on CN, not (metal, CN).
    count_distribution: dict[str, list[int]] = field(default_factory=dict)


def _unwrap_gen_mol2(cell: str) -> str:
    """Unwrap a gen_mol2 cell's {'count': N, 'mol2': '...'} string."""
    return ast.literal_eval(cell)["mol2"]


def _finalize_buckets(bucket_counts: dict, bucket_fps: dict) -> dict:
    """Turn parallel key -> Counter / key -> {smiles: fingerprint} maps
    accumulated during a build pass into key -> SignatureBucket."""
    buckets = {}
    for key, counts in bucket_counts.items():
        fps = bucket_fps[key]
        buckets[key] = SignatureBucket(
            counts=dict(counts),
            smiles_list=list(fps.keys()),
            fingerprints=[fps[s] for s in fps],
        )
    return buckets


def _refcode_metal_counts(
    refcode_list_cell: str, refcode_to_metal: dict[str, str], exclude_refcodes: set[str] | None = None
) -> Counter:
    """metal -> occurrence count across `refcode_list_cell`'s refcodes,
    skipping `exclude_refcodes` and refcodes with no known metal."""
    counts: Counter = Counter()
    for refcode in ast.literal_eval(refcode_list_cell):
        base = _COMP_SUFFIX_RE.sub("", refcode)
        if exclude_refcodes and base in exclude_refcodes:
            continue
        metal = refcode_to_metal.get(base)
        if metal is not None:
            counts[metal] += 1
    return counts


def build_ligand_corpus(
    boslig_csv: str | Path,
    bostmc_dir: str | Path,
    progress_every: int = 20000,
    progress_callback=None,
    exclude_refcodes: set[str] | None = None,
    exclude_gen_counts: dict[str, int] | None = None,
    exclude_spec_counts: dict[str, int] | None = None,
) -> LigandCorpus:
    """Build the 4-level ligand corpus, optionally excluding held-out
    refcodes/counts for validation."""
    corpus = LigandCorpus()
    refcode_to_metal = load_refcode_to_metal(bostmc_dir)

    # --- Levels 1-2: gen_gh (metal/mode-agnostic ligand identity) ---
    lig = pd.read_csv(boslig_csv)
    gen = lig.drop_duplicates(subset="gen_gh")  # one row per unique generic ligand

    gen_counts: Counter = Counter()
    gen_fps: dict[str, object] = {}
    gen_metal_bucket_counts: dict[str, Counter] = {}
    gen_metal_bucket_fps: dict[str, dict[str, object]] = {}

    n_gen_seen = 0
    for _, row in gen.iterrows():
        n_gen_seen += 1
        try:
            mol2_text = _unwrap_gen_mol2(row["gen_mol2"])
            mol = parse_mol2(mol2_text)
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            corpus.n_gen_parse_failures += 1
            if progress_callback is not None and n_gen_seen % progress_every == 0:
                progress_callback("gen", n_gen_seen, len(gen), corpus)
            continue

        effective_count = int(row["gen_count"]) - (exclude_gen_counts or {}).get(row["gen_gh"], 0)
        if effective_count > 0:
            gen_counts[smiles] += effective_count
            if smiles not in gen_fps:
                gen_fps[smiles] = morgan_fingerprint(mol)

        for metal, metal_count in _refcode_metal_counts(row["gen_refcode"], refcode_to_metal, exclude_refcodes).items():
            gen_metal_bucket_counts.setdefault(metal, Counter())[smiles] += metal_count
            mfps = gen_metal_bucket_fps.setdefault(metal, {})
            if smiles not in mfps:
                mfps[smiles] = morgan_fingerprint(mol)

        if progress_callback is not None and n_gen_seen % progress_every == 0:
            progress_callback("gen", n_gen_seen, len(gen), corpus)

    corpus.n_gen_seen = n_gen_seen
    corpus.by_ligand = SignatureBucket(
        counts=dict(gen_counts), smiles_list=list(gen_fps.keys()), fingerprints=[gen_fps[s] for s in gen_fps]
    )
    corpus.by_ligand_and_metal = _finalize_buckets(gen_metal_bucket_counts, gen_metal_bucket_fps)

    # --- Levels 3-4: spec_gh (one row per distinct binding mode) ---
    mode_bucket_counts: dict[tuple[str, ...], Counter] = {}
    mode_bucket_fps: dict[tuple[str, ...], dict[str, object]] = {}
    mode_metal_bucket_counts: dict[tuple[str, tuple[str, ...]], Counter] = {}
    mode_metal_bucket_fps: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}

    n_spec_seen = 0
    for _, row in lig.iterrows():  # spec_gh is already unique per row - no dedup needed (confirmed: 72,395/72,395)
        n_spec_seen += 1
        try:
            trimmed_mol2, groups = _strip_metal_dummy(row["spec_mol2"])
            mol = parse_mol2(trimmed_mol2)
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            signature = donor_signature(symbols, groups)
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            corpus.n_spec_parse_failures += 1
            if progress_callback is not None and n_spec_seen % progress_every == 0:
                progress_callback("spec", n_spec_seen, len(lig), corpus)
            continue

        effective_count = int(row["spec_count"]) - (exclude_spec_counts or {}).get(row["spec_gh"], 0)
        if effective_count > 0:
            mode_bucket_counts.setdefault(signature, Counter())[smiles] += effective_count
            fps = mode_bucket_fps.setdefault(signature, {})
            if smiles not in fps:
                fps[smiles] = morgan_fingerprint(mol)

        for metal, metal_count in _refcode_metal_counts(row["spec_refcode"], refcode_to_metal, exclude_refcodes).items():
            key = (metal, signature)
            mode_metal_bucket_counts.setdefault(key, Counter())[smiles] += metal_count
            mfps = mode_metal_bucket_fps.setdefault(key, {})
            if smiles not in mfps:
                mfps[smiles] = morgan_fingerprint(mol)

        if progress_callback is not None and n_spec_seen % progress_every == 0:
            progress_callback("spec", n_spec_seen, len(lig), corpus)

    corpus.n_spec_seen = n_spec_seen
    corpus.by_ligand_and_mode = _finalize_buckets(mode_bucket_counts, mode_bucket_fps)
    corpus.by_ligand_mode_and_metal = _finalize_buckets(mode_metal_bucket_counts, mode_metal_bucket_fps)

    corpus.count_distribution = {
        "by_ligand": sorted(corpus.by_ligand.counts.values()),
        "by_ligand_and_metal": sorted(
            count for bucket in corpus.by_ligand_and_metal.values() for count in bucket.counts.values()
        ),
        "by_ligand_and_mode": sorted(
            count for bucket in corpus.by_ligand_and_mode.values() for count in bucket.counts.values()
        ),
        "by_ligand_mode_and_metal": sorted(
            count for bucket in corpus.by_ligand_mode_and_metal.values() for count in bucket.counts.values()
        ),
    }

    return corpus


__all__ = ["LigandCorpus", "SignatureBucket", "build_ligand_corpus"]
