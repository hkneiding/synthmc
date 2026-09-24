"""Ligand precedent corpus: a 4-level breakdown of how often a ligand
structure has been observed, each level scoped by a different combination
of metal/coordination-mode specificity:

  1. by_ligand              - this ligand, any metal, any coordination mode
  2. by_ligand_and_metal    - this ligand with this metal, any mode
  3. by_ligand_and_mode     - this ligand in this coordination mode, any metal
  4. by_ligand_mode_and_metal - this ligand in this mode, with this metal

Levels 1-2 use BOS-Lig's `gen_gh`/`gen_mol2`/`gen_count`/`gen_refcode`
columns (metal/mode-agnostic ligand identity - a ligand that binds several
different ways still gets one gen_gh). Levels 3-4 use `spec_gh`/`spec_mol2`/
`spec_count`/`spec_refcode` (one row per distinct binding mode) plus
donor_signature.py's donor-atom-group signature, since coordination mode is
inherently a property of one specific binding instance, not of the generic
ligand identity gen_* captures.

Levels 1 and 3 are deliberately independent of donor-detection correctness
(query-time donor_groups perception, see xyz/fragments.py) - useful as a check
when 3/4 come up empty: if 1 shows a ligand is extremely common but 3 shows
no precedent for its apparent coordination mode, that's a signal the
donor_signature was probably computed wrong, not that the ligand is
actually novel.

spec_mol2 blocks carry a dummy metal atom (SYBYL type "X") whose bonds mark
the real donor atoms - donor_sites.py strips it out and returns the
remaining donor-atom groups (already mol2-index-aligned, since mol2.py's
parse_mol2 never removes any atoms, unlike an earlier version that called
RemoveHs - see its docstring). gen_mol2 has no dummy atom at all (it's
already metal-free) and is stored as a stringified {'count': N, 'mol2':
'...'} dict, unlike spec_mol2's raw text. Bond orders/aromaticity come
straight from the mol2 (SYBYL types, generated via molSimplify from the
original CSD structures) - no bond-order guessing needed on the corpus
side, unlike for a fresh query structure (see xyz/ligand_chem.py, which
perceives bonds from raw coordinates via OpenBabel).

Canonical SMILES keep every hydrogen explicit, matching how both gen_mol2
and spec_mol2 already list them - and matching xyz/ligand_chem.py's query-side
convention (OpenBabel's "h" output option), since an identical ligand has
to land on the same canonical string everywhere or exact-match lookups
would never fire.

Levels 2 and 4 need a second data source: BOS-Lig alone has no metal column
(it's deliberately metal-agnostic, per its own design), so getting "how
often with this specific metal" means cross-referencing gen_refcode/
spec_refcode against BOS-TMC's own refcode -> metal table (bostmc.py) -
this reintroduces the BOS-TMC dependency levels 1/3 don't need, on purpose,
only where the question actually requires it.
"""

from __future__ import annotations

import ast
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from synthmc.corpus.bostmc import load_refcode_to_metal
from synthmc.common.donor_signature import donor_signature
from synthmc.corpus.donor_sites import _strip_metal_dummy
from synthmc.common.fingerprint import morgan_fingerprint
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")

_COMP_SUFFIX_RE = re.compile(r"_comp_\d+$")


@dataclass
class SignatureBucket:
    """Everything precedented for one (level-dependent) key - e.g. just a
    metal, just a donor_signature, both, or neither."""

    counts: dict[str, int] = field(default_factory=dict)  # canonical SMILES -> occurrence count
    smiles_list: list[str] = field(default_factory=list)  # parallel to fingerprints, for bulk similarity search
    fingerprints: list = field(default_factory=list)  # rdkit ExplicitBitVects, same order as smiles_list


@dataclass
class LigandCorpus:
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

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LigandCorpus":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


def _unwrap_gen_mol2(cell: str) -> str:
    """gen_mol2 cells are a stringified {'count': N, 'mol2': '...'} dict -
    unlike spec_mol2, which is raw mol2 text."""
    return ast.literal_eval(cell)["mol2"]


def _finalize_buckets(bucket_counts: dict, bucket_fps: dict) -> dict:
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
    progress_every: int = 10000,
    progress_callback=None,
    exclude_refcodes: set[str] | None = None,
    exclude_gen_counts: dict[str, int] | None = None,
    exclude_spec_counts: dict[str, int] | None = None,
) -> LigandCorpus:
    """`exclude_refcodes` drops held-out refcodes from the metal-specific
    buckets (levels 2/4) - exact, since those are built by counting refcode
    presence in gen_refcode/spec_refcode, so a held-out refcode can simply
    be skipped while iterating.

    Levels 1/3 (by_ligand, by_ligand_and_mode) are different: gen_count/
    spec_count are BOS-Lig's own pre-aggregated *total* occurrence counts
    (not derived from refcode-list length - see this module's docstring on
    why that list is a deduplicated set, not a multiset), so there's no way
    to exclude a held-out refcode's contribution from those totals just by
    knowing its refcode string. `exclude_gen_counts`/`exclude_spec_counts`
    (gen_gh/spec_gh -> exact occurrence count to subtract) carry that
    information instead - see validation/reconstruct.py, which derives it
    from complex_ligands.reconstruct_ligand_multiset the same way
    corpus/environment_corpus.py and corpus/steric_corpus.py already
    reconstruct exact per-complex multiplicities, just exposed here instead
    of only being tallied internally."""
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
