from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from rdkit import DataStructs

from synthmc.corpus.ligand_corpus import SignatureBucket


def count_percentile(count: int, sorted_counts: list[int]) -> float:
    """Empirical CDF: fraction of `sorted_counts` that are <= `count`."""
    if not sorted_counts:
        return 0.0
    n_le = bisect_right(sorted_counts, count)
    return n_le / len(sorted_counts)


@dataclass
class PrecedentMatch:
    smiles: str  # the closest corpus ligand's SMILES - equals the query's own SMILES on an exact match
    similarity: float  # Tanimoto similarity to the query; 1.0 on an exact match
    count: int  # that corpus ligand's raw occurrence count
    score: float  # similarity * count_percentile(count, ...) - see module docstring


def find_closest_precedent(
    smiles: str, fingerprint, bucket: SignatureBucket | None, count_distribution: list[int] | None
) -> PrecedentMatch | None:
    """The single closest-matching ligand in `bucket` by Tanimoto
    similarity, or None if `bucket` has no data."""
    if bucket is None or not bucket.fingerprints:
        return None

    if smiles in bucket.counts:
        best_smiles, similarity, count = smiles, 1.0, bucket.counts[smiles]
    else:
        sims = DataStructs.BulkTanimotoSimilarity(fingerprint, bucket.fingerprints)
        best_idx = max(range(len(sims)), key=lambda i: (sims[i], bucket.counts[bucket.smiles_list[i]]))
        best_smiles = bucket.smiles_list[best_idx]
        similarity, count = sims[best_idx], bucket.counts[best_smiles]

    percentile = count_percentile(count, count_distribution or [])
    return PrecedentMatch(smiles=best_smiles, similarity=similarity, count=count, score=similarity * percentile)


def find_environment_count(key, bucket: dict) -> int | None:
    """Exact-match-only lookup for the coordination-environment corpus."""
    return bucket.get(key)


def environment_percentile(key, bucket: dict, count_distribution: list[int]) -> float:
    """How common this exact joint environment is, relative to every joint
    environment on record (0.0 if `key` has no match)."""
    return count_percentile(bucket.get(key, 0), count_distribution)


def find_steric_plausibility(normalized_bulk: float, values: list[float] | None) -> float | None:
    """Fraction of `values` (same-CN complexes' normalized bulk) that are
    >= `normalized_bulk`; None if `values` is empty."""
    if not values:
        return None
    n_at_or_above = len(values) - bisect_left(values, normalized_bulk)
    return n_at_or_above / len(values)


__all__ = [
    "PrecedentMatch",
    "find_closest_precedent",
    "count_percentile",
    "find_environment_count",
    "environment_percentile",
    "find_steric_plausibility",
]
