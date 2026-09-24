"""Ligand precedent lookup: the single most similar corpus ligand sharing
the query's donor_signature (same coordination mode) - always the true
nearest neighbor by Tanimoto similarity, never overridden by a less similar
but more common alternative (see find_closest_precedent's docstring for why
that matters for interpretability).

Reported as a single combined score = similarity * count_percentile, where
count_percentile is an empirical CDF against how often *other* corpus
ligands at this same level occur (see count_percentile below) - not a hand-
picked saturation constant. No similarity threshold: a weak match to
something common still gets a low score because similarity is low, not
because it was gated out - the product decays continuously instead of
falling off a cliff at some arbitrary cutoff.

`None` means "no data for this key at all" (e.g. BOS-TMC has zero lanthanum
entries, so no metal-specific bucket can ever exist for La) - a real data
gap, distinct from a real but low score, which is what a poor match now
looks like instead of a second flavor of None.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from rdkit import DataStructs

from synthmc.corpus.ligand_corpus import SignatureBucket
from synthmc.corpus.steric_corpus import StericCorpus


def count_percentile(count: int, sorted_counts: list[int]) -> float:
    """Empirical CDF: the fraction of `sorted_counts` (a level's pooled,
    sorted occurrence counts - see corpus/ligand_corpus.py's
    count_distribution) that are <= `count`. 1.0 means this count is at
    least as large as everything on record at this level (about as
    precedented as anything gets); near 0 means almost everything else at
    this level is more common than this.

    The opposite tail from find_steric_plausibility's survival function:
    there, more bulk was worse; here, more occurrences is better, so this
    is a lower-tail CDF, not an upper-tail survival function."""
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
    """The single closest-matching ligand in `bucket` (already restricted
    to the query's donor_signature, and optionally its metal - see
    corpus/ligand_corpus.py) - unconditionally the true nearest neighbor,
    never a different, less similar corpus entry chosen because it scores
    higher once its count is factored in. That matters for interpretability:
    if an exact match exists in the bucket, nothing can be more similar than
    identical, so it's always selected and its score can only ever be
    limited by its *own* count, never overridden by an unrelated but more
    common structure elsewhere in the bucket - the reported "closest
    precedent" stays a real, explainable analog, not an artifact of jointly
    optimizing similarity and frequency together.

    None only if `bucket` itself has no data (missing key or empty) - never
    because the best match was "too dissimilar", since there's no threshold
    anymore; a weak match just produces a low `score` instead.

    Ties on similarity are broken by higher occurrence count (most
    similar, and among equally similar, most precedented)."""
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
    """Exact-match-only lookup for corpus/environment_corpus.py's full
    coordination-environment precedent - just a dict lookup, but named and
    documented like the other lookups here rather than inlined, since the
    "exact match only, no fallback" choice is deliberate (see
    environment_corpus.py's module docstring for why) and worth being
    explicit about at the call site."""
    return bucket.get(key)


def find_steric_plausibility(normalized_bulk: float, cn: int, corpus: StericCorpus) -> float | None:
    """One-sided empirical survival function against corpus/steric_corpus.py's
    per-coordination-number bulk distribution: the fraction of real
    complexes at this same coordination number whose own normalized total
    bulk was >= `normalized_bulk`. 1.0 means this complex is no bulkier
    than anything on record at this CN; it only decays toward 0 as bulk
    exceeds what's actually been accommodated.

    Deliberately one-sided rather than a two-sided z-score: being *less*
    bulky than typical isn't implausible (six small ligands like water or
    fluoride around a metal is completely normal), so only the upper tail
    should be penalized.

    None if no corpus complex shares this coordination number at all - same
    "no data" meaning as find_closest_precedent's None, not "implausible"."""
    values = corpus.by_cn.get(cn)
    if not values:
        return None
    n_at_or_above = len(values) - bisect_left(values, normalized_bulk)
    return n_at_or_above / len(values)


__all__ = [
    "PrecedentMatch",
    "find_closest_precedent",
    "count_percentile",
    "find_environment_count",
    "find_steric_plausibility",
]
