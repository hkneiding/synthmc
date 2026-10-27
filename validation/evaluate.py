"""Query every held-out complex against the held-out corpora and tally
results for all three precedent terms - the actual validation step."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from synthmc.corpus.environment_corpus import RADII
from synthmc.scoring.evaluate import evaluate_complex
from synthmc.scoring.synthesizability import score_synthesizability
from validation.corpora import HoldoutCorpora
from validation.reconstruct import ReconstructedComplex

_LIGAND_LEVELS = ["by_ligand", "by_ligand_and_metal", "by_ligand_and_mode", "by_ligand_mode_and_metal"]


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


@dataclass
class PerComplexRecord:
    """One row per held-out complex - refcode-identified and aligned across
    all four scores (unlike the separate synth_*_values lists above, which
    are each independently filtered to non-None and so aren't safely
    index-aligned with each other in general). Used for plots that need to
    relate scores to each other or to metadata (metal, CN), or that need to
    name a specific example."""

    refcode: str
    metal: str
    cn: int | None
    ligand_score: float | None
    environment_score: float | None
    steric_score: float | None
    overall_score: float | None


@dataclass
class HoldoutValidationResult:
    n_holdout: int
    n_reconstructed: int
    # complexes whose reconstructed ligands were dropped because evaluate_complex raised
    # (a real parse failure, e.g. the pre-existing kekulization edge case seen in some
    # BOS-Lig SMILES - see project conversation) - evaluate_complex now raises rather than
    # silently degrading (see its docstring), so this batch-query loop has to tolerate that
    # itself, the same way xyz/adapter.py tolerates per-fragment detection failures.
    n_eval_failed: int = 0
    # per level: mean/median PrecedentMatch.score across every ligand instance that got a
    # match at all (None only means "no corpus data for this key", now a rare edge case
    # since there's no similarity threshold anymore - see scoring/matching.py)
    ligand_score_mean: dict[str, float | None] = field(default_factory=dict)
    ligand_score_median: dict[str, float | None] = field(default_factory=dict)
    ligand_n_no_data: dict[str, int] = field(default_factory=dict)  # instances with no bucket at all for this level
    # raw per-ligand-instance values behind ligand_score_mean/median above, one list per level,
    # for the per-level distribution plots
    ligand_score_values: dict[str, list[float]] = field(default_factory=dict)
    # mean/median of LigandPrecedent.precedent_score (the 4-level average) across every ligand instance
    ligand_combined_score_mean: float | None = None
    ligand_combined_score_median: float | None = None
    # per radius: fraction of held-out complexes whose joint coordination environment matched
    environment_hit_rate: dict[int, float] = field(default_factory=dict)
    environment_metal_hit_rate: dict[int, float] = field(default_factory=dict)
    steric_n_scored: int = 0  # complexes that got a non-None steric plausibility (had a corpus entry at their CN)
    steric_plausibility_mean: float | None = None
    steric_plausibility_median: float | None = None
    steric_plausibility_values: list[float] = field(default_factory=list)
    # SynthesizabilityResult's per-complex combined terms (score_synthesizability) - the
    # actual end-to-end scores a caller would see, as opposed to the diagnostic per-level/
    # per-radius breakdowns above. mean/median across held-out complexes that got a non-None
    # value for that term; n_scored is how many did.
    synth_ligand_score_mean: float | None = None
    synth_ligand_score_median: float | None = None
    synth_environment_score_mean: float | None = None
    synth_environment_score_median: float | None = None
    synth_environment_n_scored: int = 0
    synth_steric_score_mean: float | None = None
    synth_steric_score_median: float | None = None
    synth_overall_score_mean: float | None = None
    synth_overall_score_median: float | None = None
    # raw per-complex values behind the means/medians above, for plotting distributions
    synth_ligand_score_values: list[float] = field(default_factory=list)
    synth_environment_score_values: list[float] = field(default_factory=list)
    synth_overall_score_values: list[float] = field(default_factory=list)
    # per-radius percentile values behind synth_environment_score_values's per-complex
    # average (StructureResult.full_environment_percentile / full_environment_and_metal_percentile,
    # Section 3b) - kept separately per radius for the per-level distribution plots
    environment_percentile_values: dict[int, list[float]] = field(default_factory=dict)
    environment_and_metal_percentile_values: dict[int, list[float]] = field(default_factory=dict)
    # one aligned, refcode-identified row per held-out complex - see PerComplexRecord
    per_complex: list[PerComplexRecord] = field(default_factory=list)


def run_holdout_validation(
    reconstructed: dict[str, ReconstructedComplex], corpora: HoldoutCorpora, n_holdout: int
) -> HoldoutValidationResult:
    ligand_scores: dict[str, list[float]] = {level: [] for level in _LIGAND_LEVELS}
    ligand_no_data: dict[str, int] = {level: 0 for level in _LIGAND_LEVELS}
    combined_scores: list[float] = []

    env_hits = {r: 0 for r in RADII}
    env_metal_hits = {r: 0 for r in RADII}
    env_percentile_values: dict[int, list[float]] = {r: [] for r in RADII}
    env_metal_percentile_values: dict[int, list[float]] = {r: [] for r in RADII}

    steric_values: list[float] = []
    n_eval_failed = 0

    synth_ligand_scores: list[float] = []
    synth_environment_scores: list[float] = []
    synth_steric_scores: list[float] = []
    synth_overall_scores: list[float] = []
    per_complex: list[PerComplexRecord] = []

    for rc in reconstructed.values():
        try:
            result = evaluate_complex(
                rc.metal,
                rc.ligands,
                corpora.ligand,
                environment_corpus=corpora.environment,
                steric_corpus=corpora.steric,
            )
        except ValueError:
            n_eval_failed += 1
            continue

        for lig in result.ligands:
            for level in _LIGAND_LEVELS:
                match = getattr(lig, level)
                if match is not None:
                    ligand_scores[level].append(match.score)
                else:
                    ligand_no_data[level] += 1
            combined_scores.append(lig.precedent_score)

        for r in RADII:
            if result.full_environment.get(r) is not None:
                env_hits[r] += 1
            if result.full_environment_and_metal.get(r) is not None:
                env_metal_hits[r] += 1
            p = result.full_environment_percentile.get(r)
            if p is not None:
                env_percentile_values[r].append(p)
            p_metal = result.full_environment_and_metal_percentile.get(r)
            if p_metal is not None:
                env_metal_percentile_values[r].append(p_metal)

        if result.steric is not None:
            steric_values.append(result.steric.plausibility)

        synth = score_synthesizability(result)
        if synth.ligand_score is not None:
            synth_ligand_scores.append(synth.ligand_score)
        if synth.environment_score is not None:
            synth_environment_scores.append(synth.environment_score)
        if synth.steric_score is not None:
            synth_steric_scores.append(synth.steric_score)
        if synth.overall_score is not None:
            synth_overall_scores.append(synth.overall_score)

        per_complex.append(
            PerComplexRecord(
                refcode=rc.refcode,
                metal=result.metal_symbol,
                cn=result.steric.cn if result.steric is not None else None,
                ligand_score=synth.ligand_score,
                environment_score=synth.environment_score,
                steric_score=synth.steric_score,
                overall_score=synth.overall_score,
            )
        )

    n_reconstructed = len(reconstructed)
    out = HoldoutValidationResult(n_holdout=n_holdout, n_reconstructed=n_reconstructed, n_eval_failed=n_eval_failed)

    for level in _LIGAND_LEVELS:
        values = ligand_scores[level]
        out.ligand_score_mean[level] = statistics.mean(values) if values else None
        out.ligand_score_median[level] = statistics.median(values) if values else None
        out.ligand_n_no_data[level] = ligand_no_data[level]
        out.ligand_score_values[level] = values

    if combined_scores:
        out.ligand_combined_score_mean = statistics.mean(combined_scores)
        out.ligand_combined_score_median = statistics.median(combined_scores)

    for r in RADII:
        out.environment_hit_rate[r] = env_hits[r] / n_reconstructed if n_reconstructed else 0.0
        out.environment_metal_hit_rate[r] = env_metal_hits[r] / n_reconstructed if n_reconstructed else 0.0

    out.steric_n_scored = len(steric_values)
    out.steric_plausibility_values = steric_values
    if steric_values:
        out.steric_plausibility_mean = statistics.mean(steric_values)
        out.steric_plausibility_median = statistics.median(steric_values)

    if synth_ligand_scores:
        out.synth_ligand_score_mean = statistics.mean(synth_ligand_scores)
        out.synth_ligand_score_median = statistics.median(synth_ligand_scores)
    out.synth_environment_n_scored = len(synth_environment_scores)
    if synth_environment_scores:
        out.synth_environment_score_mean = statistics.mean(synth_environment_scores)
        out.synth_environment_score_median = statistics.median(synth_environment_scores)
    if synth_steric_scores:
        out.synth_steric_score_mean = statistics.mean(synth_steric_scores)
        out.synth_steric_score_median = statistics.median(synth_steric_scores)
    if synth_overall_scores:
        out.synth_overall_score_mean = statistics.mean(synth_overall_scores)
        out.synth_overall_score_median = statistics.median(synth_overall_scores)

    out.synth_ligand_score_values = synth_ligand_scores
    out.synth_environment_score_values = synth_environment_scores
    out.synth_overall_score_values = synth_overall_scores
    out.environment_percentile_values = env_percentile_values
    out.environment_and_metal_percentile_values = env_metal_percentile_values
    out.per_complex = per_complex

    return out


def print_report(result: HoldoutValidationResult) -> None:
    print(
        f"Held-out complexes: {result.n_holdout}, reconstructed cleanly: {result.n_reconstructed}, "
        f"evaluate_complex failed on: {result.n_eval_failed}"
    )
    print()
    print("Ligand precedent (score = similarity * count_percentile, per ligand instance):")
    for level in _LIGAND_LEVELS:
        mean = result.ligand_score_mean[level]
        median = result.ligand_score_median[level]
        mean_s = f"{mean:.3f}" if mean is not None else "n/a"
        median_s = f"{median:.3f}" if median is not None else "n/a"
        print(f"  {level:28s} mean={mean_s}  median={median_s}  no-data={result.ligand_n_no_data[level]}")
    if result.ligand_combined_score_mean is not None:
        print(
            f"  {'combined (4-level average)':28s} mean={result.ligand_combined_score_mean:.3f}"
            f"  median={result.ligand_combined_score_median:.3f}"
        )
    print()
    print("Coordination environment precedent (per complex):")
    for r in RADII:
        print(
            f"  radius {r}: any-metal={result.environment_hit_rate[r]:.1%}"
            f"  metal-specific={result.environment_metal_hit_rate[r]:.1%}"
        )
    print()
    print("Steric plausibility:")
    print(f"  scored: {result.steric_n_scored}/{result.n_reconstructed}")
    if result.steric_plausibility_mean is not None:
        print(f"  mean={result.steric_plausibility_mean:.3f}  median={result.steric_plausibility_median:.3f}")
    print()
    print("Combined SynthesizabilityResult (per held-out complex - the actual end-to-end score):")
    print(f"  ligand_score      mean={_fmt(result.synth_ligand_score_mean)}  median={_fmt(result.synth_ligand_score_median)}")
    print(
        f"  environment_score mean={_fmt(result.synth_environment_score_mean)}"
        f"  median={_fmt(result.synth_environment_score_median)}"
        f"  scored={result.synth_environment_n_scored}/{result.n_reconstructed}"
    )
    print(f"  steric_score      mean={_fmt(result.synth_steric_score_mean)}  median={_fmt(result.synth_steric_score_median)}")
    print(f"  overall_score     mean={_fmt(result.synth_overall_score_mean)}  median={_fmt(result.synth_overall_score_median)}")


__all__ = ["HoldoutValidationResult", "PerComplexRecord", "run_holdout_validation", "print_report"]
