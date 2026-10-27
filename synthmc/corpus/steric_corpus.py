"""The steric-plausibility precedent: for a given coordination number, how
much total ligand bulk has actually been accommodated by real complexes on
record - the third additive scoring term alongside ligand precedent
(ligand_corpus.py) and coordination-environment precedent
(environment_corpus.py).

"Bulk" here is purely topological (common/environment.py's ligand_bulk: heavy
atom count out to STERIC_RADIUS bonds from any of a ligand's donor sites,
de-duplicated so an atom shared between two nearby sites' neighborhoods -
routine for a close-bite multidentate ligand - counts once, not once per
site that reaches it; summed across every ligand in the complex, including
repeated ligands) - no conformer generation, no idealized coordination
polyhedron, no 3D geometry at all. STERIC_RADIUS=2 corresponds to "out to
the 3rd coordination sphere counted from the metal": the donor atoms
themselves (bonded directly to the metal) are the 1st sphere, so they sit at
site_environment/ligand_bulk's radius=0; radius=1 reaches the 2nd sphere;
radius=2 reaches the 3rd.

Total bulk is normalized by the metal's covalent radius before being
compared across complexes - a metal with a longer covalent radius holds its
donor atoms (and everything attached to them) physically further apart at
the same coordination number and geometry, so raw bulk isn't comparable
across different metals without accounting for that. This is a first-pass,
deliberately simple normalization (divide by covalent_radius(metal) alone,
not e.g. also scaled by the coordination geometry's bite angle) - flagged
as revisitable, not a settled physical model, the same way
METAL_DONOR_BOND_TOLERANCE is.

The corpus itself only needs to be conditioned on coordination number, not
on metal identity too - stratifying by (metal, CN) the way
environment_corpus.py stratifies by (metal, environment) would hit the same
sparsity problem the held-out environment test surfaced for rare metals.
The covalent-radius normalization is what lets metal identity influence the
comparison without fragmenting the reference population that way.

A (metal, CN)-bucketed variant (with backoff to this pooled corpus below a
minimum sample size) was built and held-out-validated as an alternative: it
measurably fixed a real, explicable distortion for specific metals whose
characteristic ligand bulk differs from the CN-pooled average for genuine
chemical reasons (Rh/Ir especially, heavily represented in BOS-TMC by
bulky-phosphine/Cp* homogeneous-catalysis complexes, so their real,
well-precedented chemistry reads as implausibly bulky against a pool
dominated by metals carrying much smaller routine ligands). Rejected anyway
in favor of this single pooled distribution: the bucketed version traded
that per-metal fix for a harder-to-reason-about mechanism overall (a hard
sample-size cutoff with no interpolation, so two metal/CN cells of nearly
identical size can be scored against entirely different reference
populations depending which side of the threshold they fall on) and a
small but real degradation in aggregate calibration (see scoring/matching.py's
find_steric_plausibility docstring and the validation writeup). One clean,
uniformly-defined reference population was judged more valuable than a
locally-more-accurate but structurally more complex one, especially since
the rejected alternative's failure mode (Rh/Ir reading as sterically
atypical) is itself well understood and can be reasoned about directly
rather than silently absorbed into the score.

See scoring/matching.py's find_steric_plausibility for how a query value is
turned into a 0-1 score from this corpus (a one-sided empirical survival
function, not a two-sided z-score - see that function's docstring for why).
"""

from __future__ import annotations

import ast
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from synthmc.common.elements import covalent_radius
from synthmc.common.environment import ligand_bulk
from synthmc.corpus.bostmc import load_complexes
from synthmc.corpus.complex_ligands import reconstruct_ligand_multiset
from synthmc.corpus.donor_sites import _strip_metal_dummy
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")

STERIC_RADIUS = 2  # see module docstring - "out to the 3rd coordination sphere from the metal"


@dataclass
class StericCorpus:
    # coordination number -> sorted list of normalized total-bulk values
    # observed across real complexes with that CN.
    by_cn: dict[int, list[float]] = field(default_factory=dict)
    n_complexes_seen: int = 0
    n_complexes_used: int = 0  # reconstructed cleanly, every candidate parsed, and metal had a tabulated radius

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "StericCorpus":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


def build_steric_corpus(
    boslig_csv: str | Path,
    bostmc_dir: str | Path,
    progress_every: int = 20000,
    progress_callback=None,
    exclude_refcodes: set[str] | None = None,
) -> StericCorpus:
    """`exclude_refcodes` supports building a proper held-out corpus for
    validation, the same way environment_corpus.py's does - complexes in
    this set are skipped entirely, as if they didn't exist in BOS-TMC."""
    corpus = StericCorpus()

    complexes = load_complexes(bostmc_dir)  # base_refcode -> {metal, coordination_formula}
    if exclude_refcodes:
        complexes = {rc: info for rc, info in complexes.items() if rc not in exclude_refcodes}

    lig = pd.read_csv(boslig_csv)

    # Parse every spec_gh row once (Mol + donor groups + molecular formula),
    # and build refcode -> list of spec_gh indices touching it. Mirrors
    # environment_corpus.py's build loop exactly (same source data, same
    # reconstruction mechanism) - only what's tallied per complex differs.
    parsed: dict[str, tuple] = {}  # spec_gh -> (mol, groups, formula_counter)
    refcode_to_spec_ghs: dict[str, list[str]] = {}
    n_seen = 0
    for _, row in lig.iterrows():
        n_seen += 1
        spec_gh = row["spec_gh"]
        try:
            trimmed_mol2, groups = _strip_metal_dummy(row["spec_mol2"])
            mol = parse_mol2(trimmed_mol2)
            formula = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
            parsed[spec_gh] = (mol, groups, formula)
        except Exception:
            pass  # this ligand just won't be a usable candidate for any refcode

        for refcode in ast.literal_eval(row["spec_refcode"]):
            refcode_to_spec_ghs.setdefault(refcode, []).append(spec_gh)

        if progress_callback is not None and n_seen % progress_every == 0:
            progress_callback("parse", n_seen, len(lig), corpus)

    n_done = 0
    for base_refcode, info in complexes.items():
        n_done += 1
        corpus.n_complexes_seen += 1

        candidate_ghs = [gh for gh in refcode_to_spec_ghs.get(base_refcode, []) if gh in parsed]
        candidate_formulas = [parsed[gh][2] for gh in candidate_ghs]

        resolved = reconstruct_ligand_multiset(info["coordination_formula"], info["metal"], candidate_formulas)
        if resolved is None:
            if progress_callback is not None and n_done % progress_every == 0:
                progress_callback("build", n_done, len(complexes), corpus)
            continue

        try:
            r_metal = covalent_radius(info["metal"])
        except (KeyError, ValueError):
            if progress_callback is not None and n_done % progress_every == 0:
                progress_callback("build", n_done, len(complexes), corpus)
            continue

        total_bulk = 0
        cn = 0
        for complex_ligand in resolved:
            mol, groups, _ = parsed[candidate_ghs[complex_ligand.candidate_index]]
            for _ in range(complex_ligand.multiplicity):
                total_bulk += ligand_bulk(mol, groups, STERIC_RADIUS)
                cn += len(groups)

        corpus.by_cn.setdefault(cn, []).append(total_bulk / r_metal)
        corpus.n_complexes_used += 1

        if progress_callback is not None and n_done % progress_every == 0:
            progress_callback("build", n_done, len(complexes), corpus)

    for values in corpus.by_cn.values():
        values.sort()

    return corpus


__all__ = ["StericCorpus", "build_steric_corpus", "STERIC_RADIUS"]
