"""Reconstruct held-out refcodes into query-ready complexes, straight from
BOS-Lig's own curated mol2 data - the same xyz-free path
corpus/donor_sites.py offers, not corpus/environment_corpus.py's internal
build loop (which only tallies aggregate stats and never needs to keep a
per-complex (SMILES, binding_atoms) list around).

Also derives the exact spec_gh/gen_gh occurrence counts to subtract from
corpus/ligand_corpus.py's pre-aggregated gen_count/spec_count totals for a
proper held-out ligand corpus - see build_ligand_corpus's docstring for why
that can't be done from the refcode set alone the way the other two corpora
manage it. Only refcodes that resolve cleanly need an entry here: one that
fails to resolve (ambiguous/missing candidate) is never queried either, so
leaving its contribution in the ligand corpus causes no leakage.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from synthmc.corpus.bostmc import load_complexes
from synthmc.corpus.complex_ligands import reconstruct_ligand_multiset
from synthmc.corpus.donor_sites import _strip_metal_dummy, spec_mol2_to_smiles_and_donor_groups
from synthmc.corpus.mol2 import parse_mol2

RDLogger.DisableLog("rdApp.*")


@dataclass
class ReconstructedComplex:
    refcode: str
    metal: str
    ligands: list[tuple[str, list[int]]]  # (canonical SMILES, flat binding-atom positions) - ready for evaluate_complex
    instances: list[tuple[str, int]]  # (spec_gh, multiplicity) - the exact resolution behind `ligands`


def reconstruct_complexes(
    boslig_csv: str | Path, bostmc_dir: str | Path, refcodes: set[str]
) -> dict[str, ReconstructedComplex]:
    """For each of `refcodes` (a subset of BOS-TMC's own complexes),
    reconstruct its exact ligand multiset the same way
    corpus/environment_corpus.py and corpus/steric_corpus.py do internally,
    except this keeps what they discard: the resolved (spec_gh,
    multiplicity) pairs and a clean (SMILES, binding_atoms) ligand list.
    Refcodes that don't resolve cleanly (~2.4% project-wide - ambiguous or
    missing BOS-Lig candidate, see complex_ligands.py) are simply absent
    from the returned dict."""
    complexes = load_complexes(bostmc_dir)
    lig = pd.read_csv(boslig_csv)

    refcode_to_rows: dict[str, list[int]] = {}
    for idx, row in lig.iterrows():
        for refcode in ast.literal_eval(row["spec_refcode"]):
            if refcode in refcodes:
                refcode_to_rows.setdefault(refcode, []).append(idx)

    result: dict[str, ReconstructedComplex] = {}
    for refcode in refcodes:
        info = complexes.get(refcode)
        if info is None:
            continue

        candidate_ghs = []
        candidate_formulas = []
        candidate_mol2 = []
        for idx in refcode_to_rows.get(refcode, []):
            spec_mol2 = lig.at[idx, "spec_mol2"]
            try:
                trimmed, _ = _strip_metal_dummy(spec_mol2)
                mol = parse_mol2(trimmed)
            except Exception:
                continue
            candidate_ghs.append(lig.at[idx, "spec_gh"])
            candidate_formulas.append(Counter(atom.GetSymbol() for atom in mol.GetAtoms()))
            candidate_mol2.append(spec_mol2)

        resolved = reconstruct_ligand_multiset(info["coordination_formula"], info["metal"], candidate_formulas)
        if resolved is None:
            continue

        ligands: list[tuple[str, list[int]]] = []
        instances: list[tuple[str, int]] = []
        for complex_ligand in resolved:
            spec_gh = candidate_ghs[complex_ligand.candidate_index]
            mol2_text = candidate_mol2[complex_ligand.candidate_index]
            smiles, groups = spec_mol2_to_smiles_and_donor_groups(mol2_text)
            binding_atoms = [i for group in groups for i in group]
            instances.append((spec_gh, complex_ligand.multiplicity))
            for _ in range(complex_ligand.multiplicity):
                ligands.append((smiles, binding_atoms))

        result[refcode] = ReconstructedComplex(
            refcode=refcode, metal=info["metal"], ligands=ligands, instances=instances
        )

    return result


def ligand_corpus_exclusions(
    reconstructed: dict[str, ReconstructedComplex], boslig_csv: str | Path
) -> tuple[dict[str, int], dict[str, int]]:
    """(exclude_spec_counts, exclude_gen_counts) for
    corpus/ligand_corpus.py's build_ligand_corpus - spec_gh/gen_gh ->
    exact total occurrence count to subtract, summed across every held-out
    complex's resolved instances."""
    exclude_spec: Counter = Counter()
    for rc in reconstructed.values():
        for spec_gh, multiplicity in rc.instances:
            exclude_spec[spec_gh] += multiplicity

    spec_gh_to_gen_gh = pd.read_csv(boslig_csv, usecols=["spec_gh", "gen_gh"]).set_index("spec_gh")["gen_gh"].to_dict()
    exclude_gen: Counter = Counter()
    for spec_gh, amount in exclude_spec.items():
        gen_gh = spec_gh_to_gen_gh.get(spec_gh)
        if gen_gh is not None:
            exclude_gen[gen_gh] += amount

    return dict(exclude_spec), dict(exclude_gen)


__all__ = ["ReconstructedComplex", "reconstruct_complexes", "ligand_corpus_exclusions"]
