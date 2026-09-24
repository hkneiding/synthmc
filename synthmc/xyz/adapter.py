"""Helper tooling: turn a raw xyz structure into the (metal, ligands) input
scoring/evaluate.py's core evaluate_complex actually wants, then call it.

This exists for the one real use case that doesn't already know its own
chemistry: evaluating existing structures from a database like tmQM (used
throughout this project's own development for testing against real
examples). It is not the primary interface - see scoring/evaluate.py's
module docstring for why detection (this module) and precedent lookup
(scoring/evaluate.py) are kept separate. Everything fragile about turning
geometry into chemistry (metal-donor distance calibration, hapticity
clustering, OpenBabel bond-order perception) lives here, isolated from the
scoring logic it feeds.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from synthmc.corpus.environment_corpus import RADII, EnvironmentCorpus
from synthmc.corpus.ligand_corpus import LigandCorpus
from synthmc.corpus.steric_corpus import StericCorpus
from synthmc.scoring.evaluate import LigandPrecedent, StructureResult, evaluate_complex
from synthmc.xyz.fragments import split_structure
from synthmc.xyz.io import iter_structures
from synthmc.xyz.ligand_chem import ligand_to_mol_with_binding_groups


def evaluate_xyz_structure(
    identifier: str,
    symbols: list[str],
    coords: np.ndarray,
    corpus: LigandCorpus,
    environment_corpus: EnvironmentCorpus | None = None,
    steric_corpus: StericCorpus | None = None,
) -> StructureResult:
    """Detect the metal and ligands in a raw xyz structure (fragments.py +
    ligand_chem.py), then evaluate them via the same core API a caller with
    already-known chemistry would use directly.

    Detection failures are handled here, not by evaluate_complex - it now
    raises on any ligand it can't parse (see its docstring), since a
    caller who already knows their own chemistry shouldn't get a silent
    partial result. This module is exactly the layer meant to tolerate and
    report per-fragment failures instead, so a fragment that fails
    detection never reaches evaluate_complex at all: it gets its own stub
    LigandPrecedent (all match fields None, .error set) spliced back into
    the result in its original position, and - since a joint environment or
    steric total is only meaningful if every ligand is actually known - a
    detection failure forces full_environment/full_environment_and_metal to
    all-None and steric to None for the whole structure, the same way
    evaluate_complex itself used to before this split."""
    split = split_structure(identifier, symbols, coords)

    detected_ligands: list[tuple[str, list[int]]] = []
    atom_indices: list[list[int]] = []
    detection_errors: list[str | None] = []

    for fragment in split.ligands:
        atom_indices.append(fragment.atom_indices)
        try:
            lig_mol, binding_groups = ligand_to_mol_with_binding_groups(fragment)
            binding_atoms = [i for group in binding_groups for i in group]
            detected_ligands.append((lig_mol.smiles, binding_atoms))
            detection_errors.append(None)
        except Exception as e:  # noqa: BLE001 - reported per-fragment; this layer is meant to tolerate it
            detection_errors.append(str(e))

    any_detection_failed = any(e is not None for e in detection_errors)

    result = evaluate_complex(
        split.metal_symbol,
        detected_ligands,
        corpus,
        identifier=identifier,
        # a detection failure means the joint environment/steric totals would be computed
        # from an incomplete ligand set - not worth computing at all, so skip asking for them
        environment_corpus=None if any_detection_failed else environment_corpus,
        steric_corpus=None if any_detection_failed else steric_corpus,
    )

    if any_detection_failed and environment_corpus is not None:
        # restore the "tried but couldn't" signal (all-None) rather than the "never asked"
        # signal (empty dict) evaluate_complex produces when environment_corpus is None
        result.full_environment = {r: None for r in RADII}
        result.full_environment_and_metal = {r: None for r in RADII}

    # Splice detected ligands' results back into original fragment order, with a stub
    # LigandPrecedent (all None) for each fragment that failed detection.
    merged: list[LigandPrecedent] = []
    detected_iter = iter(result.ligands)
    for i, detection_error in enumerate(detection_errors):
        if detection_error is None:
            lig = next(detected_iter)
        else:
            lig = LigandPrecedent(
                formula="?",
                smiles=None,
                donor_signature=None,
                by_ligand=None,
                by_ligand_and_metal=None,
                by_ligand_and_mode=None,
                by_ligand_mode_and_metal=None,
                error=detection_error,
            )
        lig.atom_indices = atom_indices[i]
        merged.append(lig)
    result.ligands = merged

    return result


def evaluate_xyz_file(
    path: str,
    corpus: LigandCorpus,
    skip_errors: bool = True,
    environment_corpus: EnvironmentCorpus | None = None,
    steric_corpus: StericCorpus | None = None,
) -> Iterator[StructureResult]:
    """Evaluate every structure in a tmQM-format multi-frame xyz file."""
    for s in iter_structures(path, skip_errors=skip_errors):
        try:
            yield evaluate_xyz_structure(
                s.identifier, s.symbols, s.coords, corpus, environment_corpus=environment_corpus, steric_corpus=steric_corpus
            )
        except Exception:
            if skip_errors:
                continue
            raise


__all__ = ["evaluate_xyz_structure", "evaluate_xyz_file"]
