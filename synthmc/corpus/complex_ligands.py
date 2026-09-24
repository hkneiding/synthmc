"""Reconstruct one BOS-TMC complex's complete ligand multiset (every
ligand present, and how many times) by matching BOS-TMC's own
`coordination formula` field against BOS-Lig's candidate structures for
that refcode.

This exists because BOS-Lig's spec_refcode lists are deduplicated *sets*
of refcodes a ligand appears in, not per-occurrence multisets - a ligand
that occurs twice in one complex (any homoleptic or bis/tris-chelate
structure - not a rare case) only gets credited once if you try to
reconstruct occurrence counts by inverting that list directly. Confirmed
this empirically: only ~32% of complexes' donor-atom counts matched
between BOS-TMC's own histogram and a naive BOS-Lig reconstruction.

`coordination formula` (e.g. "Ir (BC29H29N)2 C5H7O2") sidesteps that
entirely - it already gives exact per-ligand-formula multiplicities
directly from BOS-TMC, verified self-consistent with BOS-TMC's own
`natoms` field on a 3000-complex random sample (zero mismatches). The only
remaining job is matching each formula term to *which* BOS-Lig structure
it refers to, by comparing molecular formulas. Measured on an 8000-complex
sample: 97.6% match every term to exactly one BOS-Lig candidate
(unambiguous); 0.1% have a formula shared by more than one real BOS-Lig
structure (a genuine structural-isomer ambiguity); 2.4% have a term with
no BOS-Lig candidate at all (a coverage gap). Both of the latter are
dropped rather than guessed at - same "narrow accepted gap over a forced
guess" principle used elsewhere in this project (e.g. bostmc.py's
multi-component exclusion).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_TERM_RE = re.compile(r"\(([A-Za-z0-9]+)\)(\d+)|(\S+)")
_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_molecular_formula(formula: str) -> Counter:
    """"C29H29BN" -> Counter({'C': 29, 'H': 29, 'B': 1, 'N': 1})."""
    counts: Counter = Counter()
    for symbol, n in _ELEM_RE.findall(formula):
        if symbol:
            counts[symbol] += int(n) if n else 1
    return counts


def parse_coordination_formula(coordination_formula: str, metal: str) -> list[tuple[str, int]]:
    """"Ir (BC29H29N)2 C5H7O2" -> [("BC29H29N", 2), ("C5H7O2", 1)] - the
    metal's own term is excluded."""
    terms = []
    for m in _TERM_RE.finditer(coordination_formula):
        if m.group(1):
            terms.append((m.group(1), int(m.group(2))))
        else:
            fragment = m.group(3)
            if fragment != metal:
                terms.append((fragment, 1))
    return terms


@dataclass
class ComplexLigand:
    candidate_index: int  # index into the `candidates` list passed to reconstruct_ligand_multiset
    multiplicity: int


def reconstruct_ligand_multiset(
    coordination_formula: str,
    metal: str,
    candidate_formulas: list[Counter],
) -> list[ComplexLigand] | None:
    """Match each coordination-formula term to exactly one entry in
    `candidate_formulas` (BOS-Lig structures known to occur somewhere in
    this refcode, as parsed element-count Counters). Returns None if any
    term matches zero or more than one candidate - see module docstring."""
    terms = parse_coordination_formula(coordination_formula, metal)

    result = []
    for formula, multiplicity in terms:
        target = parse_molecular_formula(formula)
        matches = [i for i, c in enumerate(candidate_formulas) if c == target]
        if len(matches) != 1:
            return None
        result.append(ComplexLigand(candidate_index=matches[0], multiplicity=multiplicity))
    return result


__all__ = ["ComplexLigand", "parse_molecular_formula", "parse_coordination_formula", "reconstruct_ligand_multiset"]
