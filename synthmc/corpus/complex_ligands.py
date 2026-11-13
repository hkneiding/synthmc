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
    """Match each formula term to one candidate; None on an ambiguous or
    missing match."""
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
