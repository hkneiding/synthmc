"""Small, dependency-free helpers used by both corpus/ (building the
precedent corpus from BOS-Lig/BOS-TMC) and scoring/ (evaluating a
candidate against it) - and, for elements.py/graph_utils.py, by xyz/ too.
Nothing here knows about BOS-Lig, spec_mol2, or precedent lookups."""

from synthmc.common.donor_signature import donor_signature, site_signature
from synthmc.common.elements import TRANSITION_METALS, covalent_radius, is_transition_metal
from synthmc.common.environment import site_bulk, site_environment
from synthmc.common.fingerprint import morgan_fingerprint
from synthmc.common.graph_utils import group_by_mutual_bonding

__all__ = [
    "donor_signature",
    "site_signature",
    "site_environment",
    "site_bulk",
    "TRANSITION_METALS",
    "covalent_radius",
    "is_transition_metal",
    "morgan_fingerprint",
    "group_by_mutual_bonding",
]
