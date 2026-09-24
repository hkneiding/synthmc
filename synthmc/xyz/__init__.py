"""Helper tooling: turn a raw xyz structure into the (metal, ligands) input
scoring.evaluate_complex actually wants, then call it - not the primary
interface (see adapter.py's module docstring). Everything fragile about
turning geometry into chemistry (metal-donor distance calibration,
hapticity clustering, OpenBabel bond-order perception) lives in this
subpackage, isolated from the scoring logic it feeds."""

from synthmc.xyz.adapter import evaluate_xyz_file, evaluate_xyz_structure
from synthmc.xyz.fragments import LigandFragment, SplitStructure, split_structure
from synthmc.xyz.io import Structure, iter_structures

__all__ = [
    "evaluate_xyz_structure",
    "evaluate_xyz_file",
    "split_structure",
    "LigandFragment",
    "SplitStructure",
    "Structure",
    "iter_structures",
]
