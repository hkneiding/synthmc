"""Reading multi-frame xyz files in tmQM's format.

Format (confirmed against BOSTMC/synthmc's tmQM_all.xyz):

    <natoms>
    CSD_code = WELROW | q = 0 | S = 0 | Stoichiometry = C40H36LaN2P3Se6 | MND = 8 | 2020-2024 CSD
    La        8.96863758139501    6.85196554201432    3.94646167692748
    ...
    <blank line>
    <natoms>
    ...

Frames are separated by a blank line. The comment line is a '|'-delimited
key/value string; fields of interest here are CSD_code, q (net charge) and
S (spin, always 0 in tmQM - see project memory).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, TextIO

import numpy as np

_COMMENT_RE = re.compile(
    r"CSD_code\s*=\s*(?P<code>\S+)\s*\|\s*q\s*=\s*(?P<q>-?\d+)\s*\|\s*S\s*=\s*(?P<s>\d+)"
)


@dataclass
class Structure:
    """One parsed xyz frame."""

    identifier: str
    symbols: list[str]
    coords: np.ndarray  # (natoms, 3)
    charge: int | None
    spin: int | None
    raw_comment: str

    @property
    def natoms(self) -> int:
        return len(self.symbols)


def _parse_frame(header_line: str, comment_line: str, atom_lines: list[str], line_no: int) -> Structure:
    try:
        natoms = int(header_line.strip())
    except ValueError as e:
        raise ValueError(f"Line {line_no}: expected atom count, got {header_line!r}") from e

    if len(atom_lines) != natoms:
        raise ValueError(
            f"Frame near line {line_no} (comment: {comment_line.strip()!r}): "
            f"header declares {natoms} atoms but {len(atom_lines)} atom lines were read"
        )

    symbols = []
    coords = np.empty((natoms, 3), dtype=float)
    for i, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Frame near line {line_no}: malformed atom line {line!r}")
        symbols.append(parts[0])
        coords[i] = [float(x) for x in parts[1:4]]

    m = _COMMENT_RE.search(comment_line)
    if m:
        identifier = m.group("code")
        charge = int(m.group("q"))
        spin = int(m.group("s"))
    else:
        identifier = comment_line.strip() or f"frame@line{line_no}"
        charge = None
        spin = None

    return Structure(
        identifier=identifier,
        symbols=symbols,
        coords=coords,
        charge=charge,
        spin=spin,
        raw_comment=comment_line.rstrip("\n"),
    )


def iter_structures(path_or_handle: str | TextIO, skip_errors: bool = True) -> Iterator[Structure]:
    """Stream Structure objects from a tmQM-format multi-frame xyz file.

    Memory-efficient: reads line by line rather than loading the whole file,
    since tmQM_all.xyz is ~500MB / 7.5M lines.

    If skip_errors is True, malformed frames are skipped (a warning is not
    printed per-frame to avoid flooding stdout over 100k+ entries; callers
    that need failure accounting should use iter_structures_with_errors).
    """
    for structure, error in iter_structures_with_errors(path_or_handle):
        if error is not None:
            if skip_errors:
                continue
            raise error
        yield structure


def iter_structures_with_errors(
    path_or_handle: str | TextIO,
) -> Iterator[tuple[Structure | None, Exception | None]]:
    """Like iter_structures, but yields (Structure, None) or (None, Exception) for every frame."""
    close_when_done = False
    if isinstance(path_or_handle, str):
        handle = open(path_or_handle, "r")
        close_when_done = True
    else:
        handle = path_or_handle

    try:
        line_no = 0
        line = handle.readline()
        line_no += 1
        while line:
            if line.strip() == "":
                line = handle.readline()
                line_no += 1
                continue

            header_line = line
            header_line_no = line_no
            comment_line = handle.readline()
            line_no += 1
            if not comment_line:
                yield None, ValueError(f"Line {header_line_no}: truncated file (missing comment line)")
                break

            try:
                natoms = int(header_line.strip())
            except ValueError:
                yield None, ValueError(f"Line {header_line_no}: expected atom count, got {header_line!r}")
                line = handle.readline()
                line_no += 1
                continue

            atom_lines = []
            ok = True
            for _ in range(natoms):
                atom_line = handle.readline()
                line_no += 1
                if not atom_line:
                    yield None, ValueError(f"Line {header_line_no}: truncated file mid-frame")
                    ok = False
                    break
                atom_lines.append(atom_line)

            if not ok:
                break

            try:
                yield _parse_frame(header_line, comment_line, atom_lines, header_line_no), None
            except Exception as e:  # noqa: BLE001 - deliberately broad, reported to caller
                yield None, e

            line = handle.readline()
            line_no += 1
    finally:
        if close_when_done:
            handle.close()


__all__ = ["Structure", "iter_structures", "iter_structures_with_errors"]
