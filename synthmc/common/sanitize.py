"""Shared sanitization-fallback constant for a genuine, understood gap in
BOS-Lig's source data: every gen_mol2/spec_mol2 block sets all formal
charges to zero ("ZeroCharges" in its own header), even for fragments that
chemically should carry one - most commonly an odd-membered all-carbon
aromatic ring (Cp-type organometallic fragments, e.g. cyclopentadienyl),
which has no valid Kekule structure as a neutral, closed-shell molecule.
RDKit's default full sanitization requires kekulization and fails outright
on these; skipping just that step (and the aromaticity re-perception that
depends on it) lets the mol through with its declared aromaticity kept
as-is, rather than recomputed - not a chemically pristine fix, but a
faithful reflection of a real, documented gap in the source data rather
than a bug in this project's own parsing.

Three call sites need this same fallback, each on a Chem.KekulizeException
specifically (never a broader exception, so a genuinely malformed structure
still surfaces as a real error): corpus/mol2.py (parsing a mol2 block
directly), common/environment.py (radius-truncated fragments, which can cut
a ring open - plus its own further last-resort fallback for a rarer valence
issue), and scoring/evaluate.py (re-parsing a canonical SMILES that already
went through one of the other two - the same Cp-type structure they
tolerated on the way in has to be tolerated again on the way back out,
since a plain Chem.MolFromSmiles with default sanitization has no such
leniency)."""

from __future__ import annotations

from rdkit import Chem

SANITIZE_SANS_KEKULIZATION = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY

# Second, last-resort fallback: also skip valence/property checking. Found a
# real (if rare) source-data case this needs: e.g. an O atom with formal
# charge -2 bonded via an RDKit "unspecified" bond type (an unusual
# phosphazene-type fragment), or a carbene carbon in an unusual NHC pincer
# ligand with an "unspecified" (~) bond to an anionic N - valid in the
# original whole molecule's own sanitize context, but which a freshly-
# sanitized fragment/SMILES can reject on valence grounds alone, independent
# of kekulization. Accepts the structure as-is rather than raising - still a
# valid, comparable canonical string for exact-match precedent-counting
# purposes even if not a chemically pristine independent molecule.
SANITIZE_MINIMAL = SANITIZE_SANS_KEKULIZATION ^ Chem.SANITIZE_PROPERTIES

__all__ = ["SANITIZE_SANS_KEKULIZATION", "SANITIZE_MINIMAL"]
