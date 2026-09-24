"""Held-out validation of the three precedent corpora (ligand, coordination
environment, steric) against real, unseen BOS-TMC complexes - measuring how
well-precedented genuinely novel complexes turn out to be, isolated from any
xyz-detection noise (queries here come straight from BOS-Lig's own curated
mol2 data, the same clean path corpus/donor_sites.py offers, not from
OpenBabel/geometry).

    split.py       - make_holdout_split: pick a set of refcodes to hold out,
                      shared across all three corpora so "unseen" means the
                      same thing for each one.
    reconstruct.py - reconstruct_complexes: turn held-out refcodes into
                      query-ready (metal, ligands) complexes via BOS-Lig,
                      with zero xyz/OpenBabel involvement; also derives the
                      exact per-ligand-structure exclusion counts
                      corpus/ligand_corpus.py's build_ligand_corpus needs
                      (its levels 1/3 use pre-aggregated totals that can't
                      be excluded by refcode alone - see its docstring).
    corpora.py     - build_holdout_corpora: build all three corpora with the
                      held-out refcodes excluded.
    evaluate.py    - run_holdout_validation: query every held-out complex
                      against the held-out corpora and tally hit rates /
                      plausibility distributions.
    run.py         - ties the above together end to end; run directly for a
                      full report (python validation/run.py).
"""
