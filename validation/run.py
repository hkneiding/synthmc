"""End-to-end held-out validation: split, build held-out corpora, query,
report. Supersedes the earlier ad hoc scripts/holdout_environment_test.py -
this covers all three precedent terms (ligand, coordination environment,
steric) against one shared held-out split, rather than just the environment
corpus alone.

Usage:
    python validation/run.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from validation.corpora import build_holdout_corpora  # noqa: E402
from validation.evaluate import print_report, run_holdout_validation  # noqa: E402
from validation.reconstruct import reconstruct_complexes  # noqa: E402
from validation.split import make_holdout_split  # noqa: E402

HOLDOUT_SIZE = 15287  # ~10% of BOS-TMC's 152,874 unambiguous single-component refcodes
SEED = 0


def main():
    boslig_csv = ROOT / "BOSLig" / "Dataset" / "ligand_charge_dataset.csv"
    bostmc_dir = ROOT / "BOSTMC"

    t0 = time.time()

    print(f"Making held-out split (size={HOLDOUT_SIZE}, seed={SEED}) ...")
    holdout_refcodes = make_holdout_split(bostmc_dir, holdout_size=HOLDOUT_SIZE, seed=SEED)

    print("Reconstructing held-out complexes from BOS-Lig (no xyz/OpenBabel involved) ...")
    reconstructed = reconstruct_complexes(boslig_csv, bostmc_dir, holdout_refcodes)
    print(f"  {len(reconstructed)}/{len(holdout_refcodes)} resolved cleanly")

    def progress(stage, n_seen, n_total, corpus):
        elapsed = time.time() - t0
        print(f"  [{stage}] {n_seen:>7d}/{n_total} [{elapsed:.0f}s]", flush=True)

    print("Building held-out corpora (ligand, environment, steric) ...")
    corpora = build_holdout_corpora(
        boslig_csv, bostmc_dir, holdout_refcodes, reconstructed, progress_callback=progress
    )
    print(f"  done in {time.time()-t0:.0f}s")
    print()

    result = run_holdout_validation(reconstructed, corpora, n_holdout=len(holdout_refcodes))
    print_report(result)

    out_path = ROOT / "validation_result.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved full result (including raw per-complex values) to {out_path}")


if __name__ == "__main__":
    main()
