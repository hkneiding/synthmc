"""Build the steric-plausibility precedent corpus from BOS-Lig + BOS-TMC
and save it to disk. See corpus/steric_corpus.py's module docstring for
what this captures.

Usage:
    python scripts/build_steric_corpus.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthmc.corpus.steric_corpus import build_steric_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    boslig_csv = ROOT / "BOSLig" / "Dataset" / "ligand_charge_dataset.csv"
    bostmc_dir = ROOT / "BOSTMC"
    out_path = ROOT / "steric_corpus.pkl"

    t0 = time.time()

    def progress(stage, n_seen, n_total, corpus):
        elapsed = time.time() - t0
        rate = n_seen / elapsed if elapsed > 0 else 0
        print(f"  [{stage}] {n_seen:>7d}/{n_total} [{elapsed:.0f}s, {rate:.0f}/s]", flush=True)

    print(f"Building steric corpus from {boslig_csv} + {bostmc_dir} ...")
    corpus = build_steric_corpus(boslig_csv, bostmc_dir, progress_every=20000, progress_callback=progress)

    print()
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"  complexes seen: {corpus.n_complexes_seen}")
    print(f"  complexes used (reconstructed cleanly): {corpus.n_complexes_used}"
          f" ({100*corpus.n_complexes_used/corpus.n_complexes_seen:.1f}%)")
    print(f"  distinct coordination numbers: {sorted(corpus.by_cn.keys())}")
    for cn in sorted(corpus.by_cn.keys()):
        values = corpus.by_cn[cn]
        print(f"  CN {cn}: {len(values)} complexes, normalized bulk min={values[0]:.1f} "
              f"median={values[len(values)//2]:.1f} max={values[-1]:.1f}")

    corpus.save(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
