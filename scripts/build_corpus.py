"""Build the full BOS-TMC x BOS-Lig precedent corpus and save it to disk.

Usage:
    python scripts/build_corpus.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthmc.corpus import build_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    bostmc_dir = ROOT / "BOSTMC"
    boslig_csv = ROOT / "BOSLig" / "Dataset" / "ligand_charge_dataset.csv"
    out_path = ROOT / "precedent_corpus.pkl"

    t0 = time.time()

    def progress(stage, n_seen, n_total, corpus):
        elapsed = time.time() - t0
        rate = n_seen / elapsed if elapsed > 0 else 0
        print(f"  [{stage}] {n_seen:>7d}/{n_total} [{elapsed:.0f}s, {rate:.0f}/s]", flush=True)
        corpus.save(str(out_path) + ".partial")

    print(f"Building corpus from {bostmc_dir} + {boslig_csv} ...")
    corpus = build_corpus(bostmc_dir, boslig_csv, progress_every=5000, progress_callback=progress)

    print()
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"  BOS-TMC unambiguous complexes: {corpus.n_bostmc_complexes}")
    print(f"  Joined with BOS-Lig ligand data: {corpus.n_joined_complexes}")
    for level, counter in corpus.level_counts.items():
        print(f"  level={level}: {len(counter)} distinct keys, total={sum(counter.values())}")
    print(f"  fragment_counts (ligand-level, from BOS-Lig gen_gh/gen_count): {len(corpus.fragment_counts)} distinct")

    corpus.save(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
