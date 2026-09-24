"""Build the 4-level ligand precedent corpus from BOS-Lig + BOS-TMC and
save it to disk. See ligand_corpus.py's module docstring for what the 4
levels mean.

Usage:
    python scripts/build_ligand_corpus.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthmc.corpus import build_ligand_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    boslig_csv = ROOT / "BOSLig" / "Dataset" / "ligand_charge_dataset.csv"
    bostmc_dir = ROOT / "BOSTMC"
    out_path = ROOT / "ligand_corpus.pkl"

    t0 = time.time()

    def progress(stage, n_seen, n_total, corpus):
        elapsed = time.time() - t0
        rate = n_seen / elapsed if elapsed > 0 else 0
        print(f"  [{stage}] {n_seen:>7d}/{n_total} [{elapsed:.0f}s, {rate:.0f}/s]", flush=True)

    print(f"Building ligand corpus from {boslig_csv} + {bostmc_dir} ...")
    corpus = build_ligand_corpus(boslig_csv, bostmc_dir, progress_every=5000, progress_callback=progress)

    def stats(buckets):
        if isinstance(buckets, dict):
            n_smiles = sum(len(b.counts) for b in buckets.values())
            total = sum(sum(b.counts.values()) for b in buckets.values())
            return len(buckets), n_smiles, total
        return 1, len(buckets.counts), sum(buckets.counts.values())

    n1, smi1, tot1 = stats(corpus.by_ligand)
    n2, smi2, tot2 = stats(corpus.by_ligand_and_metal)
    n3, smi3, tot3 = stats(corpus.by_ligand_and_mode)
    n4, smi4, tot4 = stats(corpus.by_ligand_mode_and_metal)

    print()
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"  gen_gh rows seen: {corpus.n_gen_seen}  (parse failures: {corpus.n_gen_parse_failures})")
    print(f"  spec_gh rows seen: {corpus.n_spec_seen}  (parse failures: {corpus.n_spec_parse_failures})")
    print()
    print(f"  1. by_ligand:               {smi1} distinct SMILES, {tot1} total count")
    print(f"  2. by_ligand_and_metal:     {n2} metals, {smi2} (metal, SMILES) entries, {tot2} total count")
    print(f"  3. by_ligand_and_mode:      {n3} donor signatures, {smi3} (signature, SMILES) entries, {tot3} total count")
    print(f"  4. by_ligand_mode_and_metal: {n4} (metal, signature) buckets, {smi4} entries, {tot4} total count")

    corpus.save(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
