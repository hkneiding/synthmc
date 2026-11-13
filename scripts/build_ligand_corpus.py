from __future__ import annotations

import time

from _common import ROOT, make_progress_printer

from synthmc.corpus import build_ligand_corpus  # noqa: E402


def main():
    """Build the 4-level ligand corpus and save it to ligand_corpus.pkl."""
    boslig_csv = ROOT / "BOSLig" / "Dataset" / "ligand_charge_dataset.csv"
    bostmc_dir = ROOT / "BOSTMC"
    out_path = ROOT / "ligand_corpus.pkl"

    t0 = time.time()
    progress = make_progress_printer(t0)

    print(f"Building ligand corpus from {boslig_csv} + {bostmc_dir} ...")
    corpus = build_ligand_corpus(boslig_csv, bostmc_dir, progress_every=5000, progress_callback=progress)

    def stats(buckets):
        """(n_sub_buckets, n_distinct_smiles, total_count) for one level."""
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
