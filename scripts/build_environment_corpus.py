from __future__ import annotations

import time

from _common import ROOT, make_progress_printer

from synthmc.corpus.environment_corpus import RADII, build_environment_corpus  # noqa: E402


def main():
    """Build the coordination-environment corpus and save it to
    environment_corpus.pkl."""
    boslig_csv = ROOT / "BOSLig" / "Dataset" / "ligand_charge_dataset.csv"
    bostmc_dir = ROOT / "BOSTMC"
    out_path = ROOT / "environment_corpus.pkl"

    t0 = time.time()
    progress = make_progress_printer(t0)

    print(f"Building environment corpus from {boslig_csv} + {bostmc_dir} ...")
    corpus = build_environment_corpus(boslig_csv, bostmc_dir, progress_every=20000, progress_callback=progress)

    print()
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"  complexes seen: {corpus.n_complexes_seen}")
    print(f"  complexes used (reconstructed cleanly): {corpus.n_complexes_used}"
          f" ({100*corpus.n_complexes_used/corpus.n_complexes_seen:.1f}%)")
    for r in RADII:
        n_keys = len(corpus.by_environment[r])
        total = sum(corpus.by_environment[r].values())
        n_metal_keys = len(corpus.by_environment_and_metal[r])
        metal_total = sum(corpus.by_environment_and_metal[r].values())
        print(f"  radius {r}: {n_keys} distinct joint environments ({total} total), "
              f"{n_metal_keys} (metal, environment) keys ({metal_total} total)")

    corpus.save(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
