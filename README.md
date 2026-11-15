# synthmc

[Precedent-based synthesizability scoring for transition metal complexes](https://doi.org/10.26434/chemrxiv.15012168/v1) (TMCs), built on the [BOS-TMC](https://doi.org/10.1021/acs.jcim.6c01792) and [BOS-Lig](https://doi.org/10.1021/acs.jcim.6c01063) datasets.

## Install

```bash
git clone https://github.com/hkneiding/synthmc/
cd tmc_sa
conda create -n synthmc python=3.11
conda activate synthmc
conda install -c conda-forge rdkit ase pandas
```

The precedent corpora (`ligand_corpus.pkl`, `environment_corpus.pkl`, `steric_corpus.pkl`) are not provided; build them yourself from the [BOS-TMC](https://zenodo.org/records/20384001) and [BOS-Lig](https://zenodo.org/records/19014026) dataset repositories:

```bash
python scripts/build_ligand_corpus.py
python scripts/build_environment_corpus.py
python scripts/build_steric_corpus.py
```

This expects `BOSTMC/` and `BOSLig/` data directories at the repo root. Each script writes its `.pkl` file to the repo root; expect the environment corpus build to take a few minutes.

## Quickstart: scoring Zeise's salt

Zeise's salt, `PtCl3(η2-C2H4)`, is a square-planar Pt(II) complex with three chloride ligands and one η2-bound ethylene.

```python
import pickle
from rdkit import Chem
from synthmc import score_complex

with open("ligand_corpus.pkl", "rb") as f:
    ligand_corpus = pickle.load(f)
with open("environment_corpus.pkl", "rb") as f:
    environment_corpus = pickle.load(f)
with open("steric_corpus.pkl", "rb") as f:
    steric_corpus = pickle.load(f)

# Metal provided with chemical symbol.
metal = "Pt"
# Ligands are (SMILES, binding_atoms) pairs, where binding_atoms is the
# list of that SMILES string's own atom indices binding to the metal.
# SMILES must be explicit-H (every hydrogen written out).
ligands = [
    ("[Cl-]", [0]),
    ("[Cl-]", [0]),
    ("[Cl-]", [0]),
    ("C(=C([H])[H])([H])[H]", [0, 1]),
]

score_complex(
    metal, 
    ligands,
    ligand_corpus,
    environment_corpus,
    steric_corpus,
    identifier="Zeise's salt"
).summary()
```

Output:

```
Synthesizability summary - metal: Pt (Zeise's salt)

[1] Ligand precedent: 0.999
      4 ligand(s)
      Cl1                  precedent_score=1.000
      Cl1                  precedent_score=1.000
      Cl1                  precedent_score=1.000
      C2H4                 precedent_score=0.998

[2] Coordination environment precedent: 0.787
      radius 0: any-metal=0.932    metal-specific=0.957
      radius 1: any-metal=0.608    metal-specific=0.650
      radius 2: any-metal=0.706    metal-specific=0.756
      radius 3: any-metal=0.769    metal-specific=0.830
      radius 4: any-metal=0.799    metal-specific=0.865

[3] Steric plausibility: 0.942

Overall score: 0.910
```

High on all three terms, as expected for a textbook, extremely well-precedented complex.
