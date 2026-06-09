# data

`ofoldx.data` holds the chemical and reference data plus the dataset wrapper that
feed structures into the models — CCD component lookups, residue-geometry constants,
covalent-bond and chirality checks, rigid alignment, and the in-memory structure objects
the featurizers read.

Much of this is a faithful port of upstream AlphaFold3 / OpenFold reference data and
scoring code, kept as tested, reusable APIs rather than inline steps. The production CCD
provider is RDKit-backed and reads the prebaked Mol pickle so reference coordinates stay
byte-exact with the checkpoint conversion path; tests inject a small in-memory provider
instead, so the committed suite needs no external data. See [`DESIGN.md`](../../DESIGN.md)
and [`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md) for the full contract.

## Contents

- `structure.py` — the flat in-memory `Structure`, `AtomRecord`, and `ResidueKey` objects.
- `ccd.py`, `rdkit_ccd.py` — Chemical Component Dictionary lookup (`CcdProvider`, `ReferenceComponent`) and the RDKit CIF/pickle providers that build reference molecules.
- `residue_constants.py` — vendored AlphaFold2 amino-acid geometry tables for side-chain torsion frames.
- `bonds.py` — covalent-bond length validation and intra-/inter-chain bond classification.
- `chirality.py` — ligand chirality comparison against reference molecules (inversion detection).
- `alignment.py` — Kabsch superposition and aligned RMSD for scoring predictions against a reference.
- `dataset.py` — `StructureDataset`, the PyTorch dataset that reads structure files and returns processor outputs.
