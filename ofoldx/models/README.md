# models

`ofoldx.models` holds the model families — the concrete architectures for structure
prediction and design generation — together with the `Auto*` factories and the typed input
and output contracts they all share, so downstream code imports every architecture from a
single namespace.

Each family follows a **single-model-file policy**: it is self-contained under its own
directory and assembles the shared blocks from [`ofoldx.modules`](../modules) in one
`model.py`. Duplicated _architecture_ across families is deliberate — a family never
inherits another family's model, and only genuinely cross-family code lives in `modules`.
A family directory carries its `config.py`, typed input/output, a `convert.py` that maps an
upstream checkpoint onto the OFoldX layout, `model.py`, a self-registering
`processor.py`, and (where present) its own `README.md`. Published artifacts declare their
task and loading entrypoint in their root `manifest.json`; family directories stay
model-first internally. See
[`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md) and [`DESIGN.md`](../../DESIGN.md) for
the full contract.

## Contents

- `alphafold2/`, `alphafold3/` — AlphaFold 2/3 structure prediction entry points.
- `boltz1/`, `boltz2/` — Boltz-1 and Boltz-2 (with affinity) structure prediction.
- `boltzgen/` — BoltzGen design generation.
- `esmfold2/` — Biohub ESMFold2 structure-prediction architecture.
- `odesign/`, `pxdesign/` — ODesign and PxDesign design generation.
- `protenix/` — Protenix structure prediction with optional constraint guidance.
- `proteinmpnn/`, `ligandmpnn/` — ProteinMPNN and LigandMPNN sequence design.
- `rfdiffusion/` — RFdiffusion backbone design generation.
- `auto.py`, `registry.py` — `Auto*` factories and the architecture registry.
- `contract.py`, `config.py`, `mixin.py` — typed IO contracts, shared config blocks, and the save/load mixin.
