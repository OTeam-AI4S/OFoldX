---
language: protein
tags:
  - biology
  - protein-design
  - side-chain-packing
  - protein-ligand
license: mit
library_name: ofoldx
pipeline_tag: other
---

# MpnnPacker

MpnnPacker rebuilds protein side-chain atoms on a fixed backbone and its nearby
non-protein atomic context. It predicts a von Mises mixture over the four
side-chain torsions per residue and samples atom14 coordinates with the released
iterative denoiser. This is an unofficial OFoldX reimplementation of the
[LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1) side-chain packer. The
upstream authors did not create, review, or endorse this implementation or model
card.

The packer is a separate OFoldX family rather than part of `LigandMpnn`. Upstream
distributes it as an independently loaded checkpoint that accepts sequences from
every official designer type, so it owns its own Model, Processor, and Pipeline and
is registered as `ofoldx.mpnn_packer`.

## Qualification

The architecture has a canonical `MpnnPackerProcessor`, a differentiable
`MpnnPackerModel.forward` returning `MpnnPackerOutput` (the mixture `mean`,
`concentration`, and `mix_logits`), and the released iterative `pack` / `sample`
path returning the shared `PackResult`. It has strict in-memory checkpoint
conversion of every upstream tensor into a standard repository. The maintained
[`scripts/parity/mpnn_packer.py`](../../../scripts/parity/mpnn_packer.py) reads the
official source from `~/reference` and the checkpoint from `~/weights`; generated
reports remain local.

## Inputs and outputs

`MpnnPackerProcessor` accepts a canonical `Structure` or a local PDB/mmCIF path. It
lowers backbone N/CA/C/O coordinates, the residue sequence tokens to pack, residue
and chain indices, observed atom37 coordinates, and the nearest 16 non-protein heavy
atoms as atomic context. A target `sequence` (string, or a per-chain mapping) may
replace the observed identities before packing, and `pack_positions` restricts packing to
selected `ComponentReference` residues. `use_atom_context` toggles the ligand
context and `filter_by_occupancy` drops unoccupied observed atoms.

`pack(...)` returns `PackResult`: a generation-ordered tuple of `PackPrediction`,
each carrying a complete packed `Structure` with realized atom14 side chains and its
per-sample metrics. `pack_sequences(...)` packs several target sequences against one
source `Structure` and returns one `PackResult` per sequence. The tensor
`MpnnPackerSamplingOutput` additionally exposes the sampled chi angles, their von
Mises log densities, and the mixture parameters for callers that need the raw
distribution.

## Usage

```python
from ofoldx import pipeline
from ofoldx.models.mpnn_packer.convert import convert_mpnn_packer_checkpoint

convert_mpnn_packer_checkpoint(
    "ligandmpnn_sc_v_32_002_16.pt",
    "mpnn_packer",
)

pack_pipeline = pipeline("pack", model="mpnn_packer", device="cuda")

packed = pack_pipeline.pack(
    "backbone.cif",
    sequence="MKTAYIAKQR...",
    num_samples=4,
    num_denoising_steps=3,
    num_torsion_samples=16,
    seed=7,
)
```

The ordinary tensor path remains available and differentiable:

```python
from ofoldx import AutoModel, AutoProcessor

model = AutoModel("mpnn_packer", device="cuda")
processor = AutoProcessor("mpnn_packer", device="cuda")
batch = processor("backbone.cif", sequence="MKTAYIAKQR...")
output = model(batch)
output.concentration.square().mean().backward()
```

The gradient example demonstrates flow; it is not the unpublished official training
objective. Training orchestration is outside the current Model Hub gate.

## Checkpoints and conversion

- Validated packer checkpoint: `ligandmpnn_sc_v_32_002_16.pt`
- Official checkpoint host: `https://files.ipd.uw.edu/pub/ligandmpnn`
- Official repository: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)
- Audited source revision: `26ec57ac976ade5379920dbd43c7f97a91cf82de`

The single released packer architecture uses `k_neighbors=32`, `atom_context_num=16`,
and a `num_mix=3` von Mises mixture. Conversion interprets every upstream tensor and
assigns every live target tensor; there is no loose loading. The `002` field is the
Gaussian coordinate-noise level used during upstream training and is weight
provenance, not a runtime Model configuration.

## Real-checkpoint parity

The maintained script compares all 215 converted tensors, the default
context-enabled Processor lowering, public `encode`/`decode`/`forward`, all eight
sampling fields, and canonical `PackResult` realization against the released
checkpoint. It exercises atom-context enabled and disabled, side-chain context, and
occupancy filtering, and it lists its intentionally uncovered diagnostic cases
explicitly. Real-weight parity runs outside frequent CI; repository round-trip
evidence is maintained separately and is not attributed to that comparison.

## Limitations & intentional upstream divergences

- The packer rebuilds side-chain atoms on a fixed backbone. It does not design
  sequence, generate backbone coordinates, or move backbone atoms; residue identity
  is an input.
- `pack` requires batch size one; `pack_sequences` batches several target sequences
  behind that constraint. Fixed, backbone-only, `ALA`, `GLY`, and unpacked residues
  pass their observed atoms through unchanged.
- The PDB writer intentionally diverges from upstream in the B-factor column: fixed,
  backbone, `ALA`, `GLY`, `OXT`, modified, and non-protein atoms are written as
  `0.0` rather than upstream fixed sentinels or unused density-head values. This is a
  deliberate truthful-output choice, verified as intentional in the parity script.
- The sampler realizes atom14 side chains from the von Mises mixture; per-sample
  atom coordinates carry the ordinary iterative-denoiser stochasticity and are not
  claimed bit-identical to any single upstream random draw. Tensor-level conversion
  and the mixture parameters are the parity surface.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`. The native OFoldX code is
  separate project code.
- **Upstream code & packer weights:** MIT
  ([LigandMPNN LICENSE](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE),
  Copyright (c) 2024 Justas Dauparas).
- **Redistribution status: GREEN.** The `ligandmpnn_sc_v_32_002_16` checkpoint is
  permissively licensed and freely redistributable provided the upstream MIT notice
  is preserved. Converted weights are not relicensed as AGPL; recheck the exact
  source revision before redistribution.

## Citation

```bibtex
@article{dauparas2025atomic,
  author = {Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and others},
  title = {Atomic context-conditioned protein sequence design using LigandMPNN},
  journal = {Nature Methods},
  year = {2025},
  doi = {10.1038/s41592-025-02626-1}
}
```
