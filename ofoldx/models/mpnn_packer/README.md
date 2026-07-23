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

# MPNN Packer

MPNN Packer rebuilds protein side-chain atoms on a fixed backbone using nearby
non-protein atoms as structural context.

## Disclaimer

This is an unofficial OFoldX implementation of the side-chain packer released
with [LigandMPNN](https://github.com/dauparas/LigandMPNN). The upstream authors
did not write, review, or endorse this implementation, conversion, or model
card.

## Model Details

The side-chain packer is a separate `ofoldx.mpnn_packer` family rather
than part of LigandMPNN. Upstream distributes it as an independently loaded
checkpoint that accepts sequences from every official designer type, so it owns
its own `MpnnPackerModel`, `MpnnPackerProcessor`, and
`MpnnPackerPipeline`.

The Model predicts a three-component von Mises mixture over four side-chain chi
torsions per residue. `encode` and `decode` expose the two
tensor stages. Differentiable `forward` consumes an explicit atom14
state and returns `MpnnPackerOutput` with `mean`,
`concentration`, and `mix_logits`. `sample` runs
the released iterative denoiser and returns
`MpnnPackerSamplingOutput` with atom14 coordinates, chi angles and log
densities, mixture parameters, and the input chi representation.

`pack` returns a generation-ordered `PackResult` whose
`PackPrediction` values contain complete packed structures and
per-chi log-density metrics. `pack_sequences` packs several target
sequences against one source structure and returns one `PackResult`
per sequence.

### Variants

The released family contains one checkpoint,
`ligandmpnn_sc_v_32_002_16.pt`. The `002` field records the
Gaussian coordinate-noise level used during upstream training; it is weight
provenance, not a runtime Model configuration field.

### Model Specification

| Property             | Value                              |
| -------------------- | ---------------------------------- |
| Architecture         | `ofoldx.mpnn_packer`               |
| Model                | `MpnnPackerModel`                  |
| Processor            | `MpnnPackerProcessor`              |
| Pipeline             | `MpnnPackerPipeline`               |
| Task                 | `pack`                             |
| Result               | `PackResult`                       |
| Hidden and edge size | 128                                |
| Encoder blocks       | 3 (`num_encoder_blocks`)           |
| Atom-context blocks  | 2 (`atom_context_blocks`)          |
| Context blocks       | 2 (`context_blocks`)               |
| Decoder blocks       | 3 (`num_decoder_blocks`)           |
| Neighbors            | 32 (`num_neighbors`)               |
| Context atoms        | nearest 16 non-protein heavy atoms |
| Mixture components   | 3 (`num_mix`)                      |

The Processor accepts a canonical `Structure` or a local PDB path.
Load mmCIF through the canonical structure reader and pass the resulting
`Structure`. It lowers backbone N/CA/C/O coordinates, target sequence
tokens, residue and chain indices, observed atom37 coordinates, selected packing
positions, and nearby atomic context. A target sequence string or per-chain
mapping may replace observed identities. `use_atom_context` toggles
ligand context, and `filter_by_occupancy` excludes unoccupied observed
atoms.

### Links

- [Official source](https://github.com/dauparas/LigandMPNN)
- [Audited source revision](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de)
- [Official checkpoint host](https://files.ipd.uw.edu/pub/ligandmpnn)
- [LigandMPNN paper](https://doi.org/10.1038/s41592-025-02626-1)

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, pipeline

repository = "./mpnn-packer"
packer = pipeline("pack", model=repository, device="cuda")

packed = packer.pack(
    "backbone.pdb",
    sequence="MKTAYIAKQR...",
    num_samples=4,
    num_denoising_steps=3,
    num_torsion_samples=16,
    seed=7,
)

many = packer.pack_sequences(
    "backbone.pdb",
    sequences=("MKTAYIAKQR...", "MKTAYIAKQA..."),
    num_samples=2,
    seed=7,
)

model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")
batch = processor("backbone.pdb", sequence="MKTAYIAKQR...")
raw = model.sample(batch, num_denoising_steps=3, num_torsion_samples=16)
```

Direct `forward` additionally requires explicit
`atom14_coordinates` and `atom14_mask`. Direct stochastic
Model operations accept a caller-owned `torch.Generator`; Pipeline
operations also accept `seed`.

## Checkpoints and Conversion

Convert the acquired checkpoint:

```bash
python -m ofoldx.models.mpnn_packer.convert \
  ligandmpnn_sc_v_32_002_16.pt \
  ./mpnn-packer
```

The converter uses restricted PyTorch loading, reads the 32-neighbor and
16-context-atom metadata, maps all 215 released tensors, and strict-loads every
live target tensor. It writes `model.json`,
`processor.json`, safetensors weights, and the concrete
`pipeline.json` marker. `--repo-id` optionally uploads the
repository, and `--private` requests a private Hub repository.

## Training Details

### Training Data

The inspected inference release does not publish a complete reproducible
training-data manifest for the side-chain checkpoint.

### Training Procedure

The checkpoint name records a Gaussian coordinate-noise level of
`002`. The upstream release does not publish the complete official
packer objective and training procedure. OFoldX exposes a differentiable
`forward` over an explicit atom14 state but does not infer an
unpublished loss. Training orchestration belongs to the unified OFoldX Runner.

## Limitations

- MPNN Packer rebuilds side chains on a fixed backbone. It does not design
  sequence, generate backbone coordinates, or move backbone atoms.
- `pack` requires batch size one. `pack_sequences` batches
  several target sequences while retaining one canonical realization mapping
  per sequence.
- Fixed, backbone-only, ALA, GLY, and unselected residues preserve their
  observed atoms.
- The OFoldX PDB writer writes `0.0` in the B-factor column for fixed,
  backbone, ALA, GLY, OXT, modified, and non-protein atoms instead of upstream
  sentinels or unused density-head values.
- Atom coordinates retain ordinary iterative-denoiser stochasticity and are not
  claimed bit-identical to one upstream random draw.
- Real-weight parity has been established for standalone `pack` against the
  released checkpoint, including canonical result realization and PDB writing.
  `pack_sequences` has not yet been parity-verified.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and packer checkpoint:** MIT under the
  [LigandMPNN license](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE),
  Copyright (c) 2024 Justas Dauparas.
- The `ligandmpnn_sc_v_32_002_16.pt` checkpoint may be redistributed
  with the upstream MIT notice preserved. Converted weights are not relicensed
  by OFoldX.
