---
language: protein
tags:
  - biology
  - protein-design
  - inverse-folding
  - protein-ligand
license: mit
library_name: ofoldx
pipeline_tag: other
---

# LigandMPNN

LigandMPNN designs protein sequences on a fixed backbone using nearby
non-protein atoms as structural context.

## Disclaimer

This is an unofficial OFoldX implementation of
[LigandMPNN](https://github.com/dauparas/LigandMPNN). The upstream authors did
not write, review, or endorse this implementation, conversion, or model card.

## Model Details

The `ofoldx.ligandmpnn` architecture pairs
`LigandMpnnModel` with `LigandMpnnProcessor`.
`LigandMpnnPipeline` exposes autoregressive `design` and the
four released probability workflows:
`autoregressive_probabilities`,
`sequence_free_autoregressive_probabilities`,
`conditional_probabilities`, and
`unconditional_probabilities`.

The Model exposes `encode`, differentiable `forward`, and
`sample`. `forward` returns `LigandMpnnOutput`;
`sample` returns `LigandMpnnSamplingOutput`. Both retain the
effective decoding ranks. Probability workflows return
`MpnnProbabilityResult`, while `design` returns
`DesignResult`. Direct per-residue `tied_weights` preserve
the released weighted-symmetry control.

The upstream side-chain packer is independently loaded and accepts sequences
from every official designer type. OFoldX therefore implements it as the
separate MPNN Packer family.
`LigandMpnnDesignAndPackPipeline` owns the concrete design-to-pack
composition.

### Variants

| Family member | Released checkpoint         | Role            |
| ------------- | --------------------------- | --------------- |
| LigandMPNN    | `ligandmpnn_v_32_005_25.pt` | sequence design |
| LigandMPNN    | `ligandmpnn_v_32_010_25.pt` | sequence design |
| LigandMPNN    | `ligandmpnn_v_32_020_25.pt` | sequence design |
| LigandMPNN    | `ligandmpnn_v_32_030_25.pt` | sequence design |

The four checkpoints share one executable architecture and differ by the
Gaussian coordinate-noise level used during upstream training. That label is
weight provenance, not a runtime Model configuration field. The
`ligandmpnn_sc_v_32_002_16.pt` side-chain checkpoint belongs to
MPNN Packer, not LigandMPNN.

### Model Specification

| Property            | Value                              |
| ------------------- | ---------------------------------- |
| Architecture        | `ofoldx.ligandmpnn`                |
| Model               | `LigandMpnnModel`                  |
| Processor           | `LigandMpnnProcessor`              |
| Pipeline            | `LigandMpnnPipeline`               |
| Task                | `design`                           |
| Design result       | `DesignResult`                     |
| Probability result  | `MpnnProbabilityResult`            |
| Hidden size         | 128                                |
| Encoder blocks      | 3 (`num_encoder_blocks`)           |
| Atom-context blocks | 2 (`atom_context_blocks`)          |
| Context blocks      | 2 (`context_blocks`)               |
| Decoder blocks      | 3 (`num_decoder_blocks`)           |
| Neighbors           | 32 (`num_neighbors`)               |
| Context atoms       | nearest 25 non-protein heavy atoms |

The Processor accepts a canonical `Structure` or a local PDB path.
Load mmCIF through the canonical structure reader and pass the resulting
`Structure`. It preserves complete molecular identity while lowering
protein N/CA/C/O coordinates, observed residue atom context, residue and chain
indices, and nearby non-protein heavy atoms. Declared water molecules are
excluded. `use_atom_context`, `ligand_interface_cutoff`, and
`filter_by_occupancy` are per-call controls.

Hard `AllowedResidueIdentities` and
`TiedResidueIdentities` conditions, global and per-residue logit bias,
weighted ties, coordinate noise, and optional side-chain context are supported.
Each design candidate carries a complete coordinate-free
`MolecularSystem` and the scores
`designed_sequence_confidence`,
`ligand_interface_confidence`, and
`sequence_recovery`.

Each probability workflow retains logits and exact decoding ranks over every
requested evaluation and exposes log probabilities, probabilities, their mean,
and population standard deviation. The released workflow publishes these
aligned distributions rather than an aggregate sequence NLL, so LigandMPNN has
no scalar `score` or `ScoreResult` operation.

### Links

- [Official source](https://github.com/dauparas/LigandMPNN)
- [Audited source revision](https://github.com/dauparas/LigandMPNN/tree/26ec57ac976ade5379920dbd43c7f97a91cf82de)
- [Official checkpoint host](https://files.ipd.uw.edu/pub/ligandmpnn)
- [LigandMPNN paper](https://doi.org/10.1038/s41592-025-02626-1)

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, pipeline

repository = "./ligandmpnn-v_32_010_25"
designer = pipeline("design", model=repository, device="cuda")

result = designer.design(
    "target.pdb",
    num_candidates=4,
    use_atom_context=True,
    ligand_interface_cutoff=8.0,
    seed=7,
)
probabilities = designer.autoregressive_probabilities(
    "target.pdb",
    num_decoding_orders=8,
    seed=7,
)

model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")
batch = processor("target.pdb")
output = model(batch)
sampled = model.sample(batch, num_samples=4)
```

The direct tensor path is differentiable. Direct stochastic Model operations
accept a caller-owned `torch.Generator`; Pipeline operations also
accept `seed`.

## Checkpoints and Conversion

Convert one acquired sequence checkpoint:

```bash
python -m ofoldx.models.ligandmpnn.convert \
  ligandmpnn_v_32_010_25.pt \
  ./ligandmpnn-v_32_010_25
```

The converter uses restricted PyTorch loading, derives the hidden size, block
counts, neighbor count, and Processor context extent, maps every checkpoint
tensor, and strict-loads the native Model. It writes `model.json`,
`processor.json`, safetensors weights, and the concrete
`pipeline.json` marker. `--repo-id` optionally uploads the
repository, and `--private` requests a private Hub repository.

The side-chain checkpoint is converted with the separate
`ofoldx.models.mpnn_packer.convert` module.

## Training Details

### Training Data

The inspected upstream repository is inference-oriented and does not publish a
complete reproducible training-data manifest for the four released sequence
checkpoints.

### Training Procedure

The four checkpoint suffixes record distinct Gaussian coordinate-noise levels
used during upstream training. The released inference source does not publish a
complete reference training objective and procedure. OFoldX preserves a
differentiable teacher-forced `forward` path but does not present a
gradient-flow example as the unpublished official objective. Training
orchestration belongs to the unified OFoldX Runner.

## Limitations

- LigandMPNN designs protein sequence on a fixed backbone with atomic context.
  It does not generate backbones or pack side chains.
- Side-chain packing belongs to MPNN Packer, and combined design-to-pack
  cardinality belongs to the concrete composition.
- LigandMPNN intentionally exposes aligned probability distributions rather
  than a scalar `ScoreResult`.
- Released-weight comparison tooling covers an earlier LigandMPNN surface.
  Real-weight parity is not established for the current Model, Processor,
  Pipeline, conversion, and design-to-pack composition.

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
- **Upstream source and checkpoints:** MIT under the
  [LigandMPNN license](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE),
  Copyright (c) 2024 Justas Dauparas.
- The four sequence checkpoints and the separate side-chain checkpoint may be
  redistributed with the upstream MIT notice preserved. Converted weights are
  not relicensed by OFoldX.
