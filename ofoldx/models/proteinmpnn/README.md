---
language: protein
tags:
  - biology
  - protein-design
  - inverse-folding
license: mit
library_name: ofoldx
pipeline_tag: other
---

# ProteinMPNN

ProteinMPNN is OFoldX's implementation of the ProteinMPNN message-passing
inverse-folding family for amino-acid sequence design on a fixed protein
backbone.

## Disclaimer

This is an unofficial OFoldX implementation of
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN). The upstream authors did
not write, review, or endorse this implementation, conversion, or model card.

## Model Details

A converted repository contains a `ProteinMpnnModel` and
`ProteinMpnnProcessor` pair under the architecture ID
`ofoldx.proteinmpnn`. `ProteinMpnnPipeline` exposes:

- `design`, returning `ProteinMpnnDesignResult`;
- `score`, returning `ScoreResult`; and
- `autoregressive_probabilities`,
  `sequence_free_autoregressive_probabilities`,
  `conditional_probabilities`, and
  `unconditional_probabilities`, each returning
  `MpnnProbabilityResult`.

The Model also exposes `encode`, differentiable `forward`, and
`sample`. `forward` returns `ProteinMpnnOutput` with
raw per-residue logits and the effective decoding ranks.
`sample` returns `ProteinMpnnSamplingOutput` with sampled
tokens, in-pass log probabilities, sampling probabilities, and decoding ranks.
`ProteinMpnnDesignAndPackPipeline` composes sequence design with the
independent MPNN Packer family.

### Variants

The inspected release contains twelve checkpoint files:

| Graph         | Collection               | Checkpoints                                                |
| ------------- | ------------------------ | ---------------------------------------------------------- |
| full backbone | vanilla                  | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` |
| full backbone | soluble                  | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` |
| CA-only       | CA                       | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`                |
| full backbone | tracked training example | `training/exp_020/model_weights/epoch_last.pt`             |

All use one public Model, Processor, and Result contract. `ca_only`
selects the released coordinate and feature graph. Vanilla and soluble training
and the filename noise label are weight provenance, not additional architecture
IDs.

### Model Specification

| Property          | Value                    |
| ----------------- | ------------------------ |
| Architecture      | `ofoldx.proteinmpnn`     |
| Model             | `ProteinMpnnModel`       |
| Processor         | `ProteinMpnnProcessor`   |
| Pipeline          | `ProteinMpnnPipeline`    |
| Tasks             | `design`, `score`        |
| Hidden size       | 128                      |
| Encoder blocks    | 3 (`num_encoder_blocks`) |
| Decoder blocks    | 3 (`num_decoder_blocks`) |
| Neighbors         | 48 (`num_neighbors`)     |
| Dropout           | 0.1                      |
| Coordinate graphs | full N/CA/C/O or CA-only |

`ProteinMpnnProcessor` accepts a canonical `Structure` or a
local PDB path. Its path arm follows the released fixed-column reader: it consumes
`ATOM` records and `HETATM` MSE, keeps the first duplicate
atom, retains residue numbers and insertion codes, and ignores occupancy. Load
mmCIF through the canonical `Structure` API before passing it to the
Processor.

Full-backbone rows require finite observed N/CA/C/O coordinates; CA-only rows
require finite observed CA coordinates. `design_positions` uses
canonical `(instance_id, component_index)` references. The Processor
also accepts `AllowedResidueIdentities`,
`TiedResidueIdentities`, tied weights, per-residue logit bias, and
PSSM coefficients, probabilities, and log odds. Sampling additionally supports
global bias and omissions, PSSM mixing and thresholding, temperature, and
inference-time `backbone_noise` in angstroms.

Tied members share one causal rank and are sampled simultaneously; fixed rows
precede designed rows. Explicit `torch.long` ranks reproduce one causal
order. Each design candidate contains a complete coordinate-free
`MolecularSystem` and the scores `designed_sequence_nll`,
`global_sequence_nll`, and `sequence_recovery`.
`result.input_score` contains the repeated native-sequence score from
the same invocation. Ranking always names a score explicitly, for example
`result.best("designed_sequence_nll")`.

### Links

- [Official source](https://github.com/dauparas/ProteinMPNN)
- [Audited source revision](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57)
- [ProteinMPNN paper](https://doi.org/10.1126/science.add2187)

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, pipeline

repository = "./proteinmpnn-v_48_020"
designer = pipeline("design", model=repository, device="cuda")

result = designer.design(
    "target.pdb",
    num_candidates=4,
    num_decoding_orders=8,
    temperature=0.1,
    seed=7,
)
native_score = designer.score("target.pdb", num_decoding_orders=8, seed=7)
probabilities = designer.autoregressive_probabilities(
    "target.pdb",
    num_decoding_orders=8,
    seed=7,
)

model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")
batch = processor("target.pdb")
output = model(batch)
sampled = model.sample(batch, num_samples=4, temperature=0.2)
```

Direct Model stochastic operations accept a caller-owned
`torch.Generator`; Pipeline operations additionally accept `seed`.
OFoldX follows explicit device and dtype choices, with mixed precision controlled
by the caller's dtype or AMP policy.

## Checkpoints and Conversion

Convert an acquired checkpoint explicitly:

```bash
python -m ofoldx.models.proteinmpnn.convert \
  v_48_020.pt \
  ./proteinmpnn-v_48_020
```

The converter uses restricted PyTorch loading, derives the block counts,
neighbor count, hidden size, and `ca_only` graph from the checkpoint,
maps every live tensor, and strict-loads the native Model. It writes
`model.json`, `processor.json`, safetensors weights, and the
concrete `pipeline.json` marker. `--repo-id` optionally
uploads that repository, and `--private` requests a private Hub
repository.

Released CA checkpoints contain five tensors whose outputs are never consumed
by the upstream graph. Conversion drops exactly those five dead tensors and no
others.

## Training Details

### Training Data

The inspected release distinguishes vanilla and soluble checkpoint collections
and includes one tracked training example. It does not provide a complete
per-checkpoint training-data manifest from which every released weight can be
reproduced.

### Training Procedure

The checkpoint filename noise suffix records upstream training provenance; it is
not a runtime default or serialized Model configuration field.
`ProteinMpnnModel.forward` is differentiable, and OFoldX exposes the
family loss primitives, but OFoldX does not claim to reproduce the complete
upstream training pipeline. Training orchestration belongs to the unified OFoldX
Runner.

## Limitations

- ProteinMPNN designs and scores sequences on fixed backbones. It does not
  generate backbones or pack side-chain coordinates.
- Non-protein entities remain in the canonical system but are not Model sequence
  rows.
- The `global_label_membrane_mpnn_v_48_020` and
  `per_residue_label_membrane_mpnn_v_48_020` checkpoints require a
  distinct membrane-conditioned graph and are not ProteinMPNN variants.
- Existing released-weight comparison tooling covers an earlier ProteinMPNN
  implementation surface. Real-weight parity is not established for the current
  Model, Processor, Pipeline, and conversion behavior.
- Confirmed upstream tied-group defects are implemented according to the declared
  mathematics; details are recorded in
  [`UPSTREAM_BUGS.md`](../../../UPSTREAM_BUGS.md).

## Citation

```bibtex
@article{dauparas2022robust,
  author = {Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and others},
  title = {Robust deep learning-based protein sequence design using ProteinMPNN},
  journal = {Science},
  volume = {378},
  number = {6615},
  pages = {49--56},
  year = {2022},
  doi = {10.1126/science.add2187}
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and checkpoints:** MIT under the
  [ProteinMPNN license](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/LICENSE),
  Copyright (c) 2022 Justas Dauparas.
- The vanilla, soluble, CA-only, and membrane checkpoints may be redistributed
  with the upstream MIT notice preserved. Converted weights are not relicensed
  by OFoldX.
