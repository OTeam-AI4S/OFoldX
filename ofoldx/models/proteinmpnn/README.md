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

# ProteinMpnn

ProteinMpnn is OFoldX's implementation of the ProteinMPNN message-passing
inverse-folding model for designing amino-acid
sequences on a fixed protein backbone. OFoldX provides an independent implementation
of the released model family. The upstream authors did not create, review, or endorse
this conversion or model card.

## OFoldX API

One converted repository contains the paired `ProteinMpnnModel` and
`ProteinMpnnProcessor`, identified by `ofoldx.proteinmpnn`. The paired
`ProteinMpnnPipeline` exposes the complete biological operation surface:

- `design(...) -> ProteinMpnnDesignResult`;
- `score(...) -> ScoreResult`;
- `autoregressive_probabilities(...) -> MpnnProbabilityResult`;
- `sequence_free_autoregressive_probabilities(...) -> MpnnProbabilityResult`;
- `conditional_probabilities(...) -> MpnnProbabilityResult`; and
- `unconditional_probabilities(...) -> MpnnProbabilityResult`.

The Model additionally exposes the tensor-level `encode`, `forward`, and `sample`
operations alongside the four probability methods. `forward` returns `MpnnOutput`
with raw per-residue logits and the effective decoding ranks. One `sample` resolves
tied and untied inputs on a single causal path and returns `MpnnSamplingOutput` with
sampled tokens, raw in-pass log probabilities, sampling probabilities, and decoding
ranks.

Load a converted local or Hub repository with the paired Auto factories:

```python
from ofoldx import AutoModel, AutoProcessor, pipeline

repository = "./proteinmpnn-v_48_020"
model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")

batch = processor("target.pdb")
log_probabilities = model(batch, seed=7)
score = model.score(batch, num_decoding_orders=8, seed=7)
result = model.design(
    batch,
    num_candidates=4,
    num_decoding_orders=8,
    temperature=0.1,
    seed=7,
)

designer = pipeline("design", model=repository, device="cuda")
same_public_operation = designer.design("target.pdb", num_candidates=4, seed=7)
```

`result.candidates` is generation ordered. Each candidate contains a complete
coordinate-free `MolecularSystem` and the scores `designed_sequence_nll`,
`global_sequence_nll`, and `sequence_recovery`. `result.input_score` contains the
native repeated input-sequence score produced by the same invocation. Use
`result.best("designed_sequence_nll")` or another explicit score name when ranking
candidates; there is no implicit best-score choice.

## Inputs and controls

`ProteinMpnnProcessor` accepts a canonical `Structure` or a local PDB path. The PDB
path follows the audited ProteinMPNN reader contract rather than the general OFoldX
PDBx reader: it consumes `ATOM` records and `HETATM` MSE, keeps the first duplicate
atom, retains source residue numbering and insertion codes, and ignores occupancy.
Pass mmCIF through the canonical `Structure` API instead of passing an mmCIF path to
this family Processor.

Full-backbone mode requires finite observed N/CA/C/O coordinates. CA-only mode
requires finite observed CA coordinates. `design_positions` selects canonical
`(instance_id, component_index)` references. `sequence` supplies an alternate target
for scoring. The Processor also accepts the family-owned controls
`AllowedResidueIdentities`, `TiedResidueIdentities`, tied weights, per-residue logit
bias, and PSSM coefficients, probabilities, and log odds.

Sampling controls remain ordinary Model arguments:

```python
sampled = model.sample(
    batch,
    num_samples=4,
    temperature=0.2,
    backbone_noise=0.05,
    global_logit_bias=global_bias,
    global_omit_token_mask=global_omit,
    pssm_mixture=0.4,
    pssm_log_odds_threshold=0.0,
    seed=7,
)
```

One `sample` resolves both tied and untied Processor inputs. Generated decoding
ranks place fixed rows before designed rows; tied members share one rank and are
sampled simultaneously. A caller may pass explicit `torch.long` ranks to reproduce
one exact causal order. `model.design` samples from the processed input and realizes
the sampled identities while preserving instance and resolvable bond identity.

`backbone_noise` is an inference-time coordinate perturbation in angstroms. The
checkpoint filename's noise suffix describes training provenance and is not a
runtime default or a serialized configuration field. OFoldX follows explicit user
device and dtype choices; mixed precision remains the caller's AMP or dtype policy.

## Released checkpoints

The behavior audit used source revision
[`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57).
The complete executable inventory at that revision is:

| Graph | Collection | Checkpoints |
| --- | --- | --- |
| full backbone | vanilla | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` |
| full backbone | soluble | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt` |
| CA-only | CA | `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt` |
| full backbone | tracked training example | `training/exp_020/model_weights/epoch_last.pt` |

All twelve use the same public Model, Processor, and Result contracts. `ca_only` is
a Model and Processor configuration value because it selects the released coordinate
and feature graph. Vanilla versus soluble training and the noise label are weight
provenance, not additional architecture IDs.

The released CA checkpoints contain five tensors whose outputs are never consumed
by the upstream graph. Conversion drops exactly those five tensors and no others;
the native CA graph does not preserve a dead node-projection branch.

## Conversion

OFoldX does not fetch checkpoints during conversion. With the local source checkout
under `~/reference` and weights under `~/weights`, convert one checkpoint explicitly:

```bash
python -m ofoldx.models.proteinmpnn.convert \
  ~/weights/mpnn-weights/proteinmpnn_v_48_020.pt \
  ./proteinmpnn-v_48_020
```

The converter reads the released checkpoint with restricted PyTorch loading, maps
the complete live state dictionary, strict-loads the native Model, and writes:

```text
model.json
processor.json
model.safetensors          # or its standard shard index and shards
pipeline.json
```

`model.json` records `ofoldx.proteinmpnn` and the concrete Model config;
`processor.json` records only the paired Processor config. There is no schema,
contract version, digest, fingerprint, or duplicated checkpoint-variant field.
Repository history and reproducibility belong to the selected Hub revision.

## Manual parity

The on-demand comparison is
[`scripts/parity/proteinmpnn.py`](../../../scripts/parity/proteinmpnn.py):

```bash
python -m scripts.parity.proteinmpnn --device cpu
```

It reads the caller-supplied upstream checkout from `~/reference/ProteinMPNN` and the full and
soluble checkpoints from `~/weights/mpnn-weights`. An ordinary run checks conversion
and explicit-rank forward computation for all twelve checkpoints, then compares the
released probability, scoring, sampling, and Processor paths on representative full
and CA graphs. Corrected tied-group behavior is checked against its declared native
mathematics because the inspected upstream implementation contains confirmed defects.
Those defects and their evidence are recorded in
[`UPSTREAM_BUGS.md`](../../../UPSTREAM_BUGS.md).

This is a manual scientific diagnostic. It is not collected by pytest, does not run
in frequent CI, writes no validation report, and is not a registration or publication
gate.

## Scope

ProteinMpnn performs fixed-backbone sequence design and sequence scoring. It does
not generate backbones or pack side-chain coordinates. Non-protein entities remain
part of the canonical system but are not Model sequence rows. Training belongs to
OFoldX's unified Runner rather than a family-specific upstream training wrapper.

OFoldX code and documentation are distributed under `AGPL-3.0-or-later`. The audited
upstream ProteinMPNN repository publishes its source and checkpoint files under its
[MIT License](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/LICENSE).
Preserve the upstream notice when redistributing upstream or converted checkpoint
material. Converted weights are not relicensed by OFoldX.

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

Upstream source: [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN).

## Limitations & intentional upstream divergences

- ProteinMpnn performs fixed-backbone sequence design and scoring; it does not
  generate backbones or pack side-chain coordinates.
- **MembraneMpnn is deferred:** the `global_label_membrane_mpnn_v_48_020` and
  `per_residue_label_membrane_mpnn_v_48_020` checkpoints are present in the inventory
  but are **not** a released variant this session. The soluble and vanilla
  `*_v_48_*` variants are the qualified surface.
- A parity run applies only to the checkpoints and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** MIT
  ([ProteinMPNN LICENSE](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/LICENSE),
  Copyright (c) 2022 Justas Dauparas).
- **Redistribution status: GREEN.** The vanilla and soluble checkpoints are
  permissively licensed and freely redistributable with the upstream MIT notice
  preserved; the deferred membrane checkpoints share the same MIT terms.
