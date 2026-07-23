---
language:
  - protein
  - rna
  - dna
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# Boltz-1

Boltz-1 is OFoldX's all-atom structure-prediction implementation for
biomolecular complexes, with a recycled representation trunk, EDM coordinate
sampling, confidence scoring, pocket conditioning, and optional Boltz-1x
sampling-time steering.

## Disclaimer

This is an unofficial OFoldX conversion of the released Boltz-1 inference
checkpoint, `boltz1_conf.ckpt`. The upstream authors did not write, review, or
endorse this conversion or model card.

OFoldX converts the released weights into its native parameter layout. It does
not train, relicense, or claim ownership of the upstream model. The inspected
OFoldX checkpoint metadata does not record an upstream paper, DOI, or complete
author list, so this card does not invent one.

## Model Details

`Boltz1Model` embeds token and atom features with atom attention, refines single
and pair representations through an optional MSA module and a recycled
Pairformer trunk, predicts a distogram, and samples atom coordinates with an EDM
reverse-diffusion process. A trunk-imitating confidence head predicts pLDDT,
PAE, PDE, resolved probability, pTM-derived scores, and the released ranking
score.

`Boltz1Pipeline` owns the concrete `fold` operation. `Boltz1Processor` lowers a
`MolecularSystem`, `Structure`, or structure path, optional MSA evidence, and an
optional `Boltz1Pocket` into `Boltz1Input`. `Boltz1Model.fold` returns a canonical
`FoldResult`; `Boltz1Model.representations` separately returns the raw trunk
single and pair tensors without placing them in `FoldResult`.

### Variants

| Architecture    | Released checkpoint | Model / Processor                 | Task   | Concrete Pipeline |
| --------------- | ------------------- | --------------------------------- | ------ | ----------------- |
| `ofoldx.boltz1` | `boltz1_conf.ckpt`  | `Boltz1Model` / `Boltz1Processor` | `fold` | `Boltz1Pipeline`  |

The one released checkpoint contains both structure and confidence weights.
Boltz-1x physical-potential and Feynman-Kac steering are call-time sampling
controls on this checkpoint, not another architecture or checkpoint variant.

### Model Specification

| Property                               | Native default                     |
| -------------------------------------- | ---------------------------------- |
| Single representation                  | 384                                |
| Pair representation                    | 128                                |
| Trunk Pairformer blocks                | 48 (`num_pairformer_blocks`)       |
| MSA blocks                             | 4 (`num_msa_blocks`)               |
| Input atom-attention blocks            | 3                                  |
| Denoiser atom encoder / decoder blocks | 3 / 3                              |
| Denoiser token Transformer blocks      | 24                                 |
| Denoiser conditioning Transitions      | 2 (`num_conditioning_transitions`) |
| Confidence Pairformer blocks           | 48                                 |
| Total trunk passes                     | 4 (`recycling_steps`)              |
| Reverse-diffusion steps                | 200 (`num_sampling_steps`)         |
| Default fold samples                   | 1                                  |

The names above follow the native OFoldX graph: repeated units are Blocks,
stored under `blocks` and counted by fields ending in `num_blocks`; channel MLP
units are Transitions. Source checkpoint spellings are handled only by the
converter.

The EDM defaults are `sigma_data=16`, `rho=7`, noise levels from `160` to
`4e-4` in `sigma_data` units, `churn_rate=0.8`,
`min_churn_noise_level=1`, `noise_scale=1`, and `step_scale=1`. Sampling uses an
Euler update with optional weighted rigid alignment and a trailing zero-noise
step.

`Boltz1Pocket(binder, pocket)` names one binder `instance_id` and zero or more
canonical pocket residue references. The Processor emits the released four-way
token feature (`UNSPECIFIED`, `UNSELECTED`, `POCKET`, `BINDER`). The 6 angstrom
contact meaning comes from training and is not recomputed at fold time;
omitting the pocket leaves every token `UNSPECIFIED`.

Each prediction publishes token pLDDT on the 0-100 scale, resolved probability,
token-pair PAE/PDE in angstroms, and the scores `plddt`, `iplddt`, `pde`, `ipde`,
`ptm`, `iptm`, `ligand_iptm`, `protein_iptm`, and `confidence_score` when the
corresponding head is present. `chain_pair_iptm` is an ordered, unsymmetrized
chain-by-chain `PairMetric`. On the internal 0-1 pLDDT fraction, ranking uses

```text
interface = iptm if iptm is nonzero else ptm
confidence_score = (4 * complex_plddt + interface) / 5
```

`representations` returns `single` with shape `[B, N, 384]` and `pair` with shape
`[B, N, N, 128]`, retaining autograd when called under grad.

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz1)
- [Official Boltz source](https://github.com/jwohlwend/boltz)
- Model type: diffusion-based all-atom structure predictor with a Pairformer
  trunk and confidence/ranking head

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, fold, pipeline
from ofoldx.models.boltz1 import Boltz1Pocket

repository = "./boltz1"

result = fold(
    "complex.cif",
    model=repository,
    device="cuda",
    num_samples=5,
    seed=7,
)

folder = pipeline("fold", model=repository, device="cuda")
conditioned = folder.fold(
    "complex.cif",
    pocket=Boltz1Pocket(binder="B", pocket=(("A", 41),)),
    num_samples=5,
    seed=7,
)

model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")
batch = processor("complex.cif")
direct_result = model.fold(batch)
trunk = model.representations(batch)
```

`Boltz1Pipeline.fold` additionally accepts `recycling_steps`,
`num_sampling_steps`, `use_potentials`, and `steering_args`. Pipeline calls
accept either `seed` or a caller-owned `torch.Generator`; direct Model calls
accept the generator.

## Checkpoints and Conversion

Convert the released checkpoint explicitly:

```bash
python -m ofoldx.models.boltz1.convert \
  boltz1_conf.ckpt \
  ./boltz1
```

The converter translates the released parameter namespaces into native
`blocks` and Transition names, separates invariant denoiser conditioning from
the learned coordinate denoiser, packs adaptive-normalization projections, and
strict-loads the complete native graph. The output repository contains
`model.json` and safetensors weights; the Processor is config-less. `--dtype`
may cast floating tensors, and `--repo-id` optionally uploads the converted
repository.

## Training Details

### Training Data

The training corpus for `boltz1_conf.ckpt` is not recorded in the inspected
OFoldX checkpoint metadata. OFoldX does not redistribute or reconstruct the
upstream training data.

### Training Procedure

The released model uses an EDM coordinate-denoising objective together with a
trunk distogram objective and a confidence/ranking objective. Training noise is
drawn as
`sigma = sigma_data * exp(-1.2 + 1.5 * z)`, where `z ~ N(0, 1)`, and the
coordinate loss weight is
`(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`.

The complete upstream optimizer, schedule, batching recipe, and loss reduction
details are not present in the inspected OFoldX metadata. OFoldX conversion does
not retrain the model or claim to reproduce that training loop.

## Limitations

- Reverse diffusion is stochastic; one sampled coordinate set is not a
  deterministic statement of model quality.
- Only `boltz1_conf.ckpt` is represented. Pocket and steering behavior is
  limited to the controls exposed by `Boltz1Pipeline` and the released graph.
- Empty confidence reductions use the dtype-aware
  `torch.finfo(dtype).tiny` denominator rather than a fixed cross-family
  normalization constant.
- Recorded released-checkpoint comparisons cover the Processor, training
  forward, confidence head, and sampler (4/4), with observed maximum
  differences between 0 and `8.6e-5`. That evidence is specific to the compared
  checkpoint, inputs, dtype, and sampling controls; it does not establish parity
  for every pocket or steering configuration.
- The inspected metadata does not provide enough upstream training or citation
  provenance to reconstruct either without additional primary evidence.

## Citation

The inspected checkpoint metadata does not identify an upstream publication to
cite. If OFoldX supports your research, cite the OFoldX software:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and checkpoint:** MIT under the
  [Boltz license](https://github.com/jwohlwend/boltz/blob/main/LICENSE),
  Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, and Saro Passaro.
- `boltz1_conf.ckpt` may be redistributed with the upstream MIT notice
  preserved. Converted weights are not relicensed by OFoldX.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
