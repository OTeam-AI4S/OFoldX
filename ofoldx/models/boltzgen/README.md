---
language: protein
tags:
  - Biology
  - Protein
  - Protein Design
  - Inverse Folding
license: agpl-3.0
datasets:
  - boltzgen/boltzgen1_train
  - boltzgen/inference-data
base_model: boltz-community/boltz-2
library_name: ofoldx
pipeline_tag: other
---

# BoltzGen

BoltzGen is OFoldX's structure-based biomolecular design family for universal
binder design, controllable protein structure generation, and inverse folding.
Its concrete Pipeline combines a structure-diffusion designer with a distinct
autoregressive inverse-folding model and can optionally compose compatible fold
and affinity pairs.

## Disclaimer

This is an unofficial OFoldX conversion of
[BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494)
by Hannes Stark, Felix Faltings, MinGyu Choi, et al. The upstream authors did not
write, review, or endorse this conversion or model card.

The official repository is
[HannesStark/boltzgen](https://github.com/HannesStark/boltzgen). OFoldX converts
released weights into native parameter layouts; it does not train, relicense, or
claim ownership of the upstream models.

## Model Details

`BoltzGenModel` is the coordinate-generating structure designer. It embeds atom,
token, MSA, binding, design-mask, secondary-structure, template, token-distance,
and contact features; refines single and pair representations with a recycled
MSA/Pairformer trunk; and samples atom coordinates with EDM reverse diffusion.
The reduced structure-trained checkpoint uses Miniformer Blocks and omits the
newer token-distance and design-conditioning branches.

`BoltzGenInverseFoldingModel` encodes a fixed backbone as a k-nearest-neighbor
token graph and autoregressively predicts residue identities under design-mask,
amino-acid, and symmetry constraints. The released structure-design checkpoints
do not contain confidence or affinity heads.

`BoltzGenPipeline` owns the concrete `design` task. It runs structure design,
decodes each sampled backbone, inverse-folds the design positions, writes the
designed residue identities into the candidate `Structure`, and optionally runs
composed fold and affinity pairs. It returns `BoltzGenDesignResult` containing
generation-ordered `BoltzGenDesignCandidate` values. Its `inverse_fold` method
also exposes the inverse-folding stage on a fixed backbone.

### Variants

| Released checkpoint                     | Architecture            | Native graph                          | Pipeline role             |
| --------------------------------------- | ----------------------- | ------------------------------------- | ------------------------- |
| `boltzgen1_diverse.ckpt`                | `ofoldx.boltzgen`       | full structure designer               | required root design pair |
| `boltzgen1_adherence.ckpt`              | `ofoldx.boltzgen`       | full structure designer               | required root design pair |
| `boltzgen1_structuretrained_small.ckpt` | `ofoldx.boltzgen`       | reduced Miniformer structure designer | required root design pair |
| `boltzgen1_ifold.ckpt`                  | `ofoldx.boltzgen_ifold` | inverse-folding graph                 | required `ifold/` pair    |

The three structure checkpoints are parameter/config variants of one public
`BoltzGenModel` architecture. The inverse-folding checkpoint has a different
input and graph and therefore uses its own architecture ID. Neither pair exposes
a standard single-pair task; `ofoldx.boltzgen` routes `design` through the
multi-pair `BoltzGenPipeline`.

### Model Specification

Structure-design variants:

| Property                                    | diverse / adherence         | structure-trained small     |
| ------------------------------------------- | --------------------------- | --------------------------- |
| Single / pair size                          | 384 / 128                   | 384 / 128                   |
| Trunk Blocks                                | 64 Pairformer Blocks        | 12 Miniformer Blocks        |
| MSA Blocks                                  | 4                           | 3 Miniformer Blocks         |
| Template Blocks                             | 2                           | 2 Miniformer Blocks         |
| Token-distance Blocks                       | 2                           | disabled                    |
| Denoiser token Transformer Blocks           | 24                          | 8                           |
| Denoiser conditioning Transitions           | 2                           | 2                           |
| `recycling_steps`                           | 3 (four total trunk passes) | 3 (four total trunk passes) |
| Reverse-diffusion steps                     | 200                         | 200                         |
| Binding/design/secondary-structure channels | enabled                     | disabled                    |

Inverse-folding variant:

| Property                      | Native default           |
| ----------------------------- | ------------------------ |
| Node / edge / hidden size     | 128 / 128 / 128          |
| Graph encoder Blocks          | 6 (`num_encoder_blocks`) |
| Graph decoder Blocks          | 3 (`num_decoder_blocks`) |
| Attention heads               | 4                        |
| Neighbors                     | 30 (`num_neighbors`)     |
| Atom encoder Blocks           | 1                        |
| Inverse-fold coordinate noise | 0.2                      |
| Dropout / softmax dropout     | 0.1 / 0.2                |

The names above follow the native OFoldX graph: repeated units are Blocks,
stored under `blocks` and counted by fields ending in `num_blocks`; channel MLP
units are Transitions. Source checkpoint names remain conversion-only.

Structure sampling defaults to a dilated Karras schedule with
`sigma_data=16`, `rho=7`, noise levels from `160` to `4e-4`,
`churn_rate=0.8`, `noise_scale=0.95`, `step_scale=1.8`, and 200 steps. The
normalized interval `[0.6, 0.8]` is dilated by `2.667`. The released stochastic
churn calculation is preserved exactly:

```text
noise_var = noise_scale^2 * (sigma_hat^2 - sigma^2)
x_noisy = x + noise_scale * sqrt(max(noise_var, 0)) * epsilon
```

`BoltzGenProcessor` supports an existing context, a fully generated assembly, or
an extension of an existing system. `total_len` is the whole-assembly token
budget; `BoltzGenPlacements` can declare generated chains and their placement
identity. Supported structure-design condition records include
`BoltzGenDesignMask`, `BoltzGenBinding`, `BoltzGenSecondaryStructure`,
`BoltzGenStructureGroup`, directional `BoltzGenContact`,
`BoltzGenCovalentBond`, and `BoltzGenSymmetry`. Inverse folding separately
accepts design positions, `BoltzGenResidueConstraint` values, and symmetric
groups.

Each candidate carries a complete, coordinate-producing `Structure`, optional
fold/affinity scores, and a per-residue inverse-fold NLL metric in nats.
`BoltzGenDesignResult.best(score)` ranks by one explicitly named score and keeps
the earliest candidate on a tie.

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltzgen)
- [Official source](https://github.com/HannesStark/boltzgen)
- [BoltzGen paper](https://doi.org/10.1101/2025.11.20.689494)
- [Boltz-2 base model](https://huggingface.co/boltz-community/boltz-2)
- Developed by Hannes Stark, Felix Faltings, MinGyu Choi, et al.

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, design, pipeline

repository = "./boltzgen-diverse"

result = design(
    None,
    model=repository,
    device="cuda",
    total_len=120,
    num_candidates=4,
    seed=7,
)

designer = pipeline("design", model=repository, device="cuda")
inverse_folded = designer.inverse_fold(
    "backbone.cif",
    design=(("A", 10), ("A", 11), ("A", 12)),
    num_sequences=4,
)

model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")
batch = processor(None, total_len=120)
coordinate_output = model.design(batch, num_samples=4)
```

The converted composite contains the structure-design and inverse-folding pairs.
Compatible fold and affinity Model/Processor pairs are optional constructor
dependencies of `BoltzGenPipeline`; their presence, rather than a mode flag,
controls whether those stages run.

## Checkpoints and Conversion

Convert one structure-design variant together with the released inverse-folding
checkpoint:

```bash
python -m ofoldx.models.boltzgen.convert \
  boltzgen1_diverse.ckpt \
  boltzgen1_ifold.ckpt \
  ./boltzgen-diverse
```

Use `boltzgen1_adherence.ckpt` or
`boltzgen1_structuretrained_small.ckpt` as the first argument for those variants.
The converter reconstructs the structure graph from checkpoint
hyperparameters, auto-detects the Miniformer graph, translates source parameter
names into native Block and Transition names, packs adaptive-normalization
projections, and strict-loads both Models. The output repository stores the
structure pair at its root, the inverse-fold pair under `ifold/`, and the
`BoltzGenPipeline` marker. Optional fold and affinity pairs are not copied into
this default conversion output.

Released Lightning `.ckpt` files pickle upstream BoltzGen objects, so direct path
conversion requires an environment that can deserialize the upstream `boltzgen`
and `pydssp` classes. A dependency-free workflow may export a plain state dict or
safetensors with the upstream environment and call the state-dict converters.
`--dtype` may cast floating tensors, and `--repo-id` optionally uploads the
converted repository.

## Training Details

### Training Data

BoltzGen builds on the Boltz-2 data pipeline. The public training release
contains `targets.zip` and `msa.zip` in `boltzgen/boltzgen1_train`, with molecule
dictionaries in `boltzgen/inference-data`. The full large-model recipe may use
additional distillation data. `boltzgen1_ifold` has a distinct inverse-folding
training target, and `boltzgen1_structuretrained_small` is the released reduced
structure-trained variant. OFoldX does not redistribute the training corpus.

### Training Procedure

The structure designer is trained with an EDM denoising objective over randomly
cropped biomolecular structures and randomly selected design and conditioning
regions. Training noise follows
`sigma = sigma_data * exp(-1.2 + 1.5 * z)`, `z ~ N(0, 1)`, with loss weight
`(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`. The inverse-folding model
uses a separate structure-conditioned residue-prediction objective.

The public artifacts do not provide every optimizer, schedule, data-mixture, and
loss-reduction detail needed to reproduce all four checkpoints. OFoldX
conversion does not retrain the models or claim to reproduce the complete
upstream training system.

## Limitations

- Structure generation and temperature-based inverse folding are stochastic.
  Candidate coordinates and sequences depend on the advancing generator stream.
- The structure-design checkpoints do not contain confidence or affinity heads.
  Those scores exist only when compatible external pairs are composed into the
  Pipeline.
- The current Pipeline implements structure design, inverse folding, optional
  refolding, optional affinity scoring, and result assembly. It does not expose
  separate released `design_folding`, filtering, or composite-ranking stages.
- Before optional refolding, designed side-chain atoms are added at the origin
  and marked present, matching the released sequence-remap convention; the
  structure designer predicts the backbone coordinates, not packed side chains.
- Recorded comparisons characterize the structure denoiser (`2.4e-4`), reverse
  EDM sampler (`1.1e-4`), and inverse-fold components (worst `6.7e-6`). Three of
  the four checkpoint variants were exercised successfully. No real-reference
  parity is established for `boltzgen1_structuretrained_small.ckpt`, whose
  distinct reduced graph lacked a runnable comparison path.
- The preserved extra `noise_scale` factor in the churn equation is a
  family-specific released behavior and must not be silently normalized to the
  Boltz-1 or Boltz-2 sampler.

## Citation

```bibtex
@article{stark2025boltzgen,
  author = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and others},
  title = {BoltzGen: Toward Universal Binder Design},
  year = {2025},
  doi = {10.1101/2025.11.20.689494},
  journal = {bioRxiv}
}
```

If OFoldX supports your research, also cite the OFoldX software:

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
for questions about this implementation or model card. Questions about the
upstream model should be directed to the authors of the
[BoltzGen paper](https://doi.org/10.1101/2025.11.20.689494).

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and checkpoints:** MIT under the
  [BoltzGen license](https://github.com/HannesStark/BoltzGen/blob/main/LICENSE),
  Copyright (c) 2025 Hannes Stärk.
- The four named BoltzGen checkpoints may be redistributed with the upstream
  MIT notice preserved. Converted weights are not relicensed by OFoldX.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
