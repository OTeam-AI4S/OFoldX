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

> [!IMPORTANT]
Structure-based biomolecular design family for universal binder design, controllable protein design, and inverse folding.

> [!NOTE]
> This family registers two Model/Processor pairs (`ofoldx.boltzgen`, `ofoldx.boltzgen_ifold`) and exposes the
> canonical Processors (`BoltzGenProcessor`, `BoltzGenIFoldProcessor`), a family-owned result
> (`BoltzGenResult` / `BoltzGenCandidate` — a coordinate-producing, variable-topology result that does **not**
> reuse the frozen coordinate-free `DesignResult`), and the multi-stage `BoltzGenPipeline`. The released system
> is a 7-stage pipeline (`design -> inverse_folding -> design_folding -> folding -> affinity -> analysis ->
> filtering`); `BoltzGenPipeline.design(...)` runs design + inverse folding and composes the folding/affinity
> stages with the Boltz-2 family by **read-only** import (inject a `Boltz2Model`/`Boltz2Processor` for refold and
> a `Boltz2AffinityModel`/`Boltz2AffinityProcessor` for affinity — BoltzGen never adds a verb to boltz2). Whole-
> assembly topology is the family-owned `total_len` argument on `BoltzGenProcessor`. Fully-generated `total_len`
> assemblies are realized under the frozen §3.7 placement core (X1-B): a `BoltzGenPlacements` (or, absent one,
> a single `total_len`-long generated chain) declares ONE whole-assembly group whose inner placements are its
> chains; per-chain lengths are drawn from the `total_len` budget and the realized topology is exposed as
> `BoltzGenLayout.placement_map` (`declaration_group -> placement_id -> ordered ComponentReference occurrences`),
> resolved with `ofoldx.data.placement.resolve`. Extending an EXISTING input system with generated rows (the
> released `Structure.concatenate` X1-A path) and `BoltzGenStructureGroup` conditioning are now implemented; the
> structure-group lowering to `token_distance_mask` is ratified/frozen under §3.7.2, validated exact-equal
> against the released featurizer (2026-07-19). Neither raises.
>
> The `design` verb resolves through `BoltzGenPipeline`, not the generic single-Model `design` standard task
> (`standard_tasks=()` on both registered pairs). The released Lightning `.ckpt` files pickle upstream BoltzGen
> objects, so the path-based converter needs an upstream `boltzgen` + `pydssp` deserialization environment;
> otherwise export a plain state dict or safetensors with the upstream code and pass it to the state-dict
> converter (`convert_boltzgen_checkpoint` / `convert_boltzgen_ifold_checkpoint`).

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of [BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494) by Hannes Stark, Felix Faltings, MinGyu Choi, et al.

The OFFICIAL repository of BoltzGen is at [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen).

**The team releasing BoltzGen did not write this model card, so this model card has been written by the OFoldX team.**

## Model Details

BoltzGen is a structure-based biomolecular design family with two architectures. `BoltzGenModel` refines single and pair representations with a Pairformer / MSA trunk, adds a v1 template module, token-distance conditioning, and binding / design-mask / secondary-structure conditioning, then samples complex coordinates with EDM diffusion. `BoltzGenIFoldModel` instead encodes a backbone into a k-nearest-neighbour token graph and autoregressively decodes residue types with amino-acid and symmetry constraints. The released structure checkpoints do not contain confidence or affinity heads; those belong to the Boltz-2 families.

### Model Specification

Structure-generating designers (diffusion):

| Architecture             | Trunk Blocks   | Single Size | Pair Size | Diffusion Token Blocks | Recycling Steps | Sampler Steps |
| ------------------------ | -------------- | ----------- | --------- | ---------------------- | --------------- | ------------- |
| boltzgen                 | 64             | 384         | 128       | 24                     | 3               | 200           |
| diverse         | 64             | 384         | 128       | 24                     | 3               | 200           |
| adherence       | 64             | 384         | 128       | 24                     | 3               | 200           |
| structuretrained-small | 12 (Miniformer) | 384        | 128       | 8                      | 3               | 200           |

Inverse-folding designer (kNN graph):

| Architecture            | Encoder Layers | Decoder Layers | Node Size | Hidden Size | k-Neighbors |
| ----------------------- | -------------- | -------------- | --------- | ----------- | ----------- |
| boltzgen_ifold | 6              | 3              | 128       | 128         | 30          |

> [!NOTE]
> `diverse`, `adherence`, and `structuretrained-small` are parameter/weight variants of `BoltzGenModel`, not separate
> Python architecture classes. The small checkpoint uses 12 Miniformer trunk blocks, 3 Miniformer MSA blocks, 8
> diffusion token blocks, a Miniformer template, and no token-distance or design-conditioning channels.

### Links

- **Code**: [ofoldx.models.boltzgen](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltzgen)
- **Paper**: [BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494)
- **Developed by**: Hannes Stark, Felix Faltings, MinGyu Choi, et al.
- **Model type**: Structure-conditioned diffusion design generator with an EDM sampler, plus an inverse-folding kNN graph decoder
- **Base model**: [boltz-community/boltz-2](https://huggingface.co/boltz-community/boltz-2)
- **Original Repository**: [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen)

## Training Details

BoltzGen is trained with a diffusion objective for structure-conditioned biomolecular design, building on the Boltz-2 data pipeline. This describes the released weights; it is not an OFoldX training contract.

### Training Data

BoltzGen builds on the Boltz-2 data pipeline. The public training release includes `targets.zip` and `msa.zip` in the `boltzgen/boltzgen1_train` dataset and molecule dictionaries in `boltzgen/inference-data`; the full large-model recipe may also use additional distillation data. The `boltzgen_ifold` artifact corresponds to a distinct upstream inverse-folding training target, and `structuretrained-small` corresponds to the released small structure-trained checkpoint. OFoldX does not redistribute the training corpus.

### Training Procedure

Upstream BoltzGen trains a diffusion objective over randomly cropped biomolecular structures with randomly selected design and conditioning regions, covering a model suite that spans folding, design, and inverse folding. OFoldX converts state dicts from the released checkpoints into strict-loaded native models in memory; loading the original pickled Lightning artifact requires the upstream deserialization environment described above. Repository publication still requires the corresponding Processors and `DesignResult` decoders. Matching the upstream training loop is not a Model Hub registration requirement.

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

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OFoldX supports your research, please cite the OFoldX project as follows:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for questions or comments on the model card.

Please contact the authors of the [BoltzGen paper](https://doi.org/10.1101/2025.11.20.689494) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Limitations & intentional upstream divergences

- The diffusion noise schedule keeps the upstream-faithful `noise_scale**2` form
  rather than a simplified linear scale; this is an intentional faithfulness choice,
  not a bug.
- The release inventory carries several checkpoints (`boltzgen1_adherence`,
  `boltzgen1_diverse`, `boltzgen1_ifold`, `boltzgen1_structuretrained_small`);
  publication must state which variants are qualified. The sampler is stochastic.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** MIT
  ([boltzgen LICENSE](https://github.com/HannesStark/BoltzGen/blob/main/LICENSE),
  Copyright (c) 2025 Hannes Stärk).
- **Redistribution status: GREEN.** BoltzGen checkpoints are permissively licensed
  and freely redistributable with the upstream MIT notice preserved.
