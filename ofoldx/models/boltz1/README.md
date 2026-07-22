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

> [!IMPORTANT]
All-atom structure prediction for tokenized biomolecular complexes, combining a recycled representation trunk with an EDM diffusion coordinate sampler.

> [!NOTE]
> This family is registered as `ofoldx.boltz1` with the `fold` standard task, so `AutoModel`, `AutoProcessor`,
> `ofoldx.pipeline("fold", model=".../boltz1")`, and the top-level `ofoldx.fold` verb all resolve it through the
> generic standard fold Pipeline. The canonical `Boltz1Processor` lowers a molecular system (and optional MSA and
> `Boltz1Pocket`) to `Boltz1Input`, and `Boltz1Model.fold` decodes a self-contained `FoldResult`.
>
> **Family-owned pocket conditioning.** `Boltz1Pocket(binder, pocket=...)` names one binder `instance_id` and the
> pocket contact residues as canonical `ComponentReference` values. The Processor lowers it to the released
> four-class pocket one-hot (`UNSPECIFIED`, `UNSELECTED`, `POCKET`, `BINDER`) at the checkpoint-fixed 6 Å meaning;
> no distance is recomputed at fold time and `pocket=None` keeps the all-`UNSPECIFIED` released default. Pocket
> conditioning and Boltz-1x steering are reached through the concrete `Boltz1Pipeline.fold(pocket=..., use_potentials=...)`.
>
> **Trunk representations.** `Boltz1Model.representations(...)` is the expert method for the released
> `--write_embeddings` outputs: it returns the raw trunk `single` `[B, N, 384]` and `pair` `[B, N, N, 128]`
> tensors as `Boltz1Representations`,
> retaining autograd under grad. These carry vocabulary/layout axes and are deliberately **not** folded into `FoldResult`.
>
> **Family-specific `FoldResult` names and units.** Beyond the cross-family `plddt`/`pae`/`pde`/`ptm`/`iptm` names,
> this family publishes `iplddt` (complex interface pLDDT, 0–100), `ipde` (complex interface PDE, Å), `ligand_iptm`
> and `protein_iptm` (0–1), `confidence_score` (released ranking score, unitless), and the per-chain-pair
> `chain_pair_iptm` `PairMetric` (a `[C, C]` interface-pTM matrix over ordered chain `instance_id`s, 0–1, not
> symmetrized). pLDDT (both the per-token `Metric` and the complex `Score`) is published on the 0–100 scale.

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of the released Boltz-1 inference checkpoint (`boltz1_conf.ckpt`).

OFoldX converts the released Boltz-1 weights into its own parameter layout; it does not train, redistribute, or claim ownership of the upstream model. No upstream paper, DOI, or author list is recorded in the OFoldX metadata for this checkpoint, so none is asserted here.

**The team releasing Boltz-1 did not write this model card, so this model card has been written by the OFoldX team.**

## Model Details

Boltz-1 predicts all-atom structure for tokenized biomolecular complexes. A token/atom input embedder — built around an atom-attention encoder — feeds a recycling representation trunk (a deep Pairformer stack plus an optional MSA module) and a distogram head. An EDM reverse-diffusion sampler then denoises atom coordinates from noise, conditioned on precomputed pair/atom diffusion conditioning, and a trunk-imitating confidence/ranking head scores the sampled structure with PAE/pTM metrics.

Distinctive to this family are its atom-attention input embedder, the precomputed diffusion conditioning, the EDM coordinate sampler with the Karras noise schedule, the confidence/ranking head, and optional Boltz-1x inference-time steering (Feynman-Kac particle steering and physical-potential guidance), which is implemented on the private raw-tensor sampling path and disabled by default.

### Model Specification

| Pairformer Blocks | MSA Blocks | Single Size | Pair Size | Diffusion Blocks | Recycling Steps | Sampler Steps |
| ----------------- | ---------- | ----------- | --------- | ---------------- | --------------- | ------------- |
| 48                | 4          | 384         | 128       | 24               | 3               | 200           |

> [!NOTE]
> The dimensions above are the defaults of `Boltz1Config`, which reproduce the public Boltz-1 inference entry point:
> a 48-block Pairformer trunk, a 4-block MSA module, a 24-block / 16-head diffusion token transformer, and the
> Boltz-1 EDM sampling schedule (Euler solver, 200 reverse-diffusion steps). Boltz-1x inference-time steering is
> implemented by the private sampler and disabled by default. The parameter count is omitted because instantiating the
> full trunk on CPU is expensive; refer to the upstream checkpoint for the exact tensor budget. Released steering
> defaults are normalized locally by the private sampler; they are not persistent model configuration.

### Links

- **Code**: [ofoldx.models.boltz1](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz1)
- **Model type**: Diffusion-based all-atom biomolecular structure predictor (Pairformer trunk with an EDM coordinate sampler and a confidence/ranking head)

## Training Details

Boltz-1 is trained to predict all-atom structure with a diffusion (EDM) denoising objective over atom coordinates, paired with a distogram head on the trunk and a confidence head that scores the sampled structure. This describes the released weights; it is not an OFoldX training contract.

### Training Data

The upstream Boltz-1 training corpus is not recorded in the OFoldX metadata for this checkpoint. OFoldX neither redistributes nor reconstructs the training data; it consumes only the released inference weights.

### Training Procedure

OFoldX converts the released Boltz-1 checkpoint (`boltz1_conf.ckpt`) into a strict-loaded native model in memory. The converter splits the upstream monolithic score model, moving invariant pair/atom conditioning into `denoiser_conditioning` and learned denoising layers into `coordinate_diffusion.denoiser`, while preserving checkpoint tensor meaning. With the canonical `Boltz1Processor` and `FoldResult` decoder now in place, repository persistence via `ofoldx.artifacts.save_model` is unblocked. Exact upstream training-loop parity is not a Model Hub requirement; details of the upstream training schedule remain weight provenance.

## Citation

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

## Numerical epsilon note

- **Confidence normalization eps `1e-5`** (`model.py:3751,3760,3763,3861,3865,3877,3881,3899`) is **upstream-faithful — keep, do not unify.** Faithful to Boltz `confidence_utils`. Measured: unifying to `1e-6` (the AF3/ESMFold2 value) moves pTM parity-vs-upstream from 5.96e-8 to 2.53e-7 (Δ≈2e-7, 4 orders under the 1e-3 tol) but makes Boltz-1 diverge from its own upstream for no benefit. Boltz's `1e-5` is the outlier vs protenix/af3 (`1e-8`).
- **Internal inconsistency (recorded, not applied):** the complex-PDE normalization uses `.clamp_min(1e-5)` at `model.py:3760` where the sibling Boltz-2 and Boltz upstream use `+ 1e-5` (audit D5). Different operation, impact ≈ `1e-5/count` (~1e-7), identical at count==0. One-line fix to `+ 1e-5` for internal + upstream consistency; confidence-only, not a coordinate bug. See `EPSILON_ANALYSIS.md`.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Limitations & intentional upstream divergences

- Boltz1 is the structure + confidence pair from the single released `boltz1_conf`
  checkpoint. The diffusion sampler is stochastic; global confidence scores are the
  qualified surface, not any single per-atom random draw.
- The confidence epsilon `1e-5` is kept upstream-faithful (see the Numerical epsilon
  note) and is not unified with other families.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** MIT
  ([boltz LICENSE](https://github.com/jwohlwend/boltz/blob/main/LICENSE),
  Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro).
- **Redistribution status: GREEN.** The `boltz1_conf` checkpoint is permissively
  licensed and freely redistributable with the upstream MIT notice preserved.
