---
language: protein
tags:
  - Biology
  - Protein
  - Protein Design
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# RFdiffusion

Denoising-diffusion generation for de novo protein backbones, motif
scaffolding, inpainting, interfaces, and binder design.

## Disclaimer

This is an unofficial OFoldX implementation of
[De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8)
by Joseph L. Watson, David Juergens, Nathaniel R. Bennett, et al. The upstream
authors did not write, review, or endorse this implementation or model card.
The official repository is
[RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion).

## Model Details

RFdiffusion generates protein backbones by reversing diffusion over residue
frames. A RoseTTAFold-style denoiser combines sequence, MSA, pair, template, and
timestep features with SE(3)-equivariant structure updates. Its sampler couples
Gaussian C-alpha translation diffusion with IGSO(3) rotation diffusion. The
same architecture supports de novo generation, motif scaffolding, sequence
inpainting, symmetric oligomers, interfaces, and target-conditioned binder
design through model-specific conditioning.

### Variants

- **[oteam/rfdiffusion-base](https://huggingface.co/oteam/rfdiffusion-base)**: De novo backbone generation and motif scaffolding.
- **[oteam/rfdiffusion-complex-base](https://huggingface.co/oteam/rfdiffusion-complex-base)**: Protein complex and binder design.
- **[oteam/rfdiffusion-complex-fold](https://huggingface.co/oteam/rfdiffusion-complex-fold)**: Fold-conditioned complex design.
- **[oteam/rfdiffusion-inpaint-seq](https://huggingface.co/oteam/rfdiffusion-inpaint-seq)**: Sequence-conditioned inpainting.
- **[oteam/rfdiffusion-inpaint-fold](https://huggingface.co/oteam/rfdiffusion-inpaint-fold)**: Fold-conditioned inpainting.
- **[oteam/rfdiffusion-active-site](https://huggingface.co/oteam/rfdiffusion-active-site)**: Active-site motif scaffolding.

### Model Specification

| Main Blocks | Extra Blocks | Refinement Blocks | MSA Size | Pair Size | Diffusion Timesteps | Num Parameters (M) | FLOPs (G) |
| ----------- | ------------ | ----------------- | -------- | --------- | ------------------- | ------------------ | --------- |
| 32          | 4            | 4                 | 256      | 128       | 50                  | 59.808             | 549.535   |

FLOPs use PyTorch's `FlopCounterMode`, with each multiply-add counted as two
operations. They are measured for one batch of 128 generated residues and one
denoising timestep using the base configuration. Variant configurations may
differ in their conditioning dimensions.

### Links

- **Code**: [ofoldx.models.rfdiffusion](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/rfdiffusion)
- **Paper**: [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8)
- **Developed by**: Joseph L. Watson, David Juergens, Nathaniel R. Bennett, et al.
- **Model type**: SE(3)-equivariant protein backbone diffusion model
- **Original Repository**: [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)

## Usage

Install OFoldX:

```bash
pip install ofoldx
```

```python
import ofoldx
from ofoldx.data import GeneratedSegment
from ofoldx.models.rfdiffusion import RfDiffusionPlacements

placement = RfDiffusionPlacements({"binder": {"binder": GeneratedSegment("protein", 100, 100)}})
result = ofoldx.design(None, model="oteam/rfdiffusion-base", placement=placement)
```

## Training Details

### Training Data

RFdiffusion fine-tunes a RoseTTAFold-style structure network as a denoising
diffusion model over PDB protein structures. Refer to the RFdiffusion paper for
the authoritative upstream data construction. OFoldX does not redistribute the
training set.

### Training Procedure

The training procedure adds Gaussian C-alpha translation noise and rotational
Brownian motion to residue frames, then trains the network to denoise toward
the true frames with self-conditioning across timesteps. OFoldX exposes the
differentiable denoiser for training and fine-tuning.

## Limitations

- RFdiffusion generates protein backbones. It does not design a final amino-acid
  sequence or pack side chains; unconstrained generated residues are `UNK`.
- Sampling is stochastic and results depend on the model, topology,
  conditions, seed, sampling schedule, dtype, and backend.
- One fold-conditioning `Template` is supported. Template evidence and
  secondary-structure conditioning require a fold-conditioned variant, and
  template-conditioned generated segments must have fixed lengths.
- `RfDiffusionDesignResult` intentionally omits raw denoiser logits, diffusion
  timesteps, `pX0`, `x_{t-1}`, and backbone/torsion trajectories. Those remain
  tensor/sampler diagnostics rather than candidate values.
- The non-default upstream `rep_r_min=0` all-pairs substrate-repulsion branch is
  not represented by `RfDiffusionSubstrateInteraction`.
- OFoldX does not expose the upstream `t3` symmetry alias or the inert
  `model_only_neighbors` option as separate semantics.

## Citation

```bibtex
@article{watson2023denovo,
  author = {Watson, Joseph L. and Juergens, David and Bennett, Nathaniel R. and others},
  title = {De novo design of protein structure and function with RFdiffusion},
  journal = {Nature},
  volume = {620},
  number = {7976},
  pages = {1089--1100},
  year = {2023},
  doi = {10.1038/s41586-023-06415-8}
}
```

If OFoldX supports your research, also cite:

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
original model should be directed to the RFdiffusion authors.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
