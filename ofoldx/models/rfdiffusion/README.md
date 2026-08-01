---
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

Pre-trained model for protein backbone generation from contig specifications and optional structural motifs.

## Disclaimer

This is an UNOFFICIAL implementation of the [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8) by Joseph L. Watson, et al.

The OFFICIAL repository of RFdiffusion is at [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion).

**The team releasing RFdiffusion did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

RFdiffusion reverses diffusion over protein residue frames with a RoseTTAFold-style denoiser and SE(3)-equivariant
structure updates. Its sampler couples Gaussian C-alpha translation diffusion with IGSO(3) rotation diffusion.
Sequence, MSA, pair, template, and timestep features provide model-specific conditioning.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/rfdiffusion-base](https://huggingface.co/oteam/rfdiffusion-base)**: De novo generation and motif scaffolding (`Base_ckpt.pt`).
- **[oteam/rfdiffusion-base-epoch8](https://huggingface.co/oteam/rfdiffusion-base-epoch8)**: The earlier base checkpoint (`Base_epoch8_ckpt.pt`).
- **[oteam/rfdiffusion-complex-base](https://huggingface.co/oteam/rfdiffusion-complex-base)**: Complex and binder design (`Complex_base_ckpt.pt`).
- **[oteam/rfdiffusion-complex-beta](https://huggingface.co/oteam/rfdiffusion-complex-beta)**: Topology-diverse, beta-rich binder design (`Complex_beta_ckpt.pt`).
- **[oteam/rfdiffusion-complex-fold-base](https://huggingface.co/oteam/rfdiffusion-complex-fold-base)**: Fold-conditioned complex design (`Complex_Fold_base_ckpt.pt`).
- **[oteam/rfdiffusion-inpaint-seq](https://huggingface.co/oteam/rfdiffusion-inpaint-seq)**: Sequence-conditioned inpainting (`InpaintSeq_ckpt.pt`).
- **[oteam/rfdiffusion-inpaint-seq-fold](https://huggingface.co/oteam/rfdiffusion-inpaint-seq-fold)**: Fold-conditioned inpainting (`InpaintSeq_Fold_ckpt.pt`).
- **[oteam/rfdiffusion-active-site](https://huggingface.co/oteam/rfdiffusion-active-site)**: Active-site motif scaffolding (`ActiveSite_ckpt.pt`).

The RoseTTAFold pretraining file `RF_structure_prediction_weights.pt` is not an RFdiffusion denoising checkpoint and
cannot be converted by this converter.

### Model Specification

| Variant                                | Main Blocks | Extra Blocks | Refinement Blocks | MSA Size | Pair Size | Diffusion Timesteps | Num Parameters (M) |
| -------------------------------------- | ----------- | ------------ | ----------------- | -------- | --------- | ------------------- | ------------------ |
| Base, complex, inpaint and active-site | 32          | 4            | 4                 | 256      | 128       | 50                  | 59.879             |
| Fold-conditioned                       | 32          | 4            | 4                 | 256      | 128       | 50                  | 59.880             |

### Links

- **Code**: [ofoldx.models.rfdiffusion](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/rfdiffusion)
- **Paper**:
  [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8)
- **Developed by**: Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E.
  Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Nikita Hanikel, Samuel
  J. Pellock, Alexis Courbet, William Sheffler, Jue Wang, Preetham Venkatesh, Isaac Sappington, Susana Vázquez Torres,
  Anna Lauko, Valentin De Bortoli, Emile Mathieu, Sergey Ovchinnikov, Regina Barzilay, Tommi S. Jaakkola, Frank DiMaio,
  Minkyung Baek, and David Baker
- **Model type**: SE(3)-equivariant protein backbone diffusion model
- **Original Repository**: [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Backbone Design

You can use this model directly with a pipeline for 100-residue protein backbone generation:

```python
import ofoldx
from ofoldx.models.conditions import GeneratedSegment
from ofoldx.models.rfdiffusion import RfDiffusionPlacements

placement = RfDiffusionPlacements({"binder": {"binder": GeneratedSegment("protein", 100, 100)}})
pipeline = ofoldx.pipeline("design", model="oteam/rfdiffusion-base")
result = pipeline(None, placement=placement, seed=7)
```

Generated residues retain the unknown identity `UNK`; use a sequence-design model such as ProteinMPNN to assign amino
acids. Fixed motifs retain their source coordinates and frame. Generated or diffused residues receive N, CA, C, and O
coordinates; unpredicted side chains remain unavailable.

Placements declare copied source segments and generated positions. Backbone frames require complete, noncollinear
N, CA, and C coordinates, and topology construction checks retained connections and requested new bonds. Symmetry-related
output units share one entity and sequence.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution and composition.

## Training Details

RFdiffusion is trained to recover protein backbones from noisy coordinates with optional structural conditioning.

### Training Data

RFdiffusion fine-tunes a RoseTTAFold-style network on PDB protein structures. The paper describes the upstream data
construction; OFoldX does not redistribute the training set.

### Training Procedure

RFdiffusion adds Gaussian C-alpha translation noise and rotational Brownian motion to residue frames, then learns to
denoise them with self-conditioning across timesteps.

OFoldX exposes the differentiable denoiser for training and fine-tuning. It does not reproduce the complete coupled
translation, rotation, timestep, and self-conditioning training procedure.

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

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for any questions or comments on the model implementation or model card.

Please contact the authors of the [RFdiffusion paper](https://doi.org/10.1038/s41586-023-06415-8) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
