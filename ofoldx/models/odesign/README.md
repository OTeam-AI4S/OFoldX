---
language:
  - protein
tags:
  - Biology
  - Protein Design
  - Biomolecular Structure
  - Diffusion
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ODesign

Native PyTorch implementation of the ODesign all-atom biomolecular interaction
design model and its released protein sequence-assignment composition.

## Disclaimer

This is an unofficial OFoldX implementation of
[ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).
The upstream authors did not write, review, or endorse this implementation or
model card.

## Model Details

ODesign is an all-atom diffusion model for designing biomolecular interactions.
It combines atom-level chemical features, MSA information, geometric
constraints, and recycled Pairformer representations before an EDM denoiser
generates coordinates. Distogram and bond heads provide additional supervision
over the designed assembly. The released protein workflow assigns sequences to
generated backbones with ProteinMPNN or LigandMPNN.

### Variants

- **[oteam/odesign-flex](https://huggingface.co/oteam/odesign-flex)**: Flexible-receptor interaction design.
- **[oteam/odesign-rigid](https://huggingface.co/oteam/odesign-rigid)**: Rigid-receptor interaction design.

### Model Specification

| Pairformer Blocks | MSA Blocks | Single Size | Pair Size | Denoiser Token Blocks | Num Parameters (M) | FLOPs (G) |
| ----------------- | ---------- | ----------- | --------- | --------------------- | ------------------ | --------- |
| 48                | 4          | 384         | 128       | 24                    | 355.648            | 1260.502  |

### Links

- **Code**: [ofoldx.models.odesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/odesign)
- **Paper**: [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304)
- **Developed by**: Odin Zhang, Xujun Zhang, Haitao Lin, et al.
- **Model type**: All-atom, condition-guided biomolecular diffusion model
- **Original Repository**: [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Direct Use

#### Biomolecular Design

Use the high-level `design` API:

```python
import ofoldx

output = ofoldx.design("target.cif", model="oteam/odesign-flex")
```

## Training Details

### Training Data

Refer to the [ODesign technical report](https://arxiv.org/abs/2510.22304) for the
upstream training corpus and data construction.

### Training Procedure

OFoldX preserves the released inference graph and exposes the differentiable
one-step denoiser. It does not add an OFoldX-specific training objective to the
model. Rigid and flexible receptor behavior is a conditioning difference rather
than a separate architecture.

## Citation

```bibtex
@misc{zhang2025odesign,
  title={ODesign: A World Model for Biomolecular Interaction Design},
  author={Zhang, Odin and Zhang, Xujun and Lin, Haitao and others},
  year={2025},
  eprint={2510.22304},
  archivePrefix={arXiv},
  primaryClass={q-bio.BM}
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
