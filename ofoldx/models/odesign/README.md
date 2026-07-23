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

FLOPs use PyTorch's `FlopCounterMode`, with each multiply-add counted as two
operations. They are measured for one batch of 128 generated residues (512 atom
rows), one recycling pass, and one denoising step.

### Links

- **Code**: [ofoldx.models.odesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/odesign)
- **Paper**: [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304)
- **Developed by**: Odin Zhang, Xujun Zhang, Haitao Lin, et al.
- **Model type**: All-atom, condition-guided biomolecular diffusion model
- **Original Repository**: [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

## Usage

Install OFoldX:

```bash
pip install ofoldx
```

```python
import ofoldx

result = ofoldx.design("target.cif", model="oteam/odesign-flex")
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

## Limitations

- The released de novo ligand CLI creates atoms without sufficient element,
  reference-component, and sanitized bond identity to construct a truthful
  canonical molecule. That generation mode is not exposed.
- De novo nucleic-acid generation and non-protein OInvFold beam decoding are not
  included in this protein composition.
- Existing canonical molecules are supported as structural context.
- The core model has no per-sample confidence head. Distogram and token-bond
  logits remain raw outputs rather than fabricated ranking scores.
- Sampling is stochastic and depends on the supplied seed or generator.

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
