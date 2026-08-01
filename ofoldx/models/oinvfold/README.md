---
tags:
  - Biology
  - Inverse Folding
  - Protein Design
  - Nucleic Acid Design
  - Ligand Design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# OInvFold

Pre-trained models for polymer inverse folding and provisional ligand atom-element assignment from fixed coordinates.

## Disclaimer

This is an UNOFFICIAL implementation of the OInvFold models released with [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304) by Odin Zhang, et al.

The OFFICIAL repository of OInvFold is at [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).

> [!CAUTION]
> OFoldX uses centroid-based virtual global nodes with actual relative rotations and translations. The released polymer
> preparation supplies constant identity/zero global features instead. This changes checkpoint predictions; inference
> quality requires separate evaluation.

**The team releasing OInvFold did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

OInvFold predicts protein, DNA, or RNA identities, or ligand atom elements, from fixed coordinates. Polymer and ligand
checkpoints use separate graph models with geometric message passing. Polymer decoding produces molecular sequence
designs; ligand decoding produces ordered element assignments with unresolved bonds, formal charges, and chemical
component identities.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/oinvfold-protein](https://huggingface.co/oteam/oinvfold-protein)**: Protein sequence design.
- **[oteam/oinvfold-dna](https://huggingface.co/oteam/oinvfold-dna)**: DNA sequence design.
- **[oteam/oinvfold-rna](https://huggingface.co/oteam/oinvfold-rna)**: RNA sequence design.
- **[oteam/oinvfold-ligand](https://huggingface.co/oteam/oinvfold-ligand)**: Provisional ligand atom-element assignment.

Each repository pairs the model's output-class convention with its processor's coordinate and chemistry convention.

### Model Specification

| Variant | Hidden Size | Encoder Blocks | Attention Heads | Virtual Atoms | Released Output Classes | Num Parameters (M) |
| ------- | ----------- | -------------- | --------------- | ------------- | ----------------------- | ------------------ |
| Polymer | 128         | 12             | 4               | 32            | 25                      | 8.374              |
| Ligand  | 128         | 12             | 4               | 32            | 37                      | 6.429              |

### Links

- **Code**: [ofoldx.models.oinvfold](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/oinvfold)
- **Paper**: [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304)
- **Developed by**: The ODesign authors
- **Model type**: Geometric graph models for polymer inverse folding and ligand atom-element assignment
- **Original Repository**: [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Polymer Inverse Folding

You can use this model directly with a pipeline for protein inverse folding from a fixed backbone:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("design", model="oteam/oinvfold-protein")
result = pipeline(structure, use_beam=True, num_beams=4)
```

Each candidate contains its molecular target, design record, and total and mean log probabilities. Greedy decoding returns
one candidate; beam decoding returns candidates in rank order.

#### Ligand Element Assignment

You can use this model directly with a pipeline for element assignment to the observed atoms of a known ligand:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("ligand.cif")
pipeline = ofoldx.pipeline("design", model="oteam/oinvfold-ligand")
result = pipeline.design_from_structure(structure)
```

Candidates retain atom identifiers and coordinates with predicted elements and probabilities. Bond topology, formal
charge, and canonical component identity remain unresolved; the output is a provisional atom assignment.

### Downstream Use

#### Provisional Atom Coordinates

Call [`OInvFoldLigandPipeline.design`][ofoldx.models.oinvfold.OInvFoldLigandPipeline.design] with `coordinates` and
caller-defined `slot_ids` when the atoms do not yet belong to a canonical molecular graph. The returned assignments keep
those identifiers and unresolved connectivity.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

## Training Details

OInvFold uses structure-conditioned graph models for polymer identity prediction and ligand atom-element assignment.

### Training Data

The polymer and ligand checkpoints were released with ODesign. Please refer to the ODesign paper for training data and
curation.

### Training Procedure

OInvFold applies graph corruption in each encoder block during training.

OFoldX exposes differentiable polymer and ligand `encode` and `forward` operations, and
[`OInvFoldLoss`][ofoldx.models.oinvfold.OInvFoldLoss] provides masked polymer negative log likelihood. The ligand graph
has no target atom-identity supervision interface. Dataset construction and complete upstream training orchestration
remain the training application's responsibility.

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

Please contact the authors of the [ODesign paper](https://arxiv.org/abs/2510.22304) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our [License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.oinvfold.OInvFoldLigandPipeline.design]: https://ofoldx.valtech.ai/models/oinvfold/#ofoldx.models.oinvfold.OInvFoldLigandPipeline.design
[ofoldx.models.oinvfold.OInvFoldLoss]: https://ofoldx.valtech.ai/models/oinvfold/#ofoldx.models.oinvfold.OInvFoldLoss
