---
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

# Protenix

Pre-trained model for all-atom structure prediction of proteins, nucleic acids, small molecules, and their complexes.

## Disclaimer

This is an UNOFFICIAL implementation of the [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967) by Xinshi Chen, Yuxuan Zhang, et al.

The OFFICIAL repository of Protenix is at [bytedance/Protenix](https://github.com/bytedance/Protenix).

**The team releasing Protenix did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

Protenix combines atom and token embeddings with an MSA-conditioned recycled Pairformer trunk, EDM coordinate
diffusion, and distogram and confidence heads. Template, structure-conditioned, and constraint variants provide
additional structural evidence; the constraint model supports atom contacts, token contacts, pockets, and substructures.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/protenix](https://huggingface.co/oteam/protenix)**: The default v1.0 all-atom structure predictor.
- **[oteam/protenix-20250630](https://huggingface.co/oteam/protenix-20250630)**: The v1.0 model trained on a later data cutoff.
- **[oteam/protenix-v0.5](https://huggingface.co/oteam/protenix-v0.5)**: The v0.5 base model, also available as
  [oteam/protenix-base](https://huggingface.co/oteam/protenix-base).
- **[oteam/protenix-constraint](https://huggingface.co/oteam/protenix-constraint)**: The v0.5 model with constraint conditioning.
- **[oteam/protenix-mini-tmpl](https://huggingface.co/oteam/protenix-mini-tmpl)**: The v0.5 mini model with structure conditioning.
- **Protenix-v2**: The larger template-conditioned model, using locally converted parameters. Select the `protenix-v2` checkpoint identifier when converting its weights.

### Model Specification

| Variant                  | Pairformer Blocks | MSA Stages | Single Size | Pair Size | Recycling Steps | Num Parameters (M) |
| ------------------------ | ----------------- | ---------- | ----------- | --------- | --------------- | ------------------ |
| Base v0.5                | 48                | 4          | 384         | 128       | 10              | 368.123            |
| Constraint v0.5          | 48                | 4          | 384         | 128       | 10              | 368.339            |
| Template v1              | 48                | 4          | 384         | 128       | 10              | 368.549            |
| Mini v0.5                | 16                | 1          | 384         | 128       | 4               | 134.102            |
| Mini ESM / ISM v0.5      | 16                | 1          | 384         | 128       | 4               | 2967.899           |
| Structure condition v0.5 | 16                | 1          | 384         | 128       | 4               | 134.212            |
| Tiny v0.5                | 8                 | 1          | 384         | 128       | 4               | 109.535            |
| Protenix-v2              | 48                | 4          | 384         | 256       | 10              | 464.535            |

The Mini ESM and ISM counts include the frozen ESM2 encoder.

### Links

- **Code**: [ofoldx.models.protenix](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/protenix)
- **Paper**:
  [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
- **Developed by**: Xinshi Chen, Yuxuan Zhang, Chan Lu, Wenzhi Ma, Jiaqi Guan, Chengyue Gong, Jincai Yang, Hanyu Zhang,
  Ke Zhang, Shenghao Wu, Kuangqi Zhou, Yanping Yang, Zhenyu Liu, Lan Wang, Bo Shi, Shaochen Shi, and Wenzhi Xiao
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with optional learned constraints
- **Original Repository**: [bytedance/Protenix](https://github.com/bytedance/Protenix)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Structure Prediction

You can use this model directly with a pipeline for all-atom structure prediction:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/protenix")
result = pipeline(structure, seed=7)
```

The result contains predicted structures and confidence values. Atoms must have known elements and formal charges.

### Downstream Use

#### Chemical Guidance

You can enable training-free chemical guidance during sampling:

```python
import ofoldx
from ofoldx.io import read_structure
from ofoldx.models.protenix import ProtenixTfgGuidance

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/protenix")
result = pipeline(structure, tfg_guidance=ProtenixTfgGuidance(), seed=7)
```

Guidance is disabled by default and can be configured through `ProtenixTfgGuidance`.

Template repositories accept `templates=`. Structure-conditioned repositories accept `structure_condition=` with an
in-memory structure. Constraint repositories accept `atom_contacts`, `token_contacts`, `pocket`, and `substructure=`.
The mini template checkpoint uses the structure-conditioned graph, rather than the ordinary template graph.

## Training Details

Protenix is trained with coordinate, geometric, distogram, and confidence objectives in a staged training program.

### Training Data

The Protenix training data includes biomolecular structures from the [PDB](https://www.rcsb.org/) and AlphaFold 2
predictions used for distillation. The [upstream training guide](https://github.com/bytedance/Protenix/blob/main/docs/training_inference_instructions.md)
describes data preparation.

### Training Procedure

Protenix is trained in stages with coordinate denoising, geometric, distogram, and confidence objectives.

## Citation

```bibtex
@article{protenix2025advancing,
  author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and others},
  title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  year = {2025},
  doi = {10.1101/2025.01.08.631967},
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

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for any questions or comments on the model implementation or model card.

Please contact the authors of the [Protenix paper](https://doi.org/10.1101/2025.01.08.631967) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
