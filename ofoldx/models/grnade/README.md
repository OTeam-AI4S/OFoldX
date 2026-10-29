---
tags:
  - Biology
  - RNA Design
  - Inverse Folding
  - Geometric Deep Learning
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# gRNAde

Pre-trained models for RNA inverse folding from one or more backbone conformations.

## Disclaimer

This is an UNOFFICIAL implementation of the [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://arxiv.org/abs/2305.14749) by Chaitanya K. Joshi, et al.

The OFFICIAL repository of gRNAde is at [chaitjo/geometric-rna-design](https://github.com/chaitjo/geometric-rna-design).

**The team releasing gRNAde did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

gRNAde uses a geometric-vector perceptron (GVP) encoder and an autoregressive nucleotide decoder for RNA inverse folding.
The encoder combines one or more aligned conformations before predicting RNA identities. The v1 graph incorporates
sequence order, explicit secondary-structure base pairs, and optional P, C4', and base-nitrogen geometry; the v0.3.2 graph
uses spatial geometry without sequence or base-pair edge channels.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **v1**: [gRNAde](https://huggingface.co/oteam/grnade) and the [ODesign checkpoint](https://huggingface.co/oteam/grnade-odesign).
- **v0.3.2, 1 state**: [all](https://huggingface.co/oteam/grnade-v0.3.2-1state-all),
  [das](https://huggingface.co/oteam/grnade-v0.3.2-1state-das), [multi](https://huggingface.co/oteam/grnade-v0.3.2-1state-multi).
- **v0.3.2, 2 states**: [all](https://huggingface.co/oteam/grnade-v0.3.2-2state-all).
- **v0.3.2, 3 states**: [all](https://huggingface.co/oteam/grnade-v0.3.2-3state-all),
  [das](https://huggingface.co/oteam/grnade-v0.3.2-3state-das), [multi](https://huggingface.co/oteam/grnade-v0.3.2-3state-multi).
- **v0.3.2, 5 states**: [all](https://huggingface.co/oteam/grnade-v0.3.2-5state-all),
  [das](https://huggingface.co/oteam/grnade-v0.3.2-5state-das), [multi](https://huggingface.co/oteam/grnade-v0.3.2-5state-multi).

The v1 graph uses sequence, base-pair, and spatial features; the v0.3.2 graph uses spatial features and retains its
released conformation-count setting.

### Model Specification

| Variant | Encoder Blocks | Decoder Blocks | Node Scalar / Vector Size | Edge Scalar / Vector Size | Input Edge Channels | Num Parameters (M) |
| ------- | -------------- | -------------- | ------------------------- | ------------------------- | ------------------- | ------------------ |
| v1      | 4              | 4              | 128 / 16                  | 64 / 4                    | 132                 | 2.157              |
| v0.3.2  | 4              | 4              | 128 / 16                  | 64 / 4                    | 131                 | 2.157              |

### Links

- **Code**: [ofoldx.models.grnade](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/grnade)
- **Paper**: [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://arxiv.org/abs/2305.14749);
  [Generative inverse design of RNA structure and function with gRNAde](https://www.biorxiv.org/content/10.1101/2025.11.29.691298)
- **Developed by**: Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon V. Mathis, Alex Morehead,
  Rishabh Anand, and Pietro Liò
- **Model type**: Multi-conformation GVP encoder with an autoregressive RNA sequence decoder
- **Original Repository**: [chaitjo/geometric-rna-design](https://github.com/chaitjo/geometric-rna-design)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### RNA Sequence Design

You can use this model directly with a pipeline for RNA sequence design from a backbone and secondary structure:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("rna.cif")
pipeline = ofoldx.pipeline("design", model="oteam/grnade")
result = pipeline(structure, secondary_structure="((((....))))", num_candidates=4, temperature=0.2)
```

The dot-bracket string covers the selected RNA chain; use `instance_id` when the structure contains several RNA instances.
Each candidate contains its molecular target, design record, and sequence scores.

### Downstream Use

#### Multi-Conformation Design

You can condition one sequence design on multiple aligned conformations of the same RNA:

```python
import ofoldx
from ofoldx.io import read_structure

open_structure, _ = read_structure("rna_open.cif")
closed_structure, _ = read_structure("rna_closed.cif")
pipeline = ofoldx.pipeline("design", model="oteam/grnade")
result = pipeline((open_structure, closed_structure), secondary_structure="((((....))))")
```

The conformations must contain the same RNA sequence. The first is the design target; later conformations supply
structural context. v0.3.2 repositories retain their released conformation-count setting and spatial-only graph.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

## Training Details

gRNAde is trained to recover RNA nucleotide identities from one or more backbone conformations.

### Training Data

Please refer to the gRNAde papers for the training corpora and construction of single- and multi-conformation examples.
The released v0.3.2 checkpoints retain their one-, two-, three-, or five-conformation setting.

### Training Procedure

gRNAde uses autoregressive RNA sequence recovery from structural context. The released label-smoothing value was `0.05`.

OFoldX exposes differentiable teacher-forced `forward` and [`GrnadeLoss`][ofoldx.models.grnade.GrnadeLoss] for training
and fine-tuning. Dataset construction, screening, experiment tracking, and complete upstream training orchestration
remain the training application's responsibility.

## Citation

```bibtex
@inproceedings{joshi2025grnade,
  title={g{RNA}de: Geometric Deep Learning for 3D RNA inverse design},
  author={Joshi, Chaitanya K and Jamasb, Arian R and Vi{\~n}as, Ramon and
          Harris, Charles and Mathis, Simon V and Morehead, Alex and
          Anand, Rishabh and Li{\`o}, Pietro},
  booktitle={International Conference on Learning Representations},
  year={2025}
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

Please contact the authors of the [gRNAde paper](https://arxiv.org/abs/2305.14749) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our [License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

The upstream gRNAde source is MIT-licensed.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.grnade.GrnadeLoss]: https://ofoldx.valtech.ai/models/grnade/#ofoldx.models.grnade.GrnadeLoss
