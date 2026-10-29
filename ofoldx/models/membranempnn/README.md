---
tags:
  - Biology
  - Protein Design
  - Inverse Folding
  - Membrane Protein
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# MembraneMPNN

Pre-trained model for membrane-conditioned protein sequence design and scoring from fixed backbones.

## Disclaimer

This is an UNOFFICIAL implementation of the membrane-conditioned ProteinMPNN models released with [LigandMPNN](https://github.com/dauparas/LigandMPNN).

The OFFICIAL repository of MembraneMPNN is at [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

**The team releasing MembraneMPNN did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

MembraneMPNN encodes N, CA, C, and O backbone geometry as a spatial residue graph and predicts amino-acid identities with
an autoregressive message-passing decoder. Global or per-residue membrane labels supply additional conditioning.
Generation supports fixed positions, tied identities, position-specific amino-acid constraints, and logit biases.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/membranempnn-global](https://huggingface.co/oteam/membranempnn-global)**: Global label `0` denotes soluble
  proteins and `1` denotes transmembrane proteins.
- **[oteam/membranempnn-per-residue](https://huggingface.co/oteam/membranempnn-per-residue)**: Residue labels `0`, `1`,
  and `2` denote exposed/other, membrane-interface, and membrane-buried environments, respectively.

### Model Specification

| Backbone Atoms | Hidden Size | Encoder Blocks | Decoder Blocks | Neighbors | Num Parameters (M) |
| -------------- | ----------- | -------------- | -------------- | --------- | ------------------ |
| N, CA, C, O    | 128         | 3              | 3              | 48        | 1.691              |

### Links

- **Code**: [ofoldx.models.membranempnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/membranempnn)
- **Paper**:
  [Robust deep learning-based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187)
- **Model type**: Membrane-conditioned autoregressive fixed-backbone protein sequence design
- **Original Repository**: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Global Sequence Design

You can use this model directly with a pipeline for sequence design with a global membrane label:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("design", model="oteam/membranempnn-global")
result = pipeline(structure, membrane_labels=1)
```

Each candidate contains the designed molecular target and named sequence scores. A scalar label applies to every retained
protein residue.

#### Per-Residue Sequence Design

You can use this model directly with a pipeline for sequence design with per-residue membrane labels:

```python
import ofoldx
from ofoldx.data.molecular import ComponentRef
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("design", model="oteam/membranempnn-per-residue")
labels = {ComponentRef("A", 0): 2, ComponentRef("A", 1): 1}
result = pipeline(structure, membrane_labels=labels)
```

`ComponentRef` addresses an instance and its zero-based component position. Unspecified residues receive label `0`.

#### Sequence Scoring

You can use this model directly with a pipeline for sequence scoring with membrane labels:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("score", model="oteam/membranempnn-global")
result = pipeline(structure, membrane_labels=1, num_decoding_orders=8, seed=7)
```

The result contains designed and global sequence NLL in nats per residue.

### Downstream Use

#### Design and Packing

Combine a designer and an independently loaded [MPNN Packer](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/models/mpnn_packer/README.md) with
[`MembraneMpnnDesignAndPackPipeline`][ofoldx.models.membranempnn.MembraneMpnnDesignAndPackPipeline] to generate side-chain
coordinates after sequence design. Both operations use the coordinate-observation policy selected during preparation.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

### Checkpoint Conversion

Convert a checkpoint obtained with the [upstream download script](https://github.com/dauparas/LigandMPNN/blob/main/get_model_params.sh)
to an OFoldX repository:

```bash
python -m ofoldx.models.membranempnn.convert \
    model_params/global_label_membrane_mpnn_v_48_020.pt ./membrane-global --label-mode global
```

For the per-residue checkpoint, pass `--label-mode per_residue`. The label mode must match the selected weights.

## Training Details

MembraneMPNN is trained to recover protein sequences from backbone geometry and membrane-environment labels.

### Training Data

Please refer to the [upstream model descriptions](https://github.com/dauparas/LigandMPNN#24---model_type-global_label_membrane_mpnn)
for the membrane checkpoints' training data and label construction.

### Training Procedure

The global model predicts protein sequences with a binary soluble/transmembrane input. The per-residue model uses
separate environment labels for membrane-buried, membrane-interface, and other residues.

OFoldX exposes differentiable `encode` and `forward` operations for training and fine-tuning with membrane labels.
Dataset construction and complete upstream training orchestration remain the training application's responsibility.

## Citation

For the underlying ProteinMPNN architecture:

```bibtex
@article{dauparas2022robust,
  author = {Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and others},
  title = {Robust deep learning-based protein sequence design using ProteinMPNN},
  journal = {Science},
  volume = {378},
  number = {6615},
  pages = {49--56},
  year = {2022},
  doi = {10.1126/science.add2187}
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

Please contact the authors of the [LigandMPNN project](https://github.com/dauparas/LigandMPNN) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our [License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

The upstream repository states that its code and model parameters are available under the
[MIT license](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE). Retain the
upstream copyright and permission notice when distributing those artifacts.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.membranempnn.MembraneMpnnDesignAndPackPipeline]: https://ofoldx.valtech.ai/models/membranempnn/#ofoldx.models.membranempnn.MembraneMpnnDesignAndPackPipeline
