---
tags:
  - Biology
  - Protein Design
  - Inverse Folding
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ProteinMPNN

Pre-trained model for protein sequence design and scoring conditioned on a fixed backbone.

## Disclaimer

This is an UNOFFICIAL implementation of the [Robust deep learning-based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187) by Justas Dauparas, et al.

The OFFICIAL repository of ProteinMPNN is at [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN).

**The team releasing ProteinMPNN did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

ProteinMPNN is an autoregressive inverse-folding model that designs an amino-acid sequence for a fixed protein backbone.
It represents residues as a spatial nearest-neighbor graph, encodes backbone geometry with message-passing blocks, and
decodes residues in a sampled causal order. Generation supports fixed positions, tied identities, position-specific
amino-acid constraints, and sequence profiles.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[ProteinMPNN](https://huggingface.co/oteam/proteinmpnn)**: Full-backbone model, with training noise of
  [0.02](https://huggingface.co/oteam/proteinmpnn-noise002), [0.10](https://huggingface.co/oteam/proteinmpnn-noise010),
  [0.20](https://huggingface.co/oteam/proteinmpnn-noise020), or [0.30 Å](https://huggingface.co/oteam/proteinmpnn-noise030).
- **[SolubleMPNN](https://huggingface.co/oteam/solublempnn)**: Soluble-protein model, with training noise of
  [0.02](https://huggingface.co/oteam/solublempnn-noise002), [0.10](https://huggingface.co/oteam/solublempnn-noise010),
  [0.20](https://huggingface.co/oteam/solublempnn-noise020), or [0.30 Å](https://huggingface.co/oteam/solublempnn-noise030).
- **[CA-only](https://huggingface.co/oteam/proteinmpnn-ca-only)**: CA-only model, with training noise of
  [0.02](https://huggingface.co/oteam/proteinmpnn-ca-only-noise002), [0.10](https://huggingface.co/oteam/proteinmpnn-ca-only-noise010),
  or [0.20 Å](https://huggingface.co/oteam/proteinmpnn-ca-only).
- **[Training example](https://huggingface.co/oteam/proteinmpnn-training-example)**: Checkpoint from the official training example.

### Model Specification

| Variant  | Backbone Atoms | Hidden Size | Encoder Blocks | Decoder Blocks | Neighbors | Num Parameters (M) |
| -------- | -------------- | ----------- | -------------- | -------------- | --------- | ------------------ |
| Standard | N, CA, C, O    | 128         | 3              | 3              | 48        | 1.674              |
| Soluble  | N, CA, C, O    | 128         | 3              | 3              | 48        | 1.674              |
| CA-only  | CA             | 128         | 3              | 3              | 48        | 1.642              |

### Links

- **Code**: [ofoldx.models.proteinmpnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/proteinmpnn)
- **Paper**:
  [Robust deep learning-based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187)
- **Developed by**: Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, Hua Bai, Robert J. Ragotte, Lukas F. Milles,
  Basile I. M. Wicky, Alexis Courbet, Rob J. de Haas, Neil Bethel, Philip J. Y. Leung, Timothy F. Huddy, Sam Pellock,
  Doug Tischer, Frederick Chan, Brian Koepnick, Hannah Nguyen, Alex Kang, Banumathi Sankaran, Asim K. Bera, Neil P.
  King, and David Baker
- **Model type**: Autoregressive message-passing neural network for fixed-backbone protein sequence design
- **Original Repository**: [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Sequence Design

You can use this model directly with a pipeline for fixed-backbone protein sequence design:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("design", model="oteam/proteinmpnn")
result = pipeline(structure)
```

Each candidate contains the designed molecular target and named sequence scores.

#### Sequence Scoring

You can use this model directly with a pipeline for fixed-backbone protein sequence scoring:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("score", model="oteam/proteinmpnn")
result = pipeline(structure, num_decoding_orders=8, seed=7)
```

The result contains designed and global sequence negative log likelihood (NLL) in nats per residue.

### Downstream Use

#### Design and Packing

Combine a designer and an independently loaded [MPNN Packer](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/models/mpnn_packer/README.md) with
[`ProteinMpnnDesignAndPackPipeline`][ofoldx.models.proteinmpnn.ProteinMpnnDesignAndPackPipeline] to generate side-chain
coordinates after sequence design. Both operations use the coordinate-observation policy selected during preparation.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

## Training Details

ProteinMPNN is trained to recover amino-acid identities from protein backbone geometry.

### Training Data

The official [training example](https://github.com/dauparas/ProteinMPNN/blob/main/training/README.md) uses PDB biological
assemblies from 2 August 2021, with protein chains clustered at 30% sequence identity. The soluble variant focuses on
soluble proteins, while the CA-only variant uses reduced backbone geometry.

### Training Procedure

ProteinMPNN uses autoregressive sequence recovery with coordinate perturbations for robustness.

OFoldX exposes differentiable `encode` and `forward` operations and the masked, label-smoothed
[`MpnnSequenceLoss`][ofoldx.modules.mpnn.MpnnSequenceLoss] for training and fine-tuning. Dataset construction, parsing,
clustering, and complete upstream training orchestration remain the training application's responsibility.

## Citation

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

Please contact the authors of the [ProteinMPNN paper](https://doi.org/10.1126/science.add2187) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our [License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.proteinmpnn.ProteinMpnnDesignAndPackPipeline]: https://ofoldx.valtech.ai/models/proteinmpnn/#ofoldx.models.proteinmpnn.ProteinMpnnDesignAndPackPipeline
[ofoldx.modules.mpnn.MpnnSequenceLoss]: https://ofoldx.valtech.ai/modules/mpnn/#ofoldx.modules.mpnn.MpnnSequenceLoss
