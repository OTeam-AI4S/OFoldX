---
tags:
  - Biology
  - Protein Design
  - Inverse Folding
  - Protein-Ligand
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# LigandMPNN

Pre-trained model for protein sequence design and scoring conditioned on a backbone and non-protein atomic context.

## Disclaimer

This is an UNOFFICIAL implementation of the [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1) by Justas Dauparas, et al.

The OFFICIAL repository of LigandMPNN is at [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

**The team releasing LigandMPNN did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

LigandMPNN is an autoregressive inverse-folding model for protein sequence design with explicit atomic context. In
addition to the protein backbone, it embeds nearby ligand, nucleic-acid, metal, and other non-protein atoms. This local
chemical context conditions sequence generation around interfaces and binding sites.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/ligandmpnn-noise005](https://huggingface.co/oteam/ligandmpnn-noise005)**: 0.05 Å training coordinate noise.
- **[oteam/ligandmpnn-noise010](https://huggingface.co/oteam/ligandmpnn-noise010)**: 0.10 Å training coordinate noise;
  also available as [oteam/ligandmpnn](https://huggingface.co/oteam/ligandmpnn).
- **[oteam/ligandmpnn-noise020](https://huggingface.co/oteam/ligandmpnn-noise020)**: 0.20 Å training coordinate noise.
- **[oteam/ligandmpnn-noise030](https://huggingface.co/oteam/ligandmpnn-noise030)**: 0.30 Å training coordinate noise.

### Model Specification

| Hidden Size | Encoder Blocks | Atom-Context Blocks | Context Blocks | Decoder Blocks | Neighbors | Context Atoms | Num Parameters (M) |
| ----------- | -------------- | ------------------- | -------------- | -------------- | --------- | ------------- | ------------------ |
| 128         | 3              | 2                   | 2              | 3              | 32        | 25            | 2.632              |

### Links

- **Code**: [ofoldx.models.ligandmpnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/ligandmpnn)
- **Paper**:
  [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1)
- **Developed by**: Justas Dauparas, Gyu Rie Lee, Robert Pecoraro, Linna An, Ivan Anishchenko, Cameron Glasscock, and
  David Baker
- **Model type**: Atomic-context autoregressive message-passing neural network for protein sequence design
- **Original Repository**: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Sequence Design

You can use this model directly with a pipeline for sequence design with non-protein atomic context:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("complex.cif")
pipeline = ofoldx.pipeline("design", model="oteam/ligandmpnn")
result = pipeline(structure)
```

Each candidate contains the designed molecular target and named sequence scores.

#### Sequence Scoring

You can use this model directly with a pipeline for sequence scoring with non-protein atomic context:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("complex.cif")
pipeline = ofoldx.pipeline("score", model="oteam/ligandmpnn")
result = pipeline(structure, num_decoding_orders=8, seed=7)
```

The result contains designed and global sequence NLL in nats per residue.

### Downstream Use

#### Design and Packing

Combine a designer and an independently loaded [MPNN Packer](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/models/mpnn_packer/README.md) with
[`LigandMpnnDesignAndPackPipeline`][ofoldx.models.ligandmpnn.LigandMpnnDesignAndPackPipeline] to generate side-chain
coordinates after sequence design. Both operations use the coordinate-observation policy selected during preparation.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

## Training Details

LigandMPNN is trained to recover protein sequences from backbone geometry and non-protein atomic context.

### Training Data

LigandMPNN was trained on protein structures with local atomic context from ligands, nucleic acids, metals, and other
non-protein entities.

### Training Procedure

LigandMPNN uses autoregressive sequence recovery with coordinate perturbations for robustness. Released checkpoints
share one architecture and use training coordinate-noise levels of 0.05, 0.10, 0.20, or 0.30.

OFoldX exposes a differentiable teacher-forced `forward` path for training and fine-tuning. Dataset selection, curation,
sampling, and complete upstream training orchestration remain the training application's responsibility.

## Citation

```bibtex
@article{dauparas2025atomic,
  author = {Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and others},
  title = {Atomic context-conditioned protein sequence design using LigandMPNN},
  journal = {Nature Methods},
  year = {2025},
  doi = {10.1038/s41592-025-02626-1}
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

Please contact the authors of the [LigandMPNN paper](https://doi.org/10.1038/s41592-025-02626-1) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our [License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.ligandmpnn.LigandMpnnDesignAndPackPipeline]: https://ofoldx.valtech.ai/models/ligandmpnn/#ofoldx.models.ligandmpnn.LigandMpnnDesignAndPackPipeline
