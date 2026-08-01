---
tags:
  - Biology
  - Protein Design
  - Side-Chain Packing
  - Protein-Ligand
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# MPNN Packer

Pre-trained model for protein side-chain packing conditioned on a fixed backbone, sequence, and optional atomic context.

## Disclaimer

This is an UNOFFICIAL implementation of the side-chain packer released with [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1) by Justas Dauparas, et al.

The OFFICIAL repository of MPNN Packer is at [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

**The team releasing MPNN Packer did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

MPNN Packer predicts protein side-chain conformations on a fixed backbone. It uses LigandMPNN's local atomic-context
representation to predict a three-component von Mises mixture for each side-chain chi angle, followed by iterative
coordinate refinement. The backbone and sequence remain fixed while selected side chains are rebuilt.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Hidden Size | Encoder Blocks | Atom-Context Blocks | Context Blocks | Decoder Blocks | Neighbors | Context Atoms | Mixture Components | Num Parameters (M) |
| ----------- | -------------- | ------------------- | -------------- | -------------- | --------- | ------------- | ------------------ | ------------------ |
| 128         | 3              | 2                   | 2              | 3              | 32        | 16            | 3                  | 3.591              |

### Links

- **Code**: [ofoldx.models.mpnn_packer](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/mpnn_packer)
- **Paper**:
  [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1)
- **Developed by**: Justas Dauparas, Gyu Rie Lee, Robert Pecoraro, Linna An, Ivan Anishchenko, Cameron Glasscock, and
  David Baker
- **Model type**: Message-passing side-chain torsion predictor with iterative coordinate refinement
- **Original Repository**: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Side-Chain Packing

You can use this model directly with a pipeline for protein side-chain packing:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
pipeline = ofoldx.pipeline("pack", model="oteam/mpnn-packer")
result = pipeline(structure)
```

Each prediction contains the packed structure and chi log-density metrics. Backbone coordinates retain their source
precision; at regenerated positions, side-chain atoms without model predictions, including hydrogens, have unavailable
coordinates.

Packing requires chemistry compatible with the selected standard geometry template. Unsupported modifications and
missing chemical information raise distinct errors. To pack a designed sequence, pass the candidate's final `target`.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

## Training Details

MPNN Packer is trained to model side-chain torsion distributions conditioned on protein structure and sequence.

### Training Data

MPNN Packer was trained on protein structures with observed side-chain conformations and optional local non-protein atom
context.

### Training Procedure

The model predicts mixture distributions over side-chain chi angles and applies iterative coordinate refinement.

OFoldX exposes a differentiable `forward` over an explicit atom14 state and availability mask for custom training losses.
This family does not provide an aggregate packing-loss function. Dataset construction and complete upstream training
orchestration remain the training application's responsibility.

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
