---
tags:
  - Biology
  - Inverse Folding
  - Protein Design
  - DNA Design
  - RNA Design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# NA-MPNN

Pre-trained model for joint protein, DNA, and RNA sequence design and scoring from molecular backbones.

## Disclaimer

This is an UNOFFICIAL implementation of the [RNA sequence design and protein-DNA specificity prediction with NA-MPNN](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2) by Andrew Kubaney, et al.

The OFFICIAL repository of NA-MPNN is at [baker-laboratory/NA-MPNN](https://github.com/baker-laboratory/NA-MPNN).

**The team releasing NA-MPNN did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

NA-MPNN extends fixed-backbone inverse folding to protein, DNA, RNA, and mixed-polymer assemblies. A geometric
message-passing graph represents polymer residues, and generation restricts identities to the residue's polymer type.
Separate checkpoints support joint sequence design and protein-DNA specificity prediction.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/nampnn](https://huggingface.co/oteam/nampnn)**: Sequence-design model for protein and nucleic-acid
  assemblies.
- **[oteam/nampnn-specificity](https://huggingface.co/oteam/nampnn-specificity)**: Protein-DNA specificity model.

### Model Specification

| Vocabulary Size | Hidden Size | Encoder Blocks | Decoder Blocks | Neighbors | Atom Slots per Residue | Num Parameters (M) |
| --------------- | ----------- | -------------- | -------------- | --------- | ---------------------- | ------------------ |
| 72              | 128         | 3              | 3              | 32        | 16                     | 2.303              |

### Links

- **Code**: [ofoldx.models.nampnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/nampnn)
- **Paper**:
  [RNA sequence design and protein-DNA specificity prediction with NA-MPNN](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2)
- **Developed by**: Andrew Kubaney, Andrew Favor, Lilian McHugh, Raktim Mitra, Robert Pecoraro, Justas Dauparas,
  Cameron Glasscock, and David Baker
- **Model type**: Autoregressive message-passing neural network for protein and nucleic-acid sequence design
- **Original Repository**: [baker-laboratory/NA-MPNN](https://github.com/baker-laboratory/NA-MPNN)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Sequence Design

You can use this model directly with a pipeline for joint protein and nucleic-acid sequence design:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("complex.cif")
pipeline = ofoldx.pipeline("design", model="oteam/nampnn")
result = pipeline(structure)
```

Each candidate contains the designed molecular target and named sequence scores. Shared nucleic tokens are the released
default: RNA bases use DNA output classes during prediction and are reconstructed as RNA residues.

> [!NOTE]
> With shared nucleic tokens, `global_omit_tokens` uses DNA class names for both DNA and RNA: `"DNA_T"` excludes T and U,
> respectively. `RNA_` names are rejected in this mode.

#### Sequence Scoring

You can use this model directly with a pipeline for sequence scoring on a mixed-polymer backbone:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("complex.cif")
pipeline = ofoldx.pipeline("score", model="oteam/nampnn")
result = pipeline(structure, num_decoding_orders=8, seed=7)
```

The result contains designed and global sequence NLL in nats per residue.

#### Protein-DNA Specificity

Use the specificity checkpoint to estimate base probabilities for a protein-DNA complex:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("complex.cif")
pipeline = ofoldx.pipeline("design", model="oteam/nampnn-specificity")
result = pipeline.specificity_profile(structure, num_samples=30, seed=7)
```

The result contains a probability for each DNA base at each selected nucleotide position.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution with `prepare`, `forward`, and `postprocess`.

## Training Details

NA-MPNN is trained for structure-conditioned recovery of protein and nucleic-acid sequences.

### Training Data

NA-MPNN was trained on protein, DNA, RNA, and mixed-polymer structures. The design and specificity variants target
sequence recovery and protein-DNA specificity, respectively.

### Training Procedure

NA-MPNN uses polymer-conditioned sequence recovery, with coordinate noise and optional protein-first decoding.

OFoldX exposes differentiable `forward` and the polymer-specific [`smoothed_loss`][ofoldx.models.nampnn.smoothed_loss]
for masked sequence recovery. Dataset construction, the upstream specificity-training dataset, and complete training
orchestration remain the training application's responsibility.

## Citation

```bibtex
@article{kubaney2025nampnn,
  author = {Kubaney, Andrew and Favor, Andrew and McHugh, Lilian and Mitra, Raktim and Pecoraro, Robert and Dauparas, Justas and Glasscock, Cameron and Baker, David},
  title = {RNA sequence design and protein-DNA specificity prediction with NA-MPNN},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.10.03.679414},
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

Please contact the authors of the [NA-MPNN paper](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our [License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.nampnn.smoothed_loss]: https://ofoldx.valtech.ai/models/nampnn/#ofoldx.models.nampnn.smoothed_loss
