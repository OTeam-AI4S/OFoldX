---
language:
  - protein
  - dna
  - rna
tags:
  - biology
  - inverse-folding
  - protein-design
  - dna-design
  - rna-design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# NA-MPNN

NA-MPNN designs and scores sequences on fixed protein, DNA, RNA, and mixed
polymer backbones.

## Disclaimer

This is an unofficial OFoldX implementation of
[NA-MPNN](https://github.com/baker-laboratory/NA-MPNN). The upstream authors did
not write, review, or endorse this implementation or model card.

## Model Details

NA-MPNN extends fixed-backbone inverse folding to protein, DNA, RNA, and mixed
polymer assemblies. It represents each polymer residue with a shared geometric
message-passing graph while constraining sampled identities to the residue's
polymer type. The same architecture supports joint sequence design, sequence
scoring, and protein-DNA specificity prediction.

### Variants

- **[oteam/nampnn](https://huggingface.co/oteam/nampnn)**: Sequence-design model for protein and nucleic-acid assemblies.
- **[oteam/nampnn-specificity](https://huggingface.co/oteam/nampnn-specificity)**: Protein-DNA specificity model.

### Model Specification

| Vocabulary Size | Hidden Size | Encoder Blocks | Decoder Blocks | Neighbors | Atom Slots per Residue | Num Parameters (M) | FLOPs (G) |
| --------------- | ----------- | -------------- | -------------- | --------- | ---------------------- | ------------------ | --------- |
| 33              | 128         | 3              | 3              | 32        | 16                     | 2.293              | 12.245    |

### Links

- **Code**: [ofoldx.models.nampnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/nampnn)
- **Paper**: [RNA sequence design and protein-DNA specificity prediction with NA-MPNN](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2)
- **Developed by**: Baker Laboratory
- **Model type**: Autoregressive message-passing neural network for protein and nucleic-acid sequence design
- **Original Repository**: [baker-laboratory/NA-MPNN](https://github.com/baker-laboratory/NA-MPNN)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Sequence Design

Use the high-level `design` API:

```python
import ofoldx

output = ofoldx.design("complex.cif", model="oteam/nampnn")
```

## Training Details

### Training Data

NA-MPNN was trained on protein, DNA, RNA, and mixed-polymer structures. The
design and specificity variants target sequence recovery and protein-DNA
specificity, respectively.

### Training Procedure

`NaMpnnModel.forward` supports protein, DNA, and RNA coordinate
noise and optional protein-first decoding. OFoldX also exposes the
label-smoothed sequence objective used by the model.

## Citation

Please cite the upstream preprint,
[RNA sequence design and protein-DNA specificity prediction with NA-MPNN](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2).

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
