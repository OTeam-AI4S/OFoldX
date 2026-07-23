---
language: protein
tags:
  - biology
  - protein-design
  - inverse-folding
  - protein-ligand
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# LigandMPNN

LigandMPNN designs protein sequences on a fixed backbone using nearby
non-protein atoms as structural context.

## Disclaimer

This is an unofficial OFoldX implementation of
[LigandMPNN](https://github.com/dauparas/LigandMPNN). The upstream authors did
not write, review, or endorse this implementation or model card.

## Model Details

LigandMPNN is an autoregressive inverse-folding model for protein sequence design
with explicit atomic context. In addition to the protein backbone, it embeds
nearby ligand, nucleic-acid, metal, and other non-protein atoms, allowing sequence
generation to respond to the local chemical environment around an interface or
binding site.

### Variants

- **[oteam/ligandmpnn](https://huggingface.co/oteam/ligandmpnn)**: LigandMPNN for fixed-backbone protein sequence design with optional non-protein atom context.

Released weight variants differ in the coordinate perturbations used during
training but use the same architecture and public design interface. Side-chain
packing is provided separately by [MPNN Packer](../mpnn_packer/README.md).

### Model Specification

| Hidden Size | Encoder Blocks | Atom-Context Blocks | Context Blocks | Decoder Blocks | Neighbors | Context Atoms | Num Parameters (M) | FLOPs (G) |
| ----------- | -------------- | ------------------- | -------------- | -------------- | --------- | ------------- | ------------------ | --------- |
| 128         | 3              | 2                   | 2              | 3              | 32        | 25            | 2.619              | 34.458    |

### Links

- **Code**: [ofoldx.models.ligandmpnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/ligandmpnn)
- **Paper**: [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1)
- **Developed by**: Justas Dauparas, Gyu Rie Lee, Robert Pecoraro, et al.
- **Model type**: Atomic-context autoregressive message-passing neural network for protein sequence design
- **Original Repository**: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)

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

output = ofoldx.design("complex.cif", model="oteam/ligandmpnn")
```

## Training Details

### Training Data

LigandMPNN was trained on protein structures with local atomic context from
ligands, nucleic acids, metals, and other non-protein entities.

### Training Procedure

LigandMPNN uses autoregressive sequence recovery with coordinate perturbations
for robustness. OFoldX exposes a differentiable teacher-forced
`forward` path for training and fine-tuning.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
