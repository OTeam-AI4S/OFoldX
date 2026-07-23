---
language: protein
tags:
  - biology
  - protein-design
  - side-chain-packing
  - protein-ligand
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# MPNN Packer

MPNN Packer rebuilds protein side-chain atoms on a fixed backbone using nearby
non-protein atoms as structural context.

## Disclaimer

This is an unofficial OFoldX implementation of the side-chain packer released
with [LigandMPNN](https://github.com/dauparas/LigandMPNN). The upstream authors
did not write, review, or endorse this implementation or model card.

## Model Details

MPNN Packer predicts protein side-chain conformations on a fixed backbone. It
uses the same local atomic-context representation as LigandMPNN and predicts a
three-component von Mises mixture for each side-chain chi angle, followed by
iterative coordinate refinement. The backbone and sequence remain fixed while
selected side chains are rebuilt.

### Variants

- **[oteam/mpnn-packer](https://huggingface.co/oteam/mpnn-packer)**: Side-chain packing with optional ligand and non-protein atom context.

### Model Specification

| Hidden Size | Encoder Blocks | Atom-Context Blocks | Context Blocks | Decoder Blocks | Neighbors | Context Atoms | Mixture Components | Num Parameters (M) | FLOPs (G) |
| ----------- | -------------- | ------------------- | -------------- | -------------- | --------- | ------------- | ------------------ | ------------------ | --------- |
| 128         | 3              | 2                   | 2              | 3              | 32        | 16            | 3                  | 3.572              | 22.896    |

### Links

- **Code**: [ofoldx.models.mpnn_packer](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/mpnn_packer)
- **Paper**: [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1)
- **Developed by**: Justas Dauparas, Gyu Rie Lee, Robert Pecoraro, et al.
- **Model type**: Message-passing side-chain torsion predictor with iterative coordinate refinement
- **Original Repository**: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Side-Chain Packing

Use the high-level `pack` API:

```python
import ofoldx

output = ofoldx.pack("backbone.cif", model="oteam/mpnn-packer", sequence="MKTAYIAKQR")
```

## Training Details

### Training Data

MPNN Packer was trained on protein structures with observed side-chain
conformations and optional local non-protein atom context.

### Training Procedure

The model predicts mixture distributions over side-chain chi angles and applies
iterative coordinate refinement. OFoldX exposes a differentiable
`forward` over an explicit atom14 state.

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
