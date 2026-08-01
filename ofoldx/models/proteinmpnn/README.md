---
language: protein
tags:
  - biology
  - protein-design
  - inverse-folding
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ProteinMPNN

ProteinMPNN is OFoldX's implementation of the ProteinMPNN message-passing
inverse-folding family for amino-acid sequence design on a fixed protein
backbone.

## Disclaimer

This is an unofficial OFoldX implementation of
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN). The upstream authors did
not write, review, or endorse this implementation or model card.

## Model Details

ProteinMPNN is an autoregressive inverse-folding model that designs an amino-acid
sequence for a fixed protein backbone. It represents residues as a spatial
nearest-neighbor graph, encodes backbone geometry with message-passing blocks,
and decodes residues in a sampled causal order. Fixed positions, tied positions,
position-specific amino-acid constraints, and sequence profiles can be applied
during generation.

### Variants

- **[oteam/proteinmpnn](https://huggingface.co/oteam/proteinmpnn)**: General-purpose ProteinMPNN using N, CA, C, and O backbone atoms.
- **[oteam/solublempnn](https://huggingface.co/oteam/solublempnn)**: ProteinMPNN trained for soluble-protein sequence design.
- **[oteam/proteinmpnn-ca-only](https://huggingface.co/oteam/proteinmpnn-ca-only)**: Reduced model for backbones represented only by CA atoms.

All variants share the same autoregressive design interface. The standard and
soluble models differ in training distribution, while the CA-only model uses a
reduced geometric input. The converter covers the complete released artifact
surface: four full-backbone noise-level checkpoints, four soluble checkpoints,
three CA-only checkpoints, and the training checkpoint consumed by PXDesign.

### Model Specification

| Variant  | Backbone Atoms | Hidden Size | Encoder Blocks | Decoder Blocks | Neighbors | Num Parameters (M) | FLOPs (G) |
| -------- | -------------- | ----------- | -------------- | -------------- | --------- | ------------------ | --------- |
| Standard | N, CA, C, O    | 128         | 3              | 3              | 48        | 1.660              | 10.734    |
| Soluble  | N, CA, C, O    | 128         | 3              | 3              | 48        | 1.660              | 10.734    |
| CA-only  | CA             | 128         | 3              | 3              | 48        | 1.629              | 10.343    |

### Links

- **Code**: [ofoldx.models.proteinmpnn](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/proteinmpnn)
- **Paper**: [Robust deep learning-based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187)
- **Developed by**: Justas Dauparas, Ivan Anishchenko, Nathaniel R. Bennett, et al.
- **Model type**: Autoregressive message-passing neural network for fixed-backbone protein sequence design
- **Original Repository**: [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Sequence Design

Load the concrete design Pipeline:

```python
import ofoldx

pipeline = ofoldx.pipeline("design", model="oteam/proteinmpnn")
output = pipeline.design("backbone.cif")
```

The paired Pipeline also exposes sequence scoring plus autoregressive,
sequence-free autoregressive, conditional, and unconditional probability
routes. The raw `ProteinMpnnModel` provides differentiable `encode` and
`forward`, as well as `sample`, `score`, and decoded `design`. For workflows
that relax side chains after sequence design,
`ProteinMpnnDesignAndPackPipeline` composes this family with an explicitly
provided MPNN Packer without registering a second ProteinMPNN architecture.

## Training Details

### Training Data

ProteinMPNN was trained on fixed protein backbones paired with native amino-acid
sequences. The soluble variant focuses its training distribution on soluble
proteins, while the CA-only variant uses reduced backbone geometry.

### Training Procedure

ProteinMPNN uses an autoregressive sequence-recovery objective with coordinate
perturbations for robustness. `ProteinMpnnModel.forward` is
differentiable and exposes the logits needed to train or fine-tune the model.
`ofoldx.models.proteinmpnn.objectives.criterion` is the typed masked,
label-smoothed training criterion used by the unified Runner. The family owns
that objective directly; sampling and scoring gather their token NLL at the
point of use.

## Limitations

The unified Runner can train the differentiable raw model on caller-supplied
structures. It does not reproduce the upstream training-set construction,
parser, clustering, or training program.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
