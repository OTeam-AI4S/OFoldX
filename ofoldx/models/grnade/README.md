---
language:
  - rna
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

Geometric RNA inverse folding with a multi-conformation GVP encoder and
autoregressive nucleotide decoder.

## Disclaimer

This is an unofficial OFoldX implementation of
[gRNAde](https://github.com/chaitjo/geometric-rna-design). The upstream authors
did not write, review, or endorse this implementation or model card.

## Model Details

gRNAde predicts RNA identities from one-dimensional sequence order, explicit
secondary-structure base pairs, and optional three-dimensional backbone
geometry. `GrnadeProcessor` lowers one canonical RNA chain or aligned
conformations to P, C4', and base-nitrogen features, sequence, base-pair, and
spatial edges, native identities, and a design mask. Crossing `base_pairs`
represent pseudoknots directly. `GrnadeModel` applies the released
multi-conformation geometric-vector encoder and masked autoregressive decoder,
while `GrnadePipeline` realizes sampled identities as canonical molecular
systems.

The implementation follows gRNAde v1.0.0 revision
`2453b18778a1981ac25f314790e5395f1e9ab218`. PyTorch Geometric carriers,
structure-file parsing, checkpoint discovery, candidate screening, ranking,
training orchestration, and filesystem output do not cross the Model boundary.

### Released Artifact

The ODesign release includes `grnade.h5`, the v1.0.0 GVP checkpoint used for RNA
inverse folding. Its converter strictly maps every non-placeholder tensor into
an ordinary OFoldX Model–Processor repository. Empty upstream device
placeholders are not model state.

### Links

- **Code**: [ofoldx.models.grnade](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/grnade)
- **Method paper**: [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://arxiv.org/abs/2305.14749)
- **Experimental study**: [Generative inverse design of RNA structure and function with gRNAde](https://www.biorxiv.org/content/10.1101/2025.11.29.691298)
- **Developed by**: Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, et al.
- **Original repository**: [chaitjo/geometric-rna-design](https://github.com/chaitjo/geometric-rna-design)

## Usage

Load the concrete design Pipeline from a converted repository:

```python
from ofoldx.models.grnade import GrnadePipeline

pipeline = GrnadePipeline.from_repository("oteam/grnade")
result = pipeline.design(
    structure,
    base_pairs=((("A", 0), ("A", 7)),),
    num_candidates=4,
    temperature=0.2,
)
```

`structure` is a canonical `Structure` containing one RNA instance, or
`instance_id=` selects the target RNA chain. `design_positions=` restricts
which residue identities may change. The Result retains every sampled system,
the aligned autoregressive logits and log-probabilities, and total and
mean designed-residue log-probability scores.

Pass a sequence of aligned `Structure` values to condition one design on
multiple conformations:

```python
result = pipeline.design(
    (open_conformation, closed_conformation),
    base_pairs=(
        (("A", 0), ("A", 7)),
        (("A", 2), ("A", 9)),
    ),
)
```

The first conformation supplies the Result target, component-reference axis, and
decoded molecular system; later conformations contribute structural features
only and must contain the same RNA sequence.

`base_pairs` is the union of the secondary-structure pairs supplied to the
released graph; base-pair edges have higher priority than overlapping spatial
or sequence edges. Set `use_coordinates=False` to build the released 1D+2D-only
graph while retaining the canonical `Structure` as the sequence and decoded
system value. File parsing remains a Pipeline/I/O convenience and never enters
the Model.

## Training Details

OFoldX preserves the released model graph and differentiable teacher-forced
`forward` operation. It does not reproduce the upstream dataset, batching,
loss, screening models, experiment tracking, or training executable. A unified
OFoldX Runner or user training loop can consume the public Model output.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation. Contact the upstream authors for
questions about gRNAde or its datasets.

## License

This OFoldX implementation is licensed under the
[GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).
The inspected upstream gRNAde source is MIT-licensed.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
