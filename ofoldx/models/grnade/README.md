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
`2453b18778a1981ac25f314790e5395f1e9ab218` and v0.3.2 revision
`e473636bef037be1e1b2516cf00af71e3334bd4f`. PyTorch Geometric carriers and
structure-file parsing do not cross the Model boundary.

### Released Artifact

The converter accepts distinct explicit artifact identifiers for the ODesign
`grnade.h5`, the independently released v1 checkpoint, and all ten released
v0.3.2 autoregressive checkpoints. The latter retain their released one-, two-,
three-, or five-conformation Processor setting; split names do not change the
Model equation.

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
which residue identities may change. Each candidate reports total and mean log
probability, perplexity, and native recovery.

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

The first conformation supplies the component-reference axis and decoded
molecular systems; later conformations contribute structural features only and
must contain the same RNA sequence.

For the v1 Pipeline, `base_pairs` is the union of the secondary-structure pairs
supplied to the released graph; base-pair edges have higher priority than
overlapping spatial or sequence edges. Set `use_coordinates=False` to build the
released v1 1D+2D-only graph.

Released v0.3.2 repositories load through the paired legacy Pipeline:

```python
from ofoldx.models.grnade import GrnadeV032Pipeline

pipeline = GrnadeV032Pipeline.from_repository("oteam/grnade-v0.3.2-3state-all")
result = pipeline.design((first_conformation, second_conformation))
```

The legacy graph has no base-pair or sequence-edge channel.
`use_coordinates=False` therefore produces zero geometric features and no
spatial edges while retaining the canonical `Structure` for decoded systems.
File parsing remains a Pipeline/I/O convenience and never enters the Model.

## Training Details

OFoldX preserves the released model graph and differentiable teacher-forced
`forward` operation. `criterion(..., label_smoothing=...)` exposes the training
choice directly; the released training value was `0.05`.

## Limitations

Dataset construction, screening models, experiment tracking, and the upstream
training executable remain caller-owned. The raw teacher-forced Model output is
also available to custom training loops.

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
