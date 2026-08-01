# [OFoldX](https://ofoldx.valtech.ai/)

> [!TIP]
> Fold molecules. Unfold possibilities.

OFoldX is a PyTorch library for biomolecular structure prediction and design.
Use pretrained models to predict structures, design sequences and backbones,
score candidates, and pack side chains for proteins, nucleic acids, ligands,
and their complexes.

## Get Started

Install OFoldX from PyPI:

```shell
pip install ofoldx
```

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

Score a protein sequence with ProteinMPNN:

```python
from ofoldx import pipeline
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
scorer = pipeline("score", model="oteam/proteinmpnn")
result = scorer(structure, seed=7)
nll = result.scores["designed_sequence_nll"]
print(nll.value, nll.unit)
```

The result reports sequence negative log-likelihood in `nat/residue`. See the
[model](ofoldx/models) and [pipeline](ofoldx/pipelines) guides for structure
prediction, molecular design, scoring, and side-chain packing.

Install the latest source version when you need unreleased changes:

```shell
pip install git+https://github.com/OTeam-AI4S/OFoldX
```

## Community

- [GitHub Issues](https://github.com/OTeam-AI4S/OFoldX/issues): bugs, usage questions, and feature requests.
- [Hugging Face](https://huggingface.co/oteam): model checkpoints.

## Citation

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

## License

We believe openness is the Foundation of Research.

OFoldX is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

Please join us in building an open research community.

`SPDX-License-Identifier: AGPL-3.0-or-later`
