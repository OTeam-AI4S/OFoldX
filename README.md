# [OFoldX](https://ofoldx.valtech.ai/)

> [!TIP]
> Fold molecules. Unfold possibilities.

OFoldX is a PyTorch library for biomolecular structure prediction and design.

## Get Started

Install OFoldX from PyPI:

```shell
pip install ofoldx
```

Install the latest source version when you need unreleased changes:

```shell
pip install git+https://github.com/OTeam-AI4S/OFoldX
```

## Explore

| Entry point                     | Use it for                                                  |
| ------------------------------- | ----------------------------------------------------------- |
| [`data`](ofoldx/data)           | Represent molecular identity, coordinates and observations. |
| [`io`](ofoldx/io)               | Read and write mmCIF, PDB and native artifacts.             |
| [`chemistry`](ofoldx/chemistry) | Define chemical components and construct molecules.         |
| [`msa`](ofoldx/msa)             | Read, pair and prepare sequence alignments.                 |
| [`modules`](ofoldx/modules)     | Reuse neural-network and geometry components.               |

## Community

- [GitHub Issues](https://github.com/OTeam-AI4S/OFoldX/issues): bugs, usage questions, and feature requests.

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
