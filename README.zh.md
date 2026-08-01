# [OFoldX](https://ofoldx.valtech.ai/zh/)

> [!TIP]
> Fold molecules. Unfold possibilities.

OFoldX 是用于生物分子结构预测与设计的 PyTorch 库。
通过预训练模型，可以预测结构、设计序列和骨架、评价候选并预测侧链构象，
处理蛋白质、核酸、配体及其复合物。

## 快速开始

从 PyPI 安装 OFoldX：

```shell
pip install ofoldx
```

读取结构前，将 `OFOLDX_CCD_ARTIFACT` 指向已有 CCD 文件。

使用 ProteinMPNN 对蛋白质序列评分：

```python
from ofoldx import pipeline
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
scorer = pipeline("score", model="oteam/proteinmpnn")
result = scorer(structure, seed=7)
nll = result.scores["designed_sequence_nll"]
print(nll.value, nll.unit)
```

结果给出序列负对数似然，单位为 `nat/residue`。
[模型](ofoldx/models)和 [pipeline](ofoldx/pipelines) 指南介绍结构预测、分子设计、评分与侧链构象预测。
训练请参阅 [runner 指南](ofoldx/runners)。

如果需要未发布的最新修改，可以从源代码安装：

```shell
pip install git+https://github.com/OTeam-AI4S/OFoldX
```

## 社区

- [GitHub Issues](https://github.com/OTeam-AI4S/OFoldX/issues)：错误报告、使用问题和功能请求。
- [Hugging Face](https://huggingface.co/oteam)：模型检查点。

## 引用

> [!NOTE]
> 本仓库提供的内容是 OFoldX 项目的一部分。
> 如果 OFoldX 对你的研究有帮助，请按如下方式引用 OFoldX。

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## 许可证

我们相信开放是研究的基础。

OFoldX 在 [GNU Affero 通用公共许可证](license.md)下授权。

对于额外条款和澄清，请参阅我们的[许可协议常见问题解答](license-faq.md)。

请加入我们，共同建立一个开放的研究社区。

`SPDX-License-Identifier: AGPL-3.0-or-later`
