# [OFoldX](https://ofoldx.valtech.ai/zh/)

> [!TIP]
> Fold molecules. Unfold possibilities.

OFoldX 是用于生物分子结构预测与设计的 PyTorch 库。

## 快速开始

从 PyPI 安装 OFoldX：

```shell
pip install ofoldx
```

如果需要未发布的最新修改，可以从源代码安装：

```shell
pip install git+https://github.com/OTeam-AI4S/OFoldX
```

## 浏览

| 入口                            | 用途                                     |
| ------------------------------- | ---------------------------------------- |
| [`models`](ofoldx/models)       | 选择模型家族、转换检查点和加载模型仓库。 |
| [`data`](ofoldx/data)           | 表达分子身份、坐标和观测。               |
| [`io`](ofoldx/io)               | 读写 mmCIF、PDB 和原生制品。             |
| [`chemistry`](ofoldx/chemistry) | 定义化学组分和构建分子。                 |
| [`msa`](ofoldx/msa)             | 读取、配对和准备序列比对。               |
| [`templates`](ofoldx/templates) | 提供结构模板证据。                       |
| [`modules`](ofoldx/modules)     | 复用神经网络与几何组件。                 |
| [`results`](ofoldx/results)     | 解读候选、分数与分子指标。               |
| [`metrics`](ofoldx/metrics)     | 计算结构比较指标。                       |
| [`analysis`](ofoldx/analysis)   | 筛选和排序设计候选。                     |

## 社区

- [GitHub Issues](https://github.com/OTeam-AI4S/OFoldX/issues)：错误报告、使用问题和功能请求。

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
