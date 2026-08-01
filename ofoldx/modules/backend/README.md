# Operator Backends

`ofoldx.modules.backend` connects OFoldX's shared model operators to their PyTorch and optional accelerated
implementations. Models and pipelines accept a backend request; each operator resolves it against its own supported
implementations.

## Usage

A shared layer can select its backend directly:

```python
import torch
from ofoldx.modules.backend import LayerNorm

layer = LayerNorm(8, backend="auto")
values = torch.randn(2, 8)
output = layer(values)
print(output.shape)
```

On CPU, automatic tensor dispatch uses the operator's PyTorch default.

## Backend Requests

| Request     | Selection                                                                                             |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| `auto`      | The first installed implementation in the operator's preference order, subject to its tensor dispatch |
| `eager`     | The operator's PyTorch implementation                                                                 |
| `sdpa`      | PyTorch scaled dot-product attention where the operator exposes it                                    |
| `oops`      | The corresponding [OOps](https://github.com/OTeam-AI4S/OOps) operator where implemented               |
| `cueq`      | A supported cuEquivariance implementation                                                             |
| `apex`      | A supported Apex implementation                                                                       |
| `deepspeed` | A supported DeepSpeed implementation                                                                  |

`auto` skips unavailable packages and operators. Errors from a selected provider propagate.
The public OOps release does not implement every optimized operator; `auto` combines available OOps
operators with PyTorch fallbacks.
For NestedTensor MSAs with both sequence and residue dimensions ragged, the pair-weighted average
fallback supports eager execution but not fullgraph compilation.

An explicit `oops` request requires the OOps export for any operator that implements that backend. If the export
is unavailable, the call raises `oops_operator_unavailable`. If an operator has no implementation for a requested
public provider, it uses its public default. A model-wide request therefore allows different operators to resolve
to different implementations according to their support.

Install the optional OOps dependency with `pip install "ofoldx[oops]"`.

## Related Modules

- [Modules](../README.md): shared neural-network mathematics and compilation regions.
- [OOps](https://github.com/OTeam-AI4S/OOps): the independent optimized operator package.
