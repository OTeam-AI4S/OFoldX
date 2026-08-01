# Modules

`ofoldx.modules` provides reusable neural-network components and geometry for biomolecular models. It includes
attention, Evoformer and Pairformer blocks, atom encoders, confidence heads, coordinate denoisers, and diffusion.

## Usage

Apply a shared normalization layer with the native PyTorch backend:

```python
import torch
from ofoldx.modules.backend import LayerNorm

layer = LayerNorm(8, backend="eager")
input = torch.randn(2, 8)
output = layer(input)
print(output.shape)  # torch.Size([2, 8])
```

Use `backend="auto"` to let each operator select an available implementation. The
[backend guide](backend/README.md) describes provider requests and operator availability.

## Available Components

| Components                                | Use                                                                     |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `attention`, `transformer`, `msa`, `pair` | Attention, sequence alignment features, and pair-representation updates |
| `evoformer`, `pairformer`                 | Representation-refinement architectures                                 |
| `atom`                                    | Atom–token encoding, coordinate gathering, windows, and pair features   |
| `head`                                    | Learned distogram and confidence predictions                            |
| `denoiser`, `diffusion`                   | Coordinate denoising, EDM sampling, and denoising objectives            |
| `embedding`                               | Fourier, positional, contact, and radial features                       |
| `geometry`, `functional`, `batch`         | Geometry, tensor operations, and sample/batch layouts                   |

## Backend Selection

Use `backend="auto"` for automatic selection, or request a provider such as `"eager"` or `"oops"`.
`model.set_backend("eager")` changes the backend of a loaded model. See the
[backend guide](backend/README.md) for the available providers and selection behavior.

## Compilation

Use PyTorch's `Module.compile` on a model or a repeatedly called neural-network component:

```python
model.compile()
```

For coordinate-generation models, the denoiser can be compiled separately:

| Family                                  | Denoiser                                  |
| --------------------------------------- | ----------------------------------------- |
| AlphaFold3, Protenix, ODesign, PXDesign | `model.coordinate_denoiser`               |
| Boltz1, Boltz2, BoltzGen                | `model.coordinate_diffusion.denoiser`     |
| ESMFold2                                | `model.diffusion_structure_head.denoiser` |
| RFdiffusion                             | `model.denoiser`                          |

## Related Modules

- [Models](../models/README.md): complete architectures and scientific workflows.
- [Metrics](../metrics/README.md): independent score formulas and confidence reductions.
- [Operator Backends](backend/README.md): provider selection and accelerated implementations.
