# modules

`ofoldx.modules` holds independently meaningful, reusable neural-network
mathematics: attention, Pairformer and MSA trunks, heads, atom encoders,
denoisers, diffusion, and embeddings.

The boundary is deliberate. Each family under [`ofoldx.models`](../models) owns its
concrete graph in one `model.py` and its family-specific configuration, processing,
Pipeline, conversion, and documentation.

Composition by multiple families is strong evidence for this boundary, not a
mandatory consumer count. Operators with more than one implementation are routed
through `modules/backend`.
Their PyTorch implementation defines semantics. `auto` selects OOps when it is
available and compatible with the tensors. An explicit provider uses its kernels
where implemented and the canonical PyTorch operator for the rest of the graph.

## Contents

- `attention/`, `transformer.py` — multi-head and triangle attention, conditioned transformer stacks.
- `trunk/` — Pairformer and MSA trunk blocks.
- `atom/` — atom↔token encoders, windowing, and pair encoding.
- `head/` — shared distogram, confidence, TM-score, and frame tensor operations.
- `denoiser/`, `diffusion/` — coordinate denoisers and the EDM diffusion / sampling stack.
- `embedding/` — Fourier, positional, and template embeddings.
- `geometry.py`, `functional.py` — shared geometry and tensor utilities.
- `backend/` — operator contracts and provider-specific calling-convention adapters.
