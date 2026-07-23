# modules

`ofoldx.modules` holds the shared neural-network building blocks — attention, the
trunk (Pairformer and MSA), heads, atom encoders, denoisers, diffusion, and embeddings —
that more than one model family composes.

The boundary is deliberate. Each family under [`ofoldx.models`](../models) owns its
concrete graph and may split family-private implementation across ordinary files.
`modules` holds only code shared by at least two families; a block used by one family
stays with that family. Operators with more than one implementation are routed through `modules/backend`.
Their PyTorch implementation defines semantics. `auto` selects OOps when it is
available and compatible with the tensors. An explicit provider uses its kernels
where implemented and the canonical PyTorch operator for the rest of the graph. See
[`DESIGN.md`](../../DESIGN.md) for the full contract.

## Contents

- `attention/`, `transformer.py` — multi-head and triangle attention, conditioned transformer stacks.
- `trunk/` — Pairformer and MSA trunk blocks.
- `atom/` — atom↔token encoders, windowing, and pair encoding.
- `head/` — the shared distogram head and frame tensor helpers.
- `denoiser/`, `diffusion/` — coordinate denoisers and the EDM diffusion / sampling stack.
- `embedding/` — Fourier, positional, and template embeddings.
- `geometry.py`, `functional.py` — shared geometry and tensor utilities.
- `backend/` — operator contracts and provider-specific calling-convention adapters.
