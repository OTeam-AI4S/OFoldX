# modules

`ofoldx.modules` holds the shared neural-network building blocks — attention, the
trunk (Pairformer and MSA), heads, atom encoders, denoisers, diffusion, and embeddings —
that more than one model family composes.

The boundary is deliberate. OFoldX follows a **single-model-file policy**: each family
under [`ofoldx.models`](../models) assembles these blocks in its own `model.py`, and
duplicated _architecture_ across families is fine. `modules` holds only what is genuinely
**shared across two or more families** — a block used by a single family stays in that
family. Operators with more than one implementation are routed through `modules/backend`,
which resolves the backend once and owns the CUDA guard and the eager fallback. See
[`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md) for the full contract.

## Contents

- `attention/`, `transformer.py` — multi-head and triangle attention, conditioned transformer stacks.
- `trunk/` — Pairformer and MSA trunk blocks.
- `atom/` — atom↔token encoders, windowing, and pair encoding.
- `head/` — confidence, distogram, frames, and lDDT heads.
- `denoiser/`, `diffusion/` — coordinate denoisers and the EDM diffusion / sampling stack.
- `embedding/` — Fourier, positional, and template embeddings.
- `loss/`, `geometry.py`, `functional.py` — losses and shared tensor utilities.
- `backend/` — operator dispatch (OOps when available, eager otherwise).
