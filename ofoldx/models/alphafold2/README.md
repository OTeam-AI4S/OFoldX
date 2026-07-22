---
language: protein
tags:
  - Biology
  - Protein
  - Structure Prediction
license: cc-by-4.0
library_name: ofoldx
pipeline_tag: other
---

# AlphaFold 2

Native PyTorch AlphaFold 2 monomer, monomer-pTM, and AlphaFold-Multimer
implementation for OFoldX. This is an unofficial conversion of
[AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) and
[AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034); the official
implementations are maintained by
[Google DeepMind](https://github.com/google-deepmind/alphafold).

This family is one directory / one package exporting **two** registered
architecture IDs that share the Evoformer / extra-MSA / StructureModule trunk, its
heads, and the AF2 geometry utilities:

- `ofoldx.alphafold2` — monomer and monomer-pTM (`AlphaFold2Model` / `AlphaFold2Processor`);
- `ofoldx.alphafold2_multimer` — AlphaFold-Multimer (`AlphaFold2MultimerModel` / `AlphaFold2MultimerProcessor`).

Their Processor implementations live in [`processor.py`](processor.py) and
[`multimer_processor.py`](multimer_processor.py), respectively.

The two IDs exist because their Processor call signature and required input/output
type differ (Principle 7): the monomer Processor lowers one unbound protein instance
with an entity-bound `MSA`; the multimer Processor lowers a multi-instance
`MolecularSystem` with a paired `MSACollection` and requires the chain-identity
features (`chain_index` / `entity_index` / `symmetry_index`) the monomer contract cannot express.

Both IDs are intentionally absent from the public model registry: their single-slice
tensor graphs and checkpoint converters are implemented, but the released stochastic
MSA/recycling/ensemble inference path and complete public-output parity are not.
OFoldX does not publish a partial family. `standard_tasks` is therefore `()` on both
pairs today; `fold` is the target task for both and flips to `("fold",)` only at the
publication gate, together with real parity in `scripts/parity/alphafold2.py`.

## Contract

The two architecture IDs, their Model, Processor, and target (gated) standard task:

- `ofoldx.alphafold2` — `AlphaFold2Model` / `AlphaFold2Processor` / `fold`;
- `ofoldx.alphafold2_multimer` — `AlphaFold2MultimerModel` / `AlphaFold2MultimerProcessor` / `fold`.

The Processor accepts exactly one standard protein monomer as an inline sequence,
FASTA path, `PolymerSequence`, or one-instance `MolecularSystem`. It accepts an
entity-bound `MSA` (or a local aligned FASTA/A3M path) and aligned `Template` values.
Observed `Structure` values and structure paths are rejected rather than guessed as
positional templates; aligned coordinates must be supplied through explicit
`Template` values. Conditions, heterogeneous systems, multiple instances,
`MSACollection`, and AF-Multimer inputs are also rejected rather than discarded or
reinterpreted. Canonical `UNK` target residues map to the released X token.

`model(batch)` returns `AlphaFold2Output`, the differentiable raw tensor contract:
single/pair representations, the structure trajectory, atom37 coordinates and masks,
side-chain angles, distogram, masked-MSA, pLDDT, PAE (for pTM weights), and
experimentally-resolved logits. Gradients pass through the final recycle in both
training and evaluation modes; mode changes affect only ordinary dropout.
The processed tensor contract always includes zero-masked extra-MSA and template
rows when that evidence is absent, because the released stacks still execute.

`model.fold(batch)` returns a self-contained `FoldResult`. Every prediction contains
a canonical complete-heavy `Structure`, residue-indexed pLDDT, optional PAE, mean
pLDDT and pTM scores, and named tensor diagnostics. The multimer `fold()` adds the
interface values — `iptm` (over cross-`chain_index` pairs) and `ranking_score`
(`0.8 * iptm + 0.2 * ptm`, the released `weighted_ptm_score`, the natural
`FoldResult.best("ranking_score")` key) — all always present because the multimer
TM head is always constructed.

### Frozen stochastic `fold()` signature

Both Models freeze the public `fold()` signature
`fold(batch, *, num_samples=1, generator=None, recycling_steps=None)`.
The model graph is a pure function of `(tensors, generator)`: it accepts an injected
`torch.Generator` only and never resolves an integer `seed` or creates a generator —
the `seed`/`generator` resolution lives once at the pipeline boundary
(`pipelines/_generator.py:resolve_generator`).
The tensor graph (`forward`) is deterministic for one fixed processed feature slice.
Complete released inference is stochastic because it samples and masks MSA rows
independently across recycling and ensemble slices, so the injected `generator` seeds
that sampling and `num_samples` produces `num_samples` generation-ordered predictions
in one `FoldResult`. Until that stochastic MSA path lands for multimer, the
implementation keeps the eager guards rejecting `num_samples != 1` and `generator`; the
signature does not change when it lands, only the guard bodies. This is the one and only place
the family consumes randomness. `num_ensembles` is a multimer config extent; ensemble
averaging is deferred with the stochastic MSA path (it is numerically inert on a
single deterministic slice).

### Relaxation is deferred to a future Tik

Amber/OpenMM structure relaxation does **not** ship this version. Per the owner
decision it is a deferred follow-on: `fold()` returns only the unrelaxed §5.2
single-structure `FoldPrediction`, so the frozen single-structure contract is never
mutated or duplicated. When relaxation lands it will be a separate post-fold
operation (not a `fold(relax=True)` argument).

```python
from ofoldx.models.alphafold2 import AlphaFold2Config, AlphaFold2Model, AlphaFold2Processor

processor = AlphaFold2Processor(device="cuda", dtype=None)
batch = processor("MKAYLVGDEK")
model = AlphaFold2Model(AlphaFold2Config()).cuda().eval()
result = model.fold(batch)
structure = result.predictions[0].structure
```

## Architecture and variants

The released graph contains target/MSA embedding, recycling, template pair and
torsion embedding, the extra-MSA and 48-block Evoformer stacks, the eight-layer
invariant-point-attention StructureModule, and the standard auxiliary/confidence
heads. Runtime backend selection uses `model.set_backend("auto" | "torch" | "oops")`;
the runtime backend is not serialized in `model.json`.

Official DeepMind `model_1` through `model_5` and their `_ptm` variants share the
monomer `ofoldx.alphafold2` architecture. Models 1–2 carry template parameters;
models 3–5 do not. pTM variants carry the predicted-aligned-error head.

AlphaFold-Multimer (`model_{1..5}_multimer_v3`) is the second registered pair
`ofoldx.alphafold2_multimer`. It reuses the shared trunk and heads and differs in a
chain-relative `InputEmbedderMultimer`, an atom37 multimer template embedder, a
22-class masked-MSA head, an always-present interface-pTM branch, and its
`chain_index`/`entity_index`/`symmetry_index` chain-identity input. These labels are family records;
the architecture IDs are the two registered pairs.

## Weights and conversion

`convert_alphafold2_checkpoint(checkpoint, ...)` strict-loads and returns an
in-memory model. Its public signature is unchanged; it dispatches on the model
family name. It supports:

- official DeepMind monomer/pTM `params_model_*.npz`, a params directory, or the official tar archive;
- official DeepMind multimer `params_model_{1..5}_multimer_v3.npz` (an `is_multimer`
  branch that builds `AlphaFold2MultimerConfig` / `AlphaFold2MultimerModel` and applies
  the multimer key normalization). Multimer weights live inside
  `alphafold_params_2022-12-06.tar` alongside the monomer/pTM NPZs;
- OpenFold PyTorch checkpoints (monomer only for now).

Converters recognize the source schema and transform keys/layouts. They construct
the target on the meta device and use `load_state_dict(strict=True, assign=True)` as
the sole target key/shape authority. Official monomer and pTM NPZ weights and a full
OpenFold `finetuning_ptm_1` checkpoint have been strict-loaded; the multimer key
normalization is a best-effort port pending a strict-load against real
`model_*_multimer_v3` weights. Real weights remain caller-owned under `~/weights` and
are never downloaded by frequent CI.

```python
from ofoldx.models.alphafold2.convert import convert_alphafold2_checkpoint

model = convert_alphafold2_checkpoint(
    "params_model_1_ptm.npz",
)
```

The converter does not write a standard model repository while this family is
unregistered. Repository publication begins only after the complete Model,
Processor, repository round-trip, public fold path, and publication evidence pass
the registration gate.

Upstream DeepMind parameters are distributed under CC BY 4.0. OpenFold code is
Apache-2.0 and its released parameters are CC BY 4.0. OFoldX source code is
AGPL-3.0-or-later; users must separately comply with the terms of the weights they
obtain.

## Real-checkpoint parity

No production-weight result is stored in this repository. A maintained
`scripts/parity/alphafold2.py` public-output comparison is still required before
publication. It must read the reference checkout from `~/reference` and caller-owned
weights from `~/weights`, and leave generated reports local.

That comparison must cover the complete released inference algorithm for BOTH
registered pairs, not only a single-slice graph or strict checkpoint conversion.
Remaining work includes MSA sampling/masking, per-recycle resampling, ensembles,
variant-specific Processor limits, and complete variant/head/FoldResult parity. The
multimer half additionally requires parity on paired-MSA lowering, chain-identity
features, ensemble averaging, the multimer template embedder, the interface-pTM /
ipTM / ranking-score heads, and the multimer weight key normalization against real
`model_*_multimer_v3` weights. Published losses, initializers, and the upstream
training recipe remain useful provenance, but fresh-model training is not a Model Hub
gate.

## Numerical epsilon note

- **OPM denominator eps `1e-3`** is hidden as a constructor default (`modules/trunk/evoformer.py:418`, `msa.py:933`) rather than surfaced in `config.py`. The value matches upstream and is uniform with the AF3/Boltz OPM backend (`modules/backend/outer_product_mean.py:79`); surfacing it into `config.py` is a hygiene item only.
- **Softmax mask sentinel `-1e5`** (`model.py:1172`) is numerically inert: in fp32/bf16 the post-softmax output is bit-identical to `torch.finfo(dtype).min` (measured max-diff 0.00e+00). `finfo.min` is the dtype-safe unified form (`-1e5` overflows fp16). Unify for hygiene, zero error. See `EPSILON_ANALYSIS.md`.

## Citation

```bibtex
@article{jumper2021highly,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  title = {Highly accurate protein structure prediction with AlphaFold},
  journal = {Nature},
  volume = {596},
  pages = {583--589},
  year = {2021},
  doi = {10.1038/s41586-021-03819-2}
}
```

## Limitations & intentional upstream divergences

- OFoldX ships the monomer and multimer AF2 architecture with both the DeepMind and
  OpenFold converters. The structure module and confidence heads are the qualified
  numeric surface; the full recycling / MSA-ensemble inference orchestration and
  template featurization are audited but are not the tensor-parity target.
- Shared cross-attention carries a per-family key-normalization policy; AF2 retains
  its upstream-faithful attention scaling rather than a unified cross-family form.
- The confidence epsilon is kept upstream-faithful (see the Numerical epsilon note);
  it is not folded into other families' eps values.
- A parity run applies only to the checkpoint, converter, and settings supplied to
  the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code:** DeepMind AlphaFold2 code is Apache-2.0; OpenFold code is
  Apache-2.0 ([openfold LICENSE](https://github.com/aqlaboratory/openfold/blob/main/LICENSE)).
- **DeepMind AlphaFold2 model parameters:** distributed by DeepMind under their own
  parameter terms (CC BY 4.0 attribution framing). OFoldX does not own these weights
  and cannot confirm redistribution scope on the owner's behalf.
- **OpenFold-trained weights:** Apache-2.0.
- **Redistribution status: RED for the DeepMind-parameter variant** — owner/upstream
  confirmation of attribution and redistribution terms is required before any public
  upload. **GREEN for the OpenFold-trained weights** (Apache-2.0, redistributable
  with notice). Converted weights are not relicensed as AGPL.
