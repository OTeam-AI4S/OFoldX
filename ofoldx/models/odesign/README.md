---
license: apache-2.0
library_name: ofoldx
pipeline_tag: protein-folding
tags:
  - protein-design
  - biomolecular-structure
  - diffusion
---

# ODesign

OFoldX implementation of the released ODesign core model from the ODesign team.
ODesign is an all-atom generative world model for biomolecular interaction design,
using an atom encoder, MSA-conditioned recycling Pairformer trunk, distance-constraint
embedder, EDM coordinate denoiser, distogram head, and token-bond head.

The implementation is staged but is not a registered or published OFoldX
architecture yet. Truthful canonical representation of the released de-novo ligand
mode remains a quality-gate blocker, described below. The available protein-flex and
protein-rigid archives have both passed exact key/shape translation and strict state
loading locally; the ligand-rigid and nucleic-acid-rigid archives still require the
same manual checkpoint proof because those two multi-gigabyte files are not present
in the validation environment.

The single `ODesignModel` owns the `design` task and precise expert tensor methods:

```python
import torch

from ofoldx.models.odesign import ODesignConfig, ODesignModel, ODesignProcessor

processor = ODesignProcessor(device="cuda", dtype=torch.bfloat16)
model = ODesignModel(ODesignConfig()).to(device="cuda", dtype=torch.bfloat16)

batch = processor(structure, conditions=conditions)
tensor_output = model(
    batch,
    noisy_coordinates=noisy_coordinates,
    noise_level=noise_level,
)                                 # differentiable one-step ODesignOutput
designed = model.design(batch)     # ODesignDesignResult of backbone Structure candidates
sampled = model.sample(batch)      # expert ODesignOutput, including pairwise heads
```

`forward` is the released one-step denoising primitive: the caller supplies the
noisy coordinates and their EDM noise level, and it returns only tensors computed by
the released core graph—denoised coordinates, distogram logits, and token-bond
logits. It does not invent a training target or silently reinterpret observed input
coordinates as synthetic noise. `sample` preserves the same pairwise heads while
running the released multi-seed coordinate sampler; `design` decodes its coordinates
into canonical Results. Rigid and flexible receptor checkpoints are weight and
conditioning variants of this same graph, not separate architectures. ODesign has
no separately released fold task, so the Model does not expose `fold`.

Observed `Structure` coordinates are geometric evidence, not implicit fixation.
Partial diffusion starts from them only when the caller passes
`partial_diffusion_snr`; without that explicit argument sampling starts from full
noise even when every model atom is observed. Hard `ReferenceCoordinates` pins only
the selected atoms; soft values remain constraint evidence without pinning.
Missing/generated coordinates cannot be used for partial diffusion.
Ranged `GeneratedSegment` lengths are sampled once by `design`, from the same
seed or generator used by diffusion; `ODesignPlacements.chain_lengths` (a family-owned
per-chain length field, not a shared core invariant) can fix a chain's total length
while distributing that total across ranged segments. All samples in one call share
the realized topology as in the released inference path.

`model.set_backend(...)` accepts exactly `"auto"`, `"torch"`, or `"oops"`.
Explicit OOps selection fails when it cannot run; Results record the concrete backend
selected for that call.

ODesign core design is coordinate/structure generation. The separate upstream
OInvFold sequence-assignment checkpoints are not silently bundled into this model
identity.

## Flex and rigid: one architecture, two repositories

The two released protein checkpoints `odesign_base_prot_flex.pt` and
`odesign_base_prot_rigid.pt` are the **same graph with the same state-dict contract**;
their upstream training configs differ only in whether the receptor atoms are supplied
as fixed EDM diffusion conditions (rigid) or regenerated (flex). This is realized entirely
by whether the Processor marks existing receptor atoms into `fixed_coordinate_mask`, which
the sampler already consumes. Because there is no signature or type difference, this is
**one `ODesignModel` + one `ODesignProcessor`**, distinguished by a single Processor config
value `receptor_mode: Literal["rigid", "flex"]` (default `"flex"`). Under the registration
rule of "one pair per checkpoint," it publishes as **two repositories over one architecture**
— `ofoldx/odesign-base-prot-flex` and `ofoldx/odesign-base-prot-rigid`, whose `model.json`
both name `ofoldx.odesign` and whose `processor.json` differ only in `receptor_mode`. It is
explicitly **not** two registered architecture pairs. `rigid` auto-pins every existing
(non-generated, observed) receptor heavy atom; `flex` pins only atoms named by an explicit
hard `ReferenceCoordinates`.

## OInvFold protein composition and the M×S×K result

The full released protein deliverable is a two-stage composition under the single `design`
verb, exposed by `ODesignPipeline`: the ODesign core generates backbones over **S** seeds ×
**M** diffusion samples, and each backbone is assigned **K** top-k sequence variants by the
OInvFold sequence step. For the protein modality the OInvFold sequence module *is* the
released ProteinMPNN / LigandMPNN network, so `ODesignPipeline` **composes the already
registered `ofoldx/proteinmpnn` and `ofoldx/ligandmpnn` Pipelines read-only** (never
re-vendored), dispatching to LigandMPNN when the backbone carries ligand/nucleic atom
context and to ProteinMPNN otherwise. The sampling knobs map directly: `temperature`
(upstream `invfold_temp`) and `num_sequences` = top-k (upstream `invfold_topk`). **Beam
decoding is inert for the protein path** — the released protein sequence inference ignores
`use_beam`, so it is accepted and ignored here and never exposed as a functional protein
knob; beam exists only on the non-protein OInvFold checkpoints, which are owner-gated.

The composition returns the family-owned `ODesignDesignResult` — the released
`M×S×K` grid of `ODesignDesignCandidate`s, each tagged with integer `seed_index` /
`backbone_index` / `sequence_index` provenance and carrying its backbone `Structure` plus the
assigned-sequence `MolecularSystem`. `ODesignDesignResult` is a concrete **family Result**:
ODesign is coordinate-producing, so it owns its own carrier rather than widening the frozen
coordinate-free shared `DesignResult`. Its `best(score)` reproduces the frozen ranking
contract exactly (explicit direction, one identical unit / scope, NaN rejection, earliest
tie). Per-sequence Scores reuse the frozen ProteinMPNN / LigandMPNN semantics:
`designed_sequence_nll` (nat/residue, lower), `designed_sequence_confidence` (higher),
`sequence_recovery` (higher), and — only when the LigandMPNN atom-context model was used —
`ligand_interface_confidence` (higher). No backbone-geometry Score is fabricated: the
released core exposes no per-sample confidence head, and the distogram / token-bond logits
stay raw on `ODesignOutput`.

## Owner-gated blockers

Beyond the de-novo ligand mode above, the following remain owner-gated and are **not**
shipped as public contracts: the §3.7 placement core is now frozen (owner verdicts
ratified 2026-07-18) and the family consumes it from `ofoldx.data.placement` via the family-owned
`ODesignPlacements` condition (which re-homes the family per-chain `chain_lengths`); the hotspot split
(`ODesignHotspot(residues)` — which admits ligand components — is recorded as the target but
the running two-sided `Contact` hotspot lowering is retained); de-novo nucleic-acid generation
and the non-protein OInvFold beam paths; and the `SymbolicBond` / `DesiredCovalentLink` cyclic
closure. The `M×S×K` `ODesignDesignResult` is itself a candidate family carrier pending the
§5.3 coordinate-bearing design sub-freeze.

An existing canonical `Molecule` can be used as structural context. The released
de-novo ligand CLI path cannot yet be exposed as a generated `Molecule`: it
creates `-L` atom rows with unknown element/reference identity and sequential fake
single bonds, predicts only binary `NONE`/`ANY` bond logits, and never reconstructs
or sanitizes a chemical graph. Treating those placeholders as canonical chemistry
would be false. ODesign therefore remains unregistered until this mode has a truthful
canonical implementation; the other released modes do not lower that gate.

## Conversion

`ofoldx.models.odesign.convert.convert_odesign_checkpoint` strictly translates a
released checkpoint into the standard paired OFoldX repository format once the
architecture passes its remaining gate and is registered:

```text
model.json
processor.json
model.safetensors
README.md
```

## Upstream

- [ODesign source repository](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)
- [Technical report](https://arxiv.org/abs/2510.22304)

```bibtex
@misc{zhang2025odesign,
  title={ODesign: A World Model for Biomolecular Interaction Design},
  author={Zhang, Odin and Zhang, Xujun and Lin, Haitao and others},
  year={2025},
  eprint={2510.22304},
  archivePrefix={arXiv},
  primaryClass={q-bio.BM}
}
```

The upstream ODesign source code and released model parameters are licensed under
Apache-2.0. OFoldX library code remains governed by the OFoldX project license.

## Limitations & intentional upstream divergences

- ODesign is **owner-gated and unregistered**: the de-novo ligand CLI path emits `-L`
  atom rows with unknown element/reference identity and placeholder single bonds and
  does not reconstruct a sanitized chemical graph, so it cannot yet be exposed as a
  truthful canonical `Molecule`. De-novo nucleic-acid generation, non-protein OInvFold
  beam paths, and cyclic-closure bonds remain owner-gated.
- The flex and rigid variants share one architecture across two repositories.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** Apache-2.0
  ([ODesign LICENSE](https://github.com/odesign-team/ODesign/blob/main/LICENSE),
  Copyright 2024 ODesign Team).
- **Redistribution status: GREEN by license** (Apache-2.0, `odesign_base_prot_flex`
  and `odesign_base_prot_rigid` redistributable with notice). Publication is blocked
  by the owner-gated qualification above, not by the license.
