# PxDesign

OFoldX contains a native implementation of the released PXDesign-d v0.1 condition
embedder, coordinate denoiser, and EDM sampling loop. The official
`pxdesign_v0.1.0.pt` checkpoint has exactly 732 tensors; the native graph has the
same 732-tensor contract and loads it strictly.

`PxDesignProcessor` accepts one canonical target `Structure` (or the released
generation-only `None` path) plus exactly one `PlacementSpec` containing a separate,
fixed-length protein `GeneratedSegment`. A unit-weight soft
`Contact` with an 8 Å cutoff and zero tolerance from that segment to existing
target residues is the released binary hotspot channel; other numeric semantics are
rejected because the checkpoint has only that binary input. It lowers target
chemistry, the unknown-sequence binder atom
layout, target distance-template conditioning, and reference conformers to
`PxDesignInput`. `PxDesignModel.design` decodes every coordinate sample directly to
a canonical `Structure` in the family-owned `PxDesignResult`; the generated protein has
truthful `UNK` identity while retaining the released xpb/GLY backbone atom contract.

## Design suite / repository story

PXDesign is two separable things and OFoldX keeps them separate. The **PXDesign-d
generator** is a real native Model+Processor with the `pxdesign_v0.1.0.pt` checkpoint; it is
coordinate-producing, so per the locked §5.3 decision it returns its OWN family Result,
`PxDesignResult` (a generation-ordered collection of `PxDesignCandidate`s with the frozen
single-Score `.best`/`.ranked`), rather than the frozen coordinate-free shared
`DesignResult`. This is the one place the family diverges from the earlier prototype:
`PxDesignCandidate` is a standalone family-local record (§5.3.1 freezes the shared carrier
excluding PXDesign; §5.3.3 forbids widening it), not a subclass of the frozen `DesignCandidate`,
so nothing in the shared record changes. The generator gets the standard model-repository
entry, and `PxDesignPipeline` (a normal `_PairedPipeline`) is repository-saveable.

The **PXDesign suite** is the released composition pipeline: PXDesign-d backbones →
ProteinMPNN sequencing → AF2-IG (initial-guess) evaluation → optional Protenix evaluation →
success bucketing → rank/filter. Per §6.4 it is a **live-only composition with no single
owning Model and no checkpoint**: it would not be a `_PairedPipeline`, add no registration,
and write no `pipeline.json` model-repo artifact, being reconstructed at runtime from the
registered `ofoldx.pxdesign` + `ofoldx.proteinmpnn` + `ofoldx.alphafold2` (AF2-IG, pending)
+ `ofoldx.protenix` repositories.

Two blockers keep the suite from running end to end today, so — per Principle 13, where a
public reexport IS a publication surface — the suite composition pipeline is intentionally
**not published**: it is absent from `ofoldx.models.pxdesign` entirely until both land,
rather than shipping a class whose only behavior is to raise. The blockers are:

- **AF2-IG.** The released evaluation stage is AF2 *initial-guess multimer* (seed the
  designed backbone into the recycling `prev_pos` channel and read interface metrics
  `i_pTM` / `unscaled_i_pAE`). The current OFoldX AlphaFold2 family is deterministic monomer
  with no initial-guess entry point, so the suite cannot evaluate until the AF2 family adds
  one. It is not invented here.
- **`pxdbench` (PXDesignBench).** The un-vendored `github.com/bytedance/PXDesignBench`
  submodule is the source of the ProteinMPNN binder-mode metrics, RMSD alignment,
  secondary-structure/`Rg` geometry, the two-tier success bucketing and weighted rank
  fusion, and the exact `num_seqs` / `mpnn_temperature` / AF2 recycle+model-count defaults.
  Until it is vendored those numeric defaults and the scoring stage are unresolved.

The current canonical lowering accepts exact standard protein, RNA, and DNA
components plus non-polymer Molecules, and canonicalizes MSE (selenomethionine) to
MET before featurization exactly as released does (AlphaFold3 SI §2.1: rewrite the
residue to MET and rename its selenium `SE` atom to the sulfur `SD`; reference
`pxdesign/data/parser.py::mse_to_met`). MSE and MET are the same heavy-atom graph in
the CCD, so this is exact rather than a lossy guess. Any *other* modified polymer
component is still rejected: upstream consults the CCD canonical-parent mapping,
while the present canonical component value carries neither an asserted parent
component nor that transform, and treating every such residue as `UNK` would be
false parity. Those remain part of the released-feature parity blocker.
Target `PlacementSpec` chains currently require one complete `ExistingSpan` per retained
polymer instance, followed by the generated binder. The released
target-cropping path remains blocked until those spans are materialized as truthful
canonical chains; it is rejected rather than represented as a partial Structure on
the original full chain.

MSA evidence is not consumed by PXDesign-d. The official full suite passes target
MSAs to downstream Protenix filtering and composes ProteinMPNN, AF2-IG, and Protenix
for sequence design, confidence evaluation, filtering, and ranking. Those are
separate model/pipeline computations rather than hidden PXDesign-d Processor
features.

The released generator source is inference-only, so it does not provide a reference
loss for parity work. OFoldX does not guess one, has not implemented a PXDesign
training objective, and does not expose the two distogram heads that are commented
out upstream. `PxDesignModel.forward` is an ordinary differentiable single-step
denoiser with explicit noisy coordinates and noise level;
`PxDesignModel.sample` is the raw released inference loop, including reference
conformer augmentation and the 400-step EDM defaults.

The upstream loop draws from both global PyTorch and SciPy RNG state. OFoldX instead
uses one caller-supplied `torch.Generator`, so the same integer seed is not expected
to select the same random trajectory across implementations. With the initial
noise, rigid transforms, and churn noise held fixed, the maintained
[`sampler_parity.py`](../../../scripts/parity/pxdesign/sampler_parity.py) diagnostic
compares every released noise level, each denoiser input, and final coordinates.
Generated reports remain local. Explicit-generator reproducibility is covered in the
frequent test suite. The diagnostic isolates the EDM loop; reference-conformer
augmentation and chunked/remainder sampling remain in the public sampled-output
acceptance work. The explicit Generator is an intentional RNG implementation
improvement, not a compatibility mode or a second sampler contract.

This family is intentionally not registered yet. A public `design` task must
also complete real upstream-preprocessing and sampled-output acceptance, reproduce the
released downstream suite composition and evaluation functions, and satisfy the
publication gate. The maintained script also compares the condition embedding and
one-step denoiser against the compatible Protenix 0.6.3 reference on a shared
canonical input. Confirmed upstream dependency and released-suite bugs are
recorded in [`UPSTREAM_BUGS.md`](../../../UPSTREAM_BUGS.md).

Audited upstream source: <https://github.com/bytedance/PXDesign/tree/f788441313c84c3074fe9596ac2433f96b15c763>

Official checkpoint:
<https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt>

## Limitations & intentional upstream divergences

- PxDesign is a design suite built atop the Protenix folding core and ships the single
  released `pxdesign_v0.1.0` checkpoint; portions of the upstream design suite remain
  owner-gated and are not exposed as public contracts.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** Apache-2.0
  ([PXDesign LICENSE](https://github.com/bytedance/PXDesign/blob/main/LICENSE)).
- **Redistribution status: GREEN.** The `pxdesign_v0.1.0` checkpoint is permissively
  licensed and freely redistributable with the upstream Apache-2.0 notice preserved.
