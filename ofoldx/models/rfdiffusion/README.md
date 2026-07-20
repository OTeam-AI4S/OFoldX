---
language: protein
tags:
  - Biology
  - Protein
  - Protein Design
  - Structure Prediction
license: bsd-3-clause
library_name: ofoldx
pipeline_tag: other
---

# RFdiffusion

> [!IMPORTANT]
Denoising-diffusion generative model for de novo protein backbone design, motif scaffolding, and binder design.

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8) by Joseph L. Watson, David Juergens, Nathaniel R. Bennett, et al.

The OFFICIAL repository of RFdiffusion is at [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion).

**The team releasing RFdiffusion did not write this model card, so this model card has been written by the OFoldX team.**

## Model Details

RFdiffusion is a generative model that produces novel protein backbones by reversing a denoising-diffusion process over residue frames. A RoseTTAFold-style structure network serves as the denoising trunk: at each diffusion timestep it consumes the current noised backbone (through MSA, pair, and template embedders) and predicts a cleaner set of residue frames, refined by an SE(3)-equivariant structure module. A native reverse-diffusion sampler drives a single design trajectory from Gaussian noise down to a finished backbone, coupling a Euclidean diffuser for C-alpha translations with an IGSO3 diffuser for backbone-frame rotations. OFoldX expresses de novo segments, motif scaffolding, inpainting, interfaces, compactness, smooth coordination, substrate interactions, point-group symmetry, cyclic closure, and fold conditioning through canonical conditions rather than RFdiffusion-specific strings. Fold-conditioned checkpoints consume one canonical `Template`; its explicit target-to-template alignment conditions the aligned output rows, while unaligned target rows remain masked.

`RFdiffusionModel.forward(batch)` is the differentiable denoiser and returns every released training head. `RFdiffusionModel.design(batch)` runs reverse diffusion. Operator selection belongs to the Model: `model.set_backend("torch")`, `model.set_backend("oops")`, or `model.set_backend("auto")`.

An `ExistingSpan` keeps its canonical sequence visible by default, matching the
released contig-map semantics. Sequence is masked only when the caller explicitly
passes ordinary component references through `inpaint_sequence=(...)`; generated
segments remain masked to the denoiser. Their canonical output identity is CCD
`UNK`, while the Structure contains only the backbone sites RFdiffusion produced;
the upstream poly-glycine PDB rendering is not promoted to a chemical prediction.
`num_sampling_steps` defines the active translation, rotation, denoiser,
and timestep-feature schedules together rather than truncating a schedule built for
a different number of steps. Hard singleton residue identities and fixed placement
chain totals are preserved through canonical Result decoding.

`design` exposes the released CA/frame noise scales and independent `constant` or
`linear` schedules. `final_noise_scale_ca` and `final_noise_scale_frame` are optional
linear-interpolation endpoints; a different endpoint with a constant schedule is an
error rather than a silent no-op. The timestep-one endpoint is not itself sampled
when the final step returns the denoiser's raw `pX0`.

The raw sampler returns that upstream `pX0`. `RFdiffusionModel.design` separately
Kabsch-aligns it to a fixed motif when requested and then restores the fixed rows as
a canonical Result postcondition. Final pLDDT, `pX0`, and `x_{t-1}` diagnostics are
always present. Full reverse-order trajectories (`T, T-1, ..., final_step`) are
retained on CPU only with `return_trajectory=True`.

Because RFdiffusion is coordinate-producing and variable-topology (`ContigMap.length`
is sampled per candidate), `design` returns the family-owned `RFdiffusionResult`, not
the shared coordinate-free `DesignResult`. It lives in this family's own directory and
is not a subclass or widening of the shared record (DESIGN.md §5.3.1/§5.3.3), mirroring
the migrated `ESMFold2DesignResult`. Each `RFdiffusionDesignCandidate` owns the complete
decoded backbone `Structure` (which owns its `MolecularSystem` via `Structure.system`);
tuple position in `RFdiffusionResult.candidates` is sample identity, and
`RFdiffusionResult.best("mean_plddt")` selects the earliest best design under one named
Score. The Result carries two mapped values and nothing else (no logits, distograms,
torsion/backbone trajectories, timesteps, or `pX0`/`x_{t-1}`; no conditions or run
controls):

- `plddt` — a per-residue `Metric`, unit `"0-100"`, whose `axis` is every output
  residue (generated `UNK` + existing) in output-row order. The released expectation on
  `(0, 1]` is multiplied by 100 in decode so the §5.1 cross-family reserved `plddt` name
  carries its reserved 0-100 scale (a mapping requirement, not a sampler change).
- `mean_plddt` — a whole-design scalar `Score`, unit `"0-100"`, `preferred="higher"`,
  the arithmetic mean of `plddt` over all output residues.

`design` also owns the family-owned `length` (`ContigMap.length`) topology argument: the
combined generated length across all inpaint chains, excluding receptor `ExistingSpan`
rows. It is a concrete family operation argument (distinct from ODesign per-chain lengths
and BoltzGen `total_len`), not a shared placement-core invariant, and is accepted and validated at
the Processor. Its cross-segment realization, together with `provide_seq` on generated
rows and the placement core, remains owner-gated pending the
reopened §3.7 placement identity (§3.7.3 X1/X2).

Point-group projection uses the audited upstream C/D/T/O/I matrix order, copies the
first unit's sequence into every unit, and inserts the released residue-index unit
breaks before every network call. The polyhedral sets have an OFoldX-owned rounded
lexicographic canonical order before the fixed upstream permutation, so SciPy's
internal generator order is irrelevant. C, D, T, and I directly rotate the first unit.
Octahedral projection can additionally recenter each rotated unit and place it at
`symmetry_radius_angstrom` through the ordinary `symmetry_recenter` execution
argument. OFoldX does not copy the upstream `t3` alias or the inert
`model_only_neighbors` option as separate public semantics.

### Model Specification

| Main Blocks | Extra Blocks | Refinement Blocks | MSA Size | Pair Size | Diffusion Timesteps |
| ----------- | ------------ | ----------------- | -------- | --------- | ------------------- |
| 32          | 4            | 4                 | 256      | 128       | 50                  |

> [!NOTE]
> These dimensions describe the released `base` RFdiffusion checkpoint as configured in
> [`RFdiffusionConfig`][ofoldx.models.rfdiffusion.config.RFdiffusionConfig]. The equivariant refinement network uses
> an SE(3)-transformer structure module (a single layer of 32 channels across 2 degrees for the main blocks, and a
> wider 64-scalar top-k refinement stack). The parameter count is omitted here because RFdiffusion is a large trunk
> model that OFoldX does not instantiate at card-build time.

OFoldX represents auxiliary geometry through canonical conditions instead of upstream potential strings. In
particular, monomer and binder compactness use `RadiusOfGyration`. Dimer compactness is expressed by one
`RadiusOfGyration` condition per monomer, normally with half the desired total weight on each condition. Each term
uses its own monomer centroid. This intentionally corrects the released `dimer_ROG` helper's unambiguous typo, which
computes the second monomer's radius around the first monomer's centroid.

The released contact-number potentials map to `Coordination`; OFoldX evaluates their analytic
`1 / (1 + x**6)` form, removing the upstream quotient's undefined `0 / 0` points without changing the scientific
function. `SubstrateInteraction` implements the released default nearest-substrate attraction and clearance using
protein C-alpha distances and a substrate frame anchored to the fixed motif. The
released `rep_r_min=0` branch instead switches repulsion to an all-pairs
calculation. That is distinct scientific behavior and the complete family API must
retain it as an explicit RFdiffusion potential control rather than hide it behind
the default `SubstrateInteraction` meaning.

### Links

- **Code**: [ofoldx.models.rfdiffusion](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/rfdiffusion)
- **Paper**: [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8)
- **Developed by**: Joseph L. Watson, David Juergens, Nathaniel R. Bennett, et al.
- **Model type**: Denoising-diffusion backbone generator with a RoseTTAFold-style denoising trunk and an SE(3)-equivariant structure module
- **Original Repository**: [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)

## Training Details

RFdiffusion is trained as a denoising-diffusion generative model: a RoseTTAFold-style structure network is fine-tuned
to denoise noised residue frames back toward true protein structures, with self-conditioning across timesteps.

### Training Data

RFdiffusion fine-tunes a RoseTTAFold-style structure network as a denoising diffusion model over PDB protein
structures. OFoldX does not redistribute the training set.

### Training Procedure

Upstream RFdiffusion noises residue frames with Gaussian C-alpha translation noise and rotational Brownian motion,
trains denoising to true frames with self-conditioning, and uses checkpoint-specific inference configs. OFoldX converts
released checkpoints into strict-loaded native Models and persists them through the standard model repository format.
Its missing upstream trainer is provenance, not a blocker. Training
support is separate from the Model Hub registration gate.

## Real-checkpoint qualification

Production-weight parity is a manual, low-frequency release gate, never part of frequent CI. The maintained comparison
covers all seven official checkpoints—`Base_ckpt.pt`, `Complex_base_ckpt.pt`, `Complex_Fold_base_ckpt.pt`,
`InpaintSeq_ckpt.pt`, `InpaintSeq_Fold_ckpt.pt`, `ActiveSite_ckpt.pt`, and `Base_epoch8_ckpt.pt`—and all nine released
network outputs: the four geometric heads, sequence and resolved logits, backbone and torsion trajectories, and pLDDT
logits.

Only checkpoint fields read by the audited graph or sampler become `RFdiffusionConfig`. Stored but inert fields such as
`d_time_emb`, `use_motif_timestep`, `aa_decode_steps`, and `sequence_decode` are not turned into public configuration;
`freeze_track_motif`, which does change the structure track, is retained.

The bundled `rfdiffusion_schedule.npz` is a safe, value-preserving conversion of
`schedules/T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl` from
RosettaCommons/RFdiffusion revision `2d0c003df46b9db41d119321f15403dec3716cd9`. The source SHA-256 is
`cd79d3b77b3dc80d727ae0295f932b88cdfc0e5805e0ced29ce3c719518a1e23`; the packaged archive SHA-256 is
`89f97b26604e875e05907d465856d9b5ae5f9375482cc463dfa8d007d6e05db8`. It retains all five float64 arrays exactly.

The maintained [`scripts/parity/rfdiffusion_network.py`](../../../scripts/parity/rfdiffusion_network.py)
and companion RFdiffusion scripts reproduce the reference comparisons from code in
`~/reference` and checkpoints in `~/weights`; generated reports remain local.
Family-specific qualification lives in those maintained parity scripts. Frequent CI
covers only shared and operator invariants with tiny self-contained fixtures; it does
not run family design trajectories or download production checkpoints. RFdiffusion
is registered after passing low-frequency released-weight sampled-output acceptance,
default-Inductor full-graph compilation, and the standard
repository/Auto/Pipeline/top-level path.

## Citation

```bibtex
@article{watson2023denovo,
  author = {Watson, Joseph L. and Juergens, David and Bennett, Nathaniel R. and others},
  title = {De novo design of protein structure and function with RFdiffusion},
  journal = {Nature},
  volume = {620},
  number = {7976},
  pages = {1089--1100},
  year = {2023},
  doi = {10.1038/s41586-023-06415-8}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OFoldX supports your research, please cite the OFoldX project as follows:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for questions or comments on the model card.

Please contact the authors of the [RFdiffusion paper](https://doi.org/10.1038/s41586-023-06415-8) for questions or comments on the paper/model.

## License

The [official RFdiffusion license](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE)
is BSD 3-Clause and explicitly covers both the University of Washington source code
and the model weights referenced for download by the official README. A repository
containing converted RFdiffusion weights remains subject to that copyright notice,
its redistribution conditions, and its disclaimer; the frontmatter license describes
that model artifact.

The native OFoldX implementation is separate OFoldX project code licensed under
AGPL-3.0-or-later. OFoldX does not relicense the upstream RFdiffusion weights or
claim that the upstream authors endorsed this conversion.

The bundled schedule asset retains this upstream notice:

> BSD License
>
> Copyright (c) 2023 University of Washington. Developed at the Institute for
> Protein Design by Joseph Watson, David Juergens, Nathaniel Bennett, Brian Trippe
> and Jason Yim. This copyright and license covers both the source code and model
> weights referenced for download in the README file.
>
> Redistribution and use in source and binary forms, with or without modification,
> are permitted provided that the following conditions are met:
>
> Redistributions of source code must retain the above copyright notice, this list
> of conditions and the following disclaimer.
>
> Redistributions in binary form must reproduce the above copyright notice, this
> list of conditions and the following disclaimer in the documentation and/or other
> materials provided with the distribution.
>
> Neither the name of the University of Washington nor the names of its contributors
> may be used to endorse or promote products derived from this software without
> specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS IS”
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
> IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY
> DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
> OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
> OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
> OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
> THE POSSIBILITY OF SUCH DAMAGE.

## Limitations & intentional upstream divergences

- RFdiffusion generates backbones by diffusion; it does not design final sequence or
  pack side chains. The release inventory carries several checkpoints (`Base`,
  `Complex_base`, `ActiveSite`, `InpaintSeq`, fold-conditioned variants); publication
  must state which are qualified. The sampler is stochastic.
- The bundled schedule asset retains the upstream BSD notice.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** BSD-3-Clause; the license explicitly covers both the
  University of Washington source code and the model weights
  ([RFdiffusion LICENSE](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE),
  Copyright (c) 2023 University of Washington).
- **Redistribution status: GREEN.** RFdiffusion checkpoints are redistributable under
  BSD-3-Clause provided the copyright notice, conditions, and disclaimer are retained.
