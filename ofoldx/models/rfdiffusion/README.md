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

Denoising-diffusion generation for de novo protein backbones, motif
scaffolding, inpainting, interfaces, and binder design.

## Disclaimer

This is an unofficial OFoldX conversion of
[De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8)
by Joseph L. Watson, David Juergens, Nathaniel R. Bennett, et al. The upstream
authors did not write, review, or endorse this implementation or model card.
The official repository is
[RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion).

## Model Details

RFdiffusion generates protein backbones by reversing diffusion over residue
frames. A RoseTTAFold-style denoiser embeds MSA, pair, template, sequence, and
timestep features; main and extra trunk blocks update those representations;
SE(3)-equivariant structure modules predict cleaner frames; and a native
sampler couples Euclidean C-alpha translation diffusion with IGSO(3) rotation
diffusion.

`ofoldx.rfdiffusion` registers `RfDiffusionModel` with
`RfDiffusionProcessor`. Coordinate-bearing, variable-topology results use the
concrete `RfDiffusionPipeline`, registered for `design`, rather than the generic
coordinate-free design result.

`model.forward(input)` is one differentiable denoising pass and returns all
released heads: single and pair representations, backbone coordinates, state
and torsion values, sequence logits, pLDDT, four pair-geometry logits,
resolved logits, and backbone/torsion trajectories. `model.design(input)` runs
reverse diffusion and returns `RfDiffusionDesignResult`.

Each `RfDiffusionDesignCandidate` contains the decoded backbone `Structure`, a
realized grouped `placement_map`, post-design `condition_evaluations`, residue
`plddt` on the 0-100 scale, and `mean_plddt`. Unconstrained generated residues
decode as CCD `UNK`; existing identities and explicit singleton identity
conditions are preserved. The Structure contains the N, CA, C, and O backbone
sites produced by RFdiffusion, plus retained or explicitly referenced context
where applicable.

Topology is declared with `RfDiffusionPlacements`, `GeneratedSegment`, and
`ExistingSpan`. Ranged generated segments may realize different lengths per
candidate. The `length` argument is the combined generated-length bound across
all generated segments, excluding receptor `ExistingSpan` rows.

The Processor and Pipeline expose de novo generation, motif scaffolding,
sequence inpainting, partial diffusion, fold conditioning, hotspot and residue
identity conditioning, symbolic bonds, fixed reference coordinates, distance
and coordination guidance, radius-of-gyration guidance, substrate interaction,
secondary-structure conditioning, cyclic closure, and cyclic, dihedral,
tetrahedral, octahedral, or icosahedral symmetry. Fold-conditioned checkpoints
accept one explicitly aligned `Template`.

The sampler exposes independent C-alpha and frame noise scales with `constant`
or `linear` schedules. Distinct final noise endpoints require a linear
schedule. `num_sampling_steps` rebuilds the active translation, rotation,
denoiser, and timestep-feature schedules together. The final public coordinates
come from the upstream raw `pX0`; optional motif alignment and fixed-row
restoration are applied while decoding the canonical result.

`model.set_backend("auto" | "torch" | "oops")` selects the RFdiffusion SE(3)
operator backend.

### Variants

The released checkpoint inventory is:

- `Base_ckpt.pt`
- `Complex_base_ckpt.pt`
- `Complex_Fold_base_ckpt.pt`
- `InpaintSeq_ckpt.pt`
- `InpaintSeq_Fold_ckpt.pt`
- `ActiveSite_ckpt.pt`
- `Base_epoch8_ckpt.pt`

All use the `ofoldx.rfdiffusion` architecture identity. The converter recovers
checkpoint-specific denoiser, diffuser, and preprocessing values, including
feature widths, self-conditioning, motif-track freezing, and fold-conditioning
channels.

### Model Specification

| Property                                    | Value                                  |
| ------------------------------------------- | -------------------------------------- |
| Architecture                                | `ofoldx.rfdiffusion`                   |
| Model                                       | `RfDiffusionModel`                     |
| Processor                                   | `RfDiffusionProcessor`                 |
| Pipeline                                    | `RfDiffusionPipeline`                  |
| Task                                        | `design` through the concrete Pipeline |
| Result                                      | `RfDiffusionDesignResult`              |
| Main trunk blocks / `num_main_blocks`       | 32                                     |
| Extra trunk blocks / `num_extra_blocks`     | 4                                      |
| Refinement blocks / `num_refinement_blocks` | 4                                      |
| MSA / full-MSA / pair sizes                 | 256 / 64 / 128                         |
| Diffusion timesteps / `num_timesteps`       | 50                                     |
| Translation schedule                        | linear, beta `0.01` to `0.07`          |
| Rotation process                            | IGSO(3), linear beta `1.5` to `2.5`    |

These dimensions describe the default/base configuration. A converted
checkpoint's stored configuration is authoritative for that variant.

OFoldX represents geometric guidance with typed canonical conditions rather
than upstream potential strings. `RfDiffusionCoordination` evaluates the
released smooth coordination function

`1 / (1 + x**6)`

directly, avoiding the upstream quotient's undefined `0 / 0` points without
changing the scientific function. Radius-of-gyration conditions compute each
target set around its own centroid, including separate monomer terms for dimer
compactness. `RfDiffusionSubstrateInteraction` implements nearest-substrate
attraction and clearance from a fixed substrate frame.

Point-group projection follows the released C/D/T/O/I matrix order, copies the
first unit's sequence across units, and inserts residue-index breaks before
network calls. Octahedral projection can additionally recenter units and place
them at `symmetry_radius_angstrom`.

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/rfdiffusion)
- [Official RFdiffusion source](https://github.com/RosettaCommons/RFdiffusion)
- [RFdiffusion paper](https://doi.org/10.1038/s41586-023-06415-8)
- [Official RFdiffusion license](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE)

## Usage

```python
import ofoldx
from ofoldx.data.placement import GeneratedSegment
from ofoldx.models.rfdiffusion import RfDiffusionPlacements

placement = RfDiffusionPlacements(
    placements={
        "design": {
            "backbone": GeneratedSegment("protein", 100, 100),
        }
    }
)
design_pipeline = ofoldx.pipeline(
    "design",
    model="./rfdiffusion-base",
    device="cuda",
)
result = design_pipeline.design(
    placement=placement,
    num_candidates=4,
    seed=7,
)
candidate = result.best("mean_plddt")
structure = candidate.structure
```

For motif scaffolding or binder design, pass a canonical `Structure` or
`MolecularSystem`, combine `ExistingSpan` and `GeneratedSegment` placements,
and supply the required typed conditions. One caller-controlled `seed` or
`generator` advances across every requested candidate.

## Checkpoints and Conversion

`convert_rfdiffusion_checkpoint` strict-loads one official checkpoint into an
in-memory Model. `save_rfdiffusion_checkpoint` writes `model.json`,
`processor.json`, safetensors, and the concrete RFdiffusion `pipeline.json`
marker.

```python
from ofoldx.models.rfdiffusion.convert import (
    convert_rfdiffusion_checkpoint,
    save_rfdiffusion_checkpoint,
)

model = convert_rfdiffusion_checkpoint(
    "Base_ckpt.pt",
    schedule_source="T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl",
)
save_rfdiffusion_checkpoint(
    "Base_ckpt.pt",
    "./rfdiffusion-base",
    schedule_source="T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl",
)
```

Conversion requires the released IGSO(3) lookup table as the original pickle or
a value-equivalent NPZ. The five float64 arrays are copied into persistent
`igso3_*` buffers in `model.safetensors`; inference does not read a separate
schedule file.

The value-preserving NPZ conversion came from
`schedules/T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl`
at RosettaCommons/RFdiffusion revision
`2d0c003df46b9db41d119321f15403dec3716cd9`. The source SHA-256 is
`cd79d3b77b3dc80d727ae0295f932b88cdfc0e5805e0ced29ce3c719518a1e23`;
the converted NPZ SHA-256 is
`89f97b26604e875e05907d465856d9b5ae5f9375482cc463dfa8d007d6e05db8`.
All five arrays retain their float64 values.

Only checkpoint fields consumed by the denoiser, diffuser, or preprocessing
graph become `RfDiffusionConfig`. Stored but inert fields are not promoted to
public configuration; `freeze_track_motif`, which changes the structure track,
is retained. Supplying `repo_id` requests an upload after local conversion;
conversion does not relicense the checkpoint or schedule.

## Training Details

### Training Data

RFdiffusion fine-tunes a RoseTTAFold-style structure network as a denoising
diffusion model over PDB protein structures. Refer to the RFdiffusion paper for
the authoritative upstream data construction. OFoldX does not redistribute the
training set.

### Training Procedure

The upstream procedure adds Gaussian C-alpha translation noise and rotational
Brownian motion to residue frames, then trains the network to denoise toward
the true frames with self-conditioning across timesteps. Checkpoints carry
their architecture, diffusion, and preprocessing settings.

OFoldX exposes the differentiable denoiser and converts released inference
checkpoints. It does not claim to reproduce the complete upstream trainer.

## Limitations

- RFdiffusion generates protein backbones. It does not design a final amino-acid
  sequence or pack side chains; unconstrained generated residues are `UNK`.
- Sampling is stochastic and results depend on the checkpoint, topology,
  conditions, seed, sampling schedule, dtype, and backend.
- One fold-conditioning `Template` is supported. Template evidence and
  secondary-structure conditioning require a fold-conditioned checkpoint, and
  template-conditioned generated segments must have fixed lengths.
- `RfDiffusionDesignResult` intentionally omits raw denoiser logits, diffusion
  timesteps, `pX0`, `x_{t-1}`, and backbone/torsion trajectories. Those remain
  tensor/sampler diagnostics rather than candidate values.
- The non-default upstream `rep_r_min=0` all-pairs substrate-repulsion branch is
  not represented by `RfDiffusionSubstrateInteraction`.
- OFoldX does not expose the upstream `t3` symmetry alias or the inert
  `model_only_neighbors` option as separate semantics.
- Strict checkpoint loading validates the converted key and shape surface, not
  complete numerical equivalence for every released checkpoint or stochastic
  design trajectory.

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

If OFoldX supports your research, also cite:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Questions about the
original model should be directed to the RFdiffusion authors.

## License

The official RFdiffusion BSD 3-Clause license explicitly covers both the
University of Washington source code and the model weights referenced for
download by the official README. Converted checkpoints and the converted
schedule table remain subject to its copyright notice, redistribution
conditions, and disclaimer. The native OFoldX implementation and documentation
are separate project code licensed under AGPL-3.0-or-later; OFoldX does not
relicense the upstream weights or claim upstream endorsement.

The upstream notice is:

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
> THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
> IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY
> DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
> OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
> OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
> OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
> THE POSSIBILITY OF SUCH DAMAGE.
