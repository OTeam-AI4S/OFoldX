---
language:
  - protein
tags:
  - Biology
  - Protein Design
  - Biomolecular Structure
  - Diffusion
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ODesign

Native PyTorch implementation of the ODesign all-atom biomolecular interaction
design model and its released protein sequence-assignment composition.

## Disclaimer

This is an unofficial OFoldX implementation of
[ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).
The upstream authors did not write, review, or endorse this implementation or
model card.

## Model Details

ODesign combines an atom encoder, MSA-conditioned recycled Pairformer trunk,
distance-constraint embedder, EDM coordinate denoiser, distogram head, and
token-bond head. `forward` evaluates one differentiable denoising step from
explicit noisy coordinates and a noise level. `sample` runs coordinate
generation and preserves the pairwise heads.

The released protein operation is a composition under one `design` verb.
`ODesignModel` generates S seed groups by M diffusion samples, then
`ODesignPipeline` uses the separately registered ProteinMPNN or LigandMPNN
pipeline to assign K sequence variants to each backbone. It returns the complete
M x S x K grid as `ODesignDesignResult`.

Observed coordinates provide geometric context and are not fixed implicitly.
`partial_diffusion_snr` explicitly starts from observed coordinates. Hard
`ODesignReferenceCoordinates` conditions pin selected atoms; soft references
remain constraint evidence.

### Variants

The released `odesign_base_prot_flex.pt` and
`odesign_base_prot_rigid.pt` checkpoints use the same model graph and state
tree. Their repositories differ through the Processor's `receptor_mode`:

- `flex` regenerates receptor coordinates unless atoms are explicitly pinned.
- `rigid` fixes observed receptor heavy atoms during sampling.

### Model Specification

| Property            | Value                       |
| ------------------- | --------------------------- |
| Architecture        | `ofoldx.odesign`            |
| Model               | `ODesignModel`              |
| Processor           | `ODesignProcessor`          |
| Pipeline            | `ODesignPipeline`           |
| Task                | `design`                    |
| Result              | `ODesignDesignResult`       |
| Backbone variants   | protein flex, protein rigid |
| Sequence assignment | ProteinMPNN or LigandMPNN   |

For protein sequence assignment, `temperature` maps to the released inverse
folding temperature and `num_sequences` maps to top-k. LigandMPNN is selected
when the generated assembly carries non-protein atom context; otherwise
ProteinMPNN is used.

### Links

- [Official source](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)
- [Technical report](https://arxiv.org/abs/2510.22304)

## Usage

```python
import ofoldx

pipe = ofoldx.pipeline("design", model="./odesign-prot-flex", device="cuda")
result = pipe.design(
    "target.cif",
    placement=placement,
    num_seeds=2,
    num_backbones=4,
    num_sequences=3,
    seed=7,
)
candidate = result.candidates[0]
```

`placement` is an optional `ODesignPlacements` value describing generated
segments and per-chain lengths. Conditions may include hotspots, symbolic bonds,
and reference coordinates. All samples in one invocation share the topology
realized from ranged segments.

## Checkpoints and Conversion

The composite repository stores the ODesign backbone pair at its root and embeds
the converted ProteinMPNN and LigandMPNN designer repositories at their fixed
component paths:

```bash
python -m ofoldx.models.odesign.convert \
  odesign_base_prot_flex.pt \
  ./odesign-prot-flex \
  --protein-designer ./proteinmpnn \
  --ligand-designer ./ligandmpnn
```

The converter infers `flex` or `rigid` from the official filename unless
`--receptor-mode` is supplied. It translates the upstream state tree, strict
loads the model, writes the complete composite Pipeline repository, and can
upload it with `--repo-id oteam/odesign-prot-flex`. `--private` requests a
private Hub repository.

## Training Details

### Training Data

Refer to the [ODesign technical report](https://arxiv.org/abs/2510.22304) for the
upstream training corpus and data construction.

### Training Procedure

OFoldX preserves the released inference graph and exposes the differentiable
one-step denoiser. It does not add an OFoldX-specific training objective to the
model. Rigid and flexible receptor behavior is a conditioning difference rather
than a separate architecture.

## Limitations

- The released de novo ligand CLI creates atoms without sufficient element,
  reference-component, and sanitized bond identity to construct a truthful
  canonical molecule. That generation mode is not exposed.
- De novo nucleic-acid generation and non-protein OInvFold beam decoding are not
  included in this protein composition.
- Existing canonical molecules are supported as structural context.
- The core model has no per-sample confidence head. Distogram and token-bond
  logits remain raw outputs rather than fabricated ranking scores.
- Numerical parity applies to the converted checkpoint and evaluated settings;
  stochastic trajectories additionally depend on the supplied generator.

Confirmed upstream defects and OFoldX behavior are recorded in
[`UPSTREAM_BUGS.md`](../../../UPSTREAM_BUGS.md).

## Citation

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and released parameters:** Apache-2.0; preserve the
  upstream copyright and license notices when redistributing checkpoints.
