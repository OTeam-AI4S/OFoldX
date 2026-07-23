---
language: protein
tags:
  - Biology
  - Protein
  - Protein Design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# PxDesign

Native PyTorch implementation of the released PXDesign-d v0.1 backbone
generator for target-conditioned and de novo protein design.

## Disclaimer

This is an unofficial OFoldX implementation of
[PXDesign](https://github.com/bytedance/PXDesign). The upstream authors did not
write, review, or endorse this implementation or model card.

## Model Details

PXDesign-d combines a condition embedder, coordinate denoiser, and EDM sampler.
The processor lowers a canonical target structure, one protein placement, and
optional hotspot labels. Generated residues retain an explicit unknown sequence
identity while the model predicts their backbone coordinates.

PXDesign-d is distinct from the larger PXDesign suite. The suite additionally
composes ProteinMPNN sequence design, AlphaFold 2 initial-guess evaluation,
optional Protenix evaluation, and PXDesignBench ranking. Those downstream stages
are not hidden inside the generator model.

### Variants

The released family contains one PXDesign-d v0.1 checkpoint,
`pxdesign_v0.1.0.pt`.

### Model Specification

| Property               | Value                  |
| ---------------------- | ---------------------- |
| Architecture           | `ofoldx.pxdesign`      |
| Model                  | `PxDesignModel`        |
| Processor              | `PxDesignProcessor`    |
| Pipeline               | `PxDesignPipeline`     |
| Task                   | `design`               |
| Result                 | `PxDesignDesignResult` |
| Checkpoint tensors     | 732                    |
| Default sampling steps | 400                    |

The differentiable `forward` method evaluates one denoising step from explicit
noisy coordinates and a noise level. `sample` runs the released EDM loop, and
`design` decodes its coordinate samples into canonical structures.

### Links

- [Official source](https://github.com/bytedance/PXDesign)
- [Audited source revision](https://github.com/bytedance/PXDesign/tree/f788441313c84c3074fe9596ac2433f96b15c763)
- [Official checkpoint](https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt)

## Usage

Convert the official checkpoint first, then load the resulting repository through
the task API:

```python
import ofoldx

pipe = ofoldx.pipeline("design", model="./pxdesign-v0.1", device="cuda")
result = pipe.design(
    "target.cif",
    placement=placement,
    conditions=hotspots,
    num_candidates=4,
    seed=7,
)
best_structure = result.candidates[0].structure
```

`placement` is one canonical `PlacementSpec` containing the generated protein
segment. `conditions` is a sequence of PXDesign `Hotspot` labels. Passing
`input=None` selects the released generation-only path.

## Checkpoints and Conversion

```bash
python -m ofoldx.models.pxdesign.convert \
  pxdesign_v0.1.0.pt \
  ./pxdesign-v0.1
```

Conversion translates every upstream key into the native OFoldX state tree,
strict-loads the 732 tensors, and writes the Model, Processor, weights, and
concrete Pipeline marker. `--repo-id oteam/pxdesign-v0.1` additionally uploads
the converted repository; `--private` requests a private Hub repository.

## Training Details

### Training Data

The released PXDesign-d inference source does not provide a complete,
reproducible training-data specification.

### Training Procedure

The released source is inference-only and does not expose a reference training
objective. OFoldX therefore provides the differentiable denoiser but does not
invent a PXDesign training loss. Two distogram heads that are commented out in
the upstream source are not part of the released model graph.

## Limitations

- The full PXDesign suite is not implemented because its AlphaFold 2
  initial-guess stage and PXDesignBench ranking and geometry code are not
  available in OFoldX.
- Standard protein, RNA, DNA, and non-polymer target components are supported.
  MSE is canonicalized to MET as in the released parser; other modified polymer
  components require a truthful canonical-parent transform and are rejected.
- Target placements currently retain complete polymer chains. The released
  target-cropping path is not represented as a partial structure.
- PXDesign-d does not consume MSA evidence. The full suite passes target MSAs to
  downstream folding and filtering models.
- OFoldX uses a caller-owned `torch.Generator`. Equal integer seeds across the
  upstream mixed PyTorch/SciPy RNG loop and OFoldX do not imply equal trajectories.

Confirmed upstream defects and OFoldX behavior are recorded in
[`UPSTREAM_BUGS.md`](../../../UPSTREAM_BUGS.md).

## Citation

Please cite the PXDesign work and repository described by the
[upstream project](https://github.com/bytedance/PXDesign).

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream code and weights:** Apache-2.0; see the
  [PXDesign license](https://github.com/bytedance/PXDesign/blob/main/LICENSE).
- The released `pxdesign_v0.1.0.pt` checkpoint may be redistributed with the
  upstream Apache-2.0 notice preserved.
