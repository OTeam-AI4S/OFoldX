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

# PXDesign

Native PyTorch implementation of PXDesign-d and the released PXDesign Preview
and Extended target--binder design operations.

## Disclaimer

This is an unofficial OFoldX implementation of
[PXDesign](https://github.com/bytedance/PXDesign). The upstream authors did not
write, review, or endorse this implementation or model card.

## Model Details

PXDesign-d is an all-atom diffusion model for de novo and target-conditioned
protein backbone generation. The complete released operation assigns one
ProteinMPNN sequence per backbone, evaluates the complex and unbound binder with
AF2, and, in Extended mode, routes evaluation between Protenix base and the
structure-conditioned Protenix mini checkpoint before filtering and ranking
every candidate.

### Variants

- **PXDesign backbone**: Independently callable PXDesign-d generation through
  `PxDesignBackbonePipeline`.
- **PXDesign Preview**: 100 backbones by default, ProteinMPNN `v_48_020`,
  AF2-IG complex evaluation, ordinary AF2 unbound-binder evaluation, released
  filters, and stable ranking.
- **PXDesign Extended**: 500 backbones by default plus full-target Protenix
  routing, two-step screening, basic-pass 20-step reruns, and Extended
  filtering and ranking.

### Model Specification

| Single Size | Pair Size | Token Denoiser Blocks | Atom Encoder Blocks | Atom Decoder Blocks | Num Parameters (M) | FLOPs (G) |
| ----------- | --------- | --------------------- | ------------------- | ------------------- | ------------------ | --------- |
| 384         | 128       | 16                    | 4                   | 4                   | 139.011            | 49.888    |

### Links

- **Code**: [ofoldx.models.pxdesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/pxdesign)
- **Developed by**: ByteDance Research
- **Model type**: Target-conditioned all-atom protein backbone diffusion model
- **Original Repository**: [bytedance/PXDesign](https://github.com/bytedance/PXDesign)
- **Released Evaluation Composition**: [bytedance/PXDesignBench](https://github.com/bytedance/PXDesignBench)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Complete Design

Load a six-pair PXDesign repository and run Preview:

```python
from ofoldx.models.pxdesign import PxDesignPipeline, from_binder

pipeline = PxDesignPipeline.from_repository("/path/to/pxdesign")
result = pipeline.design(
    target_structure,
    placement=from_binder(100, [("target", [(0, 150)])]),
)
```

Extended accepts the full-target MSA and retains all 500 candidates unless a
different count is requested:

```python
result = pipeline.design(
    target_structure,
    placement=from_binder(100, [("target", [(0, 150)])]),
    msa=target_msa,
    mode="extended",
)
best = result.ranked_candidates[0]
```

The root pair of a complete repository is also independently loadable as a
`PxDesignBackbonePipeline`. It performs only PXDesign-d coordinate generation;
the concrete `PxDesignPipeline` owns the released six-pair composition.

## Checkpoints and Conversion

The independent backbone repository is written from the released PXDesign-d
checkpoint:

```bash
python -m ofoldx.models.pxdesign.convert backbone \
  pxdesign_v0.1.0.pt /path/to/pxdesign-backbone
```

The complete repository converts the fixed released composition directly. The
single official AlphaFold parameter source supplies `model_1_ptm` to both the
initial-guess complex graph and the ordinary binder-monomer graph:

```bash
python -m ofoldx.models.pxdesign.convert complete \
  pxdesign_v0.1.0.pt \
  v_48_020.pt \
  /path/to/alphafold-params \
  protenix_base_default_v0.5.0.pt \
  protenix_mini_tmpl_v0.5.0.pt \
  /path/to/pxdesign
```

The resulting repository stores the PXDesign-d pair at its root and the five
fixed child pairs under `proteinmpnn/`, `af2_complex/`, `af2_binder/`,
`protenix/`, and `protenix_structure/`. Preview uses the first four pairs.
Extended also uses both Protenix pairs, reruns every basic screening pass
(including strict passes) for 20 diffusion transitions, and never selects a
two-step screening structure as the returned structure. Target routing and AF2
globally refit after identical-sequence chain assignment; Protenix retains its
anchored-assignment RMSD. All three pair only CA coordinates resolved in both
Structures by canonical component index without renaming the Structures kept
in the result.

## Training Details

### Training Data

PXDesign-d is trained on protein structures for unconditional and
target-conditioned backbone generation.

### Training Procedure

PXDesign-d uses an EDM coordinate-denoising objective. OFoldX exposes the
differentiable one-step denoiser for training and fine-tuning.

## Limitations

PXDesign-d has no built-in Runner objective; the exposed one-step denoiser does
not reproduce its full training pipeline. Preview and Extended are fixed
multi-model inference compositions and require all converted child repositories
listed above.

## Citation

Please cite the PXDesign work and repository described by the
[upstream project](https://github.com/bytedance/PXDesign).

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
