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

Native PyTorch implementation of the PXDesign-d backbone generator for
target-conditioned and de novo protein design.

## Disclaimer

This is an unofficial OFoldX implementation of
[PXDesign](https://github.com/bytedance/PXDesign). The upstream authors did not
write, review, or endorse this implementation or model card.

## Model Details

PXDesign-d is an all-atom diffusion model for de novo and target-conditioned
protein backbone generation. It embeds target geometry, residue identity,
confidence, and hotspot conditions, then applies token- and atom-level attention
inside an EDM coordinate denoiser. The generated backbone can subsequently be
assigned a sequence and evaluated by separate sequence-design and folding
models.

### Variants

- **[oteam/pxdesign](https://huggingface.co/oteam/pxdesign)**: PXDesign-d for de novo and target-conditioned protein backbone generation.

### Model Specification

| Single Size | Pair Size | Token Denoiser Blocks | Atom Encoder Blocks | Atom Decoder Blocks | Num Parameters (M) | FLOPs (G) |
| ----------- | --------- | --------------------- | ------------------- | ------------------- | ------------------ | --------- |
| 384         | 128       | 16                    | 4                   | 4                   | 139.011            | 49.888    |

### Links

- **Code**: [ofoldx.models.pxdesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/pxdesign)
- **Developed by**: ByteDance Research
- **Model type**: Target-conditioned all-atom protein backbone diffusion model
- **Original Repository**: [bytedance/PXDesign](https://github.com/bytedance/PXDesign)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Backbone Design

Load the PXDesign design pipeline:

```python
import ofoldx
from ofoldx.models.pxdesign import from_binder

pipeline = ofoldx.pipeline("design", model="oteam/pxdesign")
output = pipeline.design(None, placement=from_binder(100))
```

## Training Details

### Training Data

PXDesign-d is trained on protein structures for unconditional and
target-conditioned backbone generation.

### Training Procedure

PXDesign-d uses an EDM coordinate-denoising objective. OFoldX exposes the
differentiable one-step denoiser for training and fine-tuning.

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
