---
language:
  - protein
  - rna
  - dna
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# Boltz-1

Boltz-1 is OFoldX's all-atom structure-prediction implementation for
biomolecular complexes, with a recycled representation trunk, EDM coordinate
sampling, confidence scoring, pocket conditioning, and optional Boltz-1x
sampling-time steering.

## Disclaimer

This is an unofficial OFoldX implementation of
[Boltz-1](https://github.com/jwohlwend/boltz). The upstream authors did not
write, review, or endorse this implementation or model card.

## Model Details

Boltz-1 predicts all-atom structures of biomolecular complexes. Atom and token
features are refined by an MSA module and recycled Pairformer trunk, an EDM
diffusion model generates coordinates, and a dedicated confidence trunk predicts
pLDDT, PAE, PDE, pTM, ipTM, and ranking scores. The same model supports pocket
conditioning and optional sampling-time steering.

### Variants

- **[oteam/boltz1](https://huggingface.co/oteam/boltz1)**: Boltz-1 structure and confidence model.

Boltz-1x physical-potential and Feynman-Kac steering are inference controls on
the same model rather than separate variants.

### Model Specification

| Pairformer Blocks | MSA Blocks | Denoiser Blocks | Confidence Blocks | Single Size | Pair Size | Trunk Passes | Sampling Steps | Num Parameters (M) | FLOPs (G) |
| ----------------- | ---------- | --------------- | ----------------- | ----------- | --------- | ------------ | -------------- | ------------------ | --------- |
| 48                | 4          | 24              | 48                | 384         | 128       | 4            | 200            | 592.008            | 1259.473  |

### Links

- **Code**: [ofoldx.models.boltz1](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz1)
- **Developed by**: Jeremy Wohlwend, Gabriele Corso, and the Boltz team
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with EDM coordinate diffusion
- **Original Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Structure Prediction

Load the concrete folding Pipeline:

```python
import ofoldx

pipeline = ofoldx.pipeline("fold", model="oteam/boltz1")
output = pipeline.fold("input.cif")
```

## Training Details

### Training Data

Refer to the official Boltz project for the training data and data-processing
procedure used by Boltz-1.

### Training Procedure

The released model uses an EDM coordinate-denoising objective together with a
trunk distogram objective and a confidence/ranking objective. Training noise is
drawn as
`sigma = sigma_data * exp(-1.2 + 1.5 * z)`, where `z ~ N(0, 1)`, and the
coordinate loss weight is
`(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`.

## Limitations

The built-in Runner objective covers the EDM coordinate-denoising term. It does
not reproduce the upstream joint schedule for distogram, confidence, and
ranking-head training, or the upstream data pipeline.

## Citation

Please cite the Boltz project described by the
[official repository](https://github.com/jwohlwend/boltz). If OFoldX supports
your research, also cite the OFoldX software:

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
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
