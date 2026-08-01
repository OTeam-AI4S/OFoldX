---
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

Pre-trained model for all-atom structure prediction of biomolecular complexes.

## Disclaimer

This is an UNOFFICIAL implementation of [Boltz-1: Democratizing Biomolecular Interaction Modeling](https://doi.org/10.1101/2024.11.19.624167) by Jeremy Wohlwend, Gabriele Corso, et al.

The OFFICIAL repository of Boltz-1 is at [jwohlwend/boltz](https://github.com/jwohlwend/boltz).

**The team releasing Boltz-1 did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

Boltz-1 refines atom and token features with an MSA module and a recycled Pairformer trunk. An EDM diffusion
model generates coordinates, and a dedicated confidence trunk predicts pLDDT, PAE, PDE, pTM, ipTM, and ranking scores.
The model also supports pocket conditioning and Boltz-1x sampling-time steering.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/boltz1](https://huggingface.co/oteam/boltz1)**: Biomolecular structure and confidence prediction.

Boltz-1x physical-potential and Feynman-Kac steering are inference controls on this model, rather than separate weights.

### Model Specification

| Pairformer Blocks | MSA Blocks | Denoiser Blocks | Confidence Blocks | Single Size | Pair Size | Trunk Passes | Sampling Steps | Num Parameters (M) |
| ----------------- | ---------- | --------------- | ----------------- | ----------- | --------- | ------------ | -------------- | ------------------ |
| 48                | 4          | 24              | 48                | 384         | 128       | 4            | 200            | 592.163            |

### Links

- **Code**: [ofoldx.models.boltz1](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz1)
- **Paper**: [Boltz-1: Democratizing Biomolecular Interaction Modeling](https://doi.org/10.1101/2024.11.19.624167)
- **Developed by**: Jeremy Wohlwend, Gabriele Corso, Saro Passaro, Noah Getz, Mateo Reveiz, Ken Leidal,
  Wojtek Swiderski, Liam Atkinson, Tally Portnoi, Itamar Chinn, Jacob Silterra, Tommi Jaakkola, and Regina Barzilay
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with EDM coordinate diffusion
- **Original Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Structure Prediction

You can use this model directly with a pipeline for all-atom structure prediction:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/boltz1")
result = pipeline(structure, seed=7)
```

The result contains predicted structures and confidence values. Reference conformers and observed coordinates remain
separate; preparation can build reference geometry for atoms without experimental observations.

### Downstream Use

#### Physical Guidance

Boltz-1x steering uses physical potentials during sampling. Active geometry and van der Waals pairs require known atomic
numbers and supported tabulated radii; unsupported elements raise errors. A soft element distribution does not determine
one radius. These empirical potentials guide sampling and do not establish chemical identity or validity.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution and composition.

## Training Details

Boltz-1 is trained for biomolecular structure prediction with coordinate denoising and confidence objectives.

### Training Data

The Boltz-1 structure model uses PDB structures and OpenFold self-distillation data. The upstream project provides
the [preprocessed data and processing pipeline](https://github.com/jwohlwend/boltz/blob/main/docs/training.md).

### Training Procedure

Boltz-1 uses EDM coordinate denoising together with trunk distogram and confidence/ranking objectives. Noise follows
`sigma = sigma_data * exp(-1.2 + 1.5 * z)` for `z ~ N(0, 1)`; coordinate errors are weighted by
`(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`.

OFoldX exposes the EDM coordinate-denoising term. It does not reproduce the joint distogram, confidence, and ranking
training schedule or the upstream data pipeline.

## Citation

```bibtex
@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and others},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
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

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for any questions or comments on the model implementation or model card.

Please contact the authors of the [Boltz-1 project](https://github.com/jwohlwend/boltz) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
