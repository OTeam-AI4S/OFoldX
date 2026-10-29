---
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
  - Binding Affinity
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# Boltz-2

Pre-trained model for all-atom structure prediction and protein–ligand binding affinity prediction.

## Disclaimer

This is an UNOFFICIAL implementation of the [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707) by Saro Passaro, et al.

The OFFICIAL repository of Boltz-2 is at [jwohlwend/boltz](https://github.com/jwohlwend/boltz).

**The team releasing Boltz-2 did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

Boltz-2 uses atom attention, template and MSA encoders, a recycled Pairformer trunk, and EDM coordinate diffusion
for structure prediction. A separate affinity model adds pairwise affinity heads. Its two-stage affinity workflow
first predicts and ranks structures, then resamples and scores the selected complex.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/boltz2](https://huggingface.co/oteam/boltz2)**: Biomolecular structure and confidence prediction.
- **[oteam/boltz2-affinity](https://huggingface.co/oteam/boltz2-affinity)**: A composite structure-and-affinity workflow for protein-ligand complexes.

### Model Specification

| Variant         | Pairformer Blocks | Template Blocks | Confidence Blocks | Affinity Blocks | Single Size | Pair Size | Trunk Passes | Num Parameters (M) |
| --------------- | ----------------- | --------------- | ----------------- | --------------- | ----------- | --------- | ------------ | ------------------ |
| Structure       | 64                | 2               | 8                 | none            | 384         | 128       | 4            | 506.762            |
| Affinity scorer | 64                | 0               | 8                 | 8 + 4           | 384         | 128       | 6            | 515.088            |

### Links

- **Code**: [ofoldx.models.boltz2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz2)
- **Paper**:
  [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
- **Developed by**: Saro Passaro, Gabriele Corso, Jeremy Wohlwend, Mateo Reveiz, Stephan Thaler, Vignesh Ram Somnath,
  Noah Getz, Tally Portnoi, Julien Roy, Hannes Stark, David Kwabi-Addo, Dominique Beaini, Tommi Jaakkola, and Regina
  Barzilay
- **Model type**: Pairformer-based all-atom structure predictor with two-stage protein-ligand affinity scoring
- **Original Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Structure Prediction

You can use this model directly with a pipeline for all-atom structure prediction:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("complex.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/boltz2")
result = pipeline(structure, seed=7)
```

The result contains predicted structures and confidence values. `designed_positions` selects already-designed component
occurrences for design-focused confidence; the exact selection and the molecular scope expanded through covalent
connections remain distinct.

#### Binding Affinity Prediction

You can use this model directly with a pipeline for binding affinity prediction on a folded complex:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("complex.cif")
pipeline = ofoldx.pipeline("score", model="oteam/boltz2-affinity")
result = pipeline(structure, seed=7)
```

The pipeline selects the highest-confidence folded structure, then resamples and scores it with its affinity model.
The result retains the structure and affinity predictions.

### Downstream Use

#### Physical Guidance

Boltz-2 steering uses physical potentials during sampling. Active geometry and van der Waals pairs require known atomic
numbers and supported tabulated radii; unsupported elements raise errors. A soft element distribution does not determine
one radius. These empirical potentials guide sampling and do not establish chemical identity or validity.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution and composition.

## Training Details

Boltz-2 is trained for biomolecular structure prediction and protein–ligand binding affinity prediction.

### Training Data

The Boltz-2 report describes PDB structures before 2023-06-01, NMR data, molecular-dynamics trajectories, AlphaFold 2
monomer distillation, and Boltz-1 complex distillation. Affinity data comes from PubChem, ChEMBL, BindingDB, and filtered
assay sources with decoy generation and structural-confidence filtering. OFoldX does not redistribute these datasets.

### Training Procedure

Boltz-2 stages structure, confidence, and affinity training. The affinity objectives include detached-trunk absolute
and difference Huber losses, together with a focal binary objective for binder classification.

OFoldX exposes the structure coordinate-denoising objective. It does not reproduce staged confidence and affinity
training or the upstream data and decoy-generation pipelines.

## Citation

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
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

Please contact the authors of the [Boltz-2 paper](https://doi.org/10.1101/2025.06.14.659707) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
