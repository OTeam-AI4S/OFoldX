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
  - Binding Affinity
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# Boltz-2

Boltz-2 is OFoldX's all-atom biomolecular structure-prediction and
protein-ligand binding-affinity family. It provides a folding pair and a
separate two-stage affinity pipeline built from structure/confidence and
affinity models.

## Disclaimer

This is an unofficial OFoldX implementation of
[Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
by Saro Passaro, Gabriele Corso, Jeremy Wohlwend, et al. The upstream authors did
not write, review, or endorse this implementation or model card.

The official Boltz-2 repository is
[jwohlwend/boltz](https://github.com/jwohlwend/boltz).

## Model Details

Boltz-2 predicts all-atom biomolecular structures and protein-ligand binding
affinity. Its structure model uses atom attention, template and MSA encoders, a
recycled Pairformer trunk, EDM coordinate diffusion, and confidence heads. The
affinity model adds pairwise affinity heads and is used in a two-stage workflow:
first predict and rank structures, then resample and score the selected complex.

### Variants

- **[oteam/boltz2](https://huggingface.co/oteam/boltz2)**: Structure prediction and confidence model.
- **[oteam/boltz2-affinity](https://huggingface.co/oteam/boltz2-affinity)**: Composite structure-and-affinity model for protein-ligand scoring.

### Model Specification

| Variant   | Pairformer Blocks | Template Blocks | Confidence Blocks | Affinity Blocks | Single Size | Pair Size | Trunk Passes | Num Parameters (M) | FLOPs (G) |
| --------- | ----------------- | --------------- | ----------------- | --------------- | ----------- | --------- | ------------ | ------------------ | --------- |
| Structure | 64                | 2               | 8                 | none            | 384         | 128       | 4            | 506.725            | 1615.889  |
| Affinity  | 64                | 0               | 8                 | 8 + 4           | 384         | 128       | 6            | 515.055            | 1806.891  |

FLOPs use PyTorch's `FlopCounterMode`, with each multiply-add counted as two
operations. They are measured for one batch of 128 residues (640 atom rows), one
query-only MSA row, and one trunk pass. The structure row includes one denoising
step; the affinity row includes the two-head affinity ensemble on supplied
coordinates.

### Links

- **Code**: [ofoldx.models.boltz2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz2)
- **Paper**: [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
- **Developed by**: Saro Passaro, Gabriele Corso, Jeremy Wohlwend, et al.
- **Model type**: Pairformer-based all-atom structure predictor with two-stage protein-ligand affinity scoring
- **Original Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)

## Usage

Install OFoldX:

```bash
pip install ofoldx
```

### Structure Prediction

```python
import ofoldx

output = ofoldx.fold("complex.cif", model="oteam/boltz2")
```

### Binding Affinity Prediction

```python
import ofoldx

output = ofoldx.score("complex.cif", model="oteam/boltz2-affinity")
```

## Training Details

### Training Data

The Boltz-2 report describes structure data from PDB entries before 2023-06-01,
experimental NMR collections, molecular-dynamics trajectories, AlphaFold 2
monomer distillation, and Boltz-1 complex distillation. It describes affinity
data from PubChem, ChEMBL, BindingDB, and related filtered assay sources, with
decoy generation and structural-confidence filtering. OFoldX does not
redistribute these datasets.

### Training Procedure

Training is staged across structure, confidence, and affinity
objectives. The report describes detached-trunk absolute and difference Huber
losses for affinity values and a focal binary objective for binder
classification.

## Limitations

- Structure and affinity sampling are stochastic. The two-stage result depends
  on both first-pass ranking and independent second-pass maximum-iPTM selection.
- Affinity scoring requires a protein receptor and one selected, non-copied
  `Molecule` instance. The released crop supports at most 256 tokens, 200 polymer
  tokens, 2048 atoms, and 128 ligand atoms after hydrogen removal.
- Molecular-weight calibration is computed from the final canonical ligand
  graph. Raw SMILES should be standardized before constructing the input.
- Native pTM reductions use the dtype-aware `torch.finfo(dtype).tiny` denominator
  for empty masks rather than a fixed cross-family epsilon.

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

If OFoldX supports your research, also cite the OFoldX software:

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
upstream model should be directed to the authors of the
[Boltz-2 paper](https://doi.org/10.1101/2025.06.14.659707).

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
