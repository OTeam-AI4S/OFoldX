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

# AlphaFold 3

Pre-trained model for all-atom structure prediction of proteins, nucleic acids, ligands, ions, modified residues, and glycan components.

## Disclaimer

This is an UNOFFICIAL implementation of the [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) by Josh Abramson, et al.

The OFFICIAL repository of AlphaFold 3 is at [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3).

**The team releasing AlphaFold 3 did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

AlphaFold 3 combines atom, token, MSA, template, bond, and relative-position features in a recycled Pairformer
trunk. A diffusion model generates atomic coordinates, while distogram and confidence heads predict pair distances,
pLDDT, PAE, PDE, pTM, and ipTM. The pipeline combines confidence, structure-derived disorder, and clashes into ranking
scores.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **AlphaFold 3**: Convert parameters obtained directly from Google DeepMind into a local OFoldX repository.

Google DeepMind's parameter terms do not grant third-party redistribution. Obtain parameters directly from Google
DeepMind; locally converted parameters retain their upstream
[parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

### Model Specification

| Pairformer Blocks | Template Blocks | Denoiser Blocks | Single Size | Pair Size | Recycling Steps | Sampling Steps | Num Parameters (M) |
| ----------------- | --------------- | --------------- | ----------- | --------- | --------------- | -------------- | ------------------ |
| 48                | 2               | 24              | 384         | 128       | 11              | 200            | 368.403            |

### Links

- **Code**: [ofoldx.models.alphafold3](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold3)
- **Paper**:
  [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
- **Developed by**: Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf
  Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, Sebastian W. Bodenstein, David A. Evans, Chia-Chun
  Hung, Michael O'Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvilė Žemgulytė, Eirini Arvaniti, Charles
  Beattie, Ottavia Bertolli, Alex Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I. Cowen-Rivers, Andrew Cowie,
  Michael Figurnov, Fabian B. Fuchs, Hannah Gladman, Rishub Jain, Yousuf A. Khan, Caroline M. R. Low, Kuba Perlin, Anna
  Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula, Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen
  D. Zhong, Michal Zielinski, Augustin Žídek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, and John M.
  Jumper
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with diffusion coordinate generation
- **Original Repository**: [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Structure Prediction

You can use this model directly with a pipeline for all-atom structure prediction with locally converted parameters:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="./alphafold3")
result = pipeline(structure, seed=7)
```

Supply the molecular target and any explicit MSA or template evidence. Predictions use a new coordinate frame;
unpredicted atoms, including hydrogens, remain without coordinates. Confidence, clash, disorder, and accessibility
values are ranking descriptors, rather than measurements of chemical validity or exact solvent accessibility.

### Checkpoint Conversion

Obtain the official parameter archive directly from Google DeepMind, then create the local repository used above:

```bash
python -m ofoldx.models.alphafold3.convert /path/to/af3.bin.zst ./alphafold3
```

The converter maps the checkpoint's 31-class features onto OFoldX's 72 canonical classes. It does not change the
upstream parameter terms.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution and composition.

## Training Details

AlphaFold 3 is trained with diffusion coordinate denoising and auxiliary distogram and confidence objectives.

### Training Data

The AlphaFold 3 paper and supplementary information describe the biomolecular structure, sequence, and chemistry
resources used for upstream training. OFoldX does not redistribute the training corpus.

### Training Procedure

AlphaFold 3 is trained end to end with diffusion coordinate, distogram, and confidence objectives.

OFoldX exposes EDM-weighted coordinate denoising with log-normal noise and centered rigid augmentation over resolved
atoms. It trains the trunk and denoiser, but has no built-in distogram or confidence loss and does not reproduce the
upstream optimizer, curriculum, or database-search pipeline. MSA pairing consumes explicitly supplied pools and does
not search databases.

## Citation

```bibtex
@article{abramson2024accurate,
  author = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
  title = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  journal = {Nature},
  volume = {630},
  pages = {493--500},
  year = {2024},
  doi = {10.1038/s41586-024-07487-w}
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

Please contact the authors of the [AlphaFold 3 paper](https://doi.org/10.1038/s41586-024-07487-w) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
