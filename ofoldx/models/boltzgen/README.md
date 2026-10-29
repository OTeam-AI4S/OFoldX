---
tags:
  - Biology
  - Protein
  - Protein Design
  - Inverse Folding
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# BoltzGen

Pre-trained model for protein and peptide binder design through joint sequence and structure generation, optional sequence redesign, and refolding-based evaluation.

## Disclaimer

This is an UNOFFICIAL implementation of [BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494) by Hannes Stark, Felix Faltings, et al.

The OFFICIAL repository of BoltzGen is at [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen).

**The team releasing BoltzGen did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

BoltzGen is an all-atom diffusion model for biomolecular binder design. A recycled Pairformer combines atom, token,
MSA, template, contact, binding-site and secondary-structure features before an EDM denoiser generates sequence and
atomic coordinates together. A separate inverse-folding model can redesign the sequence on the generated backbone.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/boltzgen](https://huggingface.co/oteam/boltzgen)**: The complete binder design and evaluation workflow.
- **[oteam/boltzgen-diverse](https://huggingface.co/oteam/boltzgen-diverse)** and
  **[oteam/boltzgen-adherence](https://huggingface.co/oteam/boltzgen-adherence)**: Independently callable structure generators.
- **[oteam/boltzgen-structure-small](https://huggingface.co/oteam/boltzgen-structure-small)**: A Miniformer structure generator.
- **[oteam/boltzgen-inversefolding](https://huggingface.co/oteam/boltzgen-inversefolding)**: The BoltzIF model for sequence assignment on fixed backbones.

### Model Specification

| Variant                 | Representation Size | Trunk                                | Denoiser / Decoder Blocks | Num Parameters (M) |
| ----------------------- | ------------------- | ------------------------------------ | ------------------------- | ------------------ |
| Diverse / Adherence     | 384 / 128           | 64 Pairformer + 4 MSA Blocks         | 24 token Blocks           | 482.213            |
| Structure-trained small | 384 / 128           | 12 Miniformer + 3 MSA Blocks         | 8 token Blocks            | 136.886            |
| Inverse folding         | 128 / 128           | 6 graph encoder Blocks, 30 neighbors | 3 graph decoder Blocks    | 3.069              |

### Links

- **Code**: [ofoldx.models.boltzgen](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltzgen)
- **Paper**: [BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494)
- **Developed by**: Hannes Stark, Felix Faltings, MinGyu Choi, Yuxin Xie, Eunsu Hur, Timothy O'Donnell, Anton Bushuiev,
  Talip Uçar, Saro Passaro, Weian Mao, Mateo Reveiz, Roman Bushuiev, Tally Portnoi, Tomáš Pluskal, Josef Sivic, Karsten
  Kreis, Arash Vahdat, Shamayeeta Ray, Jonathan T. Goldstein, Andrew Savinov, Jacob A. Hambalek, Anshika Gupta, Diego A.
  Taquiri-Diaz, Yaotian Zhang, Samuel J. Snyder, A. Katherine Hatstat, Angelika Arada, Nam Hyeong Kim, Haoyu Fan, Ethel
  Tackie-Yarboi, Dylan Boselli, Lee Schnaider, Chang C. Liu, Gene-Wei Li, Denes Hnisz, David M. Sabatini, William F.
  DeGrado, Jeremy Wohlwend, Gabriele Corso, Regina Barzilay, and Tommi Jaakkola
- **Model type**: Constraint-guided all-atom biomolecular diffusion and inverse folding
- **Original Repository**: [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

### Direct Use

#### Biomolecular Design

You can use this model directly with a pipeline for binder design with the `protein-anything` protocol:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("target.cif")
pipeline = ofoldx.pipeline("design", model="oteam/boltzgen")
result = pipeline(structure, protocol="protein-anything", total_len=120, num_candidates=20, seed=7)
best = result.ranked_candidates[0]
```

The result contains the generated candidates, their evaluation scores, and ranked and diversity-selected views.
Screening uses the atoms present in each structure. For single-chain redesign, interface scores do not describe an
actual interface.

The other released protocols are `peptide-anything`, `protein-small_molecule`, `nanobody-anything`,
`antibody-anything` and `protein-redesign`. The default inverse folder is SolubleMPNN, or BoltzIF for
`protein-small_molecule`. Use `inverse_fold_model="boltzif"` to select BoltzIF or `inverse_fold_model=None` to retain
the generated sequence and coordinates without inverse folding.

RNA and DNA can be fixed context but cannot be generated segments. When inverse folding changes a residue's identity,
the new side-chain coordinates remain unavailable until a later structure prediction.

### Downstream Use

#### Partial Workflows

Use `design_backbones()` for joint sequence and structure generation or `inverse_fold()` for sequence redesign without evaluation.
These operations return candidates in generation order; only the complete `design()` call provides analysis and ranking.
The full and Miniformer structure-generation pipelines and BoltzIF are also independently callable.

See [`BoltzGenPipeline`][ofoldx.models.boltzgen.BoltzGenPipeline] for protocol and selection options.

## Training Details

BoltzGen trains coordinate generation with EDM denoising and inverse folding with a separate structure-conditioned sequence objective.

### Training Data

BoltzGen uses experimental structures from the [Protein Data Bank](https://www.rcsb.org/) and predicted structures from
[AlphaFold DB](https://alphafold.ebi.ac.uk/) and Boltz-1, including protein–ligand, RNA and protein–DNA systems.

### Training Procedure

The structure designer learns to denoise cropped structures with sampled design regions and conditioning information.
The inverse-folding model learns to assign sequences conditioned on molecular structures. OFoldX provides
[`diffusion_loss`][ofoldx.models.boltzgen.diffusion_loss] for training and fine-tuning.

## Citation

```bibtex
@article{stark2025boltzgen,
  author = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and others},
  title = {BoltzGen: Toward Universal Binder Design},
  year = {2025},
  doi = {10.1101/2025.11.20.689494},
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

Please contact the authors of the [BoltzGen paper](https://doi.org/10.1101/2025.11.20.689494) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

[ofoldx.models.boltzgen.BoltzGenPipeline]: https://ofoldx.valtech.ai/models/boltzgen/#ofoldx.models.boltzgen.BoltzGenPipeline
[ofoldx.models.boltzgen.diffusion_loss]: https://ofoldx.valtech.ai/models/boltzgen/#ofoldx.models.boltzgen.diffusion_loss
