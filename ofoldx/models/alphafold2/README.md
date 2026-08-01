---
tags:
  - Biology
  - Protein
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# AlphaFold 2

Pre-trained model for protein structure prediction from amino-acid sequences, multiple-sequence alignments, and optional structural templates.

## Disclaimer

This is an UNOFFICIAL implementation of the [Highly accurate protein structure prediction with AlphaFold](https://doi.org/10.1038/s41586-021-03819-2) by John Jumper, et al.

The OFFICIAL repository of AlphaFold 2 is at [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold).

**The team releasing AlphaFold 2 did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

AlphaFold 2 jointly updates MSA and residue-pair representations with an Evoformer, then uses an
invariant-point-attention structure module to predict backbone frames, side-chain angles, atomic coordinates, and
confidence. AlphaFold-Multimer extends the model with paired MSAs and chain-relative features for protein complexes.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/alphafold2](https://huggingface.co/oteam/alphafold2)**: Monomer structure prediction with pLDDT confidence.
- **[oteam/alphafold2-ptm](https://huggingface.co/oteam/alphafold2-ptm)**: Monomer prediction with PAE and pTM confidence.
- **[oteam/alphafold2-multimer](https://huggingface.co/oteam/alphafold2-multimer)**: Multi-chain protein complex prediction.
- **[oteam/alphafold2-casp14](https://huggingface.co/oteam/alphafold2-casp14)**: The five-model CASP14 protocol with eight input ensembles per checkpoint.
- **[oteam/alphafold2-initial-guess](https://huggingface.co/oteam/alphafold2-initial-guess)**: The `model_1_ptm` initial-guess evaluator used by PXDesign.

### Model Specification

| Variant     | Evoformer Blocks | Extra-MSA Blocks | Structure Blocks | MSA Size | Pair Size | Single Size | Recycling Steps | Num Parameters (M) |
| ----------- | ---------------- | ---------------- | ---------------- | -------- | --------- | ----------- | --------------- | ------------------ |
| Monomer     | 48               | 4                | 8                | 256      | 128       | 384         | 4               | 93.283             |
| Monomer pTM | 48               | 4                | 8                | 256      | 128       | 384         | 4               | 93.292             |
| Multimer    | 48               | 4                | 8                | 256      | 128       | 384         | 21              | 93.278             |

### Links

- **Code**: [ofoldx.models.alphafold2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold2)
- **Paper**: [Highly accurate protein structure prediction with AlphaFold](https://doi.org/10.1038/s41586-021-03819-2); [Protein complex prediction with AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034)
- **Developed by**: John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger,
  Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl,
  Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back,
  Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer,
  Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis
  Hassabis
- **Model type**: MSA- and template-conditioned protein structure predictor with an Evoformer trunk and
  invariant-point-attention structure module
- **Original Repository**: [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Structure Prediction

You can use this model directly with a pipeline for protein structure prediction:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/alphafold2")
result = pipeline(structure, seed=7)
```

The result contains predicted structures and confidence values. Each prediction uses a new coordinate frame;
unpredicted atoms, including ligands, hydrogens, and terminal OXT, have unavailable coordinates.

### Downstream Use

#### CASP14 Ensemble

You can use the five-model ensemble to rank predictions and optionally relax the best structure:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("input.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/alphafold2-casp14")
result = pipeline(structure, num_samples=1, models_to_relax="best", platform="CPU", seed=7)
best = result.best
```

Each checkpoint evaluates eight input ensembles from shared preparation. `result.ranked` retains checkpoint names,
sample indices, and relaxation results; equal mean-pLDDT scores retain generation order. Amber relaxation requires
[OpenMM](https://docs.openmm.org/latest/userguide/application/01_getting_started.html) and
[PDBFixer](https://github.com/openmm/pdbfixer), installed in the same environment:

```bash
conda install -c conda-forge openmm pdbfixer
```

Relaxation requires a closed, standard linear protein graph. Relaxed coordinates replace experimental coordinate
metadata; source identity and independent chemical facts are retained. Ordinary unrelaxed prediction needs neither
additional package.

#### Initial-Guess Evaluation

You can evaluate a sequence-assigned target-binder complex using its existing coordinates as the initial guess:

```python
import ofoldx
from ofoldx.io import read_structure

structure, report = read_structure("target_binder.cif")
pipeline = ofoldx.pipeline("fold", model="oteam/alphafold2-initial-guess")
result = pipeline(structure, binder_instance_id="binder", num_recycles=4, seed=7)
```

The input must contain the named binder instance. The initial guess seeds recycled atom37 positions; the dedicated
processor retains the target-binder template masking used by PXDesign.

See the [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) for staged execution and composition.

## Training Details

AlphaFold 2 is trained for protein structure prediction with structural and masked-MSA objectives.

### Training Data

The AlphaFold 2 and AlphaFold-Multimer papers describe the upstream protein structure, sequence, and template corpora.
OFoldX does not redistribute those corpora or databases.

### Training Procedure

The upstream models use recycling and structural and auxiliary objectives. The pTM variants were fine-tuned with a
predicted-aligned-error head; AlphaFold-Multimer adds paired-MSA and chain-relative features and interface confidence.

OFoldX exposes a differentiable final recycle and a masked-MSA reconstruction objective for that recycle. This does not
reproduce the complete upstream structural and auxiliary-loss training program. MSA, template, and sequence-database
generation remain caller-owned.

## Citation

```bibtex
@article{jumper2021highly,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  title = {Highly accurate protein structure prediction with AlphaFold},
  journal = {Nature},
  volume = {596},
  pages = {583--589},
  year = {2021},
  doi = {10.1038/s41586-021-03819-2}
}
```

For AlphaFold-Multimer predictions, also cite
[Protein complex prediction with AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034).

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

Please contact the authors of the [AlphaFold 2 paper](https://doi.org/10.1038/s41586-021-03819-2) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
