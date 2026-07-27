---
language: protein
tags:
  - Biology
  - Protein
  - Protein Design
  - Inverse Folding
license: agpl-3.0
datasets:
  - oteam/boltzgen1
library_name: ofoldx
pipeline_tag: other
---

# BoltzGen

BoltzGen is OFoldX's structure-based biomolecular design family for universal
binder design, controllable protein structure generation, and inverse folding.
Its concrete Pipeline combines a structure-diffusion designer with a distinct
autoregressive inverse-folding model, independent Boltz-2 folding, and Boltz-2
affinity scoring.

## Disclaimer

This is an unofficial OFoldX implementation of
[BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494)
by Hannes Stark, Felix Faltings, MinGyu Choi, et al. The upstream authors did not
write, review, or endorse this implementation or model card.

The official repository is
[HannesStark/boltzgen](https://github.com/HannesStark/boltzgen).

## Model Details

BoltzGen is an all-atom diffusion model for designing proteins, peptides, and
other biomolecular binders around structural and interaction constraints. It
combines atom, token, MSA, template, contact, binding-site, and
secondary-structure features in a recycled Pairformer trunk before an EDM
denoiser generates coordinates. A separate graph neural network performs
inverse folding to assign amino-acid sequences to generated backbones.

### Variants

The released composite repository has five required checkpoint roles:

- **Diverse**: constrained structure generation at the repository root.
- **Adherence**: condition-adherent structure generation under `adherence/`.
- **Inverse folding**: backbone-conditioned sequence design under `ifold/`.
- **Folding**: Boltz-2 structure prediction under `fold/`.
- **Affinity**: Boltz-2 protein–ligand scoring under `affinity/`.

`BoltzGenConfig.structuretrained_small()` also describes the reduced
structure-trained Miniformer architecture; it is not one of these five
end-to-end checkpoint roles.

### Model Specification

| Variant                 | Representation Size | Trunk                                | Denoiser / Decoder Blocks | Num Parameters (M) | FLOPs (G) |
| ----------------------- | ------------------- | ------------------------------------ | ------------------------- | ------------------ | --------- |
| Diverse / Adherence     | 384 / 128           | 64 Pairformer + 4 MSA Blocks         | 24 token Blocks           | 482.175            | 3195.183  |
| Structure-trained small | 384 / 128           | 12 Miniformer + 3 MSA Blocks         | 8 token Blocks            | 136.848            | 333.935   |
| Inverse folding         | 128 / 128           | 6 graph encoder Blocks, 30 neighbors | 3 graph decoder Blocks    | 3.029              | 13.639    |

### Links

- **Code**: [ofoldx.models.boltzgen](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltzgen)
- **Paper**: [BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494)
- **Developed by**: Hannes Stark, Felix Faltings, MinGyu Choi, et al.
- **Model type**: Constraint-guided all-atom biomolecular diffusion and inverse folding
- **Original Repository**: [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Biomolecular Design

Load a converted five-pair BoltzGen repository and select one of the six
released protocols:

```python
import ofoldx

pipeline = ofoldx.pipeline("design", model="/path/to/boltzgen")
output = pipeline.design(
    None,
    protocol="protein-anything",
    total_len=120,
    num_candidates=20,
)

best = output.ranked_candidates[0]
best.structure
best.whole_refold
```

Each candidate analysis keeps every designed chain separate, including its exact
designed sequence, full sequence, hydrophobicity, and liability values. Binding-site
adherence is retained at all released 3--9 Å cutoffs. Optional original, refolded,
and per-target diversity reductions remain in memory:

```python
from ofoldx.models.boltzgen import analyze_diversity

diversity = analyze_diversity(
    tuple(candidate.diversity_sample("target") for candidate in output.candidates)
)
```

The other protocol values are `peptide-anything`,
`protein-small_molecule`, `nanobody-anything`, `antibody-anything`, and
`protein-redesign`. Every analyzed candidate remains in `output.candidates`;
the ranked, top-budget, and diversity-selected collections are stable views
over those candidates. `protein-small_molecule` infers its affinity ligand when
the system has exactly one molecule instance; pass `affinity_ligand="..."` to
select it explicitly in a multi-molecule system.

Convert the five official artifacts into that fixed repository layout with:

```bash
python -m ofoldx.models.boltzgen.convert \
  boltzgen1_diverse.ckpt \
  boltzgen1_adherence.ckpt \
  boltzgen1_ifold.ckpt \
  boltz2_conf_final.ckpt \
  boltz2_aff.ckpt \
  /path/to/boltzgen
```

## Training Details

### Training Data

BoltzGen builds on the Boltz-2 data pipeline with biomolecular structures, MSA
evidence, molecular dictionaries, and optional distillation data. The
inverse-folding model uses a distinct structure-conditioned sequence target.

### Training Procedure

The structure designer is trained with an EDM denoising objective over randomly
cropped biomolecular structures and randomly selected design and conditioning
regions. Training noise follows
`sigma = sigma_data * exp(-1.2 + 1.5 * z)`, `z ~ N(0, 1)`, with loss weight
`(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`. The inverse-folding model
uses a separate structure-conditioned residue-prediction objective. Its
`forward` and `design` operations use one caller generator for configured
coordinate noise, teacher-forcing visibility, decode order, and categorical
sampling.

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
[BoltzGen paper](https://doi.org/10.1101/2025.11.20.689494).

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
