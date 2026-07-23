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
base_model: bytedance/Protenix
library_name: ofoldx
pipeline_tag: other
---

# Protenix

Protenix is OFoldX's AlphaFold 3-shaped all-atom structure-prediction family for
proteins, nucleic acids, small molecules, and their complexes. OFoldX exposes
the default and constraint graphs as distinct Model/Processor/Pipeline pairs.

## Disclaimer

This is an unofficial OFoldX implementation of
[Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
by the ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al. The
upstream authors did not write, review, or endorse this implementation or model card.

The official repository is
[bytedance/Protenix](https://github.com/bytedance/Protenix).

## Model Details

Protenix is an AlphaFold 3-style all-atom structure predictor for proteins,
nucleic acids, small molecules, ions, and their complexes. It combines atom and
token embeddings with an MSA-conditioned recycled Pairformer trunk, EDM
coordinate diffusion, and distogram and confidence heads. The constraint variant
adds learned pair biases for atom contacts, token contacts, pockets, and
substructures.

### Variants

- **[oteam/protenix-base](https://huggingface.co/oteam/protenix-base)**: General biomolecular structure prediction.
- **[oteam/protenix-constraint](https://huggingface.co/oteam/protenix-constraint)**: Structure prediction with contact, pocket, and substructure conditioning.

### Model Specification

| Variant    | Pairformer Blocks | Denoiser Blocks | Confidence Blocks | Token Size | Single Size | Pair Size | Recycling Passes | Num Parameters (M) | FLOPs (G) |
| ---------- | ----------------- | --------------- | ----------------- | ---------- | ----------- | --------- | ---------------- | ------------------ | --------- |
| Base       | 48                | 24              | 4                 | 449        | 384         | 128       | 10               | 368.088            | 1329.532  |
| Constraint | 48                | 24              | 4                 | 449        | 384         | 128       | 10               | 368.304            | 1473.988  |

### Links

- **Code**: [ofoldx.models.protenix](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/protenix)
- **Paper**: [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
- **Developed by**: ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al.
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with optional learned constraints
- **Original Repository**: [bytedance/Protenix](https://github.com/bytedance/Protenix)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Structure Prediction

Use the high-level `fold` API:

```python
import ofoldx

output = ofoldx.fold("input.cif", model="oteam/protenix-base")
```

## Training Details

### Training Data

The released model reports a 2021-09-30 training-data cutoff. Its data pipeline
covers wwPDB/mmCIF bioassemblies, MSAs, CCD cache data, and RCSB sequence
clusters.

### Training Procedure

Protenix trains the trunk, distogram, confidence, and coordinate-denoising graph
jointly. The diffusion objective denoises noised ground-truth coordinates, while
the auxiliary heads supervise geometry and confidence. The constraint variant
additionally trains all four pair-bias embedders.

## Citation

```bibtex
@article{protenix2025advancing,
  author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and others},
  title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  year = {2025},
  doi = {10.1101/2025.01.08.631967},
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
[Protenix paper](https://doi.org/10.1101/2025.01.08.631967).

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
