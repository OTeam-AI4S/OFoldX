---
language: protein
tags:
  - Biology
  - Protein
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# AlphaFold 2

Native PyTorch implementation of AlphaFold 2 monomer, monomer-pTM, and
AlphaFold-Multimer for OFoldX.

## Disclaimer

This is an unofficial OFoldX implementation of
[AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) and
[AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034). The upstream
authors did not write, review, or endorse this implementation or model card.
The official implementation is maintained by
[Google DeepMind](https://github.com/google-deepmind/alphafold).

## Model Details

AlphaFold 2 predicts protein three-dimensional structure from amino-acid
sequence, multiple-sequence alignments, and optional structural templates. Its
Evoformer jointly updates MSA and residue-pair representations before an
invariant-point-attention Structure Module produces backbone frames, side-chain
angles, atomic coordinates, and confidence estimates. AlphaFold-Multimer extends
the same design with paired MSAs and chain-relative features for protein
complexes.

### Variants

- **[oteam/alphafold2](https://huggingface.co/oteam/alphafold2)**: AlphaFold 2 monomer model with pLDDT confidence.
- **[oteam/alphafold2-ptm](https://huggingface.co/oteam/alphafold2-ptm)**: Monomer model with predicted aligned error and pTM confidence.
- **[oteam/alphafold2-multimer](https://huggingface.co/oteam/alphafold2-multimer)**: AlphaFold-Multimer for multi-chain protein complexes.

### Model Specification

| Variant     | Evoformer Blocks | Extra-MSA Blocks | Structure Blocks | MSA Size | Pair Size | Single Size | Recycling Steps | Num Parameters (M) | FLOPs (G) |
| ----------- | ---------------- | ---------------- | ---------------- | -------- | --------- | ----------- | --------------- | ------------------ | --------- |
| Monomer     | 48               | 4                | 8                | 256      | 128       | 384         | 4               | 93.229             | 1282.947  |
| Monomer pTM | 48               | 4                | 8                | 256      | 128       | 384         | 4               | 93.237             | 1283.216  |
| Multimer    | 48               | 4                | 8                | 256      | 128       | 384         | 21              | 93.224             | 11592.924 |

### Links

- **Code**: [ofoldx.models.alphafold2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold2)
- **Paper**: [Highly accurate protein structure prediction with AlphaFold](https://doi.org/10.1038/s41586-021-03819-2)
- **Multimer Paper**: [Protein complex prediction with AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034)
- **Developed by**: John Jumper, Richard Evans, Alexander Pritzel, et al.
- **Model type**: MSA- and template-conditioned protein structure predictor with an Evoformer trunk and invariant-point-attention structure module
- **Original Repository**: [google-deepmind/alphafold](https://github.com/google-deepmind/alphafold)

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

output = ofoldx.fold("MKTAYIAKQRQISFVKSHFSRQ", model="oteam/alphafold2")
```

## Training Details

### Training Data

Refer to the AlphaFold 2 and AlphaFold-Multimer papers and supplementary
materials for the upstream training corpora and data construction. OFoldX does
not redistribute those corpora, sequence databases, or template databases.

### Training Procedure

The upstream monomer models use recycling, MSA and template evidence, and the
released auxiliary losses; the pTM variants were fine-tuned with the
predicted-aligned-error head. AlphaFold-Multimer adds paired-MSA and
chain-relative features and interface confidence.

OFoldX preserves a differentiable final recycle and the released inference
module boundaries.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Questions about the
original models should be directed to the upstream AlphaFold authors.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
