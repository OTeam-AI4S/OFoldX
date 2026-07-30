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

# AlphaFold 3

All-atom biomolecular structure prediction for proteins, nucleic acids,
ligands, ions, modified residues, and glycan components.

## Disclaimer

This is an unofficial OFoldX implementation of
[Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
by Josh Abramson, Jonas Adler, Jack Dunger, et al. The upstream authors did not
write, review, or endorse this implementation or model card. The official
repository is
[google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3).

## Model Details

AlphaFold 3 predicts all-atom structures of biomolecular complexes containing
proteins, DNA, RNA, small molecules, ions, modified residues, and glycans. It
combines atom-, token-, MSA-, template-, bond-, and relative-position features
in a recycled Pairformer trunk, then generates atomic coordinates with a
diffusion model. Distogram and confidence heads estimate pLDDT, PAE, PDE, pTM,
ipTM, disorder, and ranking scores.

### Variants

- **[oteam/alphafold3](https://huggingface.co/oteam/alphafold3)**: AlphaFold 3 for general biomolecular structure prediction.

### Model Specification

| Pairformer Blocks | Template Blocks | Denoiser Blocks | Single Size | Pair Size | Recycling Steps | Sampling Steps | Num Parameters (M) | FLOPs (G) |
| ----------------- | --------------- | --------------- | ----------- | --------- | --------------- | -------------- | ------------------ | --------- |
| 48                | 2               | 24              | 384         | 128       | 11              | 200            | 368.317            | 1226.198  |

### Links

- **Code**: [ofoldx.models.alphafold3](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold3)
- **Paper**: [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
- **Developed by**: Josh Abramson, Jonas Adler, Jack Dunger, et al.
- **Model type**: Pairformer-based all-atom biomolecular structure predictor with diffusion coordinate generation
- **Original Repository**: [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)

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

pipeline = ofoldx.pipeline("fold", model="oteam/alphafold3")
output = pipeline.fold("input.cif")
```

`AlphaFold3Model.sample` exposes the raw final single and pair representations,
coordinate samples, distogram logits, and confidence logits. The Pipeline
decodes those predictions into a `FoldResult`.

## Training Details

### Training Data

The upstream AlphaFold 3 publication describes the biomolecular structure,
sequence, and chemistry resources used for training. OFoldX does not
redistribute the training corpus. Refer to the paper and its supplementary
information for the authoritative data description.

### Training Procedure

AlphaFold 3 is trained end to end with a diffusion coordinate objective and
auxiliary distogram and confidence objectives. OFoldX preserves the inference
architecture and exposes the differentiable denoising step.

## Limitations

The exposed denoising operation is not a reproduction of AlphaFold 3's complete
end-to-end training program, so this family has no built-in Runner objective.
OFoldX also does not supply the upstream training corpus or evidence pipelines.

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

If OFoldX supports your research, also cite:

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
original model should be directed to the AlphaFold 3 authors.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
