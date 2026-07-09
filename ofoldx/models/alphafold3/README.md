---
library_name: ofoldx
tags:
  - biology
  - structure-prediction
  - protein-structure
pipeline_tag: other
---

# alphafold3

Official AlphaFold 3 structure-prediction architecture entry point for OFoldX. It records
AlphaFold 3 as a distinct architecture in the OFoldX runtime and provides the AlphaFold 3-shaped
all-atom model for biomolecular structure prediction.

## Usage

```python
import ofoldx

# Predict a biomolecular structure from a CIF/FASTA input.
result = ofoldx.fold("complex.cif", model="oteam/alphafold3")
result.save_pdb("out.pdb")
print(result.plddt)
```

Or load the model explicitly through the builder API:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/alphafold3")
```

## Model Details

- **Task:** biomolecular structure prediction
- **Upstream project:** [AlphaFold 3](https://github.com/google-deepmind/alphafold3)
- **Source release:** [github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)
- **Paper:** [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
- **License:** Apache-2.0 for the upstream AlphaFold 3 code; the official DeepMind parameter terms apply to the model weights.

### Training Data

The upstream AlphaFold 3 publication describes training on biomolecular structure and sequence
resources. OFoldX does not redistribute the training corpus.

### Training Procedure

OFoldX can inspect local official AlphaFold 3 parameter archives for user-side conversion work.
Official DeepMind AlphaFold 3 parameters are not uploaded by the OFoldX converter.

## Conversion

AlphaFold 3 uses a source-aware converter that reads a local official DeepMind AlphaFold 3
parameter directory. It inspects the parameters in place and does not bundle or download the
weights:

```bash
python -m ofoldx.models.alphafold3.convert --source deepmind ~/models official-af3 --inspect-only
```

Protenix checkpoints use the separate `ofoldx.models.protenix.convert` entry point so converted
artifacts keep the `protenix` architecture identity.

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
