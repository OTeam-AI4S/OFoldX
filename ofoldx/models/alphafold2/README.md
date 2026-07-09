---
library_name: ofoldx
tags:
  - biology
  - structure-prediction
  - protein-structure
pipeline_tag: other
license: cc-by-4.0
---

# AlphaFold 2

AlphaFold 2 structure-prediction architecture entry point for OFoldX.

The OFoldX runtime treats AlphaFold 2 as a distinct structure-prediction
architecture: an AF2-shaped input embedding, recycling trunk, template
embedding, extra-MSA/main Evoformer stacks, StructureModule, and confidence
heads that emit the standard OFoldX structure-prediction output.

## Usage

```python
from ofoldx import fold

# Load a converted AlphaFold 2 artifact and fold an input.
result = fold("target.fasta", model="oteam/alphafold2")
result.save_pdb("prediction.pdb")
print(result.plddt)
```

You can also build the model explicitly through the auto classes:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/alphafold2")
```

## Model Details

- **Task:** biomolecular structure prediction
- **Upstream project:** [AlphaFold 2](https://github.com/google-deepmind/alphafold)
- **Source release:** [alphafold_params_2022-12-06.tar](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar)
- **Paper:** [Highly accurate protein structure prediction with AlphaFold](https://doi.org/10.1038/s41586-021-03819-2)
- **License:** Apache-2.0 for upstream AlphaFold 2 code; CC BY 4.0 for official model parameters.

### Training Data

The upstream AlphaFold 2 release describes training on PDB structures and
sequence databases. OFoldX does not redistribute the training corpus.

### Training Procedure

The upstream AlphaFold 2 release provides official model parameters under
CC BY 4.0. OFoldX converts official monomer and pTM `params_model_*.npz`
checkpoints into `model.safetensors` with a strict-loadable AF2 PyTorch
parameter tree. OFoldX converts these weights; it does not train them.

## Conversion

DeepMind distributes AlphaFold 2 parameters as
`alphafold_params_2022-12-06.tar`. After extraction, the download root contains
`params/params_<model_name>.npz` files for the monomer CASP14 and pTM models,
plus a license file. The upstream AlphaFold code is Apache-2.0; the official
model parameters are CC BY 4.0.

Convert a checkpoint into an OFoldX model artifact with the public CLI. It
accepts a direct `params_model_*.npz`, a params/download directory, or the tar
archive:

```bash
python -m ofoldx.models.alphafold2.convert --source deepmind /data/alphafold model-af2 \
  --model-name model_1_ptm --dtype float32
```

OpenFold PyTorch checkpoints (`s3://openfold/openfold_params/`) convert through
the same entry point:

```bash
python -m ofoldx.models.alphafold2.convert --source openfold \
  /data/openfold_params/finetuning_ptm_2.pt openfold-finetuning-ptm-2
```

The output is an OFoldX `kind="model"` artifact with `manifest.json`,
`conversion_config.json`, `config.json`, and `model.safetensors`.

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
