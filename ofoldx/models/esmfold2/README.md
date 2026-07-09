---
library_name: ofoldx
tags:
  - biology
  - structure-prediction
  - protein-structure
pipeline_tag: other
license: mit
base_model: biohub/ESMFold2 + multimolecule/esmc-6b
---

# esmfold2

Biohub ESMFold2 all-atom structure-prediction head converted for OFoldX. Full
sequence inference uses ESMC loaded through MultiMolecule.

The OFoldX implementation wraps the released ESMFold2 structure-prediction
architecture: the Biohub ESMFold2 structure-prediction head is converted into the
OFoldX runtime so it can be driven through the standard task verbs. The released
checkpoint expects ESMC-6B language-model representations, and OFoldX keeps that
boundary explicit — callers can supply representations directly, attach a frozen
language model, or load ESMC through MultiMolecule.

## Usage

```python
from ofoldx import fold

# Predict all-atom structure from a protein sequence.
# ESMC-6B representations are loaded through MultiMolecule on demand.
result = fold(
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    model="biohub/ESMFold2",
)
result.save_pdb("prediction.pdb")
print(result.plddt)
```

The folding head can also consume caller-provided `lm_hidden_states` in
`ESMFold2Input`, attach a frozen language model via `set_language_model()`, or call
`load_esmc()` to load ESMC through MultiMolecule's `AutoTokenizer` / `AutoModel`
surface. To load a locally converted artifact instead of the Hub id:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("path/to/esmfold2-artifact")
```

## Model Details

- **Task:** biomolecular structure prediction (`structure_prediction`)
- **Upstream project:** [ESMFold2](https://github.com/Biohub/esm)
- **Source release:** [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2)
- **Paper:** [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf)
- **License:** MIT for the Biohub ESMFold2 release; AGPL-3.0-or-later for the
  MultiMolecule ESMC implementation.

### Training Data

The upstream ESMFold2 model card describes training on Protein Data Bank and
AlphaFold DB structures; the ESMC language model was trained on large
protein-sequence corpora. OFoldX does not redistribute the training corpora.

### Training Procedure

OFoldX converts the released Biohub ESMFold2 checkpoint into `model.safetensors`
plus an OFoldX manifest; it does not run ESMFold2 training. The ESMC component is
loaded separately through MultiMolecule (`multimolecule/esmc-6b` by default).

## Conversion

`convert.py` accepts a local or Hub `model.safetensors` source (default repo
`biohub/ESMFold2`) and writes an OFoldX artifact:

```bash
python -m ofoldx.models.esmfold2.convert biohub/ESMFold2 ./esmfold2-artifact
```

The converter writes `manifest.json`, `config.json`, and `model.safetensors`;
`AutoConfig` and `AutoModel` then load the converted artifact.

## Citation

```bibtex
@misc{candido2026language,
  title = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and others},
  year = {2026},
  url = {https://biohub.ai/papers/esm_protein.pdf},
  note = {Preprint}
}
```
