---
library_name: ofoldx
tags:
  - biology
  - structure-prediction
  - protein-structure
pipeline_tag: other
license: apache-2.0
---

# protenix

Protenix is the AlphaFold 3-shaped Protenix v0.5 structure predictor. The base checkpoint maps onto the
AF3-style trunk, EDM diffusion sampler, and distogram/confidence heads. The model also supports the
Protenix constraint pair-bias path: pocket, token-contact, atom-contact, and substructure features can be
embedded and added to the initial pair representation before the recycling trunk, and inference can
optionally apply Protenix training-free guidance.

Distinctive to this family are its guided EDM sampling loop, ten-cycle recycling trunk with no template
stack by default, and optional aggregate constraint pair-bias embedder.

Converted Protenix v0.5.0 structure-prediction weights in an AlphaFold 3-shaped OFoldX implementation.

## Usage

```python
import ofoldx

# Predict a structure from a complex file and save it to disk.
result = ofoldx.fold("complex.cif", model="oteam/protenix")
result.save_pdb("out.pdb")
print(result.plddt)
```

The same weights can be loaded through `AutoModel` when you need the typed model directly:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/protenix")
```

## Model Details

- Task: biomolecular structure prediction
- Upstream project: [Protenix](https://github.com/bytedance/Protenix)
- Source release: [supported models](https://github.com/bytedance/Protenix/blob/main/docs/supported_models.md) (`protenix_base_default_v0.5.0.pt`, keyed under title `protenix-base`)
- Paper: [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
- License: Apache-2.0 for upstream Protenix code and model parameters.

### Training Data

The upstream Protenix v0.5.0 release reports a 2021-09-30 training-data cutoff. The Protenix training-data
pipeline covers released wwPDB/mmCIF bioassemblies, MSAs, CCD cache data, and RCSB sequence clusters.
OFoldX does not redistribute the training corpus.

### Training Procedure

The upstream `protenix_base_default_v0.5.0` model is an AF3-shaped all-atom structure-prediction
checkpoint. OFoldX converts `protenix_base_default_v0.5.0.pt` to the OFoldX artifact format; it does not
retrain the model.

### Provenance

protenix-base: Source checkpoint `protenix_base_default_v0.5.0.pt`; `v0.5.0` is the upstream Protenix
checkpoint release version, not part of the public repo id.

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

## Conversion

Protenix checkpoints use the dedicated `ofoldx.models.protenix.convert` entry point so converted
artifacts keep the `protenix` architecture identity (the AlphaFold 3 converter is intentionally kept
separate):

```bash
python -m ofoldx.models.protenix.convert protenix_base_default_v0.5.0.pt ~/models/protenix
```

Pass `--dtype {keep,float32,float16,bfloat16}` to control the stored parameter precision.
