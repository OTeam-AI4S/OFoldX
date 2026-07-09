---
library_name: ofoldx
tags:
  - biology
  - design
  - protein-design
pipeline_tag: other
license: bsd-3-clause
---

# RFdiffusion

Converted RFdiffusion checkpoint for backbone generation, motif scaffolding, and binder design.
RFdiffusion is a denoising-diffusion model that generates novel protein backbones from a noised
structure. A RoseTTAFold trunk denoises the backbone at each timestep, and a native reverse-diffusion
sampler drives one design trajectory from noise down to a finished backbone.

Its distinctive pieces are the ported RoseTTAFold network with MSA/pair/template embedders and an
SE(3)-transformer structure module, plus a sampler with the Euclidean and IGSO3 diffusers, denoiser,
contig map, symmetry, and guiding potentials.

## Usage

```python
import ofoldx

# Motif scaffolding: rebuild a backbone around a fixed motif segment.
result = ofoldx.design(
    "motif.cif",
    model="oteam/rfdiffusion",
    contigs=["A10-25/80-120"],
    num_samples=1,
)
candidate = result.candidates[0]
```

De novo backbone generation is expressed the same way with a length-only contig (e.g.
`contigs=["100-150"]` and no input structure). Sampler controls such as `num_samples`,
`hotspot_res`, `symmetry`, and guiding `potentials` are forwarded as keyword arguments.

You can also load the designer directly:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/rfdiffusion")
```

## Model Details

- **Task:** design (biomolecular design generation)
- **Upstream project:** [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- **Source release:** [download_models.sh](https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh)
- **Paper:** [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8)
- **License:** BSD for upstream RFdiffusion code and referenced model weights (`bsd-3-clause`)

### Training Data

RFdiffusion fine-tunes a RoseTTAFold-style structure network as a denoising diffusion model over PDB
protein structures. OFoldX does not redistribute the training set.

### Training Procedure

Upstream RFdiffusion noises residue frames with Gaussian C-alpha translation noise and rotational
Brownian motion, trains denoising to true frames with self-conditioning, and uses checkpoint-specific
inference configs. OFoldX converts released RFdiffusion checkpoints into `model.safetensors`; it does
not run RFdiffusion training.

## Conversion

These weights are converted from the released RosettaCommons RFdiffusion checkpoints. Conversion
preserves the checkpoint tensors, renames the terse upstream `model`/`diffuser`/`preprocess` keys to
the house vocabulary (the `SE3_param_*` inner degree labels are left unchanged), and writes
`model.safetensors`.

```bash
python -m ofoldx.models.rfdiffusion.convert \
    /path/to/official_rfdiffusion.pt \
    /path/to/output_dir \
    --checkpoint-variant base
```

Pass `--dtype {keep,float32,float16,bfloat16}` to cast tensors and `--state-key` to select the source
state dict. Hub-upload flags from `ofoldx.hub` are also accepted.

## Citation

```bibtex
@article{watson2023denovo,
  author = {Watson, Joseph L. and Juergens, David and Bennett, Nathaniel R. and others},
  title = {De novo design of protein structure and function with RFdiffusion},
  journal = {Nature},
  volume = {620},
  number = {7976},
  pages = {1089--1100},
  year = {2023},
  doi = {10.1038/s41586-023-06415-8}
}
```
