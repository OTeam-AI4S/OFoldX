---
library_name: ofoldx
tags:
  - biology
  - biomolecular-design
  - protein
  - rna
  - dna
  - structure-prediction
  - protein-structure
pipeline_tag: other
---

# Boltz-1

Boltz-1 predicts all-atom structure for tokenized biomolecular complexes: a token/atom
input embedder feeds a recycling representation trunk (Pairformer plus an optional MSA
module) and a distogram head, then an EDM reverse-diffusion sampler denoises coordinates,
which the Boltz-1 confidence/ranking head scores.

Distinctive to this family are its atom-attention input embedder, precomputed diffusion
conditioning, the EDM coordinate sampler, the confidence/ranking head, and optional Boltz-1x
inference-time steering (Feynman-Kac particles and physical-potential guidance).

## Usage

Boltz-1 is a structure-prediction (folding) model, so it is reached through the Tier-0
`fold` verb. Point it at a tokenized complex and it returns a structure-prediction result
exposing `.plddt` / `.coords` / `.save_pdb`.

```python
import ofoldx

result = ofoldx.fold("complex.cif", model="oteam/boltz1")  # doctest: +SKIP
result.save_pdb("boltz1_prediction.pdb")  # doctest: +SKIP
print(result.plddt)  # doctest: +SKIP
```

To build the model explicitly (e.g. to override the config), load it through `AutoModel`:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/boltz1")  # doctest: +SKIP
```

## Model Details

- **Task:** biomolecular structure prediction
- **Architecture:** `boltz1`
- **Upstream project:** Boltz-1 / Boltz-1x
- **Source checkpoint:** the released Boltz-1 inference checkpoint `boltz1_conf.ckpt`

The default configuration reproduces the public Boltz-1 inference entry point: a 48-block
Pairformer trunk, a 4-block MSA module, an 8-head diffusion token transformer, and the
Boltz-1 EDM sampling schedule (Euler solver, 200 reverse-diffusion steps). The confidence
module imitates the trunk and emits PAE/pTM ranking metrics. Boltz-1x inference-time
steering (Feynman-Kac particle steering and physical-potential guidance) is exposed but
disabled by default.

### Provenance

OFoldX converts the released Boltz-1 checkpoint into its own parameter layout; it does not
train these weights. The converter splits the upstream monolithic score model, moving the
invariant pair/atom conditioning into `diffusion_conditioning` while keeping the denoising
layers under `structure_module.score_model`, and preserves the clean OFoldX parameter names.

## Conversion

This card describes a converted upstream checkpoint. Convert an original Boltz-1 `.ckpt`
into an OFoldX pretrained directory (writes `manifest.json`, `config.json`, and
`model.safetensors`):

```bash
python -m ofoldx.models.boltz1.convert boltz1_conf.ckpt out/boltz1
```

Options:

- `--dtype {keep,float32,float16,bfloat16}` — cast floating-point tensors during conversion
  (default `keep`).
- `--no-strict` — allow non-ignorable source checkpoint keys to remain unmapped instead of
  raising.

The command also accepts the shared hub-upload arguments, so a converted directory can be
pushed to the Hub in the same step.
