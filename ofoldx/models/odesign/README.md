---
library_name: ofoldx
tags:
  - biology
  - protein-structure-prediction
  - protein-design
pipeline_tag: other
---

# ODesign

ODesign is an all-atom structure-prediction model for protein design: a recycling
trunk (MSA, template, constraint-distogram, and Pairformer stacks) feeds an EDM
coordinate diffusion sampler, with distogram and confidence heads.

It uses the ODesign token schema and adds its own input embedder, a token
bond-type reconstruction head, an optional constraint-distogram embedder for
conditioning generation on distances, and a design-generator wrapper.

## Usage

Predict an all-atom structure with the Tier-0 verb facade:

```python
import ofoldx

# doctest: +SKIP
ofoldx.fold("complex.cif", model="ofoldx/odesign").save_pdb("out.pdb")
```

Drive the design-generator variant through the `design` verb:

```python
import ofoldx

# doctest: +SKIP
ofoldx.design("target.cif", model="ofoldx/odesign-design")
```

Or load the model and processor explicitly with `AutoModel`:

```python
from ofoldx import AutoModel, AutoProcessor

# doctest: +SKIP
processor = AutoProcessor.from_pretrained("ofoldx/odesign")
model = AutoModel.from_pretrained("ofoldx/odesign")
output = model(processor("complex.cif"))
```

## Model Details

- **Task:** all-atom protein structure prediction for design (`odesign`), with a
  design-generator entry point (`odesign_design`).
- **Architecture:** a recycling trunk (MSA, template, constraint-distogram, and
  Pairformer stacks) feeding an EDM coordinate diffusion sampler, plus distogram,
  confidence, and token bond-type heads. ODesign extends the shared AlphaFold 3 /
  Protenix trunk with two design-specific components — the token bond-type
  predictor and the constraint-distogram embedder.
- **Token schema:** restype(35) + profile(32) + deletion_mean(1) +
  is_hotspot_residue(1), giving 453 token feature channels and 386 atom feature
  channels; the trunk runs 10 recycling passes by default.

### Provenance

OFoldX converts a released ODesign checkpoint (e.g. `odesign_base_prot_flex.pt`)
into OFoldX key names and format; it does not train these weights. Every OFoldX
parameter, including the ODesign-specific `bond_type_predictor` and
`constraint_distogram_embedder` tensors, is sourced from the reference checkpoint.
The released flex checkpoint enables the constraint-distogram embedder, so the
converter builds the config with `constraint.enabled = True` to match.

## Variants

| Architecture     | Public name    | What differs                                                                                                                                                                                                 |
| ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `odesign`        | ODesign        | The all-atom structure-prediction model: recycling trunk + EDM diffusion sampler with distogram, confidence, and token bond-type heads.                                                                      |
| `odesign_design` | ODesign Design | The design-generator entry point. It shares the ODesign structure trunk and heads and returns generated structure predictions through the design-generator contract, exposed under its own architecture key. |

## Conversion

This is a converted upstream checkpoint. Convert a released ODesign checkpoint
into a pretrained OFoldX artifact directory (writes `manifest.json`,
`config.json`, and `model.safetensors`):

```bash
python -m ofoldx.models.odesign.convert path/to/odesign_base_prot_flex.pt out/odesign
```

Use `--dtype {keep,float32,float16,bfloat16}` to choose the floating-point dtype
for the persisted weights (default `keep`).
