---
library_name: ofoldx
tags:
  - biology
  - design
  - protein-design
pipeline_tag: other
---

# PxDesign

PxDesign is a protein-backbone design generator that samples backbone coordinates
from design conditioning via AlphaFold 3-style EDM diffusion. Beyond backbone
generation it drives the official design pipeline: ProteinMPNN sequence assignment
and structure-predictor filtering.

Distinctive to this family are its condition embedder (fusing an atom-to-token
encoder with restype/pLDDT/hotspot/design token features), the conditional-template
pair embedder, the EDM coordinate sampler with step-scale schedules, and the design
distogram head.

Shared neural network building blocks live in `ofoldx.modules`; this directory
holds the family's config, typed I/O, checkpoint conversion, and model.

## Usage

```python
# doctest: +SKIP
import ofoldx

# Verb facade: design backbones (and sequences) for a design spec.
result = ofoldx.design("design.json", model="ofoldx/pxdesign")
for candidate in result:
    print(candidate)
```

Or load the model and processor explicitly:

```python
# doctest: +SKIP
from ofoldx.models import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("ofoldx/pxdesign")
model = AutoModel.from_pretrained("ofoldx/pxdesign")
output = model.design(processor("design.json"))
```

## Model Details

- **Task:** biomolecular design generation (backbone design)
- **Architecture key:** `pxdesign`
- **Checkpoint variant:** `pxdesign_v0.1.0`
- **Source release:** official PxDesign v0.1.0 checkpoint (`pxdesign_v0.1.0.pt`),
  downloaded from
  `https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt`
- **License:** not declared in the OFoldX release metadata for this checkpoint;
  consult the upstream PxDesign release before redistribution.

### Provenance

OFoldX provides an OFoldX-native reimplementation of the PxDesign backbone
diffusion generator and converts the official PxDesign v0.1.0 checkpoint into a
loadable native model; it does not train these weights and does not redistribute
any training corpus. The public PxDesign v0.1.0 release ships inference-only, so
OFoldX grounds the training `forward` in the same single-step EDM denoise the
sister Protenix/ODesign models use. The design distogram head is left disabled in
the public inference model and is kept available for native runs.

## Conversion

This is a converted upstream checkpoint. `convert.py` translates an official
PxDesign `.pt` checkpoint into a single-file OFoldX pretrained directory, mapping
the official condition-embedder, atom-encoder, denoiser, and distogram keys onto
the native module names:

```bash
python -m ofoldx.models.pxdesign.convert path/to/pxdesign_v0.1.0.pt out/pxdesign
```

Options:

- `--strict` — require every native tensor to be mapped (treat the freshly
  initialized distogram head as a failure).
- `--dtype {float32,float16,bfloat16}` — cast the converted weights on save.
