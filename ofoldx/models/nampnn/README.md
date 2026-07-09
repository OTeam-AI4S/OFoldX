---
library_name: ofoldx
tags:
  - biology
  - inverse-folding
  - sequence-design
pipeline_tag: other
---

# NA-MPNN

NA-MPNN is an inverse-folding model that designs and scores sequences for fixed
protein, DNA, RNA, and mixed-polymer backbones. A single geometric message-passing
encoder reads all polymer chains over one k-nearest-neighbor residue graph, and an
autoregressive decoder produces per-position probabilities, sequence scores, and
sampled sequences across the unified protein/nucleic-acid alphabet.

The feature encoder builds a shared graph over protein and nucleic-acid atoms:
protein residues contribute an idealized virtual Cβ, and nucleic-acid residues
contribute an idealized base-nitrogen atom, so a common set of radial-basis
atom-pair and relative-position edge features spans every polymer type. The
sampling surface covers standard and unconditional scoring, temperature-controlled
autoregressive sampling with fixed positions and per-token bias, and
symmetric/tied-position decoding.

## Usage

```python
from ofoldx.models import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("oteam/nampnn-design")
model = AutoModel.from_pretrained("oteam/nampnn-design")

# Design sequences for a fixed protein/DNA/RNA complex backbone.
output = model.design(processor("complex.pdb"), num_samples=8)
```

Or through the task facade:

```python
import ofoldx

ofoldx.design("complex.pdb", model="oteam/nampnn-design", num_samples=8)
```

`model.score(...)` and `model.unconditional_probs(...)` expose the scoring paths on
the same processed input.

## Model Details

- **Task:** Inverse folding / sequence design for protein, DNA, RNA, and
  mixed-polymer backbones.
- **Upstream project:** [NA-MPNN](https://github.com/baker-laboratory/NA-MPNN)
  (Baker Laboratory).
- **Source releases:** the official `design`
  ([`models/design_model/s_19137.pt`](https://github.com/baker-laboratory/NA-MPNN/raw/main/models/design_model/s_19137.pt))
  and `specificity`
  ([`models/specificity_model/s_70114.pt`](https://github.com/baker-laboratory/NA-MPNN/raw/main/models/specificity_model/s_70114.pt))
  checkpoints.

### Provenance

OFoldX provides a native re-implementation of the NA-MPNN architecture and
converts the upstream released checkpoints into OFoldX pretrained artifacts; it
does not train these weights. The architecture configuration
([`NAMPNNConfig`][ofoldx.models.nampnn.config.NAMPNNConfig]) is inferred directly
from the official checkpoint tensor shapes during conversion, and the converted
state dict is validated to load strictly against the native model. No additional
training-data or training-procedure metadata is distributed with these files;
consult the upstream project for details.

## Variants

Both official checkpoints share the `nampnn` architecture and differ only in the
released weights they carry:

| Variant       | Public name           | Checkpoint   | What differs                   |
| ------------- | --------------------- | ------------ | ------------------------------ |
| `design`      | NA-MPNN (design)      | `s_19137.pt` | General sequence-design model. |
| `specificity` | NA-MPNN (specificity) | `s_70114.pt` | Specificity-oriented model.    |

Select a variant with `--variant` at conversion time (see below).

## Conversion

`convert.py` downloads an official NA-MPNN checkpoint (or reads a local one),
infers the config from tensor shapes, converts the state dict to native OFoldX
keys, validates a strict load, and writes an OFoldX pretrained artifact
(`manifest.json`, `config.json`, `model.safetensors`):

```bash
# Convert the official design checkpoint (auto-downloaded and cached).
python -m ofoldx.models.nampnn.convert out/nampnn-design --variant design

# Convert the specificity checkpoint from a local file.
python -m ofoldx.models.nampnn.convert out/nampnn-specificity \
    --variant specificity --checkpoint /path/to/s_70114.pt
```

Useful flags: `--cache-dir` sets the download cache, `--dtype`
(`keep`/`float32`/`float16`/`bfloat16`) casts persisted weights, and
`--checkpoint-variant` records a variant label in `config.json`. Hub-upload flags
are also accepted to push the converted artifact.
