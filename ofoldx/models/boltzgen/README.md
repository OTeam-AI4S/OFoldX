---
library_name: ofoldx
tags:
  - design
  - protein-design
  - biology
pipeline_tag: other
license: mit
base_model: boltz-community/boltz-2
datasets:
  - boltzgen/boltzgen1_train
  - boltzgen/inference-data
---

# BoltzGen

BoltzGen is a structure-based biomolecular design family with two generators: a
diffusion structure designer that samples complex coordinates under design
conditioning, and an inverse-folding designer that decodes per-token residue
types from an input backbone.

The structure designer adds a v1 template module, a token-distance module, and
binding / design-mask / secondary-structure conditioning in the input embedder,
with opt-in confidence and affinity scoring heads. The inverse-folding path
encodes the structure into a kNN token graph and autoregressively samples
sequences.

## Usage

```python
import ofoldx

# Structure-conditioned generation with a diffusion design checkpoint.
result = ofoldx.design("seed_structure.cif", model="oteam/boltzgen-diverse", num_designs=8)

# Inverse folding: design sequences for a fixed backbone.
seqs = ofoldx.design("backbone.cif", model="oteam/boltzgen-inversefolding", design_chains="A")
```

The structure generators load through
[`BoltzGenDesigner`][ofoldx.models.boltzgen.model.BoltzGenDesigner] and the
inverse-folding generator through
[`BoltzGenInverseFoldDesigner`][ofoldx.models.boltzgen.model.BoltzGenInverseFoldDesigner];
`AutoModel` resolves the right class from the checkpoint's architecture key:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/boltzgen-diverse")
```

## Model Details

- **Task:** design
- **Upstream project:** [BoltzGen](https://github.com/HannesStark/boltzgen)
- **Source release:** [boltzgen/boltzgen-1](https://huggingface.co/boltzgen/boltzgen-1)
- **Paper:** [BoltzGen: Toward Universal Binder Design](https://doi.org/10.1101/2025.11.20.689494)
- **License:** MIT for upstream BoltzGen code.

### Training Data

BoltzGen builds on the Boltz-2 data pipeline. The public training release
includes `targets.zip` and `msa.zip` in the `boltzgen/boltzgen1_train` dataset
and molecule dictionaries in `boltzgen/inference-data`; the full large-model
recipe may also use additional distillation data. Per-artifact training-data
notes are given in [Variants](#variants).

### Training Procedure

Upstream BoltzGen trains a diffusion objective over randomly cropped
biomolecular structures with randomly selected design and conditioning regions,
covering a model suite that spans folding, design, and inverse folding. OFoldX
converts the released checkpoints; it does not run BoltzGen training.

## Variants

The published `boltzgen-1` release ships four checkpoints. The base `boltzgen`
key is the umbrella structure-design configuration that the published variants
inherit from; it carries no separately published weights of its own.

| Architecture key                  | Public name              | What differs                                                                                                                                        |
| --------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `boltzgen`                        | boltzgen                 | Base structure-design configuration; registry/umbrella key inherited by the published variants. No separate published checkpoint.                   |
| `boltzgen_diverse`                | boltzgen-diverse         | Diffusion design generator optimized for diverse structure-conditioned generation (`boltzgen1_diverse.ckpt`).                                       |
| `boltzgen_adherence`              | boltzgen-adherence       | Diffusion design generator optimized for adherence to conditioning signals for controllable binder and protein design (`boltzgen1_adherence.ckpt`). |
| `boltzgen_structuretrained_small` | boltzgen-structure-small | Small structure-conditioned generation checkpoint using the compact Miniformer trunk (`boltzgen1_structuretrained_small.ckpt`).                     |
| `boltzgen_ifold`                  | boltzgen-inversefolding  | Inverse-folding generator that designs sequences from backbone or structure inputs (`boltzgen1_ifold.ckpt`).                                        |

**boltzgen-diverse** — Converted BoltzGen diverse generator checkpoint for
diverse structure-conditioned design. OFoldX converts the released diverse
checkpoint; it does not run BoltzGen training.

**boltzgen-adherence** — Converted BoltzGen adherence generator checkpoint for
controllable binder and protein design. OFoldX converts the released adherence
checkpoint; it does not run BoltzGen training.

**boltzgen-structure-small** — Converted small BoltzGen structure-trained
checkpoint for structure-conditioned generation; corresponds to the released
small structure-trained checkpoint. OFoldX converts the released BoltzGen
structure-trained checkpoint; it does not run BoltzGen training.

**boltzgen-inversefolding** — Converted BoltzGen inverse-folding checkpoint for
designing sequences from structure inputs; inverse folding is released as a
distinct upstream training target. OFoldX converts the released inverse-folding
checkpoint; it does not run BoltzGen training.

## Citation

```bibtex
@article{stark2025boltzgen,
  author = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and others},
  title = {BoltzGen: Toward Universal Binder Design},
  year = {2025},
  doi = {10.1101/2025.11.20.689494},
  journal = {bioRxiv}
}
```

## Conversion

These weights are converted from the upstream BoltzGen release. `convert.py`
ports a BoltzGen `.ckpt` into an OFoldX pretrained directory with OFoldX
parameter names, routing the three top-level prefixes (`input_embedder` /
`inverse_folding_encoder` / `structure_module`) to the matching OFoldX modules
and reusing the Boltz-2 atom converters for the input embedder.

```bash
# Structure-design checkpoints (auto-detected):
python -m ofoldx.models.boltzgen.convert boltzgen1_diverse.ckpt out/boltzgen_diverse

# Inverse-folding checkpoint (requires --kind ifold):
python -m ofoldx.models.boltzgen.convert boltzgen1_ifold.ckpt out/boltzgen_ifold --kind ifold
```

Options: `--architecture` (one of the five architecture keys), `--kind`
(`auto` / `structure-design` / `ifold`), `--state-key` (`auto`, `state_dict`,
`model_state_dict`, `model`, or `root`), and `--dtype` (`keep`, `float32`,
`float16`, `bfloat16`), plus the shared hub-upload arguments.
