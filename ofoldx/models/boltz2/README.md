---
library_name: ofoldx
tags:
  - structure-prediction
  - protein-structure
  - binding-affinity
  - affinity-prediction
  - protein-ligand
  - biology
pipeline_tag: other
license: mit
base_model: boltz-community/boltz-2
---

# Boltz-2

Boltz-2-shaped structure-prediction model converted for the OFoldX runtime, together
with a Boltz-2 affinity model with a pair-only head for protein-ligand affinity scoring.

The structure-prediction weights (`boltz2`) are a _Converted Boltz-2 structure-prediction
checkpoint for biomolecular interaction modeling_: an atom-conditioned input embedder
builds the initial single/pair representations, a recycled trunk (template + MSA +
Pairformer) refines them, an EDM diffusion sampler draws atom coordinates, and a
token-level confidence head scores the sample. Distinctive to this family are its
atom-conditioned input embedder, V2 template embedder, and the EDM sampler with in-loop
weighted rigid alignment.

The affinity weights (`boltz2_affinity`) are a _Converted Boltz-2 affinity checkpoint for
protein-ligand binding-affinity prediction_: a full Boltz-2 model plus a pair-only head
that scores protein-ligand binding affinity.

## Usage

Structure prediction with the `boltz2` weights, via the Tier-0 verb facade:

```python
import ofoldx

result = ofoldx.fold("complex.cif", model="oteam/boltz2")
result.save_pdb("out.pdb")
```

Binding-affinity scoring with the `boltz2_affinity` weights:

```python
import ofoldx

score = ofoldx.score("complex.cif", model="oteam/boltz2-affinity")
```

Or load a specific architecture directly through `AutoModel`:

```python
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/boltz2")
```

## Model Details

- **Task:** structure prediction (`boltz2`) and binding-affinity prediction (`boltz2_affinity`).
- **Upstream project:** [Boltz](https://github.com/jwohlwend/boltz)
- **Source release:** [boltz-community/boltz-2](https://huggingface.co/boltz-community/boltz-2)
- **Base model:** [boltz-community/boltz-2](https://huggingface.co/boltz-community/boltz-2)
- **Paper:** [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
- **License:** MIT for upstream Boltz code and weights.

### Training Data

The Boltz-2 report describes structure data from PDB structures before 2023-06-01,
experimental NMR collections, MD trajectory datasets, AF2 monomer distillation, Boltz-1
complex distillation, and affinity data from PubChem, ChEMBL, BindingDB, and related
filtered assay sources. OFoldX does not redistribute the training set.

The Boltz-2 affinity heads use filtered assay data from sources such as PubChem, ChEMBL,
and BindingDB, with decoy generation and structural-confidence filters described by the
upstream report. OFoldX does not redistribute the training set.

### Training Procedure

Upstream Boltz-2 training is staged across structure, confidence, and affinity objectives.
This OFoldX artifact converts a released checkpoint into `model.safetensors` plus an OFoldX
manifest; it does not run Boltz-2 training.

Upstream affinity training detaches the trunk and optimizes regression and binary
binder/decoy objectives. OFoldX converts the released affinity checkpoint into
`model.safetensors` plus an OFoldX manifest; it does not run Boltz-2 training.

## Variants

| Architecture      | Public name       | Task                 | What differs                                                                                                                                                                                                          |
| ----------------- | ----------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `boltz2`          | `boltz2`          | Structure prediction | Full folding model: atom-conditioned input embedder, recycled template + MSA + Pairformer trunk, EDM diffusion sampler, and token-level confidence head.                                                              |
| `boltz2_affinity` | `boltz2-affinity` | Binding affinity     | Adds a pair-only head scoring protein-ligand binding affinity (a two-module ensemble of 8- and 4-block pair-only Pairformer stacks with a molecular-weight correction); templates and the B-factor head are disabled. |

The affinity checkpoint is a full Boltz-2 model plus an extra masked pair head;
`Boltz2AffinityConfig` subclasses `Boltz2Config` and turns off `use_templates` /
`use_templates_v2` / `predict_bfactor`.

## Citation

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}
```

## Conversion

These weights are converted from a public upstream Boltz-2 checkpoint. `convert.py` remaps
a public Boltz-2 `.ckpt` onto OFoldX's idiomatic parameter names (dropping weight-tied
duplicates) so the converted `state_dict` loads into the OFoldX modules. Convert a
checkpoint with:

```bash
python -m ofoldx.models.boltz2.convert boltz2_conf.ckpt out/boltz2 --architecture boltz2
```

The structure checkpoint converts from `boltz2_conf_final.ckpt`; the affinity checkpoint
converts from `boltz2_aff.ckpt` (pass `--architecture boltz2_affinity`). Both produce
`model.safetensors` plus an OFoldX manifest. `--state-key` selects the checkpoint state key
(`auto`, `state_dict`, `model_state_dict`, `model`, or `root`) and `--dtype` optionally
recasts the weights.
