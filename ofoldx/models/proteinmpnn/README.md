---
library_name: ofoldx
tags:
  - biology
  - design
  - protein-design
pipeline_tag: other
license: mit
---

# ProteinMPNN

ProteinMPNN is an inverse-folding model that designs and scores amino-acid
sequences for a fixed protein backbone. A geometric message-passing encoder reads
the backbone over a k-nearest-neighbor residue graph, and an autoregressive decoder
produces per-residue probabilities, sequence scores, and sampled sequences.

Distinctive here are its backbone-feature encoder, encoder/decoder message-passing
stacks, and the full sampling surface: standard and single-residue scoring,
autoregressive sampling with fixed positions, per-residue bias, PSSM, and omit-AA
conditioning, plus symmetric and tied-position decoding. This family covers the
vanilla ProteinMPNN checkpoints, the SolubleMPNN checkpoints, and two membrane
ProteinMPNN checkpoints (global and per-residue membrane labels).

## Usage

```python
import ofoldx

# Design a sequence for a fixed backbone with the verb facade.
result = ofoldx.design("target.cif", model="oteam/proteinmpnn")
print(result.best.sequence)
```

```python
# Or load the model explicitly via AutoModel.
from ofoldx.models import AutoModel

model = AutoModel.from_pretrained("oteam/proteinmpnn")
```

Swap the repository id for a SolubleMPNN or membrane checkpoint to use the
corresponding variant.

## Model Details

- **Task:** design (fixed-backbone inverse folding)
- **Upstream project:** ProteinMPNN / LigandMPNN release — [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- **Source release:** [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)
- **Paper:** [Robust deep learning-based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187)
- **License:** MIT for upstream ProteinMPNN and LigandMPNN code/model parameters.

### Training Data

ProteinMPNN was trained for fixed-backbone protein sequence design on PDB biounits
with a 2021-08-02 data snapshot. OFoldX does not redistribute the training set.

### Training Procedure

Upstream ProteinMPNN trains graph-neural inverse-folding models with 48-neighbor
checkpoints and multiple Gaussian backbone-noise variants. OFoldX converts released
ProteinMPNN-family checkpoints into `model.safetensors`; it does not run ProteinMPNN
training.

## Variants

All variants share the 48-neighbor ProteinMPNN architecture. Checkpoint metadata
records `k_neighbors=48`; the `noiseXXX` suffix identifies the training-noise variant.

| Architecture                                | Project              | What differs                                                                                    |
| ------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------- |
| `proteinmpnn`                               | ProteinMPNN          | Vanilla fixed-backbone inverse folding; no per-residue node features.                           |
| `solublempnn`                               | SolubleMPNN          | ProteinMPNN-style model trained with transmembrane PDB entries excluded, for soluble backbones. |
| `proteinmpnn-membrane-global-noise020`      | Membrane ProteinMPNN | Adds a binary global soluble/membrane label to the input representation (0.20 Å noise).         |
| `proteinmpnn-membrane-per-residue-noise020` | Membrane ProteinMPNN | Adds three per-residue membrane-context classes — buried, interface, other (0.20 Å noise).      |

### SolubleMPNN

Converted SolubleMPNN sequence-design checkpoint for soluble protein backbones.
Repository: [dauparas/ProteinMPNN — soluble_model_weights](https://github.com/dauparas/ProteinMPNN/tree/main/soluble_model_weights).
Paper: [Computational design of soluble and functional membrane protein analogues](https://doi.org/10.1038/s41586-024-07601-y).

The SolubleMPNN work trains like ProteinMPNN on PDB assemblies as of 2021-08-02,
filtered to X-ray/cryo-EM structures better than 3.5 Å and fewer than 10,000
residues, while excluding annotated transmembrane PDB entries. OFoldX does not
redistribute the training set. Upstream SolubleMPNN follows ProteinMPNN-style
fixed-backbone inverse-folding training with 48-neighbor noisy-backbone checkpoint
variants. OFoldX converts released SolubleMPNN checkpoints into `model.safetensors`;
it does not run SolubleMPNN training.

### Membrane ProteinMPNN (global label)

Converted membrane ProteinMPNN checkpoint with a global soluble/membrane label.
Repository: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

This checkpoint is distributed in the LigandMPNN parameter release. The global-label
model adds a binary soluble/membrane feature to the ProteinMPNN-style input
representation. OFoldX does not redistribute the training set. Upstream training
follows the ProteinMPNN fixed-backbone inverse-folding setup with a global membrane
label and a 0.20 Å noisy-backbone checkpoint. OFoldX only converts the released
checkpoint.

### Membrane ProteinMPNN (per-residue labels)

Converted membrane ProteinMPNN checkpoint with per-residue membrane labels.
Repository: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

This checkpoint is distributed in the LigandMPNN parameter release. The per-residue
model adds three membrane-context classes per residue: buried, interface, and other.
OFoldX does not redistribute the training set. Upstream training follows the
ProteinMPNN fixed-backbone inverse-folding setup with per-residue membrane labels and
a 0.20 Å noisy-backbone checkpoint. OFoldX only converts the released checkpoint.

## Citation

```bibtex
@article{dauparas2022robust,
  author = {Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and others},
  title = {Robust deep learning-based protein sequence design using ProteinMPNN},
  journal = {Science},
  volume = {378},
  number = {6615},
  pages = {49--56},
  year = {2022},
  doi = {10.1126/science.add2187}
}
```

```bibtex
@article{solublempnn2024membrane,
  title = {Computational design of soluble and functional membrane protein analogues},
  journal = {Nature},
  year = {2024},
  doi = {10.1038/s41586-024-07601-y}
}
```

## Conversion

These are converted upstream checkpoints. To convert an official
ProteinMPNN-family `.pt` checkpoint into an OFoldX pretrained directory (writing
`model.safetensors` + `config.json`), use the `convert.py` CLI:

```bash
python -m ofoldx.models.proteinmpnn.convert \
    proteinmpnn_v_48_020.pt \
    out/proteinmpnn \
    --checkpoint-variant proteinmpnn_v_48_020
```

- `--checkpoint-variant` records the released checkpoint tag (e.g.
  `proteinmpnn_v_48_020`) in `config.json`.
- `--dtype {keep,float32,float16,bfloat16}` controls the stored weight precision
  (`keep` preserves the source dtype).

The converter also accepts the shared hub-upload flags for pushing the converted
directory to a model repository.
