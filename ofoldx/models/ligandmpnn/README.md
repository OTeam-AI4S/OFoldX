---
library_name: ofoldx
tags:
  - design
  - protein-design
  - side-chain-packing
  - biology
pipeline_tag: other
license: mit
---

# LigandMPNN

LigandMPNN is a message-passing inverse-folding model that designs and scores
amino-acid sequences for a protein-ligand complex, conditioning on backbone
geometry and nearby ligand/context atoms. The converted sequence-design
checkpoint carries ligand and non-protein atom context.

Its `LigandMPNNDesigner` runs a k-nearest-neighbor graph encoder and an
autoregressive decoder that scores a given sequence, scores one masked residue
at a time, or samples sequences. The family also ships a `LigandMPNNPacker` that
predicts side-chain chi angles as a Von Mises mixture and denoises them into atom
coordinates.

## Usage

Design a sequence for a protein-ligand complex with the Tier-0 verb facade:

```python
import ofoldx

# Sequence design conditioned on backbone + ligand/atom context.
result = ofoldx.design("complex.cif", model="oteam/ligandmpnn")

# Side-chain packing around a fixed sequence.
packed = ofoldx.pack("backbone.cif", model="oteam/ligandmpnn-packer")
```

Or load the model explicitly via `AutoModel`:

```python
from ofoldx.models import AutoModel

designer = AutoModel.from_pretrained("oteam/ligandmpnn")
packer = AutoModel.from_pretrained("oteam/ligandmpnn-packer")
```

## Model Details

- **Task:** sequence design (`ligandmpnn`) and side-chain packing (`ligandmpnn_packer`)
- **Upstream project:** [LigandMPNN](https://github.com/dauparas/LigandMPNN)
- **Paper:** [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1)
- **License:** MIT for upstream LigandMPNN code and model parameters

### Training Data

LigandMPNN was trained on PDB assemblies as of 2022-12-16 from X-ray
crystallography or cryo-EM structures better than 3.5 A and shorter than 6,000
residues, with 30% sequence-identity clustering and non-protein atomic context
such as ligands, nucleic acids, and metals. OFoldX does not redistribute the
training set.

The side-chain packing checkpoint is distributed with the LigandMPNN model
parameters and shares the upstream LigandMPNN release provenance.

### Training Procedure

Upstream LigandMPNN uses categorical cross-entropy sequence-design training with
Adam, mixed precision/checkpointing, and released noisy-backbone checkpoint
variants. OFoldX converts released LigandMPNN checkpoints into `model.safetensors`;
it does not run LigandMPNN training.

The upstream packer predicts side-chain chi torsions from backbone, ligand atoms,
and sequence with a LigandMPNN-style architecture. OFoldX converts the released
side-chain packing checkpoint; it does not run training.

## Variants

This family covers two upstream architectures with distinct tensor sets and
artifact entrypoints:

| Architecture        | Public name         | Task               | What differs                                                                                                                                                                                            |
| ------------------- | ------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ligandmpnn`        | `ligandmpnn`        | Sequence design    | Ligand/atom-context sequence-design model. Checkpoint metadata: `k_neighbors=32`, `atom_context_num=25`; a `noiseXXX` suffix identifies the training-noise variant.                                     |
| `ligandmpnn_packer` | `ligandmpnn-packer` | Side-chain packing | Standalone `ligandmpnn_sc` packer that packs rotamers around a fixed sequence. Checkpoint metadata: `k_neighbors=32`, `atom_context_num=16`; a `noiseXXX` suffix identifies the training-noise variant. |

The two configs are kept intentionally separate because the released sequence
model and side-chain packer checkpoints have different tensor sets and different
artifact entrypoints.

## Citation

```bibtex
@article{dauparas2025atomic,
  author = {Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and others},
  title = {Atomic context-conditioned protein sequence design using LigandMPNN},
  journal = {Nature Methods},
  year = {2025},
  doi = {10.1038/s41592-025-02626-1}
}
```

## Conversion

Convert an official LigandMPNN `.pt` checkpoint into an OFoldX-native pretrained
directory with `convert.py`:

```bash
python -m ofoldx.models.ligandmpnn.convert \
    ligandmpnn_v_32_010_25.pt out/ligandmpnn \
    --checkpoint-variant ligandmpnn_v_32_010_25 \
    --dtype keep
```

Pass `--checkpoint-kind packer` to convert a side-chain packing checkpoint
instead of a sequence-design one (`auto` detects the packer weights), and use
the shared `--hub-*` upload flags to push the converted directory to the Hub.
