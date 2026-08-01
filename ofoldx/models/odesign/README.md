---
tags:
  - Biology
  - Protein Design
  - Nucleic Acid Design
  - Ligand Design
  - Biomolecular Structure
  - Diffusion
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# ODesign

Pre-trained model for biomolecular interaction design through coordinate diffusion followed by sequence or element assignment.

## Disclaimer

This is an UNOFFICIAL implementation of [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304) by Odin Zhang, Xujun Zhang, Haitao Lin, et al.

The OFFICIAL repository of ODesign is at [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).

**The team releasing ODesign did not write this model card for this model so this model card has been written by the OFoldX team.**

## Model Details

ODesign uses an all-atom diffusion model for biomolecular interaction design. Its four released variants share the
`ODesignModel` architecture and differ in receptor flexibility and the type of molecule being generated. Inverse-folding
children assign sequences or elements after coordinate generation.

The protein routes use ProteinMPNN without ligand atom context and LigandMPNN otherwise. The released DNA route uses
OInvFold, the RNA route uses gRNAde, and the provisional ligand route uses ligand OInvFold. Please refer to the
[Training Details](#training-details) section for more information on the training process.

### Variants

- **[oteam/odesign-protein-flex](https://huggingface.co/oteam/odesign-protein-flex)**: Protein backbone generation with flexible receptor context and MPNN sequence assignment.
- **[oteam/odesign-protein-rigid](https://huggingface.co/oteam/odesign-protein-rigid)**: Protein backbone generation
  with rigid receptor context and the same context-selected MPNN assignment.
- **[oteam/odesign-dna-rigid](https://huggingface.co/oteam/odesign-dna-rigid)**: DNA generation with rigid receptor
  context and OInvFold sequence assignment.
- **[oteam/odesign-rna-rigid](https://huggingface.co/oteam/odesign-rna-rigid)**: RNA generation with rigid receptor
  context and gRNAde sequence assignment.
- **[oteam/odesign-ligand-rigid](https://huggingface.co/oteam/odesign-ligand-rigid)**: Provisional ligand coordinate generation with rigid context and element assignment.

### Model Specification

| Single Size | Pair Size | Pairformer Blocks | MSA Stages | Token Denoiser Blocks | Atom Encoder Blocks | Atom Decoder Blocks | Num Parameters (M) |
| ----------- | --------- | ----------------- | ---------- | --------------------- | ------------------- | ------------------- | ------------------ |
| 384         | 128       | 48                | 4          | 24                    | 3                   | 3                   | 355.687            |

The parameter count is for the coordinate-generation model.

### Links

- **Code**: [ofoldx.models.odesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/odesign)
- **Paper**: [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304)
- **Developed by**: Odin Zhang, Xujun Zhang, Haitao Lin, et al.
- **Model type**: All-atom coordinate diffusion with sequence or element assignment
- **Original Repository**: [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

## Usage

The model file depends on the [`ofoldx`](https://ofoldx.valtech.ai/) library. You can install it using pip:

```bash
pip install ofoldx
```

Before reading a structure, set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact.

### Direct Use

#### Polymer Design

You can use this model directly with a pipeline for polymer design over instance `B` of a conditioning complex:

```python
import ofoldx
from ofoldx.io import read_structure
from ofoldx.models.design import PositionSelection

structure, _ = read_structure("complex.cif")
pipeline = ofoldx.pipeline("design", model="oteam/odesign-protein-flex")
result = pipeline(structure, design=PositionSelection(instances=("B",)), num_backbones=2, num_sequences=4, seed=7)
```

Protein, DNA and RNA candidates contain canonical `Structure` predictions and the inverse folder's named scores.
Without a `design` selection, every position of the route's monomer type is generated. For de novo generation, supply
a typed `PolymerSequence` of unknown residues; the [chemistry guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/chemistry/README.md) describes these inputs.

Rigid context retains its source coordinates; flexible context can move. Regeneration clears old observations for
moved atoms that the model does not place, while explicitly fixed atoms retain their coordinates and observations.

#### Provisional Ligand Design

You can use this model directly with a pipeline for provisional ligand design with 8–16 atom slots around an observed pocket:

```python
import ofoldx
from ofoldx.io import read_structure

structure, _ = read_structure("pocket.cif")
pipeline = ofoldx.pipeline("design", model="oteam/odesign-ligand-rigid")
result = pipeline(structure, generated_atoms=range(8, 17), seed=7)
```

Ligand candidates contain ordered coordinates, predicted elements, per-atom probabilities and pair bond logits.
Their bonds, formal charges and chemical-component identities remain undetermined; they are provisional designs.

### Downstream Use

#### Alternative Inverse Folding

`ODesignOInvFoldProteinPipeline` and `ODesignOInvFoldRnaPipeline` combine ODesign generation with independently runnable
OInvFold alternatives. They are separate from the released protein MPNN and RNA gRNAde routes. Complete repositories
include their required children; use `load_component` to load one component independently. Backbone decoding returns
`ODesignBackboneSample` objects containing the `structure` and any `unbound_linkers`.

The [pipeline guide](https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md) describes staged execution and composition. Family pipeline docstrings
specify sample counts, seed order, beam decoding and component roles.

### Checkpoint Conversion

Convert a protein-flex checkpoint with the released MPNN children:

```bash
python -m ofoldx.models.odesign.convert protein-flex checkpoint.pt ./odesign-protein-flex \
  --protein-designer oteam/proteinmpnn --ligand-designer oteam/ligandmpnn
```

Run `python -m ofoldx.models.odesign.convert --help` for the rigid, nucleic-acid, ligand and explicit OInvFold alternatives.

## Training Details

ODesign learns coordinate denoising together with geometric and chemical reconstruction objectives.

### Training Data

ODesign was trained on [PDB](https://www.rcsb.org/) crystal structures released before 30 September 2021 and distilled
structures predicted by AlphaFold 2 and OpenFold. The [official repository](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign#training) provides the training data.

### Training Procedure

`ODesignModel.forward` corrupts explicit coordinate targets and performs differentiable denoising. `odesign_loss`
combines the five released objectives: weighted coordinate error, covalent-bond distance, smooth lDDT, distogram
classification and binary token-bond prediction.

These operations support custom training loops without reproducing the upstream data curation or complete training
program. Known zero-occupancy
coordinates are unresolved by default for conditioning and truth; set `filter_by_occupancy=False` to ignore occupancy.

## Citation

```bibtex
@misc{zhang2025odesign,
  title={ODesign: A World Model for Biomolecular Interaction Design},
  author={Zhang, Odin and Zhang, Xujun and Lin, Haitao and others},
  year={2025},
  eprint={2510.22304},
  archivePrefix={arXiv},
  primaryClass={q-bio.BM}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OFoldX supports your research, please cite the OFoldX project as follows:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for any questions or comments on the model implementation or model card.

Please contact the authors of the [ODesign paper](https://arxiv.org/abs/2510.22304) for questions or comments on the paper/model.

## License

The model weights, implementation, configuration, and documentation distributed by OFoldX are licensed under the
[GNU Affero General Public License](https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md).

For additional terms and clarifications, please refer to our
[License FAQ](https://github.com/OTeam-AI4S/OFoldX/blob/main/license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
