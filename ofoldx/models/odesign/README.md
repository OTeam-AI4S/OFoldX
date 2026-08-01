---
language:
  - protein
  - dna
  - rna
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

Native PyTorch implementation of the four released ODesign backbone artifacts
and their released ProteinMPNN, LigandMPNN, gRNAde, and OInvFold compositions.

## Disclaimer

This is an unofficial OFoldX implementation of
[ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).
The upstream authors did not write, review, or endorse this implementation or
model card.

## Model details

ODesign is an all-atom coordinate-diffusion model for biomolecular interaction
design. One `ODesignModel` graph serves four released artifacts; receptor mode
and generated chemistry are Processor facts, not switches that change the model
topology.

| Released artifact  | Processor                                         | Concrete composition                                          |
| ------------------ | ------------------------------------------------- | ------------------------------------------------------------- |
| Protein flex       | `ODesignProcessor(artifact="protein_flex")`       | ProteinMPNN without ligand atom context, otherwise LigandMPNN |
| Protein rigid      | `ODesignProcessor(artifact="protein_rigid")`      | ProteinMPNN without ligand atom context, otherwise LigandMPNN |
| Nucleic-acid rigid | `ODesignProcessor(artifact="nucleic_acid_rigid")` | DNA uses OInvFold; RNA uses gRNAde                            |
| Ligand rigid       | `ODesignLigandProcessor`                          | Ligand OInvFold assigns provisional atom elements             |

DNA and RNA are separate concrete Pipelines because their inverse-folding
models differ. `ODesignOInvFoldProteinPipeline` and
`ODesignOInvFoldRnaPipeline` retain the independently runnable OInvFold
alternatives without presenting them as the official ODesign routes.

Protein, DNA, and RNA results contain canonical `Structure` predictions. Ligand
results remain provisional: they retain ordered coordinates, predicted elements,
per-atom probabilities, and generated-slot pair bond logits, but do not invent
bonds, formal charges, or a chemical-component identity.

## Conversion

Choose the released artifact and supply the repositories owned by its concrete
Pipeline:

```bash
python -m ofoldx.models.odesign.convert protein-flex \
  released-odesign.ckpt /path/to/odesign-protein-flex \
  --protein-designer /path/to/proteinmpnn \
  --ligand-designer /path/to/ligandmpnn
```

The other official selectors are `protein-rigid`, `dna-rigid`, `rna-rigid`,
and `ligand-rigid`. The explicit alternatives are
`oinvfold-protein-flex`, `oinvfold-protein-rigid`, and
`oinvfold-rna-rigid`. Run the converter with `--help` for each route's required
child repository.

## Usage

### Polymer design

Placement is an ordinary `PlacementSpec`; no ODesign-specific request wrapper
is required:

```python
import ofoldx
from ofoldx.data import GeneratedSegment

pipeline = ofoldx.pipeline(
    "design",
    model="/path/to/odesign-protein-flex",
)
placement = {
    "designed_chain": {
        "design": GeneratedSegment("protein", 100, 100),
    },
}
result = pipeline.design(
    None,
    placement=placement,
    num_backbones=2,
    num_sequences=4,
)
```

Use the converted DNA or RNA repository and change the generated segment to
`"dna"` or `"rna"` for those concrete routes.

### Provisional ligand design

Ligand atom count is the direct public argument. An integer fixes it; a `range`
lets the Processor sample it from the caller's generator.

```python
import ofoldx

pipeline = ofoldx.pipeline(
    "design",
    model="/path/to/odesign-ligand-rigid",
)
result = pipeline.design(
    "target.cif",
    generated_atoms=range(8, 17),
)
```

## Training

`ODesignProcessor.process_training` produces the tensor-native model Input and
the coordinate Target. `ODesignModel.forward` implements stochastic target
corruption and differentiable denoising. `odesign_loss` combines the five
released objectives: weighted coordinate error, covalent-bond distance, smooth
lDDT, distogram classification, and binary token-bond prediction. A caller may
use those functions with the unified Runner or in a custom training loop; OFoldX
does not reproduce the upstream training program.

Explicitly zero-occupancy coordinates are unresolved by default for both
conditioning and training targets. Set `filter_by_occupancy=False` only when the
source occupancy field should be ignored.

## Links

- **Code**: [ofoldx.models.odesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/odesign)
- **Paper**: [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304)
- **Original repository**: [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

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

## License

This OFoldX implementation is licensed under the
[GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
