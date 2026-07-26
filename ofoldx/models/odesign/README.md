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
and their protein, nucleic-acid, and ligand OInvFold compositions.

## Disclaimer

This is an unofficial OFoldX implementation of
[ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).
The upstream authors did not write, review, or endorse this implementation or
model card.

## Model Details

ODesign is an all-atom diffusion model for designing biomolecular interactions.
It combines atom-level chemical features, MSA information, geometric
constraints, and recycled Pairformer representations before an EDM denoiser
generates coordinates. Distogram and bond heads retain additional predictions
over the designed assembly. OFoldX composes each backbone artifact with the
released OInvFold model appropriate to its generated chemistry.

### Released Artifacts

The converter exposes four exact artifact selectors:

- `protein-flex` uses `ODesignModel` with a protein/flexible
  `ODesignProcessor`, then `ODesignProteinPipeline` assigns generated chains with
  a protein `OInvFoldPolymerPipeline`.
- `protein-rigid` uses the same model graph with a protein/rigid Processor and
  the same protein OInvFold composition.
- `nucleic-acid-rigid` uses the same model graph with a nucleic-acid/rigid
  Processor. `ODesignNucleicAcidPipeline` routes each generated chain to its
  exact DNA or RNA OInvFold child and realizes canonical nucleotide identities.
- `ligand-rigid` uses `ODesignModel` with `ODesignLigandProcessor`.
  `ODesignLigandPipeline` sends the generated coordinate slots to an
  `OInvFoldLigandPipeline` for element assignment.

Receptor mode and generated modality are persisted Processor facts, not runtime
model-topology switches. Protein, DNA, and RNA candidates become canonical
`Structure` values after OInvFold assignment. Ligand generation is deliberately
provisional: candidates retain ordered slot identities, coordinates, predicted
elements, probabilities, and ODesign bond logits, but their bonds, formal
charges, and component identity remain unresolved. OFoldX does not fabricate a
canonical molecule from those predictions.

### Links

- **Code**: [ofoldx.models.odesign](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/odesign)
- **Paper**: [ODesign: A World Model for Biomolecular Interaction Design](https://arxiv.org/abs/2510.22304)
- **Developed by**: Odin Zhang, Xujun Zhang, Haitao Lin, et al.
- **Model type**: All-atom, condition-guided biomolecular diffusion model
- **Original Repository**: [The-Institute-for-AI-Molecular-Design/ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

## Usage

The model depends on the [`ofoldx`](https://github.com/OTeam-AI4S/OFoldX)
library. Install it with pip:

```bash
pip install ofoldx
```

### Polymer Design

Use the concrete Pipeline because placement and seed/backbone/sequence axes are
family-specific:

```python
import ofoldx
from ofoldx.data import GeneratedSegment
from ofoldx.models.odesign import ODesignPlacements

pipeline = ofoldx.pipeline(
    "design",
    model="/path/to/converted-odesign-protein-flex",
)
placement = ODesignPlacements(
    {"designed_chain": {"design": GeneratedSegment("protein", 100, 100)}}
)
output = pipeline.design(
    None,
    placement=placement,
    num_backbones=2,
)
```

The nucleic-acid Pipeline uses the same placement form with `"dna"` or `"rna"`
generated segments.

### Provisional Ligand Design

Ligand artifacts take an explicit generated-atom request:

```python
import ofoldx
from ofoldx.models.odesign import ODesignGeneratedAtoms

pipeline = ofoldx.pipeline(
    "design",
    model="/path/to/converted-odesign-ligand-rigid",
)
output = pipeline.design(
    "target.cif",
    generated_atoms=ODesignGeneratedAtoms("ligand", 8, 16),
)
```

The returned ligand slots are not a chemically complete molecule until an
external chemical-realization step resolves connectivity, charge, and component
identity.

## Training Details

### Training Data

Refer to the [ODesign technical report](https://arxiv.org/abs/2510.22304) for the
upstream training corpus and data construction.

### Training Procedure

OFoldX preserves the released inference graph and exposes the differentiable
one-step denoiser. It does not add an OFoldX-specific training objective to the
model. Rigid and flexible receptor behavior is Processor conditioning rather
than a separate architecture.

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

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
