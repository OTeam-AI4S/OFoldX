---
language:
  - protein
  - dna
  - rna
tags:
  - Biology
  - Inverse Folding
  - Protein Design
  - Nucleic Acid Design
  - Ligand Design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# OInvFold

Native PyTorch implementation of the four released OInvFold inverse-folding
checkpoints for protein, DNA, RNA, and ligand backbones.

## Disclaimer

This is an unofficial OFoldX implementation of the OInvFold models released
with [ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign).
The upstream authors did not write, review, or endorse this implementation or
model card.

## Model Details

OInvFold is an independently runnable graph model family. It predicts polymer
identities or ligand atom elements from fixed coordinates; it is not an alias
for ProteinMPNN, LigandMPNN, or gRNAde.

### Released Checkpoints

| Checkpoint | Model and Processor pair                                                  | Decoded output                                                                         |
| ---------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Protein    | `OInvFoldPolymerModel` and `OInvFoldPolymerProcessor(modality="protein")` | Protein sequence and a `Structure` with only design-masked residue identities replaced |
| DNA        | `OInvFoldPolymerModel` and `OInvFoldPolymerProcessor(modality="dna")`     | DNA sequence and canonical nucleotide identities                                       |
| RNA        | `OInvFoldPolymerModel` and `OInvFoldPolymerProcessor(modality="rna")`     | RNA sequence and canonical nucleotide identities                                       |
| Ligand     | `OInvFoldLigandModel` and `OInvFoldLigandProcessor`                       | Ordered atom-element assignments with unresolved bonds                                 |

Protein, DNA, and RNA share the registered `oinvfold` graph and its
25-token unified vocabulary. Their repository Processor configuration fixes the
coordinate convention and allowed chemistry. Ligand uses the distinct
`oinvfold_ligand` graph and released 37-token element vocabulary.

### Decoding and Scores

`OInvFoldPolymerPipeline.design` and `OInvFoldLigandPipeline.design` expose the
released decoding controls without conflating them:

- With `use_beam=False`, decoding is greedy and returns one candidate; `topk`
  does not request additional greedy candidates.
- With `use_beam=True`, `topk` is the beam width and controls the number of
  returned candidates.
- Candidate selection always uses the released unscaled logits and raw log
  probabilities.
- `temperature` changes only the returned `probabilities` view. It does not
  change the selected greedy or beam sequences.
- Every candidate names both `total_log_probability` and
  `mean_log_probability`; OFoldX does not collapse the upstream scoring
  conventions into one ambiguous score.

Raw results retain logits, raw log probabilities, and temperature-view
probabilities.

### Ligand Identity Boundary

The ligand Pipeline consumes caller-owned coordinates and slot identifiers, or
atom sites from a known input `Structure`. Its candidates retain slot identifiers,
coordinates, predicted elements, and per-atom probabilities, while
`bond_state` remains `"unresolved"`. Element assignment alone does not determine
bond topology, formal charge, or a canonical chemical-component identity, so
the Pipeline never fabricates a `Molecule` or generated-atom `Structure`.

## Usage

Install OFoldX:

```bash
pip install ofoldx
```

### Polymer Inverse Folding

Load the converted checkpoint repository whose Processor has the desired
protein, DNA, or RNA modality:

```python
import ofoldx
from ofoldx.io import read_structure

backbone = read_structure("backbone.cif")
pipeline = ofoldx.pipeline(
    "design",
    model="/path/to/converted-oinvfold-protein",
)
result = pipeline.design(
    backbone,
    use_beam=True,
    topk=4,
    temperature=0.8,
)
```

### Ligand Element Assignment

For a known canonical ligand, `design_structure` extracts its observed atom
sites while keeping decoded connectivity unresolved:

```python
import ofoldx
from ofoldx.io import read_structure

ligand = read_structure("ligand.cif")
pipeline = ofoldx.pipeline(
    "design",
    model="/path/to/converted-oinvfold-ligand",
)
result = pipeline.design_structure(ligand)
```

Call `pipeline.design(coordinates, slot_ids=(...))` instead when the coordinates
belong to provisional atoms that do not yet have a canonical molecular graph.

## Conversion

The converter selects one of the four checkpoint modalities explicitly:

```bash
python -m ofoldx.models.oinvfold.convert \
  protein released.ckpt /path/to/converted-oinvfold-protein
```

Replace `protein` with `dna`, `rna`, or `ligand` for the corresponding released
checkpoint. Conversion writes an ordinary routable Model/Processor/Pipeline
repository.

## Training Details

OFoldX preserves the checkpoint-backed encoder corruption and provides the
released masked polymer negative log likelihood as the built-in Runner
objective. Polymer and ligand `encode`/`forward`, and the polymer
autoregressive operation, accept one optional `torch.Generator` that controls
every per-block graph corruption draw.

## Limitations

The built-in Runner objective covers polymer identity prediction only; ligand
graphs carry no target atom identities. Ligand decoding assigns elements only;
connectivity, formal charge, and canonical component identity remain unresolved.

## Citation

Please cite the ODesign work and repository accompanying the released OInvFold
checkpoints.

If OFoldX supports your research, also cite the OFoldX software:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

This model implementation is licensed under the
[GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
