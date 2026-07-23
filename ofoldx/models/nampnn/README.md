---
language:
  - protein
  - dna
  - rna
tags:
  - biology
  - inverse-folding
  - protein-design
  - dna-design
  - rna-design
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# NA-MPNN

NA-MPNN designs and scores sequences on fixed protein, DNA, RNA, and mixed
polymer backbones.

## Disclaimer

This is an unofficial OFoldX implementation of
[NA-MPNN](https://github.com/baker-laboratory/NA-MPNN). The upstream authors did
not write, review, or endorse this implementation, conversion, or model card.

## Model Details

The `ofoldx.nampnn` architecture pairs `NaMpnnModel` with
`NaMpnnProcessor`. `NaMpnnPipeline` exposes:

- `design`, returning `DesignResult`;
- `score` and `unconditional_probabilities`, returning
  `MpnnProbabilityResult`; and
- `specificity_profile`, returning `ScoreResult`.

The Model additionally exposes `encode`, differentiable
`forward`, and `sample`. `forward` returns
`NaMpnnOutput`, while `sample` returns
`NaMpnnSamplingOutput` with sampled tokens, in-pass log
probabilities, sampling probabilities, and decoding ranks.

### Variants

| Variant     | Released file                         |  Step | Released operation default |
| ----------- | ------------------------------------- | ----: | -------------------------: |
| design      | `models/design_model/s_19137.pt`      | 19137 |            temperature 0.1 |
| specificity | `models/specificity_model/s_70114.pt` | 70114 |            temperature 0.6 |

The two files are different weight distributions trained for different
scientific purposes, but they share one 123-tensor executable graph. The
converted Model config contains no variant or temperature field. The 0.1 and
0.6 values are defaults on the corresponding design and specificity operations
and remain explicit caller controls.

Both released training configurations use shared nucleic-acid tokens. DNA and
RNA A/C/G/T or U occupy token IDs 21-24, and the canonical row's polymer type
determines whether a sampled token realizes as `DA/DC/DG/DT` or
`A/C/G/U`. `NaMpnnProcessorConfig(shared_nucleic_tokens=False)`
uses RNA IDs 26-29 and the corresponding released smoothing semantics.

### Model Specification

| Property                | Value                    |
| ----------------------- | ------------------------ |
| Architecture            | `ofoldx.nampnn`          |
| Model                   | `NaMpnnModel`            |
| Processor               | `NaMpnnProcessor`        |
| Pipeline                | `NaMpnnPipeline`         |
| Tasks                   | `design`, `score`        |
| Vocabulary size         | 33                       |
| Hidden and feature size | 128                      |
| Encoder blocks          | 3 (`num_encoder_blocks`) |
| Decoder blocks          | 3 (`num_decoder_blocks`) |
| Neighbors               | 32 (`num_neighbors`)     |
| Atom slots per residue  | 16                       |
| Dropout                 | 0.1                      |

The Processor accepts a canonical `Structure` or a local PDB/mmCIF
path through the general structure reader. It retains the complete input
`MolecularSystem`, including non-polymer entities and covalent bonds,
while lowering protein, DNA, and RNA rows into this released atom order:

```text
N CA C O OP1 OP2 P O5' C5' C4' O4' C3' O3' C2' O2' C1'
```

Protein rows require N/CA/C/O; DNA rows require the eleven released
deoxyribose and phosphate sites; RNA rows additionally require O2'. Incomplete
rows and residues participating in explicit assembly bonds remain fixed.
Sampling is restricted to identities valid for each row's polymer type.

The Processor consumes `AllowedResidueIdentities`,
`TiedResidueIdentities`, explicit symmetry groups, tied weights,
per-residue bias, pair bias, and global token omissions. Shared instances of one
canonical entity are tied automatically. `DesignResult` candidates
preserve complete molecular identity and carry
`designed_sequence_confidence` and
`sequence_recovery` scores.

`score` evaluates a supplied sequence over independent
autoregressive orders. Evaluating each order independently corrects the released
batched implementation's reuse of its first autoregressive mask.
`unconditional_probabilities` returns the sequence-free,
all-equal-rank distribution. Both operations retain raw per-position logits and
effective decoding ranks and expose log probabilities, probabilities, their
mean, and population standard deviation. `specificity_profile` fixes protein
context, samples nucleic-acid rows, and returns per-position mean base
probabilities for adenine, cytosine, guanine, and thymine/uracil.

### Links

- [Official source](https://github.com/baker-laboratory/NA-MPNN)
- [Audited source revision](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef)
- [NA-MPNN preprint](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2)

## Usage

```python
import ofoldx

repository = "./nampnn-design"
design = ofoldx.design(
    "complex.cif",
    model=repository,
    num_candidates=8,
    seed=7,
)
probabilities = ofoldx.score(
    "complex.cif",
    model=repository,
    num_decoding_orders=8,
    seed=7,
)

pipe = ofoldx.pipeline("design", model=repository, device="cuda")
profile = pipe.specificity_profile(
    "complex.cif",
    num_samples=30,
    temperature=0.6,
    seed=7,
)
```

Direct typed execution remains ordinary PyTorch:

```python
from ofoldx import AutoModel, AutoProcessor

processor = AutoProcessor(repository, device="cuda")
model = AutoModel(repository, device="cuda")
batch = processor("complex.cif")
output = model(batch)
raw = model.sample(batch, num_samples=8)
```

Direct stochastic Model operations accept a caller-owned
`torch.Generator`; Pipeline operations additionally accept `seed`.

## Checkpoints and Conversion

Convert either acquired checkpoint with the same converter:

```bash
python -m ofoldx.models.nampnn.convert \
  s_19137.pt \
  ./nampnn-design
```

Conversion uses restricted PyTorch loading, derives the vocabulary, hidden and
feature sizes, block counts, and fixed 32-neighbor graph, translates the released
key mapping, and strict-loads the 123-tensor Model. It writes
`model.json`, `processor.json`, safetensors weights, and the
concrete `pipeline.json` marker. The converter has no
`--variant` option; choose the output repository name and operation
defaults explicitly. `--repo-id` optionally uploads the repository,
and `--private` requests a private Hub repository.

## Training Details

### Training Data

The inspected release identifies distinct design and specificity training
configurations but does not publish a complete per-checkpoint training-data
manifest in the material summarized here.

### Training Procedure

The released configurations differ in training purpose and step count.
`NaMpnnModel.forward` exposes training-time protein, DNA, and RNA
coordinate noise and optional protein-first decoding. OFoldX also exposes the
released label-smoothed loss primitive. These are training primitives, not a
claim that OFoldX reproduces every upstream data, augmentation, optimizer, or
schedule choice. Training orchestration belongs to the unified OFoldX Runner.

## Limitations

- NA-MPNN changes sequence identity on an existing fixed backbone. It does not
  generate coordinates or pack side chains.
- At least one complete supported polymer row is required; coordinate-free
  systems are not accepted.
- A canonical protein residue cannot realize as a nucleic acid, or vice versa,
  so cross-polymer tokens are masked before sampling.
- Released-weight comparison tooling covers an earlier NA-MPNN surface.
  Real-weight parity is not established for the current Model, Processor,
  Pipeline, and conversion behavior.

## Citation

Please cite the upstream preprint,
[RNA sequence design and protein-DNA specificity prediction with NA-MPNN](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2).

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and weights:** the top-level
  [NA-MPNN license](https://github.com/baker-laboratory/NA-MPNN/blob/aafa32c87b65046a02db152f6e6235a7078e5fef/LICENSE)
  grants MIT terms for NA-MPNN and separately records BSD-3-Clause terms for
  `cifutils.py`, `obutils.py`, and `geometry.py`,
  stating that the BSD notice covers its named source and model weights.
- Retain every applicable upstream notice when redistributing source or weights.
  Converted weights are not relicensed by OFoldX, and OFoldX provides no
  warranty.
