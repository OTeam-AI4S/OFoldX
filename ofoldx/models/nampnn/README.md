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

# NaMpnn

NaMpnn designs and scores sequences on fixed protein, DNA, RNA, and mixed
polymer backbones. This is an unofficial OFoldX conversion of
[baker-laboratory/NA-MPNN](https://github.com/baker-laboratory/NA-MPNN). The
upstream authors did not create, review, or endorse this conversion or model card.

## Repository contract

- Architecture ID: `ofoldx.nampnn`
- Model / Processor: `NaMpnnModel` / `NaMpnnProcessor`
- Standard tasks: `design`, `score`
- Results: `DesignResult`, `MpnnProbabilityResult`, `ScoreResult`

The design and specificity checkpoints are repository variants of one 123-tensor
graph. They differ in training data/procedure and their released inference
temperature: `0.1` for design and `0.6` for specificity. That temperature is stored
in `model.json`; neither checkpoint creates another architecture.

Both released training configs use shared nucleic-acid tokens. DNA and RNA A/C/G/T
or U occupy token IDs 21–24; the canonical row's polymer type determines whether a
sampled token realizes as `DA/DC/DG/DT` or `A/C/G/U`. The upstream
`na_shared_tokens=0` control remains available through
`NaMpnnProcessorConfig(shared_nucleic_tokens=False)`, which uses RNA IDs 26–29 and
the matching released smoothing semantics.

## Inputs and Results

The Processor accepts one canonical `Structure` or a local PDB/mmCIF path. It keeps
the complete input `MolecularSystem`, including non-polymer entities and covalent
bonds, while lowering every protein, DNA, and RNA residue into the released 16-atom
order:

```text
N CA C O OP1 OP2 P O5' C5' C4' O4' C3' O3' C2' O2' C1'
```

Protein rows require N/CA/C/O, DNA rows require the eleven released deoxyribose and
phosphate sites, and RNA rows additionally require O2'. Incomplete rows and residues
participating in explicit assembly bonds remain fixed. The Processor consumes hard
`AllowedResidueIdentities` and `TiedResidueIdentities`; shared instances of one
canonical entity are tied automatically. Sampling is restricted to identities valid
for each row's polymer type.

`DesignResult` contains complete coordinate-free `MolecularSystem` candidates,
aligned condition outcomes and score mappings, immutable sampling-probability
diagnostics, and run provenance. Unchanged entities, instances, component graphs,
and bonds survive exactly.

`score` and `unconditional_probabilities` return the shared `MpnnProbabilityResult`.
`score` evaluates the supplied sequence autoregressively over `num_decoding_orders`
independent orders, while `unconditional_probabilities` returns the sequence-free
all-equal-rank distribution. Each retains the raw per-position `logits` and effective
decoding ranks and exposes `log_probabilities`, `probabilities`, `mean_probabilities`,
and `probability_standard_deviation`. Evaluating each requested order independently
intentionally fixes the released batched `score` implementation's reuse of its first
autoregressive mask; each native order matches a separate released single-order call.

`specificity_profile` returns a `ScoreResult` whose `Metric` values are the mean
predicted base PPM (`adenine`, `cytosine`, `guanine`, `thymine_uracil`) over the
sampled nucleic-acid rows.

## Usage

```python
import ofoldx

design = ofoldx.design(
    "complex.cif",
    model="oteam/nampnn-design",
    num_candidates=8,
    seed=7,
)
print(design.candidates[0].sequences)

probabilities = ofoldx.score(
    "complex.cif",
    model="oteam/nampnn-design",
    seed=7,
)
print(probabilities.mean_probabilities.shape)
```

The upstream specificity protocol is sequence sampling, not autoregressive
scoring. `model.specificity_profile(batch, num_samples=30, seed=7)` fixes the protein
context, samples nucleic-acid rows, and returns the mean predicted PPM directly.

Direct typed execution remains ordinary PyTorch:

```python
import torch
from ofoldx import AutoModel, AutoProcessor

processor = AutoProcessor("oteam/nampnn-design", device="cuda", dtype=torch.bfloat16)
model = AutoModel("oteam/nampnn-design", device="cuda", dtype=torch.bfloat16)
batch = processor("complex.cif")
output = model(batch)
result = model.design(batch, seed=7)
raw = model.sample(batch, num_samples=8, seed=7)
conditional = model.score(batch, num_decoding_orders=8, seed=7)
unconditional = model.unconditional_probabilities(batch, num_decoding_orders=8, seed=7)
```

`sample` exposes the released sequences, log probabilities, sampling
probabilities, and decoding orders as the shared `MpnnSamplingOutput`. Position bias and pair
bias enter through the Processor; hard allowed/fixed and tied identities enter as
Conditions; `tied_weights` supplies the released weighted-symmetry control.

`model(batch)` remains an ordinary differentiable PyTorch call. Training objectives,
augmentation, targets, and metrics are intentionally deferred to OFoldX's unified
Runner rather than published as family-specific Model Hub APIs.

## Released checkpoints and parity

The behavior audit used source revision
[`aafa32c87b65046a02db152f6e6235a7078e5fef`](https://github.com/baker-laboratory/NA-MPNN/tree/aafa32c87b65046a02db152f6e6235a7078e5fef).

| Variant | Released file | Step | Default temperature |
| --- | --- | ---: | ---: |
| design | `models/design_model/s_19137.pt` | 19137 | 0.1 |
| specificity | `models/specificity_model/s_70114.pt` | 70114 | 0.6 |

The maintained
[`scripts/parity/nampnn_released.py`](../../../scripts/parity/nampnn_released.py)
comparison covers both files on CPU. It exercises repository conversion and reload,
teacher-forced and unconditional likelihoods, raw sampling, predicted specificity
PPMs, the explicit training loss, position/pair bias, weighted symmetry, separate
RNA tokens, the neighbor override, protein-first decoding, and canonical
independent-order sequence NLL. It reads caller-owned assets from `~/reference` and
`~/weights`; generated reports remain local. This production-weight run is
deliberately outside frequent CI.

Convert an acquired local checkpoint into `model.json`, `processor.json`,
safetensors weights, and this card:

```bash
python -m ofoldx.models.nampnn.convert \
  NA-MPNN/models/design_model/s_19137.pt \
  ./nampnn-design
```

For a renamed or copied checkpoint, pass `--variant design` or
`--variant specificity`. Conversion uses restricted PyTorch loading, translates
only the released key mapping, and lets strict state-dict loading authoritatively
check every target key and shape. Uploading is a separate operation.

## Limitations

- NaMpnn designs sequence identity on an existing fixed backbone; it does not
  generate coordinates or pack side chains.
- It requires at least one complete supported polymer row. Coordinate-free systems
  are not accepted.
- The canonical result cannot represent a protein residue sampled as nucleic acid,
  or vice versa, so OFoldX masks cross-polymer tokens before sampling.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Provenance and terms

The upstream repository describes the work as “RNA sequence design and protein–DNA
specificity prediction with NA-MPNN” and links
[bioRxiv 2025.10.03.679414v2](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2).

OFoldX code and documentation are `AGPL-3.0-or-later`. The audited upstream
repository's top-level license grants MIT terms for NA-MPNN and separately records
BSD-3-Clause terms for `cifutils.py`, `obutils.py`, and `geometry.py`, stating that
the latter notice covers its named source and model weights. Publishers must retain
the applicable upstream notices and recheck the exact source revision. OFoldX does
not relicense upstream weights and provides no warranty.

## Limitations & intentional upstream divergences

- NaMpnn is **in transition**: it currently remains unregistered pending its family
  result carrier, so it is not yet on the registration or Pipeline publication
  surface.
- It designs sequence identity on an existing fixed backbone; it does not generate
  coordinates or pack side chains, requires at least one complete supported polymer
  row, and masks cross-polymer tokens before sampling (a protein residue cannot be
  sampled as nucleic acid or vice versa).
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** the NA-MPNN repository grants MIT for NA-MPNN and
  separately records BSD-3-Clause for `cifutils.py`, `obutils.py`, and `geometry.py`,
  stating that the BSD notice covers its named source and model weights
  ([NA-MPNN LICENSE](https://github.com/dauparas/NA-MPNN/blob/main/LICENSE)).
- **Redistribution status: GREEN** (permissive MIT + BSD-3), but publication is
  gated by the in-transition registration work above rather than by license. Retain
  the applicable upstream notices; converted weights are not relicensed as AGPL.
