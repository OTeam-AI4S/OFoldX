---
language: protein
tags:
  - biology
  - protein-design
  - inverse-folding
  - protein-ligand
license: mit
library_name: ofoldx
pipeline_tag: other
---

# LigandMpnn

LigandMpnn designs protein sequences from a fixed backbone and nearby
non-protein atomic context. This is an unofficial OFoldX reimplementation of
[LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1). The upstream authors did
not create, review, or endorse this implementation or model card.

## Qualification

The sequence architecture has a canonical `LigandMpnnProcessor`, a differentiable
`LigandMpnnModel.forward` returning the shared `MpnnOutput`, one standard
autoregressive `design` path, all four released autoregressive/single-residue
probability workflows returning the shared `MpnnProbabilityResult`, and a
complete tensor `sample` output carrying probabilities, log probabilities, and the
actual decoding order. `design` accepts direct per-residue `tied_weights` rather
than forcing one symmetry weighting policy. It also has strict in-memory checkpoint
conversion into a standard repository. It is registered as `ofoldx.ligandmpnn`
for `design`.

The upstream project also distributes an independently loaded side-chain Packer.
It accepts sequences from every official designer type and therefore belongs to the
separate OFoldX `MpnnPacker` family, not to the LigandMpnn Model. The complete
released design-to-pack composition remains part of the upstream surface audit.
The audited upstream repository is inference-only; reproducing unpublished training
choices is not a Model Hub requirement.

The maintained
[`scripts/parity/ligandmpnn.py`](../../../scripts/parity/ligandmpnn.py)
comparison reads the official source from `~/reference` and checkpoints from
`~/weights`; generated reports remain local.

## Inputs and outputs

`LigandMpnnProcessor` accepts a canonical `Structure` or a local PDB/mmCIF path.
It preserves full molecular identity while lowering protein N/CA/C/O coordinates,
observed atom37 coordinates, residue and chain indices, and the nearest 25
non-protein heavy atoms. Declared water molecules are excluded. Hard
`AllowedResidueIdentities` and `TiedResidueIdentities` conditions are supported.

`design(...)` implements the released autoregressive sampler and returns the shared
`DesignResult`. Each candidate carries a complete coordinate-free `MolecularSystem`
that retains ligands and topology, realizes tied sequences, records aligned condition
outcomes and score mappings, and carries sampling diagnostics and provenance.

The four probability operations `autoregressive_probabilities`,
`sequence_free_autoregressive_probabilities`, `conditional_probabilities`, and
`unconditional_probabilities` each return the shared `MpnnProbabilityResult`. Over
`num_decoding_orders` evaluations they retain the raw per-position `logits` and the
effective decoding ranks and expose `log_probabilities`, `probabilities`,
`mean_probabilities`, and `probability_standard_deviation`. The released command
saves those raw aligned distributions, their mean, and population standard deviation
rather than an aggregate NLL, so LigandMpnn publishes no scalar `score`/`ScoreResult`
path.

## Usage

```python
from ofoldx import pipeline
from ofoldx.models.ligandmpnn.convert import convert_ligandmpnn_checkpoint

convert_ligandmpnn_checkpoint(
    "ligandmpnn_v_32_010_25.pt",
    "ligandmpnn",
)

design_pipeline = pipeline("design", model="ligandmpnn", device="cuda")

candidates = design_pipeline.design("target.cif", num_candidates=4, seed=7)
probabilities = design_pipeline.autoregressive_probabilities(
    "target.cif", num_decoding_orders=8, seed=7
)
```

The ordinary tensor path remains available and differentiable:

```python
from ofoldx import AutoModel, AutoProcessor

model = AutoModel("ligandmpnn", device="cuda")
processor = AutoProcessor("ligandmpnn", device="cuda")
batch = processor("target.cif")
output = model(batch)
output.logits.square().mean().backward()

sampled = model.sample(batch, num_samples=4)
conditional = model.conditional_probabilities(batch, num_decoding_orders=8)
autoregressive = model.autoregressive_probabilities(batch, num_decoding_orders=8)
```

That example demonstrates gradient flow; it is not presented as the unpublished
official training objective. Training orchestration is outside the current Model Hub
gate.

## Real-checkpoint parity

The maintained script compares all four released sequence checkpoints, canonical
Processor, and public APIs with the inspected official source. Its
comparison surface includes:

- backbone, sequence, masks, ligand coordinates/types/masks, interface identity,
  relative residue offsets, and same-chain features;
- teacher-forced forward log probabilities;
- autoregressive and target-specific log probabilities, both with sequence context
  and with backbone context only;
- four-order sequence and ligand-interface scoring;
- full autoregressive design tokens, sampling probabilities, per-step log
  probabilities, and actual decoding order.
- atom context enabled and disabled, side-chain context, and non-uniform tied
  weights.

Real-weight parity runs outside frequent CI. Frequent tests use tiny self-contained
fixtures to cover shared Hub, Result, Condition, and standard-Pipeline invariants;
family-specific qualification lives in the maintained parity scripts above.

## Side-chain packing composition

The released workflow can pass LigandMpnn designs to the independent
`ligandmpnn_sc_v_32_002_16.pt` Packer, but that checkpoint is not part of the
LigandMpnn Model graph. Its implementation and direct packing API live with
`ofoldx.models.mpnn_packer`; a concrete composition owns the designer × pack
cardinality and transfer of selected sequences.

[`scripts/parity/mpnn_packer.py`](../../../scripts/parity/mpnn_packer.py)
compares all 215 converted tensors, the default context-enabled Processor lowering,
public encode/decode/forward, all eight sampling fields, and canonical `PackResult`
realization against the released checkpoint. The script lists its intentionally
uncovered diagnostic cases explicitly; repository round-trip evidence is maintained
separately and is not attributed to that comparison.

## Upstream provenance

- Official repository: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)
- Audited source revision: `26ec57ac976ade5379920dbd43c7f97a91cf82de`
- Official checkpoint host: `https://files.ipd.uw.edu/pub/ligandmpnn`
- Validated sequence checkpoints: `ligandmpnn_v_32_{005,010,020,030}_25.pt`
- Validated packer checkpoint: `ligandmpnn_sc_v_32_002_16.pt`

The audited revision records the source inspected for this behavior inventory. It
does not pin an OFoldX dependency or constrain which checkout a parity run may use.

The sequence checkpoints use the same executable architecture and differ by the
Gaussian coordinate-noise level used during upstream training. That value is weight
provenance, not a runtime Model configuration field.

## Provenance, terms, and redistribution

OFoldX's Python implementation and documentation are licensed under
`AGPL-3.0-or-later`. The audited upstream LigandMPNN code and released checkpoint
files are distributed under the upstream
[MIT License](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE).
Converted weights are not relicensed as AGPL; preserve the upstream notice and
recheck the exact source revision before redistribution.

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

## Limitations & intentional upstream divergences

- LigandMpnn designs sequence on a fixed backbone with atomic context. It does not
  generate backbones or pack side chains; the independent packer is the separate
  `MpnnPacker` family.
- The four released sequence checkpoints share one executable architecture and differ
  only by the training coordinate-noise level; that value is weight provenance, not a
  runtime config field.
- The released command saves raw aligned distributions, their mean, and population
  standard deviation rather than a scalar NLL, so LigandMpnn intentionally publishes
  no `ScoreResult` path.
- A parity run applies only to the checkpoints and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** MIT
  ([LigandMPNN LICENSE](https://github.com/dauparas/LigandMPNN/blob/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE),
  Copyright (c) 2024 Justas Dauparas).
- **Redistribution status: GREEN.** The four `ligandmpnn_v_32_*_25` checkpoints are
  permissively licensed and freely redistributable with the upstream MIT notice
  preserved.
