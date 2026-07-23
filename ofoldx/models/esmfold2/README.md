---
language: protein
tags:
  - Biology
  - Protein
  - Structure Prediction
license: agpl-3.0
base_model: biohub/ESMFold2 + multimolecule/esmc-6b
library_name: ofoldx
pipeline_tag: other
---

# ESMFold2

All-atom biomolecular structure prediction and protein binder design driven by
ESMC representations, with optional protein MSA evidence for supported checkpoints.

## Disclaimer

This is an unofficial OFoldX implementation and conversion of
[ESMFold2](https://github.com/Biohub/esm), described in
[Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf).
The upstream authors did not write, review, or endorse this implementation,
conversion, or model card.

## Model Details

The `ofoldx.esmfold2` architecture pairs `EsmFold2Model` with
`EsmFold2Processor`. `EsmFold2Pipeline` exposes the released `fold` operation
and returns `FoldResult`. The separate `ofoldx.esmfold2_experimental`
architecture pairs `EsmFold2ExperimentalModel` with
`EsmFold2ExperimentalProcessor`; `EsmFold2DesignPipeline` exposes the
experimental binder `design` operation and returns the family-owned
`EsmFold2DesignResult`.

The released folding input supports standard and modified proteins, DNA, RNA,
reference-conformer ligands, and covalent assemblies. The Processor accepts a
protein sequence string, FASTA path, `PolymerSequence`, a sequence of
`PolymerSequence` values, `MolecularSystem`, or `Structure`. A `Structure`
contributes its canonical molecular system rather than its observed
coordinates. Optional `MSA` and `MSACollection` inputs retain aligned protein
rows, pairing, and A3M deletion features. The Fast variants reject explicit MSA
evidence.

The native Model state tree begins with `input_embeddings`,
`left_pair_projection`, `right_pair_projection`,
`relative_position_embedder`, and `bond_projection`. ESMC layer states enter
through `language_model_pair_encoder`; the released graph then uses its
optional `language_model_encoder` and `msa_encoder`, `folding_trunk`, linear-recurrent
`parcae_input_norm`, `parcae_log_decay`, `parcae_log_delta`, `parcae_input_matrix`,
`parcae_readout`, `parcae_coda`, `diffusion_structure_head`,
`distogram_head`, and `confidence_head`. Repeated `EsmFold2FoldingBlock`
values live under `folding_trunk.blocks`. Each Block owns its coupled triangle
updates and an `EsmFold2Transition`; each Transition owns the inner
`EsmFold2FeedForward` computation. The experimental graph instead carries the
previous pair representation through `recycling_norm` and
`recycling_projection` and uses its narrower confidence head.

The released Model owns a frozen MultiMolecule `EsmCModel` as its public
`esmc` child. Converted release repositories store the ESMC and folding-head
weights in one Model state dict, so `AutoModel` does not perform a hidden
second repository lookup. Experimental design repositories store the selected
ESMC masked-language-model child and tokenizer under `esmc/`; the concrete
design Pipeline restores that child and its language-model head. Binder design
requires the language-model head whenever the configured sequence-prior weight
is nonzero.

`EsmFold2Model.embeddings` returns the released token and pair embeddings
before diffusion. `EsmFold2Model.forward` is a differentiable single-step
denoising primitive, while `fold` runs recurrent refinement, diffusion
sampling, confidence prediction, and canonical Result decoding.

Binder design optimizes per-position soft amino-acid logits through the
experimental recurrent graph, combines the structure objective with the
configured ESMC pseudo-perplexity prior, selects the best distogram proxy
during optimization, and discretely refolds each selected sequence. A
`binder_prompt` fixes amino acids and marks mutable positions with `#`;
`antibody_cdr_prompt` constructs a Chothia CDR-only prompt for an antibody Fv.
The configured sequence-prior weight is `0.15` for general binders and `0.05`
for antibody-framework design.
Each `EsmFold2DesignCandidate` owns the complete folded `target|binder`
`Structure`, scalar scores, token and atom pLDDT, and PAE.

### Variants

| Upstream checkpoint group                                                           | Type         | Folding Blocks | MSA      | ESMC             | Confidence and role                             |
| ----------------------------------------------------------------------------------- | ------------ | -------------: | -------- | ---------------- | ----------------------------------------------- |
| `biohub/ESMFold2`                                                                   | release      |             48 | optional | 6B               | full confidence; folding                        |
| `biohub/ESMFold2-Fast`                                                              | release      |             24 | no       | 6B               | full confidence; faster single-sequence folding |
| `biohub/ESMFold2-Experimental-Fast` and `-Cutoff2025`                               | experimental |             24 | no       | 6B               | pLDDT/PAE; binder inversion                     |
| `biohub/ESMFold2-Experimental` and `-Cutoff2025`                                    | experimental |             48 | yes      | 6B               | pLDDT/PAE; hero critic and refold               |
| `biohub/ESMFold2-Experimental-Fast-base{300M,600M,6B}-step{250,500,750,1000,1500}k` | experimental |             24 | no       | named scale/step | distogram-only scaling critics                  |

The two release checkpoints share one executable architecture. ESMFold2-Fast
disables its MSA encoder and uses a `0.1` language-model masking default;
ESMFold2 enables MSA conditioning and uses a zero masking default. Experimental
checkpoints use a distinct recurrent graph and checkpoint-derived ESMC
dimensions. Scaling critics omit the confidence head and cannot produce
confidence outputs.

### Model Specification

| Property                               | Released graph                  | Experimental graph                          |
| -------------------------------------- | ------------------------------- | ------------------------------------------- |
| Architecture                           | `ofoldx.esmfold2`               | `ofoldx.esmfold2_experimental`              |
| Model                                  | `EsmFold2Model`                 | `EsmFold2ExperimentalModel`                 |
| Processor                              | `EsmFold2Processor`             | `EsmFold2ExperimentalProcessor`             |
| Pipeline                               | `EsmFold2Pipeline`              | `EsmFold2DesignPipeline`                    |
| Task                                   | `fold`                          | `design`                                    |
| Result                                 | `FoldResult`                    | `EsmFold2DesignResult`                      |
| Input size                             | 451                             | 451                                         |
| Single / pair size                     | 384 / 256                       | 384 / 256                                   |
| Atom / token size                      | 128 / 768                       | 128 / 768                                   |
| Folding Blocks                         | 48 or 24                        | 48 or 24, from checkpoint                   |
| Source loops / total recurrent updates | 3 / 4                           | 3 / 4                                       |
| Default sampler steps                  | 14                              | 14; 15 for scaling critics                  |
| ESMC dependency                        | 6B, 80 layers, hidden size 2560 | 30/960, 36/1152, or 80/2560 from checkpoint |
| ESMC context                           | 2048 tokens per protein chain   | selected ESMC checkpoint limit              |

Each released `FoldPrediction` carries a complete non-hydrogen `Structure`;
`ptm`, `iptm`, `mean_plddt`, and `complex_iplddt` scores; token and atom pLDDT;
PAE and PDE in angstroms; and chain-pair ipTM. `fold` exposes recurrence,
sample count, diffusion steps, language-model masking and dropout, MSA depth
and masking, sampler scales, maximum inference sigma, and RMSD-based denoising
exit. Stochastic Model calls accept a caller-owned `torch.Generator`; Pipeline
calls also accept `seed`.

### Links

- [Official source](https://github.com/Biohub/esm)
- [Inspected source revision](https://github.com/Biohub/esm/tree/ba4d7124864eed323a93bf3cfefcd958f573b75a)
- [ESMFold2 checkpoint](https://huggingface.co/biohub/ESMFold2)
- [ESMFold2-Fast checkpoint](https://huggingface.co/biohub/ESMFold2-Fast)
- [Experimental Fast checkpoint](https://huggingface.co/biohub/ESMFold2-Experimental-Fast)
- [Experimental MSA checkpoint](https://huggingface.co/biohub/ESMFold2-Experimental)
- [Released binder-design workflow](https://github.com/Biohub/esm/blob/ba4d7124864eed323a93bf3cfefcd958f573b75a/cookbook/tutorials/binder_design.py)
- [MultiMolecule ESMC-6B](https://huggingface.co/multimolecule/esmc-6b)
- [Paper](https://biohub.ai/papers/esm_protein.pdf)
- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/esmfold2)

## Usage

Convert the required checkpoints first, then load the resulting repository:

```python
from ofoldx import AutoModel, AutoProcessor, pipeline

sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQR"
repository = "./esmfold2"

folder = pipeline("fold", model=repository, device="cuda")
result = folder.fold(
    sequence,
    num_samples=4,
    num_sampling_steps=14,
    seed=7,
)
best_prediction = result.best("ptm")

model = AutoModel(repository, device="cuda")
processor = AutoProcessor(repository, device="cuda")
batch = processor(sequence)
token_embeddings, pair_embeddings = model.embeddings(batch)
direct_result = model.fold(batch, num_samples=1, seed=7)
```

The MSA-capable checkpoint accepts `msa=` as an `MSA`, `MSACollection`, aligned
FASTA, or A3M path. For multi-protein systems, use `MSACollection` to retain
explicit entity and instance bindings.

Load a converted experimental design repository through its concrete Pipeline:

```python
from ofoldx import pipeline

designer = pipeline(
    "design",
    model="./esmfold2-experimental-fast",
    device="cuda",
)
designs = designer.design(
    "EVQLVESGGGLVQPGGSLRLSCAAS",
    binder_length=60,
    num_candidates=4,
    num_optimization_steps=150,
    seed=7,
)
best_design = designs.best("iptm")
```

Pass `binder_prompt` to mix fixed and mutable binder positions. For antibody
CDR design, build the prompt with `antibody_cdr_prompt` and use an experimental
artifact configured with the `0.05` antibody sequence-prior weight rather than
the `0.15` general-binder default.

## Checkpoints and Conversion

Convert a released folding checkpoint together with the paired MultiMolecule
ESMC-6B child:

```bash
python -m ofoldx.models.esmfold2.convert \
  biohub/ESMFold2 \
  ./esmfold2 \
  --esmc multimolecule/esmc-6b
```

Use `biohub/ESMFold2-Fast` as the source for the Fast variant. Convert an
experimental binder-design head with its masked-language-model child and
tokenizer:

```bash
python -m ofoldx.models.esmfold2.convert \
  biohub/ESMFold2-Experimental-Fast \
  ./esmfold2-experimental-fast \
  --experimental \
  --esmc multimolecule/esmc-6b
```

The converter reads graph dimensions and checkpoint type from the source
`config.json`. It translates source checkpoint names into the descriptive
native state tree: for example, source `z_init_1`, `z_init_2`, and
`token_bonds` parameters become `left_pair_projection`,
`right_pair_projection`, and `bond_projection`. Source transition branches are
combined into each native `transition.feed_forward.gate_value_projection`, and
attention, conditioning, diffusion, and confidence parameters are mapped to
their owning Modules.

Release conversion rejects missing head parameters and every unexpected
parameter after discarding exactly four confidence tensors that the released
source forward path never reads. Experimental heads strict-load without an
ESMC child, after which the selected child is attached. A release repository
stores `model.json`, `processor.json`, and the combined safetensors state. A
design repository stores the experimental root pair, the selected ESMC model
and tokenizer under `esmc/`, and the `ofoldx.esmfold2_design` Pipeline marker.
Converted repositories are self-contained.

The 15 scaling critics name the ESMC scale and training step in their
checkpoint IDs. Their source configs also name the corresponding ESMC
checkpoint; conversion must use that matching dependency rather than assuming
ESMC-6B.

## Training Details

### Training Data

Biohub reports that ESMFold2 was trained on sequences from the Protein Data
Bank (PDB) and AlphaFold DB (AFDB). The release model cards give a September
2021 data cutoff for ESMFold2 and ESMFold2-Fast. ESMC was pretrained separately
on large protein-sequence corpora; consult the selected ESMC artifact's model
card for its data and checkpoint-specific provenance. OFoldX does not
redistribute the upstream training corpora.

### Training Procedure

The paper describes diffusion MSE, smooth-LDDT, polymer-ligand bond,
distogram, and confidence losses for the folding system. The released source
and model cards do not publish every target construction, mask, mini-rollout,
batching rule, and reduction needed to reproduce head training exactly.
OFoldX exposes the differentiable single-step denoising path but does not
present it as a complete reproduction of the unpublished training pipeline.

Binder design is inference-time optimization rather than checkpoint training.
It performs gradient descent over soft binder-sequence logits under the
released distogram contact objective and ESMC pseudo-perplexity regularizer,
then refolds a discrete selected sequence.

## Limitations

- ESMFold2 predictions are computational hypotheses, not experimental
  measurements. Structures and designed binders require independent
  experimental validation.
- PDB and AFDB contain uneven coverage of protein families, conformational
  states, modifications, disorder, and non-protein biomolecules. Accuracy may
  degrade for underrepresented chemistry and biology.
- Quality depends on inference-time sampling and recurrent refinement.
  Reducing samples or steps trades accuracy for speed, while pair and atom
  computations make large systems memory-intensive.
- The ESMC dependency has a 2048-token context. The Processor runs one ESMC row
  per protein chain; MSA evidence applies only to protein entities, and Fast
  checkpoints reject it.
- Canonical small-molecule inputs must carry an explicit, complete reference
  conformer. The Processor does not generate geometry from SMILES. Observed
  coordinates in a `Structure` are not folding features.
- The released source emits distance-conditioning tensors that its released
  folding forward path does not consume. OFoldX rejects this ineffective
  conditioning rather than advertising it as supported.
- Binder design currently targets one single-chain protein and one binder per
  call. One `EsmFold2DesignPipeline` owns one converted experimental head; it
  does not automatically reproduce the released notebook's multi-inversion,
  hero-critic, and 15-checkpoint scaling-critic ensemble.
- In one recorded released-ESMFold2 comparison, repeated native runs had zero
  observed difference, global-score differences were about `2e-4`, the
  reported coordinate discrepancy was about `0.215` angstrom, and the
  reported PAE discrepancy was about `0.108` angstrom. Those measurements
  characterize that checkpoint, input, dtype, seed, and settings only; they do
  not establish numerical identity.
- Equivalent released-weight comparison has not been established for
  ESMFold2-Fast, the experimental checkpoints, binder design, or canonical
  generic nucleic-acid N/DN full inputs.

## Citation

```bibtex
@misc{candido2026language,
  title = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and others},
  year = {2026},
  url = {https://biohub.ai/papers/esm_protein.pdf},
  note = {Preprint}
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Contact the Biohub
authors for questions about ESMFold2, its checkpoints, or the paper.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Biohub ESM source and ESMFold2 checkpoint heads:** MIT, Copyright 2026
  Chan Zuckerberg Biohub, Inc., with the
  [Biohub third-party notice](https://github.com/Biohub/esm/blob/ba4d7124864eed323a93bf3cfefcd958f573b75a/THIRD_PARTY_NOTICE.md)
  retained where applicable.
- **`multimolecule/esmc-6b` implementation and artifact:** AGPL-3.0-or-later,
  subject to the linked
  [MultiMolecule license materials](https://huggingface.co/multimolecule/esmc-6b/tree/62eb94c06f7013fea59a5075955658c8a5a72263).
- Converted artifacts combine components with distinct provenance. Conversion
  does not relicense upstream weights or remove their notices.
