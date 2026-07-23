---
language: protein
tags:
  - Biology
  - Protein
  - Structure Prediction
license: cc-by-4.0
library_name: ofoldx
pipeline_tag: other
---

# AlphaFold 2

Native PyTorch implementation of AlphaFold 2 monomer, monomer-pTM, and
AlphaFold-Multimer for OFoldX.

## Disclaimer

This is an unofficial OFoldX conversion of
[AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) and
[AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034). The upstream
authors did not write, review, or endorse this implementation or model card.
The official implementation is maintained by
[Google DeepMind](https://github.com/google-deepmind/alphafold).

Official DeepMind parameters are not part of the OFoldX source license. They are
distributed under CC BY 4.0 and must be obtained and used under their upstream
terms.

## Model Details

The family exposes two registered Model, Processor, and Pipeline pairs, both
with the standard `fold` task:

- `ofoldx.alphafold2`: `AlphaFold2Model`, `AlphaFold2Processor`, and
  `AlphaFold2Pipeline` for monomer and monomer-pTM checkpoints.
- `ofoldx.alphafold2_multimer`: `AlphaFold2MultimerModel`,
  `AlphaFold2MultimerProcessor`, and `AlphaFold2MultimerPipeline` for
  AlphaFold-Multimer checkpoints.

Both models contain token/MSA embedding, recycling, template embedding, an
extra-MSA stack, a 48-block Evoformer stack, an eight-block invariant-point-
attention StructureModule, and the released auxiliary and confidence heads.
`model.set_backend("auto" | "torch" | "oops")` selects the runtime operator
backend; backend choice is not serialized in `model.json`.

The monomer Processor accepts one protein as an inline sequence, FASTA path,
`PolymerSequence`, one-instance `MolecularSystem`, or `Structure`. A supplied
`Structure` contributes topology only; its coordinates are not treated as a
template. MSA evidence is an entity-bound `MSA` or an aligned FASTA/A3M path,
and structural evidence must be supplied as explicitly aligned `Template`
values. Heterogeneous systems, multiple instances, and `MSACollection` inputs
are rejected by the monomer Processor.

The multimer Processor accepts a multi-instance protein `MolecularSystem`, a
sequence of `PolymerSequence` chains, or a mapping of chain names to
`PolymerSequence` values. It lowers paired `MSACollection` evidence and derives
the required `chain_index`, `entity_index`, and `symmetry_index` features.
Multimer templates use atom37 coordinates, pseudo-beta geometry, and backbone
frames rather than the monomer template-torsion MSA row.

`model(input)` returns `AlphaFold2Output`: single and pair representations,
StructureModule frames and coordinates, side-chain angles, distogram logits,
masked-MSA logits, pLDDT logits, optional PAE logits, and
experimentally-resolved logits. The final recycle remains differentiable;
`train()` and `eval()` affect ordinary dropout, not which recycle carries
gradients.

`fold` performs released stochastic MSA sampling, masking, clustering, and
cropping for each recycle and configured representation ensemble. The Pipeline
accepts an integer `seed` or a `torch.Generator`; the Model accepts the resolved
generator. `num_samples` returns generation-ordered predictions in one
`FoldResult`.

Each prediction contains a canonical complete-heavy `Structure`, residue pLDDT,
mean pLDDT, optional PAE, and pTM when the checkpoint has a PAE head. Multimer
predictions always include PAE, pTM, ipTM, and
`ranking_score = 0.8 * iptm + 0.2 * ptm`.

### Variants

The official `alphafold_params_2022-12-06.tar` release contains:

- `model_1` through `model_5` for monomer prediction;
- `model_1_ptm` through `model_5_ptm` with the PAE/pTM head; and
- `model_1_multimer_v3` through `model_5_multimer_v3` for
  AlphaFold-Multimer.

Monomer models 1 and 2 include template parameters; models 3 through 5 do not.
The corresponding pTM variants use the same template split. Multimer v3
checkpoints use the multimer architecture, chain-relative input embedding,
22-class masked-MSA head, multimer template embedder, and always-enabled
interface-TM branch.

### Model Specification

| Property                                  | Value                                             |
| ----------------------------------------- | ------------------------------------------------- |
| Architectures                             | `ofoldx.alphafold2`, `ofoldx.alphafold2_multimer` |
| Task                                      | `fold`                                            |
| Evoformer blocks / `num_blocks`           | 48                                                |
| Extra-MSA blocks / `num_extra_msa_blocks` | 4                                                 |
| StructureModule blocks / `num_blocks`     | 8                                                 |
| Template pair blocks / `num_blocks`       | 2 when templates are enabled                      |
| MSA / pair / single sizes                 | 256 / 128 / 384                                   |
| Default recycling steps                   | 4 monomer, 21 multimer                            |
| Default representation ensembles          | 1                                                 |
| Monomer masked-MSA classes                | 23                                                |
| Multimer masked-MSA classes               | 22                                                |

The monomer and multimer graphs share the Evoformer and output-head
implementations. Multimer applies outer-product mean first within each
Evoformer block, uses a StructureModule position scale of 20 rather than 10,
and always constructs the PAE/interface-TM head. The outer-product-mean
denominator epsilon is `1e-3`; masked softmax uses a dtype-safe finite minimum,
including for float16.

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold2)
- [Official AlphaFold source](https://github.com/google-deepmind/alphafold)
- [AlphaFold 2 paper](https://doi.org/10.1038/s41586-021-03819-2)
- [AlphaFold-Multimer paper](https://doi.org/10.1101/2021.10.04.463034)
- [OpenFold source](https://github.com/aqlaboratory/openfold)

## Usage

```python
import ofoldx
from ofoldx.data.molecular import PolymerSequence

monomer = ofoldx.pipeline(
    "fold",
    model="./alphafold2-model-1-ptm",
    device="cuda",
)
monomer_result = monomer.fold("MKAYLVGDEK", seed=7)
monomer_prediction = monomer_result.best("mean_plddt")

chains = (
    PolymerSequence.from_one_letter("MKAYLVGDEK"),
    PolymerSequence.from_one_letter("GAMGKT"),
)
multimer = ofoldx.pipeline(
    "fold",
    model="./alphafold2-model-1-multimer-v3",
    device="cuda",
)
multimer_result = multimer.fold(chains, seed=7)
complex_prediction = multimer_result.best("ranking_score")
```

Both Pipelines also accept explicit MSA and template evidence, `num_samples`,
and `recycling_steps`. Direct Model use first lowers the biological input
through the matching Processor, then calls
`model.fold(processed, generator=...)`.

## Checkpoints and Conversion

`convert_alphafold2_checkpoint` strict-loads a checkpoint and returns an
in-memory Model. `save_alphafold2_checkpoint` additionally writes
`model.json`, `processor.json`, and safetensors, selecting the monomer or
multimer Processor from the converted Model.

```python
from ofoldx.models.alphafold2.convert import (
    convert_alphafold2_checkpoint,
    save_alphafold2_checkpoint,
)

monomer_model = convert_alphafold2_checkpoint(
    "params_model_1_ptm.npz",
    source="deepmind",
)
multimer_model = convert_alphafold2_checkpoint(
    "alphafold_params_2022-12-06.tar",
    source="deepmind",
    model_name="model_1_multimer_v3",
)
save_alphafold2_checkpoint(
    "params_model_1_ptm.npz",
    "./alphafold2-model-1-ptm",
    source="deepmind",
)
```

The DeepMind source accepts one official NPZ, a params directory/download root,
or the official tar archive. `model_name` selects a member when a directory or
archive contains more than one. Monomer, pTM, and multimer names dispatch to
their matching configs and Models. The OpenFold source accepts PyTorch
checkpoints for the monomer architecture and selects a recognized EMA or model
state; OpenFold multimer conversion is not exposed.

Official monomer and pTM NPZs and an OpenFold `finetuning_ptm_1` checkpoint have
been strict-loaded. A real official `model_1_multimer_v3` conversion loaded 330
source tensors into 5,062 native state entries with no missing, unexpected, or
shape-mismatched keys under strict loading. At a matched recycle count, its
single and pair representations agreed with OpenFold 2.x to fp32 relative-L2
on the order of `3e-7`. These observations are specific to those checkpoints
and comparisons.

Supplying `repo_id` to `save_alphafold2_checkpoint` requests a Hub upload;
omitting it keeps conversion local. Conversion does not change the license of
the source checkpoint.

## Training Details

### Training Data

Refer to the AlphaFold 2 and AlphaFold-Multimer papers and supplementary
materials for the upstream training corpora and data construction. OFoldX does
not redistribute those corpora, sequence databases, or template databases.

### Training Procedure

The upstream monomer models use recycling, MSA and template evidence, and the
released auxiliary losses; the pTM variants were fine-tuned with the
predicted-aligned-error head. AlphaFold-Multimer adds paired-MSA and
chain-relative features and interface confidence.

OFoldX preserves a differentiable final recycle and the released inference
module boundaries. Checkpoint conversion and inference do not claim to
reproduce the complete upstream training system.

## Limitations

- AlphaFold 2 predicts structures; it does not provide experimental validation.
- Monomer input is limited to one standard protein instance. Multimer input is
  limited to protein polymer chains; ligands, nucleic acids, and heterogeneous
  assemblies are not accepted by this family.
- Coordinates on a monomer `Structure` input are not consumed as template
  evidence. Templates require explicit canonical alignments.
- Amber/OpenMM relaxation is not included; `fold` returns unrelaxed structures.
- OpenFold conversion currently covers monomer checkpoints only.
- Strict loading establishes key and shape compatibility, not complete
  numerical equivalence. Reported multimer agreement covers one checkpoint,
  matched settings, and the single/pair representations; it does not establish
  every variant, stochastic MSA realization, confidence value, or decoded
  structure.
- Sampling is stochastic unless the caller supplies a seed or generator, and
  results also depend on MSA, template, recycling, ensemble, dtype, and backend
  settings.

## Citation

```bibtex
@article{jumper2021highly,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  title = {Highly accurate protein structure prediction with AlphaFold},
  journal = {Nature},
  volume = {596},
  pages = {583--589},
  year = {2021},
  doi = {10.1038/s41586-021-03819-2}
}
```

For AlphaFold-Multimer predictions, also cite
[Protein complex prediction with AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034).

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Questions about the
original models should be directed to the upstream AlphaFold authors.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Google DeepMind AlphaFold source:** Apache-2.0.
- **Official AlphaFold 2 parameters:** CC BY 4.0. Preserve attribution and the
  upstream license notice when redistributing converted parameters.
- **OpenFold source:** Apache-2.0.
- **OpenFold checkpoints:** retain the terms supplied with the particular
  checkpoint. Loading or converting a checkpoint does not relicense it.

The front-matter license describes the official DeepMind model parameters, not
the separate OFoldX source-code license.
