---
language:
  - protein
  - rna
  - dna
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# AlphaFold 3

All-atom biomolecular structure prediction for proteins, nucleic acids,
ligands, ions, modified residues, and glycan components.

## Disclaimer

This is an unofficial OFoldX conversion of
[Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
by Josh Abramson, Jonas Adler, Jack Dunger, et al. The upstream authors did not
write, review, or endorse this implementation or model card. The official
repository is
[google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3).

Official AlphaFold 3 parameters are obtained separately from Google DeepMind
and are governed by the
[Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
and
[Prohibited Use Policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md).
OFoldX does not redistribute those parameters.

## Model Details

`ofoldx.alphafold3` registers one `AlphaFold3Model` and
`AlphaFold3Processor` pair. Its seed-group-preserving result requires the
concrete `AlphaFold3Pipeline`, which is registered for `fold`; the Model pair
does not advertise a generic standard task.

AlphaFold 3 embeds atom, token, MSA, template, bond, and relative-position
features into single and pair representations. A recycled 48-block Pairformer
trunk refines those representations, a 24-block EDM coordinate denoiser samples
all-atom coordinates, and distogram and confidence heads produce pLDDT, PAE,
PDE, resolved-atom estimates, and scalar/chain-level confidence summaries. The
released template path is represented by a two-block template stack.

The Processor accepts an inline protein sequence, FASTA or structure path,
`PolymerSequence`, a sequence of polymer chains, `MolecularSystem`, or
`Structure`, together with explicit MSA and `Template` evidence. Bare CCD-code
ligands, ions, modified residues, and glycan monomers are resolved to embedded
chemistry at the Processor boundary using the pinned wwPDB CCD revision
`2022-09-28`; the tensor graph performs no live registry lookup.

`model.forward` is the differentiable one-step denoising graph and requires
caller-supplied noisy coordinates and a noise level. `model.fold` runs trunk
recycling, stochastic reverse diffusion, confidence decoding, clash detection,
and disorder-aware ranking. One trunk pass is shared by the samples in a seed
group.

`AlphaFold3FoldResult` preserves `AlphaFold3SeedGroup` values rather than
flattening samples into a shared `FoldResult`. Each `FoldPrediction` contains a
canonical `Structure`, atom-level pLDDT and resolved probability, token-level
PAE/PDE values, chain and chain-pair summaries, pTM, ipTM,
`ptm_iptm_average = 0.8 * iptm + 0.2 * ptm`, clash/disorder values, and ranking
scores. `best(score)` ranks across every sample in every seed group.

`model.set_backend("auto" | "torch" | "oops")` selects the operator backend
for shared attention, transition, atom, MSA, and triangle modules.

### Variants

OFoldX exposes one AlphaFold 3 architecture identity,
`ofoldx.alphafold3`. The converter reads official single-file or sharded
record-stream parameters, compressed or uncompressed. When a directory
contains more than one official model stem, `model_name` selects that upstream
name; OFoldX does not assign additional synthetic variant names.

### Model Specification

| Property                                | Value                                |
| --------------------------------------- | ------------------------------------ |
| Architecture                            | `ofoldx.alphafold3`                  |
| Model                                   | `AlphaFold3Model`                    |
| Processor                               | `AlphaFold3Processor`                |
| Pipeline                                | `AlphaFold3Pipeline`                 |
| Task                                    | `fold` through the concrete Pipeline |
| Result                                  | `AlphaFold3FoldResult`               |
| Pairformer trunk blocks                 | 48                                   |
| Template blocks / `template.num_blocks` | 2                                    |
| Denoiser token blocks                   | 24                                   |
| Single / pair sizes                     | 384 / 128                            |
| Default recycling steps                 | 4                                    |
| Default diffusion samples               | 5                                    |
| Default sampling steps                  | 200                                  |
| Maximum lowered MSA rows                | 16,384                               |

Confidence post-processing keeps two numerical contracts separate:
predicted-TM normalization uses the dtype's positive `tiny` value, while the
contact-weighted PAE ranking metric uses `rank_metric_eps` (`1e-6` by default).

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold3)
- [Official AlphaFold 3 source](https://github.com/google-deepmind/alphafold3)
- [AlphaFold 3 paper](https://doi.org/10.1038/s41586-024-07487-w)
- [Official parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)

## Usage

```python
import ofoldx

fold_pipeline = ofoldx.pipeline(
    "fold",
    model="./alphafold3",
    device="cuda",
)
result = fold_pipeline.fold(
    "complex.cif",
    num_samples=5,
    seed=7,
)
prediction = result.best("ranking_score")
structure = prediction.structure
```

The Pipeline also accepts explicit `msa`, `templates`, `recycling_steps`, and
`num_sampling_steps`. `seed` and `generator` are Pipeline-level controls for
the diffusion rollouts. Direct Model use requires an `AlphaFold3Input` built by
the Processor when the output must be decoded into canonical structures.

## Checkpoints and Conversion

`convert_alphafold3_model` strict-loads official parameters into an in-memory
Model. `convert_alphafold3_checkpoint` writes a complete Pipeline repository
containing `model.json`, `processor.json`, `pipeline.json`, and safetensors.

```python
from ofoldx.models.alphafold3.convert import (
    convert_alphafold3_checkpoint,
    convert_alphafold3_model,
)

model = convert_alphafold3_model("af3.bin.zst")
convert_alphafold3_checkpoint(
    "af3.bin.zst",
    "./alphafold3",
)
```

The reader accepts official `.bin`, `.bin.zst`, and their supported sharded
filename forms. It translates the official record-stream tensors, rejects
unexpected records outside the known non-inference metadata/projections, and
strict-loads the native state tree.

Conversion is local when `repo_id` is omitted. The converter also exposes
explicit `repo_id` and `private` arguments, but those arguments do not determine
or grant legal permission to upload AlphaFold 3 parameters. Users must comply
with the official parameter terms, including restrictions on sharing.

## Training Details

### Training Data

The upstream AlphaFold 3 publication describes the biomolecular structure,
sequence, and chemistry resources used for training. OFoldX does not
redistribute the training corpus. Refer to the paper and its supplementary
information for the authoritative data description.

### Training Procedure

Upstream AlphaFold 3 is trained end to end with a diffusion coordinate
objective and auxiliary distogram and confidence objectives. OFoldX preserves
the released inference architecture and exposes the differentiable denoising
step; converting official parameters does not reproduce or claim the complete
upstream training system.

## Limitations

- AlphaFold 3 outputs are predictions with varying confidence and are not
  experimental evidence. They are not intended or validated for clinical use.
- Inference is stochastic and computationally demanding. Results depend on the
  parameter set, biological evidence, seed, sample count, recycling and
  sampling steps, dtype, and backend.
- The Processor lowers supplied MSA and template evidence; it does not perform
  the upstream database-search pipeline.
- A reference comparison reported pair-representation relative-L2 of `0.0587`,
  above a tight numerical target. Treat full numerical parity as incomplete.
- The complete reverse-diffusion sampler and DSSP-derived disorder/ranking
  terms have not been independently validated end to end against the official
  implementation. Strict checkpoint loading alone does not establish that
  equivalence.
- Bare CCD codes resolve against the pinned `2022-09-28` snapshot; callers that
  require different chemistry must provide a fully specified canonical system.

## Citation

```bibtex
@article{abramson2024accurate,
  author = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
  title = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  journal = {Nature},
  volume = {630},
  pages = {493--500},
  year = {2024},
  doi = {10.1038/s41586-024-07487-w}
}
```

If OFoldX supports your research, also cite:

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
for questions about this implementation or model card. Questions about the
original model should be directed to the AlphaFold 3 authors.

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Official AlphaFold 3 source:** Apache-2.0.
- **Official AlphaFold 3 parameters:** governed by the
  [Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
  and
  [Prohibited Use Policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md),
  not by AGPL or Apache-2.0.

The official terms permit only specified non-commercial uses by or on behalf
of non-commercial organizations, prohibit commercial use, and prohibit
publishing or sharing the parameters except within an organization as allowed
by those terms. Parameters may be used only when received directly from Google.
Consult the current official terms before use; conversion does not relicense
the parameters or grant redistribution rights. The front-matter license
describes the OFoldX implementation, not the official parameters.
