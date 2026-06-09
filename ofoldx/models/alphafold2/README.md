# AlphaFold 2

This directory owns the AlphaFold 2 architecture entry point.

Currently implemented here:

- AF2-owned config tree with explicit `input_embedding`, `recycling`,
  `evoformer`, `template`, `heads`, and `recycling_steps` groups.
- A save/load-compatible `AlphaFold2Model` that consumes `AlphaFold2Input`,
  runs AF2-shaped input embedding, recycling, template embedding,
  extra-MSA/main Evoformer stacks, structure/head modules, and emits the standard
  OFoldX
  `StructurePredictionOutput`.
- An `AlphaFold2Processor` registered under the `alphafold2` architecture key,
  with protein structure, raw sequence, FASTA, and precomputed A3M inputs.
- Official-parameter discovery, inspection, key normalization, tensor layout
  conversion, strict loading, and standard OFoldX model-artifact packaging.
- OpenFold PyTorch checkpoint inspection, key normalization, strict loading, and
  standard OFoldX model-artifact packaging.

Current status:

- Official monomer and pTM tensors strict-load into the PyTorch
  `AlphaFold2Model` state dict with no missing, unexpected, or shape-mismatched
  keys under the matching preset. `model_1`/`model_2` use templates;
  `model_3`/`model_4`/`model_5` are no-template presets.
- Evoformer, extra-MSA, template pair embedding, StructureModule, masked-MSA,
  experimentally-resolved, distogram, pLDDT, and optional PAE parameter trees are
  represented with AF2-owned module boundaries.
- The inference-side StructureModule geometry now includes invariant point
  attention, rigid-frame updates, atom14 sidechains, and atom14-to-atom37
  projection; FAPE and structural-violation losses are not ported yet.
- The converter writes a standard `kind="model"` artifact for official monomer
  and pTM checkpoints, including a processor component for `AutoProcessor` and
  `Pipeline.from_pretrained`. AF-Multimer checkpoints are rejected until the
  multimer architecture and feature path are implemented explicitly.
- OpenFold `initial_training`, `finetuning_*`, `finetuning_ptm_*`, and
  `finetuning_no_templ*` checkpoints convert through the same entry point. The
  converter infers template and pTM-head availability from the checkpoint tensor
  tree and writes strict-loadable OFoldX artifacts.
- Full database-backed AF2 feature generation is still outside this architecture
  slice: JackHMMER/HHblits search, HHsearch template hit featurization, feature
  caching, ranking, and output writing need an AF2-specific feature pipeline
  above the runnable model artifact.

Quality boundary:

- Strict checkpoint loading is not public-output parity. It proves the converted
  tensors cover the current PyTorch parameter tree, but AF2 is not signed off
  until the same public input produces the same public outputs as the official
  AlphaFold 2 reference path with an explicit numerical diff.
- The committed tests stay cheap and self-contained; AF2 numerical parity belongs
  in the local parity harness and must be rerun after Evoformer, template,
  recycling, or StructureModule changes.

## Official DeepMind parameter conversion

DeepMind distributes AlphaFold 2 parameters as `alphafold_params_2022-12-06.tar`.
After extraction, the download root contains `params/params_<model_name>.npz`
files for five monomer CASP14 models, five monomer pTM models, and five multimer
v3 models, plus a license file. The upstream AlphaFold code is Apache-2.0; the
official model parameters are CC BY 4.0.

The converter accepts a direct `params_model_*.npz`, a params/download directory,
or the tar archive via the unified source-aware entry point:

```bash
python -m ofoldx.models.alphafold2.convert --source deepmind /data/alphafold model-af2 \
  --model-name model_1_ptm --dtype float32
```

OpenFold distributes trained PyTorch checkpoints at
`s3://openfold/openfold_params/` (also reachable as
`https://openfold.s3.amazonaws.com/openfold_params/`). Download the desired
`.pt` file, then run:

```bash
python -m ofoldx.models.alphafold2.convert --source openfold \
  /data/openfold_params/finetuning_ptm_2.pt openfold-finetuning-ptm-2
```

`--source openfold --inspect-only` can summarize torch checkpoint containers
(`state_dict`, `model_state_dict`, `model`, `params`, or EMA params) without
writing an artifact.

The output is an OFoldX `kind="model"` artifact with:

- `manifest.json`
- `conversion_config.json`
- `config.json`
- `model.safetensors`

The converter strict-loads the official parameters into `AlphaFold2Model` before
writing and, by default, reloads the saved artifact with
`AlphaFold2Model.from_pretrained(..., strict=True)`. That is a conversion gate,
not final AlphaFold 2 parity signoff.
