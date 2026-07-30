# OFoldX

OFoldX is a research toolkit for biomolecular structure prediction and design. It
follows a small, consistent shape: processors turn structures into typed model inputs,
models consume those inputs, and pipelines package the processor/model boundary for
inference.

OFoldX is pre-release. A family package or prototype is not a support claim;
registration is reserved for a family only after all of its released scientific
functions, checkpoint variants, canonical mappings, and conversion paths are
complete.

Training and evaluation run through one config-only `ofoldx.runners.Runner`
(Danling-backed). It resolves the exact registered architecture, builds the Model and a
config-consistent Processor, and applies the exact built-in objective and batching
behavior owned by the Runner. Training support is not a model-registration
requirement.

## Get Started

Install OFoldX:

```bash
pip install ofoldx
```

For development, install from source:

```bash
git clone https://github.com/OTeam-AI4S/OFoldX
cd OFoldX
pip install -e .
```

Design operations have family-specific scientific controls. Load the concrete
Pipeline selected by the repository and call its exact interface:

```python
import ofoldx
from ofoldx.data import Structure

structure: Structure = ...  # construct or obtain one canonical Structure
pipeline = ofoldx.pipeline("design", model="oteam/proteinmpnn")
result = pipeline.design(structure)
print(result.candidates[0].system)
```

The same concrete Pipeline exposes that family's complete controls:

```python
result = pipeline.design(structure, num_candidates=4)
```

Use the `Auto*` factories only when you need the raw registered pair:

```python
from ofoldx import AutoModel, AutoProcessor

processor = AutoProcessor("oteam/model-name")
model = AutoModel("oteam/model-name")
```

Train a family with a built-in Runner objective from one ordinary configuration:

```python
from ofoldx.runners import Runner

runner = Runner(
    {
        "epochs": 1,
        "model": {"name": "proteinmpnn"},
        "data": {"train": {"source": "structures/"}},
        "optim": {"type": "adamw", "lr": 1e-4},
        "dataloader": {"batch_size": 1},
    }
)
runner.train()
runner.close()
```

Families without a faithful built-in objective remain available through their
Model and Pipeline APIs; the Runner does not fabricate a loss for them.

Write one registered Model/Processor pair as a standard repository with the sole
public artifact writer:

```python
from ofoldx.artifacts import save_model

save_model("./model-repository", model=model, processor=processor)
```

The writer emits `model.json`, `processor.json`, and standard safetensors weights.
A registered concrete Pipeline additionally writes `pipeline.json` through its
own `save()` method and is reconstructed by `ofoldx.pipeline(...)`. Model-card
authorship and Hub publishing remain separate.

Canonical PDB/mmCIF input and output live in `ofoldx.io`; MultiMolecule owns FASTA
and the other biological-sequence formats used by family Processors.

`AutoModel` and `AutoProcessor` accept either a local model directory or a Hugging
Face Hub repository ID. Pipeline repositories persist only the concrete Pipeline
ID; the class owns how its model/processor pair or fixed composition is loaded.

## Explore

| Package                                | Purpose                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| [`ofoldx.models`](ofoldx/models)       | Model registration; concrete family packages are imported explicitly.                 |
| `ofoldx.artifacts`                     | The one public writer for a validated standard Model/Processor repository.            |
| `ofoldx.hub`                           | Hugging Face Hub repository resolution, publication, and loading.                     |
| [`ofoldx.pipelines`](ofoldx/pipelines) | Concrete family Pipeline registration and shared paired/composite persistence.        |
| [`ofoldx.modules`](ofoldx/modules)     | Shared neural-network building blocks: attention, trunk, heads, denoising, diffusion. |
| [`ofoldx.io`](ofoldx/io)               | Canonical PDB/mmCIF structure I/O; sequence-format I/O belongs to MultiMolecule.      |
| [`ofoldx.data`](ofoldx/data)           | Canonical molecular, evidence, placement, and scientific domain values.               |
| `ofoldx.runners`                       | Config-only Danling Runner for unified training and evaluation.                       |

The low-level values exported by `ofoldx.data` are part of the domain API: atom, bond,
component, attachment, entity, instance, and site values are what make an in-memory
`MolecularSystem` and `Structure` referentially closed.

## Status

OFoldX is pre-release research software under the AGPL-3.0 license. V1 is not complete
or released until all normative verbs, inputs, Results, and complete PDB/mmCIF I/O
pass their automated release gates.
