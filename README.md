# OFoldX

OFoldX is a research toolkit for biomolecular structure prediction and design. It
follows a small, consistent shape: processors turn structures into typed model inputs,
models consume those inputs, and pipelines package the processor/model boundary for
inference.

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

Load a task pipeline directly from a local directory or Hub repository:

```python
from ofoldx.pipelines import Pipeline

pipeline = Pipeline.from_pretrained("oteam/model-name")
output = pipeline("complex.cif")
```

Drop down to the `Auto*` factories when you need direct model or processor control:

```python
from ofoldx import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("oteam/model-name")
model = AutoModel.from_pretrained("oteam/model-name")
batch = processor("complex.cif")
output = model(batch)
```

Pretrained objects use `save_pretrained` / `from_pretrained` and are compatible with local
directories or Hugging Face Hub repositories.

## Hub Round Trip

Any saved model, processor, or pipeline is an artifact directory with a root `manifest.json`.
Push helpers generate a Hub `README.md` with task metadata, static examples, and a pipeline-first usage path:

```python
from ofoldx.pipelines import Pipeline

pipeline = Pipeline.from_pretrained("./protein-model")
pipeline.push_to_hub("oteam/protein-model", private=True, tag="v0.1.0")
loaded = Pipeline.from_pretrained("oteam/protein-model")
```

Converters expose the same upload switches:

```bash
python -m ofoldx.models.rfdiffusion.convert Base_ckpt.pt out/rfdiffusion \
  --push-to-hub --repo-id oteam/rfdiffusion-base --tag v0.1.0
```

Release helpers keep Hub cards and pipeline packages reproducible:

```bash
python scripts/publish_hub_cards.py --root artifacts/hub/ofoldx --namespace oteam --upload --verify-remote
python scripts/build_pipeline_artifacts.py --root artifacts/hub/ofoldx --output-root artifacts/hub/ofoldx-pipelines \
  --copy --overwrite --upload --delete-existing-files --verify-remote
python scripts/validate_hub_artifacts.py --root artifacts/hub/ofoldx-pipelines --load-pipeline
```

## Explore

| Package                                  | Purpose                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| [`ofoldx.models`](ofoldx/models)         | Model families — per-family config, typed input/output, checkpoint conversion, and the model. |
| [`ofoldx.processors`](ofoldx/processors) | Structure/MSA/template processors that build typed model inputs, plus `AutoProcessor`.        |
| [`ofoldx.pipelines`](ofoldx/pipelines)   | Saveable processor + model inference boundaries.                                              |
| [`ofoldx.modules`](ofoldx/modules)       | Shared neural-network building blocks: attention, trunk, heads, denoising, diffusion.         |
| [`ofoldx.io`](ofoldx/io)                 | Structure parsing and writing.                                                                |
| [`ofoldx.data`](ofoldx/data)             | Chemical/reference data and dataset helpers.                                                  |
| [`ofoldx.runners`](ofoldx/runners)       | DanLing-based train / predict / design / evaluate entry points above the models.              |

## Design

- [`DESIGN.md`](DESIGN.md) — architecture blueprint and locked design decisions.
- [`DESIGN_PRINCIPLES.md`](DESIGN_PRINCIPLES.md) — the review bar: contracts, protocols, and style.
- [`docs/naming.md`](docs/naming.md) — canonical vocabulary.
- [`docs/docstrings.md`](docs/docstrings.md) — docstring & doctest convention.

OFoldX is pre-alpha research software under the AGPL; see
[`DESIGN_PRINCIPLES.md`](DESIGN_PRINCIPLES.md) for the compatibility and conversion
policy.
