# pipelines

`ofoldx.pipelines` holds the task-level inference boundaries — a `Pipeline` pairs a
model with its structure processor behind a single callable. A pipeline accepts a parsed
[`Structure`](../data/structure.py), a structure file path, or an already-typed model
input; it runs the processor when one is needed, moves the batch to the model device,
runs the forward pass, and normalizes the raw model output into a task result.

`from_config` builds the registered model and wraps it in the matching pipeline, so a
task is described by a single config. `save_pretrained` writes the model and processor
side by side under a root `manifest.json`, and `from_pretrained`
round-trips that pair — dispatching to the concrete task class from either a plain model
artifact or a saved pipeline package. See [`DESIGN.md`](../../DESIGN.md) and
[`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md) for the full contract.

## Contents

- `base.py` — the `Pipeline` inference boundary, the `StructureProcessor` protocol, and the input predicate that decides when a structure must be processed first.
- `design.py` — `DesignPipeline` and `DesignInput`; generates design candidates and coerces the several generator outputs into one `DesignOutput`.
- `structure_prediction.py` — `StructurePredictionPipeline` and `StructurePredictionInput`; runs a prediction model under inference mode and returns its `StructurePredictionOutput`.
- `auto.py` — pipeline-package manifest helpers: build, read, and validate `manifest.json`, and resolve the concrete pipeline class it names.
