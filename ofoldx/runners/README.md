# runners

`ofoldx.runners` holds the [DanLing](https://danling.org)-based runners that sit above the
models and drive end-to-end **train**, **evaluate**, **predict**, and **design** over inputs —
booting the config, building the model, wiring datasets and loss, and owning the loop and IO.

The direction is one-way. Runners import [`ofoldx.models`](../models); models never import
runners. A runner is the infrastructure ring that the model deliberately does not carry: the model
owns only its forward pass and capability verb, input prep lives in the processor and [`io`](../io)
layers, and the runner owns the batching, the sample loop, and writing results to disk. Multi-model
composition (e.g. the backbone → sequence → structure-predictor filter workflow) lives in the generic
composite [`AllAtomDesignPipeline`](../pipelines/design.py), which injects the sibling providers and the
family's [`stages`](../models/pxdesign/stages.py) glue, not here.
See [`DESIGN.md`](../../DESIGN.md) and [`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md).

## Contents

- `runner.py` — `Runner`, the default `danling.Runner` specialization that trains, evaluates, and runs inference with any OFoldX model.
- `config.py` — `Config` and the `DataConfig` / `LossConfig` / `SamplingConfig` sections that parameterize a run.
- `registry.py` — `RUNNERS`, the config-driven registry mapping a `runner` name to its class.
