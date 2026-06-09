# runners

`ofoldx.runners` holds the [DanLing](https://danling.org)-based runners that sit above the
models and drive end-to-end **train**, **evaluate**, **predict**, and **design** over inputs —
booting the config, building the model, wiring datasets and loss, and owning the loop and IO.

The direction is one-way. Runners import [`ofoldx.models`](../models); models never import
runners. A runner is the infrastructure ring that the model deliberately does not carry: the model
owns only its forward / `generate` pass, input prep lives in the processor and [`io`](../io) layers,
and the runner owns the batching, the sample loop, and writing results to disk. This is also the
only ring allowed to compose sibling models — the PxDesign design pipeline runs backbone generation,
then ProteinMPNN sequence assignment and structure-predictor filtering, each a separate model.
See [`DESIGN.md`](../../DESIGN.md) and [`DESIGN_PRINCIPLES.md`](../../DESIGN_PRINCIPLES.md).

## Contents

- `runner.py` — `Runner`, the default `danling.Runner` specialization that trains, evaluates, and runs inference with any OFoldX model.
- `pxdesign.py` — `PxDesignInferenceRunner` and `build_pxdesign_inference_runner`, the end-to-end PxDesign design pipeline (generate, assign sequences, filter, dump).
- `config.py` — `Config` and the `DataConfig` / `LossConfig` / `SamplingConfig` sections that parameterize a run.
- `registry.py` — `RUNNERS`, the config-driven registry mapping a `runner` name to its class.
