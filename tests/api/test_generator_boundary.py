r"""Lock the single seed|generator resolution point at the Pipeline boundary.

Per the OWNER principle the model graph is a pure function of ``(tensors, generator)``: a Pipeline
accepts ``seed=`` / ``generator=`` / neither, resolves them once, and forwards the resulting
``torch.Generator`` (or ``None``) to the Model -- never an integer ``seed``. Generic task Processors
are deterministic and do not receive the stream. These tests drive all four standard Pipelines with
a synthetic Model/Processor pair and assert both sides of that boundary.
"""

from __future__ import annotations

import pytest
import torch

from ofoldx.pipelines import (
    DesignPipeline,
    FoldPipeline,
    PackPipeline,
    ScorePipeline,
)

# (Pipeline class, task method name) for the four standard one-shot tasks.
_PIPELINES = [
    (FoldPipeline, "fold"),
    (DesignPipeline, "design"),
    (ScorePipeline, "score"),
    (PackPipeline, "pack"),
]


def _make_pipeline(pipeline_class, tiny_classes):
    _, TinyProcessor, TinyModel = tiny_classes()
    model = TinyModel({"dim": 2})
    processor = TinyProcessor()
    return pipeline_class(model=model, processor=processor), model


@pytest.mark.parametrize(("pipeline_class", "method"), _PIPELINES)
@pytest.mark.parametrize("rng", ["seed", "generator", "neither"])
def test_pipeline_accepts_seed_generator_or_neither(pipeline_class, method, rng, tiny_classes):
    r"""Every standard Pipeline accepts seed / generator / neither with no TypeError."""

    pipeline, model = _make_pipeline(pipeline_class, tiny_classes)
    kwargs = {"seed": 123} if rng == "seed" else {"generator": torch.Generator()} if rng == "generator" else {}

    getattr(pipeline, method)("input", **kwargs)

    forwarded = model.received
    assert forwarded["method"] == method
    assert "generator" not in pipeline.processor.received
    if rng == "neither":
        assert forwarded["generator"] is None
    else:
        assert isinstance(forwarded["generator"], torch.Generator)


@pytest.mark.parametrize(("pipeline_class", "method"), _PIPELINES)
def test_pipeline_never_leaks_seed_into_model(pipeline_class, method, tiny_classes):
    r"""The integer seed is resolved at the boundary and never forwarded to the model graph."""

    pipeline, model = _make_pipeline(pipeline_class, tiny_classes)

    getattr(pipeline, method)("input", seed=7)

    assert "seed" not in model.received


@pytest.mark.parametrize(("pipeline_class", "method"), _PIPELINES)
def test_pipeline_rejects_seed_and_generator_together(pipeline_class, method, tiny_classes):
    r"""seed and generator are mutually exclusive at the single resolution point."""

    pipeline, _ = _make_pipeline(pipeline_class, tiny_classes)

    with pytest.raises(ValueError, match="mutually exclusive"):
        getattr(pipeline, method)("input", seed=1, generator=torch.Generator())
