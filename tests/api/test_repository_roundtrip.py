r"""Lock the repository save/load round-trip through the public task API.

DESIGN §9.2 requires end-to-end coverage of a single Model/Processor pair: classify a local reference,
write a repository (``model.json`` architecture marker + safetensors weights, and -- when the concrete
Pipeline is registered -- a ``pipeline.json`` marker), then load it back through ``ofoldx.api.pipeline``
and recover the same Pipeline class, the same paired Model/Processor types, and the same weights.
"""

from __future__ import annotations

import json

import torch

from ofoldx.api import pipeline as load_pipeline
from ofoldx.models.registry import register_model
from ofoldx.pipelines import FoldPipeline
from ofoldx.pipelines.registry import register_pipeline
from ofoldx.references import classify_reference


def _register_pair(tiny_classes, *, model_id="test.tiny"):
    _, TinyProcessor, TinyModel = tiny_classes()
    register_model(model_id, model=TinyModel, processor=TinyProcessor, standard_tasks={"fold"})
    model = TinyModel({"dim": 3})
    # Distinctive weights so the round-trip actually verifies weight transport, not just shape.
    with torch.no_grad():
        model.weight.copy_(torch.arange(3, dtype=torch.float32))
    return model, TinyProcessor(), TinyModel, TinyProcessor


def test_standard_pipeline_round_trip(tmp_path, tiny_classes):
    r"""A standard FoldPipeline (no marker) saves and reloads via api.pipeline('fold')."""

    model, processor, TinyModel, TinyProcessor = _register_pair(tiny_classes)
    FoldPipeline(model=model, processor=processor).save(tmp_path)

    # No concrete Pipeline is registered, so no pipeline.json marker is written.
    assert not (tmp_path / "pipeline.json").exists()
    assert (tmp_path / "model.json").exists()
    # classify_reference resolves a local directory to a filesystem Path (not a Hub repo id).
    from pathlib import Path

    assert isinstance(classify_reference(tmp_path), Path)

    loaded = load_pipeline("fold", model=tmp_path)

    assert type(loaded) is FoldPipeline
    assert isinstance(loaded.model, TinyModel)
    assert isinstance(loaded.processor, TinyProcessor)
    assert torch.equal(loaded.model.weight, torch.arange(3, dtype=torch.float32))


def test_registered_pipeline_round_trip_via_marker(tmp_path, tiny_classes):
    r"""A registered concrete Pipeline writes a pipeline.json marker that the loader honors."""

    model, processor, TinyModel, TinyProcessor = _register_pair(tiny_classes)

    class MarkerFoldPipeline(FoldPipeline):
        r"""A concrete, registered Fold pipeline so save() emits a pipeline.json marker."""

    register_pipeline("test.tiny.fold", pipeline=MarkerFoldPipeline)
    MarkerFoldPipeline(model=model, processor=processor).save(tmp_path)

    marker = tmp_path / "pipeline.json"
    assert marker.exists()
    assert json.loads(marker.read_text()) == {"pipeline": "test.tiny.fold"}

    loaded = load_pipeline("fold", model=tmp_path)

    # The marker -- not the task default -- selects the concrete class, and pairs it back up.
    assert type(loaded) is MarkerFoldPipeline
    assert isinstance(loaded.model, TinyModel)
    assert isinstance(loaded.processor, TinyProcessor)
    assert torch.equal(loaded.model.weight, torch.arange(3, dtype=torch.float32))
