from __future__ import annotations

import pytest
import torch.nn as nn

import ofoldx
from ofoldx import tasks
from ofoldx.pipelines import (
    BindingAffinityPipeline,
    DesignPipeline,
    SideChainPackingPipeline,
    StructurePredictionPipeline,
)


@pytest.mark.parametrize(
    ("task", "cls"),
    [
        ("fold", StructurePredictionPipeline),
        ("structure-prediction", StructurePredictionPipeline),
        ("design", DesignPipeline),
        ("score", BindingAffinityPipeline),
        ("pack", SideChainPackingPipeline),
    ],
)
def test_task_maps_to_pipeline_class(task, cls):
    assert tasks._pipeline_class(task) is cls


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        tasks._pipeline_class("teleport")


def test_verb_requires_capability():
    # a provider lacking the verb yields a clear capability error, not an AttributeError
    with pytest.raises(TypeError, match="does not implement the 'fold' capability"):
        ofoldx.fold(object(), model=nn.Identity())


def test_default_model_catalog(monkeypatch):
    monkeypatch.setattr(tasks, "_DEFAULT_MODEL", {})
    with pytest.raises(ValueError, match="no model given and no default"):
        ofoldx.pipeline("design")
    ofoldx.register_default_model("design", "oteam/x")
    assert tasks._resolve_model("design", "oteam/y") == "oteam/y"
    assert tasks._resolve_model("design", None) == "oteam/x"
