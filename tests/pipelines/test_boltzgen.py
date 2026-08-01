from __future__ import annotations

from ofoldx.models.boltz2 import Boltz2AffinityScorePipeline, Boltz2Pipeline
from ofoldx.models.boltzgen import (
    BoltzGenBackbonePipeline,
    BoltzGenInverseFoldingPipeline,
    BoltzGenPipeline,
)
from ofoldx.pipelines import PIPELINES


class TestBoltzGenPipeline:
    def test_components(self) -> None:
        assert PIPELINES.lookup_pipeline("boltzgen_backbone").pipeline is BoltzGenBackbonePipeline
        assert PIPELINES.lookup_pipeline("boltzgen_inverse_fold").pipeline is BoltzGenInverseFoldingPipeline

        pairs = {pair.role: (pair.subdirectory, pair.component) for pair in BoltzGenPipeline.pair_layout}
        assert pairs == {
            "design_diverse": ("", BoltzGenBackbonePipeline),
            "design_adherence": ("adherence", BoltzGenBackbonePipeline),
            "inverse_fold": ("ifold", BoltzGenInverseFoldingPipeline),
            "fold": ("fold", Boltz2Pipeline),
            "affinity": ("affinity", Boltz2AffinityScorePipeline),
        }
