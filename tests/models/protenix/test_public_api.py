from __future__ import annotations

from ofoldx.models.protenix import ProtenixEvaluationOutput, ProtenixTrainingOutput
from ofoldx.models.protenix.model import (
    ProtenixEvaluationOutput as ModelProtenixEvaluationOutput,
)
from ofoldx.models.protenix.model import ProtenixTrainingOutput as ModelProtenixTrainingOutput


def test_training_output_is_exported_from_package_surface() -> None:
    assert ProtenixTrainingOutput is ModelProtenixTrainingOutput


def test_evaluation_output_is_exported_from_package_surface() -> None:
    assert ProtenixEvaluationOutput is ModelProtenixEvaluationOutput
