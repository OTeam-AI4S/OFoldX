from __future__ import annotations

import pytest

from ofoldx.models import AutoConfig
from ofoldx.models.alphafold2.processor import AlphaFold2Processor
from ofoldx.models.alphafold3.processor import AlphaFold3Processor
from ofoldx.models.boltz2.processor import Boltz2Processor
from ofoldx.models.boltzgen.processor import BoltzGenProcessor
from ofoldx.models.odesign.processor import ODesignProcessor
from ofoldx.models.rfdiffusion.processor import RfDiffusionProcessor
from ofoldx.processors import AutoProcessor


@pytest.mark.parametrize(
    ("architecture", "processor_class"),
    [
        ("alphafold2", AlphaFold2Processor),
        ("alphafold3", AlphaFold3Processor),
        ("boltz2_affinity", Boltz2Processor),
        ("boltzgen_diverse", BoltzGenProcessor),
        ("boltzgen_adherence", BoltzGenProcessor),
        ("boltzgen_structuretrained_small", BoltzGenProcessor),
        ("odesign_design", ODesignProcessor),
    ],
)
def test_new_model_architectures_have_default_processors(architecture: str, processor_class: type) -> None:
    processor = AutoProcessor.from_config(AutoConfig.from_config({"architecture": architecture}))

    assert isinstance(processor, processor_class)


def test_rfdiffusion_has_default_processor() -> None:
    processor = AutoProcessor.from_config({"architecture": "rfdiffusion"})

    assert isinstance(processor, RfDiffusionProcessor)
