# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

"""Builders shared by the ESMFold2 tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.msa import MSA


@pytest.fixture
def referenced_repository(tmp_path, esm_config):
    from multimolecule import EsmCConfig, EsmCModel

    from ofoldx.models.esmfold2 import EsmFold2Model, EsmFold2Pipeline, EsmFold2Processor, EsmFold2ProcessorConfig

    config = esm_config(False)
    backbone = EsmCModel(EsmCConfig(**config.esmc.dict()), add_pooling_layer=False)
    config.esmc_repository = str(tmp_path / "esmc")
    backbone.save_pretrained(config.esmc_repository)
    pipeline = EsmFold2Pipeline(
        model=EsmFold2Model(config, esmc=backbone).eval().set_backend("eager"),
        processor=EsmFold2Processor(EsmFold2ProcessorConfig(msa_conditioning=False)),
    )
    destination = tmp_path / "native"
    pipeline.save(destination)
    return destination, pipeline


@pytest.fixture
def msa_pool():
    """``msa_pool(query, *rows)``: the pool of the ``peptide`` fixture's chain ``A``, with zero deletions."""

    def build(query, *rows):
        unpaired = (query, *rows)
        counts = np.zeros((len(unpaired), len(query)), dtype=np.uint32)
        return {"A": MSA("P", unpaired, counts, (query,), np.zeros((1, len(query)), dtype=np.uint32), ("query",))}

    return build


@pytest.fixture
def experimental_referenced_repository(tmp_path, esm_config):
    from multimolecule import EsmCConfig, EsmCForMaskedLM

    from ofoldx.models.esmfold2 import (
        EsmFold2ExperimentalModel,
        EsmFold2ExperimentalPipeline,
        EsmFold2ExperimentalProcessor,
    )

    config = esm_config(True)
    config.confidence.enabled = False
    torch.manual_seed(23)
    language_model = EsmCForMaskedLM(EsmCConfig(**config.esmc.dict()))
    config.esmc_repository = str(tmp_path / "esmc")
    language_model.save_pretrained(config.esmc_repository)
    pipeline = EsmFold2ExperimentalPipeline(
        model=EsmFold2ExperimentalModel(config, esmc=language_model).eval().set_backend("eager"),
        processor=EsmFold2ExperimentalProcessor(),
    )
    destination = tmp_path / "native"
    pipeline.save(destination)
    return destination, pipeline
