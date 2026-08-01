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

import pytest

from ofoldx.models.mpnn_packer import (
    MpnnPackerConfig,
    MpnnPackerModel,
    MpnnPackerPipeline,
    MpnnPackerProcessor,
    MpnnPackerProcessorConfig,
)


@pytest.fixture
def small_packer():
    config = MpnnPackerConfig(
        hidden_size=8,
        edge_size=8,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        num_rbf=2,
        num_neighbors=1,
        num_context_atoms=1,
        num_mix=1,
        dropout=0.0,
    )
    return MpnnPackerPipeline(
        model=MpnnPackerModel(config).eval(),
        processor=MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=1)),
    )
