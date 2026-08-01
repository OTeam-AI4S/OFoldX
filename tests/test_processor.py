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

from __future__ import annotations

import torch

from ofoldx.processor import Processor


class TestProcessor:
    def test_placement(self) -> None:
        processor = Processor()
        assert processor.device == torch.device("cpu")
        assert processor.dtype == torch.float32

        processor = Processor(device="meta", dtype=torch.float64)
        assert processor.to() is processor
        assert processor.device == torch.device("meta")
        assert processor.dtype == torch.float64

        processor.to(device="cpu", dtype=torch.float32)
        assert processor.device == torch.device("cpu")
        assert processor.dtype == torch.float32


def test_prepare_composes_input_and_target():
    class ExampleProcessor(Processor):
        def prepare_input(self, source, *, selection):
            return source[selection]

        def prepare_target(self, truth, *, prepared):
            return truth + prepared

    processor = ExampleProcessor()
    source = torch.tensor([2.0, 5.0, 11.0])
    input_only = processor.prepare_input(source, selection=1)
    prepared, absent = processor.prepare(source, selection=1)
    torch.testing.assert_close(prepared, input_only)
    assert absent is None
    result = processor.prepare(source, target=torch.tensor(7.0), selection=1)
    assert type(result) is tuple
    combined, target = result
    torch.testing.assert_close(combined, input_only)
    torch.testing.assert_close(target, processor.prepare_target(torch.tensor(7.0), prepared=input_only))
