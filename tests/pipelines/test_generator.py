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

import pytest
import torch

from ofoldx.pipelines import resolve_generator


class TestResolveGenerator:
    def test_default(self) -> None:
        assert resolve_generator(None, None) is None

    def test_supplied(self) -> None:
        supplied = torch.Generator()

        assert resolve_generator(None, supplied, device="cuda") is supplied

    def test_seed(self) -> None:
        generator = resolve_generator(0, None, device="cpu")

        assert generator is not None
        assert generator.device == torch.device("cpu")
        assert generator.initial_seed() == 0

    def test_conflict(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_generator(0, torch.Generator())
