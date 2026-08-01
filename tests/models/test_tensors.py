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

"""Behavior of ofoldx.models.tensors."""

from __future__ import annotations

import torch

from ofoldx.models.tensors import to_tensors


class TestToTensors:
    def test_columns_of_both_records(self, peptide):
        structure = peptide("GLY")
        tensors = to_tensors(structure, ("coordinates", "atomic_number"), device=torch.device("cpu"))
        assert tensors["coordinates"].shape == (len(structure.coordinates), 3)
        assert tensors["atomic_number"].tolist() == structure.system.atomic_number.tolist()
