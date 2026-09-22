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

"""Behaviour of ofoldx.io.tables: the selected model's checks raise, the other models' are recorded."""

import pytest

from ofoldx.errors import ContradictionError
from ofoldx.io.source import parse_source


class TestModelView:
    @pytest.fixture
    def source(self, mmcif_text):
        # model 2's zinc names an asym _struct_asym does not declare (1ts6)
        return parse_source(mmcif_text.replace("HETATM 6 ZN ZN . ZN B", "HETATM 6 ZN ZN . ZN C"), format="mmcif")

    def test_other_model_recorded(self, source):
        records = source.tables(1).records
        detail = {"model": 2, "check": "asym_undeclared", "asym": "C"}
        assert (records.counts(), records[0].detail) == ({"entry_level_violation": 1}, detail)

    def test_selected_model_refused(self, source):
        with pytest.raises(ContradictionError, match="asym_undeclared"):
            source.tables(2)
