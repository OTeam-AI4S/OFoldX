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

import pickle

import pytest

from ofoldx.errors import ContradictionError
from ofoldx.io import parse_source, read_structure


class TestReadReport:
    def test_summary_counts_bulk(self, library, dipeptide):
        _, report = read_structure(parse_source(dipeptide(), format="mmcif"), library=library)
        assert report.summary()["bulk.standard_junction_endpoint"] == 2

    def test_pickles_with_its_refusal(self, library, disulfide):  # a process pool returns both
        source = parse_source(disulfide, format="mmcif")
        _, report = read_structure(source, library=library, on_conflict="record")
        assert pickle.loads(pickle.dumps(report)).summary() == report.summary()
        with pytest.raises(ContradictionError) as raised:
            read_structure(source, library=library)
        again = pickle.loads(pickle.dumps(raised.value))
        assert again.code == "read_conflicts" and again.detail["report"].summary() == report.summary()
