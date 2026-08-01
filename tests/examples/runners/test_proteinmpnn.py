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

import json
import math


def test_runs(examples):
    assert examples["proteinmpnn"] == ""


def test_scoring_example(examples, examples_directory):
    assert examples["proteinmpnn"] == ""
    assert examples["proteinmpnn_score"] == ""
    scores = json.loads((examples_directory / "proteinmpnn-scores.json").read_text())
    assert set(scores) == {"designed_sequence_nll", "global_sequence_nll"}
    expected_scope = [{"instance": "A", "component": index} for index in range(3)]
    for score in scores.values():
        assert math.isfinite(score["value"]) and score["value"] >= 0
        assert score["unit"] == "nat/residue"
        assert score["preferred"] == "lower"
        assert score["scope"] == expected_scope
