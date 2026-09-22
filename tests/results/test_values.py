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

from ofoldx.results.values import (
    DesignAndPackResult,
    DesignCandidate,
    DesignResult,
    FoldPrediction,
    FoldResult,
    PackResult,
    Score,
)


@pytest.fixture
def structure(peptide):
    return peptide("ALA")


class TestFoldResult:
    def test_best_by(self, structure) -> None:
        first = FoldPrediction(structure, scores={"quality": Score(1.0, preferred="higher")})
        second = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="higher")})
        tied = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="higher")})

        assert FoldResult((first, second, tied)).best_by("quality") is second

    def test_direction(self, structure) -> None:
        higher = FoldPrediction(structure, scores={"quality": Score(1.0, preferred="higher")})
        lower = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="lower")})

        with pytest.raises(ValueError):
            FoldResult((higher, lower)).best_by("quality")


class TestDesignResult:
    def test_best_by(self, structure) -> None:
        first = DesignCandidate(structure, scores={"quality": Score(2.0, preferred="lower")})
        second = DesignCandidate(structure, scores={"quality": Score(1.0, preferred="lower")})

        assert DesignResult((first, second)).best_by("quality") is second

    def test_candidates(self, structure) -> None:
        candidate = DesignCandidate(structure)
        result = DesignAndPackResult(DesignResult((candidate,)), (PackResult(()),))

        assert result.candidates == (candidate,)
