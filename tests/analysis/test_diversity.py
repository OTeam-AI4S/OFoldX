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

from ofoldx.analysis.diversity import greedy_diversity_selection


@pytest.mark.parametrize("budget", [2, 9])
def test_size_quotas_apply_below_and_above_collection_size(budget: int) -> None:
    selected = greedy_diversity_selection(
        ("A", "AA", "AAA", "AAAA", "AAAAA"),
        (1.0, 0.9, 0.8, 0.7, 0.6),
        budget=budget,
        alpha=0.0,
        size_buckets=((1, 3, 1), (3, 5, 1)),
    )
    assert selected == ((0, 2) if budget == 2 else (0, 2, 4))


@pytest.mark.parametrize("budget", [1, 5])
def test_first_candidate_must_be_eligible(budget: int) -> None:
    selected = greedy_diversity_selection(
        ("AA", "AAA", "AAAAAA"),
        (1.0, 0.9, 0.8),
        budget=budget,
        alpha=0.0,
        size_buckets=((2, 3, 0), (3, 4, 1)),
    )
    assert selected == ((1,) if budget == 1 else (1, 2))


def test_zero_budget_selects_nothing() -> None:
    assert greedy_diversity_selection(("AA", "AAA"), (1.0, 0.5), budget=0) == ()


def test_entirely_excluded_candidates() -> None:
    assert greedy_diversity_selection(("AA", "AAA"), (1.0, 0.5), budget=5, size_buckets=((1, 8, 0),)) == ()


def test_size_bucket_boundaries_are_half_open() -> None:
    assert greedy_diversity_selection(
        ("A", "AA", "AAA", "AAAA", "AAAAA"),
        (0.6, 1.0, 0.9, 0.8, 0.7),
        budget=5,
        alpha=0.0,
        size_buckets=((2, 4, 0),),
    ) == (0, 3, 4)


def test_overlapping_buckets_use_the_first_matching_range() -> None:
    assert greedy_diversity_selection(
        ("AAA", "AAAA", "AAAAAA"),
        (1.0, 0.9, 0.8),
        budget=3,
        alpha=0.0,
        size_buckets=((1, 4, 0), (3, 6, 1)),
    ) == (1, 2)


def test_equal_gains_keep_earliest_candidates_and_stable_output_order() -> None:
    assert greedy_diversity_selection(
        ("AA", "AA", "AA", "AAAA"),
        (1.0, 1.0, 1.0, 0.5),
        budget=3,
        size_buckets=((2, 3, 2),),
    ) == (0, 1, 3)


def test_alignment_gain_can_override_quality() -> None:
    # AAAA/AAAC have normalized alignment .75; AAAA/CCCC have score zero.
    # With alpha=.8, their gains are .38 and .96 respectively.
    assert greedy_diversity_selection(
        ("AAAA", "AAAC", "CCCC"),
        (1.0, 0.9, 0.8),
        budget=2,
        alpha=0.8,
    ) == (0, 2)


def test_unrestricted_full_collection_keeps_original_order() -> None:
    assert greedy_diversity_selection(("AA", "AAA", "AAAA"), (0.3, 0.9, 0.6), budget=5) == (0, 1, 2)
