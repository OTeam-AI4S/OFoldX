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

"""Behaviour of ofoldx.templates.alignment: template alignments, target resolution and frames."""

from __future__ import annotations

import pytest

from ofoldx.data.molecular import MonomerType
from ofoldx.errors import UnsupportedError
from ofoldx.templates import ResolvedTemplateAlignment, TemplateAlignment, resolve_template, template_alignment_frames


class TestTemplate:
    @pytest.mark.parametrize(("pairs", "valid"), [(((0, 0),), True), (((1, 1),), True), (((0, 1),), False)])
    def test_one_position_per_target(self, template, invalid, pairs, valid):
        first = TemplateAlignment("1", None, "X", ((0, 0),))
        second = TemplateAlignment("1", "A", "X", pairs)
        if valid:
            template(first, second)
        else:
            invalid(template, first, second)

    def test_alignment_pairs_increase(self, invalid):
        invalid(TemplateAlignment, "1", None, "X", ((1, 1), (0, 2)))


class TestResolveTemplate:
    def test_broadcast_binds_instances(self, structure, template):
        resolved = resolve_template(
            template(TemplateAlignment("1", None, "X", ((0, 0), (2, 1)))),
            structure.system,
            monomer_types={MonomerType.PROTEIN},
        )
        assert [(r.template_instance_id, r.target_instance_index) for r in resolved] == [("X", 0), ("X", 1)]

    @pytest.mark.parametrize(
        ("entity", "instance", "pairs"),
        [("9", None, ((0, 0),)), ("1", "L", ((0, 0),)), ("1", "Z", ((0, 0),)), ("1", None, ((3, 0),))],
        ids=["entity", "other-entity", "instance", "past-instance"],
    )
    def test_unknown_target(self, structure, template, invalid, entity, instance, pairs):
        alignment = TemplateAlignment(entity, instance, "X", pairs)
        invalid(
            resolve_template,
            template(alignment),
            structure.system,
            monomer_types={MonomerType.PROTEIN},
            code="template_target_unknown",
        )

    def test_unsupported_target(self, structure, template):
        with pytest.raises(UnsupportedError) as caught:
            resolve_template(
                template(TemplateAlignment("1", "A", "X", ((0, 0),))), structure.system, monomer_types={MonomerType.RNA}
            )
        assert caught.value.code == "template_target_unsupported"


class TestTemplateAlignmentFrames:
    FIRST = ResolvedTemplateAlignment("X", 0, ((0, 0), (1, 1)))
    DISJOINT = ResolvedTemplateAlignment("X", 1, ((0, 2),))
    REUSED = ResolvedTemplateAlignment("X", 0, ((2, 0),))

    @pytest.mark.parametrize(
        ("alignments", "frames"),
        [((FIRST, DISJOINT), (0, 0)), ((FIRST, REUSED), (1, 2)), ((FIRST, FIRST, REUSED), (1, 1, 2))],
        ids=["disjoint", "reused", "duplicate"],
    )
    def test_frames(self, alignments, frames):
        assert template_alignment_frames(alignments) == frames

    def test_incompatible_frames(self):
        with pytest.raises(UnsupportedError) as caught:
            template_alignment_frames((self.FIRST, self.REUSED, ResolvedTemplateAlignment("X", 0, ((1, 1),))))
        assert caught.value.code == "template_frame_incompatible"
