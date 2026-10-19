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

from ofoldx.chemistry.attachments import rule_l
from ofoldx.data.molecular import EndpointState
from ofoldx.errors import CODE_NAMES

from .builders import edge, problem

S = EndpointState


def names(component, atoms) -> set[str]:
    return {component.atom_names[a] for a in atoms}


def reason(outcome) -> str:
    return CODE_NAMES[outcome.reason]


class TestRuleL:
    def test_designated_group_leaves_whole(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["type_symbol"][9] = "C"  # HXT as a carbon: a two-heavy-atom group designated at C
        ester = define(items(**columns))
        outcome = rule_l(problem(ester, "C", [edge(1)], observed={"O", "CA"}), ester)
        assert (outcome.state, names(ester, outcome.atoms)) == (S.DERIVED_FROM_REFERENCE, {"OXT", "HXT"})
        assert outcome.phantom == ()

    def test_unknown_element_has_no_class(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["type_symbol"][0] = "?"  # N of unknown element: neither class A nor class B
        gly = define(items(**columns))
        outcome = rule_l(problem(gly, "N", [edge(1)], observed={"CA"}), gly)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "reference_incomplete_absent")
