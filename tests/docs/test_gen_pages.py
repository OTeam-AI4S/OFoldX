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

import pytest

from docs.gen_pages import ROOT, rewrite_readme_links


class TestReadmeLinks:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("license.md", "../about/license.md"),
            ("license.md#conditions", "../about/license.md#conditions"),
            (
                "https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md",
                "../about/license.md",
            ),
            ("README.md", "../index.md"),
            ("ofoldx/chemistry", "../chemistry/index.md"),
            ("ofoldx/msa", "../msa/index.md"),
            ("ofoldx/templates", "../templates/index.md"),
            ("ofoldx/reference", "../reference-data/index.md"),
            ("docs", "https://github.com/OTeam-AI4S/OFoldX/tree/main/docs"),
            ("pyproject.toml", "https://github.com/OTeam-AI4S/OFoldX/blob/main/pyproject.toml"),
            ("https://example.org/guide", "https://example.org/guide"),
            ("#installation", "#installation"),
        ],
    )
    def test_readme_links(self, target, expected):
        result = rewrite_readme_links(f"[Guide]({target})", readme=ROOT / "README.md", page="guide/index.md")
        assert result == f"[Guide]({expected})"
