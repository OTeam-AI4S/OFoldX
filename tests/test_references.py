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

from pathlib import Path

import pytest

from ofoldx.references import classify_reference


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("./missing", Path("./missing")),
        ("~/missing", Path("~/missing").expanduser()),
        (r"C:\missing", Path(r"C:\missing")),
        ("owner/repo", "owner/repo"),
        ("", ""),
    ],
)
def test_classification(reference: str, expected: Path | str) -> None:
    assert classify_reference(reference) == expected


def test_existing_path(tmp_path: Path) -> None:
    assert classify_reference(str(tmp_path)) == tmp_path
