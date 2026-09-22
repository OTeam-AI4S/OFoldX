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

r"""Bundled crystallographic space-group reference data."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

__all__ = ["load_table"]


def load_table() -> dict[str, Any]:
    r"""Load the bundled space-group table and its provenance."""

    text = (
        resources.files("ofoldx.reference.space_groups")
        .joinpath("space_group_operators.json")
        .read_text(encoding="utf-8")
    )
    return cast(dict[str, Any], json.loads(text))
