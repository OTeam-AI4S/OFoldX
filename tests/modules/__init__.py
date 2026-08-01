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

r"""
Module tests.
"""

import pytest


def require_oops(*operators: str) -> None:
    """Skip an operator-specific case only when its OOps exports are unavailable."""

    oops = pytest.importorskip("oops")
    missing = tuple(name for name in operators if not callable(getattr(oops, name, None)))
    if missing:
        pytest.skip(f"installed OOps does not provide: {', '.join(missing)}")
