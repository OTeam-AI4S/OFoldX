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

from pathlib import Path

import pytest

from ofoldx.chemistry.definitions import define_component
from ofoldx.data.molecular import DefinitionScope
from ofoldx.io.cif import read_cif
from ofoldx.io.mmcif.decode import decode_components


@pytest.fixture(scope="module")
def inosine_components():
    # DI and I are cut from the wwPDB CCD snapshot.
    # Source SHA256: d5be0434a719108e2dd50708b17b0af26b088c1ce7d751a668aeecf9bdc26949.
    fixture = Path(__file__).parent / "fixtures" / "inosine.cif.gz"
    return {
        block.name: define_component(
            decode_components(block.categories, block=block.name).components[block.name], scope=DefinitionScope.ARCHIVE
        )
        for block in read_cif(fixture).blocks
    }
