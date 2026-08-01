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

r"""Copy canonical NumPy columns to model tensors.

Canonical arrays are read-only. Model tensors receive independent storage on the
requested device, so modifying a feature leaves the source value unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ..data.molecular import MolecularSystem
from ..data.structure import Structure


def to_tensors(
    value: Structure | MolecularSystem, columns: Sequence[str], *, device: torch.device
) -> Mapping[str, torch.Tensor]:
    r"""Copy named canonical columns to PyTorch tensors on a device.

    Args:
        value: A molecular system or structure supplying the columns. A structure
            exposes both its own columns and its system's columns.
        columns: Names of the columns to copy.
        device: Destination device.

    Returns:
        Tensors indexed by column name, with storage independent of the source arrays.
        Memory use is proportional to the selected columns.
    """

    own = value.arrays()
    system = value.system.arrays() if isinstance(value, Structure) else own
    return {name: torch.from_numpy(np.array(own[name] if name in own else system[name])).to(device) for name in columns}
