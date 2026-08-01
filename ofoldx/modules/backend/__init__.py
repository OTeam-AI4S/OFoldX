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
Dispatch across supported acceleration backends.
"""

from __future__ import annotations

from .adaptive_norm import adaptive_layer_norm
from .atom_window import atom_to_token_mean, token_pair_to_atom_window_gather, token_to_atom_broadcast
from .attention import multihead_attention, projected_attention, triangle_attention
from .categorical_msa_projection import categorical_msa_projection
from .equivariant import Se3Basis, SegmentSoftmaxCsr, SegmentSumCsr, WeightedTensorProduct
from .factorized_pair_linear import factorized_pair_linear
from .knn import gather_nodes, masked_knn
from .normalization import LayerNorm, layer_norm
from .outer_product_mean import outer_product_mean
from .pair_norm_projection import pair_norm_projection
from .pair_weighted_average import pair_weighted_average
from .resolve import Backend, resolve_config_backend
from .transition import transition
from .triangle_multiplication import triangle_multiplication
