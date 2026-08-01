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
Scientific scores and confidence expectations.
"""

from .confidence import (
    categorical_entropy_from_logits,
    categorical_expectation,
    categorical_mean,
    categorical_probability_below,
    interface_confidence_weights,
    minimum_confidence_error,
    plddt_from_logits,
    resolved_probability_from_logits,
)
from .diversity import normalized_psd_projection, sequence_vendi_score, vendi_score
from .sequence import (
    amino_acid_fractions,
    sequence_alignment_score,
    sequence_hydrophobicity,
    sequence_isoelectric_point,
    sequence_nll,
    sequence_recovery,
    token_recovery,
)
from .structure import (
    aligned_rmsd,
    backbone_secondary_structure,
    binding_site_adherence,
    binding_site_adherence_curve,
    lddt,
    lddt_from_distances,
    secondary_structure_agreement,
    secondary_structure_fractions,
    structure_tm_scores,
)
from .tm import (
    chain_pair_mean_tm_scores,
    chain_pair_tm_scores,
    directional_ipsae,
    mean_partner_tm_scores,
    predicted_tm_score,
    tm_bin_weights,
    tm_length_scale,
    tm_weight_from_pae,
)
