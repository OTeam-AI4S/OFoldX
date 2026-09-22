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
Public in-memory scientific analysis over generated structures and sequences.
"""

from __future__ import annotations

from .diversity import (
    ConditionedVendiScores,
    DiversityAggregate,
    DiversityMetrics,
    DiversityResult,
    DiversitySample,
    FoldseekMatch,
    FoldseekNoveltyOperation,
    StructureVendiScores,
    analyze_diversity,
    deduplicate_sequences,
    foldseek_novelty,
    greedy_diversity_selection,
    structure_vendi_scores,
)
from .ranking import AnalysisValue, CandidateFilter, CandidateRanking, candidate_filter_results, rank_candidates
from .sequence import (
    Liability,
    LiabilityModality,
    PeptideType,
    liability_metrics,
    sequence_liabilities,
)
from .structure import (
    AnalysisUnavailableError,
    NoncovalentContacts,
    PresentAtomContacts,
    PresentAtomCoverage,
    PresentAtomHydrophobicPatch,
    PresentAtomSasa,
    PresentAtomUnavailableError,
    TargetDeltaSasa,
    assign_secondary_structure,
    count_noncovalent_contacts,
    largest_hydrophobic_patch_area,
    present_atom_largest_hydrophobic_patch_area,
    present_atom_noncovalent_contacts,
    present_atom_target_delta_sasa,
    structure_coordinates,
    structure_design_lddt,
    structure_rmsd_metrics,
    target_delta_sasa,
)
