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

"""Behavior of sequence composition annotations."""

import pytest

from ofoldx.analysis.sequence import liability_metrics, sequence_liabilities


class TestComposition:
    @pytest.mark.parametrize("operation", [sequence_liabilities, liability_metrics])
    @pytest.mark.parametrize(
        "sequence,options",
        [("", {}), ("A", {"modality": "unknown"}), ("A", {"peptide_type": "unknown"})],
    )
    def test_invalid_request(self, operation, sequence, options):
        with pytest.raises(ValueError):
            operation(sequence, **options)

    @pytest.mark.parametrize(
        ("modality", "peptide_type"),
        [("antibody", "linear"), ("peptide", "linear"), ("peptide", "cyclic")],
    )
    @pytest.mark.parametrize("sequence", ["C", "CAC", "CACAC"])
    def test_cysteine_count(self, modality, peptide_type, sequence):
        annotations = sequence_liabilities(sequence, modality=modality, peptide_type=peptide_type)
        counts = [item for item in annotations if item.motif == "OddCysteineCount"]
        if sequence.count("C") % 2:
            assert len(counts) == 1
            assert counts[0].position is None
            assert counts[0].details == f"{sequence.count('C')} cysteines"
        else:
            assert not counts
        assert all(item.motif != "UnpairedCys" for item in annotations)

    @pytest.mark.parametrize("sequence,excess", [("KKRDE", 1), ("KKRD", 2), ("H", 0)])
    def test_basic_residue_excess(self, sequence, excess):
        annotations = sequence_liabilities(sequence)
        basic = [item for item in annotations if item.motif == "BasicResidueExcess"]
        if excess > 1:
            assert len(basic) == 1
            assert basic[0].details == f"K+R-D-E={excess}"
        else:
            assert not basic
        assert all(item.motif != "HighNetCharge" for item in annotations)

    def test_report_composition(self):
        report = liability_metrics("CKK")
        assert report["liability_OddCysteineCount_count"] == 1
        assert report["liability_OddCysteineCount_position"] == -1
        assert report["liability_BasicResidueExcess_details"] == "K+R-D-E=2"
