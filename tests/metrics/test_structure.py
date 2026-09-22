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

import numpy as np
import pytest

from ofoldx.metrics import (
    backbone_secondary_structure,
    binding_site_adherence,
    binding_site_adherence_curve,
    secondary_structure_agreement,
    secondary_structure_fractions,
    structure_tm_scores,
)


def test_tm_rigid_motion() -> None:
    # A nondegenerate protein-like helix with one C-alpha position per residue.
    angle = np.arange(40) * (2 * np.pi / 3.6)
    coordinates = np.column_stack((2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * np.arange(40)))
    moved = coordinates @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + [5, 7, -2]
    rmsd_score, homology_score = structure_tm_scores(coordinates, moved)
    assert rmsd_score == pytest.approx(1.0, abs=1e-4)
    assert homology_score == pytest.approx(1.0, abs=1e-4)


def test_cross_chain_beta_contacts(secondary_backbones) -> None:
    backbone = secondary_backbones["beta"]
    edges = np.array([[0, 2], [1, 3]])
    np.testing.assert_array_equal(backbone_secondary_structure(backbone, edges, np.ones(4, bool)), list("LELE"))
    np.testing.assert_array_equal(
        backbone_secondary_structure(backbone, edges, np.array([True, False, True, True])), list("LLLL")
    )


def test_helix_uses_peptide_edges(secondary_backbones) -> None:
    backbone = secondary_backbones["helix"]
    edges = np.stack((np.arange(6), np.arange(1, 7)))
    np.testing.assert_array_equal(backbone_secondary_structure(backbone, edges, np.ones(7, bool)), list("LHHHHLL"))
    np.testing.assert_array_equal(
        backbone_secondary_structure(backbone, edges[:, [0, 1, 2, 4, 5]], np.ones(7, bool)), list("LLLLLLL")
    )


@pytest.mark.parametrize("circular", [False, True])
def test_storage_permutation(circular, secondary_backbones) -> None:
    backbone = secondary_backbones["helix"]
    edges = np.stack((np.arange(6), np.arange(1, 7)))
    if circular:
        edges = np.concatenate((edges, [[6], [0]]), axis=1)
    labels = backbone_secondary_structure(backbone, edges, np.ones(7, bool))
    permutation = np.array([6, 2, 0, 4, 1, 5, 3])
    inverse = np.argsort(permutation)
    np.testing.assert_array_equal(
        backbone_secondary_structure(backbone[permutation], inverse[edges], np.ones(7, bool)), labels[permutation]
    )


def test_secondary_structure_empty_observations() -> None:
    np.testing.assert_array_equal(
        backbone_secondary_structure(np.empty((0, 4, 3)), np.empty((2, 0), dtype=int), np.empty(0, dtype=bool)),
        np.empty(0, dtype="U1"),
    )


def test_secondary_structure_fractions_uses_values_and_empty_scopes() -> None:
    assert secondary_structure_fractions({("A", 0): "L", ("B", 0): "H", ("B", 1): "H", ("B", 2): "E"}) == {
        "loop": 0.25,
        "helix": 0.5,
        "sheet": 0.25,
    }
    assert secondary_structure_fractions(("L", "H", "H", "E")) == {"loop": 0.25, "helix": 0.5, "sheet": 0.25}
    assert all(np.isnan(value) for value in secondary_structure_fractions(()).values())


@pytest.mark.parametrize("conditioned", [("L", 2, 3, "E", None), (1, 2, 3, 3, 0)])
def test_secondary_structure_agreement_scopes_jaccard_and_accuracy(conditioned) -> None:
    observed = secondary_structure_agreement(("L", "H", "H", "E", "H"), conditioned)
    assert observed == {
        "precision_loop": 1.0,
        "recall_loop": 1.0,
        "jaccard_loop": 1.0,
        "precision_helix": 0.5,
        "recall_helix": 1.0,
        "jaccard_helix": 0.5,
        "precision_sheet": 1.0,
        "recall_sheet": 0.5,
        "jaccard_sheet": 0.5,
        "accuracy_overall": 0.75,
    }
    empty = secondary_structure_agreement(("L", "H"), (None, None))
    assert all(np.isnan(value) for value in empty.values())


def test_binding_site_adherence_uses_nearest_centers_and_strict_thresholds() -> None:
    binding_sites = np.array([[0.0, 0, 0], [3.0, 4.0, 0], [10.0, 0, 0]])
    design = np.array([[0.0, 0, 0]])
    observed = binding_site_adherence_curve(binding_sites, design, thresholds=(0.0, 5.0, 5.1, 10.0, 11.0))
    assert observed == {0.0: 0.0, 5.0: 1 / 3, 5.1: 2 / 3, 10.0: 2 / 3, 11.0: 1.0}
    assert binding_site_adherence(binding_sites, design, distance_threshold=5.0) == 1 / 3
    assert binding_site_adherence(binding_sites, np.empty((0, 3))) == 0.0
    assert np.isnan(binding_site_adherence(np.empty((0, 3)), design))
