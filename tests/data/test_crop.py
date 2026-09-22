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

"""Behaviour of ofoldx.data.crop: crops keep whole component groups and honour required and anchor tokens."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from ofoldx.data.crop import (
    continuous_crop,
    crop_token_layout,
    instance_anchored_crop,
    interface_anchored_crop,
    interface_crop,
    resolved_token_mask,
    spatial_crop,
)
from ofoldx.data.tokenization import tokenize

POLICIES = [continuous_crop, spatial_crop, interface_crop]


@pytest.fixture
def layout(structure):
    """ALA and GLY one token each, LIG and HOH one token per atom (9 tokens; HA in ALA's token)."""
    return tokenize(
        structure.system,
        residue_token=np.array([True, True, False, False]),
        centre_atom=np.array([1, 7, -1, -1]),
    )


def groups(layout, keep):
    return sorted(set(layout.token_component[keep].tolist()))


class TestCrop:
    @pytest.mark.parametrize("policy", POLICIES)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_whole_groups_within_limit(self, policy, seed, structure, layout):
        keep = policy(structure, layout, max_tokens=7, generator=np.random.default_rng(seed))
        assert keep.sum() <= 7
        for component in groups(layout, keep):
            assert keep[layout.token_component == component].all()

    @pytest.mark.parametrize("policy", POLICIES)
    def test_required_groups_are_kept(self, policy, structure, layout):
        required = np.zeros(len(layout.token_component), dtype=bool)
        required[8] = True
        keep = policy(structure, layout, max_tokens=3, required=required, generator=np.random.default_rng(0))
        assert keep[8]

    def test_unfitting_required_is_invalid(self, structure, layout, invalid):
        required = np.zeros(len(layout.token_component), dtype=bool)
        required[2] = True
        invalid(continuous_crop, structure, layout, max_tokens=3, required=required, generator=np.random.default_rng(0))

    @pytest.mark.parametrize("required", [np.array([2]), np.ones(8, dtype=bool)], ids=["indices", "short"])
    def test_required_must_be_mask(self, structure, layout, invalid, required):  # never read as token indices
        generator = np.random.default_rng(0)
        invalid(continuous_crop, structure, layout, max_tokens=7, required=required, generator=generator)

    def test_crop_selects_the_layout(self, structure, layout):
        cropped = crop_token_layout(
            structure, layout, partial(spatial_crop, max_tokens=4), generator=np.random.default_rng(3)
        )
        assert len(cropped.token_component) <= 4 and cropped.token_atom_ptr[-1] == len(cropped.token_atoms)

    def test_instance_anchor(self, structure, layout):
        cropper = partial(continuous_crop, max_tokens=2)
        keep = instance_anchored_crop(
            structure, layout, cropper=cropper, instances=["W"], generator=np.random.default_rng(0)
        )
        assert keep[8]

    def test_interface_anchor_keeps_contact(self, structure, layout):
        cropper = partial(spatial_crop, max_tokens=4)
        keep = interface_anchored_crop(
            structure, layout, cropper=cropper, instances=["A", "W"], generator=np.random.default_rng(0)
        )
        assert keep[8] and keep[:2].any()

    def test_self_interface_is_invalid(self, structure, layout, invalid):
        cropper = partial(spatial_crop, max_tokens=4)
        generator = np.random.default_rng(0)
        invalid(interface_anchored_crop, structure, layout, cropper=cropper, instances=["A", "A"], generator=generator)


class TestResolvedTokenMask:
    def test_unplaced_token_is_unresolved(self, structure, system):
        layout = tokenize(system, residue_token=np.zeros(4, dtype=bool), centre_atom=np.full(4, -1))
        resolved = resolved_token_mask(structure, layout)
        assert not resolved[5] and resolved.sum() == len(resolved) - 1
