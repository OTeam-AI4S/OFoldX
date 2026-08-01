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

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.embedding import RelativePositionEmbedder, generate_relative_position_features


def _features(
    residue_index,
    *,
    chain_index=None,
    entity_index=None,
    symmetry_index=None,
    cyclic_period=None,
    chain_gate="same_entity",
    dtype=torch.float32,
):
    r"""
    Relative-position buckets from a ``(1, N)`` residue index, defaulting metadata to a single chain.
    """
    zeros = torch.zeros_like(residue_index)
    return generate_relative_position_features(
        residue_index=residue_index,
        chain_index=zeros if chain_index is None else chain_index,
        entity_index=zeros if entity_index is None else entity_index,
        symmetry_index=zeros if symmetry_index is None else symmetry_index,
        token_index=residue_index,
        cyclic_period=cyclic_period,
        chain_gate=chain_gate,
        dtype=dtype,
    )


class TestGenerateRelativePositionFeatures:
    def test_metadata_buckets(self) -> None:
        idx = torch.tensor([[0, 1, 0, 1]])
        chain = torch.tensor([[0, 0, 1, 1]])
        buckets = _features(idx, chain_index=chain, entity_index=chain, symmetry_index=chain)

        assert torch.equal(
            buckets.residue_bucket,
            torch.tensor([[[32, 31, 65, 65], [33, 32, 65, 65], [65, 65, 32, 31], [65, 65, 33, 32]]]),
        )
        assert torch.equal(
            buckets.token_bucket,
            torch.tensor([[[32, 65, 65, 65], [65, 32, 65, 65], [65, 65, 32, 65], [65, 65, 65, 32]]]),
        )
        assert torch.equal(
            buckets.chain_bucket,
            torch.tensor([[[2, 2, 5, 5], [2, 2, 5, 5], [5, 5, 2, 2], [5, 5, 2, 2]]]),
        )
        assert torch.equal(buckets.same_entity, chain[..., :, None] == chain[..., None, :])

    @pytest.mark.parametrize("period", [None, 0.0, 5.0])
    def test_cyclic_period(self, period: float | None) -> None:
        idx = torch.tensor([[0, 5]])
        cyclic = None if period is None else torch.full((1, 2), period)

        buckets = _features(idx, cyclic_period=cyclic)

        expected = torch.tensor([[[32, 32], [32, 32]]]) if period else torch.tensor([[[32, 27], [37, 32]]])
        assert torch.equal(buckets.residue_bucket, expected)

    def test_token_periods(self) -> None:
        buckets = _features(torch.tensor([[0, 5]]), cyclic_period=torch.tensor([[0.0, 5.0]]))

        assert torch.equal(buckets.residue_bucket, torch.tensor([[[32, 32], [37, 32]]]))

    def test_chain_gate(self) -> None:
        buckets = _features(
            torch.tensor([[0, 1, 0, 0]]),
            chain_index=torch.tensor([[0, 0, 1, 2]]),
            entity_index=torch.tensor([[0, 0, 0, 1]]),
            symmetry_index=torch.tensor([[0, 0, 1, 2]]),
            chain_gate="not_same_chain",
        )

        assert torch.equal(
            buckets.chain_bucket,
            torch.tensor([[[5, 5, 1, 0], [5, 5, 1, 0], [3, 3, 5, 1], [4, 4, 3, 5]]]),
        )


class TestRelativePositionEmbedder:
    def test_projection(self) -> None:
        torch.manual_seed(0)
        embedder = RelativePositionEmbedder(pair_size=8)
        buckets = _features(torch.tensor([[0, 1, 2]]))

        output = embedder(buckets)
        residue_size, chain_size = buckets.residue_size, buckets.chain_size
        dense = torch.cat(
            [
                F.one_hot(buckets.residue_bucket, residue_size).float(),
                F.one_hot(buckets.token_bucket, residue_size).float(),
                buckets.same_entity.unsqueeze(-1).float(),
                F.one_hot(buckets.chain_bucket, chain_size).float(),
            ],
            dim=-1,
        )

        torch.testing.assert_close(output, embedder.output_projection(dense), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_dtype(self, dtype: torch.dtype) -> None:
        embedder = RelativePositionEmbedder(pair_size=8)

        assert embedder(_features(torch.arange(3).unsqueeze(0), dtype=dtype)).dtype == dtype
