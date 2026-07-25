from __future__ import annotations

import pytest
import torch

from ofoldx.modules.attention import ProjectedAttention


class TestProjectedAttention:
    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_pair_first_masks_when_batch_equals_query_length(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(2, 1, 2, 1)
        key = torch.zeros(2, 1, 3, 1)
        value = torch.tensor(
            [
                [[[-1.0], [2.0], [3.0]]],
                [[[-10.0], [20.0], [30.0]]],
            ]
        )

        pair_mask = torch.tensor([[True, False, False], [False, True, False]])
        pair_output = attention(query, key, value, attention_mask=pair_mask)
        expected_pair_output = torch.tensor(
            [
                [[[-1.0], [2.0]]],
                [[[-10.0], [20.0]]],
            ]
        )
        torch.testing.assert_close(pair_output, expected_pair_output)

        batched_key_mask = torch.tensor(
            [
                [[True, False, False]],
                [[False, False, True]],
            ]
        )
        key_output = attention(query, key, value, attention_mask=batched_key_mask)
        expected_key_output = torch.tensor(
            [
                [[[-1.0], [-1.0]]],
                [[[30.0], [30.0]]],
            ]
        )
        torch.testing.assert_close(key_output, expected_key_output)

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_fully_masked_raw_attention_row_is_zero(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(1, 1, 2, 1)
        key = torch.zeros(1, 1, 3, 1)
        value = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        mask = torch.tensor([[False, False, False], [True, False, False]])

        output = attention(query, key, value, attention_mask=mask)

        torch.testing.assert_close(output[..., 0, :], torch.zeros_like(output[..., 0, :]))
        torch.testing.assert_close(output[..., 1, :], torch.ones_like(output[..., 1, :]))

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_head_specific_key_mask(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(1, 2, 3, 1)
        key = torch.zeros(1, 2, 2, 1)
        value = torch.tensor([[[[1.0], [2.0]], [[10.0], [20.0]]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        output = attention(query, key, value, attention_mask=mask)

        expected = torch.tensor([[[[1.0], [1.0], [1.0]], [[20.0], [20.0], [20.0]]]])
        torch.testing.assert_close(output, expected)
