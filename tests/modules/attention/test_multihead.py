from __future__ import annotations

import pytest
import torch

from ofoldx.modules.attention import GatedMultiheadAttention, MultiheadAttentionProjection, ProjectedAttention


class TestProjectedAttention:
    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_mask_layouts(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(2, 1, 2, 1)
        key = torch.zeros(2, 1, 3, 1)
        value = torch.tensor(
            [
                [[[-1.0], [2.0], [3.0]]],
                [[[-10.0], [20.0], [30.0]]],
            ]
        )

        pair_output = attention(
            query,
            key,
            value,
            attention_mask=torch.tensor([[True, False, False], [False, True, False]]),
        )
        key_output = attention(
            query,
            key,
            value,
            attention_mask=torch.tensor(
                [
                    [[True, False, False]],
                    [[False, False, True]],
                ]
            ),
        )

        torch.testing.assert_close(
            pair_output,
            torch.tensor(
                [
                    [[[-1.0], [2.0]]],
                    [[[-10.0], [20.0]]],
                ]
            ),
        )
        torch.testing.assert_close(
            key_output,
            torch.tensor(
                [
                    [[[-1.0], [-1.0]]],
                    [[[30.0], [30.0]]],
                ]
            ),
        )

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_empty_row(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(1, 1, 2, 1)
        key = torch.zeros(1, 1, 3, 1)
        value = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        mask = torch.tensor([[False, False, False], [True, False, False]])

        output = attention(query, key, value, attention_mask=mask)

        torch.testing.assert_close(output[..., 0, :], torch.zeros_like(output[..., 0, :]))
        torch.testing.assert_close(output[..., 1, :], torch.ones_like(output[..., 1, :]))

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_head_mask(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        query = torch.zeros(1, 2, 3, 1)
        key = torch.zeros(1, 2, 2, 1)
        value = torch.tensor([[[[1.0], [2.0]], [[10.0], [20.0]]]])
        mask = torch.tensor([[[True, False], [False, True]]])

        output = attention(query, key, value, attention_mask=mask)

        expected = torch.tensor([[[[1.0], [1.0], [1.0]], [[20.0], [20.0], [20.0]]]])
        torch.testing.assert_close(output, expected)

    @pytest.mark.parametrize("backend", ["eager", "sdpa"])
    def test_mask_precedes_bias(self, backend: str) -> None:
        attention = ProjectedAttention(backend=backend)
        output = attention(
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 2, 1),
            torch.tensor([[[[1.0], [9.0]]]]),
            biases=(torch.tensor([[[[0.0, 2_000_000.0]]]]),),
            attention_mask=torch.tensor([[True, False]]),
        )

        torch.testing.assert_close(output, torch.ones_like(output))


class TestMultiheadAttentionProjection:
    def test_invalid_heads(self) -> None:
        with pytest.raises(ValueError):
            MultiheadAttentionProjection(hidden_size=8, num_attention_heads=3)

    def test_biases(self) -> None:
        projection = MultiheadAttentionProjection(
            hidden_size=8,
            num_attention_heads=2,
            query_bias=False,
            output_bias=True,
            gate_bias=True,
            gate_bias_init=1.0,
        )

        assert projection.query_projection.bias is None
        assert projection.key_projection.bias is None
        assert projection.value_projection.bias is None
        assert projection.output_projection.bias is not None
        assert projection.gate_projection.bias is not None
        torch.testing.assert_close(projection.gate_projection.bias, torch.ones(8))

    def test_head_dim(self) -> None:
        projection = MultiheadAttentionProjection(hidden_size=8, num_attention_heads=3, head_dim=4)
        hidden = torch.randn(1, 3, 8)

        query, key, value = projection.project(hidden, hidden, hidden)
        output = projection.merge(torch.randn(1, 3, 3, 4), hidden)

        assert query.shape == key.shape == value.shape == (1, 3, 3, 4)
        assert output.shape == hidden.shape


class TestGatedMultiheadAttention:
    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError):
            GatedMultiheadAttention(hidden_size=8, num_attention_heads=2, backend="bogus")
