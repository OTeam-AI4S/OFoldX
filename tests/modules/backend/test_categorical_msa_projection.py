from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.backend import categorical_msa_projection


class TestCategoricalMsaProjection:
    def test_equation(self) -> None:
        tokens = torch.tensor([[[0, 2, 4], [3, 1, 0], [4, 3, 2]]])
        continuous = torch.randn(1, 3, 3, 2, requires_grad=True)
        weight = torch.randn(4, 7, requires_grad=True)
        expected_continuous = continuous.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()
        features = torch.cat((F.one_hot(tokens, num_classes=5), expected_continuous), dim=-1)

        expected = F.linear(features, expected_weight)
        actual = categorical_msa_projection(tokens, continuous, weight, backend="eager")
        expected.square().sum().backward()
        actual.square().sum().backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(continuous.grad, expected_continuous.grad)
        torch.testing.assert_close(weight.grad, expected_weight.grad)

    def test_shape(self) -> None:
        with pytest.raises(ValueError, match="continuous_features must have shape"):
            categorical_msa_projection(
                torch.zeros(2, 3, dtype=torch.long),
                torch.zeros(1, 2, 3, 2),
                torch.zeros(4, 7),
                backend="eager",
            )
