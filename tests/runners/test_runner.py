from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from ofoldx.models import BaseInput
from ofoldx.runners import Runner


@dataclass(slots=True, kw_only=True)
class CustomInput(BaseInput):
    r"""
    Minimal external model input used to verify runner collation.
    """

    value: torch.Tensor


def test_runner_collate_preserves_one_custom_base_input() -> None:
    batch = CustomInput(value=torch.ones(3))

    assert Runner.collate_fn([batch]) is batch


def test_runner_collate_rejects_multiple_custom_base_inputs() -> None:
    with pytest.raises(ValueError, match="already batched"):
        Runner.collate_fn([CustomInput(value=torch.ones(3)), CustomInput(value=torch.ones(3))])


def test_runner_collate_delegates_plain_tensors() -> None:
    output = Runner.collate_fn([torch.ones(2), torch.zeros(2)])

    torch.testing.assert_close(output, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
