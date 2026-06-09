from __future__ import annotations

import pytest
import torch

from ofoldx.modules.transition import RowwiseDropout, Transition


def test_rowwise_dropout_eval_and_zero_probability_are_identity() -> None:
    hidden = torch.arange(1, 2 * 3 * 4 * 5 + 1, dtype=torch.float32).reshape(2, 3, 4, 5)

    torch.testing.assert_close(RowwiseDropout(0.5).eval()(hidden), hidden)
    torch.testing.assert_close(RowwiseDropout(0.0).train()(hidden), hidden)


def test_rowwise_dropout_shares_mask_across_rows() -> None:
    hidden = torch.arange(1, 2 * 3 * 4 * 5 + 1, dtype=torch.float32).reshape(2, 3, 4, 5)

    torch.manual_seed(0)
    output = RowwiseDropout(0.5).train()(hidden)

    ratio = output / hidden
    torch.testing.assert_close(ratio[:, 0], ratio[:, 1])
    torch.testing.assert_close(ratio[:, 1], ratio[:, 2])
    assert bool(((ratio == 0.0) | torch.isclose(ratio, torch.tensor(2.0))).all())
    assert bool((ratio == 0.0).any()) and bool(torch.isclose(ratio, torch.tensor(2.0)).any())


def test_transition_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        Transition(hidden_size=8, operator_backend="bogus")
