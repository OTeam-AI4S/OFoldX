from __future__ import annotations

import torch

from ofoldx.modules.contact import ContactConditioning


class TestContactConditioning:
    def test_bias(self) -> None:
        conditioning = ContactConditioning(pair_size=2, cutoff_min=2.0, cutoff_max=6.0)
        with torch.no_grad():
            conditioning.unspecified_bias.copy_(torch.tensor([1.0, 2.0]))
            conditioning.unselected_bias.copy_(torch.tensor([3.0, 4.0]))
            conditioning.threshold_fourier.projection.weight.zero_()
            conditioning.threshold_fourier.projection.bias.zero_()
            conditioning.constraint_encoder.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0, 0.0, 3.0, 5.0, 0.0],
                        [1.5, 0.0, 0.0, -2.0, 0.0, 0.5],
                    ]
                )
            )
            conditioning.constraint_encoder.bias.copy_(torch.tensor([7.0, 2.0]))

        contact_class = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        actual = conditioning(contact_class, torch.tensor([2.0, 6.0, 4.0]))

        torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0], [3.0, 4.0], [15.5, 3.0]]))
