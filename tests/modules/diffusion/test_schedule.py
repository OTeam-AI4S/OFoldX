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

from ofoldx.modules.diffusion import EdmScheduler


class TestEdmScheduler:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_schedule(self, dtype: torch.dtype) -> None:
        scheduler = EdmScheduler(max_noise_level=10.0, min_noise_level=0.1, rho=2.0, sigma_data=16.0)

        schedule = scheduler(num_sampling_steps=4, dtype=dtype)

        max_root, min_root = 10.0**0.5, 0.1**0.5
        expected = [16.0 * (max_root + index * 0.25 * (min_root - max_root)) ** 2 for index in range(5)]
        expected[-1] = 0.0
        assert schedule.shape == (5,)
        assert schedule.dtype == dtype
        torch.testing.assert_close(schedule, schedule.new_tensor(expected))
        assert torch.all(schedule[:-2] > schedule[1:-1])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"num_sampling_steps": 0}, "num_sampling_steps"),
            ({"max_noise_level": 0.0}, "max_noise_level"),
        ],
    )
    def test_invalid_input(self, kwargs: dict[str, int | float], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            EdmScheduler()(**kwargs, dtype=torch.float32)

    def test_maximum_override(self) -> None:
        schedule = EdmScheduler(max_noise_level=10.0, sigma_data=16.0)(
            num_sampling_steps=1,
            dtype=torch.float32,
            max_noise_level=2.0,
        )

        assert schedule.tolist() == pytest.approx([32.0, 0.0])

    def test_dynamic_compile(self) -> None:
        scheduler = EdmScheduler(max_noise_level=10.0, min_noise_level=0.1, rho=2.0, sigma_data=16.0)
        compiled = torch.compile(scheduler, backend="eager", fullgraph=True, dynamic=True)

        for steps, maximum in ((2, 2.0), (3, 3.0)):
            expected = scheduler(steps, dtype=torch.float32, max_noise_level=maximum)
            actual = compiled(steps, dtype=torch.float32, max_noise_level=maximum)
            torch.testing.assert_close(actual, expected)
