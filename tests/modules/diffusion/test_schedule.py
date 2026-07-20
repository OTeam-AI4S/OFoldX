from __future__ import annotations

import pytest
import torch

from ofoldx.modules.diffusion import EdmScheduler


@pytest.fixture
def scheduler() -> EdmScheduler:
    return EdmScheduler(max_noise_level=10.0, min_noise_level=0.1, rho=2.0, sigma_data=16.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_schedule(scheduler: EdmScheduler, dtype: torch.dtype) -> None:
    schedule = scheduler(num_sampling_steps=4, dtype=dtype)

    assert schedule.shape == (5,)
    assert schedule.dtype == dtype
    # rho=2 closed form: sigma_data * (max_root + i/N*(min_root - max_root))**2, last clamped to 0.
    max_root, min_root = 10.0**0.5, 0.1**0.5
    expected = [16.0 * (max_root + i * 0.25 * (min_root - max_root)) ** 2 for i in range(5)]
    expected[-1] = 0.0
    torch.testing.assert_close(schedule, schedule.new_tensor(expected))
    # first level is sigma_data * max_noise_level, terminal level is exactly zero, strictly decreasing.
    assert schedule[0].item() == pytest.approx(160.0)
    assert schedule[-1].item() == 0.0
    assert torch.all(schedule[:-2] > schedule[1:-1])


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_sampling_steps"):
        EdmScheduler()(num_sampling_steps=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="max_noise_level"):
        EdmScheduler()(dtype=torch.float32, max_noise_level=0.0)


def test_call_can_override_the_initial_noise_level() -> None:
    schedule = EdmScheduler(max_noise_level=10.0, sigma_data=16.0)(
        num_sampling_steps=1,
        dtype=torch.float32,
        max_noise_level=2.0,
    )

    assert schedule.tolist() == pytest.approx([32.0, 0.0])
