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

"""RFdiffusion raw template geometry carries an explicit observation mask."""

import numpy as np
import pytest
import torch

from ofoldx.data.tokenization import TOKENS
from ofoldx.models.rfdiffusion.model import (
    RfDiffusionIgso3Diffuser,
    RfDiffusionSamplerStep,
    RfDiffusionTemplateEmbedder,
    _initial_coordinates,
)


def _rotation_density(angles, sigma):
    degrees = torch.arange(64, dtype=torch.float64)
    weights = (2 * degrees + 1) * torch.exp(-degrees * (degrees + 1) * sigma**2 / 2)
    frequencies = degrees[1:]
    tails = weights.flip(0).cumsum(0).flip(0)[1:]
    phase = angles[..., None] * frequencies
    density = weights.sum() + 2 * (tails * phase.cos()).sum(-1)
    derivative = -2 * (tails * frequencies * phase.sin()).sum(-1)
    return density, derivative / density


@pytest.fixture(scope="class", params=(0.9, 2.0))
def rotation_diffuser(request):
    sigma = float(torch.tensor(request.param, dtype=torch.float32))
    angles = torch.linspace(0, torch.pi, 1001, dtype=torch.float64)[1:]
    density, scores = _rotation_density(angles, sigma)
    marginal = density * (1 - angles.cos()) / torch.pi
    degree = torch.arange(64, dtype=torch.float64)
    mass = (2 * degree + 1).square() * torch.exp(-degree * (degree + 1) * sigma**2 / 2)
    slope = -(mass * degree * (degree + 1)).sum() / (3 * mass.sum())
    diffuser = RfDiffusionIgso3Diffuser(
        num_timesteps=50,
        min_sigma=0.02,
        max_sigma=2 * sigma - 0.02,
        min_b=2 * (sigma - 0.02),
        max_b=2 * (sigma - 0.02),
        igso3_schedule={
            "discrete_sigma": torch.tensor([sigma], dtype=torch.float64),
            "discrete_omega": angles,
            "score_norm": scores[None],
            "cdf": (marginal.cumsum(0) * torch.pi / 1000)[None],
            "exp_score_norms": ((scores.square() * marginal).sum() / marginal.sum()).sqrt()[None],
        },
        backend="eager",
    )
    return diffuser, sigma, slope


class TestRotationScore:
    def test_values_and_gradients(self, rotation_diffuser):
        diffuser, sigma, slope = rotation_diffuser
        first = torch.pi / 1000
        angles = torch.tensor([0, 1e-12, 1e-10, first / 4, first * 0.99, first * 1.01, 0.4], dtype=torch.float64)
        angles.requires_grad_()
        actual = diffuser.score_norm(25, angles)
        _, expected = _rotation_density(angles, sigma)
        # The released 1000-point grid's first-cell secant has <=1.3e-6 relative error.
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-15)
        gradient = torch.autograd.grad(actual[:4].sum(), angles)[0]
        torch.testing.assert_close(gradient[:4], slope.expand(4), rtol=2e-6, atol=1e-12)

    def test_reverse_limit(self, rotation_diffuser):
        diffuser, _sigma, slope = rotation_diffuser
        increments = torch.tensor([1e-12, 1e-10, torch.pi / 4000], dtype=torch.float64)
        angles = torch.cat((torch.zeros(1), torch.stack((increments, -increments), -1).flatten()))
        rotations = torch.eye(3, dtype=torch.float64).repeat(len(angles), 1, 1)
        rotations[:, 1, 1] = rotations[:, 2, 2] = angles.cos()
        rotations[:, 1, 2], rotations[:, 2, 1] = -angles.sin(), angles.sin()
        actual = diffuser.reverse_sample_vectorized(
            rotations, torch.eye(3).expand_as(rotations), 25, noise_level=0, return_perturb=True
        )
        tangent = torch.atan2(actual[:, 2, 1], actual[:, 1, 1]).double()
        assert tangent[0] == 0
        finite_difference = (tangent[1::2] - tangent[2::2]) / (2 * increments)
        sigma = 0.02 + 0.5 * diffuser.min_b
        coefficient_squared = 2 * sigma * diffuser.min_b
        expected = slope * coefficient_squared / 50
        # Finite differences independently check the vector slope.
        torch.testing.assert_close(finite_difference, expected.expand(3), rtol=3e-6, atol=1e-10)


def test_initial_coordinates_half_centroid():
    coordinates = torch.full((1, 2, 4, 27, 3), torch.nan, dtype=torch.float16)
    for residue, x in ((0, 40000), (1, 40032), (3, 40064)):
        coordinates[0, 0, residue, :3] = torch.tensor([[x - 32, 32, 0], [x, 0, 0], [x + 32, 0, 0]], dtype=torch.float16)
    ideal = torch.full((27, 3), torch.nan, dtype=torch.float16)
    ideal[:5] = torch.tensor(
        [[-0.5272, 1.3593, 0], [0, 0, 0], [1.5233, 0, 0], [2.0505, 1.3593, 0], [-0.5291, -0.7749, -1.2056]]
    )
    expected = coordinates.double()
    observed = torch.tensor([0, 1, 3])
    center = expected[0, 0, observed, 1].mean(0)
    expected[0, 0] -= center
    # The missing row is tied between known rows 1 and 3; nearest-known uses
    # the first row, whose centered CA is zero. The second template is empty.
    expected[0, 0, 2] = ideal.double()
    expected[0, 1] = ideal.double()
    actual = _initial_coordinates(coordinates)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected.half(), equal_nan=True)
    assert torch.isfinite(actual[0, 0, :, :3]).all()


def test_unavailable_template_gradients():
    model = RfDiffusionTemplateEmbedder(
        pair_size=8,
        state_size=8,
        template_size=8,
        num_blocks=1,
        num_attention_heads=2,
        hidden_size=4,
        dropout=0,
    )
    coordinates = torch.randn(1, 1, 2, 27, 3)
    mask = torch.ones(1, 1, 2, 27, dtype=torch.bool)
    mask[:, :, 1] = False
    coordinates[:, :, 1] = float("nan")
    coordinates.requires_grad_()
    pair, state = model(
        torch.randn(1, 1, 2, len(TOKENS) + 2),  # the canonical residue plane and upstream's two extra channels
        torch.randn(1, 1, 2, 2, 44),
        torch.zeros(1, 1, 2, 30),
        coordinates,
        torch.randn(1, 2, 2, 8),
        torch.randn(1, 2, 8),
        coordinate_mask=mask,
    )
    assert torch.isfinite(pair).all() and torch.isfinite(state).all()
    gradient = torch.autograd.grad(pair.square().sum() + state.square().sum(), coordinates)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[:, :, 1], torch.zeros(1, 1, 27, 3))


def _alignment_reference(predicted, final, motif):
    points = predicted[motif, :3].reshape(-1, 3)
    targets = final[motif, :3].reshape(-1, 3)
    center = points.mean(0)
    target_center = targets.mean(0)
    left, _, right = np.linalg.svd((points - center).T @ (targets - target_center))
    orientation = np.eye(3)
    orientation[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ orientation @ right
    return (predicted - center) @ rotation + target_center


@pytest.fixture
def alignment_coordinates():
    generator = torch.Generator().manual_seed(413)
    predicted = torch.randn(3, 14, 3, dtype=torch.float64, generator=generator)
    rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    translation = torch.tensor([1 + 2**-30, -0.25 + 2**-29, 1.75], dtype=torch.float64)
    final = predicted @ rotation.T + translation
    motif = torch.tensor([True, True, False])
    return predicted, final, motif


class TestAlignment:
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_coordinates(self, alignment_coordinates, dtype):
        predicted, final, motif = alignment_coordinates
        predicted = predicted.to(dtype)
        predicted[2, 13] = torch.nan
        actual = RfDiffusionSamplerStep.align_prediction_to_noisy_motif(predicted, final, motif, backend="eager")
        reference = _alignment_reference(predicted.double().numpy(), final.numpy(), motif.numpy())
        expected = torch.from_numpy(reference).to(dtype)
        tolerance = {torch.float64: 2e-13, torch.float32: 2e-6, torch.float16: 0.004, torch.bfloat16: 0.03}[dtype]
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
        torch.testing.assert_close(torch.isnan(actual), torch.isnan(predicted))

    def test_gradients(self, alignment_coordinates):
        predicted, final, motif = alignment_coordinates
        final = final + torch.linspace(-0.08, 0.11, final.numel()).reshape_as(final)
        predicted.requires_grad_()
        final.requires_grad_()
        cotangent = torch.linspace(-0.9, 0.7, predicted.numel(), dtype=torch.float64).reshape_as(predicted)
        actual = RfDiffusionSamplerStep.align_prediction_to_noisy_motif(predicted, final, motif, backend="eager")
        gradients = torch.autograd.grad((actual * cotangent).sum(), (predicted, final))
        source = predicted.detach().numpy()
        target = final.detach().numpy()
        selection = motif.numpy()
        for index, gradient in enumerate(gradients):
            direction = np.cos(np.arange(predicted.numel()).reshape(predicted.shape) * (0.19 + index * 0.13))
            step = 1e-6
            operands = [source.copy(), target.copy()]
            operands[index] += direction * step
            plus = _alignment_reference(*operands, selection)
            operands[index] -= 2 * direction * step
            minus = _alignment_reference(*operands, selection)
            expected = np.sum((plus - minus) * cotangent.numpy()) / (2 * step)
            observed = np.sum(gradient.detach().numpy() * direction)
            assert observed == pytest.approx(expected, rel=2e-7, abs=2e-7)
