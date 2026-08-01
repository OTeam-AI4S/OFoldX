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

from ofoldx.errors import InvalidRequestError
from ofoldx.modules.backend import pair_norm_projection
from tests.modules import require_oops


def _centered_projection_reference(
    pair: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    projection_weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    normalization_dtype: torch.dtype | None = None,
    projection_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    with torch.autocast(device_type=pair.device.type, enabled=False):
        values = pair.double()
        centered = values - values.mean(dim=-1, keepdim=True)
        normalized = centered / (centered.square().mean(dim=-1, keepdim=True) + eps).sqrt()
        normalized = normalized * norm_weight.double()
        if norm_bias is not None:
            normalized = normalized + norm_bias.double()
        normalized = normalized.to(pair.dtype if normalization_dtype is None else normalization_dtype)
        if projection_dtype is not None:
            normalized = normalized.to(projection_dtype)
            projection_weight = projection_weight.to(projection_dtype)
        return torch.einsum("...c,hc->...h", normalized, projection_weight)


def _assert_precision_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    precision: torch.dtype,
    absolute_error_bound: torch.Tensor | None = None,
) -> None:
    tolerance = {
        torch.float16: (5e-3, 3e-3),
        torch.bfloat16: (3e-2, 2e-2),
        torch.float32: (2e-5, 2e-5),
        torch.float64: (2e-10, 2e-10),
    }[precision]
    assert torch.isfinite(actual).all()
    if absolute_error_bound is None:
        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    else:
        difference = (actual.double() - expected.double()).abs()
        bound = absolute_error_bound + tolerance[0] * expected.double().abs() + tolerance[1]
        assert torch.all(difference <= bound), (difference / bound).amax().item()


def _normalization_conditioning_bounds(
    pair: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    has_bias: bool,
    output_layout: str,
) -> tuple[torch.Tensor, ...]:
    values = pair.detach().double()
    mean = values.mean(dim=-1, keepdim=True)
    centered = values - mean
    rstd = (centered.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
    # Native reduced-precision CPU LayerNorm centers its variance, but its
    # second pass and affine VJP evaluate x*rstd - mean*rstd in FP32. Their
    # absolute rounding error scales with those large terms before cancellation.
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    normalized_error = unit_roundoff * (values.abs() + mean.abs()) * rstd
    affine_error = normalized_error * norm_weight.detach().double().abs()
    weight = projection_weight.detach().double().abs()
    output_error = torch.einsum("...c,hc->...h", affine_error, weight)
    if output_layout == "head_major":
        cotangent = cotangent.movedim(-3, -1)
        output_error = output_error.movedim(-1, -3)
    cotangent = cotangent.detach().double().abs()
    normalization_adjoint_bound = torch.einsum("...h,hc->...c", cotangent, weight)
    norm_weight_error = (normalization_adjoint_bound * normalized_error).sum(dim=tuple(range(pair.ndim - 1)))
    projection_error = torch.einsum("...h,...c->hc", cotangent, affine_error)
    # Input and shift VJPs passed their ordinary precision bounds; only the
    # actual expanded-affine terms receive this conditioning-derived allowance.
    bounds = (output_error, torch.zeros_like(values), norm_weight_error)
    if has_bias:
        bounds = (*bounds, torch.zeros_like(norm_weight_error))
    return (*bounds, projection_error)


def _assert_output_and_vjps(
    actual: torch.Tensor,
    expected: torch.Tensor,
    operands: tuple[torch.Tensor, ...],
    precision: torch.dtype,
    *,
    cotangent: torch.Tensor | None = None,
    absolute_error_bounds: tuple[torch.Tensor, ...] | None = None,
) -> None:
    _assert_precision_close(
        actual, expected, precision, None if absolute_error_bounds is None else absolute_error_bounds[0]
    )
    if cotangent is None:
        cotangent = torch.linspace(-0.8, 1.1, actual.numel(), device=actual.device, dtype=actual.dtype).reshape_as(
            actual
        )
    actual_gradients = torch.autograd.grad(actual, operands, grad_outputs=cotangent)
    expected_gradients = torch.autograd.grad(expected, operands, grad_outputs=cotangent)
    for index, (actual_gradient, expected_gradient) in enumerate(
        zip(actual_gradients, expected_gradients, strict=True)
    ):
        gradient_precision = torch.bfloat16 if actual_gradient.dtype == torch.bfloat16 else precision
        _assert_precision_close(
            actual_gradient,
            expected_gradient,
            gradient_precision,
            None if absolute_error_bounds is None else absolute_error_bounds[index + 1],
        )


@pytest.mark.parametrize("output_layout", ["head_last", "head_major"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "dtype, distribution",
    [
        (torch.float16, "random"),
        (torch.float16, "offset"),
        (torch.float16, "constant"),
        (torch.bfloat16, "random"),
        (torch.bfloat16, "offset"),
        (torch.float32, "random"),
        (torch.float64, "random"),
    ],
)
def test_centered_equation_and_all_operand_vjps(
    dtype: torch.dtype, distribution: str, has_bias: bool, output_layout: str
) -> None:
    torch.manual_seed(0)
    pair_size = 128 if distribution == "offset" else 5
    pair = torch.randn(2, 3, 5, pair_size, dtype=dtype)
    if distribution == "offset":
        pair.fill_(1024)
        pair[..., -1] += 8 if dtype == torch.bfloat16 else 1
    elif distribution == "constant":
        pair.fill_(1000.5)
    pair.requires_grad_()
    norm_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    norm_weight = torch.randn(pair_size, dtype=norm_dtype, requires_grad=True)
    norm_bias = torch.randn(pair_size, dtype=norm_dtype, requires_grad=True) if has_bias else None
    projection_weight = (0.1 * torch.randn(3, pair_size, dtype=dtype)).requires_grad_()

    actual = pair_norm_projection(
        pair,
        norm_weight,
        norm_bias,
        projection_weight,
        output_layout=output_layout,
        backend="eager",
    )
    expected = _centered_projection_reference(pair, norm_weight, norm_bias, projection_weight)
    if output_layout == "head_major":
        expected = expected.movedim(-1, -3).contiguous()
    operands = (
        (pair, norm_weight, projection_weight)
        if norm_bias is None
        else (pair, norm_weight, norm_bias, projection_weight)
    )
    cotangent = torch.linspace(-0.8, 1.1, actual.numel(), dtype=dtype).reshape_as(actual)
    bounds = (
        _normalization_conditioning_bounds(
            pair, norm_weight, projection_weight, cotangent, has_bias=has_bias, output_layout=output_layout
        )
        if dtype in (torch.float16, torch.bfloat16)
        else None
    )
    _assert_output_and_vjps(actual, expected, operands, dtype, cotangent=cotangent, absolute_error_bounds=bounds)
    assert actual.dtype == dtype
    assert actual.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast requires CUDA")
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_layout", ["head_last", "head_major"])
@pytest.mark.parametrize("backend", ["eager", "auto", "oops"])
def test_cuda_autocast_centered_equation_and_vjps(
    autocast_dtype: torch.dtype, input_dtype: torch.dtype, output_layout: str, backend: str
) -> None:
    if autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    if backend == "oops":
        require_oops("norm_projection")
    torch.manual_seed(1)
    pair = torch.randn(2, 3, 5, 8, device="cuda", dtype=input_dtype, requires_grad=True)
    norm_weight = torch.randn(8, device="cuda", requires_grad=True)
    norm_bias = torch.randn(8, device="cuda", requires_grad=True)
    projection_weight = torch.randn(64, 8, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=autocast_dtype):
        actual = pair_norm_projection(
            pair, norm_weight, norm_bias, projection_weight, output_layout=output_layout, backend=backend
        )
    expected = _centered_projection_reference(
        pair,
        norm_weight,
        norm_bias,
        projection_weight,
        normalization_dtype=torch.float32,
        projection_dtype=autocast_dtype,
    )
    if output_layout == "head_major":
        expected = expected.movedim(-1, -3).contiguous()
    _assert_output_and_vjps(actual, expected, (pair, norm_weight, norm_bias, projection_weight), autocast_dtype)
    assert actual.dtype == autocast_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_missing_oops_projection_is_explicit_and_auto_falls_back() -> None:
    oops = pytest.importorskip("oops")
    if callable(getattr(oops, "norm_projection", None)):
        pytest.skip("installed OOps provides norm_projection")
    pair = torch.randn(1, 3, 3, 8, device="cuda", requires_grad=True)
    norm_weight = torch.randn(8, device="cuda", requires_grad=True)
    norm_bias = torch.randn(8, device="cuda", requires_grad=True)
    projection_weight = torch.randn(4, 8, device="cuda", requires_grad=True)
    inputs = (pair, norm_weight, norm_bias, projection_weight)

    with pytest.raises(InvalidRequestError, match="oops_operator_unavailable.*norm_projection"):
        pair_norm_projection(*inputs, backend="oops")
    actual = pair_norm_projection(*inputs, backend="auto")
    expected = pair_norm_projection(*inputs, backend="eager")
    _assert_output_and_vjps(actual, expected, inputs, torch.float32)


@pytest.mark.parametrize("backend", ["eager", "oops"])
def test_public_shape_domain(backend: str) -> None:
    pair = torch.randn(3, 8)
    norm_weight = torch.randn(8)

    with pytest.raises(ValueError, match="pair must have shape"):
        pair_norm_projection(pair, norm_weight, None, torch.randn(2, 8), backend=backend)
    with pytest.raises(ValueError, match="H > 0"):
        pair_norm_projection(pair[None], norm_weight, None, torch.empty(0, 8), backend=backend)
