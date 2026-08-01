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
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import nn

from ofoldx.modules.backend import LayerNorm, normalization
from tests.modules import require_oops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast requires CUDA")
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("input_dtype", "affine_dtype"),
    [
        (torch.float16, None),
        (torch.bfloat16, None),
        (torch.float32, None),
        (torch.float64, None),
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_auto_matches_native_cuda_autocast(input_dtype: torch.dtype, affine_dtype, amp_dtype: torch.dtype) -> None:
    if torch.bfloat16 in (input_dtype, affine_dtype, amp_dtype) and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    _assert_auto_matches_native(input_dtype, affine_dtype, amp_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("affine", [False, True])
def test_auto_matches_native_cuda_without_autocast(input_dtype: torch.dtype, affine: bool) -> None:
    if input_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    _assert_auto_matches_native(input_dtype, input_dtype if affine else None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_explicit_oops_keeps_layer_norm_dtype_under_autocast(input_dtype: torch.dtype, amp_dtype: torch.dtype) -> None:
    require_oops("layer_norm")
    if torch.bfloat16 in (input_dtype, amp_dtype) and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    torch.manual_seed(22)
    input = torch.randn(2, 3, 8, device="cuda", dtype=input_dtype, requires_grad=True)
    weight = torch.randn(8, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=amp_dtype):
        actual = normalization.layer_norm(input, (8,), weight, bias, 1e-5, backend="oops")
    with torch.autocast("cuda", enabled=False):
        expected = F.layer_norm(input.float(), (8,), weight, bias, 1e-5).to(input_dtype)
    assert actual.dtype == expected.dtype == input_dtype
    rtol, atol = (3e-2, 2e-2) if input_dtype == torch.bfloat16 else (5e-3, 3e-3)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    grad = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, (input, weight, bias), grad_outputs=grad)
    expected_grads = torch.autograd.grad(expected, (input, weight, bias), grad_outputs=grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_nested_cuda_autocast_matches_dense_gradients_and_fullgraph() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    torch.manual_seed(23)
    weight = torch.randn(8, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)

    def run(input: NestedTensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            return normalization.layer_norm(input, (8,), weight, bias, 1e-5, backend="auto").concat

    compiled = torch.compile(run, backend="eager", fullgraph=True, dynamic=True)
    for lengths in ((3, 5), (4, 7)):
        template = NestedTensor([torch.empty(length, 8) for length in lengths], ragged_dims=(0,))
        values = torch.randn(template.concat.shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        input = template.packed_like(values)
        actual = compiled(input, weight, bias)
        with torch.autocast("cuda", dtype=torch.float16):
            expected = torch.cat([F.layer_norm(part, (8,), weight, bias, 1e-5) for part in torch.unbind(input)])
        assert actual.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        grad = torch.randn_like(expected)
        actual_grads = torch.autograd.grad(actual, (values, weight, bias), grad_outputs=grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (values, weight, bias), grad_outputs=grad)
        torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=3e-2, atol=2e-2)
        for actual_grad, expected_grad in zip(actual_grads[1:], expected_grads[1:], strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)


def _assert_auto_matches_native(input_dtype: torch.dtype, affine_dtype, amp_dtype: torch.dtype | None) -> None:
    torch.manual_seed(21)
    input = torch.randn(2, 3, 8, device="cuda", dtype=input_dtype, requires_grad=True)
    weight = None if affine_dtype is None else torch.randn(8, device="cuda", dtype=affine_dtype, requires_grad=True)
    bias = None if affine_dtype is None else torch.randn(8, device="cuda", dtype=affine_dtype, requires_grad=True)
    inputs = tuple(value for value in (input, weight, bias) if value is not None)
    with torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
        actual = normalization.layer_norm(input, (8,), weight, bias, 1e-5, backend="auto")
        expected = F.layer_norm(input, (8,), weight, bias, 1e-5)
    assert actual.dtype == expected.dtype
    if expected.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    elif expected.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=3e-3)
    elif expected.dtype == torch.float64:
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    grad = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, inputs, grad_outputs=grad)
    expected_grads = torch.autograd.grad(expected, inputs, grad_outputs=grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        if expected_grad.dtype == torch.bfloat16:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=2e-2)
        elif expected_grad.dtype == torch.float16:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-3, atol=3e-3)
        elif expected_grad.dtype == torch.float64:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-10, atol=2e-10)
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)


class TestLayerNorm:
    def test_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*args, **kwargs):
            raise RuntimeError("provider failure")

        monkeypatch.setattr(normalization, "_oops_layer_norm", fail)

        with pytest.raises(RuntimeError, match="provider failure"):
            LayerNorm(8, backend="oops")(torch.randn(2, 8))

    @pytest.mark.parametrize("elementwise_affine", [False, True])
    @pytest.mark.parametrize("bias", [False, True])
    def test_torch_parity(self, elementwise_affine: bool, bias: bool) -> None:
        reference = nn.LayerNorm(8, elementwise_affine=elementwise_affine, bias=bias)
        actual = LayerNorm(8, elementwise_affine=elementwise_affine, bias=bias, backend="eager")
        actual.load_state_dict(reference.state_dict())
        hidden = torch.randn(2, 3, 8)

        torch.testing.assert_close(actual(hidden), reference(hidden))
        assert tuple(actual.state_dict()) == tuple(reference.state_dict())

    @pytest.mark.parametrize("backend", ["sdpa", "cueq", "apex", "deepspeed"])
    def test_fallback(self, backend: str) -> None:
        norm = LayerNorm(8, backend="eager")
        hidden = torch.randn(2, 3, 8)
        expected = norm(hidden)
        norm.backend = backend

        torch.testing.assert_close(norm(hidden), expected)
