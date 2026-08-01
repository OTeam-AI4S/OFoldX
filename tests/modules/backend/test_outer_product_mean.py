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

"""Outer-product dispatch selects unavailable inputs before provider math."""

from copy import deepcopy

import pytest
import torch

from ofoldx.modules.backend import outer_product_mean
from ofoldx.modules.msa import OuterProductMean, PostProjectionOuterProductMean


def test_provider_input_availability(monkeypatch):
    import importlib

    owner = importlib.import_module("ofoldx.modules.backend.outer_product_mean")

    def provider(msa, mask, **kwargs):
        assert torch.isfinite(msa).all()
        assert torch.equal(msa[~mask], torch.zeros_like(msa[~mask]))
        return msa.sum(dim=-3)

    monkeypatch.setattr(owner, "_oops_outer_product_mean", provider)
    monkeypatch.setattr(owner, "resolve_tensor_backend", lambda *args, **kwargs: "oops")
    msa = torch.tensor([[[[1.0, 2.0], [torch.nan, torch.inf]]]], requires_grad=True)
    mask = torch.tensor([[[True, False]]])
    result = outer_product_mean(
        msa,
        mask,
        msa_norm_weight=torch.ones(2),
        msa_norm_bias=torch.zeros(2),
        left_weight=torch.ones(2, 2),
        right_weight=torch.ones(2, 2),
        output_weight=torch.ones(2, 4),
        output_bias=torch.zeros(2),
        msa_norm_eps=1e-5,
        denominator_eps=0.0,
        divide_before_projection=True,
        clamp_denominator=False,
        backend="oops",
    )
    gradient = torch.autograd.grad(result.sum(), msa)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the OOps backend")
def test_auto_falls_back_for_unsupported_float64():
    oops = pytest.importorskip("oops")
    if not hasattr(oops, "outer_product_mean") or not hasattr(oops, "OOpsUnsupportedError"):
        pytest.skip("this OOps installation does not provide outer-product mean")
    expected = OuterProductMean(msa_size=4, hidden_size=2, pair_size=3, backend="eager").double().cuda()
    actual = deepcopy(expected)
    actual.backend = "auto"
    msa = torch.randn(1, 2, 3, 4, dtype=torch.float64, device="cuda", requires_grad=True)
    torch.testing.assert_close(actual(msa), expected(msa))
    actual.backend = "oops"
    with pytest.raises(oops.OOpsUnsupportedError):
        actual(msa)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the OOps backend")
@pytest.mark.parametrize("module_type", [OuterProductMean, PostProjectionOuterProductMean])
@pytest.mark.parametrize("prefix", [(), (2,), (2, 3)])
@pytest.mark.parametrize("broadcast_mask", [False, True])
def test_oops_outer_product_mean_with_leading_sample_axes(module_type, prefix, broadcast_mask):
    oops = pytest.importorskip("oops")
    if not hasattr(oops, "outer_product_mean"):
        pytest.skip("this OOps installation does not provide outer-product mean")
    torch.manual_seed(31)
    reference = module_type(msa_size=8, hidden_size=4, pair_size=6, backend="eager").cuda()
    actual = deepcopy(reference)
    actual.backend = "auto"
    msa = torch.randn(*prefix, 3, 5, 8, device="cuda", requires_grad=True)
    ref_msa = msa.detach().clone().requires_grad_()
    mask_prefix = () if broadcast_mask else prefix
    mask = torch.rand(*mask_prefix, 3, 5, device="cuda") > 0.3
    out = actual(msa, mask)
    expected = reference(ref_msa, mask)
    assert out.shape == (*prefix, 5, 5, 6)
    torch.testing.assert_close(out, expected, rtol=3e-4, atol=3e-5)
    for got, want in zip(
        torch.autograd.grad(out.square().sum(), (msa, *actual.parameters())),
        torch.autograd.grad(expected.square().sum(), (ref_msa, *reference.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=8e-4, atol=8e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the OOps backend")
@pytest.mark.parametrize("module_type", [OuterProductMean, PostProjectionOuterProductMean])
@pytest.mark.parametrize("shape", [(2, 3, 0, 5, 8), (2, 3, 2, 0, 8), (2, 0, 3, 5, 8)])
def test_oops_outer_product_mean_with_empty_axes(module_type, shape):
    oops = pytest.importorskip("oops")
    if not hasattr(oops, "outer_product_mean"):
        pytest.skip("this OOps installation does not provide outer-product mean")
    reference = module_type(msa_size=8, hidden_size=4, pair_size=6, backend="eager").cuda()
    actual = deepcopy(reference)
    actual.backend = "auto"
    msa = torch.randn(*shape, device="cuda", requires_grad=True)
    ref_msa = msa.detach().clone().requires_grad_()
    mask = torch.ones(*shape[:-1], dtype=torch.bool, device="cuda")
    out, expected = actual(msa, mask), reference(ref_msa, mask)
    assert out.shape == (*shape[:-3], shape[-2], shape[-2], 6)
    torch.testing.assert_close(out, expected)
    for got, want in zip(
        torch.autograd.grad(out.sum(), (msa, *actual.parameters())),
        torch.autograd.grad(expected.sum(), (ref_msa, *reference.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the OOps backend")
@pytest.mark.parametrize("module_type", [OuterProductMean, PostProjectionOuterProductMean])
def test_oops_outer_product_mean_broadcast_float_mask_gradients(module_type):
    oops = pytest.importorskip("oops")
    if not hasattr(oops, "outer_product_mean"):
        pytest.skip("this OOps installation does not provide outer-product mean")
    torch.manual_seed(43)
    reference = module_type(msa_size=8, hidden_size=4, pair_size=6, backend="eager").cuda()
    actual = deepcopy(reference)
    actual.backend = "auto"
    msa = torch.randn(2, 4, 3, 5, 8, device="cuda").transpose(1, 2).requires_grad_()
    ref_msa = msa.detach().clone().requires_grad_()
    mask = torch.rand(1, 1, 4, 5, device="cuda", requires_grad=True)
    ref_mask = mask.detach().clone().requires_grad_()
    out, expected = actual(msa, mask), reference(ref_msa, ref_mask)
    torch.testing.assert_close(out, expected, rtol=3e-4, atol=3e-5)
    for got, want in zip(
        torch.autograd.grad(out.square().sum(), (msa, mask, *actual.parameters())),
        torch.autograd.grad(expected.square().sum(), (ref_msa, ref_mask, *reference.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=8e-4, atol=8e-5)
