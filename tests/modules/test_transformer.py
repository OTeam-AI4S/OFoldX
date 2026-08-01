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

from copy import deepcopy
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.transformer import (
    AtomCrossAttentionTransformerEncoderBlock,
    ConditionedTransformerEncoder,
    ConditionedTransformerEncoderBlock,
    NormalizedKeyAtomCrossAttentionTransformerEncoderBlock,
    PairBiasedConditionedTransformerEncoder,
    PostNormConditionedTransformerEncoderBlock,
    PostNormQueryKeyNormConditionedTransformerEncoderBlock,
    QueryKeyNormConditionedTransformerEncoderBlock,
)
from ofoldx.modules.transition import ConditionedTransition


def _learned_parameters(module, generator):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = torch.randn(parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=generator)
            parameter.copy_(1 + 0.1 * values if name.endswith("norm.weight") else 0.08 * values)
    return {name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()}


def _centered_norm(value):
    centered = value - value.mean(dim=-1, keepdim=True)
    return centered / (centered.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()


def _affine_reference(value, name, parameters):
    return F.linear(value, parameters[name + ".weight"], parameters.get(name + ".bias"))


def _norm_reference(value, name, parameters):
    return _centered_norm(value) * parameters[name + ".weight"] + parameters.get(name + ".bias", 0)


def _adaptive_reference(hidden, conditioning, name, parameters):
    condition = _norm_reference(conditioning, name + ".conditioning_norm", parameters)
    scale, shift = _affine_reference(condition, name + ".conditioning_projection", parameters).chunk(2, dim=-1)
    return (scale + parameters[name + ".conditioning_scale_bias"]).sigmoid() * _centered_norm(hidden) + shift


def _conditioned_block_reference(
    hidden, conditioning, bias, attention_mask, parameters, *, prefix="", gather=None, key_mode="self", qk_norm=False
):
    """Independent complete learned equation, including the explicit mask's real axes."""
    prefix = prefix + "." if prefix else ""
    query_input = _adaptive_reference(hidden, conditioning, prefix + "conditioned_norm", parameters)
    key_input = query_input
    if key_mode != "self":
        key_base = hidden if key_mode == "raw" else query_input
        key_input = _adaptive_reference(key_base, conditioning, prefix + "key_value_norm", parameters)
    if gather is not None:
        key_input = gather(key_input)
    projection = prefix + "attention.projection"
    query = _affine_reference(query_input, projection + ".query_projection", parameters).unflatten(-1, (2, 4))
    key = _affine_reference(key_input, projection + ".key_projection", parameters).unflatten(-1, (2, 4))
    value = _affine_reference(key_input, projection + ".value_projection", parameters).unflatten(-1, (2, 4))
    if qk_norm:
        query = _norm_reference(query, projection + ".query_norm", parameters)
        key = _norm_reference(key, projection + ".key_norm", parameters)
    while bias.ndim < hidden.ndim + 1:
        bias = bias.unsqueeze(-4)
    logits = torch.einsum("...qhd,...khd->...hqk", query, key) / 2 + bias.movedim(-1, -3)
    keep = attention_mask
    while keep.ndim < logits.ndim:
        keep = keep.unsqueeze(-3)
    probabilities = logits.masked_fill(~keep, -torch.inf).softmax(dim=-1)
    attended = torch.einsum("...hqk,...khd->...qhd", probabilities, value).flatten(-2)
    attended = attended * _affine_reference(query_input, projection + ".gate_projection", parameters).sigmoid()
    update = _affine_reference(attended, projection + ".output_projection", parameters)
    hidden = hidden + _affine_reference(conditioning, prefix + "output_gate_projection", parameters).sigmoid() * update
    transition = prefix + "transition"
    normalized = _adaptive_reference(hidden, conditioning, transition + ".conditioned_norm", parameters)
    if transition + ".expand_projection.weight" in parameters:
        activated = (
            F.silu(_affine_reference(normalized, transition + ".gate_projection", parameters))
            * _affine_reference(normalized, transition + ".value_projection", parameters)
            * _affine_reference(normalized, transition + ".expand_projection", parameters)
        )
        projected = _affine_reference(activated, transition + ".output_projection", parameters)
    else:
        activated = F.silu(
            _affine_reference(normalized, transition + ".feed_forward.gate_projection", parameters)
        ) * _affine_reference(normalized, transition + ".feed_forward.value_projection", parameters)
        projected = _affine_reference(activated, transition + ".feed_forward.output_projection", parameters)
    hidden = (
        hidden
        + _affine_reference(conditioning, transition + ".conditioning_gate_projection", parameters).sigmoid()
        * projected
    )
    if prefix + "post_norm.weight" in parameters:
        hidden = _norm_reference(hidden, prefix + "post_norm", parameters)
    return hidden


def _compare_complete_vjps(
    actual, expected, inputs, expected_inputs, module, parameters, generator, *, rtol=5e-10, atol=5e-11
):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    cotangent = torch.randn(actual.shape, dtype=actual.dtype, device=actual.device, generator=generator)
    named = list(module.named_parameters())
    actual_gradients = torch.autograd.grad((actual * cotangent).sum(), (*inputs, *(value for _, value in named)))
    expected_gradients = torch.autograd.grad(
        (expected * cotangent).sum(), (*expected_inputs, *(parameters[name] for name, _ in named))
    )
    for name, actual_gradient, expected_gradient in zip(
        [*(f"input{index}" for index in range(len(inputs))), *(name for name, _ in named)],
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol, msg=name)


_BLOCKS = (
    (ConditionedTransformerEncoderBlock, "self", False),
    (PostNormConditionedTransformerEncoderBlock, "self", False),
    (QueryKeyNormConditionedTransformerEncoderBlock, "self", True),
    (PostNormQueryKeyNormConditionedTransformerEncoderBlock, "self", True),
    (AtomCrossAttentionTransformerEncoderBlock, "raw", False),
    (NormalizedKeyAtomCrossAttentionTransformerEncoderBlock, "normalized", False),
    (
        partial(ConditionedTransformerEncoderBlock, transition=partial(ConditionedTransition, expansion=2)),
        "self",
        False,
    ),
)


@pytest.mark.parametrize("block_type,key_mode,qk_norm", _BLOCKS)
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("indices", [None, (2, 0, 3, 1), (3, 1, 0)])
def test_complete_encoder_validity_equation_and_all_named_vjps(block_type, key_mode, qk_norm, sampled, indices):
    generator = torch.Generator().manual_seed(816)
    encoder = ConditionedTransformerEncoder(2, 2, 8, 6, backend="eager", block=block_type).double().eval()
    parameters = _learned_parameters(encoder, generator)
    prefix = (4, 3) if sampled else (4,)
    key_indices = None if indices is None else torch.tensor(indices)
    key_count = 4 if indices is None else len(indices)
    inputs = (
        torch.randn(*prefix, 4, 8, dtype=torch.float64, generator=generator).requires_grad_(),
        torch.randn(*prefix, 4, 6, dtype=torch.float64, generator=generator).requires_grad_(),
        torch.randn(4, 4, key_count, 4, dtype=torch.float64, generator=generator).requires_grad_(),
    )
    expected_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    validity = ~torch.eye(4, dtype=torch.bool)
    gather = None if key_indices is None else lambda value: value.index_select(-2, key_indices)
    actual = encoder(*inputs, mask=validity, to_keys=gather)
    key_validity = validity if key_indices is None else validity[:, key_indices]
    attention_mask = key_validity.unsqueeze(-2)
    expected = expected_inputs[0]
    for index in range(2):
        expected = _conditioned_block_reference(
            expected,
            expected_inputs[1],
            expected_inputs[2][..., index * 2 : (index + 1) * 2],
            attention_mask,
            parameters,
            prefix=f"blocks.{index}",
            gather=gather,
            key_mode=key_mode,
            qk_norm=qk_norm,
        )
    _compare_complete_vjps(actual, expected, inputs, expected_inputs, encoder, parameters, generator)


def test_actual_block_nonfactorizable_pair_mask_complete_equation_and_vjps():
    generator = torch.Generator().manual_seed(827)
    block = ConditionedTransformerEncoderBlock(8, 6, 2, backend="eager").double().eval()
    parameters = _learned_parameters(block, generator)
    inputs = tuple(
        torch.randn(4, 4, width, dtype=torch.float64, generator=generator).requires_grad_() for width in (8, 6)
    )
    bias = torch.randn(4, 4, 4, 2, dtype=torch.float64, generator=generator).requires_grad_()
    inputs = (*inputs, bias)
    expected_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    pair_mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True], [False, True, True, True], [True, True, True, False]]
    ).expand(4, -1, -1)
    actual = block(*inputs, attention_mask=pair_mask)
    expected = _conditioned_block_reference(*expected_inputs, pair_mask, parameters)
    _compare_complete_vjps(actual, expected, inputs, expected_inputs, block, parameters, generator)


class TestDoubleProductConditionedTransformerEncoderBlock:
    def test_accepts_key_mask(self) -> None:
        block = ConditionedTransformerEncoderBlock(
            hidden_size=4,
            conditioning_size=3,
            num_attention_heads=1,
            backend="eager",
            transition=partial(ConditionedTransition, expansion=2),
        ).eval()

        output = block(
            hidden=torch.randn(2, 4, 4),
            conditioning=torch.randn(2, 4, 3),
            bias=torch.zeros(2, 4, 4, 1),
            attention_mask=torch.tensor([[True, False, True, True], [True, True, False, True]]).unsqueeze(-2),
        )

        assert output.shape == (2, 4, 4)
        assert torch.isfinite(output).all()


class TestConditionedTransformerEncoder:
    def test_gradient_checkpointing(self) -> None:
        eager = ConditionedTransformerEncoder(
            num_blocks=2,
            num_attention_heads=1,
            hidden_size=4,
            conditioning_size=3,
            backend="eager",
            block=partial(ConditionedTransformerEncoderBlock, transition=partial(ConditionedTransition, expansion=2)),
        )
        checkpointed = ConditionedTransformerEncoder(
            num_blocks=2,
            num_attention_heads=1,
            hidden_size=4,
            conditioning_size=3,
            backend="eager",
            block=partial(ConditionedTransformerEncoderBlock, transition=partial(ConditionedTransition, expansion=2)),
        )
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        eager_hidden = torch.randn(2, 4, 4, requires_grad=True)
        checkpointed_hidden = eager_hidden.detach().clone().requires_grad_()
        conditioning = torch.randn(2, 4, 3)
        bias = torch.zeros(2, 4, 4, 2)
        validity = torch.tensor([[True, True, False, True], [True, False, True, True]])
        indices = torch.tensor([2, 0, 3, 1])

        def gather(value):
            return value.index_select(-2, indices)

        eager_output = eager(eager_hidden, conditioning, bias, validity, gather)
        checkpointed_output = checkpointed(checkpointed_hidden, conditioning, bias, validity, gather)
        eager_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_hidden.grad, eager_hidden.grad)
        for checkpointed_parameter, eager_parameter in zip(
            checkpointed.parameters(),
            eager.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(checkpointed_parameter.grad, eager_parameter.grad)


def test_complete_encoder_validity_dynamic_fullgraph() -> None:
    generator = torch.Generator().manual_seed(853)
    encoder = (
        ConditionedTransformerEncoder(1, 2, 8, 6, backend="eager", block=ConditionedTransformerEncoderBlock)
        .double()
        .eval()
    )
    parameters = _learned_parameters(encoder, generator)
    compiled = torch.compile(encoder, backend="eager", fullgraph=True, dynamic=True)
    for num_positions in (4, 7):
        inputs = (
            torch.randn(4, num_positions, 8, dtype=torch.float64, generator=generator).requires_grad_(),
            torch.randn(4, num_positions, 6, dtype=torch.float64, generator=generator).requires_grad_(),
            torch.randn(4, num_positions, num_positions, 2, dtype=torch.float64, generator=generator).requires_grad_(),
        )
        expected_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        validity = torch.arange(num_positions)[None] != torch.arange(4)[:, None]
        actual = compiled(*inputs, mask=validity)
        expected = _conditioned_block_reference(*expected_inputs, validity.unsqueeze(-2), parameters, prefix="blocks.0")
        _compare_complete_vjps(actual, expected, inputs, expected_inputs, encoder, parameters, generator)


class TestPairBiasedConditionedTransformerEncoder:
    def test_identity(self) -> None:
        encoder = PairBiasedConditionedTransformerEncoder(
            num_blocks=0,
            token_hidden_size=4,
            conditioning_size=3,
            pair_size=2,
            num_attention_heads=1,
            backend="eager",
        )
        token_hidden = torch.randn(2, 5, 4)

        output = encoder(
            token_hidden,
            conditioning=torch.randn(2, 5, 3),
            pair=torch.randn(2, 5, 5, 2),
        )

        torch.testing.assert_close(output, token_hidden)

    @pytest.mark.parametrize(
        "mask",
        [
            torch.tensor([[True, True, False, True, True], [True, False, True, True, True]]).unsqueeze(-2),
            torch.tensor(
                [
                    [[True, False, False, True, True]] + [[True, True, False, True, True]] * 4,
                    [[True, False, True, True, True]] * 4 + [[False, False, True, True, True]],
                ]
            ),
        ],
        ids=("key", "pair"),
    )
    def test_masks(self, mask: torch.Tensor) -> None:
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        encoder = (
            PairBiasedConditionedTransformerEncoder(
                num_blocks=1,
                token_hidden_size=8,
                conditioning_size=6,
                pair_size=4,
                num_attention_heads=2,
                backend="eager",
            )
            .double()
            .eval()
        )
        parameters = _learned_parameters(encoder, generator)
        inputs = tuple(
            torch.randn(shape, dtype=torch.float64, generator=generator).requires_grad_()
            for shape in ((2, 5, 8), (2, 5, 6), (2, 5, 5, 4))
        )
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        output = encoder(*inputs, pair_mask=mask)
        bias = _affine_reference(
            _norm_reference(reference_inputs[2], "pair_norms.0", parameters), "bias_projections.0", parameters
        )
        expected = _conditioned_block_reference(*reference_inputs[:2], bias, mask, parameters, prefix="blocks.0")
        _compare_complete_vjps(output, expected, inputs, reference_inputs, encoder, parameters, generator)

        pair_keep = mask.expand(2, 5, 5)
        excluded_keys = ~pair_keep.any(dim=-2)
        altered_inputs = tuple(value.detach().clone() for value in inputs)
        with torch.no_grad():
            for value in altered_inputs[:2]:
                value[excluded_keys] = 3 * torch.randn(
                    value[excluded_keys].shape, dtype=value.dtype, generator=generator
                )
            altered_inputs[2][~pair_keep] = 3 * inputs[2][~pair_keep].flip(-1)
            altered_output = encoder(*altered_inputs, pair_mask=mask)
            torch.testing.assert_close(altered_output[~excluded_keys], output[~excluded_keys], rtol=5e-10, atol=5e-11)

        assert output.shape == (2, 5, 8)
        assert torch.isfinite(output).all()

    def test_checkpointing(self) -> None:
        torch.manual_seed(1)
        eager = PairBiasedConditionedTransformerEncoder(
            num_blocks=2,
            token_hidden_size=4,
            conditioning_size=3,
            pair_size=2,
            num_attention_heads=1,
            backend="eager",
        )
        checkpointed = deepcopy(eager)
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        eager_inputs = (
            torch.randn(2, 5, 4, requires_grad=True),
            torch.randn(2, 5, 3, requires_grad=True),
            torch.randn(2, 5, 5, 2, requires_grad=True),
        )
        checkpointed_inputs = tuple(value.detach().clone().requires_grad_() for value in eager_inputs)
        pair_mask = torch.ones(2, 5, 5, dtype=torch.bool)
        pair_mask[:, :, -1] = False

        eager_output = eager(*eager_inputs, pair_mask)
        checkpointed_output = checkpointed(*checkpointed_inputs, pair_mask)
        eager_output.square().sum().backward()
        checkpointed_output.square().sum().backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        for actual, expected in zip(checkpointed_inputs, eager_inputs, strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
