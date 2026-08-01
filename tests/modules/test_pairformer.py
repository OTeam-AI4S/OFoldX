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

from functools import partial
from typing import TypedDict

import pytest
import torch
from danling.runners.compile import Compiler
from danling.runners.config import CompileConfig
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.pairformer import PairformerBlock, PairformerStack


class _PairformerArgs(TypedDict):
    single_size: int
    pair_size: int
    triangle_multiplication_hidden_size: int
    triangle_attention_head_size: int
    num_triangle_attention_heads: int
    num_pair_attention_heads: int
    dropout: float
    backend: str


def _kwargs(*, dropout: float = 0.0) -> _PairformerArgs:
    return {
        "single_size": 8,
        "pair_size": 4,
        "triangle_multiplication_hidden_size": 4,
        "triangle_attention_head_size": 2,
        "num_triangle_attention_heads": 2,
        "num_pair_attention_heads": 2,
        "dropout": dropout,
        "backend": "eager",
    }


class TestPairformerBlock:
    def test_shapes(self) -> None:
        single = torch.randn(1, 3, 8)
        pair = torch.randn(1, 3, 3, 4)

        output_single, output_pair = PairformerBlock(**_kwargs())(single, pair)

        assert output_single.shape == single.shape
        assert output_pair.shape == pair.shape


class TestPairformerStack:
    def test_shapes(self) -> None:
        single = torch.randn(1, 3, 8)
        pair = torch.randn(1, 3, 3, 4)

        output_single, output_pair = PairformerStack(num_blocks=2, **_kwargs())(single, pair)

        assert output_single.shape == single.shape
        assert output_pair.shape == pair.shape

    def test_gradient_checkpointing(self) -> None:
        eager = PairformerStack(num_blocks=2, **_kwargs())
        checkpointed = PairformerStack(num_blocks=2, **_kwargs())
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        eager_single = torch.randn(1, 3, 8, requires_grad=True)
        eager_pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        checkpointed_single = eager_single.detach().clone().requires_grad_()
        checkpointed_pair = eager_pair.detach().clone().requires_grad_()

        eager_output = eager(eager_single, eager_pair)
        checkpointed_output = checkpointed(checkpointed_single, checkpointed_pair)
        sum(output.sum() for output in eager_output).backward()
        sum(output.sum() for output in checkpointed_output).backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_single.grad, eager_single.grad)
        torch.testing.assert_close(checkpointed_pair.grad, eager_pair.grad)
        for actual, expected in zip(checkpointed.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("num_blocks", [0, 1, 3, 5])
def test_grouped_dropout_gradients(num_blocks: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = _kwargs(dropout=0.25)
    reference = PairformerStack(num_blocks=num_blocks, **kwargs).to(device)
    grouped = PairformerStack(num_blocks=num_blocks, checkpoint_group_size=4, **kwargs).to(device)
    _assert_grouped_gradients(reference, grouped, device=device)


def test_regional_grouped_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = _kwargs(dropout=0.25)
    reference = PairformerStack(num_blocks=5, **kwargs).to(device)
    grouped = PairformerStack(num_blocks=5, checkpoint_group_size=4, **kwargs).to(device)
    compiler = Compiler(
        CompileConfig({"enabled": True, "backend": "aot_eager", "fullgraph": True, "optimize_ddp": None})
    )
    for module in (reference, grouped):
        regions = module.local_compilation_region_groups()
        assert regions == (tuple(module.blocks),)
        compiler.compile_modules(regions[0])
    _assert_grouped_gradients(reference, grouped, device=device)


def test_invalid_checkpoint_group() -> None:
    with pytest.raises(ValueError, match="checkpoint_group_size must be positive"):
        PairformerStack(checkpoint_group_size=0, **_kwargs())


def _assert_grouped_gradients(reference: PairformerStack, grouped: PairformerStack, *, device: torch.device) -> None:
    grouped.load_state_dict(reference.state_dict(), strict=True)
    single = torch.randn(1, 4, 8, device=device)
    pair = torch.randn(1, 4, 4, 4, device=device)
    pair_mask = torch.ones(1, 4, 4, dtype=torch.bool, device=device)
    pair_mask[..., -1] = False
    results = []
    for module in (reference, grouped):
        module.gradient_checkpointing = True
        module._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        single_input = single.detach().clone().requires_grad_()
        pair_input = pair.detach().clone().requires_grad_()
        torch.manual_seed(23)
        outputs = module(single_input, pair_input, pair_mask)
        forward_cpu_rng = torch.get_rng_state().clone()
        forward_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        sum(output.square().sum() for output in outputs).backward()
        torch.testing.assert_close(torch.get_rng_state(), forward_cpu_rng, rtol=0, atol=0)
        if forward_cuda_rng is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(device), forward_cuda_rng, rtol=0, atol=0)
        results.append(
            (
                tuple(output.detach() for output in outputs),
                single_input.grad,
                pair_input.grad,
                tuple(parameter.grad for parameter in module.parameters()),
                forward_cpu_rng,
                forward_cuda_rng,
            )
        )
    torch.testing.assert_close(results[1], results[0])
