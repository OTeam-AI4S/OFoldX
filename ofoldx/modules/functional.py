r"""
Functional tensor helpers for modules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def checkpoint_blocks(
    blocks: Sequence[Callable[..., Tensor | tuple[Tensor, ...]]],
    args: tuple[Tensor, ...],
    blocks_per_checkpoint: int | None,
) -> tuple[Tensor, ...]:
    r"""
    Run a sequence of blocks in order, optionally trading compute for memory via checkpointing.

    Each block consumes the previous block's outputs and returns a tensor or a tuple of tensors.
    When ``blocks_per_checkpoint`` is set (and gradients are enabled) the stack is split into
    contiguous segments of that many blocks, and each segment is wrapped in
    [torch.utils.checkpoint.checkpoint][] so its activations are recomputed in the backward pass
    instead of stored. Checkpointing is skipped when ``blocks_per_checkpoint`` is ``None`` or when
    running under [torch.no_grad][].

    Args:
        blocks: Sequence of callables that consume and return tensors or tensor tuples.
        args: Initial tensor arguments for the first block.
        blocks_per_checkpoint: Number of blocks per checkpoint segment, or ``None`` to disable
            checkpointing.

    Returns:
        Final block outputs, always normalized to a tuple.

    Raises:
        ValueError: If ``blocks_per_checkpoint`` is outside ``[1, len(blocks)]``.

    Examples:
        >>> import torch
        >>> double = lambda x: x * 2
        >>> add_one = lambda x: x + 1
        >>> out, = checkpoint_blocks([double, add_one], (torch.tensor([1.0, 2.0]),), None)
        >>> out.tolist()
        [3.0, 5.0]
        >>> out, = checkpoint_blocks([double, add_one], (torch.tensor([1.0, 2.0]),), 1)
        >>> out.tolist()
        [3.0, 5.0]
    """

    if blocks_per_checkpoint is None or not torch.is_grad_enabled():
        return _run_blocks(blocks, args)
    if blocks_per_checkpoint < 1 or blocks_per_checkpoint > len(blocks):
        raise ValueError("blocks_per_checkpoint must be between 1 and len(blocks)")

    outputs = args
    for start in range(0, len(blocks), blocks_per_checkpoint):
        stop = start + blocks_per_checkpoint
        outputs = checkpoint(
            _block_runner(blocks[start:stop]),
            *outputs,
            use_reentrant=False,
        )
        outputs = _as_tuple(outputs)
    return outputs


def _block_runner(blocks: Sequence[Callable[..., Tensor | tuple[Tensor, ...]]]) -> Callable[..., tuple[Tensor, ...]]:
    r"""
    Close over a block segment and return a flat-args callable for [torch.utils.checkpoint.checkpoint][].

    Args:
        blocks: The block segment to run when the returned callable is invoked.

    Returns:
        A callable that runs the segment on positional tensor arguments and returns a tuple.
    """

    def run(*args: Tensor) -> tuple[Tensor, ...]:
        r"""
        Run the captured block segment.

        Args:
            *args: Segment input tensors.

        Returns:
            Segment output tensors.
        """

        return _run_blocks(blocks, args)

    return run


def _run_blocks(
    blocks: Sequence[Callable[..., Tensor | tuple[Tensor, ...]]],
    args: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    r"""
    Thread ``args`` through ``blocks`` in order, normalizing each block's output to a tuple.

    Args:
        blocks: Blocks to run in sequence.
        args: Inputs for the first block.

    Returns:
        The final block's outputs as a tuple.
    """

    outputs = args
    for block in blocks:
        outputs = _as_tuple(block(*outputs))
    return outputs


def _as_tuple(output: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    r"""
    Wrap a single tensor in a one-element tuple, leaving existing tuples unchanged.

    Args:
        output: A tensor or a tuple of tensors.

    Returns:
        ``output`` if it is already a tuple, otherwise ``(output,)``.
    """

    return output if isinstance(output, tuple) else (output,)


def pair_mask_from_token_mask(token_mask: Tensor) -> Tensor:
    r"""
    Build a pair-validity mask by taking the outer conjunction of a token mask.

    A token pair ``(i, j)`` is valid only when both token ``i`` and token ``j`` are valid, so the
    result is ``token_mask[..., :, None] & token_mask[..., None, :]``.

    Args:
        token_mask: `[..., N_token]` token-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask.

    Examples:
        >>> import torch
        >>> token_mask = torch.tensor([[1, 1, 0]])
        >>> pair_mask_from_token_mask(token_mask).int().tolist()
        [[[1, 1, 0], [1, 1, 0], [0, 0, 0]]]
    """

    token_mask = token_mask.bool()
    return token_mask[..., :, None] & token_mask[..., None, :]


def pair_mask_from_msa_mask(msa_mask: Tensor) -> Tensor:
    r"""
    Build a token-pair mask from an MSA mask, keeping tokens with any valid MSA row.

    A token is treated as valid when at least one MSA row covers it (``any`` over the MSA axis);
    the per-token mask is then expanded into a pair mask via
    [pair_mask_from_token_mask][ofoldx.modules.functional.pair_mask_from_token_mask].

    Args:
        msa_mask: `[..., N_msa, N_token]` MSA-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask for tokens with at least one valid
        MSA row.

    Examples:
        >>> import torch
        >>> msa_mask = torch.tensor([[[1, 0, 0], [0, 1, 0]]])
        >>> pair_mask_from_msa_mask(msa_mask).int().tolist()
        [[[1, 1, 0], [1, 1, 0], [0, 0, 0]]]
    """

    token_mask = msa_mask.bool().any(dim=-2)
    return pair_mask_from_token_mask(token_mask)
