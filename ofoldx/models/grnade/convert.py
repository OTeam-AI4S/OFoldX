r"""
Convert the released ODesign ``grnade.h5`` checkpoint.

The source layout is geometric-rna-design v1.0.0 revision
``2453b18778a1981ac25f314790e5395f1e9ab218``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .config import GrnadeConfig
from .model import GrnadeModel

_PLACEHOLDER_KEYS = frozenset(
    {
        "W_v.1.dummy_param",
        "W_e.1.dummy_param",
        "W_out.dummy_param",
    }
).union(
    {
        f"{family}.{block}.{path}.dummy_param"
        for family in ("encoder_layers", "decoder_layers")
        for block in range(4)
        for path in (
            "conv.message_func.0",
            "conv.message_func.1",
            "conv.message_func.2",
            "dropout.0.vdropout",
            "dropout.1.vdropout",
            "ff_func.0",
            "ff_func.1",
        )
    }
)


def convert_grnade_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Strictly convert the released raw state dictionary and save its paired Pipeline.
    """

    from .pipeline import GrnadePipeline
    from .processor import GrnadeProcessor

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("the released gRNAde checkpoint must be a raw state dictionary")
    model = GrnadeModel(GrnadeConfig())
    model.load_state_dict(
        convert_grnade_state_dict(cast(Mapping[str, Tensor], checkpoint)),
        strict=True,
        assign=True,
    )
    model.eval()
    GrnadePipeline(
        model=model,
        processor=GrnadeProcessor(),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_grnade_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    r"""
    Translate every released tensor name to the native gRNAde module layout.

    ``W_v``/``W_e`` become the node/edge embeddings,
    ``encoder_layers``/``decoder_layers`` become their native blocks,
    ``W_s`` becomes the token embedding, and ``W_out`` becomes the output
    projection. Empty upstream device-placeholder parameters are not model state.
    """

    converted: dict[str, Tensor] = {}
    for source_key, value in state_dict.items():
        if source_key in _PLACEHOLDER_KEYS:
            if value.numel() != 0:
                raise ValueError(f"gRNAde placeholder parameter {source_key!r} is not empty")
            continue
        target_key = _convert_key(source_key)
        if target_key in converted:
            raise ValueError(f"multiple gRNAde tensors map to {target_key!r}")
        converted[target_key] = value
    return converted


def _convert_key(key: str) -> str:
    top_level = {
        "W_v.": "node_embedding.",
        "W_e.": "edge_embedding.",
        "encoder_layers.": "encoder_blocks.",
        "W_s.": "token_embedding.",
        "decoder_layers.": "decoder_blocks.",
        "W_out.": "output_projection.",
    }
    for source, target in top_level.items():
        if key.startswith(source):
            key = target + key.removeprefix(source)
            break
    else:
        raise KeyError(f"unrecognized gRNAde checkpoint tensor: {key}")

    replacements = (
        (".conv.message_func.", ".message_passing.message."),
        (".norm.", ".norms."),
        (".ff_func.", ".transition."),
        (".wh.", ".hidden_vector_projection."),
        (".ws.", ".scalar_projection."),
        (".wv.", ".output_vector_projection."),
        (".wsv.", ".vector_gate_projection."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def main(argv: list[str] | None = None) -> None:
    r"""
    Convert the released gRNAde checkpoint from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    convert_grnade_checkpoint(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
