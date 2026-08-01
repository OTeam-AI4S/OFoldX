r"""
Convert released NA-MPNN checkpoints to native OFoldX repositories.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor

from .config import NaMpnnConfig
from .model import NaMpnnModel

# NA-MPNN uses a fixed 32-neighbor residue graph; the value is not stored in the
# released state dict, so conversion pins it directly.
_NUM_NEIGHBORS = 32

_ENCODER_SUBLAYERS = {
    "norm1.weight": "node_norm.weight",
    "norm1.bias": "node_norm.bias",
    "norm2.weight": "transition_norm.weight",
    "norm2.bias": "transition_norm.bias",
    "norm3.weight": "edge_norm.weight",
    "norm3.bias": "edge_norm.bias",
    "W1.weight": "node_message.input_projection.weight",
    "W1.bias": "node_message.input_projection.bias",
    "W2.weight": "node_message.hidden_projection.weight",
    "W2.bias": "node_message.hidden_projection.bias",
    "W3.weight": "node_message.output_projection.weight",
    "W3.bias": "node_message.output_projection.bias",
    "W11.weight": "edge_message.input_projection.weight",
    "W11.bias": "edge_message.input_projection.bias",
    "W12.weight": "edge_message.hidden_projection.weight",
    "W12.bias": "edge_message.hidden_projection.bias",
    "W13.weight": "edge_message.output_projection.weight",
    "W13.bias": "edge_message.output_projection.bias",
    "dense.W_in.weight": "transition.input_projection.weight",
    "dense.W_in.bias": "transition.input_projection.bias",
    "dense.W_out.weight": "transition.output_projection.weight",
    "dense.W_out.bias": "transition.output_projection.bias",
}

_DECODER_SUBLAYERS = {
    "norm1.weight": "message_norm.weight",
    "norm1.bias": "message_norm.bias",
    "norm2.weight": "transition_norm.weight",
    "norm2.bias": "transition_norm.bias",
    "W1.weight": "message.input_projection.weight",
    "W1.bias": "message.input_projection.bias",
    "W2.weight": "message.hidden_projection.weight",
    "W2.bias": "message.hidden_projection.bias",
    "W3.weight": "message.output_projection.weight",
    "W3.bias": "message.output_projection.bias",
    "dense.W_in.weight": "transition.input_projection.weight",
    "dense.W_in.bias": "transition.input_projection.bias",
    "dense.W_out.weight": "transition.output_projection.weight",
    "dense.W_out.bias": "transition.output_projection.bias",
}


def convert_nampnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert one released NA-MPNN checkpoint and save its paired Pipeline.

    Saving is local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo via the paired Pipeline's ``save``; no organization or token is assumed
    and nothing is published without an explicit ``repo_id``.
    """

    from .pipeline import NaMpnnPipeline
    from .processor import NaMpnnProcessor

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint["model_state_dict"]
    config = NaMpnnConfig(
        hidden_size=state_dict["W_s.weight"].shape[1],
        feature_size=state_dict["W_v.weight"].shape[1],
        num_encoder_blocks=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("encoder_layers.")}),
        num_decoder_blocks=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("decoder_layers.")}),
        num_neighbors=_NUM_NEIGHBORS,
    )
    model = NaMpnnModel(config)
    model.load_state_dict(convert_nampnn_state_dict(state_dict), strict=True, assign=True)
    model.eval()
    NaMpnnPipeline(model=model, processor=NaMpnnProcessor()).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_nampnn_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Translate every released NA-MPNN tensor name to its native OFoldX name.
    """

    return {_convert_key(key): value for key, value in state_dict.items()}


def _convert_key(key: str) -> str:
    r"""
    Translate one released NA-MPNN tensor name to the folded native name.
    """

    top_level = {
        "features.embeddings.linear.weight": "feature_encoder.relative_position.projection.weight",
        "features.embeddings.linear.bias": "feature_encoder.relative_position.projection.bias",
        "features.node_embedding.weight": "feature_encoder.node_projection.weight",
        "features.norm_nodes.weight": "feature_encoder.node_norm.weight",
        "features.norm_nodes.bias": "feature_encoder.node_norm.bias",
        "features.edge_embedding.weight": "feature_encoder.edge_projection.weight",
        "features.norm_edges.weight": "feature_encoder.edge_norm.weight",
        "features.norm_edges.bias": "feature_encoder.edge_norm.bias",
        "W_e.weight": "edge_embedding.weight",
        "W_e.bias": "edge_embedding.bias",
        "W_v.weight": "node_embedding.weight",
        "W_v.bias": "node_embedding.bias",
        "W_s.weight": "token_embedding.weight",
        "W_out.weight": "output_projection.weight",
        "W_out.bias": "output_projection.bias",
    }
    if key in top_level:
        return top_level[key]
    if key.startswith("encoder_layers."):
        block_index, _, sublayer = key.removeprefix("encoder_layers.").partition(".")
        return f"encoder_blocks.{block_index}.{_ENCODER_SUBLAYERS[sublayer]}"
    if key.startswith("decoder_layers."):
        block_index, _, sublayer = key.removeprefix("decoder_layers.").partition(".")
        return f"decoder_blocks.{block_index}.{_DECODER_SUBLAYERS[sublayer]}"
    raise KeyError(f"unrecognized NA-MPNN checkpoint tensor: {key}")


def main(argv: list[str] | None = None) -> None:
    r"""
    Convert one released NA-MPNN checkpoint from the command line.
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
    convert_nampnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
