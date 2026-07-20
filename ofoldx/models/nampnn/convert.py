r"""Convert released NA-MPNN checkpoints to native OFoldX repositories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor

from .config import NaMpnnConfig

# NA-MPNN uses a fixed 32-neighbor residue graph; the value is not stored in the
# released state dict, so conversion pins it directly.
_K_NEIGHBORS = 32


def convert_nampnn_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""Translate every released NA-MPNN tensor name to its native OFoldX name."""

    return {_convert_key(key): value for key, value in state_dict.items()}


def convert_nampnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> None:
    r"""Convert one released NA-MPNN checkpoint and save its paired Pipeline."""

    from .model import NaMpnnModel
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
        vocab_size=state_dict["W_out.weight"].shape[0],
        hidden_size=state_dict["W_s.weight"].shape[1],
        feature_size=state_dict["W_v.weight"].shape[1],
        num_encoder_layers=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("encoder_layers.")}),
        num_decoder_layers=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("decoder_layers.")}),
        k_neighbors=_K_NEIGHBORS,
    )
    model = NaMpnnModel(config)
    model.load_state_dict(convert_nampnn_state_dict(state_dict), strict=True, assign=True)
    NaMpnnPipeline(model=model, processor=NaMpnnProcessor()).save(output_dir)


def _convert_key(key: str) -> str:
    r"""Translate one released NA-MPNN tensor name to the folded native name."""

    top_level = {
        "features.embeddings.linear.weight": "feature_encoder.relative_position.linear.weight",
        "features.embeddings.linear.bias": "feature_encoder.relative_position.linear.bias",
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
        layer, _, sublayer = key.removeprefix("encoder_layers.").partition(".")
        return f"encoder_layers.{layer}.{_ENCODER_SUBLAYERS[sublayer]}"
    if key.startswith("decoder_layers."):
        layer, _, sublayer = key.removeprefix("decoder_layers.").partition(".")
        return f"decoder_layers.{layer}.{_DECODER_SUBLAYERS[sublayer]}"
    raise KeyError(f"unrecognized NA-MPNN checkpoint tensor: {key}")


_ENCODER_SUBLAYERS = {
    "norm1.weight": "node_norm.weight",
    "norm1.bias": "node_norm.bias",
    "norm2.weight": "feed_forward_norm.weight",
    "norm2.bias": "feed_forward_norm.bias",
    "norm3.weight": "edge_norm.weight",
    "norm3.bias": "edge_norm.bias",
    "W1.weight": "node_message.input.weight",
    "W1.bias": "node_message.input.bias",
    "W2.weight": "node_message.middle.weight",
    "W2.bias": "node_message.middle.bias",
    "W3.weight": "node_message.output.weight",
    "W3.bias": "node_message.output.bias",
    "W11.weight": "edge_message.input.weight",
    "W11.bias": "edge_message.input.bias",
    "W12.weight": "edge_message.middle.weight",
    "W12.bias": "edge_message.middle.bias",
    "W13.weight": "edge_message.output.weight",
    "W13.bias": "edge_message.output.bias",
    "dense.W_in.weight": "feed_forward.input.weight",
    "dense.W_in.bias": "feed_forward.input.bias",
    "dense.W_out.weight": "feed_forward.output.weight",
    "dense.W_out.bias": "feed_forward.output.bias",
}
_DECODER_SUBLAYERS = {
    "norm1.weight": "message_norm.weight",
    "norm1.bias": "message_norm.bias",
    "norm2.weight": "feed_forward_norm.weight",
    "norm2.bias": "feed_forward_norm.bias",
    "W1.weight": "message.input.weight",
    "W1.bias": "message.input.bias",
    "W2.weight": "message.middle.weight",
    "W2.bias": "message.middle.bias",
    "W3.weight": "message.output.weight",
    "W3.bias": "message.output.bias",
    "dense.W_in.weight": "feed_forward.input.weight",
    "dense.W_in.bias": "feed_forward.input.bias",
    "dense.W_out.weight": "feed_forward.output.weight",
    "dense.W_out.bias": "feed_forward.output.bias",
}


def main(argv: list[str] | None = None) -> None:
    r"""Convert one released NA-MPNN checkpoint from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    convert_nampnn_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
