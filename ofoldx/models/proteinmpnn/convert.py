r"""Convert released ProteinMPNN checkpoints to native OFoldX repositories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor

from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import ProteinMpnnModel

# The first edge-embedding Linear (``features.edge_embedding``) consumes a
# 167-dim edge feature in the Cα graph
# (``num_positional_embeddings + num_rbf * 9 + 7`` == ``16 + 144 + 7``) versus
# 416 for the full backbone (``num_positional_embeddings + num_rbf * 25``).
# Membrane checkpoints keep a live ``W_v``, so ``W_v`` presence does not identify
# a Cα checkpoint; the 167-wide edge input does.
CA_EDGE_INPUT_WIDTH = 167


def convert_proteinmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert one released ProteinMPNN checkpoint and save its paired Pipeline.

    Saving is local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo via the paired Pipeline's ``save``; no organization or token is assumed
    and nothing is published without an explicit ``repo_id``.
    """

    from .pipeline import ProteinMpnnPipeline
    from .processor import ProteinMpnnProcessor

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint["model_state_dict"]
    config = ProteinMpnnConfig(
        hidden_size=state_dict["W_s.weight"].shape[1],
        num_encoder_blocks=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("encoder_layers.")}),
        num_decoder_blocks=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("decoder_layers.")}),
        num_neighbors=checkpoint["num_edges"],
        ca_only=state_dict["features.edge_embedding.weight"].shape[1] == CA_EDGE_INPUT_WIDTH,
    )
    model = ProteinMpnnModel(config)
    model.load_state_dict(convert_proteinmpnn_state_dict(state_dict), strict=True, assign=True)
    processor = ProteinMpnnProcessor(ProteinMpnnProcessorConfig(ca_only=config.ca_only))
    ProteinMpnnPipeline(model=model, processor=processor).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_proteinmpnn_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""Translate every live released ProteinMPNN tensor to its native name.

    The released Cα checkpoints contain five parameters whose outputs are never
    consumed by the upstream model. Those parameters are the only tensors omitted
    from conversion.
    """

    top_level = {
        "features.embeddings.linear.weight": "feature_encoder.relative_position.projection.weight",
        "features.embeddings.linear.bias": "feature_encoder.relative_position.projection.bias",
        "features.edge_embedding.weight": "feature_encoder.edge_projection.weight",
        "features.norm_edges.weight": "feature_encoder.edge_norm.weight",
        "features.norm_edges.bias": "feature_encoder.edge_norm.bias",
        "W_e.weight": "edge_embedding.weight",
        "W_e.bias": "edge_embedding.bias",
        "W_s.weight": "token_embedding.weight",
        "W_out.weight": "output_projection.weight",
        "W_out.bias": "output_projection.bias",
    }
    encoder_sublayers = {
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
    decoder_sublayers = {
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
    dead_ca_parameters = {
        "W_v.weight",
        "W_v.bias",
        "features.node_embedding.weight",
        "features.norm_nodes.weight",
        "features.norm_nodes.bias",
    }

    converted: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key in dead_ca_parameters:
            continue
        target = top_level.get(key)
        if target is None:
            if key.startswith("encoder_layers."):
                block_index, _, sublayer = key.removeprefix("encoder_layers.").partition(".")
                target = f"encoder_blocks.{block_index}.{encoder_sublayers[sublayer]}"
            elif key.startswith("decoder_layers."):
                block_index, _, sublayer = key.removeprefix("decoder_layers.").partition(".")
                target = f"decoder_blocks.{block_index}.{decoder_sublayers[sublayer]}"
            else:
                raise KeyError(f"unrecognized ProteinMPNN checkpoint tensor: {key}")
        converted[target] = value
    return converted


def main(argv: list[str] | None = None) -> None:
    r"""Run ProteinMPNN checkpoint conversion from the command line."""

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
    convert_proteinmpnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
