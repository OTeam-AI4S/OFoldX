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

r"""
ProteinMPNN checkpoint conversion.

Converts released ProteinMPNN checkpoints to native OFoldX repositories.

Notes:
    The supported release includes four full-backbone noise-level checkpoints, four soluble checkpoints,
    three CA-only checkpoints, and the training checkpoint used by PXDesign. Standard and soluble checkpoints
    share the autoregressive operations but differ in training distribution; CA-only checkpoints use reduced
    backbone geometry.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import RequestRef, fail
from ..vocabulary import canonical_word_embeddings
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import PROTEINMPNN_VOCABULARY, ProteinMpnnModel
from .pipeline import ProteinMpnnPipeline
from .processor import ProteinMpnnProcessor

# The first edge-embedding Linear (``features.edge_embedding``) consumes a
# 167-dim edge feature in the Cα graph
# (``positional_embedding_size + num_rbf * 9 + 7`` == ``16 + 144 + 7``) versus
# 416 for the full backbone (``positional_embedding_size + num_rbf * 25``).
# Membrane checkpoints keep a live ``W_v``, so ``W_v`` presence does not identify
# a Cα checkpoint; the 167-wide edge input does.
_CA_EDGE_INPUT_SIZE = 167


def convert_proteinmpnn_model(
    checkpoint_path: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    expected_noise_level: float | None = None,
) -> ProteinMpnnModel:
    r"""
    Strict-load one released ProteinMPNN checkpoint into its native Model.

    Args:
        checkpoint_path: Released ProteinMPNN checkpoint.
        dtype: Optional floating-point dtype for converted weights.
        expected_noise_level: Optional released training-noise identity required
            by a fixed composite model.

    Returns:
        The converted Model. Repository writers pair it with a Processor whose
        ``ca_only`` setting comes from the converted Config.
    """

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if expected_noise_level is not None and float(checkpoint["noise_level"]) != expected_noise_level:
        fail("value_invalid", RequestRef("noise_level", str(checkpoint["noise_level"])), expected=expected_noise_level)
    state_dict = checkpoint["model_state_dict"]
    config = ProteinMpnnConfig(
        hidden_size=state_dict["W_s.weight"].shape[1],
        num_encoder_blocks=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("encoder_layers.")}),
        num_decoder_blocks=len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith("decoder_layers.")}),
        num_neighbors=checkpoint["num_edges"],
        ca_only=state_dict["features.edge_embedding.weight"].shape[1] == _CA_EDGE_INPUT_SIZE,
    )
    converted = convert_proteinmpnn_state_dict(state_dict)
    if dtype is not None:
        converted = {
            key: value.to(dtype=dtype) if value.is_floating_point() else value for key, value in converted.items()
        }
    model = ProteinMpnnModel(config)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def convert_proteinmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert one released ProteinMPNN checkpoint and save its paired Pipeline.

    Saving is local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo via the paired Pipeline's ``save``; no organization or token is assumed
    and nothing is published without an explicit ``repo_id``.
    """

    model = convert_proteinmpnn_model(checkpoint_path)
    processor = ProteinMpnnProcessor(ProteinMpnnProcessorConfig(ca_only=model.config.ca_only))
    save_repository(
        ProteinMpnnPipeline(model=model, processor=processor),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=__name__ + ".convert_proteinmpnn_checkpoint",
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_proteinmpnn_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Translate every live released ProteinMPNN tensor to its native name, with the token embedding and the output
    projection regenerated onto the 72 canonical classes.

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
        "norm2.weight": "feed_forward_norm.weight",
        "norm2.bias": "feed_forward_norm.bias",
        "norm3.weight": "edge_norm.weight",
        "norm3.bias": "edge_norm.bias",
        "W1.weight": "node_message.input_projection.weight",
        "W1.bias": "node_message.input_projection.bias",
        "W2.weight": "node_message.intermediate_projections.0.weight",
        "W2.bias": "node_message.intermediate_projections.0.bias",
        "W3.weight": "node_message.output_projection.weight",
        "W3.bias": "node_message.output_projection.bias",
        "W11.weight": "edge_message.input_projection.weight",
        "W11.bias": "edge_message.input_projection.bias",
        "W12.weight": "edge_message.intermediate_projections.0.weight",
        "W12.bias": "edge_message.intermediate_projections.0.bias",
        "W13.weight": "edge_message.output_projection.weight",
        "W13.bias": "edge_message.output_projection.bias",
        "dense.W_in.weight": "feed_forward.input_projection.weight",
        "dense.W_in.bias": "feed_forward.input_projection.bias",
        "dense.W_out.weight": "feed_forward.output_projection.weight",
        "dense.W_out.bias": "feed_forward.output_projection.bias",
    }
    decoder_sublayers = {
        "norm1.weight": "message_norm.weight",
        "norm1.bias": "message_norm.bias",
        "norm2.weight": "feed_forward_norm.weight",
        "norm2.bias": "feed_forward_norm.bias",
        "W1.weight": "message.input_projection.weight",
        "W1.bias": "message.input_projection.bias",
        "W2.weight": "message.intermediate_projections.0.weight",
        "W2.bias": "message.intermediate_projections.0.bias",
        "W3.weight": "message.output_projection.weight",
        "W3.bias": "message.output_projection.bias",
        "dense.W_in.weight": "feed_forward.input_projection.weight",
        "dense.W_in.bias": "feed_forward.input_projection.bias",
        "dense.W_out.weight": "feed_forward.output_projection.weight",
        "dense.W_out.bias": "feed_forward.output_projection.bias",
    }
    dead_ca_parameters = {
        "W_v.weight",
        "W_v.bias",
        "features.node_embedding.weight",
        "features.norm_nodes.weight",
        "features.norm_nodes.bias",
    }

    converted: dict[str, Tensor] = {}
    ca_only = state_dict["features.edge_embedding.weight"].shape[1] == _CA_EDGE_INPUT_SIZE
    for key, value in state_dict.items():
        if ca_only and key in dead_ca_parameters:
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
                fail("value_invalid", RequestRef("checkpoint", key))
        converted[target] = value
    (converted["token_embedding.weight"],) = canonical_word_embeddings(
        converted["token_embedding.weight"], vocabulary=PROTEINMPNN_VOCABULARY
    )
    converted["output_projection.weight"], converted["output_projection.bias"] = canonical_word_embeddings(
        converted["output_projection.weight"], converted["output_projection.bias"], vocabulary=PROTEINMPNN_VOCABULARY
    )
    return converted


def main(argv: list[str] | None = None) -> None:
    r"""
    Run ProteinMPNN checkpoint conversion from the command line.
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
    convert_proteinmpnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
