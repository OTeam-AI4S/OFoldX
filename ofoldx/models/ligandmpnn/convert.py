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
LigandMPNN checkpoint conversion.

Translate the released LigandMPNN ``.pt`` sequence checkpoints into OFoldX-native config objects and state
dicts. Upstream tensor names live only here: the key maps rewrite the terse official names (``W_e``,
``features.embeddings.linear``, ...) into the house vocabulary owned directly by the root Model, without the
former ``backbone.`` checkpoint prefix. The live architecture facts (one width, encoder/decoder depths,
neighbor count) are derived from the state and metadata; ``num_context_atoms`` is written only to the
Processor config. This module owns only the ``ligand_mpnn`` context architecture -- the four active
``ligandmpnn_v_32_{005,010,020,030}_25.pt`` checkpoints, which share one graph and differ only in their
training coordinate-noise label. The released side-chain checkpoint belongs to
`ofoldx.models.mpnn_packer`, and the label-conditioned ``global_label_membrane_mpnn`` /
``per_residue_label_membrane_mpnn`` checkpoints form a distinct architecture with a live node embedding and
no context layers; their unique tensors are unrecognized here and raise rather than map onto ligand-context
keys.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import RequestRef, fail
from ..vocabulary import canonical_word_embeddings
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import LIGANDMPNN_VOCABULARY, LigandMpnnModel
from .pipeline import LigandMpnnPipeline
from .processor import LigandMpnnProcessor


def convert_ligandmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert a released LigandMPNN checkpoint into a registered concrete Pipeline repository.

    Derives every live architecture fact from the checkpoint, strict-loads the flat Model with
    ``assign=True`` to preserve the serialized dtype, writes ``num_context_atoms`` to the paired Processor
    config, and saves the paired concrete [`LigandMpnnPipeline`]
    [ofoldx.models.ligandmpnn.pipeline.LigandMpnnPipeline] artifact under the exact ``ligandmpnn`` ID.

    Saving is local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo via the paired Pipeline's ``save``; no organization or token is assumed
    and nothing is published without an explicit ``repo_id``.

    Args:
        checkpoint_path: Path to a released ``ligandmpnn_v_*.pt`` sequence checkpoint.
        output_dir: New repository directory to write.
        repo_id: Optional Hub repo id (e.g. ``owner/name``) to publish to; local-only when omitted.
        token: Optional Hugging Face token authenticating the ``repo_id`` upload.
        private: Create the ``repo_id`` repository as private when it does not yet exist.
    """

    checkpoint = load_ligandmpnn_checkpoint(checkpoint_path)
    config = ligandmpnn_config_from_checkpoint(checkpoint)
    model = LigandMpnnModel(config)
    model.load_state_dict(
        convert_ligandmpnn_state_dict(checkpoint["model_state_dict"]),
        strict=True,
        assign=True,
    )
    processor = LigandMpnnProcessor(ligandmpnn_processor_config_from_checkpoint(checkpoint))
    save_repository(
        LigandMpnnPipeline(model=model, processor=processor),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=__name__ + ".convert_ligandmpnn_checkpoint",
        repo_id=repo_id,
        token=token,
        private=private,
    )


def load_ligandmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    r"""
    Load one released LigandMPNN checkpoint with PyTorch's restricted loader.
    """

    return torch.load(checkpoint_path, map_location=map_location, weights_only=True, mmap=True)


def convert_ligandmpnn_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Convert an official LigandMPNN sequence checkpoint to native OFoldX tensor names, with the token embedding and
    the output projection regenerated onto the 72 canonical classes.

    Rewrites every tensor key via [`convert_original_state_dict_key`]
    [ofoldx.models.ligandmpnn.convert.convert_original_state_dict_key], preserving the serialized dtype.

    Args:
        state_dict: The official sequence-model state dict.

    Returns:
        State dict keyed by OFoldX tensor names.

    Raises:
        InvalidRequestError: ``value_invalid`` if a checkpoint tensor name is not recognized.
    """

    converted = {convert_original_state_dict_key(key): value for key, value in state_dict.items()}
    (converted["token_embedding.weight"],) = canonical_word_embeddings(
        converted["token_embedding.weight"], vocabulary=LIGANDMPNN_VOCABULARY
    )
    converted["output_projection.weight"], converted["output_projection.bias"] = canonical_word_embeddings(
        converted["output_projection.weight"], converted["output_projection.bias"], vocabulary=LIGANDMPNN_VOCABULARY
    )
    return converted


def convert_original_state_dict_key(key: str) -> str:
    r"""
    Translate one official LigandMPNN sequence tensor name to its native OFoldX name.

    Handles the feature-encoder table, the top-level projection/embedding table, and the per-layer encoder /
    decoder / context / atom-context groups. This is the single place upstream key jargon is resolved. The
    native names live on the root Model directly; the upstream ``backbone.`` prefix is not preserved.

    Args:
        key: An official checkpoint tensor name (e.g. ``"W_e.weight"`` or ``"encoder_layers.0.W1.weight"``).

    Returns:
        The corresponding OFoldX tensor name.

    Raises:
        InvalidRequestError: ``value_invalid`` if the key matches no known feature, top-level, or per-layer pattern.
    """

    feature_replacements = {
        "features.embeddings.linear.weight": "feature_encoder.relative_position.projection.weight",
        "features.embeddings.linear.bias": "feature_encoder.relative_position.projection.bias",
        "features.edge_embedding.weight": "feature_encoder.edge_projection.weight",
        "features.norm_edges.weight": "feature_encoder.edge_norm.weight",
        "features.norm_edges.bias": "feature_encoder.edge_norm.bias",
        "features.node_project_down.weight": "feature_encoder.atom_projection.weight",
        "features.node_project_down.bias": "feature_encoder.atom_projection.bias",
        "features.norm_nodes.weight": "feature_encoder.atom_norm.weight",
        "features.norm_nodes.bias": "feature_encoder.atom_norm.bias",
        "features.type_linear.weight": "feature_encoder.element_projection.weight",
        "features.type_linear.bias": "feature_encoder.element_projection.bias",
        "features.y_nodes.weight": "feature_encoder.context_node_projection.weight",
        "features.y_edges.weight": "feature_encoder.context_edge_projection.weight",
        "features.norm_y_edges.weight": "feature_encoder.context_edge_norm.weight",
        "features.norm_y_edges.bias": "feature_encoder.context_edge_norm.bias",
        "features.norm_y_nodes.weight": "feature_encoder.context_node_norm.weight",
        "features.norm_y_nodes.bias": "feature_encoder.context_node_norm.bias",
    }
    if key in feature_replacements:
        return feature_replacements[key]
    top_level_replacements = {
        "W_e.weight": "edge_embedding.weight",
        "W_e.bias": "edge_embedding.bias",
        "W_s.weight": "token_embedding.weight",
        "W_v.weight": "atom_context_embedding.weight",
        "W_v.bias": "atom_context_embedding.bias",
        "W_c.weight": "residue_context_input.weight",
        "W_c.bias": "residue_context_input.bias",
        "W_nodes_y.weight": "atom_node_embedding.weight",
        "W_nodes_y.bias": "atom_node_embedding.bias",
        "W_edges_y.weight": "atom_edge_embedding.weight",
        "W_edges_y.bias": "atom_edge_embedding.bias",
        "V_C.weight": "context_output.weight",
        "V_C_norm.weight": "context_norm.weight",
        "V_C_norm.bias": "context_norm.bias",
        "W_out.weight": "output_projection.weight",
        "W_out.bias": "output_projection.bias",
    }
    if key in top_level_replacements:
        return top_level_replacements[key]
    if key.startswith("encoder_layers."):
        return _convert_encoder_key(key)
    if key.startswith("decoder_layers."):
        return _convert_decoder_key(key, "decoder_blocks")
    if key.startswith("context_encoder_layers."):
        return _convert_decoder_key(key, "context_blocks")
    if key.startswith("y_context_encoder_layers."):
        return _convert_decoder_key(key, "atom_context_blocks")
    fail("value_invalid", RequestRef("checkpoint", key))


def _convert_encoder_key(key: str) -> str:
    r"""
    Translate one official ``encoder_layers.<i>.*`` tensor name to its native name.
    """

    parts = key.split(".")
    if len(parts) < 4:
        fail("value_invalid", RequestRef("checkpoint", key))
    block_index = parts[1]
    suffix = ".".join(parts[2:])
    replacements = {
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
    if suffix not in replacements:
        fail("value_invalid", RequestRef("checkpoint", key))
    return f"encoder_blocks.{block_index}.{replacements[suffix]}"


def _convert_decoder_key(key: str, target_prefix: str) -> str:
    r"""
    Translate one official decoder-style ``*_layers.<i>.*`` tensor name to its native name.

    Shared by the decoder, context-encoder, and atom-context-encoder groups, which use the same sublayer
    layout under different native prefixes.
    """

    parts = key.split(".")
    if len(parts) < 4:
        fail("value_invalid", RequestRef("checkpoint", key))
    block_index = parts[1]
    suffix = ".".join(parts[2:])
    replacements = {
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
    if suffix not in replacements:
        fail("value_invalid", RequestRef("checkpoint", key))
    return f"{target_prefix}.{block_index}.{replacements[suffix]}"


def ligandmpnn_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> LigandMpnnConfig:
    r"""
    Derive the executable LigandMPNN architecture from a released checkpoint.

    The one live width comes from the token embedding, the encoder/decoder depths from the distinct layer
    indices present in the state, and the neighbor count from checkpoint metadata. Training coordinate-noise
    metadata and the padded context extent are not architecture and are excluded here.
    """

    state_dict = checkpoint["model_state_dict"]
    return LigandMpnnConfig(
        hidden_size=state_dict["W_s.weight"].shape[1],
        num_encoder_blocks=_block_count(state_dict, "encoder_layers."),
        num_decoder_blocks=_block_count(state_dict, "decoder_layers."),
        num_neighbors=checkpoint["num_edges"],
    )


def _block_count(state_dict: Mapping[str, Tensor], prefix: str) -> int:
    r"""
    Count the distinct layer indices under one released ``<prefix>.<i>.*`` group.
    """

    return len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith(prefix)})


def ligandmpnn_processor_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> LigandMpnnProcessorConfig:
    r"""
    Read the padded nearest-context extent into the Processor config.
    """

    return LigandMpnnProcessorConfig(num_context_atoms=checkpoint["atom_context_num"])


def main(argv: list[str] | None = None) -> None:
    r"""
    Convert one released LigandMPNN checkpoint from the command line.
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
    convert_ligandmpnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
