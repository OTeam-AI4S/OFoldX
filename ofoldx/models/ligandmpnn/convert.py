r"""
LigandMPNN checkpoint conversion helpers.

Translate the released LigandMPNN ``.pt`` sequence checkpoints into OFoldX-native config objects and state
dicts. Upstream tensor names live only here: the key maps rewrite the terse official names (``W_e``,
``features.embeddings.linear``, ...) into the house vocabulary owned directly by the root Model, without the
former ``backbone.`` checkpoint prefix. The live architecture facts (one width, encoder/decoder depths,
neighbor count) are derived from the state and metadata; ``num_context_atoms`` is written only to the
Processor config. This module owns only the ``ligand_mpnn`` context architecture -- the four active
``ligandmpnn_v_32_{005,010,020,030}_25.pt`` checkpoints, which share one graph and differ only in their
training coordinate-noise label. The released side-chain checkpoint belongs to
`ofoldx.models.mpnn_packer`, and the label-conditioned ``global_label_membrane_mpnn`` /
``per_residue_label_membrane_mpnn`` checkpoints are a distinct architecture (a live node embedding, no
context layers) owned by the frozen §4.9 membrane vertical; their unique tensors are unrecognized here and
raise rather than mis-map onto the ligand context keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
from torch import Tensor

from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import LigandMpnnModel


def convert_ligandmpnn_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Convert an official LigandMPNN sequence checkpoint to native OFoldX tensor names.

    Rewrites every tensor key via [`convert_original_state_dict_key`]
    [ofoldx.models.ligandmpnn.convert.convert_original_state_dict_key], preserving the serialized dtype.

    Args:
        state_dict: The official sequence-model state dict.

    Returns:
        State dict keyed by OFoldX tensor names.

    Raises:
        KeyError: If a checkpoint tensor name is not recognized.
    """

    return {convert_original_state_dict_key(key): value for key, value in state_dict.items()}


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
        KeyError: If the key does not match any known feature, top-level, or per-layer pattern.

    Examples:
        >>> from ofoldx.models.ligandmpnn.convert import convert_original_state_dict_key
        >>> convert_original_state_dict_key("W_e.weight")
        'edge_embedding.weight'
        >>> convert_original_state_dict_key("encoder_layers.0.W1.weight")
        'encoder_layers.0.node_message.input.weight'
    """

    feature_replacements = {
        "features.embeddings.linear.weight": "feature_encoder.relative_position.linear.weight",
        "features.embeddings.linear.bias": "feature_encoder.relative_position.linear.bias",
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
        return _convert_decoder_key(key, "decoder_layers")
    if key.startswith("context_encoder_layers."):
        return _convert_decoder_key(key, "context_layers")
    if key.startswith("y_context_encoder_layers."):
        return _convert_decoder_key(key, "atom_context_layers")
    raise KeyError(f"unrecognized LigandMPNN checkpoint key: {key}")


def _convert_encoder_key(key: str) -> str:
    r"""Translate one official ``encoder_layers.<i>.*`` tensor name to its native name."""

    parts = key.split(".")
    if len(parts) < 4:
        raise KeyError(f"malformed encoder key: {key}")
    layer = parts[1]
    suffix = ".".join(parts[2:])
    replacements = {
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
    if suffix not in replacements:
        raise KeyError(f"unrecognized encoder key: {key}")
    return f"encoder_layers.{layer}.{replacements[suffix]}"


def _convert_decoder_key(key: str, target_prefix: str) -> str:
    r"""Translate one official decoder-style ``*_layers.<i>.*`` tensor name to its native name.

    Shared by the decoder, context-encoder, and atom-context-encoder groups, which use the same sublayer
    layout under different native prefixes.
    """

    parts = key.split(".")
    if len(parts) < 4:
        raise KeyError(f"malformed decoder key: {key}")
    layer = parts[1]
    suffix = ".".join(parts[2:])
    replacements = {
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
    if suffix not in replacements:
        raise KeyError(f"unrecognized decoder key: {key}")
    return f"{target_prefix}.{layer}.{replacements[suffix]}"


def load_ligandmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    r"""Load one released LigandMPNN checkpoint with PyTorch's restricted loader."""

    return torch.load(checkpoint_path, map_location=map_location, weights_only=True, mmap=True)


def _layer_count(state_dict: Mapping[str, Tensor], prefix: str) -> int:
    r"""Count the distinct layer indices under one released ``<prefix>.<i>.*`` group."""

    return len({int(key.split(".", 2)[1]) for key in state_dict if key.startswith(prefix)})


def ligandmpnn_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> LigandMpnnConfig:
    r"""Derive the executable LigandMPNN architecture from a released checkpoint.

    The one live width comes from the token embedding, the encoder/decoder depths from the distinct layer
    indices present in the state, and the neighbor count from checkpoint metadata. Training coordinate-noise
    metadata and the padded context extent are not architecture and are excluded here.
    """

    state_dict = checkpoint["model_state_dict"]
    return LigandMpnnConfig(
        hidden_size=state_dict["W_s.weight"].shape[1],
        num_encoder_layers=_layer_count(state_dict, "encoder_layers."),
        num_decoder_layers=_layer_count(state_dict, "decoder_layers."),
        num_neighbors=checkpoint["num_edges"],
    )


def ligandmpnn_processor_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> LigandMpnnProcessorConfig:
    r"""Read the padded nearest-context extent into the Processor config."""

    return LigandMpnnProcessorConfig(num_context_atoms=checkpoint["atom_context_num"])


def convert_ligandmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
) -> None:
    r"""
    Convert a released LigandMPNN checkpoint into a standard Pipeline repository.

    Derives every live architecture fact from the checkpoint, strict-loads the flat Model with
    ``assign=True`` to preserve the serialized dtype, writes ``num_context_atoms`` to the paired Processor
    config, and saves the paired concrete [`LigandMpnnPipeline`]
    [ofoldx.models.ligandmpnn.pipeline.LigandMpnnPipeline] artifact under the exact ``ofoldx.ligandmpnn`` ID.

    Args:
        checkpoint_path: Path to a released ``ligandmpnn_v_*.pt`` sequence checkpoint.
        output_dir: New repository directory to write.

    Examples:
        >>> convert_ligandmpnn_checkpoint("ligandmpnn_v_32_010_25.pt", "ligandmpnn")  # doctest: +SKIP
    """

    from .pipeline import LigandMpnnPipeline
    from .processor import LigandMpnnProcessor

    checkpoint = load_ligandmpnn_checkpoint(checkpoint_path)
    config = ligandmpnn_config_from_checkpoint(checkpoint)
    model = LigandMpnnModel(config)
    model.load_state_dict(
        convert_ligandmpnn_state_dict(checkpoint["model_state_dict"]),
        strict=True,
        assign=True,
    )
    processor = LigandMpnnProcessor(ligandmpnn_processor_config_from_checkpoint(checkpoint))
    LigandMpnnPipeline(model=model, processor=processor).save(output_dir)
