r"""Convert released ``ligandmpnn_sc`` checkpoints to MpnnPacker repositories."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

import torch
from torch import Tensor

from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import MpnnPackerModel

_PACKER_TOP_LEVEL = {
    "W_e": "edge_embedding",
    "W_v": "node_embedding",
    "W_f": "decode_edge_embedding",
    "W_v_sc": "side_chain_node_embedding",
    "linear_down": "linear_down",
    "W_torsions": "torsion_projection",
    "W_c": "residue_context_input",
    "W_e_context": "context_edge_embedding",
    "W_nodes_y": "atom_node_embedding",
    "W_edges_y": "atom_edge_embedding",
    "V_C": "context_output",
    "V_C_norm": "context_norm",
    "features.enc_edge_embedding": "features.edge_projection",
    "features.enc_norm_edges": "features.edge_norm",
    "features.enc_node_embedding": "features.node_projection",
    "features.enc_norm_nodes": "features.node_norm",
    "features.W_XY_project_down1": "features.context_down_projection",
    "features.dec_edge_embedding1": "features.decode_edge_projection",
    "features.dec_norm_edges1": "features.decode_edge_norm",
    "features.dec_node_embedding1": "features.decode_node_projection",
    "features.dec_norm_nodes1": "features.decode_node_norm",
    "features.node_project_down": "features.context_node_projection",
    "features.norm_nodes": "features.context_node_norm",
    "features.type_linear": "features.element_projection",
    "features.y_nodes": "features.atom_node_projection",
    "features.y_edges": "features.atom_edge_projection",
    "features.norm_y_edges": "features.atom_edge_norm",
    "features.norm_y_nodes": "features.atom_node_norm",
    "features.positional_embeddings.linear": "features.relative_position.linear",
}
_PACKER_ENCODER_SUBLAYER = {
    "norm1": "node_norm",
    "norm2": "feed_forward_norm",
    "norm3": "edge_norm",
    "W1": "node_message.input",
    "W2": "node_message.middle",
    "W3": "node_message.output",
    "W11": "edge_message.input",
    "W12": "edge_message.middle",
    "W13": "edge_message.output",
    "dense.W_in": "feed_forward.input",
    "dense.W_out": "feed_forward.output",
}
_PACKER_DECODER_SUBLAYER = {
    "norm1": "message_norm",
    "norm2": "feed_forward_norm",
    "W1": "message.input",
    "W2": "message.middle",
    "W3": "message.output",
    "dense.W_in": "feed_forward.input",
    "dense.W_out": "feed_forward.output",
}
_PACKER_LAYER_GROUPS = {
    "encoder_layers": ("encoder_layers", _PACKER_ENCODER_SUBLAYER),
    "decoder_layers": ("decoder_layers", _PACKER_DECODER_SUBLAYER),
    "context_encoder_layers": ("context_layers", _PACKER_DECODER_SUBLAYER),
    "y_context_encoder_layers": ("atom_context_layers", _PACKER_DECODER_SUBLAYER),
}


def convert_mpnn_packer_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    r"""
    Convert an official LigandMPNN side-chain MpnnPacker checkpoint to native OFoldX keys.

    The MpnnPacker is the ``ligandmpnn_sc`` checkpoint family (Von Mises torsion weights), distinct from
    the sequence backbone. Load the result into
    [`MpnnPackerModel`][ofoldx.models.mpnn_packer.model.MpnnPackerModel].

    Args:
        state_dict: The official side-chain packer state dict.
    Returns:
        State dict with OFoldX MpnnPacker tensor names.

    Raises:
        KeyError: If a checkpoint tensor name is not recognized.
    """

    return {_convert_packer_key(key): value for key, value in state_dict.items()}


def _convert_packer_key(key: str) -> str:
    r"""Translate one official MpnnPacker tensor name to the native OFoldX name.

    Args:
        key: An official ``ligandmpnn_sc`` MpnnPacker tensor name.

    Returns:
        The corresponding native MpnnPacker tensor name.

    Raises:
        KeyError: If the key matches no known MpnnPacker layer-group or top-level pattern.
    """

    for prefix, (target, sublayer) in _PACKER_LAYER_GROUPS.items():
        if key.startswith(prefix + "."):
            parts = key.split(".")
            remainder = ".".join(parts[2:])
            for original, replacement in sublayer.items():
                if remainder.startswith(original + "."):
                    return f"{target}.{parts[1]}.{replacement}{remainder[len(original):]}"
            raise KeyError(f"unrecognized MpnnPacker sublayer key: {key}")
    for original, replacement in _PACKER_TOP_LEVEL.items():
        if key in (f"{original}.weight", f"{original}.bias"):
            return replacement + key[len(original) :]
    raise KeyError(f"unrecognized MpnnPacker checkpoint key: {key}")


def convert_mpnn_packer_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert an official ``ligandmpnn_sc`` checkpoint into a standard packer repository.

    Saving is local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo via the paired Pipeline's ``save``; no organization or token is assumed
    and nothing is published without an explicit ``repo_id``.

    Args:
        checkpoint_path: Path to an official ``ligandmpnn_sc_v_*.pt`` checkpoint.
        output_dir: New repository directory to write.
        repo_id: Optional Hub repo id (e.g. ``owner/name``) to publish to; local-only when omitted.
        token: Optional Hugging Face token authenticating the ``repo_id`` upload.
        private: Create the ``repo_id`` repository as private when it does not yet exist.

    Examples:
        >>> convert_mpnn_packer_checkpoint(  # doctest: +SKIP
        ...     "ligandmpnn_sc_v_32_002_16.pt", "mpnn-packer"
        ... )
    """

    from .pipeline import MpnnPackerPipeline
    from .processor import MpnnPackerProcessor

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    config = MpnnPackerConfig(
        num_neighbors=checkpoint["num_edges"],
        num_context_atoms=checkpoint["atom_context_num"],
    )
    model = MpnnPackerModel(config)
    state_dict = convert_mpnn_packer_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    processor = MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=config.num_context_atoms))
    MpnnPackerPipeline(model=model, processor=processor).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )
