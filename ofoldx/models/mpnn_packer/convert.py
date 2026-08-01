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
MPNN Packer checkpoint conversion.

Converts released ``ligandmpnn_sc`` checkpoints to MpnnPacker repositories.
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
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import MPNN_PACKER_VOCABULARY, MpnnPackerModel
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

_PACKER_TOP_LEVEL = {
    "W_e": "edge_embedding",
    "W_v": "node_embedding",
    "W_f": "decode_edge_embedding",
    "W_v_sc": "sidechain_node_embedding",
    "linear_down": "decoder_node_projection",
    "W_torsions": "torsion_projection",
    "W_c": "residue_context_input",
    "W_e_context": "context_edge_embedding",
    "W_nodes_y": "atom_node_embedding",
    "W_edges_y": "atom_edge_embedding",
    "V_C": "context_output",
    "V_C_norm": "context_norm",
    "features.enc_edge_embedding": "feature_encoder.edge_projection",
    "features.enc_norm_edges": "feature_encoder.edge_norm",
    "features.enc_node_embedding": "feature_encoder.node_projection",
    "features.enc_norm_nodes": "feature_encoder.node_norm",
    "features.W_XY_project_down1": "feature_encoder.context_down_projection",
    "features.dec_edge_embedding1": "feature_encoder.decode_edge_projection",
    "features.dec_norm_edges1": "feature_encoder.decode_edge_norm",
    "features.dec_node_embedding1": "feature_encoder.decode_node_projection",
    "features.dec_norm_nodes1": "feature_encoder.decode_node_norm",
    "features.node_project_down": "feature_encoder.context_node_projection",
    "features.norm_nodes": "feature_encoder.context_node_norm",
    "features.type_linear": "feature_encoder.element_projection",
    "features.y_nodes": "feature_encoder.atom_node_projection",
    "features.y_edges": "feature_encoder.atom_edge_projection",
    "features.norm_y_edges": "feature_encoder.atom_edge_norm",
    "features.norm_y_nodes": "feature_encoder.atom_node_norm",
    "features.positional_embeddings.linear": "feature_encoder.relative_position.projection",
}
_PACKER_ENCODER_SUBLAYER = {
    "norm1": "node_norm",
    "norm2": "feed_forward_norm",
    "norm3": "edge_norm",
    "W1": "node_message.input_projection",
    "W2": "node_message.intermediate_projections.0",
    "W3": "node_message.output_projection",
    "W11": "edge_message.input_projection",
    "W12": "edge_message.intermediate_projections.0",
    "W13": "edge_message.output_projection",
    "dense.W_in": "feed_forward.input_projection",
    "dense.W_out": "feed_forward.output_projection",
}
_PACKER_DECODER_SUBLAYER = {
    "norm1": "message_norm",
    "norm2": "feed_forward_norm",
    "W1": "message.input_projection",
    "W2": "message.intermediate_projections.0",
    "W3": "message.output_projection",
    "dense.W_in": "feed_forward.input_projection",
    "dense.W_out": "feed_forward.output_projection",
}
_PACKER_BLOCK_GROUPS = {
    "encoder_layers": ("encoder_blocks", _PACKER_ENCODER_SUBLAYER),
    "decoder_layers": ("decoder_blocks", _PACKER_DECODER_SUBLAYER),
    "context_encoder_layers": ("context_blocks", _PACKER_DECODER_SUBLAYER),
    "y_context_encoder_layers": ("atom_context_blocks", _PACKER_DECODER_SUBLAYER),
}


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
    """

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
    model.load_state_dict(state_dict, strict=True, assign=True)
    processor = MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=config.num_context_atoms))
    save_repository(
        MpnnPackerPipeline(model=model, processor=processor),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=__name__ + ".convert_mpnn_packer_checkpoint",
        repo_id=repo_id,
        token=token,
        private=private,
    )


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
        State dict with OFoldX MpnnPacker tensor names, the residue one-hot inputs regenerated onto the 72 canonical
        classes.
    """

    converted = {_convert_packer_key(key): value for key, value in state_dict.items()}
    node = "feature_encoder.node_projection.weight"
    (converted[node],) = canonical_word_embeddings(converted[node], vocabulary=MPNN_PACKER_VOCABULARY, dim=-1)
    edge = "feature_encoder.decode_edge_projection.weight"
    size = len(MPNN_PACKER_VOCABULARY)
    tensor = converted[edge]
    own, neighbour = canonical_word_embeddings(
        tensor[..., -2 * size : -size], tensor[..., -size:], vocabulary=MPNN_PACKER_VOCABULARY, dim=-1
    )
    converted[edge] = torch.cat((tensor[..., : -2 * size], own, neighbour), dim=-1)
    return converted


def _convert_packer_key(key: str) -> str:
    r"""
    Translate one official MpnnPacker tensor name to the native OFoldX name.

    Args:
        key: An official ``ligandmpnn_sc`` MpnnPacker tensor name.

    Returns:
        The corresponding native MpnnPacker tensor name.

    """

    for prefix, (target, sublayer) in _PACKER_BLOCK_GROUPS.items():
        if key.startswith(prefix + "."):
            parts = key.split(".")
            remainder = ".".join(parts[2:])
            for original, replacement in sublayer.items():
                if remainder.startswith(original + "."):
                    return f"{target}.{parts[1]}.{replacement}{remainder[len(original):]}"
            fail("value_invalid", RequestRef("checkpoint", key))
    for original, replacement in _PACKER_TOP_LEVEL.items():
        if key in (f"{original}.weight", f"{original}.bias"):
            return replacement + key[len(original) :]
    fail("value_invalid", RequestRef("checkpoint", key))


def main(argv: list[str] | None = None) -> None:
    r"""
    Convert one released MPNN Packer checkpoint from the command line.
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
    convert_mpnn_packer_checkpoint(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
