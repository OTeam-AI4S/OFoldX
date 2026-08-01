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
OInvFold checkpoint conversion.

Converts the four released OInvFold Lightning checkpoints.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import RequestRef, fail
from ..vocabulary import canonical_word_embeddings
from .config import (
    OInvFoldLigandConfig,
    OInvFoldPolymerConfig,
    OInvFoldPolymerProcessorConfig,
)
from .model import OInvFoldLigandModel, OInvFoldPolymerModel
from .pipeline import OInvFoldLigandPipeline, OInvFoldPolymerPipeline
from .processor import OInvFoldLigandProcessor, OInvFoldPolymerProcessor

# The released polymer output columns (``ACDEFGHIKLMNPQRSTVWY`` then nucleic A, C, G, T, U) read by each canonical
# class: RNA and DNA share the A, C and G columns, and inosine reads A as the released preparation does.
_CHECKPOINT_ROWS: tuple[tuple[str, int], ...] = (
    *((letter, column) for column, letter in enumerate("ACDEFGHIKLMNPQRSTVWY")),
    *((f"{kind}_{letter}", 20 + offset) for offset, letter in enumerate("ACG") for kind in ("RNA", "DNA")),
    ("RNA_I", 20),
    ("DNA_I", 20),
    ("DNA_T", 23),
    ("RNA_U", 24),
)


def convert_oinvfold_protein_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the released protein OInvFold checkpoint.
    """

    _convert_polymer_checkpoint(
        checkpoint_path,
        output_dir,
        modality="protein",
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_oinvfold_dna_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the released DNA OInvFold checkpoint.
    """

    _convert_polymer_checkpoint(
        checkpoint_path,
        output_dir,
        modality="dna",
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_oinvfold_rna_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the released RNA OInvFold checkpoint.
    """

    _convert_polymer_checkpoint(
        checkpoint_path,
        output_dir,
        modality="rna",
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_oinvfold_ligand_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the released ligand OInvFold checkpoint.
    """

    model = OInvFoldLigandModel(OInvFoldLigandConfig())
    state_dict = _lightning_state_dict(checkpoint_path)
    model.load_state_dict(
        _live_state_dict(
            state_dict,
            model.state_dict(),
            _ligand_dead_source_keys(model.config.num_encoder_blocks),
            architecture="ligand",
            config=model.config,
        ),
        strict=True,
        assign=True,
    )
    model.eval()
    save_repository(
        OInvFoldLigandPipeline(
            model=model,
            processor=OInvFoldLigandProcessor(),
        ),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=__name__ + ".convert_oinvfold_ligand_checkpoint",
        options={"modality": "ligand"},
        repo_id=repo_id,
        token=token,
        private=private,
    )


def _convert_polymer_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    modality: Literal["protein", "dna", "rna"],
    repo_id: str | None,
    token: str | None,
    private: bool,
) -> None:

    model = OInvFoldPolymerModel(OInvFoldPolymerConfig(modality=modality))
    state_dict = _lightning_state_dict(checkpoint_path)
    model.load_state_dict(
        convert_oinvfold_polymer_classes(
            _live_state_dict(
                state_dict,
                model.state_dict(),
                _polymer_omitted_source_keys(model.config.num_encoder_blocks),
                architecture="polymer",
                config=model.config,
            )
        ),
        strict=True,
        assign=True,
    )
    model.eval()
    processor = OInvFoldPolymerProcessor(OInvFoldPolymerProcessorConfig(modality=modality))
    save_repository(
        OInvFoldPolymerPipeline(model=model, processor=processor),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=f"{__name__}.convert_oinvfold_{modality}_checkpoint",
        options={"modality": modality},
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_oinvfold_polymer_classes(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Regenerate the polymer class axis of a translated state dict onto the 72 canonical classes.

    The tied token embedding with its output bias and the classifier projection are indexed by the released 25 output
    columns; each canonical class of ``_CHECKPOINT_ROWS`` takes its column's row, every other class is regenerated by
    ``canonical_word_embeddings``.
    """

    vocabulary = tuple(name for name, _ in _CHECKPOINT_ROWS)
    columns = torch.tensor([column for _, column in _CHECKPOINT_ROWS])
    converted = dict(state_dict)
    for names in (
        ("autoregressive_decoder.token_embedding.weight", "autoregressive_decoder.output_bias"),
        ("decoder.output_projection.weight", "decoder.output_projection.bias"),
    ):
        rows = (converted[name].index_select(0, columns) for name in names)
        converted.update(zip(names, canonical_word_embeddings(*rows, vocabulary=vocabulary)))
    return converted


def _lightning_state_dict(
    checkpoint_path: str | PathLike[str],
) -> Mapping[str, Tensor]:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = checkpoint["state_dict"]
    return cast(Mapping[str, Tensor], state_dict)


def _live_state_dict(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    omitted_source_keys: frozenset[str],
    *,
    architecture: Literal["polymer", "ligand"],
    config: OInvFoldPolymerConfig | OInvFoldLigandConfig,
) -> dict[str, Tensor]:
    prefixed = tuple(key.startswith("model.") for key in source)
    if all(prefixed):
        normalized = {key.removeprefix("model."): value for key, value in source.items()}
    elif not any(prefixed):
        normalized = dict(source)
    else:
        fail("value_invalid", RequestRef("checkpoint", "mixed prefixed and unprefixed state keys"))

    source_keys = set(normalized)
    missing_omissions = omitted_source_keys - source_keys
    converted = {
        _target_key(key, architecture, config): value
        for key, value in normalized.items()
        if key not in omitted_source_keys
    }
    target_keys = set(target)
    unexpected = set(converted) - target_keys
    missing = target_keys - set(converted)
    if unexpected:
        fail("value_invalid", RequestRef("checkpoint", "unrecognized tensors"), keys=sorted(unexpected))
    if missing_omissions:
        fail("value_invalid", RequestRef("checkpoint", "missing source-only tensors"), keys=sorted(missing_omissions))
    if missing:
        fail("value_invalid", RequestRef("checkpoint", "missing live tensors"), keys=sorted(missing))
    return {key: converted[key] for key in target}


def _target_key(
    key: str,
    architecture: Literal["polymer", "ligand"],
    config: OInvFoldPolymerConfig | OInvFoldLigandConfig,
) -> str:
    for embedding in ("node_embedding", "edge_embedding"):
        translated = _normalized_relu_target_key(
            key,
            source_prefix=f"{embedding}.",
            target_prefix=f"{embedding}.",
            num_layers=2,
        )
        if translated is not None:
            return translated

    encoder_prefix = "encoder.encoder_layers."
    if key.startswith(encoder_prefix):
        block_index_text, separator, block_key = key.removeprefix(encoder_prefix).partition(".")
        if not separator or not block_index_text.isdecimal():
            return key
        block_index = int(block_index_text)
        if block_index >= config.num_encoder_blocks:
            return key

        source_prefix = f"{encoder_prefix}{block_index}."
        target_prefix = f"encoder.blocks.{block_index}."
        translated = _normalized_relu_target_key(
            key,
            source_prefix=f"{source_prefix}geofeat.we_condition.",
            target_prefix=f"{target_prefix}geometric_edge_update.geometry_feed_forward.",
            num_layers=config.num_geometric_layers,
        )
        if translated is not None:
            return translated
        translated = _gelu_feed_forward_target_key(
            key,
            source_prefix=f"{source_prefix}update_edge.W.",
            target_prefix=f"{target_prefix}edge_update.feed_forward.",
            num_layers=config.num_edge_layers,
        )
        if translated is not None:
            return translated

        replacements = (
            ("mask_token.", "mask_token."),
            (
                "geofeat.virtual_atom.",
                "geometric_edge_update.node_vector_projection.",
            ),
            (
                "geofeat.virtual_direct.",
                "geometric_edge_update.edge_vector_projection.",
            ),
            (
                "geofeat.MergeEG.",
                "geometric_edge_update.output_projection.",
            ),
            ("attention.W_V.0.", "attention.value_projection."),
            ("attention.Bias.0.", "attention.logit_feed_forward.input_projection."),
            ("attention.Bias.2.", "attention.logit_feed_forward.intermediate_projections.0."),
            ("attention.Bias.4.", "attention.logit_feed_forward.output_projection."),
            ("attention.W_O.", "attention.output_projection."),
            ("attention.gate.", "attention.output_gate."),
            ("update_node.dense.0.", "node_update.transition_input_norm."),
            (
                "update_node.dense.1.",
                "node_update.feed_forward.input_projection.",
            ),
            (
                "update_node.dense.3.",
                "node_update.feed_forward.output_projection.",
            ),
            (
                "update_node.dense.4.",
                "node_update.transition_output_norm.",
            ),
            (
                "update_node.V_MLP_g.0.",
                "node_update.graph_gate_feed_forward.input_projection.",
            ),
            (
                "update_node.V_MLP_g.2.",
                "node_update.graph_gate_feed_forward.intermediate_projections.0.",
            ),
            (
                "update_node.V_MLP_g.4.",
                "node_update.graph_gate_feed_forward.output_projection.",
            ),
            ("update_edge.norm.", "edge_update.output_norm."),
        )
        for source, destination in replacements:
            if block_key.startswith(source):
                return target_prefix + destination + block_key.removeprefix(source)
        return key

    if key.startswith("encoder.s."):
        return "encoder.block_gate." + key.removeprefix("encoder.s.")

    decoder_replacements: tuple[tuple[str, str], ...]
    if architecture == "polymer":
        if key == "decoder.bos_embed":
            return "autoregressive_decoder.bos_embedding"
        autoregressive_replacements = (
            ("decoder.token_emb.", "autoregressive_decoder.token_embedding."),
            ("decoder.pos_emb.", "autoregressive_decoder.position_embedding."),
            ("decoder.struct_proj.", "autoregressive_decoder.structure_projection."),
        )
        for source, destination in autoregressive_replacements:
            if key.startswith(source):
                return destination + key.removeprefix(source)
        if key == "decoder.readout.bias":
            return "autoregressive_decoder.output_bias"
        decoder_block_prefix = "decoder.blocks."
        if key.startswith(decoder_block_prefix):
            block_index_text, separator, block_key = key.removeprefix(decoder_block_prefix).partition(".")
            if not separator or not block_index_text.isdecimal():
                return key
            target_prefix = f"autoregressive_decoder.blocks.{block_index_text}."
            block_replacements = (
                ("ln1.", "attention_norm."),
                ("self_attn.", "attention."),
                ("ln2.", "transition_norm."),
                ("ffn.0.", "transition_input_projection."),
                ("ffn.3.", "transition_output_projection."),
            )
            for source, destination in block_replacements:
                if block_key.startswith(source):
                    return target_prefix + destination + block_key.removeprefix(source)
            return key
        decoder_replacements = (
            ("decoder_mlp.ffn.0.", "decoder.transition_norm."),
            (
                "decoder_mlp.ffn.1.",
                "decoder.transition_input_projection.",
            ),
            (
                "decoder_mlp.ffn.4.",
                "decoder.transition_output_projection.",
            ),
            ("decoder_mlp.readout.", "decoder.output_projection."),
        )
    else:
        decoder_replacements = (("decoder.readout.", "decoder.output_projection."),)
    for source, destination in decoder_replacements:
        if key.startswith(source):
            return destination + key.removeprefix(source)
    return key


def _normalized_relu_target_key(
    key: str,
    *,
    source_prefix: str,
    target_prefix: str,
    num_layers: int,
) -> str | None:
    if not key.startswith(source_prefix):
        return None
    sequence_index_text, separator, suffix = key.removeprefix(source_prefix).partition(".")
    if not separator or not sequence_index_text.isdecimal():
        return key
    sequence_index = int(sequence_index_text)
    num_intermediate_layers = max(num_layers - 2, 0)
    # Upstream stores each hidden stage as Linear, BatchNorm, Dropout, ReLU.
    sequence_members = {
        0: "input_projection",
        1: "input_norm",
        4 + 4 * num_intermediate_layers: "output_projection",
    }
    for layer_index in range(num_intermediate_layers):
        sequence_members[4 + 4 * layer_index] = f"intermediate_projections.{layer_index}"
        sequence_members[5 + 4 * layer_index] = f"intermediate_norms.{layer_index}"
    member = sequence_members.get(sequence_index)
    if member is None:
        return key
    return f"{target_prefix}{member}.{suffix}"


def _gelu_feed_forward_target_key(
    key: str,
    *,
    source_prefix: str,
    target_prefix: str,
    num_layers: int,
) -> str | None:
    if not key.startswith(source_prefix):
        return None
    sequence_index_text, separator, suffix = key.removeprefix(source_prefix).partition(".")
    if not separator or not sequence_index_text.isdecimal():
        return key
    sequence_index = int(sequence_index_text)
    num_intermediate_layers = max(num_layers - 2, 0)
    # Upstream stores each hidden stage as Linear, Dropout, GELU.
    sequence_members = {
        0: "input_projection",
        3 + 3 * num_intermediate_layers: "output_projection",
    }
    for layer_index in range(num_intermediate_layers):
        sequence_members[3 + 3 * layer_index] = f"intermediate_projections.{layer_index}"
    member = sequence_members.get(sequence_index)
    if member is None:
        return key
    return f"{target_prefix}{member}.{suffix}"


# The released checkpoints retain quaternion predictors whose outputs are never
# consumed by either encoder equation.
def _common_dead_source_keys(num_encoder_blocks: int) -> frozenset[str]:
    return frozenset(
        key
        for block_index in range(num_encoder_blocks)
        for key in (
            f"encoder.encoder_layers.{block_index}.update_edge.pred_quat.weight",
            f"encoder.encoder_layers.{block_index}.update_edge.pred_quat.bias",
        )
    )


def _polymer_dead_source_keys(num_encoder_blocks: int) -> frozenset[str]:
    return _common_dead_source_keys(num_encoder_blocks) | frozenset(
        {
            "chain_embeddings.weight",
        }
    )


def _polymer_omitted_source_keys(num_encoder_blocks: int) -> frozenset[str]:
    # ``decoder.readout.weight`` aliases ``decoder.token_emb.weight`` upstream.
    # The local decoder reads the embedding weight directly so ``assign=True``
    # cannot break the tie; this is a redundant source alias, not dead state.
    tied_source_aliases = frozenset({"decoder.readout.weight"})
    return _polymer_dead_source_keys(num_encoder_blocks) | tied_source_aliases


def _ligand_dead_source_keys(num_encoder_blocks: int) -> frozenset[str]:
    return _common_dead_source_keys(num_encoder_blocks) | frozenset(
        {
            "seq_embedding.weight",
            "type_embedding.weight",
            "task_embedding.weight",
            "edge_rel_emb.weight",
            "edge_pairtype_emb.weight",
        }
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Convert one released OInvFold checkpoint from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modality",
        choices=("protein", "dna", "rna", "ligand"),
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    converter = {
        "protein": convert_oinvfold_protein_checkpoint,
        "dna": convert_oinvfold_dna_checkpoint,
        "rna": convert_oinvfold_rna_checkpoint,
        "ligand": convert_oinvfold_ligand_checkpoint,
    }[args.modality]
    converter(
        args.checkpoint,
        args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
