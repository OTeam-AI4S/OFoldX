r"""
Convert the four released OInvFold Lightning checkpoints.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor

from .config import (
    OInvFoldLigandConfig,
    OInvFoldConfig,
    OInvFoldProcessorConfig,
)
from .model import OInvFoldLigandModel, OInvFoldModel


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

    from .pipeline import OInvFoldLigandPipeline
    from .processor import OInvFoldLigandProcessor

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
    OInvFoldLigandPipeline(
        model=model,
        processor=OInvFoldLigandProcessor(),
    ).save(
        output_dir,
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
    from .pipeline import OInvFoldPipeline
    from .processor import OInvFoldProcessor

    model = OInvFoldModel(OInvFoldConfig())
    state_dict = _lightning_state_dict(checkpoint_path)
    model.load_state_dict(
        _live_state_dict(
            state_dict,
            model.state_dict(),
            _polymer_omitted_source_keys(model.config.num_encoder_blocks),
            architecture="polymer",
            config=model.config,
        ),
        strict=True,
        assign=True,
    )
    model.eval()
    processor = OInvFoldProcessor(OInvFoldProcessorConfig(modality=modality))
    OInvFoldPipeline(model=model, processor=processor).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


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
    config: OInvFoldConfig | OInvFoldLigandConfig,
) -> dict[str, Tensor]:
    prefixed = tuple(key.startswith("model.") for key in source)
    if all(prefixed):
        normalized = {key.removeprefix("model."): value for key, value in source.items()}
    elif not any(prefixed):
        normalized = dict(source)
    else:
        raise KeyError("OInvFold checkpoint mixes prefixed and unprefixed state keys")

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
        raise KeyError(f"unrecognized OInvFold checkpoint tensors: {sorted(unexpected)}")
    if missing_omissions:
        raise KeyError(
            f"released OInvFold checkpoint is missing expected source-only tensors: {sorted(missing_omissions)}"
        )
    if missing:
        raise KeyError(f"OInvFold checkpoint is missing live tensors: {sorted(missing)}")
    return {key: converted[key] for key in target}


def _target_key(
    key: str,
    architecture: Literal["polymer", "ligand"],
    config: OInvFoldConfig | OInvFoldLigandConfig,
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
            target_prefix=f"{target_prefix}geometric_edge_update.geometry_transition.",
            num_layers=config.num_geometric_layers,
        )
        if translated is not None:
            return translated
        translated = _gelu_transition_target_key(
            key,
            source_prefix=f"{source_prefix}update_edge.W.",
            target_prefix=f"{target_prefix}edge_update.transition.",
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
            ("attention.Bias.0.", "attention.logit_input_projection."),
            ("attention.Bias.2.", "attention.logit_hidden_projection."),
            ("attention.Bias.4.", "attention.logit_output_projection."),
            ("attention.W_O.", "attention.output_projection."),
            ("attention.gate.", "attention.output_gate."),
            ("update_node.dense.0.", "node_update.transition_input_norm."),
            (
                "update_node.dense.1.",
                "node_update.transition_hidden_projection.",
            ),
            (
                "update_node.dense.3.",
                "node_update.transition_output_projection.",
            ),
            (
                "update_node.dense.4.",
                "node_update.transition_output_norm.",
            ),
            (
                "update_node.V_MLP_g.0.",
                "node_update.graph_gate_input_projection.",
            ),
            (
                "update_node.V_MLP_g.2.",
                "node_update.graph_gate_hidden_projection.",
            ),
            (
                "update_node.V_MLP_g.4.",
                "node_update.graph_gate_output_projection.",
            ),
            ("update_edge.norm.", "edge_update.output_norm."),
        )
        for source, destination in replacements:
            if block_key.startswith(source):
                return target_prefix + destination + block_key.removeprefix(source)
        return key

    if key.startswith("encoder.s."):
        return "encoder.layer_gate." + key.removeprefix("encoder.s.")

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


def _gelu_transition_target_key(
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
