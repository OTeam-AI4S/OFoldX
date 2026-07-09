from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.models import AutoModel
from ofoldx.models.ligandmpnn import LigandMPNNPacker, LigandMPNNPackerConfig
from ofoldx.models.ligandmpnn.convert import (
    _PACKER_LAYER_GROUPS,
    _PACKER_TOP_LEVEL,
    convert_ligandmpnn_checkpoint,
)


def test_convert_ligandmpnn_checkpoint_auto_routes_side_chain_packer(tmp_path: Path) -> None:
    config = LigandMPNNPackerConfig(
        {
            "hidden_size": 16,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "num_rbf": 4,
            "k_neighbors": 4,
            "atom_context_num": 2,
            "num_positional_embeddings": 4,
        }
    )
    source_model = LigandMPNNPacker(config)
    checkpoint_path = tmp_path / "ligandmpnn_sc_tiny.pt"
    torch.save(
        {
            "model_state_dict": _official_packer_state_dict(source_model),
            "num_edges": config.k_neighbors,
            "atom_context_num": config.atom_context_num,
            "noise_level": 0.02,
        },
        checkpoint_path,
    )

    output_dir = tmp_path / "artifact"
    report = convert_ligandmpnn_checkpoint(checkpoint_path, output_dir)
    loaded = AutoModel.from_pretrained(output_dir)

    assert report["architecture"] == "ligandmpnn_packer"
    assert report["checkpoint_variant"] == "ligandmpnn_sc_tiny"
    assert report["tensor_count"] == len(source_model.state_dict())
    assert isinstance(loaded, LigandMPNNPacker)
    assert loaded.config.architecture == "ligandmpnn_packer"
    assert loaded.config.k_neighbors == 4
    assert loaded.config.atom_context_num == 2
    assert loaded.config.noise_level == 0.02
    for key, tensor in source_model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)

    with pytest.raises(ValueError, match="checkpoint_kind='sequence'"):
        convert_ligandmpnn_checkpoint(checkpoint_path, tmp_path / "wrong-kind", checkpoint_kind="sequence")


def _official_packer_state_dict(model: LigandMPNNPacker) -> dict[str, torch.Tensor]:
    return {_native_to_official_packer_key(key): tensor.clone() for key, tensor in model.state_dict().items()}


def _native_to_official_packer_key(key: str) -> str:
    for official_group, (native_group, sublayer) in _PACKER_LAYER_GROUPS.items():
        if not key.startswith(native_group + "."):
            continue
        _, layer, remainder = key.split(".", 2)
        inverse_sublayers = sorted(
            ((native_prefix, official_prefix) for official_prefix, native_prefix in sublayer.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for native_prefix, official_prefix in inverse_sublayers:
            if remainder.startswith(native_prefix + "."):
                return f"{official_group}.{layer}.{official_prefix}{remainder[len(native_prefix):]}"
        raise AssertionError(f"unmapped LigandMPNNPacker layer key: {key}")

    inverse_top_level = sorted(
        ((native_prefix, official_prefix) for official_prefix, native_prefix in _PACKER_TOP_LEVEL.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for native_prefix, official_prefix in inverse_top_level:
        if key in (f"{native_prefix}.weight", f"{native_prefix}.bias"):
            return official_prefix + key[len(native_prefix) :]
    raise AssertionError(f"unmapped LigandMPNNPacker key: {key}")
