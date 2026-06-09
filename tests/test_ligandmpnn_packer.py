from __future__ import annotations

from chanfig import Config

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME
from ofoldx.models import AutoConfig, AutoModel
from ofoldx.models.ligandmpnn import Packer, PackerConfig


def test_packer_saves_and_loads_with_auto_model(tmp_path) -> None:
    config = PackerConfig(
        {
            "hidden_size": 16,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "num_rbf": 4,
            "k_neighbors": 4,
            "atom_context_num": 2,
            "num_positional_embeddings": 4,
            "checkpoint_variant": "tiny-packer",
        }
    )
    model = Packer(config)

    model.save_pretrained(tmp_path)

    manifest = Config.from_json(tmp_path / ARTIFACT_MANIFEST_NAME)
    loaded_config = AutoConfig.from_pretrained(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path)

    assert manifest["kind"] == "model"
    assert manifest["architecture"] == "ligandmpnn_packer"
    assert manifest["entrypoint"] == "ofoldx.models.ligandmpnn.packer.Packer"
    assert loaded_config.architecture == "ligandmpnn_packer"
    assert loaded_config.checkpoint_variant == "tiny-packer"
    assert isinstance(loaded, Packer)
    assert len(loaded.state_dict()) == len(model.state_dict())
