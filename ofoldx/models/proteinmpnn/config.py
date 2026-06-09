r"""
ProteinMPNN configuration.
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


@MODEL_CONFIGS.register("proteinmpnn")
class ProteinMPNNConfig(Config):
    r"""
    Configuration for the official ProteinMPNN architecture.

    The public knobs of the OFoldX-native ProteinMPNN design generator. Default dimensions match the released
    ``proteinmpnn_v_48_*.pt`` checkpoint family, so instantiating with the defaults reproduces the reference
    architecture. ``checkpoint_variant`` and ``noise_level`` stay unset for a freshly-built model and are populated
    only by explicit checkpoint conversion or loading (see
    [`proteinmpnn_config_from_checkpoint`][ofoldx.models.proteinmpnn.convert.proteinmpnn_config_from_checkpoint]).

    Attributes:
        architecture: Registry key selecting this architecture (`"proteinmpnn"`).
        vocab_size: Number of amino-acid tokens, including the unknown/gap token (`21`).
        edge_feature_size: Width of the geometric edge features produced by the backbone encoder.
        hidden_size: Width of the node and edge hidden representations in the message-passing stack.
        num_encoder_layers: Number of geometric encoder layers.
        num_decoder_layers: Number of autoregressive decoder layers.
        k_neighbors: Number of nearest neighbors used to build the sparse residue graph.
        coord_noise_scale: Standard deviation of Gaussian noise added to backbone coords before featurization.
        dropout: Dropout probability applied inside the encoder and decoder layers.
        node_feature_size: Optional per-residue node feature width. Membrane ProteinMPNN checkpoints use three
            membrane-label channels; vanilla ProteinMPNN leaves this at zero and starts encoder nodes from zeros.
        checkpoint_variant: Released checkpoint tag (e.g. `"proteinmpnn_v_48_020"`) once a checkpoint is loaded;
            `None` for a freshly-created model.
        noise_level: Training backbone-noise level recorded in the source checkpoint; `None` when unknown.

    Examples:
        >>> config = ProteinMPNNConfig()
        >>> config.architecture
        'proteinmpnn'
        >>> config.k_neighbors
        48
        >>> config.checkpoint_variant is None
        True
    """

    architecture: str = "proteinmpnn"
    vocab_size: int = 21
    edge_feature_size: int = 128
    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    k_neighbors: int = 48
    coord_noise_scale: float = 0.0
    dropout: float = 0.0
    node_feature_size: int = 0
    checkpoint_variant: str | None = None
    noise_level: float | None = None
