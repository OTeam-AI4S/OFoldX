r"""
Hyperparameter schema defining a LigandMPNN design generator (widths, depths, graph and ligand-context sizes).
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


@MODEL_CONFIGS.register("ligandmpnn")
class LigandMPNNConfig(Config):
    r"""
    Configuration for the OFoldX-native LigandMPNN design generator.

    Stores the knobs that define a
    [`LigandMPNNDesignGenerator`][ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator]: the
    message-passing widths, encoder/decoder depths, graph connectivity, and the ligand-context budget.
    Instantiating with the defaults yields the released LigandMPNN sequence checkpoint family: 32 protein
    graph neighbors and 25 ligand context atoms per residue. ``checkpoint_variant`` stays ``None`` unless a
    concrete checkpoint is loaded (see
    [`config_from_ligandmpnn_checkpoint`][ofoldx.models.ligandmpnn.convert.config_from_ligandmpnn_checkpoint]).

    Attributes:
        architecture: Registry key selecting this architecture (`"ligandmpnn"`).
        checkpoint_variant: Name of the loaded official checkpoint, or ``None`` for a fresh config.
        vocab_size: Number of residue tokens; 20 amino acids plus one unknown/mask token.
        node_size: Width of the per-residue node features in the feature encoder.
        edge_size: Width of the per-edge features in the feature encoder.
        hidden_size: Width of the message-passing hidden state and token embedding.
        num_encoder_layers: Number of backbone-graph encoder layers.
        num_decoder_layers: Number of autoregressive decoder layers.
        k_neighbors: Number of nearest-neighbor residues in the protein graph.
        atom_context_num: Number of ligand/context atoms kept per residue.
        coord_noise_scale: Standard deviation of Gaussian noise added to input coordinates; ``0`` disables it.
        noise_level: Training noise level recorded from the source checkpoint, or ``None`` when unknown.
        dropout: Dropout probability shared by the encoder/decoder layers.
        use_side_chain_context: Whether to fold observed side-chain atoms into the ligand context.

    Examples:
        >>> config = LigandMPNNConfig()
        >>> config.architecture
        'ligandmpnn'
        >>> config.k_neighbors
        32
        >>> from ofoldx.models.ligandmpnn.model import LigandMPNNDesignGenerator
        >>> model = LigandMPNNDesignGenerator(config)  # doctest: +SKIP
    """

    architecture: str = "ligandmpnn"
    checkpoint_variant: str | None = None
    vocab_size: int = 21
    node_size: int = 128
    edge_size: int = 128
    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    k_neighbors: int = 32
    atom_context_num: int = 25
    coord_noise_scale: float = 0.0
    noise_level: float | None = None
    dropout: float = 0.0
    use_side_chain_context: bool = False


@MODEL_CONFIGS.register("ligandmpnn_packer")
class PackerConfig(Config):
    r"""
    Configuration for the LigandMPNN side-chain Packer.

    The Packer is the standalone ``ligandmpnn_sc`` architecture: it consumes backbone, sequence, and
    context features and predicts side-chain torsion mixtures. It is intentionally separate from
    [`LigandMPNNConfig`][ofoldx.models.ligandmpnn.config.LigandMPNNConfig], because the released sequence
    model and side-chain packer checkpoints have different tensor sets and different artifact entrypoints.

    Attributes:
        architecture: Registry key selecting the side-chain packer architecture.
        checkpoint_variant: Name of the loaded official checkpoint, or ``None`` for a fresh config.
        hidden_size: Width of the message-passing hidden state.
        num_encoder_layers: Number of backbone-graph encoder layers.
        num_decoder_layers: Number of torsion decoder layers.
        num_rbf: Number of radial basis functions for distance features.
        k_neighbors: Number of nearest-neighbor residues in the protein graph.
        atom_context_num: Number of ligand/context atoms kept per residue.
        num_mix: Number of Von Mises mixture components per chi angle.
        num_positional_embeddings: Width of the relative-position embedding.
        noise_level: Training noise level recorded from the source checkpoint, or ``None`` when unknown.
        dropout: Dropout probability.
        lower_bound: Lower distance bound for context RBFs.
        upper_bound: Upper distance bound for context RBFs.

    Examples:
        >>> config = PackerConfig()
        >>> config.architecture
        'ligandmpnn_packer'
        >>> config.atom_context_num
        16
    """

    architecture: str = "ligandmpnn_packer"
    checkpoint_variant: str | None = None
    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_rbf: int = 16
    k_neighbors: int = 32
    atom_context_num: int = 16
    num_mix: int = 3
    num_positional_embeddings: int = 16
    noise_level: float | None = None
    dropout: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 20.0
