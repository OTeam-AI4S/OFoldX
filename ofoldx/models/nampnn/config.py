r"""
NA-MPNN model configuration.
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


@MODEL_CONFIGS.register("nampnn")
class NAMPNNConfig(Config):
    r"""
    NA-MPNN architecture configuration.

    Attributes:
        architecture: Registry key selecting this architecture (`"nampnn"`).
        checkpoint_variant: Source checkpoint name once converted, or ``None`` for a fresh model.
        vocab_size: Number of protein/DNA/RNA residue tokens in the official NA-MPNN alphabet.
        hidden_size: Width of the message-passing hidden states.
        node_feature_size: Width of polymer-type node features after projection.
        edge_feature_size: Width of geometric edge features after projection.
        num_encoder_layers: Number of graph encoder layers.
        num_decoder_layers: Number of autoregressive decoder layers.
        k_neighbors: Number of nearest neighbors in the residue graph.
        dropout: Dropout probability in encoder and decoder layers.
        coord_noise_scale: Shared coordinate-noise scale used when no polymer-specific scale is set.
        protein_coord_noise_scale: Protein-coordinate training noise scale.
        dna_coord_noise_scale: DNA-coordinate training noise scale.
        rna_coord_noise_scale: RNA-coordinate training noise scale.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial-basis bins for atom-pair distances.
        num_polymer_types: Number of polymer-type node categories.
        num_atom_types: Number of explicit atom slots consumed by the feature encoder.
        na_ref_atom: Nucleic-acid reference atom used for graph centering.
        include_predicted_na_n: Whether to append the idealized nucleobase N atom feature.
        shared_nucleic_acid_tokens: Whether RNA residues share the DNA token ids used by the released checkpoints.

    Examples:
        >>> config = NAMPNNConfig()
        >>> config.architecture
        'nampnn'
        >>> config.vocab_size
        33
    """

    architecture: str = "nampnn"
    checkpoint_variant: str | None = None
    vocab_size: int = 33
    hidden_size: int = 128
    node_feature_size: int = 128
    edge_feature_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    k_neighbors: int = 32
    dropout: float = 0.0
    coord_noise_scale: float = 0.0
    protein_coord_noise_scale: float | None = None
    dna_coord_noise_scale: float | None = None
    rna_coord_noise_scale: float | None = None
    num_positional_embeddings: int = 16
    num_rbf: int = 16
    num_polymer_types: int = 6
    num_atom_types: int = 16
    na_ref_atom: str = "C1'"
    include_predicted_na_n: bool = True
    shared_nucleic_acid_tokens: bool = True
