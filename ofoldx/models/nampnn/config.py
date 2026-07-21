r"""Exact NA-MPNN Model and Processor configuration."""

from __future__ import annotations

from chanfig import Config


class NaMpnnConfig(Config):
    r"""
    NA-MPNN architecture configuration.

    Both released checkpoints share one 33-token architecture; ``mode`` selects a
    weight distribution and released sampling defaults, not a config branch. Only
    checkpoint-shape facts survive here: one independently live ``hidden_size`` and
    one shared ``feature_size`` for the node and edge projections, whose upstream
    node/edge width pair is false because its constructor swaps those arguments.
    Temperature, neighbor override, protein/DNA/RNA coordinate noise,
    protein-first teacher forcing, and seed or generator are operation arguments,
    not persistent config.

    Attributes:
        vocab_size: Number of protein/DNA/RNA residue tokens in the NA-MPNN alphabet.
        hidden_size: Width of the message-passing hidden states.
        feature_size: Shared width of the polymer node and geometric edge features.
        num_encoder_layers: Number of graph encoder layers.
        num_decoder_layers: Number of autoregressive decoder layers.
        num_neighbors: Number of nearest neighbors in the residue graph.
        dropout: Dropout probability in encoder and decoder layers.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial-basis bins for atom-pair distances.
        num_polymer_types: Number of polymer-type node categories.
        num_atom_types: Number of explicit atom slots consumed by the feature encoder.
        include_predicted_na_n: Whether to append the idealized nucleobase N feature.

    Examples:
        >>> config = NaMpnnConfig()
        >>> config.vocab_size
        33
    """

    vocab_size: int = 33
    hidden_size: int = 128
    feature_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 32
    dropout: float = 0.1
    num_positional_embeddings: int = 16
    num_rbf: int = 16
    num_polymer_types: int = 6
    num_atom_types: int = 16
    include_predicted_na_n: bool = True


class NaMpnnProcessorConfig(Config):
    r"""Persistent NA-MPNN tokenization configuration."""

    shared_nucleic_tokens: bool = True
