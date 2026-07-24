r"""
Configuration values for the NA-MPNN model and processor.
"""

from __future__ import annotations

from chanfig import Config


class NaMpnnConfig(Config):
    r"""
    Configure the NA-MPNN architecture.

    Args:
        vocab_size: Number of protein/DNA/RNA residue tokens in the NA-MPNN alphabet.
        hidden_size: Width of the message-passing hidden states.
        feature_size: Shared width of the polymer node and geometric edge features.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest neighbors in the residue graph.
        dropout: Dropout probability in encoder and decoder blocks.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial-basis bins for atom-pair distances.
        num_polymer_types: Number of polymer-type node categories.
        num_atom_types: Number of explicit atom slots consumed by the feature encoder.
        include_predicted_na_n: Whether to append the idealized nucleobase N feature.
    """

    vocab_size: int = 33
    hidden_size: int = 128
    feature_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 32
    dropout: float = 0.1
    num_positional_embeddings: int = 16
    num_rbf: int = 16
    num_polymer_types: int = 6
    num_atom_types: int = 16
    include_predicted_na_n: bool = True


class NaMpnnProcessorConfig(Config):
    r"""
    Configure NA-MPNN input processing.

    Args:
        shared_nucleic_tokens: Whether RNA and DNA share nucleic-acid token ids.
    """

    shared_nucleic_tokens: bool = True
