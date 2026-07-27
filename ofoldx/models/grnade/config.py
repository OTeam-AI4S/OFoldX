r"""
Configuration for the released gRNAde model and canonical RNA lowering.
"""

from __future__ import annotations

from chanfig import Config


class GrnadeConfig(Config):
    r"""
    Configure the released gRNAde GVP encoder and autoregressive decoder.

    Args:
        node_scalar_size: Number of scalar input channels per RNA residue.
        node_vector_size: Number of vector input channels per RNA residue.
        node_hidden_scalar_size: Number of hidden scalar node channels.
        node_hidden_vector_size: Number of hidden vector node channels.
        edge_scalar_size: Number of scalar input channels per graph edge.
        edge_vector_size: Number of vector input channels per graph edge.
        edge_hidden_scalar_size: Number of hidden scalar edge channels.
        edge_hidden_vector_size: Number of hidden vector edge channels.
        num_blocks: Number of GVP blocks in both encoder and decoder.
        dropout: Dropout probability in GVP blocks.
        vocabulary_size: Number of canonical RNA identities.

    Examples:
        >>> from ofoldx.models.grnade import GrnadeConfig
        >>> config = GrnadeConfig(num_blocks=2)
        >>> config.num_blocks
        2
    """

    node_scalar_size: int = 15
    node_vector_size: int = 4
    node_hidden_scalar_size: int = 128
    node_hidden_vector_size: int = 16
    edge_scalar_size: int = 132
    edge_vector_size: int = 3
    edge_hidden_scalar_size: int = 64
    edge_hidden_vector_size: int = 4
    num_blocks: int = 4
    dropout: float = 0.5
    vocabulary_size: int = 4


class GrnadeProcessorConfig(Config):
    r"""
    Configure canonical RNA graph construction.

    Args:
        num_neighbors: Number of spatial and sequence neighbors per residue.
        num_rbf: Number of radial-basis channels for each atom-pair distance.
        num_positional_embeddings: Width of the sinusoidal sequence-offset encoding.
        distance_epsilon: Released configurable smoothing used by centroid
            construction, edge lengths, and the node-length log offset. Other
            normalized geometric features retain the released fixed ``1e-5``.
    """

    num_neighbors: int = 32
    num_rbf: int = 32
    num_positional_embeddings: int = 32
    distance_epsilon: float = 1e-5
