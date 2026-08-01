r"""
Configuration for the released gRNAde model and canonical RNA lowering.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config


class GrnadeConfig(Config):
    r"""
    Configure the released gRNAde GVP encoder and autoregressive decoder.

    Attributes:
        node_hidden_scalar_size: Number of hidden scalar node channels.
        node_hidden_vector_size: Number of hidden vector node channels.
        edge_scalar_size: Number of scalar input channels per graph edge.
        edge_hidden_scalar_size: Number of hidden scalar edge channels.
        edge_hidden_vector_size: Number of hidden vector edge channels.
        num_blocks: Number of GVP blocks in both encoder and decoder.
        dropout: Dropout probability in GVP blocks.
        num_token_embeddings: Number of released decoder input tokens.

    Examples:
        >>> from ofoldx.models.grnade import GrnadeConfig
        >>> config = GrnadeConfig(num_blocks=2)
        >>> config.num_blocks
        2
    """

    node_hidden_scalar_size: int = 128
    node_hidden_vector_size: int = 16
    edge_scalar_size: int = 132
    edge_hidden_scalar_size: int = 64
    edge_hidden_vector_size: int = 4
    num_blocks: int = 4
    dropout: float = 0.5
    num_token_embeddings: int = 5


class GrnadeV032ProcessorConfig(Config):
    r"""
    Configure released gRNAde v0.3.2 conformation selection.

    Attributes:
        max_num_conformations: Number of conformation rows presented to the
            released multi-state encoder.

    Examples:
        >>> from ofoldx.models.grnade import GrnadeV032ProcessorConfig
        >>> GrnadeV032ProcessorConfig(max_num_conformations=3).max_num_conformations
        3
    """

    max_num_conformations: Literal[1, 2, 3, 5] = 1
