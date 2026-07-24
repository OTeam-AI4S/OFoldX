r"""
Configuration values for the ProteinMPNN model and processor.
"""

from __future__ import annotations

from chanfig import Config


class ProteinMpnnConfig(Config):
    r"""
    Configure architecture values shared by the ProteinMPNN graphs.

    Args:
        hidden_size: Width of node, edge, and decoder hidden states.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest neighbors in each residue graph.
        dropout: Dropout probability in encoder and decoder blocks.
        ca_only: Whether to build the CA-only graph instead of the backbone graph.
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 48
    dropout: float = 0.1
    ca_only: bool = False


class ProteinMpnnProcessorConfig(Config):
    r"""
    Configure ProteinMPNN input processing.

    Args:
        ca_only: Whether inputs contain only C-alpha coordinates instead of the
            N, C-alpha, C, and O backbone atoms.
    """

    ca_only: bool = False
