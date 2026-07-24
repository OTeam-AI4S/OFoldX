r"""
Configuration values for the LigandMPNN model and processor.
"""

from __future__ import annotations

from chanfig import Config


class LigandMpnnConfig(Config):
    r"""
    Configure the shared architecture of the LigandMPNN sequence graphs.

    Args:
        hidden_size: Shared width of node, edge, and decoder hidden states.
        num_encoder_blocks: Number of graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest neighbors in each residue graph.
        dropout: Dropout probability in encoder and decoder blocks.
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 32
    dropout: float = 0.0


class LigandMpnnProcessorConfig(Config):
    r"""
    Configure LigandMPNN input processing.

    Args:
        num_context_atoms: Number of nearby context atoms retained per residue.
    """

    num_context_atoms: int = 25
