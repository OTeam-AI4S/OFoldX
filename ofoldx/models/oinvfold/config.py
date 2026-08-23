r"""
Configuration for the released OInvFold model graphs.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config


class OInvFoldConfig(Config):
    r"""
    Configure the shared protein, DNA, and RNA OInvFold graph.

    Attributes:
        hidden_size: Width of node, edge, and decoder representations.
        num_encoder_blocks: Number of graph encoder blocks.
        num_virtual_atoms: Number of learned geometric vectors per graph node.
        num_attention_heads: Number of graph-attention heads.
        num_geometric_layers: Number of layers in each geometric edge update.
        num_edge_layers: Number of layers in each learned edge update.
        dropout: Dropout probability in the graph encoder and autoregressive decoder.
        decoder_hidden_size: Width of the polymer decoder transition.

    Examples:
        >>> from ofoldx.models.oinvfold import OInvFoldConfig
        >>> config = OInvFoldConfig()
        >>> config.num_encoder_blocks
        12
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 12
    num_virtual_atoms: int = 32
    num_attention_heads: int = 4
    num_geometric_layers: int = 3
    num_edge_layers: int = 3
    dropout: float = 0.0
    decoder_hidden_size: int = 256


class OInvFoldLigandConfig(Config):
    r"""
    Configure the released ligand atom-element OInvFold graph.

    Attributes:
        hidden_size: Width of node and edge representations.
        num_encoder_blocks: Number of graph encoder blocks.
        num_virtual_atoms: Number of learned geometric vectors per graph node.
        num_attention_heads: Number of graph-attention heads.
        num_geometric_layers: Number of layers in each geometric edge update.
        num_edge_layers: Number of layers in each learned edge update.
        dropout: Dropout probability in the graph encoder.

    Examples:
        >>> from ofoldx.models.oinvfold import OInvFoldLigandConfig
        >>> config = OInvFoldLigandConfig()
        >>> config.num_encoder_blocks
        12
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 12
    num_virtual_atoms: int = 32
    num_attention_heads: int = 4
    num_geometric_layers: int = 3
    num_edge_layers: int = 3
    dropout: float = 0.0


class OInvFoldProcessorConfig(Config):
    r"""
    Select the polymer chemistry represented by one OInvFold repository.

    Attributes:
        modality: Protein, DNA, or RNA coordinate and vocabulary convention.
    """

    modality: Literal["protein", "dna", "rna"] = "protein"
