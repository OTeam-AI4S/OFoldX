r"""
Configuration for the released OInvFold model graphs.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config


class OInvFoldPolymerConfig(Config):
    r"""
    Configure the shared protein, DNA, and RNA OInvFold graph.

    Args:
        hidden_size: Width of node, edge, and decoder representations.
        num_encoder_blocks: Number of graph encoder blocks.
        num_virtual_embeddings: Number of learned global-frame embeddings.
        num_virtual_atoms: Number of learned geometric vectors per graph node.
        num_attention_heads: Number of graph-attention heads.
        num_geometric_layers: Number of layers in each geometric edge update.
        num_edge_layers: Number of layers in each learned edge update.
        dropout: Dropout probability in the graph encoder and autoregressive decoder.
        node_feature_size: Width of lowered polymer node features.
        edge_feature_size: Width of lowered polymer edge features.
        vocabulary_size: Size of the unified protein/DNA/RNA vocabulary.
        decoder_hidden_size: Width of the polymer decoder transition.
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 12
    num_virtual_embeddings: int = 30
    num_virtual_atoms: int = 32
    num_attention_heads: int = 4
    num_geometric_layers: int = 3
    num_edge_layers: int = 3
    dropout: float = 0.0
    node_feature_size: int = 133
    edge_feature_size: int = 310
    vocabulary_size: int = 25
    decoder_hidden_size: int = 256


class OInvFoldLigandConfig(Config):
    r"""
    Configure the released ligand atom-element OInvFold graph.

    Args:
        hidden_size: Width of node and edge representations.
        num_encoder_blocks: Number of graph encoder blocks.
        num_virtual_embeddings: Number of learned global-frame embeddings.
        num_virtual_atoms: Number of learned geometric vectors per graph node.
        num_attention_heads: Number of graph-attention heads.
        num_geometric_layers: Number of layers in each geometric edge update.
        num_edge_layers: Number of layers in each learned edge update.
        dropout: Dropout probability in the graph encoder.
        node_feature_size: Width of lowered atom-node features.
        edge_feature_size: Width of lowered atom-edge features.
        vocabulary_size: Size of the released element vocabulary.
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 12
    num_virtual_embeddings: int = 30
    num_virtual_atoms: int = 32
    num_attention_heads: int = 4
    num_geometric_layers: int = 3
    num_edge_layers: int = 3
    dropout: float = 0.0
    node_feature_size: int = 114
    edge_feature_size: int = 272
    vocabulary_size: int = 37


class OInvFoldPolymerProcessorConfig(Config):
    r"""
    Select the polymer chemistry represented by one OInvFold repository.

    Args:
        modality: Protein, DNA, or RNA coordinate and vocabulary convention.
    """

    modality: Literal["protein", "dna", "rna"] = "protein"
