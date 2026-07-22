r"""Configuration for the released MpnnPacker."""

from __future__ import annotations

from chanfig import Config


class MpnnPackerConfig(Config):
    r"""Architecture values for the released ``ligandmpnn_sc`` checkpoint."""

    hidden_size: int = 128
    edge_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_rbf: int = 16
    num_neighbors: int = 32
    num_context_atoms: int = 16
    num_mix: int = 3
    num_positional_embeddings: int = 16
    dropout: float = 0.0
    coordinate_noise: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 20.0


class MpnnPackerProcessorConfig(Config):
    r"""Tensor-shape choices for canonical Structure lowering."""

    num_context_atoms: int = 16
