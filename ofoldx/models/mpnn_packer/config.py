r"""Configuration for the released MpnnPacker."""

from __future__ import annotations

from chanfig import Config


class MpnnPackerProcessorConfig(Config):
    r"""Tensor-shape choices for canonical Structure lowering."""

    atom_context_num: int = 16


class MpnnPackerConfig(Config):
    r"""Architecture values for the released ``ligandmpnn_sc`` checkpoint."""

    hidden_size: int = 128
    edge_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_rbf: int = 16
    k_neighbors: int = 32
    atom_context_num: int = 16
    num_mix: int = 3
    num_positional_embeddings: int = 16
    dropout: float = 0.0
    coordinate_noise: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 20.0
