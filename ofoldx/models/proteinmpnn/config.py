r"""Configuration values for the ProteinMPNN model and processor."""

from __future__ import annotations

from chanfig import Config


class ProteinMpnnConfig(Config):
    r"""Architecture values shared by the released ProteinMPNN graphs."""

    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48
    dropout: float = 0.1
    ca_only: bool = False


class ProteinMpnnProcessorConfig(Config):
    r"""Persistent ProteinMPNN input-lowering configuration."""

    ca_only: bool = False
