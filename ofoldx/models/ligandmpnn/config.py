r"""Configuration values for the LigandMPNN model and processor."""

from __future__ import annotations

from chanfig import Config


class LigandMpnnConfig(Config):
    r"""Architecture values shared by the released LigandMPNN sequence graphs.

    The released checkpoints expose one usable node/edge/hidden width, so a single
    ``hidden_size`` replaces the three nominally independent axes. The four active
    ``ligandmpnn_v_32_{005,010,020,030}_25.pt`` checkpoints differ only in their
    training coordinate-noise label, which never changes the executable graph.
    """

    hidden_size: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 32
    dropout: float = 0.0


class LigandMpnnProcessorConfig(Config):
    r"""Persistent LigandMPNN input-lowering configuration.

    ``num_context_atoms`` fixes only the padded nearest-context extent the Processor
    emits per residue; it controls no learned tensor shape. ``use_atom_context``,
    the ligand-interface cutoff, and occupancy filtering are per-call operation
    controls rather than persistent configuration.
    """

    num_context_atoms: int = 25
