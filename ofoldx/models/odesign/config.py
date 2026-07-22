r"""Persistent configuration for the ODesign architecture and its Processor."""

from __future__ import annotations

from chanfig import Config


class ODesignConstraintEmbeddingsConfig(Config):
    r"""Observed condition-geometry embedder dimensions."""

    num_blocks: int = 2
    hidden_size: int = 64
    num_bins: int = 39
    min_distance: float = 3.25
    max_distance: float = 50.75
    dropout: float = 0.25


class ODesignTrunkConfig(Config):
    r"""Pairformer and MSA dimensions that determine the released graph."""

    num_pairformer_blocks: int = 48
    num_msa_blocks: int = 4
    pairformer_dropout: float = 0.25
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25


class ODesignDenoiserConfig(Config):
    r"""All-atom EDM denoiser dimensions."""

    sigma_data: float = 16.0
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    token_hidden_size: int = 768
    num_token_encoder_blocks: int = 24
    num_token_encoder_attention_heads: int = 16
    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    num_atom_decoder_blocks: int = 3
    num_atom_decoder_attention_heads: int = 4
    num_query_atoms: int = 32
    num_key_atoms: int = 128


class ODesignPairwiseConfig(Config):
    r"""Distogram and bond-reconstruction head dimensions."""

    num_distogram_bins: int = 64
    bond_hidden_size: int = 128
    num_bond_types: int = 2


class ODesignConfig(Config):
    r"""Exact construction parameters for the released ODesign core graph.

    Runtime placement, operator selection, Hub transport, and training policy are
    deliberately absent.  ``rigid`` and ``flex`` repositories differ in weights and
    conditioning inputs, not in this graph or its state-dict contract.
    """

    token_feature_size: int = 453
    atom_feature_size: int = 386
    single_size: int = 384
    pair_size: int = 128
    constraint_embeddings: ODesignConstraintEmbeddingsConfig = ODesignConstraintEmbeddingsConfig()
    relative_position_size: int = 139
    trunk: ODesignTrunkConfig = ODesignTrunkConfig()
    recycling_steps: int = 10
    denoiser: ODesignDenoiserConfig = ODesignDenoiserConfig()
    pairwise: ODesignPairwiseConfig = ODesignPairwiseConfig()


class ODesignProcessorConfig(Config):
    r"""Persistent configuration for canonical ODesign lowering.

    ``receptor_mode`` selects flex or rigid receptor conditioning. It is the only setting
    that differs between the two protein
    repositories ``ofoldx/odesign-base-prot-flex`` and ``ofoldx/odesign-base-prot-rigid``:
    the graph, the ``ODesignConfig``, the ``__call__`` signature, and the
    ``ODesignInput``/``ODesignOutput`` types are identical. Both repositories therefore use
    the same architecture and select behavior through processor configuration. ``"rigid"``
    marks every existing (non-generated)
    receptor heavy atom as a hard fixed coordinate (the released ``diffusion`` condition,
    receptor held rigid); ``"flex"`` regenerates receptor coordinates and pins only atoms
    named by an explicit hard ``ODesignReferenceCoordinates``. This is additive over the
    per-atom ``ODesignReferenceCoordinates`` path in ``processor.py``.
    """

    receptor_mode: str = "flex"  # "rigid" | "flex" (str, not Literal: chanfig validates fields via isinstance)
