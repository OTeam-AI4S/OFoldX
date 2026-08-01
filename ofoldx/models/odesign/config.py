r"""
Persistent configuration for the ODesign architecture and its Processor.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config


class ODesignConstraintEmbeddingsConfig(Config):
    r"""
    Configure the observed condition-geometry embedder.

    Attributes:
        num_blocks: Number of condition embedding blocks.
        hidden_size: Width of condition pair features.
        num_distogram_bins: Number of condition distance bins.
        min_distance: Lower condition distance bound, in angstroms.
        max_distance: Upper condition distance bound, in angstroms.
        dropout: Dropout probability in condition embedding blocks.
    """

    num_blocks: int = 2
    hidden_size: int = 64
    num_distogram_bins: int = 39
    min_distance: float = 3.25
    max_distance: float = 50.75
    dropout: float = 0.25


class ODesignTrunkConfig(Config):
    r"""
    Configure the Pairformer and MSA trunk.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks.
        num_msa_stages: Number of MSA stages, including the final pair-only update.
        pairformer_dropout: Dropout probability inside Pairformer blocks.
        msa_dropout: Dropout probability along the MSA path.
        pair_dropout: Dropout probability along the pair path.
    """

    num_pairformer_blocks: int = 48
    num_msa_stages: int = 4
    pairformer_dropout: float = 0.25
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25


class ODesignDenoiserConfig(Config):
    r"""
    Configure the all-atom EDM denoiser.

    Attributes:
        sigma_data: Data noise scale used by EDM preconditioning.
        atom_hidden_size: Width of atom single features.
        atom_pair_hidden_size: Width of atom pair features.
        token_hidden_size: Width of denoiser token features.
        num_token_encoder_blocks: Number of token encoder blocks.
        num_token_encoder_attention_heads: Number of token encoder attention heads.
        num_atom_encoder_blocks: Number of atom encoder blocks.
        num_atom_encoder_attention_heads: Number of atom encoder attention heads.
        num_atom_decoder_blocks: Number of atom decoder blocks.
        num_atom_decoder_attention_heads: Number of atom decoder attention heads.
        num_query_atoms: Query-window size for local atom attention.
        num_key_atoms: Key-window size for local atom attention.
    """

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
    r"""
    Configure distogram and bond-reconstruction heads.

    Attributes:
        num_distogram_bins: Number of distogram output bins.
        bond_hidden_size: Width of bond-reconstruction hidden features.
    """

    num_distogram_bins: int = 64
    bond_hidden_size: int = 128


class ODesignConfig(Config):
    r"""
    Configure the core graph shared by rigid and flexible ODesign models.

    Attributes:
        single_size: Width of the token representation.
        pair_size: Width of the pair representation.
        constraint_embeddings: Condition-geometry embedding architecture.
        max_relative_residue: Largest unclipped residue and token offset.
        max_relative_chain: Largest unclipped symmetry-copy offset.
        trunk: Pairformer and MSA trunk architecture.
        denoiser: All-atom EDM denoiser architecture.
        pairwise: Distogram and bond-reconstruction head configuration.

    Examples:
        >>> from ofoldx.models.odesign import ODesignConfig
        >>> config = ODesignConfig(single_size=256)
        >>> config.single_size
        256
    """

    single_size: int = 384
    pair_size: int = 128
    constraint_embeddings: ODesignConstraintEmbeddingsConfig = ODesignConstraintEmbeddingsConfig()
    max_relative_residue: int = 32
    max_relative_chain: int = 2
    trunk: ODesignTrunkConfig = ODesignTrunkConfig()
    denoiser: ODesignDenoiserConfig = ODesignDenoiserConfig()
    pairwise: ODesignPairwiseConfig = ODesignPairwiseConfig()


class ODesignProcessorConfig(Config):
    r"""
    Configure one polymer-backbone ODesign artifact.

    Attributes:
        artifact: Exact released Processor identity. The nucleic-acid artifact
            realizes both DNA and RNA chemistry.
    """

    artifact: Literal[
        "protein_flex",
        "protein_rigid",
        "nucleic_acid_rigid",
    ] = "protein_flex"
