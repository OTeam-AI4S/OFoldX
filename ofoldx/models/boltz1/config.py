# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Boltz-1 architecture and inference configuration.
"""

from __future__ import annotations

from chanfig import Config

from ...data.tokenization import TOKENS


class Boltz1InputEmbeddingsConfig(Config):
    r"""
    Configure the Boltz-1 atom-attention input embedder and token feature widths.

    Attributes:
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        vocab_size: Residue-type vocabulary size feeding the one-hot residue feature.
        pocket_feature_size: Width of the token-level pocket one-hot conditioning feature.
        use_atom_encoder: Whether to encode atom features into token states.
    """

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = len(TOKENS)
    pocket_feature_size: int = 4
    use_atom_encoder: bool = True


class Boltz1TrunkConfig(Config):
    r"""
    Configure the Boltz-1 Pairformer trunk and optional MSA trunk.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk.
        num_pair_attention_heads: Head count for the single-track pair-biased attention.
        num_msa_blocks: Number of MSA-module blocks.
        msa_size: Width of the MSA representation channel.
        msa_attention_head_size: Per-head dimension inside the MSA attention.
        num_msa_attention_heads: Head count for the MSA pair-weighted-averaging.
        outer_product_hidden_size: Hidden width of the MSA outer-product-mean projection.
        msa_triangle_attention_head_size: Per-head width of the MSA triangle attention.
        num_msa_triangle_attention_heads: Head count for the MSA triangle attention.
        triangle_attention_head_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Head count for the triangle attention.
        use_paired_feature: Whether the MSA carries the cross-chain pairing feature.
        pair_dropout: Dropout applied to the Pairformer pair track.
        msa_dropout: Dropout applied inside the MSA trunk.
        msa_pair_dropout: Pair-track dropout applied inside the MSA trunk.
        update_single: Whether the Pairformer trunk updates the single track.
    """

    num_pairformer_blocks: int = 48
    num_pair_attention_heads: int = 16
    num_msa_blocks: int = 4
    msa_size: int = 64
    msa_attention_head_size: int = 32
    num_msa_attention_heads: int = 8
    outer_product_hidden_size: int = 32
    msa_triangle_attention_head_size: int = 32
    num_msa_triangle_attention_heads: int = 4
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    use_paired_feature: bool = False
    pair_dropout: float = 0.25
    msa_dropout: float = 0.15
    msa_pair_dropout: float = 0.25
    update_single: bool = True


class Boltz1DenoiserConfig(Config):
    r"""
    Configure the Boltz-1 EDM token encoder and atom encoder-decoder.

    Attributes:
        num_token_encoder_blocks: Number of conditioned token-encoder blocks in the score model.
        num_token_encoder_attention_heads: Head count for the token encoder.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks in the denoiser.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks in the denoiser.
        num_atom_attention_heads: Head count shared by the denoiser atom encoder and decoder.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        num_query_atoms: Query-window size for the local atom attention.
        num_key_atoms: Key-window size for the local atom attention.
        num_conditioning_transitions: Number of transitions in each diffusion conditioner.
        sigma_data: EDM data-scale hyperparameter used by the preconditioner and noise schedule.
        alignment_reverse_diff: Whether to weighted-rigid-align each reverse-diffusion step to the denoised coords.
    """

    num_token_encoder_blocks: int = 24
    num_token_encoder_attention_heads: int = 16
    num_atom_encoder_blocks: int = 3
    num_atom_decoder_blocks: int = 3
    num_atom_attention_heads: int = 4
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    num_conditioning_transitions: int = 2
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True


class Boltz1SamplerConfig(Config):
    r"""
    Configure the Boltz-1 EDM reverse-diffusion schedule.

    Attributes:
        num_sampling_steps: Number of reverse-diffusion steps.
        rho: Karras schedule power controlling the sigma spacing.
        max_noise_level: Highest noise level (start of the schedule), in ``sigma_data`` units.
        min_noise_level: Lowest non-zero noise level (end of the schedule), in ``sigma_data`` units.
        churn_rate: Stochastic-churn factor added above ``min_churn_noise_level``.
        min_churn_noise_level: Noise level below which churn is disabled.
        noise_scale: Multiplier on the churn-injected noise.
        step_scale: Multiplier on the Euler update step.
    """

    num_sampling_steps: int = 200
    rho: float = 8.0
    max_noise_level: float = 160.0
    min_noise_level: float = 4e-4
    churn_rate: float = 0.605
    min_churn_noise_level: float = 1.107
    noise_scale: float = 0.901
    step_scale: float = 1.638


class Boltz1ConfidenceConfig(Config):
    r"""
    Configure the Boltz-1 confidence head and combined ranking score.

    Attributes:
        enabled: Whether to build and run the confidence/ranking head at all.
        imitate_trunk: Whether the confidence head re-runs a full trunk-like stack instead of reusing trunk outputs.
        compute_pae: Whether to emit PAE logits and pTM/ipTM ranking metrics.
        num_pairformer_blocks: Number of Pairformer blocks inside the confidence head.
        num_pair_attention_heads: Head count for the confidence-head pair-biased attention.
        triangle_attention_head_size: Per-head width of the confidence triangle attention.
        num_triangle_attention_heads: Confidence triangle-attention head count.
        num_distogram_bins: Number of bins for the confidence distance embedding.
        max_distance: Upper distance bound (Angstrom) for the confidence distogram binning.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of PAE output bins.
        num_pde_bins: Number of PDE output bins.
        add_single_to_pair_product: Whether to add the single-input outer product into the pair track.
        add_single_input_to_single: Whether to inject the single-input representation into the single track.
        add_pair_input_to_pair: Whether to inject relative-position and bond features into the pair track.
        use_single_diffusion: Whether to condition on the accumulated final diffusion token representation.
        pair_dropout: Pair-track dropout for a trunk-reuse confidence head.
        update_single: Whether a trunk-reuse confidence head updates the single track.
        ranking_plddt_weight: Weight on complex pLDDT when forming the combined ranking score.
        ranking_ptm_weight: Weight on pTM/ipTM when forming the combined ranking score.
    """

    enabled: bool = True
    imitate_trunk: bool = True
    compute_pae: bool = True
    num_pairformer_blocks: int = 48
    num_pair_attention_heads: int = 16
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    add_single_to_pair_product: bool = True
    add_single_input_to_single: bool = True
    add_pair_input_to_pair: bool = True
    use_single_diffusion: bool = True
    pair_dropout: float = 0.25
    update_single: bool = True
    ranking_plddt_weight: float = 4.0
    ranking_ptm_weight: float = 1.0


class Boltz1Config(Config):
    r"""
    Configure the Boltz-1 structure-prediction architecture and inference schedule.

    Attributes:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single representation.
        atom_pair_hidden_size: Width of the windowed atom-pair representation.
        atom_feature_size: Expected width of the raw per-atom feature vector.
        num_distogram_bins: Number of bins for the trunk distogram head.
        fourier_embedding_size: Width of the frozen Fourier noise embedding.
        relative_position_size: Width of the relative-position feature fed to the position embedder.
        num_recycles: Total number of trunk passes.
        use_msa: Whether to build and run the MSA trunk.
        input_embeddings: Nested input-embeddings configuration.
        trunk: Nested representation-trunk configuration.
        denoiser: Nested EDM denoiser configuration.
        sampler: Nested reverse-diffusion sampler configuration.
        confidence: Nested confidence/ranking module configuration.

    Examples:
        >>> from ofoldx.models.boltz1 import Boltz1Config
        >>> config = Boltz1Config(num_recycles=2)
        >>> config.num_recycles
        2
    """

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 389
    num_distogram_bins: int = 64
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    num_recycles: int = 4
    use_msa: bool = True

    input_embeddings: Boltz1InputEmbeddingsConfig = Boltz1InputEmbeddingsConfig()
    trunk: Boltz1TrunkConfig = Boltz1TrunkConfig()
    denoiser: Boltz1DenoiserConfig = Boltz1DenoiserConfig()
    sampler: Boltz1SamplerConfig = Boltz1SamplerConfig()
    confidence: Boltz1ConfidenceConfig = Boltz1ConfidenceConfig()


class Boltz1ProcessorConfig(Config):
    r"""Configure Boltz-1 input processing.

    Attributes:
        seed: Seed used to embed missing reference conformers.
    """

    seed: int = 42
