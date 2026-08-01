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
Boltz-2 architecture and inference configuration.
"""

from __future__ import annotations

from chanfig import Config

from ...data.tokenization import TOKENS


class Boltz2InputEmbeddingsConfig(Config):
    r"""
    Configure the independent Boltz-2 atom input encoder and token conditioning.

    Attributes:
        num_atom_encoder_blocks: Number of atom-attention encoder blocks.
        num_atom_encoder_attention_heads: Head count for the atom-attention encoder.
        vocab_size: Residue-type vocabulary size.
        num_method_types: Number of experimental-method ids for method conditioning.
        num_molecule_types: Number of molecule types for molecule-type conditioning.
        add_method_conditioning: Whether to add the experimental-method embedding.
        add_modified_flag: Whether to add the modified-residue embedding.
        add_cyclic_flag: Whether to add the cyclic-position projection.
        add_mol_type_feat: Whether to add the molecule-type embedding.
    """

    num_atom_encoder_blocks: int = 3
    num_atom_encoder_attention_heads: int = 4
    vocab_size: int = len(TOKENS)
    num_method_types: int = 12
    num_molecule_types: int = 4
    add_method_conditioning: bool = True
    add_modified_flag: bool = True
    add_cyclic_flag: bool = True
    add_mol_type_feat: bool = True


class Boltz2TrunkConfig(Config):
    r"""
    Configure the Boltz-2 Pairformer and MSA representation trunk.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the trunk stack.
        num_pair_attention_heads: Head count for the trunk pair-biased attention.
        num_msa_blocks: Number of MSA-module blocks.
        msa_size: Width of the MSA representation channel.
        msa_attention_head_size: Per-head dimension of the MSA attention.
        num_msa_attention_heads: Pair-weighted-averaging head count.
        outer_product_hidden_size: Hidden width of the outer-product-mean block.
        msa_triangle_attention_head_size: Per-head width of the MSA triangle attention.
        num_msa_triangle_attention_heads: Head count for the MSA triangle attention.
        triangle_attention_head_size: Per-head width of the triangle attention.
        num_triangle_attention_heads: Triangle-attention head count.
        use_paired_feature: Whether the paired-MSA feature is consumed.
        pair_dropout: Pair-track dropout for the Pairformer.
        msa_dropout: Dropout applied inside the MSA trunk.
        msa_pair_dropout: Pair-track dropout applied inside the MSA trunk.
        post_layer_norm: Whether each Pairformer block normalizes its updated single track.
    """

    num_pairformer_blocks: int = 64
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
    use_paired_feature: bool = True
    pair_dropout: float = 0.25
    msa_dropout: float = 0.15
    msa_pair_dropout: float = 0.25
    post_layer_norm: bool = False


class Boltz2TemplateConfig(Config):
    r"""
    Configure Boltz-2 on-the-fly template distance and unit-vector embedding.

    Attributes:
        hidden_size: Width of the template pair hidden channel.
        num_blocks: Number of pair-only triangle blocks in the template stack.
        num_distogram_bins: Number of on-the-fly distogram bins.
        vocab_size: Residue-type vocabulary size feeding the template projection.
        expansion: Transition expansion factor.
        triangle_attention_head_size: Per-head width of the template triangle attention.
        num_triangle_attention_heads: Template triangle-attention head count.
        dropout: Dropout probability in the template pair-update stack.
        min_distance: Lower edge of the distogram binning range, in angstroms.
        max_distance: Upper edge of the distogram binning range, in angstroms.
    """

    hidden_size: int = 64
    num_blocks: int = 2
    num_distogram_bins: int = 38
    vocab_size: int = len(TOKENS)
    expansion: int = 4
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    dropout: float = 0.25
    min_distance: float = 3.25
    max_distance: float = 50.75


class Boltz2DenoiserConfig(Config):
    r"""
    Configure the Boltz-2 learned coordinate denoiser.

    Attributes:
        num_token_encoder_blocks: Number of conditioned token-encoder blocks in the score model.
        num_token_encoder_attention_heads: Head count for the token encoder.
        num_atom_encoder_blocks: Number of atom-attention encoder blocks in the score model.
        num_atom_decoder_blocks: Number of atom-attention decoder blocks in the score model.
        num_atom_attention_heads: Head count shared by the atom encoder and decoder.
        post_layer_norm: Whether denoiser atom-encoder blocks apply post-layer normalization.
        atom_hidden_size: Width of the atom single channel.
        atom_pair_hidden_size: Width of the atom pair channel.
        num_query_atoms: Query-window size for windowed atom attention.
        num_key_atoms: Key-window size for windowed atom attention.
        num_conditioning_transitions: Number of transitions in each diffusion conditioner.
        sigma_data: EDM data standard deviation used for preconditioning.
        alignment_reverse_diff: Whether to weighted-rigid-align each reverse-diffusion step.
    """

    num_token_encoder_blocks: int = 24
    num_token_encoder_attention_heads: int = 16
    num_atom_encoder_blocks: int = 3
    num_atom_decoder_blocks: int = 3
    num_atom_attention_heads: int = 4
    post_layer_norm: bool = False
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    num_query_atoms: int = 32
    num_key_atoms: int = 128
    num_conditioning_transitions: int = 2
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True


class Boltz2SamplerConfig(Config):
    r"""
    Configure Boltz-2 Euler sampling with in-loop weighted rigid alignment.

    Attributes:
        num_sampling_steps: Number of EDM sampling steps.
        rho: Exponent of the Karras noise schedule.
        max_noise_level: Highest sigma at the start of the schedule.
        min_noise_level: Lowest non-zero sigma before the trailing zero.
        step_scale: Scale on the Euler update step.
        noise_scale: Multiplier on the injected churn noise.
        churn_rate: Stochastic churn added to each step above the churn threshold.
    """

    num_sampling_steps: int = 200
    rho: float = 7.0
    max_noise_level: float = 160.0
    min_noise_level: float = 1e-4
    step_scale: float = 1.5
    noise_scale: float = 1.003
    churn_rate: float = 0.8


class Boltz2ConfidenceConfig(Config):
    r"""
    Configure token-level Boltz-2 confidence with separate intra/inter PAE and PDE.

    Attributes:
        num_pairformer_blocks: Number of Pairformer blocks in the confidence head.
        num_pair_attention_heads: Head count for the confidence pair-biased attention.
        triangle_attention_head_size: Per-head width of the confidence triangle attention.
        num_triangle_attention_heads: Confidence triangle-attention head count.
        num_distogram_bins: Number of distogram bins used to bin representative-atom distances.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_plddt_bins: Number of pLDDT output bins.
        num_pae_bins: Number of predicted-aligned-error output bins.
        num_pde_bins: Number of predicted-distance-error output bins.
        pair_dropout: Pair-track dropout for the confidence Pairformer.
        post_layer_norm: Whether each confidence Pairformer block normalizes its updated single track.
        normalize_single_input: Whether to normalize the incoming trunk single representation.
    """

    num_pairformer_blocks: int = 8
    num_pair_attention_heads: int = 16
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_plddt_bins: int = 50
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    pair_dropout: float = 0.25
    post_layer_norm: bool = False
    normalize_single_input: bool = True


class Boltz2AffinityHeadConfig(Config):
    r"""
    Configure one Boltz-2 masked-pair affinity head and its triangle pair updates.

    Attributes:
        num_pair_update_blocks: Number of triangle pair-update blocks in the affinity head.
        num_distogram_bins: Number of distogram bins used to condition the pair features.
        max_distance: Upper edge of the distogram binning range, in angstroms.
        num_conditioning_transitions: Number of transitions in the pairwise conditioner.
        expansion: Transition expansion factor in the pairwise conditioner.
        triangle_attention_head_size: Per-head width of the affinity triangle attention.
        num_triangle_attention_heads: Affinity triangle-attention head count.
        dropout: Dropout applied in the triangle pair-update stack.
    """

    num_pair_update_blocks: int = 4
    num_distogram_bins: int = 64
    max_distance: float = 22.0
    num_conditioning_transitions: int = 2
    expansion: int = 2
    triangle_attention_head_size: int = 32
    num_triangle_attention_heads: int = 4
    dropout: float = 0.25


class Boltz2Config(Config):
    r"""
    Configure the Boltz-2 folding and confidence architecture.

    Attributes:
        single_size: Width of the token (single) representation.
        pair_size: Width of the pair representation.
        atom_hidden_size: Width of the atom single channel in the input embedder.
        atom_pair_hidden_size: Width of the atom pair channel in the input embedder.
        atom_feature_size: Width of the per-atom reference feature vector.
        num_distogram_bins: Number of distogram / B-factor output bins.
        fourier_embedding_size: Width of the Fourier noise-level embedding.
        relative_position_size: Width of the relative-position feature vector.
        num_bond_types: Bond-type vocabulary size.
        num_recycles: Number of trunk recycling passes.
        use_templates: Whether template embedding is enabled.
        predict_b_factor: Whether the B-factor head is instantiated.
        min_conditioning_cutoff: Lower distance cutoff for contact conditioning, in angstroms.
        max_conditioning_cutoff: Upper distance cutoff for contact conditioning, in angstroms.
        input_embeddings: Input-embeddings sub-config.
        trunk: Representation-trunk sub-config.
        template: Template-embedder sub-config.
        denoiser: Learned coordinate-denoiser sub-config.
        sampler: EDM-sampler sub-config.
        confidence: Pairformer confidence-module sub-config.

    Examples:
        >>> from ofoldx.models.boltz2 import Boltz2Config
        >>> config = Boltz2Config(num_recycles=2)
        >>> config.num_recycles
        2
    """

    single_size: int = 384
    pair_size: int = 128
    atom_hidden_size: int = 128
    atom_pair_hidden_size: int = 16
    atom_feature_size: int = 388
    num_distogram_bins: int = 64
    fourier_embedding_size: int = 256
    relative_position_size: int = 139
    num_bond_types: int = 7
    num_recycles: int = 4

    use_templates: bool = True
    predict_b_factor: bool = True

    min_conditioning_cutoff: float = 4.0
    max_conditioning_cutoff: float = 20.0
    input_embeddings: Boltz2InputEmbeddingsConfig = Boltz2InputEmbeddingsConfig()
    trunk: Boltz2TrunkConfig = Boltz2TrunkConfig()
    template: Boltz2TemplateConfig = Boltz2TemplateConfig()
    denoiser: Boltz2DenoiserConfig = Boltz2DenoiserConfig()
    sampler: Boltz2SamplerConfig = Boltz2SamplerConfig()
    confidence: Boltz2ConfidenceConfig = Boltz2ConfidenceConfig()


class Boltz2AffinityConfig(Boltz2Config):
    r"""
    Configure affinity prediction by overriding the inherited Boltz-2 graph and adding two fixed heads.

    Attributes:
        use_templates: Whether to run the inherited template embedder; fixed to ``False`` for affinity.
        predict_b_factor: Whether to build the inherited B-factor head; fixed to ``False`` for affinity.
        num_recycles: Number of trunk recycling passes; overridden to six for affinity.
        affinity_heads: Two affinity-head configs in ensemble order.

    Examples:
        >>> from ofoldx.models.boltz2 import Boltz2AffinityConfig
        >>> config = Boltz2AffinityConfig()
        >>> config.num_recycles
        6
    """

    use_templates: bool = False
    predict_b_factor: bool = False
    num_recycles: int = 6

    affinity_heads: tuple[Boltz2AffinityHeadConfig, Boltz2AffinityHeadConfig] = (
        Boltz2AffinityHeadConfig({"num_pair_update_blocks": 8}),
        Boltz2AffinityHeadConfig({"num_pair_update_blocks": 4}),
    )


class Boltz2ProcessorConfig(Config):
    r"""Configure Boltz-2 folding and affinity input processing.

    Attributes:
        seed: Seed used to embed missing reference conformers.
    """

    seed: int = 42
