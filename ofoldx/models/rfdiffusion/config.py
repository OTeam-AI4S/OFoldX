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
RFdiffusion model and processor configuration.
"""

from __future__ import annotations

from typing import Literal

from chanfig import Config


class RfDiffusionSe3Config(Config):
    r"""
    Configure one SE(3)-Transformer stack.

    Attributes:
        num_blocks: Number of SE(3)-Transformer blocks.
        hidden_size: Number of channels per irreducible representation.
        num_degrees: Number of representation degrees.
        num_attention_heads: Number of equivariant attention heads.
        channel_reduction_factor: Divisor used to size intermediate equivariant channels.
        input_scalar_size: Number of scalar input features.
        output_scalar_size: Number of scalar output features.
        input_vector_size: Number of vector input features.
        output_vector_size: Number of vector output features.
        edge_size: Width of edge features.
    """

    num_blocks: int = 1
    hidden_size: int = 32
    num_degrees: int = 2
    num_attention_heads: int = 4
    channel_reduction_factor: int = 4
    input_scalar_size: int = 8
    output_scalar_size: int = 8
    input_vector_size: int = 3
    output_vector_size: int = 2
    edge_size: int = 32


class RfDiffusionDenoiserConfig(Config):
    r"""
    Configure the RoseTTAFold denoiser architecture.

    Attributes:
        num_extra_blocks: Number of extra-MSA blocks.
        num_main_blocks: Number of main RoseTTAFold blocks.
        num_refinement_blocks: Number of structural refinement blocks.
        msa_size: Width of the clustered MSA representation.
        full_msa_size: Width of the full-MSA representation.
        pair_size: Width of the pair representation.
        template_size: Width of template representations.
        num_msa_attention_heads: Number of MSA attention heads.
        num_pair_attention_heads: Number of pair attention heads.
        num_template_attention_heads: Number of template attention heads.
        hidden_size: Width of projected structure features.
        template_hidden_size: Width of projected template features.
        dropout: Dropout probability in denoiser blocks.
        trunk_se3: SE(3)-Transformer used in the main trunk.
        refinement_se3: Wider SE(3)-Transformer used for structural refinement.
        freeze_track_motif: Whether motif residues remain fixed in the structure track.
    """

    num_extra_blocks: int = 4
    num_main_blocks: int = 32
    num_refinement_blocks: int = 4
    msa_size: int = 256
    full_msa_size: int = 64
    pair_size: int = 128
    template_size: int = 64
    num_msa_attention_heads: int = 8
    num_pair_attention_heads: int = 4
    num_template_attention_heads: int = 4
    hidden_size: int = 32
    template_hidden_size: int = 32
    dropout: float = 0.15
    trunk_se3: RfDiffusionSe3Config = RfDiffusionSe3Config()
    refinement_se3: RfDiffusionSe3Config = RfDiffusionSe3Config(
        {
            "input_scalar_size": 64,
            "output_scalar_size": 64,
            "edge_size": 64,
        }
    )
    freeze_track_motif: bool = False


class RfDiffusionDiffuserConfig(Config):
    r"""
    Configure the RFdiffusion forward-diffusion process.

    Attributes:
        num_timesteps: Number of discrete diffusion timesteps.
        beta_initial: Initial translation variance.
        beta_final: Final translation variance.
        coordinate_scale: Scale applied to Cartesian coordinates.
        rotation_schedule_type: Rotation variance schedule.
        min_rotation_beta: Minimum rotation beta.
        max_rotation_beta: Maximum rotation beta.
        min_rotation_sigma: Minimum rotation noise scale.
        max_rotation_sigma: Maximum rotation noise scale for the exponential
            schedule. The linear schedule derives its endpoint from the beta
            integral and does not consume this value.
    """

    num_timesteps: int = 50
    beta_initial: float = 1e-2
    beta_final: float = 7e-2
    coordinate_scale: float = 0.25
    rotation_schedule_type: Literal["linear", "exponential"] = "linear"
    min_rotation_beta: float = 1.5
    max_rotation_beta: float = 2.5
    min_rotation_sigma: float = 0.02
    max_rotation_sigma: float = 1.5


class RfDiffusionPreprocessConfig(Config):
    r"""
    Configure the denoiser input feature schema.

    Attributes:
        sidechain_input: Whether to include general sidechain coordinates.
        motif_sidechain_input: Whether to include motif sidechain coordinates.
        template_single_feature_size: Width of per-token template features.
        template_pair_feature_size: Width of pairwise template features.
        structure_self_conditioning: Whether preprocessing exposes self-conditioned structure features.
    """

    sidechain_input: bool = False
    motif_sidechain_input: bool = True
    template_single_feature_size: int = 22
    template_pair_feature_size: int = 44
    structure_self_conditioning: bool = False


class RfDiffusionConfig(Config):
    r"""
    Configure an RFdiffusion denoiser architecture.

    Attributes:
        denoiser: RoseTTAFold denoiser architecture.
        diffuser: Forward-diffusion process.
        preprocess: Denoiser input feature schema.

    Examples:
        >>> from ofoldx.models.rfdiffusion import RfDiffusionConfig
        >>> config = RfDiffusionConfig()
        >>> config.diffuser.num_timesteps
        50
    """

    denoiser: RfDiffusionDenoiserConfig = RfDiffusionDenoiserConfig()
    diffuser: RfDiffusionDiffuserConfig = RfDiffusionDiffuserConfig()
    preprocess: RfDiffusionPreprocessConfig = RfDiffusionPreprocessConfig()
