r"""
RFdiffusion model and processor configuration.
"""

from __future__ import annotations

from chanfig import Config


class RfDiffusionSe3Config(Config):
    r"""
    Configure one SE(3)-Transformer stack.

    Args:
        num_blocks: Number of SE(3)-Transformer blocks.
        num_channels: Number of channels per irreducible representation.
        num_degrees: Number of representation degrees.
        num_attention_heads: Number of equivariant attention heads.
        div: Divisor used to size intermediate equivariant channels.
        l0_in_features: Number of scalar input features.
        l0_out_features: Number of scalar output features.
        l1_in_features: Number of vector input features.
        l1_out_features: Number of vector output features.
        num_edge_features: Width of edge features.
    """

    num_blocks: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    num_attention_heads: int = 4
    div: int = 4
    l0_in_features: int = 8
    l0_out_features: int = 8
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 32


class RfDiffusionDenoiserConfig(Config):
    r"""
    Configure the RoseTTAFold denoiser architecture.

    Args:
        vocab_size: Number of sequence token types.
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

    vocab_size: int = 21
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
            "l0_in_features": 64,
            "l0_out_features": 64,
            "num_edge_features": 64,
        }
    )
    freeze_track_motif: bool = False


class RfDiffusionDiffuserConfig(Config):
    r"""
    Configure the RFdiffusion forward-diffusion process.

    Args:
        num_timesteps: Number of discrete diffusion timesteps.
        beta_initial: Initial translation variance.
        beta_final: Final translation variance.
        schedule_type: Translation variance schedule.
        rotation_diffuser: Rotational diffusion formulation.
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
    schedule_type: str = "linear"
    rotation_diffuser: str = "igso3"
    coordinate_scale: float = 0.25
    rotation_schedule_type: str = "linear"
    min_rotation_beta: float = 1.5
    max_rotation_beta: float = 2.5
    min_rotation_sigma: float = 0.02
    max_rotation_sigma: float = 1.5


class RfDiffusionPreprocessConfig(Config):
    r"""
    Configure the denoiser input feature schema.

    Args:
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

    Args:
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

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self) -> None:
        r"""
        Validate the supported diffusion schedule and feature dimensions.

        Raises:
            ValueError: If a schedule type or feature width is unsupported.
        """

        super().validate()
        if self.diffuser.schedule_type != "linear":
            raise ValueError("RFdiffusion supports the linear translation schedule")
        if self.diffuser.rotation_diffuser != "igso3":
            raise ValueError("RFdiffusion supports the IGSO(3) rotation diffuser")
        if self.preprocess.template_single_feature_size not in (22, 24, 28):
            raise ValueError("preprocess.template_single_feature_size must be 22, 24, or 28")
        if self.preprocess.template_pair_feature_size not in (44, 47):
            raise ValueError("preprocess.template_pair_feature_size must be 44 or 47")
