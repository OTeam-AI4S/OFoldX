r"""Persistent RFdiffusion model and processor configuration."""

from __future__ import annotations

from chanfig import Config


class RfDiffusionSe3Config(Config):
    r"""Architecture of one released SE(3)-Transformer stack."""

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
    r"""RoseTTAFold denoiser dimensions stored in official checkpoints."""

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
    r"""Forward-diffusion process the checkpoint was trained against."""

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
    r"""Checkpoint-dependent input feature schema."""

    sidechain_input: bool = False
    motif_sidechain_input: bool = True
    template_1d_size: int = 22
    template_2d_size: int = 44
    structure_self_conditioning: bool = False


class RfDiffusionConfig(Config):
    r"""Exact persistent configuration of one RFdiffusion denoiser.

    Sampling steps, noise scales, symmetry, conditions, paths, placement, logging,
    backend selection, and training-loop controls are deliberately absent.  They
    are execution inputs, not checkpoint architecture.
    """

    denoiser: RfDiffusionDenoiserConfig = RfDiffusionDenoiserConfig()
    diffuser: RfDiffusionDiffuserConfig = RfDiffusionDiffuserConfig()
    preprocess: RfDiffusionPreprocessConfig = RfDiffusionPreprocessConfig()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self) -> None:
        super().validate()
        if self.diffuser.schedule_type != "linear":
            raise ValueError("RFdiffusion supports the released linear translation schedule")
        if self.diffuser.rotation_diffuser != "igso3":
            raise ValueError("RFdiffusion supports the released IGSO(3) rotation diffuser")
        if self.preprocess.template_1d_size not in (22, 24, 28):
            raise ValueError("preprocess.template_1d_size must be 22, 24, or 28")
        if self.preprocess.template_2d_size not in (44, 47):
            raise ValueError("preprocess.template_2d_size must be 44 or 47")


class RfDiffusionProcessorConfig(Config):
    r"""RFdiffusion lowering has no persistent options."""
