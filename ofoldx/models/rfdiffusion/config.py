r"""
RfDiffusion configuration.
"""

from __future__ import annotations

from chanfig import Config

from ..registry import MODEL_CONFIGS


class RfDiffusionSE3Config(Config):
    r"""
    SE(3)-transformer sizing metadata stored by released RfDiffusion checkpoints.

    Mirrors the ``SE3_param_full`` block of the upstream ``config_dict``. The fields feed
    [`SE3TransformerWrapper`][ofoldx.models.rfdiffusion.model.SE3TransformerWrapper] and describe the
    per-layer channel widths, degrees, and edge dimension of the equivariant refinement network. The
    ``l0_*``/``l1_*`` names are SE(3)-standard degree labels and are carried through verbatim from the
    checkpoint rather than renamed to the house vocabulary.

    Attributes:
        num_layers: Number of SE(3)-transformer layers.
        num_channels: Hidden channel width of the equivariant fibers.
        num_degrees: Number of spherical-harmonic degrees (types) represented in the hidden fiber.
        num_attention_heads: Number of equivariant attention heads.
        div: Channel-division factor for the value fiber inside each attention block.
        l0_in_features: Width of the incoming degree-0 (scalar) node feature.
        l0_out_features: Width of the outgoing degree-0 (scalar) node feature; also the state width.
        l1_in_features: Number of incoming degree-1 (vector) node features.
        l1_out_features: Number of outgoing degree-1 (vector) node features.
        num_edge_features: Width of the scalar edge feature.

    Examples:
        >>> config = RfDiffusionSE3Config()
        >>> config.num_channels
        32
        >>> config.l0_out_features
        8
    """

    num_layers: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    num_attention_heads: int = 4
    div: int = 4
    l0_in_features: int = 8
    l0_out_features: int = 8
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 32


class RfDiffusionTopKSE3Config(RfDiffusionSE3Config):
    r"""
    SE(3)-transformer sizing for the top-k refinement stack of RfDiffusion.

    Same schema as [`RfDiffusionSE3Config`][ofoldx.models.rfdiffusion.config.RfDiffusionSE3Config]
    (the ``SE3_param_topk`` checkpoint block), but with the wider scalar and edge widths the top-k
    refinement blocks use. Only the overridden fields are listed; the rest are inherited.

    Attributes:
        l0_in_features: Width of the incoming degree-0 (scalar) node feature (`64`).
        l0_out_features: Width of the outgoing degree-0 (scalar) node feature (`64`); also the state width.
        num_edge_features: Width of the scalar edge feature (`64`).

    Examples:
        >>> config = RfDiffusionTopKSE3Config()
        >>> config.l0_out_features
        64
        >>> config.num_edge_features
        64
    """

    l0_in_features: int = 64
    l0_out_features: int = 64
    num_edge_features: int = 64


class RfDiffusionModelConfig(Config):
    r"""
    RoseTTAFold network sizing recorded in the RfDiffusion checkpoint ``config_dict["model"]``.

    Holds the architecture hyper-parameters of the RoseTTAFold trunk that
    [`RoseTTAFoldModule`][ofoldx.models.rfdiffusion.model.RoseTTAFoldModule] is built from. Upstream terse
    keys (``d_msa``, ``n_head_pair``, ...) are renamed to the house vocabulary during conversion; the mapping
    lives in [`convert`][ofoldx.models.rfdiffusion.convert]. Defaults match the released ``base`` checkpoint.

    Attributes:
        vocab_size: Number of amino-acid tokens the sequence head predicts over.
        num_extra_blocks: Number of extra-MSA IterBlocks in the trunk.
        num_main_blocks: Number of main IterBlocks in the trunk.
        num_refinement_blocks: Number of structure-refinement iterations after the main blocks.
        msa_size: Width of the latent MSA representation.
        full_msa_size: Width of the extra (full) MSA representation.
        pair_size: Width of the pair representation.
        template_size: Width of the template pair representation.
        num_msa_attention_heads: Number of attention heads in MSA attention.
        num_pair_attention_heads: Number of attention heads in pair (axial) attention.
        num_template_attention_heads: Number of attention heads in template attention.
        hidden_size: Per-head hidden width shared across the attention stacks.
        template_hidden_size: Per-head hidden width used inside the template stack.
        dropout: Dropout probability applied throughout the trunk.
        time_embedding_size: Width of the diffusion-time embedding (`0` when time is injected elsewhere).
        time_embedding_projection_size: Width of the time-embedding projection.
        se3_param_full: SE(3) sizing for the main-block structure module.
        se3_param_topk: SE(3) sizing for the top-k refinement structure module.
        freeze_track_motif: Whether to stop gradients on motif residues in the track.
        use_motif_timestep: Whether motif residues carry their own diffusion timestep feature.

    Examples:
        >>> config = RfDiffusionModelConfig()
        >>> config.num_main_blocks
        32
        >>> config.pair_size
        128
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
    time_embedding_size: int = 0
    time_embedding_projection_size: int = 10
    se3_param_full: RfDiffusionSE3Config = RfDiffusionSE3Config()
    se3_param_topk: RfDiffusionTopKSE3Config = RfDiffusionTopKSE3Config()
    freeze_track_motif: bool = False
    use_motif_timestep: bool = False


class RfDiffusionDiffuserConfig(Config):
    r"""
    Forward-diffusion (noising) schedule recorded in the checkpoint ``config_dict["diffuser"]``.

    Controls the coupled translation (Euclidean) and rotation (IGSO3) noise processes used by
    [`Diffuser`][ofoldx.models.rfdiffusion.model.Diffuser]. The upstream timestep keys ``T`` and
    ``partial_T`` are renamed to ``num_timesteps`` and ``partial_timesteps`` during conversion. Defaults match
    the released ``base`` checkpoint.

    Attributes:
        num_timesteps: Total number of diffusion timesteps.
        b_0: Initial translation-noise beta.
        b_T: Final translation-noise beta.
        schedule_type: Translation beta-schedule shape (only `"linear"` is supported).
        so3_type: Rotation diffuser family (only `"igso3"` is supported).
        crd_scale: Coordinate-scaling factor applied before noising.
        partial_timesteps: Number of steps for partial diffusion; `None` for full diffusion.
        so3_schedule_type: IGSO3 sigma-schedule shape (`"linear"` or `"exponential"`).
        min_b: Minimum IGSO3 drift coefficient (linear schedule).
        max_b: Maximum IGSO3 drift coefficient (linear schedule).
        min_sigma: Minimum IGSO3 noise sigma.
        max_sigma: Maximum IGSO3 noise sigma.
        truncation_level: Number of series terms used when precomputing IGSO3 densities.
        igso3_num_omega: Number of discretized rotation angles for the IGSO3 lookup tables.
        igso3_num_sigma: Number of discretized sigma levels for the IGSO3 lookup tables.

    Examples:
        >>> config = RfDiffusionDiffuserConfig()
        >>> config.num_timesteps
        50
        >>> config.schedule_type
        'linear'
    """

    num_timesteps: int = 50
    b_0: float = 1e-2
    b_T: float = 7e-2
    schedule_type: str = "linear"
    so3_type: str = "igso3"
    crd_scale: float = 0.25
    partial_timesteps: int | None = None
    so3_schedule_type: str = "linear"
    min_b: float = 1.5
    max_b: float = 2.5
    min_sigma: float = 0.02
    max_sigma: float = 1.5
    truncation_level: int = 2000
    igso3_num_omega: int = 1000
    igso3_num_sigma: int = 500


class RfDiffusionPreprocessConfig(Config):
    r"""
    Input-feature dimensions and self-conditioning flags from ``config_dict["preprocess"]``.

    Describes how the sampler assembles the RoseTTAFold network inputs each timestep (template feature
    widths, whether motif sidechains are fed in, and self-conditioning behavior). The upstream ``d_t1d`` and
    ``d_t2d`` keys are renamed to ``template_1d_size`` and ``template_2d_size`` during conversion.

    Attributes:
        sidechain_input: Whether design-residue sidechain coords are fed to the network.
        motif_sidechain_input: Whether fixed-motif sidechain coords are fed to the network.
        template_1d_size: Width of the 1D template feature; also gates hotspot/SS feature emission.
        template_2d_size: Width of the 2D template feature; also gates block-adjacency feature emission.
        prob_self_cond: Probability of applying self-conditioning during training.
        str_self_cond: Whether structural self-conditioning is enabled.
        predict_previous: Whether the network predicts the previous rather than the clean structure.

    Examples:
        >>> config = RfDiffusionPreprocessConfig()
        >>> config.template_1d_size
        22
        >>> config.motif_sidechain_input
        True
    """

    sidechain_input: bool = False
    motif_sidechain_input: bool = True
    template_1d_size: int = 22
    template_2d_size: int = 44
    prob_self_cond: float = 0.0
    str_self_cond: bool = False
    predict_previous: bool = False


class RfDiffusionDenoiserConfig(Config):
    r"""
    Reverse-diffusion (denoising) noise schedule consumed by the sampler.

    Sets the per-step noise scales the [`Denoiser`][ofoldx.models.rfdiffusion.model.Denoiser] injects
    when stepping from ``x_t`` toward ``x_0`` for CA translations and backbone frames.

    Attributes:
        noise_scale_ca: Initial noise scale for CA-translation reverse steps.
        final_noise_scale_ca: Final noise scale for CA-translation reverse steps.
        ca_noise_schedule_type: CA-translation noise-schedule shape (`"constant"` or `"linear"`).
        noise_scale_frame: Initial noise scale for backbone-frame (rotation) reverse steps.
        final_noise_scale_frame: Final noise scale for backbone-frame reverse steps.
        frame_noise_schedule_type: Frame noise-schedule shape (`"constant"` or `"linear"`).

    Examples:
        >>> config = RfDiffusionDenoiserConfig()
        >>> config.ca_noise_schedule_type
        'constant'
        >>> config.noise_scale_frame
        1.0
    """

    noise_scale_ca: float = 1.0
    final_noise_scale_ca: float = 1.0
    ca_noise_schedule_type: str = "constant"
    noise_scale_frame: float = 1.0
    final_noise_scale_frame: float = 1.0
    frame_noise_schedule_type: str = "constant"


class RfDiffusionInferenceConfig(Config):
    r"""
    Design-time sampler controls (I/O, symmetry, self-conditioning, and step bounds).

    Populated from ``config_dict["inference"]`` and overridable at call time. These knobs steer the sampling
    loop of [`NativeRfDiffusionSampler`][ofoldx.models.rfdiffusion.model.NativeRfDiffusionSampler]
    rather than the network weights.

    Attributes:
        input_pdb: Path to the target PDB, or `None` to sample from batch tensors.
        num_designs: Number of designs to generate.
        design_startnum: Starting index for design output naming.
        ckpt_override_path: Optional checkpoint path overriding the configured weights.
        symmetry: Global symmetry group (e.g. `"c3"`), or `None` for asymmetric design.
        recenter: Whether to recenter symmetric subunits.
        radius: Placement radius for polyhedral symmetry subunits.
        model_only_neighbors: Whether to feed only neighboring subunits to the network.
        output_prefix: Filesystem prefix for written designs.
        write_trajectory: Whether to write the full denoising trajectory.
        empty_cache_per_design: Whether to empty the CUDA cache between designs.
        scaffold_guided: Whether scaffold-guided sampling is requested.
        model_runner: Sampler runner name (e.g. `"SelfConditioning"`).
        cautious: Whether to skip designs whose output files already exist.
        align_motif: Whether to Kabsch-align predictions onto the fixed motif each step.
        symmetric_self_cond: Whether self-conditioning respects symmetry.
        final_step: Last (smallest) timestep index to denoise to.
        deterministic: Whether to seed RNGs per design for reproducibility.
        trb_save_ckpt_path: Optional path to save the run metadata (.trb) file.
        schedule_directory_path: Directory for cached IGSO3 schedule tables.
        model_directory_path: Directory containing model weights.
        cyclic: Whether to apply cyclic-peptide backbone constraints.
        cyc_chains: Chain labels to cyclize (lowercase).

    Examples:
        >>> config = RfDiffusionInferenceConfig()
        >>> config.num_designs
        1
        >>> config.model_runner
        'SelfConditioning'
    """

    input_pdb: str | None = None
    num_designs: int = 1
    design_startnum: int = 0
    ckpt_override_path: str | None = None
    symmetry: str | None = None
    recenter: bool = True
    radius: float = 10.0
    model_only_neighbors: bool = False
    output_prefix: str = "samples/design"
    write_trajectory: bool = False
    empty_cache_per_design: bool = False
    scaffold_guided: bool = False
    model_runner: str = "SelfConditioning"
    cautious: bool = True
    align_motif: bool = True
    symmetric_self_cond: bool = True
    final_step: int = 1
    deterministic: bool = False
    trb_save_ckpt_path: str | None = None
    schedule_directory_path: str | None = None
    model_directory_path: str | None = None
    cyclic: bool = False
    cyc_chains: str = "a"


class RfDiffusionContigMapConfig(Config):
    r"""
    Contig and inpainting specification for placing motif and designed segments.

    Feeds [`ContigMap`][ofoldx.models.rfdiffusion.model.ContigMap], which resolves these contig strings
    into reference, output, and RF-relative residue indices. Sequence- and structure-inpainting masks may be
    given globally or per secondary-structure class.

    Attributes:
        contigs: Contig strings describing motif segments and variable-length gaps.
        inpaint_seq: Residue spans whose sequence is masked (redesigned).
        inpaint_str: Residue spans whose structure is masked (rebuilt).
        inpaint_str_helix: Structure-inpainting spans forced to helix.
        inpaint_str_strand: Structure-inpainting spans forced to strand.
        inpaint_str_loop: Structure-inpainting spans forced to loop.
        provide_seq: Residue spans whose sequence is provided (unmasked).
        length: Target total design length or range, as a spec string.

    Examples:
        >>> config = RfDiffusionContigMapConfig()
        >>> config.contigs is None
        True
    """

    contigs: list[str] | None = None
    inpaint_seq: list[str] | None = None
    inpaint_str: list[str] | None = None
    inpaint_str_helix: list[str] | None = None
    inpaint_str_strand: list[str] | None = None
    inpaint_str_loop: list[str] | None = None
    provide_seq: list[str] | None = None
    length: str | None = None


class RfDiffusionPPIConfig(Config):
    r"""
    Protein-protein-interface conditioning via hotspot residues.

    Names target residues the designed binder should contact; the sampler emits a hotspot feature for them
    when the checkpoint's template feature width provides for it.

    Attributes:
        hotspot_res: Hotspot residue ids as `"<chain><number>"` strings (e.g. `"A30"`), or `None`.

    Examples:
        >>> config = RfDiffusionPPIConfig()
        >>> config.hotspot_res is None
        True
    """

    hotspot_res: list[str] | None = None


class RfDiffusionPotentialsConfig(Config):
    r"""
    Guiding-potential controls that bias the reverse-diffusion trajectory.

    Parsed by [`PotentialManager`][ofoldx.models.rfdiffusion.model.PotentialManager] into scalar
    potentials whose gradients nudge CA coords toward user goals (radius of gyration, contacts, substrate
    proximity). Each potential string uses the terse upstream token form (e.g. ``"type:monomer_ROG,weight:1"``).

    Attributes:
        guiding_potentials: Potential specification strings, or `None` for no guidance.
        guide_scale: Overall gradient scale applied to the summed potential.
        guide_decay: Time-decay shape for ``guide_scale`` (`"constant"`, `"linear"`, `"quadratic"`, `"cubic"`).
        olig_inter_all: Force all inter-chain oligomer contacts when set.
        olig_intra_all: Force all intra-chain oligomer contacts when set.
        olig_custom_contact: Custom oligomer contact specification string.
        substrate: Ligand residue name targeted by the ``substrate_contacts`` potential.

    Examples:
        >>> config = RfDiffusionPotentialsConfig()
        >>> config.guide_scale
        10.0
        >>> config.guide_decay
        'constant'
    """

    guiding_potentials: list[str] | None = None
    guide_scale: float = 10.0
    guide_decay: str = "constant"
    olig_inter_all: float | None = None
    olig_intra_all: float | None = None
    olig_custom_contact: str | None = None
    substrate: str | None = None


class RfDiffusionScaffoldGuidedConfig(Config):
    r"""
    Secondary-structure and block-adjacency guided (fold-conditioned) sampling controls.

    Drives the scaffold branch of the sampler
    ([`BlockAdjacency`][ofoldx.models.rfdiffusion.model.BlockAdjacency] and
    [`Target`][ofoldx.models.rfdiffusion.model.Target]): sample a fold topology from precomputed
    secondary-structure/adjacency tensors, optionally conditioned on a fixed target.

    Attributes:
        scaffoldguided: Master switch enabling scaffold-guided sampling.
        target_pdb: Whether a fixed target PDB conditions the design.
        target_path: Path to the target PDB.
        scaffold_list: List of scaffold names or a `.txt` path listing them.
        scaffold_dir: Directory holding `*_ss.pt` / `*_adj.pt` scaffold tensors.
        sampled_insertion: Range spec for random insertions into scaffold loops.
        sampled_N: Range spec for random N-terminal extension length.
        sampled_C: Range spec for random C-terminal extension length.
        ss_mask: Number of residues to erode from secondary-structure segment ends.
        systematic: Whether to iterate scaffolds in order rather than sampling randomly.
        target_ss: Path to a precomputed target secondary-structure tensor.
        target_adj: Path to a precomputed target block-adjacency tensor.
        mask_loops: Whether loop residues are masked in the SS/adjacency features.
        contig_crop: Contig-crop specification restricting the target to a region.

    Examples:
        >>> config = RfDiffusionScaffoldGuidedConfig()
        >>> config.scaffoldguided
        False
        >>> config.mask_loops
        True
    """

    scaffoldguided: bool = False
    target_pdb: bool = False
    target_path: str | None = None
    scaffold_list: str | None = None
    scaffold_dir: str | None = None
    sampled_insertion: int = 0
    sampled_N: int = 0
    sampled_C: int = 0
    ss_mask: int = 0
    systematic: bool = False
    target_ss: str | None = None
    target_adj: str | None = None
    mask_loops: bool = True
    contig_crop: str | None = None


class RfDiffusionLoggingConfig(Config):
    r"""
    Debug-logging controls carried through from the checkpoint ``config_dict``.

    Attributes:
        inputs: Whether to log the per-step network input tensors.

    Examples:
        >>> config = RfDiffusionLoggingConfig()
        >>> config.inputs
        False
    """

    inputs: bool = False


@MODEL_CONFIGS.register("rfdiffusion")
class RfDiffusionConfig(Config):
    r"""
    Top-level configuration for the OFoldX-native RfDiffusion design generator.

    Aggregates the per-concern config groups (model sizing, diffuser/denoiser schedules, preprocessing,
    inference controls, contigs, PPI, potentials, scaffold guidance, and logging). The groups mirror the
    released RosettaCommons checkpoint ``config_dict`` so converted artifacts preserve training-time model,
    diffuser, and preprocessing settings. Building
    [`RfDiffusionDesignGenerator`][ofoldx.models.rfdiffusion.model.RfDiffusionDesignGenerator] with the
    defaults reproduces the reference ``base`` architecture.

    Args:
        *args: Positional overrides forwarded to the base `Config` (e.g. a mapping of nested settings);
            when both ``args`` and ``kwargs`` are empty the default scaffold-guidance group is installed.
        **kwargs: Keyword overrides forwarded to the base `Config`, keyed by attribute name.

    Attributes:
        architecture: Registry key selecting this architecture (`"rfdiffusion"`).
        source: Upstream repository the checkpoints originate from.
        checkpoint_variant: Released checkpoint tag once a checkpoint is loaded (default `"base"`).
        model: RoseTTAFold trunk sizing
            ([`RfDiffusionModelConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionModelConfig]).
        diffuser: Forward-noising schedule
            ([`RfDiffusionDiffuserConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionDiffuserConfig]).
        preprocess: Input-feature dimensions
            ([`RfDiffusionPreprocessConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionPreprocessConfig]).
        denoiser: Reverse-noising schedule
            ([`RfDiffusionDenoiserConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionDenoiserConfig]).
        inference: Design-time sampler controls
            ([`RfDiffusionInferenceConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionInferenceConfig]).
        contigmap: Contig/inpainting specification
            ([`RfDiffusionContigMapConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionContigMapConfig]).
        ppi: Protein-protein-interface hotspots
            ([`RfDiffusionPPIConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionPPIConfig]).
        potentials: Guiding-potential controls
            ([`RfDiffusionPotentialsConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionPotentialsConfig]).
        scaffoldguided: Fold-conditioned sampling controls
            ([`RfDiffusionScaffoldGuidedConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionScaffoldGuidedConfig]).
        logging: Debug-logging controls
            ([`RfDiffusionLoggingConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionLoggingConfig]).

    Examples:
        >>> config = RfDiffusionConfig()
        >>> config.architecture
        'rfdiffusion'
        >>> config.diffuser.num_timesteps
        50
        >>> model = RfDiffusionDesignGenerator(config)  # doctest: +SKIP
    """

    architecture: str = "rfdiffusion"
    source: str = "RosettaCommons/RFdiffusion"
    checkpoint_variant: str = "base"

    model: RfDiffusionModelConfig = RfDiffusionModelConfig()
    diffuser: RfDiffusionDiffuserConfig = RfDiffusionDiffuserConfig()
    preprocess: RfDiffusionPreprocessConfig = RfDiffusionPreprocessConfig()
    denoiser: RfDiffusionDenoiserConfig = RfDiffusionDenoiserConfig()
    inference: RfDiffusionInferenceConfig = RfDiffusionInferenceConfig()
    contigmap: RfDiffusionContigMapConfig = RfDiffusionContigMapConfig()
    ppi: RfDiffusionPPIConfig = RfDiffusionPPIConfig()
    potentials: RfDiffusionPotentialsConfig = RfDiffusionPotentialsConfig()
    scaffoldguided: RfDiffusionScaffoldGuidedConfig = RfDiffusionScaffoldGuidedConfig()
    logging: RfDiffusionLoggingConfig = RfDiffusionLoggingConfig()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not args and not kwargs:
            self["scaffoldguided"] = RfDiffusionScaffoldGuidedConfig()
