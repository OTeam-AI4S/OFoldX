"""Boltz-1 real-denoiser and reverse-sampler-mechanics parity.

Mirrors ``scripts.parity.boltz2.sampler_parity``. It first compares the real checkpoint denoiser
(``preconditioned_network_forward``) at fixed noise levels, driving the OFoldX
:class:`~ofoldx.models.boltz1.model.Boltz1DiffusionScoreModel` + precomputed
:class:`~ofoldx.models.boltz1.model.DiffusionConditioning` against the released Boltz-1
``AtomDiffusion`` (whose ``DiffusionModule`` recomputes conditioning internally) on shared trunk reps and
features. It then compares the reverse-sampler mechanics under the released public inference parameters
(``BoltzDiffusionParams``: ``gamma_0=0.605 gamma_min=1.107 noise_scale=0.901 rho=8 step_scale=1.638``),
identical injected randomness, and one identical deterministic denoiser. This decomposition isolates the
EDM noise schedule, the stochastic-churn ``sigma_hat``/``noise_var`` scalars, the weighted-rigid-align, and
the Euler update from denoiser roundoff.

The reference and OFoldX stacks share the zyc environment, so this runs in one process (CPU/fp32).

    python -m scripts.parity.boltz1.sampler_parity
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity.boltz1.forward_parity import (  # noqa: E402
    B,
    M,
    N,
    _ensure_fairscale_stub,
    allo_input,
    build_feats,
    load_checkpoint,
    subset,
)
from scripts.parity.paths import reference_root, weight_path  # noqa: E402

REFERENCE_SRC = reference_root("boltz") / "src"
if str(REFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SRC))

TOKEN_S = 384


def report(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-3) -> float:
    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape:
        print(f"[SHAPE] {name}: {tuple(actual.shape)} vs {tuple(expected.shape)}", flush=True)
        return float("inf")
    delta = (actual - expected).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    mean_abs = float(delta.mean()) if delta.numel() else 0.0
    status = "ok " if max_abs < atol else "BAD"
    print(f"[{status}] {name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} shape={tuple(actual.shape)}", flush=True)
    return max_abs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=weight_path("boltz1_conf.ckpt"))
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sampler-seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def released_inference_diffusion_args() -> dict[str, object]:
    r"""The exact public Boltz-1 ``BoltzDiffusionParams`` (boltz.main), overriding the checkpoint defaults."""

    return {
        "gamma_0": 0.605,
        "gamma_min": 1.107,
        "noise_scale": 0.901,
        "rho": 8,
        "step_scale": 1.638,
        "sigma_min": 0.0004,
        "sigma_max": 160.0,
        "sigma_data": 16.0,
        "alignment_reverse_diff": True,
        "coordinate_augmentation": True,
        "synchronize_sigmas": True,
    }


def diffusion_kwargs_for_reference(diff_args: dict[str, object], n_step: int) -> dict[str, object]:
    allowed = {
        "sigma_min",
        "sigma_max",
        "sigma_data",
        "rho",
        "gamma_0",
        "gamma_min",
        "noise_scale",
        "step_scale",
        "coordinate_augmentation",
        "alignment_reverse_diff",
        "synchronize_sigmas",
    }
    kwargs = {key: diff_args[key] for key in allowed if key in diff_args}
    kwargs["num_sampling_steps"] = n_step
    return kwargs


def diffusion_kwargs_for_ofoldx(diff_args: dict[str, object], n_step: int) -> dict[str, object]:
    return {
        "sigma_data": float(diff_args["sigma_data"]),
        "min_noise_level": float(diff_args["sigma_min"]),
        "max_noise_level": float(diff_args["sigma_max"]),
        "rho": float(diff_args["rho"]),
        "churn_rate": float(diff_args["gamma_0"]),
        "min_churn_noise_level": float(diff_args["gamma_min"]),
        "noise_scale": float(diff_args["noise_scale"]),
        "step_scale": float(diff_args["step_scale"]),
        "num_sampling_steps": n_step,
        "alignment_reverse_diff": bool(diff_args["alignment_reverse_diff"]),
    }


def make_sampler_draws(n_step: int, seed: int) -> dict[str, object]:
    r"""Shared fixed stochastic draws (init noise, per-step rigid augmentation, per-step churn eps)."""

    generator = torch.Generator().manual_seed(seed)

    def rotation() -> torch.Tensor:
        # A fixed proper rotation shared by both samplers (via QR with a positive-determinant fix).
        q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q.unsqueeze(0)  # (B=1, 3, 3)

    return {
        "init": torch.randn(B, M, 3, generator=generator),
        "rotation": [rotation() for _ in range(n_step)],
        "translation": [torch.randn(B, 1, 3, generator=generator) for _ in range(n_step)],
        "eps": [torch.randn(B, M, 3, generator=generator) for _ in range(n_step)],
    }


def prepare_models(args: argparse.Namespace):
    _ensure_fairscale_stub()
    from boltz.model.modules.diffusion import AtomDiffusion as RefAtomDiffusion

    from ofoldx.models.boltz1 import Boltz1Config
    from ofoldx.models.boltz1.convert import convert_boltz1_state_dict
    from ofoldx.models.boltz1.model import Boltz1Model

    state, hparams = load_checkpoint(args.checkpoint)
    diff_args = released_inference_diffusion_args()
    device = torch.device("cpu")

    # Reference AtomDiffusion (builds its own DiffusionModule internally from score_model_args).
    unused = {"activation_checkpointing", "offload_to_cpu"}
    score_model_args = {key: value for key, value in hparams["score_model_args"].items() if key not in unused}
    score_model_args = {
        **score_model_args,
        "token_s": int(hparams["token_s"]),
        "token_z": int(hparams["token_z"]),
        "atom_s": int(hparams["atom_s"]),
        "atom_z": int(hparams["atom_z"]),
        "atoms_per_window_queries": int(hparams["atoms_per_window_queries"]),
        "atoms_per_window_keys": int(hparams["atoms_per_window_keys"]),
        "atom_feature_dim": int(hparams["atom_feature_dim"]),
    }
    ref_sampler = RefAtomDiffusion(
        score_model_args=score_model_args,
        **diffusion_kwargs_for_reference(diff_args, args.n_step),
    )
    ref_sampler.load_state_dict(subset(state, "structure_module"), strict=False)
    ref_sampler.to(device).eval()

    # OFoldX model: reuse the released converter, then borrow its sampler + conditioning.
    with torch.device("meta"):
        model = Boltz1Model(Boltz1Config())
    model.load_state_dict(convert_boltz1_state_dict(state), strict=True, assign=True)
    model = model.to(device).eval().set_backend("torch")
    ofoldx_sampler = model.structure_module
    for key, value in diffusion_kwargs_for_ofoldx(diff_args, args.n_step).items():
        setattr(ofoldx_sampler, key, value)
    # The token-repr accumulator does not affect the sampled coordinates; the reference AtomDiffusion is built
    # with accumulate_token_repr=False, so disable it here for a coords-only signoff (return_token_repr=False).
    ofoldx_sampler.out_token_feat_update = None
    ofoldx_conditioning = model.diffusion_conditioning
    return ref_sampler, ofoldx_sampler, ofoldx_conditioning, diff_args


def prepare_conditioning(feats: dict[str, torch.Tensor], ofoldx_conditioning, seed: int):
    r"""Random trunk reps shared by both denoisers; OFoldX conditioning precomputed from them."""

    generator = torch.Generator().manual_seed(seed)
    single_trunk = torch.randn(B, N, TOKEN_S, generator=generator)
    single_inputs = torch.randn(B, N, feats["res_type"].shape[-1] * 2 + TOKEN_S + 1 + 4, generator=generator)
    pair_trunk = torch.randn(B, N, N, 128, generator=generator)
    relative_position = torch.randn(B, N, N, 128, generator=generator)
    batch = allo_input(feats)
    atom_to_token = feats["atom_to_token"].argmax(-1)

    with torch.no_grad():
        conditioning = ofoldx_conditioning(
            single_trunk,
            pair_trunk,
            relative_position,
            atom_features=batch.atom_features,
            reference_positions=feats["ref_pos"],
            atom_mask=feats["atom_pad_mask"],
            reference_space_id=feats["ref_space_uid"],
            atom_to_token=atom_to_token,
        )

    ref_net = {
        "s_inputs": single_inputs,
        "s_trunk": single_trunk,
        "z_trunk": pair_trunk,
        "relative_position_encoding": relative_position,
        "feats": feats,
    }
    ofoldx_net = {
        "single_inputs": single_inputs,
        "single_trunk": single_trunk,
        "conditioning": conditioning,
        "atom_to_token": atom_to_token,
        "token_mask": feats["token_pad_mask"],
    }
    return ref_net, ofoldx_net


def run_manual_sampler(ref_sampler, ofoldx_sampler, feats, draws, args):
    r"""Step the EDM loop by hand with shared draws + a deterministic (0.75x) denoiser.

    Both sides use ``.item()`` (float64) schedule scalars, so this isolates the sampler *algorithm* (schedule,
    churn, align, Euler) from the internal float precision each real sampler actually uses.
    """

    from boltz.model.loss.diffusion import weighted_rigid_align as ref_align

    from ofoldx.modules.geometry import weighted_rigid_align as ofoldx_align

    ref_sigmas = ref_sampler.sample_schedule(args.n_step)
    ofoldx_sigmas = ofoldx_sampler.sample_schedule(args.n_step, device=torch.device("cpu"))
    worst = report("sample_schedule", ofoldx_sigmas, ref_sigmas, atol=args.atol)
    ref_gammas = torch.where(ref_sigmas > ref_sampler.gamma_min, ref_sampler.gamma_0, 0.0)
    ofoldx_gammas = torch.where(ofoldx_sigmas > ofoldx_sampler.min_churn_noise_level, ofoldx_sampler.churn_rate, 0.0)
    worst = max(worst, report("gammas", ofoldx_gammas, ref_gammas, atol=args.atol))

    ref_coords = ref_sigmas[0] * draws["init"].clone()
    ofoldx_coords = ofoldx_sigmas[0] * draws["init"].clone()
    ref_denoised: torch.Tensor | None = None
    ofoldx_denoised: torch.Tensor | None = None
    atom_mask = feats["atom_pad_mask"]

    for step_idx, (ref_sigma_tm, ref_sigma_t, ref_gamma) in enumerate(
        zip(ref_sigmas[:-1], ref_sigmas[1:], ref_gammas[1:])
    ):
        ofoldx_sigma_tm = ofoldx_sigmas[step_idx]
        ofoldx_sigma_t = ofoldx_sigmas[step_idx + 1]
        ofoldx_gamma = ofoldx_gammas[step_idx + 1]
        rotation = draws["rotation"][step_idx]
        translation = draws["translation"][step_idx]
        eps = draws["eps"][step_idx]

        ref_coords = (
            torch.einsum("bmd,bds->bms", ref_coords - ref_coords.mean(dim=-2, keepdim=True), rotation) + translation
        )
        ofoldx_coords = (
            torch.einsum("bmd,bds->bms", ofoldx_coords - ofoldx_coords.mean(dim=-2, keepdim=True), rotation)
            + translation
        )
        if ref_denoised is not None and ofoldx_denoised is not None:
            ref_denoised = (
                torch.einsum("bmd,bds->bms", ref_denoised - ref_denoised.mean(dim=-2, keepdim=True), rotation)
                + translation
            )
            ofoldx_denoised = (
                torch.einsum("bmd,bds->bms", ofoldx_denoised - ofoldx_denoised.mean(dim=-2, keepdim=True), rotation)
                + translation
            )

        ref_t_hat = float(ref_sigma_tm.item() * (1 + ref_gamma.item()))
        ofoldx_t_hat = float(ofoldx_sigma_tm.item() * (1 + ofoldx_gamma.item()))
        ref_noise_var = ref_sampler.noise_scale**2 * (ref_t_hat**2 - float(ref_sigma_tm.item()) ** 2)
        ofoldx_noise_var = ofoldx_sampler.noise_scale**2 * (ofoldx_t_hat**2 - float(ofoldx_sigma_tm.item()) ** 2)
        ref_noisy = ref_coords + math.sqrt(max(ref_noise_var, 0.0)) * eps
        ofoldx_noisy = ofoldx_coords + math.sqrt(max(ofoldx_noise_var, 0.0)) * eps

        ref_denoised = 0.75 * ref_noisy
        ofoldx_denoised = 0.75 * ofoldx_noisy

        ref_aligned = ref_align(ref_noisy.float(), ref_denoised.float(), atom_mask.float(), atom_mask.float()).to(
            ref_denoised
        )
        ofoldx_aligned = ofoldx_align(
            source=ofoldx_noisy.float(), target=ofoldx_denoised.float(), weight=atom_mask.float()
        ).to(ofoldx_denoised)

        ref_next = ref_aligned + ref_sampler.step_scale * (float(ref_sigma_t.item()) - ref_t_hat) * (
            (ref_aligned - ref_denoised) / ref_t_hat
        )
        ofoldx_next = ofoldx_aligned + ofoldx_sampler.step_scale * (float(ofoldx_sigma_t.item()) - ofoldx_t_hat) * (
            (ofoldx_aligned - ofoldx_denoised) / ofoldx_t_hat
        )

        worst = max(worst, report(f"step{step_idx}:noisy", ofoldx_noisy, ref_noisy, atol=args.atol))
        worst = max(worst, report(f"step{step_idx}:aligned_noisy", ofoldx_aligned, ref_aligned, atol=args.atol))
        worst = max(worst, report(f"step{step_idx}:coords_next", ofoldx_next, ref_next, atol=args.atol))
        ref_coords = ref_next
        ofoldx_coords = ofoldx_next

    return ref_coords, ofoldx_coords, worst


def run_reference_public_sample(ref_sampler, feats, ref_net, draws, args) -> torch.Tensor:
    from boltz.model.modules import diffusion as ref_diffusion

    state = {"init": True, "aug": 0, "eps": 0}
    original_randn = ref_diffusion.torch.randn
    original_aug = ref_diffusion.compute_random_augmentation
    original_denoise = ref_sampler.preconditioned_network_forward
    token_repr_shape = (1, N, 2 * TOKEN_S)

    def compute_augmentation(multiplicity, s_trans=1.0, device=None, dtype=torch.float32):
        del multiplicity, s_trans, device
        idx = state["aug"]
        state["aug"] += 1
        return draws["rotation"][idx].to(dtype=dtype), draws["translation"][idx].to(dtype=dtype)

    def randn(*shape, **kwargs):
        del kwargs
        if state["init"]:
            state["init"] = False
            return draws["init"].clone()
        idx = state["eps"]
        state["eps"] += 1
        return draws["eps"][idx].clone()

    def denoise(noised, sigma, network_condition_kwargs, training=True):
        del sigma, network_condition_kwargs, training
        return 0.75 * noised, torch.zeros(token_repr_shape, dtype=noised.dtype, device=noised.device)

    ref_diffusion.compute_random_augmentation = compute_augmentation
    ref_diffusion.torch.randn = randn
    ref_sampler.preconditioned_network_forward = denoise
    try:
        with torch.no_grad():
            return ref_sampler.sample(
                feats["atom_pad_mask"],
                num_sampling_steps=args.n_step,
                multiplicity=1,
                max_parallel_samples=1,
                steering_args={"fk_steering": False, "physical_guidance_update": False},
                **ref_net,
            )["sample_atom_coords"]
    finally:
        ref_diffusion.torch.randn = original_randn
        ref_diffusion.compute_random_augmentation = original_aug
        ref_sampler.preconditioned_network_forward = original_denoise


def run_ofoldx_public_sample(ofoldx_sampler, feats, ofoldx_net, draws, args) -> torch.Tensor:
    import ofoldx.models.boltz1.model as ofoldx_model

    state = {"init": True, "aug": 0, "eps": 0, "translation": 0}
    original_randn = ofoldx_model.torch.randn
    original_rotation = ofoldx_model.uniform_random_rotation
    original_denoise = ofoldx_sampler.preconditioned_network_forward

    def rotation(num_samples=1, *, device=None, dtype=None, generator=None):
        del num_samples, generator
        idx = state["aug"]
        state["aug"] += 1
        return draws["rotation"][idx].to(device=device, dtype=dtype)

    def randn(*shape, **kwargs):
        del kwargs
        if len(shape) == 1 and isinstance(shape[0], (tuple, torch.Size)):
            requested = tuple(shape[0])
        else:
            requested = tuple(int(item) for item in shape)
        if state["init"]:
            state["init"] = False
            return draws["init"].clone()
        if requested == (B, 1, 3):
            idx = state["translation"]
            state["translation"] += 1
            return draws["translation"][idx].clone()
        idx = state["eps"]
        state["eps"] += 1
        return draws["eps"][idx].clone()

    def denoise(noised, sigma, **kwargs):
        del sigma, kwargs
        return 0.75 * noised

    ofoldx_model.uniform_random_rotation = rotation
    ofoldx_model.torch.randn = randn
    ofoldx_sampler.preconditioned_network_forward = denoise
    try:
        with torch.no_grad():
            return ofoldx_sampler._sample(
                feats["atom_pad_mask"],
                num_sampling_steps=args.n_step,
                multiplicity=1,
                steering_args={
                    "fk_steering": False,
                    "physical_guidance_update": False,
                    "num_particles": 1,
                    "fk_lambda": 4.0,
                    "fk_resampling_interval": 3,
                    "num_gd_steps": 16,
                },
                **ofoldx_net,
            )
    finally:
        ofoldx_model.uniform_random_rotation = original_rotation
        ofoldx_model.torch.randn = original_randn
        ofoldx_sampler.preconditioned_network_forward = original_denoise


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    feats = build_feats(seed=1, device=torch.device("cpu"))
    ref_sampler, ofoldx_sampler, ofoldx_conditioning, diff_args = prepare_models(args)
    ref_net, ofoldx_net = prepare_conditioning(feats, ofoldx_conditioning, args.seed)
    draws = make_sampler_draws(args.n_step, args.sampler_seed)

    print("[sampler args]", diffusion_kwargs_for_ofoldx(diff_args, args.n_step), flush=True)

    print("\n=== Step 1: REAL DENOISER (preconditioned_network_forward) ===", flush=True)
    worst = 0.0
    noisy = torch.randn(B, M, 3, generator=torch.Generator().manual_seed(args.seed + 1000))
    for sigma in (16.0, 5.0, 0.5):
        with torch.no_grad():
            ref_denoised, _ = ref_sampler.preconditioned_network_forward(
                noisy, sigma, network_condition_kwargs={**ref_net, "multiplicity": 1}, training=False
            )
            ofoldx_denoised = ofoldx_sampler.preconditioned_network_forward(
                noisy, sigma, multiplicity=1, atom_mask=feats["atom_pad_mask"], **ofoldx_net
            )
        worst = max(worst, report(f"denoised@sigma={sigma}", ofoldx_denoised, ref_denoised, atol=args.atol))

    print("\n=== Step 2: MANUAL SAMPLER LOOP (diagnostic localization) ===", flush=True)
    ref_manual, ofoldx_manual, manual_worst = run_manual_sampler(ref_sampler, ofoldx_sampler, feats, draws, args)
    worst = max(worst, manual_worst)

    print("\n=== Step 3: PUBLIC sample() OUTPUTS (signoff) ===", flush=True)
    ref_public = run_reference_public_sample(ref_sampler, feats, ref_net, draws, args)
    ofoldx_public = run_ofoldx_public_sample(ofoldx_sampler, feats, ofoldx_net, draws, args)
    worst = max(worst, report("reference sample() vs manual", ref_public, ref_manual, atol=args.atol))
    worst = max(worst, report("OFoldX sample() vs manual", ofoldx_public, ofoldx_manual, atol=args.atol))
    worst = max(worst, report("FINAL sampled coords", ofoldx_public, ref_public, atol=args.atol))
    print(f"\n[boltz1] worst max_abs = {worst:.3e}  tolerance = {args.atol:.1e}", flush=True)
    raise SystemExit(0 if worst < args.atol else 1)


if __name__ == "__main__":
    main()
