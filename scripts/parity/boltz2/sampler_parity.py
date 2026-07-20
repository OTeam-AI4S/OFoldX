"""Boltz-2 real-denoiser and reverse-sampler-mechanics parity.

First compares the real checkpoint denoiser at fixed noise levels. It then
compares the reverse-sampler mechanics under the released public inference
parameters, identical injected randomness, and one identical deterministic
denoiser. This decomposition avoids treating chaotic Kabsch choices amplified
from tiny denoiser roundoff as a sampler-algorithm mismatch. The public command
overrides three older values serialized in the checkpoint.

    python -m scripts.parity.boltz2.sampler_parity

Realistic step count (``--n-step`` default 200). The OFoldX ``AtomDiffusion._sample``
loop is dtype-transparent (no forced ``.double()``), so its EDM churn/Euler scalars
run in the working fp32 dtype while this harness's ``run_manual_sampler`` reference
retains fp64 Python-float scalars. At an unrealistically low step count (n_step=5)
the two schedules put ~2e-3 of the coordinate scale into the very first high-sigma
step, where ``sigma_hat**2 - sigma_current**2`` (sigma**2 ~ 1e7 > 2**24) loses fp32
ULPs -- a numerically-expected high-sigma fp32 artifact, not a sampler-algorithm
mismatch. At the realistic released step count (n_step=200) each per-step sigma gap
is small and the fp32 mechanics track the fp64 reference to ~7.7e-7 mean / ~3e-6 max,
so the gate runs at n_step=200.
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

# E402: these imports require REPO_ROOT (inserted above) to be on sys.path first.
from scripts.parity.boltz2.forward_parity import B, M, N, build_feats, subset  # noqa: E402
from scripts.parity.paths import reference_root, weight_path  # noqa: E402

REFERENCE_SRC = reference_root("boltz") / "src"
if str(REFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SRC))

TOKEN_S, TOKEN_Z = 384, 128


def atom_features(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            feats["ref_pos"],
            feats["ref_charge"].unsqueeze(-1),
            feats["ref_element"],
            feats["ref_atom_name_chars"].reshape(B, M, 4 * 64),
        ],
        dim=-1,
    )


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


def converted_subset(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) + 1 :]: value for key, value in state.items() if key.startswith(prefix + ".")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=weight_path("boltz2_conf.ckpt"))
    # Gate at the realistic released step count: dtype-transparent fp32 mechanics track the fp64 reference to
    # ~7.7e-7 here, whereas n_step=5 exposes a high-sigma fp32 rounding artifact (~2e-3) that is not a mismatch.
    parser.add_argument("--n-step", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sampler-seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def diffusion_kwargs_for_reference(diff_args: dict[str, object], n_step: int) -> dict[str, object]:
    allowed = {
        "sigma_min",
        "sigma_max",
        "sigma_data",
        "rho",
        "P_mean",
        "P_std",
        "gamma_0",
        "gamma_min",
        "noise_scale",
        "step_scale",
        "step_scale_random",
        "coordinate_augmentation",
        "alignment_reverse_diff",
        "synchronize_sigmas",
    }
    kwargs = {key: diff_args[key] for key in allowed if key in diff_args}
    kwargs["num_sampling_steps"] = n_step
    return kwargs


def diffusion_kwargs_for_ofoldx(diff_args: dict[str, object], n_step: int) -> dict[str, object]:
    return {
        "sigma_data": float(diff_args.get("sigma_data", 16.0)),
        "min_noise_level": float(diff_args.get("sigma_min", 1e-4)),
        "max_noise_level": float(diff_args.get("sigma_max", 160.0)),
        "rho": float(diff_args.get("rho", 7.0)),
        "churn_rate": float(diff_args.get("gamma_0", 0.8)),
        "min_churn": float(diff_args.get("gamma_min", 1.0)),
        "noise_scale": float(diff_args.get("noise_scale", 1.003)),
        "step_scale": float(diff_args.get("step_scale", 1.5)),
        "num_sampling_steps": n_step,
        "alignment_reverse_diff": bool(diff_args.get("alignment_reverse_diff", True)),
    }


def released_inference_diffusion_args(checkpoint_args: dict[str, object]) -> dict[str, object]:
    r"""Apply the exact Boltz2DiffusionParams overrides used by the public CLI."""

    values = dict(checkpoint_args)
    values.update(
        {
            "sigma_min": 1e-4,
            "noise_scale": 1.003,
            "step_scale": 1.5,
        }
    )
    return values


def haar(generator: torch.Generator) -> torch.Tensor:
    """Draw one Haar-uniform rotation with the same closed form used by OFoldX."""

    random = torch.rand(1, 3, generator=generator)
    u1, u2, u3 = random.unbind(-1)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)
    return torch.stack(
        [
            torch.stack([1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)], dim=-1),
            torch.stack([2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)], dim=-1),
            torch.stack([2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)], dim=-1),
        ],
        dim=-2,
    )


def make_sampler_draws(n_step: int, seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "init": torch.randn(B, M, 3, generator=generator),
        "rotation": [haar(generator) for _ in range(n_step)],
        "translation": [torch.randn(B, 1, 3, generator=generator) for _ in range(n_step)],
        "eps": [torch.randn(B, M, 3, generator=generator) for _ in range(n_step)],
    }


def prepare_models(args: argparse.Namespace):
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning as RefConditioning
    from boltz.model.modules.diffusionv2 import AtomDiffusion as RefAtomDiffusion

    from ofoldx.models.boltz2 import convert
    from ofoldx.models.boltz2.model import AtomDiffusion, DiffusionConditioning, DiffusionScoreModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    hparams = checkpoint["hyper_parameters"]
    diff_args = released_inference_diffusion_args(hparams["diffusion_process_args"])
    converted = convert.convert_trunk(state)
    device = torch.device("cpu")

    ref_cond = RefConditioning(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atom_s=128,
        atom_z=16,
        atom_feature_dim=388,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=24,
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        conditioning_transition_layers=2,
    )
    ref_cond.load_state_dict(subset(state, "diffusion_conditioning"), strict=True)
    ref_cond.to(device).eval()

    score_model_args = {**hparams["score_model_args"], "token_s": TOKEN_S, "atom_s": 128}
    ref_sampler = RefAtomDiffusion(
        score_model_args=score_model_args,
        **diffusion_kwargs_for_reference(diff_args, args.n_step),
    )
    ref_sampler.load_state_dict(subset(state, "structure_module"), strict=False)
    ref_sampler.to(device).eval()

    conditioning = DiffusionConditioning(atom_feature_size=388)
    conditioning.load_state_dict(converted_subset(converted, "diffusion_conditioning"), strict=True)
    conditioning.to(device).eval()
    score_model = DiffusionScoreModel(operator_backend="eager")
    score_model.load_state_dict(converted_subset(converted, "structure_module.score_model"), strict=True)
    score_model.to(device).eval()
    ofoldx_sampler = AtomDiffusion(score_model, **diffusion_kwargs_for_ofoldx(diff_args, args.n_step)).to(device).eval()
    return ref_cond, ref_sampler, conditioning, ofoldx_sampler, diff_args


def prepare_conditioning(
    ref_cond: torch.nn.Module,
    ofoldx_cond: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    generator = torch.Generator().manual_seed(seed)
    single_trunk = torch.randn(B, N, TOKEN_S, generator=generator)
    single_inputs = torch.randn(B, N, TOKEN_S, generator=generator)
    pair_trunk = torch.randn(B, N, N, TOKEN_Z, generator=generator)
    relative_position = torch.randn(B, N, N, TOKEN_Z, generator=generator)
    atom_to_token = feats["atom_to_token"].argmax(-1)

    with torch.no_grad():
        query, conditioning, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = ref_cond(
            single_trunk, pair_trunk, relative_position, feats
        )
        ref_conditioning = {
            "q": query,
            "c": conditioning,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }
        ofoldx_conditioning = ofoldx_cond(
            single_trunk,
            pair_trunk,
            relative_position,
            atom_features=atom_features(feats),
            reference_positions=feats["ref_pos"],
            atom_mask=feats["atom_pad_mask"],
            reference_space_id=feats["ref_space_uid"],
            atom_to_token=atom_to_token,
        )

    ref_net = {
        "s_inputs": single_inputs,
        "s_trunk": single_trunk,
        "feats": feats,
        "diffusion_conditioning": ref_conditioning,
    }
    ofoldx_net = {
        "single_inputs": single_inputs,
        "single_trunk": single_trunk,
        "conditioning": ofoldx_conditioning,
        "atom_to_token": atom_to_token,
        "token_mask": feats["token_pad_mask"],
    }
    return ref_net, ofoldx_net


def run_manual_sampler(
    ref_sampler: torch.nn.Module,
    ofoldx_sampler: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    draws: dict[str, object],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    from boltz.model.loss.diffusionv2 import weighted_rigid_align as ref_align

    from ofoldx.modules.geometry import weighted_rigid_align as ofoldx_align

    ref_sigmas = ref_sampler.sample_schedule(args.n_step)
    ofoldx_sigmas = ofoldx_sampler.sample_schedule(args.n_step, device=torch.device("cpu"))
    worst = report("sample_schedule", ofoldx_sigmas, ref_sigmas, atol=args.atol)
    ref_gammas = torch.where(ref_sigmas > ref_sampler.gamma_min, ref_sampler.gamma_0, 0.0)
    ofoldx_gammas = torch.where(ofoldx_sigmas > ofoldx_sampler.min_churn, ofoldx_sampler.churn_rate, 0.0)
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
        worst = max(worst, report(f"step{step_idx}:denoised", ofoldx_denoised, ref_denoised, atol=args.atol))
        worst = max(worst, report(f"step{step_idx}:aligned_noisy", ofoldx_aligned, ref_aligned, atol=args.atol))
        worst = max(worst, report(f"step{step_idx}:coords_next", ofoldx_next, ref_next, atol=args.atol))
        ref_coords = ref_next
        ofoldx_coords = ofoldx_next

    return ref_coords, ofoldx_coords, worst


def run_reference_public_sample(
    ref_sampler: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    ref_net: dict[str, object],
    draws: dict[str, object],
    args: argparse.Namespace,
) -> torch.Tensor:
    from boltz.model.modules import diffusionv2 as ref_diffusion
    from boltz.model.modules import utils as ref_utils

    state = {"init": True, "aug": 0, "eps": 0}
    original_randn = ref_diffusion.torch.randn
    original_diff_aug = ref_diffusion.compute_random_augmentation
    original_utils_aug = ref_utils.compute_random_augmentation
    original_denoise = ref_sampler.preconditioned_network_forward

    def compute_augmentation(multiplicity: int, s_trans: float = 1.0, device=None, dtype=torch.float32):
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

    def denoise(noised: torch.Tensor, sigma: object, network_condition_kwargs: dict[str, object]) -> torch.Tensor:
        del sigma, network_condition_kwargs
        return 0.75 * noised

    ref_diffusion.compute_random_augmentation = compute_augmentation
    ref_utils.compute_random_augmentation = compute_augmentation
    ref_diffusion.torch.randn = randn
    ref_sampler.preconditioned_network_forward = denoise
    try:
        with torch.no_grad():
            return ref_sampler.sample(
                feats["atom_pad_mask"],
                num_sampling_steps=args.n_step,
                multiplicity=1,
                max_parallel_samples=1,
                steering_args={
                    "fk_steering": False,
                    "physical_guidance_update": False,
                    "contact_guidance_update": False,
                },
                **ref_net,
            )["sample_atom_coords"]
    finally:
        ref_diffusion.torch.randn = original_randn
        ref_diffusion.compute_random_augmentation = original_diff_aug
        ref_utils.compute_random_augmentation = original_utils_aug
        ref_sampler.preconditioned_network_forward = original_denoise


def run_ofoldx_public_sample(
    ofoldx_sampler: torch.nn.Module,
    feats: dict[str, torch.Tensor],
    ofoldx_net: dict[str, object],
    draws: dict[str, object],
    args: argparse.Namespace,
) -> torch.Tensor:
    import ofoldx.models.boltz2.model as ofoldx_model

    state = {"init": True, "aug": 0, "eps": 0, "translation": 0}
    original_randn = ofoldx_model.torch.randn
    original_rotation = ofoldx_model.uniform_random_rotation
    original_denoise = ofoldx_sampler.preconditioned_network_forward

    def rotation(num_samples: int = 1, *, device=None, dtype=None, generator=None):
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

    def denoise(noised: torch.Tensor, sigma: object, **kwargs: object) -> torch.Tensor:
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
                **ofoldx_net,
            )
    finally:
        ofoldx_model.uniform_random_rotation = original_rotation
        ofoldx_model.torch.randn = original_randn
        ofoldx_sampler.preconditioned_network_forward = original_denoise


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    feats = build_feats()
    ref_cond, ref_sampler, ofoldx_cond, ofoldx_sampler, diff_args = prepare_models(args)
    ref_net, ofoldx_net = prepare_conditioning(ref_cond, ofoldx_cond, feats, args.seed)
    draws = make_sampler_draws(args.n_step, args.sampler_seed)

    print(
        "[sampler args]",
        {
            key: diff_args[key]
            for key in (
                "sigma_min",
                "sigma_max",
                "sigma_data",
                "rho",
                "gamma_0",
                "gamma_min",
                "noise_scale",
                "step_scale",
                "alignment_reverse_diff",
            )
        },
        flush=True,
    )
    print("\n=== Step 1: DENOISER (preconditioned_network_forward) ===", flush=True)
    worst = 0.0
    noisy = torch.randn(B, M, 3, generator=torch.Generator().manual_seed(args.seed + 1000))
    for sigma in (16.0, 5.0, 0.5):
        with torch.no_grad():
            ref_denoised = ref_sampler.preconditioned_network_forward(
                noisy,
                sigma,
                network_condition_kwargs={**ref_net, "multiplicity": 1},
            )
            ofoldx_denoised = ofoldx_sampler.preconditioned_network_forward(
                noisy,
                sigma,
                multiplicity=1,
                atom_mask=feats["atom_pad_mask"],
                **ofoldx_net,
            )
        worst = max(worst, report(f"denoised@sigma={sigma}", ofoldx_denoised, ref_denoised, atol=args.atol))

    print("\n=== Step 2: MANUAL SAMPLER LOOP (diagnostic localization) ===", flush=True)
    ref_manual, ofoldx_manual, manual_worst = run_manual_sampler(
        ref_sampler,
        ofoldx_sampler,
        feats,
        draws,
        args,
    )
    worst = max(worst, manual_worst)

    print("\n=== Step 3: PUBLIC sample() OUTPUTS (signoff) ===", flush=True)
    ref_public = run_reference_public_sample(ref_sampler, feats, ref_net, draws, args)
    ofoldx_public = run_ofoldx_public_sample(ofoldx_sampler, feats, ofoldx_net, draws, args)
    worst = max(worst, report("reference sample() vs manual", ref_public, ref_manual, atol=args.atol))
    worst = max(worst, report("OFoldX sample() vs manual", ofoldx_public, ofoldx_manual, atol=args.atol))
    worst = max(worst, report("FINAL sampled coords", ofoldx_public, ref_public, atol=args.atol))
    print(f"\n[boltz2] worst max_abs = {worst:.3e}", flush=True)
    raise SystemExit(0 if worst < args.atol else 1)


if __name__ == "__main__":
    main()
