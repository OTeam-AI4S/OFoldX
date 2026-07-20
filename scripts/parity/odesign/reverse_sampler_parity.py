r"""Multi-step reverse-diffusion sampler parity for ODesign.

``forward_parity`` validates a single denoiser call plus one EDM combine. It does *not*
exercise the released multi-step reverse-diffusion loop: the per-step churn RNG, the
per-step SE(3) centre-random-augmentation (rotation + translation) and its reversal, the
predictor-corrector noise injection, and the accumulation of state across N sampling
steps are all unvalidated by that harness.

This script closes that gap. It drives the OFoldX
[`CoordinateEdmSampler`][ofoldx.modules.diffusion.sampling.CoordinateEdmSampler] loop and
the upstream ODesign ``sample_diffusion`` loop for the *same* number of steps on the same
trunk embeddings and compares the full coordinate trajectory (per-step, in the original
frame) plus the final coordinates.

RNG alignment (the crux). The two loops draw randomness from different sources and in
subtly different tensor shapes, so a fair trajectory comparison requires the random
streams to be made identical explicitly rather than merely co-seeded:

* Rotations. Upstream draws SE(3) rotations with ``scipy.spatial.transform.Rotation.random``
  (NumPy RNG, quaternion sampling); OFoldX draws them with a torch Shoemake construction
  (``torch.rand``). Different generators *and* different algorithms -- co-seeding cannot
  align them, and the torch path additionally consumes the torch RNG stream that the churn
  noise also draws from. Both ``uniform_random_rotation`` symbols are therefore patched to
  serve identical pre-generated matrices from a shared pool, consuming no torch RNG.
* Torch draws (initial noise, per-step translation, per-step churn noise). Upstream uses
  the global torch generator; OFoldX threads an explicit ``generator``. Passing
  ``generator=None`` routes OFoldX to the global generator too, and the two loops draw the
  same shapes in the same order (init -> [translation, churn] x N_step), so re-seeding the
  global generator identically before each run aligns them.

With the streams aligned and the sampler math being algebraically identical (verified by
inspection: same schedule, gamma0==churn_rate, gamma_min==min_churn_noise_level,
lambda==noise_scale, eta==step_scale, same predictor-corrector update), any surviving
divergence is a real bug in the OFoldX reverse-sampler port -- not an RNG/precision
artifact. The torch operator backend is forced on both sides so the denoiser numerics
match the reference (as in ``forward_parity``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Importing the forward-parity module runs its flash_attn / chanfig import shims and
# exposes the shared model/input builders, so the reverse harness reuses exactly the same
# weights, config, and synthetic system.
from scripts.parity.odesign import forward_parity as fp
from scripts.parity.paths import reference_root, weight_path

REFERENCE_ROOT = reference_root("ODesign")
CHECKPOINT = weight_path("odesign_base_prot_flex.pt")


class RotationPool:
    r"""Deterministic shared source of SE(3) rotation matrices for both samplers.

    Pre-generates ``capacity`` rotations with the upstream sampling algorithm
    (``scipy`` quaternion sampling) so the distribution is faithful, then serves them in
    order via :meth:`take`. :meth:`reset` rewinds the cursor before each sampler run so the
    reference loop and the OFoldX loop consume byte-identical matrices step-for-step.
    Consumes no torch RNG, keeping the torch stream (noise/translation) aligned.
    """

    def __init__(self, capacity: int, *, seed: int) -> None:
        from scipy.spatial.transform import Rotation

        matrices = Rotation.random(num=capacity, random_state=seed).as_matrix()
        self.matrices = torch.from_numpy(matrices).float()  # [capacity, 3, 3], CPU
        self.cursor = 0

    def reset(self) -> None:
        self.cursor = 0

    def take(self, count: int) -> torch.Tensor:
        end = self.cursor + count
        if end > self.matrices.shape[0]:
            raise RuntimeError(f"rotation pool exhausted: need {end}, have {self.matrices.shape[0]}")
        chunk = self.matrices[self.cursor : end]
        self.cursor = end
        return chunk


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()
    fp.configure_environment(args.reference_root)
    if not args.allow_tf32:
        # A100 matmuls default to TF32 (~10-bit mantissa). Both implementations would use it,
        # but with different matmul shapes/orderings the ~1e-3 per-op TF32 error diverges and
        # is amplified by network input magnitude -- a precision confound, not a port bug.
        # forward_parity never caught it because it feeds ~0.06-scale inputs; the sampler
        # drives near-unit-scale inputs at high sigma. Force true fp32 to isolate real logic.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)

    fp.log_stage(started_at, "build_config")
    reference_config = fp.build_reference_config(args.reference_root, n_cycle=args.n_cycle)
    fp.log_stage(started_at, "load_checkpoint")
    reference_state = fp.load_reference_state(args.checkpoint)
    fp.log_stage(started_at, "build_reference_model")
    reference_model = fp.build_reference_model(reference_config, reference_state, device=device)
    fp.log_stage(started_at, "build_ofoldx_model")
    ofoldx_model = fp.build_ofoldx_model(reference_config, reference_state, device=device)
    fp.log_stage(started_at, "build_inputs")
    reference_input, reference_diffusion_input, ofoldx_input = fp.build_inputs(
        n_token=args.n_token,
        n_msa=args.n_msa,
        seed=args.seed,
        device=device,
    )

    if args.probe_denoiser:
        fp.log_stage(started_at, "probe_denoiser")
        probe = run_denoiser_sigma_probe(
            reference_model=reference_model,
            ofoldx_model=ofoldx_model,
            reference_input=reference_input,
            reference_diffusion_input=reference_diffusion_input,
            ofoldx_input=ofoldx_input,
            num_sampling_steps=args.num_sampling_steps,
            n_cycle=args.n_cycle,
            seed=args.seed,
            device=device,
        )
        print(json.dumps({"denoiser_sigma_probe": probe}, indent=2, sort_keys=True))
        return

    # One rotation per augmentation call: chunk_size 1 => 1 augmentation per step, and one
    # chunk per sample => num_samples * num_sampling_steps rotations for a full run.
    pool = RotationPool(
        capacity=args.num_samples * args.num_sampling_steps,
        seed=args.seed + 101,
    )
    reference_reverse_trajectory: list[torch.Tensor] = []
    ofoldx_reverse_trajectory: list[torch.Tensor] = []

    fp.log_stage(started_at, "sample_reference")
    reference_final = run_reference_sampler(
        reference_model=reference_model,
        reference_input=reference_input,
        reference_diffusion_input=reference_diffusion_input,
        pool=pool,
        trajectory=reference_reverse_trajectory,
        num_samples=args.num_samples,
        num_sampling_steps=args.num_sampling_steps,
        n_cycle=args.n_cycle,
        seed=args.seed,
        device=device,
        surrogate=args.surrogate,
    )

    fp.log_stage(started_at, "sample_ofoldx")
    ofoldx_final = run_ofoldx_sampler(
        ofoldx_model=ofoldx_model,
        ofoldx_input=ofoldx_input,
        pool=pool,
        trajectory=ofoldx_reverse_trajectory,
        num_samples=args.num_samples,
        num_sampling_steps=args.num_sampling_steps,
        n_cycle=args.n_cycle,
        seed=args.seed,
        surrogate=args.surrogate,
    )

    fp.log_stage(started_at, "compare")
    payload = compare_trajectories(
        reference_trajectory=reference_reverse_trajectory,
        ofoldx_trajectory=ofoldx_reverse_trajectory,
        reference_final=reference_final,
        ofoldx_final=ofoldx_final,
        rmsd_tol=args.rmsd_tol,
        rel_tol=args.rel_tol,
    )
    payload.update(
        {
            "checkpoint": str(args.checkpoint),
            "reference_root": str(args.reference_root),
            "n_cycle": args.n_cycle,
            "n_token": args.n_token,
            "n_msa": args.n_msa,
            "num_samples": args.num_samples,
            "num_sampling_steps": args.num_sampling_steps,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["ok"]:
        raise SystemExit(1)


def run_reference_sampler(
    *,
    reference_model: Any,
    reference_input: Any,
    reference_diffusion_input: Any,
    pool: RotationPool,
    trajectory: list[torch.Tensor],
    num_samples: int,
    num_sampling_steps: int,
    n_cycle: int,
    seed: int,
    device: torch.device,
    surrogate: bool = False,
) -> torch.Tensor:
    r"""Run the upstream ``sample_diffusion`` loop with the shared rotation pool.

    The trunk is run once (outside the RNG-sensitive region), then the global torch
    generator is re-seeded immediately before the sampler loop. ``uniform_random_rotation``
    is patched to serve pool matrices and ``reverse_centre_random_augmentation`` is wrapped
    to record the per-step coordinates in the original frame.
    """

    from src.api.model_interface import GroundTruth
    from src.model.modules import generator as reference_generator
    from src.utils.model import misc as reference_misc

    n_atom = reference_diffusion_input.atom_to_token_idx.size(-1)

    with torch.no_grad():
        # The reference trunk consumes global RNG (rowwise dropout / MSA subsampling), so it
        # must be seeded to a fixed draw before the call -- otherwise its pair output varies
        # ~2.8% run-to-run and diverges from the deterministic OFoldX trunk, corrupting the
        # denoiser conditioning and the whole trajectory. (The sampler loop below is re-seeded
        # separately so its churn/augmentation RNG stays aligned with OFoldX.)
        torch.manual_seed(seed)
        representations = reference_model.get_pairformer_output(
            reference_input,
            N_cycle=n_cycle,
            inplace_safe=False,
            chunk_size=None,
        )

    ground_truth = GroundTruth(
        coordinate=torch.zeros(n_atom, 3, device=device),
        coordinate_mask=torch.ones(n_atom, dtype=torch.bool, device=device),
    )

    # Zero-update surrogate: replaces the network with an analytic denoiser (update == 0,
    # so denoised == c_skip * x_noisy). Both loops then run identical closed-form math, so
    # any surviving trajectory divergence is a genuine sampler-loop bug rather than network
    # numerics amplifying through the chaotic ODE.
    denoise_net: Any = reference_model.diffusion_module
    if surrogate:

        class _ZeroUpdate(torch.nn.Module):
            def forward(self, *, x_noisy, t_hat_noise_level, input_data, input_embedding, chunk_size, inplace_safe):
                return torch.zeros_like(x_noisy)

        denoise_net = _ZeroUpdate().to(device).eval()

    original_uniform = reference_misc.uniform_random_rotation
    original_reverse = reference_generator.reverse_centre_random_augmentation

    def patched_uniform(N_sample: int = 1) -> torch.Tensor:
        return pool.take(N_sample).clone()

    def patched_reverse(x_augment_coords, trans, rot, x_center):
        result = original_reverse(x_augment_coords, trans, rot, x_center)
        trajectory.append(result.detach().float().cpu().clone())
        return result

    reference_misc.uniform_random_rotation = patched_uniform
    reference_generator.reverse_centre_random_augmentation = patched_reverse
    try:
        pool.reset()
        torch.manual_seed(seed)
        with torch.no_grad():
            output = reference_generator.sample_diffusion(
                denoise_net=denoise_net,
                input_data=reference_diffusion_input,
                input_embedding=representations,
                ground_truth=ground_truth,
                noise_schedulers=reference_model.inference_noise_schedulers,
                N_sample=num_samples,
                N_step=num_sampling_steps,
                diffusion_chunk_size=1,
                inplace_safe=False,
                attn_chunk_size=None,
            )
    finally:
        reference_misc.uniform_random_rotation = original_uniform
        reference_generator.reverse_centre_random_augmentation = original_reverse

    return output.x_denoised.detach().float().cpu()  # [N_sample, N_atom, 3]


def run_ofoldx_sampler(
    *,
    ofoldx_model: Any,
    ofoldx_input: Any,
    pool: RotationPool,
    trajectory: list[torch.Tensor],
    num_samples: int,
    num_sampling_steps: int,
    n_cycle: int,
    seed: int,
    surrogate: bool = False,
) -> torch.Tensor:
    r"""Run the OFoldX ``CoordinateEdmSampler`` loop with the shared rotation pool.

    Mirrors :func:`run_reference_sampler`: trunk once, re-seed the global torch generator,
    pass ``generator=None`` so the sampler draws from that same global stream, patch the
    sampler's ``uniform_random_rotation`` to the pool, and wrap the instance's
    ``_reverse_augmentation`` to record per-step coordinates in the original frame.
    """

    from ofoldx.modules.diffusion import sampling as ofoldx_sampling

    sampler = ofoldx_model.coordinate_sampler

    with torch.no_grad():
        token_embedding, single, pair, relative_features = ofoldx_model._run_trunk(
            ofoldx_input,
            recycling_steps=n_cycle,
        )
        context = ofoldx_model._denoise_context(
            ofoldx_input,
            token_embedding,
            single,
            pair,
            relative_features,
        )

    preconditioner = ofoldx_model.coordinate_denoiser.preconditioner

    def surrogate_denoise(inputs: Any) -> torch.Tensor:
        return preconditioner.combine_update(
            inputs.noisy_coords,
            inputs.noise_level,
            torch.zeros_like(inputs.noisy_coords),
        )

    denoise_fn: Any = surrogate_denoise if surrogate else ofoldx_model.coordinate_denoiser

    original_uniform = ofoldx_sampling.uniform_random_rotation
    original_reverse = type(sampler)._reverse_augmentation

    def patched_uniform(num_samples, *, device=None, dtype=None, generator=None):
        return pool.take(num_samples).to(device=device, dtype=dtype)

    def patched_reverse(coords, transform):
        result = original_reverse(coords, transform)
        trajectory.append(result.detach().float().cpu().squeeze(0).clone())
        return result

    ofoldx_sampling.uniform_random_rotation = patched_uniform
    sampler._reverse_augmentation = patched_reverse  # shadow the staticmethod on the instance
    try:
        pool.reset()
        torch.manual_seed(seed)
        with torch.no_grad():
            coordinates = sampler(
                denoise_fn,
                context=context,
                num_samples=num_samples,
                num_sampling_steps=num_sampling_steps,
                sample_chunk_size=1,
                initial_coords=None,
                condition_coords=ofoldx_input.observed_positions,
                condition_mask=ofoldx_input.fixed_coordinate_mask,
                max_noise_level=None,
                generator=None,
            )
    finally:
        ofoldx_sampling.uniform_random_rotation = original_uniform
        try:
            del sampler._reverse_augmentation
        except AttributeError:
            pass

    # OFoldX carries a leading batch axis: [1, N_sample, N_atom, 3] -> [N_sample, N_atom, 3].
    return coordinates.detach().float().cpu().squeeze(0)


def compare_trajectories(
    *,
    reference_trajectory: list[torch.Tensor],
    ofoldx_trajectory: list[torch.Tensor],
    reference_final: torch.Tensor,
    ofoldx_final: torch.Tensor,
    rmsd_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    r"""Per-step and final trajectory comparison: direct RMSD, Kabsch RMSD, relative error."""

    if len(reference_trajectory) != len(ofoldx_trajectory):
        raise RuntimeError(
            "trajectory length mismatch: " f"reference={len(reference_trajectory)} ofoldx={len(ofoldx_trajectory)}"
        )

    steps = []
    worst_rmsd = 0.0
    worst_rel = 0.0
    for index, (reference, ofoldx) in enumerate(zip(reference_trajectory, ofoldx_trajectory)):
        metrics = pairwise_metrics(ofoldx, reference)
        steps.append({"step": index, **metrics})
        worst_rmsd = max(worst_rmsd, metrics["rmsd"])
        worst_rel = max(worst_rel, metrics["rel"])

    final_metrics = pairwise_metrics(ofoldx_final, reference_final)
    ok = final_metrics["rmsd"] <= rmsd_tol and final_metrics["rel"] <= rel_tol

    return {
        "ok": bool(ok),
        "num_trajectory_steps": len(reference_trajectory),
        "final": final_metrics,
        "worst_step_rmsd": worst_rmsd,
        "worst_step_rel": worst_rel,
        "rmsd_tol": rmsd_tol,
        "rel_tol": rel_tol,
        "per_step": steps,
    }


def pairwise_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    r"""Direct RMSD, Kabsch-aligned RMSD, relative Frobenius error, and max abs diff.

    Both tensors are `[N_sample, N_atom, 3]`; metrics are averaged over samples. The direct
    RMSD is the meaningful number here (the frames are already aligned because the per-step
    augmentation is reversed with the shared rotations); the Kabsch RMSD additionally
    factors out any residual global rigid drift to isolate genuine shape divergence.
    """

    actual = actual.detach().float()
    expected = expected.detach().float()
    diff = actual - expected
    rmsd = float(torch.sqrt((diff**2).sum(dim=-1).mean()))
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    denom = float(torch.sqrt((expected**2).sum())) + 1e-9
    rel = float(torch.sqrt((diff**2).sum())) / denom

    kabsch_values = [kabsch_rmsd(actual[sample], expected[sample]) for sample in range(actual.shape[0])]
    kabsch = float(sum(kabsch_values) / len(kabsch_values)) if kabsch_values else 0.0

    return {"rmsd": rmsd, "kabsch_rmsd": kabsch, "rel": rel, "max_abs": max_abs}


def kabsch_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    r"""RMSD after optimal rigid superposition of ``actual`` onto ``expected`` ([N_atom, 3])."""

    actual_centered = actual - actual.mean(dim=0, keepdim=True)
    expected_centered = expected - expected.mean(dim=0, keepdim=True)
    covariance = actual_centered.transpose(-1, -2) @ expected_centered
    u, _, vh = torch.linalg.svd(covariance.double())
    determinant = torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    correction = torch.eye(3, dtype=torch.double)
    correction[2, 2] = torch.sign(determinant)
    rotation = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    aligned = (rotation @ actual_centered.double().transpose(-1, -2)).transpose(-1, -2)
    diff = aligned - expected_centered.double()
    return float(torch.sqrt((diff**2).sum(dim=-1).mean()))


def run_denoiser_sigma_probe(
    *,
    reference_model: Any,
    ofoldx_model: Any,
    reference_input: Any,
    reference_diffusion_input: Any,
    ofoldx_input: Any,
    num_sampling_steps: int,
    n_cycle: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    r"""Compare the reference vs OFoldX denoised clean coords across the schedule's sigmas.

    forward_parity validates the denoiser only at a single moderate noise level. The reverse
    sampler however evaluates it across the full EDM schedule, including the extreme first
    ``t_hat`` (~4600). This probe feeds *identical* noisy coordinates to both denoisers at
    several schedule sigmas (first / quartiles / last predictor-corrector step) and reports
    the denoised-clean-coord agreement, ruling out a hidden extreme-sigma denoiser bug.
    """

    scheduler = reference_model.inference_noise_schedulers["coordinate"]
    scheduler.set_noise_schedule(N_step=num_sampling_steps, device=device, dtype=torch.float32)
    schedule = scheduler.noise_schedule  # [N_step + 1]
    sigma_data = float(scheduler.sigma_data)
    gamma0 = float(scheduler.gamma0)
    gamma_min = float(scheduler.gamma_min)

    with torch.no_grad():
        # The reference trunk consumes global RNG (rowwise dropout / MSA subsampling in the
        # msa+pairformer stack), so it must be seeded to a fixed draw before every call --
        # otherwise its pair output varies ~2.8% run-to-run and spuriously diverges from the
        # deterministic OFoldX trunk. Seed matches forward_parity's ``torch.manual_seed(seed)``.
        torch.manual_seed(seed)
        representations = reference_model.get_pairformer_output(
            reference_input, N_cycle=n_cycle, inplace_safe=False, chunk_size=None
        )
        token_embedding, single, pair, relative_features = ofoldx_model._run_trunk(
            ofoldx_input, recycling_steps=n_cycle
        )
        context = ofoldx_model._denoise_context(ofoldx_input, token_embedding, single, pair, relative_features)

    n_atom = reference_diffusion_input.atom_to_token_idx.size(-1)

    # Control: replicate forward_parity's exact denoiser inputs (unit noisy coords, the
    # training-style exp noise level) using the SAME representations/context objects. If this
    # reproduces ~1e-6, the probe machinery is sound and any schedule-sigma divergence is real.
    control_generator = torch.Generator(device=device).manual_seed(seed + 1)
    control_noisy_unit = torch.randn(1, 1, n_atom, 3, device=device, generator=control_generator)
    control_noise_level = 16 * torch.exp(-1.2 + 1.5 * torch.randn(1, 1, device=device, generator=control_generator))
    control_scaled = control_noisy_unit / torch.sqrt(16**2 + control_noise_level.square().unsqueeze(-1).unsqueeze(-1))
    with torch.no_grad():
        control_reference = reference_model.diffusion_module(
            x_noisy=control_scaled.squeeze(0),
            t_hat_noise_level=control_noise_level.squeeze(0),
            input_data=reference_diffusion_input,
            input_embedding=representations,
            chunk_size=None,
            inplace_safe=False,
        )
        control_ofoldx = ofoldx_model.coordinate_denoiser.predict_coordinate_update(
            context.to_denoising_input(control_scaled, control_noise_level)
        ).squeeze(0)
    control_metrics = pairwise_metrics(control_ofoldx.cpu(), control_reference.cpu())

    # Localize: do the trunk embeddings feeding the two denoisers actually match here?
    def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
        actual = actual.detach().float().cpu()
        expected = expected.detach().float().cpu()
        return float(torch.sqrt(((actual - expected) ** 2).sum())) / (float(torch.sqrt((expected**2).sum())) + 1e-9)

    with torch.no_grad():
        torch.manual_seed(seed)
        representations_again = reference_model.get_pairformer_output(
            reference_input, N_cycle=n_cycle, inplace_safe=False, chunk_size=None
        )
        _, _, pair_again, _ = ofoldx_model._run_trunk(ofoldx_input, recycling_steps=n_cycle)
    embedding_rel = {
        "s_inputs": _rel(token_embedding.squeeze(0), representations.s_inputs),
        "single": _rel(single.squeeze(0), representations.s),
        "pair": _rel(pair.squeeze(0), representations.z),
        "reference_pair_nondeterminism": _rel(representations_again.z, representations.z),
        "ofoldx_pair_nondeterminism": _rel(pair_again.squeeze(0), pair.squeeze(0)),
        "pair_ref_maxabs": float(representations.z.abs().max()),
        "pair_ofx_maxabs": float(pair.abs().max()),
        "pair_diff_maxabs": float((pair.squeeze(0) - representations.z).abs().max()),
    }
    print(json.dumps({"embedding_rel": embedding_rel}, indent=2, sort_keys=True), flush=True)

    probe_steps = sorted(
        {0, num_sampling_steps // 4, num_sampling_steps // 2, 3 * num_sampling_steps // 4, num_sampling_steps - 1}
    )
    results = []
    for step_index in probe_steps:
        c_tau_last = float(schedule[step_index])
        c_tau = float(schedule[step_index + 1])
        gamma = gamma0 if c_tau > gamma_min else 0.0
        t_hat = c_tau_last * (gamma + 1)  # the loop's actual denoiser noise level

        generator = torch.Generator(device=device).manual_seed(seed + step_index)
        noisy = t_hat * torch.randn(1, n_atom, 3, device=device, generator=generator)
        noise_level = torch.tensor([t_hat], device=device)
        scale = (sigma_data**2 + t_hat**2) ** 0.5
        # Pre-scale ONCE so both denoisers receive bit-identical network input (r_l).
        scaled = noisy / scale

        with torch.no_grad():
            reference_update = reference_model.diffusion_module(
                x_noisy=scaled,
                t_hat_noise_level=noise_level,
                input_data=reference_diffusion_input,
                input_embedding=representations,
                chunk_size=None,
                inplace_safe=False,
            )
            reference_update_again = reference_model.diffusion_module(
                x_noisy=scaled,
                t_hat_noise_level=noise_level,
                input_data=reference_diffusion_input,
                input_embedding=representations,
                chunk_size=None,
                inplace_safe=False,
            )
            # OFoldX raw update on the SAME pre-scaled coords (predict_coordinate_update takes
            # already-scaled coords, exactly as forward_parity feeds it).
            ofoldx_update = ofoldx_model.coordinate_denoiser.predict_coordinate_update(
                context.to_denoising_input(
                    noisy_coords=scaled.unsqueeze(0),
                    noise_level=noise_level.reshape(1, 1),
                )
            ).squeeze(0)

        update_metrics = pairwise_metrics(ofoldx_update.cpu(), reference_update.cpu())
        reference_nondeterminism = float((reference_update - reference_update_again).abs().max())
        results.append(
            {
                "step": step_index,
                "sigma": c_tau_last,
                "t_hat": t_hat,
                "control_forward_parity_update_rel": control_metrics["rel"],
                "control_forward_parity_update_max_abs": control_metrics["max_abs"],
                "scaled_input_std": float(scaled.std()),
                "reference_update_std": float(reference_update.std()),
                "reference_nondeterminism_max_abs": reference_nondeterminism,
                "update_rel": update_metrics["rel"],
                "update_max_abs": update_metrics["max_abs"],
                "update_rmsd": update_metrics["rmsd"],
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-cycle", type=int, default=1)
    parser.add_argument("--n-token", type=int, default=5)
    parser.add_argument("--n-msa", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    # A faithful 40-step port lands at ~1e-3 A direct RMSD / ~1e-6 relative (pure fp32
    # accumulation over the chaotic sampler); the RNG/embedding-misaligned bug this harness
    # caught sat at ~7.5 A / ~1e-2, so these bounds cleanly separate parity from regression.
    parser.add_argument("--rmsd-tol", type=float, default=5e-3)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="Replace both denoisers with a zero-update analytic surrogate to isolate the sampler loop.",
    )
    parser.add_argument(
        "--probe-denoiser",
        action="store_true",
        help="Compare denoised clean coords at the schedule's actual sigmas (incl. extreme t_hat) and exit.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Leave TF32 matmuls enabled (default: force true fp32 to avoid a precision confound).",
    )
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
