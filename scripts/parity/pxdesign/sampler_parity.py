r"""Compare PXDesign sampler mechanics with fixed shared stochastic draws.

The released implementation mixes PyTorch's global RNG with SciPy's global RNG,
whereas OFoldX accepts one explicit ``torch.Generator``. Exact seeded trajectories
therefore are not a meaningful cross-implementation contract. This diagnostic
injects the same initial noise, rigid transforms, and churn noise into both EDM loops
and compares every denoiser input and Euler update.

The real checkpoint denoiser is validated separately; this script isolates the
scientific sampling equations with a shared deterministic denoiser.

Example:

    python -m scripts.parity.pxdesign.sampler_parity \
      --protenix-root /path/to/protenix-0.6.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor, nn

from scripts.parity.paths import reference_root, repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=reference_root("PXDesign"))
    parser.add_argument("--protenix-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--atoms", type=int, default=9)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--atol", type=float, default=3e-4)
    return parser.parse_args()


def configure_imports(reference: Path, protenix: Path) -> None:
    os.environ.setdefault("LAYERNORM_TYPE", "normal")
    for path in (repo_root(), reference, protenix):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def make_context(num_atoms: int):
    from ofoldx.modules.diffusion import CoordinateDenoiseContext

    num_tokens = 2
    return CoordinateDenoiseContext(
        atom_features=torch.zeros(1, num_atoms, 1),
        atom_to_token=torch.zeros(1, num_atoms, dtype=torch.long),
        reference_positions=torch.zeros(1, num_atoms, 3),
        reference_charges=torch.zeros(1, num_atoms, 1),
        reference_mask=torch.ones(1, num_atoms, dtype=torch.bool),
        atom_mask=torch.ones(1, num_atoms, dtype=torch.bool),
        token_embedding=torch.zeros(1, num_tokens, 1),
        single=torch.zeros(1, num_tokens, 1),
        pair=torch.zeros(1, num_tokens, num_tokens, 1),
        relative_features=None,  # type: ignore[arg-type]
    )


def make_draws(*, steps: int, samples: int, atoms: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    shape = (1, samples, atoms, 3)
    initial = torch.randn(shape, generator=generator)
    translations: list[Tensor] = []
    churn: list[Tensor] = []
    for _ in range(steps):
        translations.append(torch.randn(1, samples, 3, generator=generator))
        churn.append(torch.randn(shape, generator=generator))

    rotation_generator = torch.Generator().manual_seed(seed + 1)
    rotations = [fixed_rotations(samples, generator=rotation_generator) for _ in range(steps)]
    return {
        "initial": initial,
        "translations": translations,
        "churn": churn,
        "rotations": rotations,
    }


def fixed_rotations(count: int, *, generator: torch.Generator) -> Tensor:
    from ofoldx.modules.geometry import uniform_random_rotation

    return uniform_random_rotation(
        count,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=generator,
    )


def deterministic_denoise(coordinates: Tensor, noise_level: Tensor) -> Tensor:
    scale = 0.25 + 0.05 / (1 + noise_level[..., None, None])
    return coordinates * scale


class NativeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor]] = []

    def forward(self, inputs) -> Tensor:
        self.calls.append((inputs.noisy_coords.detach().clone(), inputs.noise_level.detach().clone()))
        return deterministic_denoise(inputs.noisy_coords, inputs.noise_level)


class ReferenceDenoiser:
    def __init__(self) -> None:
        self.calls: list[tuple[Tensor, Tensor]] = []

    def __call__(self, *, x_noisy: Tensor, t_hat_noise_level: Tensor, **_kwargs) -> Tensor:
        self.calls.append((x_noisy.detach().clone(), t_hat_noise_level.detach().clone()))
        return deterministic_denoise(x_noisy, t_hat_noise_level)


def reference_randn(draws: dict[str, Any]):
    values = [draws["initial"]]
    for translation, churn in zip(draws["translations"], draws["churn"]):
        values.extend((translation.unsqueeze(-2), churn))
    queue = iter(values)

    def randn(*args, **kwargs) -> Tensor:
        value = next(queue)
        size = kwargs.get("size", args)
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = size[0]
        if tuple(size) != tuple(value.shape):
            raise AssertionError(f"reference requested {tuple(size)}, fixed draw is {tuple(value.shape)}")
        return value.to(device=kwargs.get("device"), dtype=kwargs.get("dtype"))

    return randn


def rotation_source(rotations: list[Tensor]):
    queue = iter(rotations)

    def draw(count: int = 1, *args, **kwargs) -> Tensor:
        count = kwargs.pop("N_sample", count)
        value = next(queue)
        if value.shape[0] != count:
            raise AssertionError(f"reference requested {count} rotations, fixed draw has {value.shape[0]}")
        return value.to(device=kwargs.get("device"), dtype=kwargs.get("dtype"))

    return draw


def compare(name: str, actual: Tensor, expected: Tensor, *, atol: float) -> dict[str, float | bool]:
    delta = (actual.float() - expected.float()).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    mean_abs = float(delta.mean()) if delta.numel() else 0.0
    return {"ok": max_abs <= atol, "max_abs": max_abs, "mean_abs": mean_abs}


def main() -> None:
    args = parse_args()
    configure_imports(args.reference_root, args.protenix_root)

    import protenix.model.utils as reference_utils
    import pxdesign.model.generator as reference_generator

    import ofoldx.models.pxdesign.model as native_module
    from ofoldx.models.pxdesign import PxDesignConfig

    config = PxDesignConfig()
    native_sampler = native_module.PxDesignSampler(config)
    reference_schedule = reference_generator.InferenceNoiseScheduler(
        s_max=config.sampler.max_noise_level,
        s_min=config.sampler.min_noise_level,
        rho=config.sampler.rho,
        sigma_data=config.denoiser.sigma_data,
    )(N_step=args.steps, device=torch.device("cpu"), dtype=torch.float32)
    native_schedule = native_sampler.scheduler(
        num_sampling_steps=args.steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    draws = make_draws(
        steps=args.steps,
        samples=args.samples,
        atoms=args.atoms,
        seed=args.seed,
    )

    reference_denoiser = ReferenceDenoiser()
    with (
        patch.object(reference_generator.torch, "randn", side_effect=reference_randn(draws)),
        patch.object(
            reference_utils,
            "uniform_random_rotation",
            side_effect=rotation_source(draws["rotations"]),
        ),
    ):
        reference_output = reference_generator.sample_diffusion(
            reference_denoiser,
            input_feature_dict={"atom_to_token_idx": torch.zeros(1, args.atoms, dtype=torch.long)},
            s_inputs=torch.zeros(1, 2, 1),
            s_trunk=torch.zeros(1, 2, 1),
            z_trunk=torch.zeros(1, 2, 2, 1),
            noise_schedule=reference_schedule,
            N_sample=args.samples,
            gamma0=config.sampler.gamma0,
            gamma_min=config.sampler.gamma_min,
            noise_scale_lambda=config.sampler.noise_scale,
            step_scale_eta={
                "type": config.sampler.step_scale_schedule,
                "min": config.sampler.step_scale_min,
                "max": config.sampler.step_scale_max,
            },
        )

    native_denoiser = NativeDenoiser()
    with patch.object(
        native_module,
        "uniform_random_rotation",
        side_effect=rotation_source(draws["rotations"]),
    ):
        native_output = native_sampler(
            native_denoiser,
            context=make_context(args.atoms),
            num_samples=args.samples,
            num_sampling_steps=args.steps,
            sample_chunk_size=None,
            generator=torch.Generator().manual_seed(args.seed),
        )

    results = {
        "schedule": compare("schedule", native_schedule, reference_schedule, atol=args.atol),
        "output": compare("output", native_output, reference_output, atol=args.atol),
    }
    initial = draws["initial"] * native_schedule[0]
    centered = initial - initial.mean(dim=-2, keepdim=True)
    rotation = draws["rotations"][0].reshape(1, args.samples, 3, 3)
    augmented = torch.einsum("...ij,...nj->...ni", rotation, centered)
    augmented = augmented + draws["translations"][0].unsqueeze(-2)
    sigma_hat = native_schedule[0] * (config.sampler.gamma0 + 1)
    expected_first_noisy = augmented + (
        config.sampler.noise_scale * torch.sqrt(sigma_hat.square() - native_schedule[0].square()) * draws["churn"][0]
    )
    results["step_0_native_against_fixed_draws"] = compare(
        "step_0_native_against_fixed_draws",
        native_denoiser.calls[0][0],
        expected_first_noisy,
        atol=args.atol,
    )
    results["step_0_reference_against_fixed_draws"] = compare(
        "step_0_reference_against_fixed_draws",
        reference_denoiser.calls[0][0],
        expected_first_noisy,
        atol=args.atol,
    )
    for index, ((native_x, native_sigma), (reference_x, reference_sigma)) in enumerate(
        zip(native_denoiser.calls, reference_denoiser.calls)
    ):
        results[f"step_{index}_noisy_coordinates"] = compare(
            f"step_{index}_noisy_coordinates",
            native_x,
            reference_x,
            atol=args.atol,
        )
        results[f"step_{index}_noise_level"] = compare(
            f"step_{index}_noise_level",
            native_sigma,
            reference_sigma,
            atol=args.atol,
        )

    payload = {
        "ok": len(native_denoiser.calls) == len(reference_denoiser.calls) == args.steps
        and all(result["ok"] for result in results.values()),
        "reference": str(args.reference_root),
        "protenix": str(args.protenix_root),
        "steps": args.steps,
        "samples": args.samples,
        "seed": args.seed,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
