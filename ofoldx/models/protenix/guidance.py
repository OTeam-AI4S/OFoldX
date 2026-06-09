r"""
Native Protenix training-free guidance utilities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ...modules.diffusion import CoordinateDenoiseContext

if TYPE_CHECKING:
    from .model import ProtenixInput

_VDW_RADII = (
    1.2,
    1.4,
    2.2,
    1.9,
    1.8,
    1.7,
    1.6,
    1.55,
    1.5,
    1.54,
    2.4,
    2.2,
    2.1,
    2.1,
    1.95,
    1.8,
    1.8,
    1.88,
    2.8,
    2.4,
    2.3,
    2.15,
    2.05,
    2.05,
    2.05,
    2.05,
    2.0,
    2.0,
    2.0,
    2.1,
    2.1,
    2.1,
    2.05,
    1.9,
    1.9,
    2.02,
    2.9,
    2.55,
    2.4,
    2.3,
    2.15,
    2.1,
    2.05,
    2.05,
    2.0,
    2.05,
    2.1,
    2.2,
    2.2,
    2.25,
    2.2,
    2.1,
    2.1,
    2.16,
    3.0,
    2.7,
    2.7,
    2.5,
    2.48,
    2.47,
    2.45,
    2.43,
    2.42,
    2.4,
    2.38,
    2.37,
    2.35,
    2.33,
    2.32,
    2.3,
    2.28,
    2.27,
    2.25,
    2.2,
    2.1,
    2.05,
    2.0,
    2.0,
    2.05,
    2.1,
    2.05,
    2.2,
    2.3,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.4,
    2.0,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
)


_REQUIRED_FEATURES: dict[str, set[str]] = {
    "PairwiseDistancePotential": {
        "pairwise_distance_index",
        "pairwise_distance_is_bond",
        "pairwise_distance_is_angle",
        "pairwise_distance_upper_bound",
        "pairwise_distance_lower_bound",
        "ref_element",
    },
    "InterchainBondPotential": {"interchain_bond_index"},
    "VinaStericPotential": {
        "asym_id",
        "atom_to_token_idx",
        "ref_element",
        "interchain_bond_index",
    },
    "StereoBondPotential": {"stereo_bond_index", "stereo_bond_orientation"},
    "ChiralAtomPotential": {
        "chiral_index",
        "chiral_orientation",
        "asym_id",
        "atom_to_token_idx",
    },
    "PlanarImproperPotential": {
        "planar_improper_index",
        "planar_improper_is_carbonyl",
    },
    "LinearBondPotential": {"linear_triple_bond_index"},
    "ExperimentalTorsionPotential": {
        "experimental_torsion_index",
        "experimental_torsion_force_constant",
        "experimental_torsion_sign",
    },
}


@dataclass(slots=True)
class ProtenixGuidanceFeatures:
    r"""
    Bundle of training-free-guidance (TFG) geometry feature tensors keyed by Protenix names.

    Carries the per-atom/per-bond geometry tensors (element identities, bond/torsion/chiral indices,
    distance bounds, ...) the TFG potentials consume during guided sampling. The keys are the official
    Protenix feature names; consumers look tensors up by name.

    Attributes:
        tensors: Mapping from official Protenix feature name to its tensor.
    """

    tensors: dict[str, Tensor]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Tensor]) -> ProtenixGuidanceFeatures:
        r"""
        Build a feature bundle from a mapping of Protenix feature names to tensors.

        Args:
            values: Mapping from official Protenix feature name to tensor; copied into the bundle.

        Returns:
            A new bundle wrapping a shallow copy of ``values``.
        """

        return cls(tensors=dict(values))

    def as_dict(self) -> dict[str, Tensor]:
        r"""
        Return a shallow copy of the underlying feature dictionary.

        Returns:
            Mapping from official Protenix feature name to tensor.
        """

        return dict(self.tensors)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> ProtenixGuidanceFeatures:
        r"""
        Return a copy with every feature tensor moved to the given device/dtype.

        Only floating-point tensors are cast to ``dtype``; integer index tensors keep their dtype.

        Args:
            device: Target device, or ``None`` to leave devices unchanged.
            dtype: Target floating dtype, or ``None`` to leave dtypes unchanged.
            non_blocking: Whether to request an asynchronous host-to-device copy.
            copy: Whether to force a copy even when no move is required.

        Returns:
            A new bundle with the moved tensors.
        """

        moved = {key: _move_tensor(value, device, dtype, non_blocking, copy) for key, value in self.tensors.items()}
        return type(self)(moved)


class Schedule:
    r"""
    Abstract base for a scalar TFG schedule evaluated at a normalized sampling time.

    A schedule maps a normalized time ``t`` in ``[0, 1]`` (0 at the last diffusion step, 1 at the
    first) to a scalar weight/parameter value. Subclasses implement ``__call__``.
    """

    def __call__(self, t: float) -> float:
        r"""
        Evaluate the schedule at normalized time ``t``.

        Args:
            t: Normalized sampling time in ``[0, 1]``.

        Returns:
            The scalar schedule value.

        Raises:
            TypeError: Always, since the base schedule is abstract.
        """

        raise TypeError("Schedule is an abstract TFG schedule base")


@dataclass(frozen=True)
class Constant(Schedule):
    r"""
    Schedule that returns a fixed value regardless of sampling time.

    Attributes:
        value: The constant value returned for every ``t``.

    Examples:
        >>> schedule = Constant(0.5)
        >>> schedule(0.0)
        0.5
        >>> schedule(1.0)
        0.5
    """

    value: float

    def __call__(self, t: float) -> float:
        r"""
        Return the constant value, ignoring ``t``.

        Args:
            t: Normalized sampling time (unused).

        Returns:
            The constant `value`.
        """

        return float(self.value)


@dataclass(frozen=True)
class ExponentialInterpolation(Schedule):
    r"""
    Schedule interpolating from ``start`` to ``end`` with optional exponential warping.

    With ``alpha == 0`` the interpolation is linear in ``t``; otherwise it follows the normalized
    curve ``(exp(alpha * t) - 1) / (exp(alpha) - 1)``, so larger ``|alpha|`` concentrates the change
    near one end of the ``[0, 1]`` interval.

    Attributes:
        start: Value at ``t = 0``.
        end: Value at ``t = 1``.
        alpha: Exponential warping factor; ``0`` yields linear interpolation.

    Examples:
        >>> schedule = ExponentialInterpolation(start=0.0, end=1.0)
        >>> schedule(0.0)
        0.0
        >>> schedule(0.5)
        0.5
        >>> schedule(1.0)
        1.0
    """

    start: float
    end: float
    alpha: float = 0.0

    def __call__(self, t: float) -> float:
        r"""
        Evaluate the interpolated (optionally warped) value at ``t``.

        Args:
            t: Normalized sampling time in ``[0, 1]``.

        Returns:
            The interpolated value between `start` and `end`.
        """

        if self.alpha == 0.0:
            return float(self.start + (self.end - self.start) * t)
        numerator = math.exp(self.alpha * t) - 1.0
        denominator = math.exp(self.alpha) - 1.0
        return float(self.start + (self.end - self.start) * numerator / denominator)


def schedule_from_cfg(obj: Any) -> Schedule:
    r"""
    Build a [`Schedule`][] from a config value.

    Accepts a ready-made schedule, a scalar (wrapped as a [`Constant`][]), or a mapping with a
    ``type`` key selecting ``"const"`` or ``"exp_interpolation"``.

    Args:
        obj: Schedule instance, scalar, or config mapping describing the schedule.

    Returns:
        The resolved schedule.

    Raises:
        ValueError: If the mapping specifies an unknown schedule ``type``.
        TypeError: If ``obj`` is not a schedule, scalar, or mapping.

    Examples:
        >>> schedule_from_cfg(0.5)(0.0)
        0.5
        >>> schedule_from_cfg({"type": "const", "value": 2.0})(1.0)
        2.0
        >>> schedule_from_cfg({"type": "exp_interpolation", "start": 0.0, "end": 1.0})(0.5)
        0.5
    """

    if isinstance(obj, Schedule):
        return obj
    if isinstance(obj, (int, float)):
        return Constant(float(obj))
    if isinstance(obj, Mapping):
        cfg = dict(obj)
        schedule_type = str(cfg.pop("type")).lower()
        if schedule_type == "const":
            return Constant(float(cfg["value"]))
        if schedule_type == "exp_interpolation":
            return ExponentialInterpolation(
                start=float(cfg["start"]),
                end=float(cfg["end"]),
                alpha=float(cfg.get("alpha", 0.0)),
            )
        raise ValueError(f"unknown TFG schedule type: {schedule_type!r}")
    raise TypeError(f"unsupported TFG schedule config type: {type(obj)!r}")


@dataclass(slots=True)
class Term:
    r"""
    A single scheduled TFG potential with its activation cadence and parameters.

    Pairs a `Potential` with a time-dependent weight schedule, per-parameter schedules, and a
    step interval controlling when the term contributes during guided sampling.

    Attributes:
        name: Potential name, matching a key in the potential registry.
        interval: Step cadence; the term is active on steps whose index is a multiple of this value.
        weight: Schedule producing the term's scalar weight at a given normalized time.
        param_templates: Per-parameter values, each either a scalar or a [`Schedule`][].
        potential: The underlying `Potential` implementation.
        enable_projection: Whether this term participates in the projection phase.
    """

    name: str
    interval: int
    weight: Schedule
    param_templates: dict[str, Any]
    potential: Potential
    enable_projection: bool = True

    def required_features(self) -> set[str]:
        r"""
        Return the guidance feature names this potential requires.

        Returns:
            Set of required feature names, empty when the potential has no requirements.
        """

        return set(_REQUIRED_FEATURES.get(self.name, set()))

    def active(self, step_index: int) -> bool:
        r"""
        Return whether the term contributes at the given diffusion step.

        Args:
            step_index: Zero-based diffusion step index.

        Returns:
            ``True`` when `interval` is positive and divides ``step_index``.
        """

        return self.interval > 0 and step_index % self.interval == 0

    def params_at(self, t: float) -> dict[str, Any]:
        r"""
        Resolve the term's parameters at normalized time ``t``.

        Any parameter that is a [`Schedule`][] is evaluated at ``t``; scalars pass through.

        Args:
            t: Normalized sampling time in ``[0, 1]``.

        Returns:
            Mapping of parameter name to its resolved scalar value.
        """

        return {key: value(t) if isinstance(value, Schedule) else value for key, value in self.param_templates.items()}

    def energy(self, coords: Tensor, features: Mapping[str, Tensor], t: float) -> Tensor:
        r"""
        Return the weighted potential energy for a coordinate batch.

        Short-circuits to zero when the schedule weight is zero at ``t``.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            t: Normalized sampling time in ``[0, 1]``.

        Returns:
            Per-batch energy ``[...]`` scaled by the term weight.
        """

        weight = float(self.weight(t))
        if weight == 0.0:
            return _zeros_energy(coords)
        return self.potential.energy(coords, features, self.params_at(t)) * weight

    def energy_and_grad(self, coords: Tensor, features: Mapping[str, Tensor], t: float) -> tuple[Tensor, Tensor]:
        r"""
        Return the weighted potential energy and its coordinate gradient.

        Short-circuits to zeros when the schedule weight is zero at ``t``.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            t: Normalized sampling time in ``[0, 1]``.

        Returns:
            ``(energy, grad)`` with shapes ``[...]`` and ``[..., N_atom, 3]``, both scaled by the weight.
        """

        weight = float(self.weight(t))
        if weight == 0.0:
            return _zeros_energy_and_grad(coords)
        energy, grad = self.potential.energy_and_grad(coords, features, self.params_at(t))
        return energy * weight, grad * weight

    def project(self, coords: Tensor, features: Mapping[str, Tensor], t: float) -> Tensor:
        r"""
        Return the term's constraint-projection displacement for the coordinates.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            t: Normalized sampling time in ``[0, 1]``.

        Returns:
            Coordinate displacement ``[..., N_atom, 3]`` that reduces the term's constraint violation.
        """

        return self.potential.project(coords, features, self.params_at(t))


@dataclass(frozen=True, slots=True)
class TFGConfig:
    r"""
    Resolved training-free-guidance configuration for the constraint sampler.

    Groups the guidance step counts, Monte-Carlo smoothing settings, path-shift/gradient strengths,
    and the scheduled potential terms. Produced by [`parse_tfg_config`][].

    Attributes:
        enable: Whether guidance is active during sampling.
        rho: Strength of the denoiser-path-shift step.
        mu: Strength of the inner log-probability gradient ascent step.
        eps_std: Standard deviation of the Monte-Carlo smoothing noise.
        eps_batch: Number of Monte-Carlo smoothing samples (at least 1).
        outer_steps: Number of outer guidance iterations per diffusion step.
        inner_steps: Number of inner gradient-ascent iterations per outer step.
        projection_outer_steps: Number of outer constraint-projection passes.
        projection_inner_steps: Number of inner constraint-projection iterations per outer pass.
        terms: Scheduled potential terms applied during guidance.
        log_last_step_energy: Whether to evaluate per-term energy on the final step for logging.
    """

    enable: bool
    rho: float
    mu: float
    eps_std: float
    eps_batch: int
    outer_steps: int
    inner_steps: int
    projection_outer_steps: int
    projection_inner_steps: int
    terms: tuple[Term, ...]
    log_last_step_energy: bool = False


def parse_tfg_config(config: Mapping[str, Any] | None) -> TFGConfig:
    r"""
    Parse a raw guidance-config mapping into a validated [`TFGConfig`][].

    A ``None`` config yields a disabled configuration with no terms. Unknown top-level keys raise, and
    enabling guidance without any configured terms is rejected.

    Args:
        config: Raw guidance config with keys ``enable``, ``rho``, ``mu``, ``mc``, ``steps``,
            ``terms``, and ``log_last_step_energy``; or ``None`` to disable guidance.

    Returns:
        The resolved guidance configuration.

    Raises:
        KeyError: If ``config`` contains unsupported top-level keys.
        ValueError: If guidance is enabled but no terms are configured.

    Examples:
        >>> config = parse_tfg_config(None)
        >>> config.enable
        False
        >>> config.terms
        ()
    """

    if config is None:
        return TFGConfig(
            enable=False,
            rho=0.0,
            mu=0.0,
            eps_std=0.0,
            eps_batch=1,
            outer_steps=1,
            inner_steps=0,
            projection_outer_steps=0,
            projection_inner_steps=0,
            terms=(),
        )
    cfg = dict(config)
    allowed = {"enable", "rho", "mu", "mc", "steps", "terms", "log_last_step_energy"}
    extra = set(cfg) - allowed
    if extra:
        raise KeyError(f"unsupported TFG config keys: {sorted(extra)}")

    terms = tuple(_build_terms(cfg.get("terms", {})))
    enable = bool(cfg.get("enable", False))
    if enable and not terms:
        raise ValueError("TFG is enabled but no terms are configured")
    mc = dict(cfg.get("mc", {}))
    steps = dict(cfg.get("steps", {}))
    return TFGConfig(
        enable=enable,
        rho=float(cfg.get("rho", 0.0)),
        mu=float(cfg.get("mu", 0.0)),
        eps_std=float(mc.get("std", 0.0)),
        eps_batch=max(1, int(mc.get("batch", 1))),
        outer_steps=max(1, int(steps.get("tfg_outer", 1))),
        inner_steps=max(0, int(steps.get("tfg_inner", 10))),
        projection_outer_steps=max(0, int(steps.get("projection_outer", 2))),
        projection_inner_steps=max(0, int(steps.get("projection_inner", 10))),
        terms=terms,
        log_last_step_energy=bool(cfg.get("log_last_step_energy", False)),
    )


def validate_features(features: Mapping[str, Tensor], terms: Iterable[Term]) -> None:
    r"""
    Check that every required guidance feature is present for the given terms.

    Args:
        features: Available guidance feature tensors keyed by Protenix names.
        terms: Scheduled potential terms whose feature requirements are checked.

    Raises:
        KeyError: If any term is missing one or more required feature tensors; the message lists the
            offending term names and the absent feature keys.

    Examples:
        >>> validate_features({}, [])
    """

    missing: dict[str, list[str]] = {}
    for term in terms:
        absent = sorted(key for key in term.required_features() if key not in features)
        if absent:
            missing[term.name] = absent
    if missing:
        lines = ["TFG is missing required input features:"]
        for name, absent in missing.items():
            lines.append(f"- {name}: {absent}")
        raise KeyError("\n".join(lines))


class ProtenixTFGEngine:
    r"""
    Training-free-guidance update driving each step of the Protenix coordinate sampler.

    Wraps the guidance configuration and applies, per diffusion step, a denoiser-path-shift,
    constraint projection, and inner log-probability gradient ascent so the sampled coordinates
    satisfy the configured geometry potentials. Instantiated per sampling call by the model.

    Args:
        config: Resolved guidance configuration (see [`parse_tfg_config`][]).
        device: Device on which guidance tensors are allocated.
        dtype: Floating dtype used for guidance computations.
    """

    def __init__(self, config: TFGConfig, *, device: torch.device, dtype: torch.dtype) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self._projection_terms = sorted(config.terms, key=_projection_order)

    def step(
        self,
        denoise: Any,
        *,
        context: CoordinateDenoiseContext,
        coords: Tensor,
        noise_level: Tensor,
        next_noise_level: Tensor,
        step_scale: Tensor,
        step_index: int,
        num_diffusion_steps: int,
        features: Mapping[str, Tensor],
        generator: torch.Generator | None,
    ) -> Tensor:
        r"""
        Advance one diffusion step under training-free guidance.

        Applies the denoiser-path shift, one denoise call, constraint projection, and (when ``mu``
        is nonzero) inner log-probability gradient ascent, then takes the sampler step from
        ``noise_level`` to ``next_noise_level``. Feature availability is validated on the first step.

        Args:
            denoise: Callable coordinate denoiser accepting a denoising input and returning coords.
            context: Denoiser context bundling the trunk representations and atom features.
            coords: Current noisy coordinates ``[..., N_sample, N_atom, 3]``.
            noise_level: Current per-sample noise level ``[..., N_sample]``.
            next_noise_level: Target per-sample noise level for the step ``[..., N_sample]``.
            step_scale: Sampler step-size scaling for this transition.
            step_index: Zero-based diffusion step index.
            num_diffusion_steps: Total number of diffusion transitions.
            features: Guidance feature tensors keyed by Protenix names.
            generator: Optional RNG for reproducible noise draws.

        Returns:
            Coordinates after the guided step ``[..., N_sample, N_atom, 3]``.

        Raises:
            KeyError: On the first step, if a configured term is missing a required feature.
        """

        if step_index == 0:
            validate_features(features, self.config.terms)

        t = 1.0 - float(step_index) / float(max(1, num_diffusion_steps))
        eps = _sample_eps(
            self.config.eps_std,
            coords.shape,
            count=self.config.eps_batch,
            device=coords.device,
            dtype=coords.dtype,
            generator=generator,
        )

        work = coords
        next_coords = coords
        for outer_index in range(self.config.outer_steps):
            shift = self._denoiser_path_shift(
                denoise,
                context=context,
                coords=work,
                noise_level=noise_level,
                features=features,
                eps=eps,
                t=t,
                step_index=step_index,
            )
            with torch.no_grad():
                denoised = denoise(context.to_denoising_input(noisy_coords=work + shift, noise_level=noise_level))

            refined = denoised.detach()
            refined = refined + self._project(refined, features, t=t, step_index=step_index)
            for inner_index in range(self.config.inner_steps):
                if self.config.mu == 0.0:
                    break
                log_components = (
                    self.config.log_last_step_energy
                    and step_index == num_diffusion_steps - 1
                    and outer_index == self.config.outer_steps - 1
                    and inner_index == self.config.inner_steps - 1
                )
                _, grad = self._logp_and_grad_x0(
                    refined,
                    eps,
                    features,
                    t=t,
                    step_index=step_index,
                    log_components=log_components,
                )
                refined = refined + grad * float(self.config.mu)

            direction = (work + shift - refined) / noise_level[..., None, None]
            dt = next_noise_level - noise_level
            next_coords = work + shift + step_scale * dt[..., None, None] * direction
            sigma = torch.sqrt((noise_level**2 - next_noise_level**2).clamp_min(0.0))
            noise = torch.empty_like(next_coords).normal_(generator=generator)
            work = next_coords + sigma[..., None, None] * noise
        return next_coords

    def _denoiser_path_shift(
        self,
        denoise: Any,
        *,
        context: CoordinateDenoiseContext,
        coords: Tensor,
        noise_level: Tensor,
        features: Mapping[str, Tensor],
        eps: Tensor,
        t: float,
        step_index: int,
    ) -> Tensor:
        r"""
        Return the guidance path shift added before the denoise call.

        Computes ``rho`` times the gradient of the smoothed log-probability with respect to the
        noisy coordinates; returns zeros when ``rho`` is zero or the gradient does not flow.

        Args:
            denoise: Coordinate denoiser callable.
            context: Denoiser context bundle.
            coords: Current noisy coordinates ``[..., N_atom, 3]``.
            noise_level: Current noise level.
            features: Guidance feature tensors.
            eps: Monte-Carlo smoothing noise ``[count, ..., N_atom, 3]``.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.

        Returns:
            Coordinate shift ``[..., N_atom, 3]``.
        """

        if self.config.rho == 0.0:
            return torch.zeros_like(coords)
        with torch.enable_grad():
            variable = coords.detach().requires_grad_(True)
            denoised = denoise(context.to_denoising_input(noisy_coords=variable, noise_level=noise_level))
            logp = self._logp_x0(denoised, eps, features, t=t, step_index=step_index)
            grad = torch.autograd.grad(logp.sum(), variable, allow_unused=True)[0]
        if grad is None:
            return torch.zeros_like(coords)
        return grad.detach() * float(self.config.rho)

    def _logp_x0(
        self,
        coords: Tensor,
        eps: Tensor,
        features: Mapping[str, Tensor],
        *,
        t: float,
        step_index: int,
    ) -> Tensor:
        r"""
        Return the Monte-Carlo-smoothed log-probability of the clean coordinates.

        Evaluates the total potential energy over ``eps``-perturbed copies of ``coords`` and reduces
        with a log-sum-exp over the smoothing samples (identity when there is a single sample).

        Args:
            coords: Clean coordinate estimate ``[..., N_atom, 3]``.
            eps: Monte-Carlo smoothing noise ``[count, ..., N_atom, 3]``.
            features: Guidance feature tensors.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.

        Returns:
            Per-batch log-probability ``[...]``.
        """

        count = eps.shape[0]
        coords_eps = (coords.unsqueeze(0) + eps).reshape(-1, *coords.shape[-2:])
        energy = self._energy(coords_eps, features, t=t, step_index=step_index)
        logp = -energy.reshape(count, *coords.shape[:-2])
        if count == 1:
            return logp.squeeze(0)
        return torch.logsumexp(logp, dim=0) - math.log(count)

    def _logp_and_grad_x0(
        self,
        coords: Tensor,
        eps: Tensor,
        features: Mapping[str, Tensor],
        *,
        t: float,
        step_index: int,
        log_components: bool,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Return the smoothed log-probability and its gradient for the clean coordinates.

        Softmax-weights the per-sample energy gradients (importance weights from the smoothed
        log-probability) so gradient ascent on the returned gradient raises the log-probability.

        Args:
            coords: Clean coordinate estimate ``[..., N_atom, 3]``.
            eps: Monte-Carlo smoothing noise ``[count, ..., N_atom, 3]``.
            features: Guidance feature tensors.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.
            log_components: Whether to additionally evaluate per-term energies for logging.

        Returns:
            ``(log_probability, gradient)`` with shapes ``[...]`` and ``[..., N_atom, 3]``.
        """

        count = eps.shape[0]
        coords_eps = (coords.unsqueeze(0) + eps).reshape(-1, *coords.shape[-2:])
        energy, grad_energy = self._energy_and_grad(coords_eps, features, t=t, step_index=step_index)
        energy = energy.reshape(count, *coords.shape[:-2])
        grad_energy = grad_energy.reshape(count, *coords.shape)
        logp = -energy
        if count == 1:
            avg_logp = logp.squeeze(0)
            grad = (-grad_energy).squeeze(0)
        else:
            avg_logp = torch.logsumexp(logp, dim=0) - math.log(count)
            weights = torch.softmax(logp, dim=0)
            grad = (weights[..., None, None] * -grad_energy).sum(dim=0)
        if log_components:
            self._log_last_step_energy(coords, features, t=t, step_index=step_index)
        return avg_logp, grad

    def _energy_and_grad(
        self,
        coords: Tensor,
        features: Mapping[str, Tensor],
        *,
        t: float,
        step_index: int,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Sum the energy and gradient of every term active at the given step.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.

        Returns:
            ``(energy, gradient)`` with shapes ``[...]`` and ``[..., N_atom, 3]``.
        """

        energy = torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)
        grad = torch.zeros_like(coords)
        for term in self.config.terms:
            if not term.active(step_index):
                continue
            term_energy, term_grad = term.energy_and_grad(coords, features, t)
            energy = energy + term_energy
            grad = grad + term_grad
        return energy, grad

    def _energy(
        self,
        coords: Tensor,
        features: Mapping[str, Tensor],
        *,
        t: float,
        step_index: int,
    ) -> Tensor:
        r"""
        Sum the energy of every term active at the given step.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.

        Returns:
            Per-batch total energy ``[...]``.
        """

        energy = torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)
        for term in self.config.terms:
            if term.active(step_index):
                energy = energy + term.energy(coords, features, t)
        return energy

    def _project(self, coords: Tensor, features: Mapping[str, Tensor], *, t: float, step_index: int) -> Tensor:
        r"""
        Return the accumulated constraint-projection displacement across active terms.

        Runs the configured outer/inner projection passes over the projection-capable terms, ordered
        so higher-priority constraints (chiral, then pairwise distance) project first.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.

        Returns:
            Coordinate displacement ``[..., N_atom, 3]`` reducing constraint violations.
        """

        if self.config.projection_outer_steps <= 0 or self.config.projection_inner_steps <= 0:
            return torch.zeros_like(coords)
        delta = torch.zeros_like(coords)
        terms = [term for term in self._projection_terms if term.potential.has_projection]
        for _ in range(self.config.projection_outer_steps):
            for term in terms:
                if not term.active(step_index) or not term.enable_projection:
                    continue
                for _ in range(self.config.projection_inner_steps):
                    delta = delta + term.project(coords + delta, features, t)
        return delta

    def _log_last_step_energy(
        self,
        coords: Tensor,
        features: Mapping[str, Tensor],
        *,
        t: float,
        step_index: int,
    ) -> None:
        r"""
        Evaluate each active term's energy on the final step for logging side effects.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors.
            t: Normalized sampling time in ``[0, 1]``.
            step_index: Zero-based diffusion step index.
        """

        for term in self.config.terms:
            if term.active(step_index):
                _ = term.energy(coords, features, t)


class Potential:
    r"""
    Base class for a training-free-guidance geometry potential.

    A potential scores how well a coordinate set satisfies a geometric constraint (bond length,
    angle, dihedral, steric clash, ...) and can supply gradients and, optionally, a constraint
    projection. Subclasses override ``_eval`` (and ``project`` when they support projection) and
    declare per-potential default parameters via the constructor.

    Args:
        defaults: Default parameter values merged under any per-call overrides.
    """

    has_projection = False

    def __init__(self, defaults: Mapping[str, Any] | None = None) -> None:
        self.defaults = dict(defaults or {})

    def energy(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any]) -> Tensor:
        r"""
        Return the potential energy for a coordinate batch.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            params: Per-call parameter overrides merged over the potential defaults.

        Returns:
            Per-batch energy ``[...]``.
        """

        return self._eval(coords, features, self._resolve(params), need_grad=False)

    def energy_and_grad(
        self,
        coords: Tensor,
        features: Mapping[str, Tensor],
        params: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor]:
        r"""
        Return the potential energy and its coordinate gradient.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            params: Per-call parameter overrides merged over the potential defaults.

        Returns:
            ``(energy, gradient)`` with shapes ``[...]`` and ``[..., N_atom, 3]``.
        """

        return self._eval(coords, features, self._resolve(params), need_grad=True)

    def project(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any]) -> Tensor:
        r"""
        Return a constraint-projection displacement for the coordinates.

        The base implementation performs no projection; projection-capable subclasses override it.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            params: Per-call parameter overrides merged over the potential defaults.

        Returns:
            Coordinate displacement ``[..., N_atom, 3]``; zeros for non-projecting potentials.
        """

        return torch.zeros_like(coords)

    def _resolve(self, params: Mapping[str, Any]) -> dict[str, Any]:
        r"""
        Merge per-call parameters over the potential defaults.

        Args:
            params: Per-call parameter overrides.

        Returns:
            The merged parameter dictionary.
        """

        merged = dict(self.defaults)
        merged.update(params)
        return merged

    def _eval(
        self,
        coords: Tensor,
        features: Mapping[str, Tensor],
        params: Mapping[str, Any],
        need_grad: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Evaluate the potential (and optionally its gradient); implemented by subclasses.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            params: Resolved parameters.
            need_grad: Whether to also return the coordinate gradient.

        Returns:
            The energy, or an ``(energy, gradient)`` tuple when ``need_grad`` is set.

        Raises:
            TypeError: Always, since the base potential is abstract.
        """

        raise TypeError("Potential is an abstract TFG potential base")


class InterchainBondPotential(Potential):
    r"""
    Penalize interchain bonds whose length exceeds an upper bound.

    Uses a flat-bottom linear penalty on the distance of each interchain bonded atom pair, keeping
    covalently linked chains within a ``buffer`` distance.
    """

    def __init__(self) -> None:
        super().__init__({"buffer": 2.0})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the interchain-bond energy (and gradient if requested)."""
        index = features["interchain_bond_index"].to(device=coords.device)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _distance_value_and_grad(coords, index, need_grad)
        upper = torch.full((index.shape[1],), float(params["buffer"]), device=coords.device, dtype=value.dtype)
        energy, d_energy = _flat_bottom_linear(value, torch.ones_like(value), None, upper)
        if not need_grad:
            return _sum_energy(energy)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)


class PairwiseDistancePotential(Potential):
    r"""
    Keep bonded, angular, and non-bonded atom pairs within RDKit-derived distance bounds.

    Applies a flat-bottom parabolic penalty between per-pair lower/upper distance bounds. Bounds come
    from the RDKit bounds matrix, tightened by bond/angle/clash buffers and by van-der-Waals radii for
    non-bonded pairs. Supports constraint projection for bond and angle pairs.
    """

    has_projection = True

    def __init__(self) -> None:
        super().__init__({"bond_buffer": 0.05, "angle_buffer": 0.05, "clash_buffer": 0.05})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the flat-bottom parabolic distance-bound energy (and gradient if requested)."""
        index, lower, upper = self._bounds(features, coords.device, coords.dtype, params)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _distance_value_and_grad(coords, index, need_grad)
        energy, d_energy = _flat_bottom_parabolic(value, torch.ones_like(value), lower, upper)
        if not need_grad:
            return _sum_energy(energy)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)

    def project(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any]) -> Tensor:
        r"""
        Project angle pairs then bond pairs onto their distance bounds, returning the displacement.

        Angles are projected first, then bonds are projected on the angle-corrected coordinates so the
        two passes compose into a single displacement.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            params: Per-call parameter overrides merged over the potential defaults.

        Returns:
            Coordinate displacement ``[..., N_atom, 3]``; zeros when no bond or angle pair is violated.
        """
        index, lower, upper = self._bounds(features, coords.device, coords.dtype, params)
        if index.numel() == 0:
            return torch.zeros_like(coords)
        is_bond = features["pairwise_distance_is_bond"].to(device=coords.device).bool()
        is_angle = features["pairwise_distance_is_angle"].to(device=coords.device).bool()
        delta_angle = self._project_mask(coords, index, lower, upper, is_angle)
        delta_bond = self._project_mask(coords + delta_angle, index, lower, upper, is_bond)
        return delta_bond + delta_angle

    def _bounds(
        self,
        features: Mapping[str, Tensor],
        device: torch.device,
        dtype: torch.dtype,
        params: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Build the per-pair index and tightened lower/upper distance bounds.

        Applies the bond/angle/clash buffer tables per pair state and clamps non-bonded pairs by their
        van-der-Waals contact radius (bonds are capped by it, non-bonds floored by it).

        Args:
            features: Guidance feature tensors keyed by Protenix names.
            device: Target device.
            dtype: Target floating dtype for the bounds.
            params: Resolved buffer parameters.

        Returns:
            ``(index, lower, upper)`` with ``index`` shape ``[2, P]`` and bounds shape ``[P]``.
        """

        index = features["pairwise_distance_index"].to(device=device, dtype=torch.long)
        lower = features["pairwise_distance_lower_bound"].to(device=device, dtype=dtype)
        upper = features["pairwise_distance_upper_bound"].to(device=device, dtype=dtype)
        if index.numel() == 0:
            return index, lower, upper
        is_bond = features["pairwise_distance_is_bond"].to(device=device, dtype=torch.long)
        is_angle = features["pairwise_distance_is_angle"].to(device=device, dtype=torch.long)
        state = is_bond + (is_angle << 1)
        lower_table = torch.tensor(
            [
                float(params["clash_buffer"]),
                float(params["bond_buffer"]),
                float(params["angle_buffer"]),
                min(float(params["bond_buffer"]), float(params["angle_buffer"])),
            ],
            device=device,
            dtype=dtype,
        )
        upper_table = torch.tensor(
            [
                0.0,
                float(params["bond_buffer"]),
                float(params["angle_buffer"]),
                min(float(params["bond_buffer"]), float(params["angle_buffer"])),
            ],
            device=device,
            dtype=dtype,
        )
        lower = lower * (1.0 - lower_table[state])
        upper = upper * (1.0 + upper_table[state])
        upper = torch.where(state == 0, torch.full_like(upper, float("inf")), upper)

        element = features["ref_element"].to(device=device)
        radii = _vdw_radii(device).to(dtype=dtype)
        atom_radii = radii.index_select(0, element.argmax(dim=-1).to(torch.long).clamp_max(117))
        vdw_limit = 0.35 + 0.5 * (atom_radii[index[0]] + atom_radii[index[1]])
        bond_mask = (state & 1).bool()
        lower = torch.where(~bond_mask, torch.maximum(lower, vdw_limit), lower)
        upper = torch.where(bond_mask, torch.minimum(upper, vdw_limit), upper)
        return index, lower, upper

    def _project_mask(self, coords: Tensor, index: Tensor, lower: Tensor, upper: Tensor, mask_index: Tensor) -> Tensor:
        r"""
        Project the masked subset of pairs onto their distance bounds.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            index: Pair index ``[2, P]``.
            lower: Per-pair lower bounds ``[P]``.
            upper: Per-pair upper bounds ``[P]``.
            mask_index: Boolean mask selecting the pairs to project.

        Returns:
            Coordinate displacement ``[..., N_atom, 3]``; zeros when no masked pair is violated.
        """

        index = index[:, mask_index]
        if index.numel() == 0:
            return torch.zeros_like(coords)
        value, grad_value = _distance_value_and_grad(coords, index, True)
        lower = lower[mask_index]
        upper = upper[mask_index]
        mask_low = value < lower
        mask_high = value > upper
        active = mask_low | mask_high
        if not bool(active.any()):
            return torch.zeros_like(coords)
        violation = value - torch.where(mask_low, lower, upper)
        return _solve_constraint_projection(coords, index, violation, grad_value, active)


class StereoBondPotential(Potential):
    r"""
    Enforce E/Z double-bond stereochemistry via the substituent dihedral.

    Applies a flat-bottom linear penalty on the absolute dihedral of each stereo double bond, driving
    E bonds toward ``pi`` and Z bonds toward ``0`` within a ``buffer`` tolerance.
    """

    def __init__(self) -> None:
        super().__init__({"buffer": 0.52360})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the stereo-bond dihedral energy (and gradient if requested)."""
        index = features["stereo_bond_index"].to(device=coords.device)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _abs_dihedral_value_and_grad(coords, index, need_grad)
        orientation = features["stereo_bond_orientation"].to(device=coords.device, dtype=value.dtype)
        buffer = float(params["buffer"])
        lower = torch.where(orientation > 0.5, torch.full_like(orientation, math.pi - buffer), -torch.inf)
        upper = torch.where(orientation > 0.5, torch.full_like(orientation, torch.inf), buffer)
        energy, d_energy = _flat_bottom_linear(value, torch.ones_like(value), lower, upper)
        if not need_grad:
            return _sum_energy(energy)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)


class ChiralAtomPotential(Potential):
    r"""
    Enforce tetrahedral chirality via the signed substituent dihedral.

    Applies a flat-bottom linear penalty on the signed chiral dihedral so its sign matches the
    reference orientation, keeping it beyond a ``buffer`` from the planar (racemization) value.
    Supports constraint projection.
    """

    has_projection = True

    def __init__(self) -> None:
        super().__init__({"buffer": 0.34906, "scale_x": True})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the chiral-dihedral energy (and gradient if requested)."""
        index = features["chiral_index"].to(device=coords.device)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _dihedral_value_and_grad(coords, index, need_grad)
        orientation = features["chiral_orientation"].to(device=coords.device, dtype=value.dtype)
        lower = torch.where(orientation > 0, torch.full_like(orientation, float(params["buffer"])), -torch.inf)
        upper = torch.where(orientation > 0, torch.full_like(orientation, torch.inf), -float(params["buffer"]))
        energy, d_energy = _flat_bottom_linear(value, torch.ones_like(value), lower, upper)
        if not need_grad:
            return _sum_energy(energy)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)

    def project(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any]) -> Tensor:
        r"""
        Project chiral centers on the wrong side of the buffer back past it, returning the displacement.

        Only centers whose signed dihedral has fallen short of the ``buffer`` (wrong orientation) are
        moved; correctly oriented centers are left untouched.

        Args:
            coords: Atom coordinates ``[..., N_atom, 3]``.
            features: Guidance feature tensors keyed by Protenix names.
            params: Per-call parameter overrides merged over the potential defaults.

        Returns:
            Coordinate displacement ``[..., N_atom, 3]``; zeros when no chiral center is violated.
        """
        index = features["chiral_index"].to(device=coords.device)
        if index.numel() == 0:
            return torch.zeros_like(coords)
        orientation = features["chiral_orientation"].to(device=coords.device, dtype=coords.dtype)
        value, grad_value = _dihedral_value_and_grad(coords, index, True)
        value = value * orientation - float(params["buffer"])
        grad_value = grad_value * orientation.reshape(*([1] * len(coords.shape[:-2])), 1, -1, 1)
        active = value < 0
        if not bool(active.any()):
            return torch.zeros_like(coords)
        return _solve_constraint_projection(coords, index, value, grad_value, active)


class PlanarImproperPotential(Potential):
    r"""
    Keep sp2 centers planar via their improper dihedral.

    Applies a flat-bottom linear penalty on the absolute improper dihedral of each sp2 center,
    driving carbonyl centers toward ``pi`` and other planar centers toward ``0`` within a ``buffer``.
    """

    def __init__(self) -> None:
        super().__init__({"buffer": 0.1309})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the planar-improper dihedral energy (and gradient if requested)."""
        index = features["planar_improper_index"].to(device=coords.device)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _abs_dihedral_value_and_grad(coords, index, need_grad)
        carbonyl = features["planar_improper_is_carbonyl"].to(device=coords.device, dtype=value.dtype)
        lower = torch.where(carbonyl > 0.5, torch.full_like(carbonyl, math.pi - float(params["buffer"])), -torch.inf)
        upper = torch.where(carbonyl > 0.5, torch.full_like(carbonyl, torch.inf), float(params["buffer"]))
        energy, d_energy = _flat_bottom_linear(value, torch.ones_like(value), lower, upper)
        if not need_grad:
            return _sum_energy(energy)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)


class LinearBondPotential(Potential):
    r"""
    Keep triple-bonded (sp) centers linear via the substituent bond angle.

    Applies a flat-bottom linear penalty that drives each substituent-to-triple-bond angle toward
    ``pi`` within a ``buffer`` tolerance.
    """

    def __init__(self) -> None:
        super().__init__({"buffer": 0.08726646259})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the linear-bond angle energy (and gradient if requested)."""
        index = features["linear_triple_bond_index"].to(device=coords.device)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _angle_value_and_grad(coords, index, need_grad)
        lower = torch.full(
            (index.shape[1],),
            math.pi - float(params["buffer"]),
            device=coords.device,
            dtype=value.dtype,
        )
        energy, d_energy = _flat_bottom_linear(value, torch.ones_like(value), lower, None)
        if not need_grad:
            return _sum_energy(energy)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)


class ExperimentalTorsionPotential(Potential):
    r"""
    Score torsions against RDKit experimental-torsion Fourier profiles.

    Evaluates the ETKDG cosine-series energy ``sum_k V_k * (1 + sign_k * cos(k * phi))`` for each
    torsion, biasing dihedrals toward their empirically preferred angles.
    """

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the experimental-torsion Fourier energy (and gradient if requested)."""
        index = features["experimental_torsion_index"].to(device=coords.device)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        value, grad_value = _dihedral_value_and_grad(coords, index, need_grad)
        force = features["experimental_torsion_force_constant"].to(device=coords.device, dtype=value.dtype)
        sign = features["experimental_torsion_sign"].to(device=coords.device, dtype=value.dtype)
        cos1 = torch.cos(value)
        cos2 = 2.0 * cos1**2 - 1.0
        cos3 = 4.0 * cos1**3 - 3.0 * cos1
        cos4 = 8.0 * cos1**4 - 8.0 * cos1**2 + 1.0
        cos5 = 16.0 * cos1**5 - 20.0 * cos1**3 + 5.0 * cos1
        cos6 = 32.0 * cos1**6 - 48.0 * cos1**4 + 18.0 * cos1**2 - 1.0
        cosines = torch.stack((cos1, cos2, cos3, cos4, cos5, cos6), dim=-1)
        energy = (force * (1.0 + sign * cosines)).sum(dim=-1)
        if not need_grad:
            return _sum_energy(energy)
        sin1 = torch.sin(value)
        dcos = torch.stack(
            (
                torch.ones_like(value),
                4.0 * cos1,
                12.0 * cos1**2 - 3.0,
                32.0 * cos1**3 - 16.0 * cos1,
                80.0 * cos1**4 - 60.0 * cos1**2 + 5.0,
                192.0 * cos1**5 - 192.0 * cos1**3 + 36.0 * cos1,
            ),
            dim=-1,
        )
        d_energy = -sin1 * (force * sign * dcos).sum(dim=-1)
        return _sum_energy(energy), _aggregate_atom_gradients(coords, index, grad_value, d_energy)


class VinaStericPotential(Potential):
    r"""
    Penalize interchain steric clashes with an AutoDock-Vina-style term.

    Builds interchain, non-bonded atom-pair candidates and, for pairs closer than their
    van-der-Waals equilibrium distance (minus ``buffer``), applies the Vina Gaussian-plus-repulsion
    energy to relieve clashes between chains.
    """

    def __init__(self) -> None:
        super().__init__({"buffer": 0.225})

    def _eval(self, coords: Tensor, features: Mapping[str, Tensor], params: Mapping[str, Any], need_grad: bool):
        r"""Evaluate the Vina steric-clash energy (and gradient if requested)."""
        index, equilibrium = self._collision_candidates(features, coords.device, coords.dtype)
        if index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)
        buffer = float(params["buffer"])
        batch_shape = coords.shape[:-2]
        num_atoms = int(coords.shape[-2])
        flat_batch = math.prod(batch_shape) if batch_shape else 1
        coords_b = coords.reshape(flat_batch, num_atoms, 3)
        value, _ = _distance_value_and_grad(coords_b, index, False)
        active = value < equilibrium.to(value.dtype).unsqueeze(0) * (1.0 - buffer)
        batch_index, pair_index = active.nonzero(as_tuple=True)
        if batch_index.numel() == 0:
            return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)

        atom_i = index[0].index_select(0, pair_index)
        atom_j = index[1].index_select(0, pair_index)
        eq = equilibrium.index_select(0, pair_index).to(coords.dtype)
        distance = value[batch_index, pair_index].to(coords.dtype)
        diff = distance - eq
        norm = diff / 0.5
        g1 = -0.0356 * torch.exp(-(norm**2))
        g2 = -0.00516 * torch.exp(-(((diff - 3.0) / 2.0) ** 2))
        repulsion = 0.840 * torch.where(diff < 0.0, diff**2, 0.0)
        pair_energy = g1 + g2 + repulsion
        energy = torch.zeros((flat_batch,), device=coords.device, dtype=coords.dtype)
        energy.scatter_add_(0, batch_index, pair_energy)
        if not need_grad:
            return energy[0] if not batch_shape else energy.reshape(*batch_shape)

        ri = coords_b[batch_index, atom_i]
        rj = coords_b[batch_index, atom_j]
        vector = ri - rj
        length = torch.linalg.norm(vector, dim=-1).clamp_min(1e-8)
        unit = vector / length.unsqueeze(-1)
        d_g1 = -2.0 * g1 * norm * 2.0
        d_g2 = -0.5 * g2 * (diff - 3.0)
        d_repulsion = 0.840 * torch.where(diff < 0.0, 2.0 * diff, 0.0)
        d_total = d_g1 + d_g2 + d_repulsion
        grad_i = d_total.unsqueeze(-1) * unit
        grad = torch.zeros((flat_batch * num_atoms, 3), device=coords.device, dtype=coords.dtype)
        flat_i = batch_index * num_atoms + atom_i
        flat_j = batch_index * num_atoms + atom_j
        grad.scatter_add_(0, flat_i.unsqueeze(-1).expand(-1, 3), grad_i)
        grad.scatter_add_(0, flat_j.unsqueeze(-1).expand(-1, 3), -grad_i)
        grad = grad.reshape(flat_batch, num_atoms, 3).reshape(*coords.shape)
        return (energy[0] if not batch_shape else energy.reshape(*batch_shape)), grad

    def _collision_candidates(
        self,
        features: Mapping[str, Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Enumerate interchain, non-bonded atom pairs and their vdW equilibrium distances.

        Returns empty tensors when the complex is single-chain. Interchain covalent bonds are
        excluded so bonded links are not treated as clashes.

        Args:
            features: Guidance feature tensors keyed by Protenix names.
            device: Target device.
            dtype: Target floating dtype for the equilibrium distances.

        Returns:
            ``(index, equilibrium)`` with ``index`` shape ``[2, P]`` and ``equilibrium`` shape ``[P]``.
        """

        atom_to_token = features["atom_to_token_idx"].to(device=device, dtype=torch.long)
        asym_id = features["asym_id"].to(device=device, dtype=torch.long)
        if atom_to_token.numel() == 0:
            return _empty_index(2, device), torch.empty((0,), device=device, dtype=dtype)
        atom_chain = asym_id.index_select(0, atom_to_token)
        if torch.all(atom_chain == atom_chain[:1]):
            return _empty_index(2, device), torch.empty((0,), device=device, dtype=dtype)
        pair = torch.combinations(torch.arange(atom_to_token.numel(), device=device), r=2).T
        keep = atom_chain[pair[0]] != atom_chain[pair[1]]
        interchain_bond = features.get("interchain_bond_index")
        if interchain_bond is not None and interchain_bond.numel() > 0:
            bonds = interchain_bond.to(device=device, dtype=torch.long)
            bond_key = torch.minimum(bonds[0], bonds[1]) * atom_to_token.numel() + torch.maximum(bonds[0], bonds[1])
            pair_key = torch.minimum(pair[0], pair[1]) * atom_to_token.numel() + torch.maximum(pair[0], pair[1])
            keep = keep & ~torch.isin(pair_key, bond_key)
        pair = pair[:, keep]
        if pair.numel() == 0:
            return _empty_index(2, device), torch.empty((0,), device=device, dtype=dtype)
        element = features["ref_element"].to(device=device)
        radii = _vdw_radii(device).to(dtype=dtype)
        atom_radii = radii.index_select(0, element.argmax(dim=-1).to(torch.long).clamp_max(117))
        equilibrium = atom_radii[pair[0]] + atom_radii[pair[1]]
        return pair, equilibrium


_POTENTIALS: dict[str, type[Potential]] = {
    "InterchainBondPotential": InterchainBondPotential,
    "PairwiseDistancePotential": PairwiseDistancePotential,
    "StereoBondPotential": StereoBondPotential,
    "ChiralAtomPotential": ChiralAtomPotential,
    "PlanarImproperPotential": PlanarImproperPotential,
    "LinearBondPotential": LinearBondPotential,
    "ExperimentalTorsionPotential": ExperimentalTorsionPotential,
    "VinaStericPotential": VinaStericPotential,
}


def _build_terms(config: Mapping[str, Any] | None) -> list[Term]:
    r"""
    Build the list of scheduled [`Term`][] objects from a raw terms config.

    Args:
        config: Mapping from potential name to its raw term config (interval, weight, per-parameter
            schedules, and ``enable_projection``); ``None`` yields no terms.

    Returns:
        The constructed terms, one per configured potential.

    Raises:
        TypeError: If ``config`` is not a mapping.
        KeyError: If a configured potential name is not registered.
    """

    if config is None:
        return []
    if not isinstance(config, Mapping):
        raise TypeError(f"TFG terms must be a mapping, got {type(config)!r}")
    terms = []
    for name, raw in config.items():
        if name not in _POTENTIALS:
            raise KeyError(f"unknown TFG potential {name!r}; available: {sorted(_POTENTIALS)}")
        raw_config = {} if raw is None else dict(raw)
        interval = int(raw_config.pop("interval", 1))
        weight = schedule_from_cfg(raw_config.pop("weight", 0.0))
        enable_projection = bool(raw_config.pop("enable_projection", True))
        params = {
            key: schedule_from_cfg(value) if isinstance(value, Mapping) and "type" in value else value
            for key, value in raw_config.items()
        }
        terms.append(
            Term(
                name=str(name),
                interval=interval,
                weight=weight,
                param_templates=params,
                potential=_POTENTIALS[str(name)](),
                enable_projection=enable_projection,
            )
        )
    return terms


def _projection_order(term: Term) -> int:
    r"""
    Return a sort key ordering terms for the projection phase.

    Chiral constraints project first, then pairwise-distance constraints, then the rest.

    Args:
        term: The term to rank.

    Returns:
        A small integer priority (lower projects earlier).
    """

    if term.name == "ChiralAtomPotential":
        return 0
    if term.name == "PairwiseDistancePotential":
        return 1
    return 2


def _sample_eps(
    std: float,
    shape: torch.Size,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    r"""
    Draw Monte-Carlo smoothing noise for the guidance log-probability.

    Args:
        std: Noise standard deviation; ``0`` yields a single zero sample.
        shape: Shape of one coordinate tensor to perturb.
        count: Number of smoothing samples.
        device: Target device.
        dtype: Target dtype.
        generator: Optional RNG for reproducibility.

    Returns:
        Noise tensor ``[count, *shape]`` (``[1, *shape]`` zeros when ``std`` is zero).
    """

    if std == 0.0:
        return torch.zeros((1, *shape), device=device, dtype=dtype)
    return std * torch.empty((count, *shape), device=device, dtype=dtype).normal_(generator=generator)


def _distance_value_and_grad(coords: Tensor, index: Tensor, need_grad: bool) -> tuple[Tensor, Tensor | None]:
    r"""
    Compute pairwise distances and their per-endpoint gradients.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.
        index: Atom-pair index ``[2, P]``.
        need_grad: Whether to also return the per-endpoint unit-vector gradients.

    Returns:
        ``(distance, grad)`` with ``distance`` shape ``[..., P]`` and ``grad`` shape ``[..., 2, P, 3]``
        (``None`` when ``need_grad`` is false).
    """

    vector = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    value = torch.linalg.norm(vector, dim=-1).clamp_min(1e-8)
    if not need_grad:
        return value, None
    unit = vector / value.unsqueeze(-1)
    return value, torch.stack((unit, -unit), dim=-3)


def _angle_value_and_grad(coords: Tensor, index: Tensor, need_grad: bool) -> tuple[Tensor, Tensor | None]:
    r"""
    Compute bond angles at the center atom and their per-atom gradients.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.
        index: Atom-triple index ``[3, A]`` as ``(i, center, k)``.
        need_grad: Whether to also return the per-atom gradients.

    Returns:
        ``(angle, grad)`` with ``angle`` shape ``[..., A]`` and ``grad`` shape ``[..., 3, A, 3]``
        (``None`` when ``need_grad`` is false).
    """

    ji = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    jk = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
    cross = torch.linalg.cross(ji, jk, dim=-1)
    dot = torch.linalg.vecdot(ji, jk) + 1e-8
    angle = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), dot)
    if not need_grad:
        return angle, None
    ji2 = torch.linalg.vecdot(ji, ji) + 1e-8
    jk2 = torch.linalg.vecdot(jk, jk) + 1e-8
    cross_norm = torch.linalg.vector_norm(cross, dim=-1).clamp_min(1e-8)
    grad_i = torch.linalg.cross(ji, cross, dim=-1) / (ji2 * cross_norm).unsqueeze(-1)
    grad_k = torch.linalg.cross(cross, jk, dim=-1) / (jk2 * cross_norm).unsqueeze(-1)
    grad_j = -grad_i - grad_k
    return angle, torch.stack((grad_i, grad_j, grad_k), dim=-3)


def _dihedral_value_and_grad(coords: Tensor, index: Tensor, need_grad: bool) -> tuple[Tensor, Tensor | None]:
    r"""
    Compute dihedral angles and their per-atom gradients via autograd.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.
        index: Atom-quadruple index ``[4, D]``.
        need_grad: Whether to also return the per-atom gradients.

    Returns:
        ``(angle, grad)`` with ``angle`` shape ``[..., D]`` and ``grad`` shape ``[..., 4, D, 3]``
        (``None`` when ``need_grad`` is false).
    """

    with torch.enable_grad():
        selected = coords.index_select(-2, index.reshape(-1)).reshape(*coords.shape[:-2], 4, index.shape[1], 3)
        if need_grad:
            selected = selected.detach().requires_grad_(True)
        p0, p1, p2, p3 = selected.unbind(dim=-3)
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
        v = b0 - (torch.linalg.vecdot(b0, b1_norm).unsqueeze(-1) * b1_norm)
        w = b2 - (torch.linalg.vecdot(b2, b1_norm).unsqueeze(-1) * b1_norm)
        x = torch.linalg.vecdot(v, w)
        y = torch.linalg.vecdot(torch.linalg.cross(b1_norm, v, dim=-1), w)
        angle = torch.atan2(y, x)
        if not need_grad:
            return angle, None
        grad_selected = torch.autograd.grad(angle.sum(), selected, create_graph=False)[0]
    return angle.detach(), grad_selected.detach()


def _abs_dihedral_value_and_grad(coords: Tensor, index: Tensor, need_grad: bool) -> tuple[Tensor, Tensor | None]:
    r"""
    Compute absolute dihedral angles and their sign-corrected gradients.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.
        index: Atom-quadruple index ``[4, D]``.
        need_grad: Whether to also return the per-atom gradients.

    Returns:
        ``(abs_angle, grad)`` with ``abs_angle`` shape ``[..., D]`` and ``grad`` shape ``[..., 4, D, 3]``
        (``None`` when ``need_grad`` is false).
    """

    value, grad = _dihedral_value_and_grad(coords, index, need_grad)
    if not need_grad:
        return torch.abs(value), None
    sign = torch.where(value < 0, -1.0, 1.0)
    return torch.abs(value), grad * sign[..., None, :, None]


def _flat_bottom_linear(
    value: Tensor,
    stiffness: Tensor,
    lower: Tensor | None,
    upper: Tensor | None,
) -> tuple[Tensor, Tensor]:
    r"""
    Flat-bottom linear penalty: zero inside ``[lower, upper]``, linear outside.

    Args:
        value: The scored quantity ``[...]`` (distance/angle/dihedral).
        stiffness: Per-element penalty slope ``[...]``.
        lower: Lower bound, or ``None`` to omit the lower penalty.
        upper: Upper bound, or ``None`` to omit the upper penalty.

    Returns:
        ``(energy, d_energy)`` where ``d_energy`` is the derivative with respect to ``value``.
    """

    energy = torch.zeros_like(value)
    grad = torch.zeros_like(value)
    if lower is not None:
        low = lower - value
        mask = low > 0
        energy = energy + torch.where(mask, stiffness * low, 0.0)
        grad = grad - torch.where(mask, stiffness, 0.0)
    if upper is not None:
        high = value - upper
        mask = high > 0
        energy = energy + torch.where(mask, stiffness * high, 0.0)
        grad = grad + torch.where(mask, stiffness, 0.0)
    return energy, grad


def _flat_bottom_parabolic(
    value: Tensor,
    stiffness: Tensor,
    lower: Tensor | None,
    upper: Tensor | None,
) -> tuple[Tensor, Tensor]:
    r"""
    Flat-bottom parabolic penalty: zero inside ``[lower, upper]``, quadratic outside.

    Args:
        value: The scored quantity ``[...]`` (typically a distance).
        stiffness: Per-element penalty stiffness ``[...]``.
        lower: Lower bound, or ``None`` to omit the lower penalty.
        upper: Upper bound, or ``None`` to omit the upper penalty.

    Returns:
        ``(energy, d_energy)`` where ``d_energy`` is the derivative with respect to ``value``.
    """

    energy = torch.zeros_like(value)
    grad = torch.zeros_like(value)
    if lower is not None:
        low = lower - value
        mask = low > 0
        energy = energy + torch.where(mask, 0.5 * stiffness * low**2, 0.0)
        grad = grad - torch.where(mask, stiffness * low, 0.0)
    if upper is not None:
        high = value - upper
        mask = high > 0
        energy = energy + torch.where(mask, 0.5 * stiffness * high**2, 0.0)
        grad = grad + torch.where(mask, stiffness * high, 0.0)
    return energy, grad


def _aggregate_atom_gradients(coords: Tensor, index: Tensor, value_grad: Tensor | None, d_energy: Tensor) -> Tensor:
    r"""
    Scatter per-constraint gradients onto the atoms they involve.

    Applies the chain rule (``d_energy`` times the per-endpoint value gradient) and scatter-adds the
    result into a full per-atom gradient tensor.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]`` (used for shape/device).
        index: Constraint atom index ``[W, C]`` (W endpoints per constraint).
        value_grad: Per-endpoint value gradients ``[..., W, C, 3]``.
        d_energy: Energy derivative with respect to each constraint value ``[..., C]``.

    Returns:
        Per-atom energy gradient ``[..., N_atom, 3]``.

    Raises:
        RuntimeError: If ``value_grad`` is ``None``.
    """

    if value_grad is None:
        raise RuntimeError("internal TFG error: value_grad is required")
    combined = d_energy.unsqueeze(-2).unsqueeze(-1) * value_grad
    combined = combined.flatten(start_dim=-3, end_dim=-2)
    flat_index = index.flatten()
    batch_shape = coords.shape[:-2]
    target = flat_index.view(*([1] * len(batch_shape)), -1, 1).expand(*batch_shape, -1, 3)
    output = torch.zeros_like(coords)
    output.scatter_add_(-2, target, combined)
    return output


def _solve_constraint_projection(
    coords: Tensor,
    index: Tensor,
    value: Tensor,
    value_grad: Tensor | None,
    mask: Tensor,
) -> Tensor:
    r"""
    Solve the least-norm coordinate displacement that zeros the active constraint violations.

    Builds the constraint Jacobian for the active constraints and solves the normal equations
    (Tikhonov-regularized) per batch element, returning the resulting atom displacement.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.
        index: Constraint atom index ``[W, C]``.
        value: Per-constraint violation ``[..., C]``.
        value_grad: Per-endpoint value gradients ``[..., W, C, 3]``.
        mask: Boolean mask ``[..., C]`` selecting the active (violated) constraints.

    Returns:
        Least-norm coordinate displacement ``[..., N_atom, 3]``.

    Raises:
        RuntimeError: If ``value_grad`` is ``None``.
    """

    if value_grad is None:
        raise RuntimeError("internal TFG error: projection requires gradients")
    batch_shape = coords.shape[:-2]
    num_atoms = coords.shape[-2]
    width = index.shape[0]
    num_constraints = index.shape[1]
    flat_batch = math.prod(batch_shape) if batch_shape else 1
    value_b = value.reshape(flat_batch, num_constraints)
    mask_b = mask.reshape(flat_batch, num_constraints)
    grad_b = value_grad.reshape(flat_batch, width, num_constraints, 3)
    delta = torch.zeros(flat_batch, num_atoms, 3, device=coords.device, dtype=coords.dtype)
    for batch_index in range(flat_batch):
        active = mask_b[batch_index]
        active_count = int(active.sum().item())
        if active_count == 0:
            continue
        active_index = index[:, active]
        active_value = value_b[batch_index, active]
        active_grad = grad_b[batch_index, :, active]
        jacobian = torch.zeros(active_count, num_atoms, 3, device=coords.device, dtype=coords.dtype)
        rows = torch.arange(active_count, device=coords.device)
        for atom_axis in range(width):
            jacobian[rows, active_index[atom_axis]] += active_grad[atom_axis]
        jacobian = jacobian.reshape(active_count, num_atoms * 3)
        work_dtype = torch.float32 if coords.dtype in (torch.float16, torch.bfloat16) else coords.dtype
        lhs = jacobian.to(work_dtype) @ jacobian.to(work_dtype).T
        lhs = lhs + 1e-8 * torch.eye(active_count, device=coords.device, dtype=work_dtype)
        solution = torch.linalg.solve(lhs, active_value.to(work_dtype))
        delta[batch_index] = (-(jacobian.to(work_dtype).T @ solution)).to(coords.dtype).reshape(num_atoms, 3)
    return delta.reshape(*coords.shape)


def _sum_energy(energy: Tensor) -> Tensor:
    r"""
    Reduce per-constraint energy to per-batch energy by summing the trailing axis.

    Args:
        energy: Per-constraint energy ``[..., C]`` or a scalar.

    Returns:
        Per-batch energy ``[...]`` (the input unchanged when it is already scalar).
    """

    if energy.ndim == 0:
        return energy
    return energy.sum(dim=-1)


def _zeros_energy(coords: Tensor) -> Tensor:
    r"""
    Return a zero per-batch energy tensor matching ``coords``.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.

    Returns:
        Zero energy ``[...]`` on the coords' device and dtype.
    """

    return torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)


def _zeros_energy_and_grad(coords: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Return matching zero energy and zero coordinate gradient tensors.

    Args:
        coords: Atom coordinates ``[..., N_atom, 3]``.

    Returns:
        ``(zeros_energy, zeros_grad)`` with shapes ``[...]`` and ``[..., N_atom, 3]``.
    """

    return _zeros_energy(coords), torch.zeros_like(coords)


def _vdw_radii(device: torch.device) -> Tensor:
    r"""
    Return the padded van-der-Waals radius lookup table indexed by element number.

    Args:
        device: Device on which to allocate the table.

    Returns:
        Radius table ``[128]`` with zeros beyond the tabulated elements.
    """

    radii = torch.zeros(128, dtype=torch.float32, device=device)
    radii[: len(_VDW_RADII)] = torch.tensor(_VDW_RADII, dtype=torch.float32, device=device)
    return radii


def _empty_index(width: int, device: torch.device) -> Tensor:
    r"""
    Return an empty ``[width, 0]`` long index tensor on the given device.

    Args:
        width: Number of endpoints per (absent) constraint.
        device: Target device.

    Returns:
        Empty index tensor ``[width, 0]``.
    """

    return torch.empty((width, 0), dtype=torch.long, device=device)


def _move_tensor(
    tensor: Tensor,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    non_blocking: bool,
    copy: bool,
) -> Tensor:
    r"""
    Move a tensor to a device/dtype, casting only floating-point tensors.

    Args:
        tensor: The tensor to move.
        device: Target device, or ``None`` to leave unchanged.
        dtype: Target floating dtype; ignored for integer tensors.
        non_blocking: Whether to request an asynchronous copy.
        copy: Whether to force a copy.

    Returns:
        The moved tensor.
    """

    cast_dtype = dtype if tensor.is_floating_point() else None
    return tensor.to(device=device, dtype=cast_dtype, non_blocking=non_blocking, copy=copy)


_GUIDANCE_EXPECTED_NDIMS = {
    "asym_id": 1,
    "atom_to_token_idx": 1,
    "ref_element": 2,
    "interchain_bond_index": 2,
    "symmetric_chain_index": 2,
    "pairwise_distance_index": 2,
    "pairwise_distance_upper_bound": 1,
    "pairwise_distance_lower_bound": 1,
    "pairwise_distance_is_bond": 1,
    "pairwise_distance_is_angle": 1,
    "experimental_torsion_index": 2,
    "experimental_torsion_force_constant": 2,
    "experimental_torsion_sign": 2,
    "linear_triple_bond_index": 2,
    "chiral_index": 2,
    "chiral_orientation": 1,
    "stereo_bond_index": 2,
    "stereo_bond_orientation": 1,
    "planar_improper_index": 2,
    "planar_improper_is_carbonyl": 1,
}


_GUIDANCE_INDEX_WIDTHS = {
    "interchain_bond_index": 2,
    "symmetric_chain_index": 2,
    "pairwise_distance_index": 2,
    "experimental_torsion_index": 4,
    "linear_triple_bond_index": 3,
    "chiral_index": 4,
    "stereo_bond_index": 4,
    "planar_improper_index": 4,
}


def _guidance_features_from_batch(
    batch: ProtenixInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    r"""
    Assemble the full guidance feature dictionary for a batch on the target device/dtype.

    Starts from the defaults derived from the batch, overlays any caller-provided guidance features,
    then normalizes each tensor's batch dimension, dtype, and empty-index width.

    Args:
        batch: The model input carrying atom/chain fields and optional guidance features.
        device: Target device for the returned tensors.
        dtype: Target floating dtype for float features.

    Returns:
        Mapping from official Protenix feature name to a prepared tensor.
    """

    features = _default_guidance_features_from_batch(batch)
    if batch.guidance_features is not None:
        if isinstance(batch.guidance_features, ProtenixGuidanceFeatures):
            provided = batch.guidance_features.as_dict()
        else:
            provided = dict(batch.guidance_features)
        features.update(provided)
    return {key: _prepare_guidance_tensor(key, value, device=device, dtype=dtype) for key, value in features.items()}


def _default_guidance_features_from_batch(batch: ProtenixInput) -> dict[str, Tensor]:
    r"""
    Derive the minimal default guidance features from a batch's structural fields.

    Provides ``asym_id``, ``atom_to_token_idx``, and a ``ref_element`` one-hot (from the atom feature
    slice when present), plus empty index/bound tensors for the geometry potentials.

    Args:
        batch: The model input carrying atom/chain/atom-feature fields.

    Returns:
        Mapping from official Protenix feature name to its default tensor.
    """

    atom_to_token = _strip_guidance_batch(batch.atom_to_token, "atom_to_token_idx", 1).to(dtype=torch.long)
    chain_index = _strip_guidance_batch(batch.chain_index, "asym_id", 1).to(dtype=torch.long)
    atom_features = _strip_guidance_batch(batch.atom_features, "atom_features", 2)
    num_atoms = atom_to_token.shape[-1]
    ref_element = torch.zeros(num_atoms, 128, dtype=atom_features.dtype, device=atom_features.device)
    if atom_features.shape[-1] >= 129:
        ref_element = atom_features[..., 1:129]

    empty = {
        key: torch.empty(width, 0, dtype=torch.long, device=atom_to_token.device)
        for key, width in _GUIDANCE_INDEX_WIDTHS.items()
    }
    return {
        "asym_id": chain_index,
        "atom_to_token_idx": atom_to_token,
        "ref_element": ref_element,
        **empty,
        "pairwise_distance_upper_bound": torch.empty(0, dtype=ref_element.dtype, device=ref_element.device),
        "pairwise_distance_lower_bound": torch.empty(0, dtype=ref_element.dtype, device=ref_element.device),
        "pairwise_distance_is_bond": torch.empty(0, dtype=torch.long, device=atom_to_token.device),
        "pairwise_distance_is_angle": torch.empty(0, dtype=torch.long, device=atom_to_token.device),
        "experimental_torsion_force_constant": torch.empty(0, 6, dtype=ref_element.dtype, device=ref_element.device),
        "experimental_torsion_sign": torch.empty(0, 6, dtype=ref_element.dtype, device=ref_element.device),
        "chiral_orientation": torch.empty(0, dtype=ref_element.dtype, device=ref_element.device),
        "stereo_bond_orientation": torch.empty(0, dtype=ref_element.dtype, device=ref_element.device),
        "planar_improper_is_carbonyl": torch.empty(0, dtype=ref_element.dtype, device=ref_element.device),
    }


def _prepare_guidance_tensor(key: str, value: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    r"""
    Normalize a single guidance feature tensor's batch dim, device, dtype, and empty width.

    Args:
        key: Official Protenix feature name (drives expected ndim and index width).
        value: The raw feature tensor.
        device: Target device.
        dtype: Target floating dtype; integer/index features keep their dtype.

    Returns:
        The prepared tensor, with empty index features reshaped to their canonical ``[width, 0]``.
    """

    expected_ndim = _GUIDANCE_EXPECTED_NDIMS.get(key)
    if expected_ndim is not None:
        value = _strip_guidance_batch(value, key, expected_ndim)
    cast_dtype = None if _guidance_tensor_is_integral(key, value) else dtype
    value = value.to(device=device, dtype=cast_dtype)
    width = _GUIDANCE_INDEX_WIDTHS.get(key)
    if width is not None and value.numel() == 0:
        return torch.empty(width, 0, dtype=torch.long, device=device)
    return value


def _strip_guidance_batch(value: Tensor, name: str, expected_ndim: int) -> Tensor:
    r"""
    Squeeze leading singleton batch dimensions down to the expected rank.

    Args:
        value: The feature tensor.
        name: Feature name, used in error messages.
        expected_ndim: The required number of dimensions after stripping.

    Returns:
        The tensor reduced to ``expected_ndim`` dimensions.

    Raises:
        ValueError: If a non-singleton batch dimension is present, or the rank does not match.
    """

    while value.ndim > expected_ndim:
        if value.shape[0] != 1:
            raise ValueError(f"TFG feature {name!r} only supports singleton batch dimensions, got {tuple(value.shape)}")
        value = value.squeeze(0)
    if value.ndim != expected_ndim:
        raise ValueError(f"TFG feature {name!r} must have {expected_ndim} dimensions, got {tuple(value.shape)}")
    return value


def _guidance_tensor_is_integral(key: str, value: Tensor) -> bool:
    r"""
    Return whether a guidance feature should keep an integer dtype (no float cast).

    Index tensors, known integer-valued keys, and already-integral tensors are treated as integral.

    Args:
        key: Official Protenix feature name.
        value: The feature tensor.

    Returns:
        ``True`` if the tensor must not be cast to the float compute dtype.
    """

    return (
        key.endswith("_index")
        or key
        in {
            "asym_id",
            "atom_to_token_idx",
            "pairwise_distance_is_bond",
            "pairwise_distance_is_angle",
        }
        or not value.is_floating_point()
    )
