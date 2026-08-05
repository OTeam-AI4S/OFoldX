from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from ofoldx.models.protenix.model import (
    ProtenixModel,
    ProtenixOutput,
    ProtenixStructureConditionModel,
    ProtenixStructureTarget,
    ProtenixTrainingOutput,
    _draw_keep_mask,
    _float32_coordinate_denoise_context,
)
from ofoldx.modules.atom import LocalAtomGeometry, LocalAtomWindowInfo
from ofoldx.modules.diffusion import CoordinateDenoiseContext, CoordinateDenoisingOutput
from ofoldx.modules.diffusion.loss import edm_denoising_loss
from ofoldx.modules.embedding import RelativePositionBuckets


class _DenoiseContext:
    def __init__(self, batch_size: int, atom_count: int) -> None:
        self.batch_shape = torch.Size([batch_size])
        self.num_atoms = atom_count

    def to_denoising_input(self, *, noisy_coords: Tensor, noise_level: Tensor) -> Any:
        return SimpleNamespace(noisy_coords=noisy_coords, noise_level=noise_level)


class _CountingDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))
        self.preconditioner = SimpleNamespace(sigma_data=1.0)
        self.calls: list[int] = []
        self.input_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
        self.autocast_enabled: list[bool] = []

    def forward(self, inputs: Any) -> Tensor:
        self.calls.append(inputs.noisy_coords.shape[-3])
        self.input_dtypes.append((inputs.noisy_coords.dtype, inputs.noise_level.dtype))
        self.autocast_enabled.append(torch.is_autocast_enabled(inputs.noisy_coords.device.type))
        return self.scale * inputs.noisy_coords + 0.1 * inputs.noise_level[..., None, None]


class _TinyProtenix(ProtenixModel):
    r"""Exercise the real training forward while replacing only expensive blocks."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._recycling_steps = 1
        self.coordinate_denoiser = _CountingDenoiser()
        self.trunk_calls = 0
        self.assemble_calls = 0

    def _run_trunk(self, input: Any, **kwargs: Any) -> Any:
        del kwargs
        self.trunk_calls += 1
        batch_size = input.atom_mask.shape[0]
        return SimpleNamespace(single=torch.zeros(batch_size, 1, 2))

    def _denoise_context(self, input: Any, representations: Any, **kwargs: Any) -> _DenoiseContext:
        del representations, kwargs
        return _DenoiseContext(input.atom_mask.shape[0], input.atom_mask.shape[1])

    def _assemble_output(self, input: Any, representations: Any, coordinates: Tensor) -> Any:
        del representations
        self.assemble_calls += 1
        batch_size, sample_count, atom_count = coordinates.shape[:3]
        token_count = 1
        return SimpleNamespace(
            coordinates=coordinates,
            distogram_logits=torch.zeros(batch_size, token_count, token_count, 1),
            plddt_logits=torch.zeros(batch_size, sample_count, atom_count, 1),
            pae_logits=torch.zeros(batch_size, sample_count, token_count, token_count, 1),
            pde_logits=torch.zeros(batch_size, sample_count, token_count, token_count, 1),
            resolved_logits=torch.zeros(batch_size, sample_count, atom_count, 1),
        )


def _training_pair(batch_size: int = 2, atom_count: int = 4) -> tuple[Any, Any]:
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )[:atom_count]
    coordinates = torch.stack([base + index for index in range(batch_size)])
    atom_mask = torch.ones(batch_size, atom_count, dtype=torch.bool)
    model_input = SimpleNamespace(
        reference_positions=torch.zeros_like(coordinates),
        atom_index=torch.arange(atom_count).expand(batch_size, atom_count),
        atom_mask=atom_mask,
    )
    target = SimpleNamespace(
        coordinates=coordinates,
        coordinate_mask=atom_mask.clone(),
    )
    return model_input, target


def test_default_single_sample_still_returns_full_protenix_output() -> None:
    model = _TinyProtenix()
    model_input, target = _training_pair()

    output = model(
        model_input,
        target,
        coordinate_augmentation=False,
        generator=torch.Generator().manual_seed(11),
    )

    assert isinstance(output, ProtenixOutput)
    assert output.coordinates.shape == (2, 1, 4, 3)
    assert output.augmented_target_coordinates.shape == (2, 1, 4, 3)
    assert output.noise_level.shape == (2, 1)
    assert model.trunk_calls == 1
    assert model.coordinate_denoiser.calls == [1]
    assert model.assemble_calls == 1


def test_diffusion_batch_chunks_only_denoiser_and_supports_backward() -> None:
    model = _TinyProtenix()
    model_input, target = _training_pair()

    output = model(
        model_input,
        target,
        num_samples=5,
        sample_chunk_size=2,
        compute_auxiliary_heads=False,
        generator=torch.Generator().manual_seed(19),
    )
    loss = edm_denoising_loss(output, torch.ones(2, 4, dtype=torch.bool), sigma_data=1.0)
    loss.backward()

    assert isinstance(output, CoordinateDenoisingOutput)
    assert output.coordinates.shape == (2, 5, 4, 3)
    assert output.augmented_target_coordinates.shape == (2, 5, 4, 3)
    assert output.noise_level.shape == (2, 5)
    assert torch.isfinite(loss)
    assert model.coordinate_denoiser.scale.grad is not None
    assert torch.isfinite(model.coordinate_denoiser.scale.grad)
    assert model.trunk_calls == 1
    assert model.coordinate_denoiser.calls == [2, 2, 1]
    assert model.assemble_calls == 0


def test_seed_reproduces_batch_but_samples_are_independent() -> None:
    model = _TinyProtenix()
    model_input, target = _training_pair()

    def run() -> CoordinateDenoisingOutput:
        return model(  # type: ignore[return-value]
            model_input,
            target,
            num_samples=3,
            sample_chunk_size=2,
            compute_auxiliary_heads=False,
            generator=torch.Generator().manual_seed(23),
        )

    first = run()
    second = run()

    torch.testing.assert_close(first.coordinates, second.coordinates)
    torch.testing.assert_close(first.augmented_target_coordinates, second.augmented_target_coordinates)
    torch.testing.assert_close(first.noise_level, second.noise_level)
    assert not torch.equal(first.coordinates[:, 0], first.coordinates[:, 1])
    assert not torch.equal(first.noise_level[:, 0], first.noise_level[:, 1])


def _loss_and_scale_gradient(
    sample_coordinates: Tensor,
    target_coordinates: Tensor,
    noise_level: Tensor,
) -> tuple[Tensor, Tensor]:
    scale = torch.tensor(0.8, requires_grad=True)
    prediction = SimpleNamespace(
        coordinates=scale * sample_coordinates,
        augmented_target_coordinates=target_coordinates,
        noise_level=noise_level,
    )
    loss = edm_denoising_loss(
        prediction,
        torch.ones(sample_coordinates.shape[0], sample_coordinates.shape[-2], dtype=torch.bool),
        sigma_data=1.0,
    )
    (gradient,) = torch.autograd.grad(loss, scale)
    return loss.detach(), gradient.detach()


def test_repeating_one_diffusion_sample_does_not_scale_loss_or_gradient() -> None:
    base = _training_pair(batch_size=1)[1].coordinates[:, None]
    distorted = base + torch.tensor([[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.3]]]])
    noise = torch.tensor([[0.7]])

    single = _loss_and_scale_gradient(distorted, base, noise)
    repeated = _loss_and_scale_gradient(
        distorted.expand(-1, 4, -1, -1),
        base.expand(-1, 4, -1, -1),
        noise.expand(-1, 4),
    )

    torch.testing.assert_close(repeated[0], single[0])
    torch.testing.assert_close(repeated[1], single[1])


def test_joint_diffusion_loss_and_gradient_equal_mean_of_individual_samples() -> None:
    base = _training_pair(batch_size=1)[1].coordinates[:, None]
    first = base + torch.tensor([[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.3]]]])
    second = base + torch.tensor([[[[0.1, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, -0.2], [-0.1, 0.0, 0.0]]]])
    first_result = _loss_and_scale_gradient(first, base, torch.tensor([[0.5]]))
    second_result = _loss_and_scale_gradient(second, base, torch.tensor([[1.2]]))

    joint_result = _loss_and_scale_gradient(
        torch.cat([first, second], dim=1),
        base.expand(-1, 2, -1, -1),
        torch.tensor([[0.5, 1.2]]),
    )

    torch.testing.assert_close(joint_result[0], (first_result[0] + second_result[0]) / 2)
    torch.testing.assert_close(joint_result[1], (first_result[1] + second_result[1]) / 2)


class _ConfidenceSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.pred_requires_grad: bool | None = None
        self.embedding_requires_grad: tuple[bool, bool, bool] | None = None
        self.pair_values: Tensor | None = None
        self.embedding_dtypes: tuple[torch.dtype, torch.dtype, torch.dtype] | None = None
        self.pred_dtype: torch.dtype | None = None
        self.autocast_enabled: bool | None = None

    def forward(
        self,
        single: Tensor,
        *,
        pair: Tensor,
        token_embedding: Tensor,
        pred_coords: Tensor,
        token_to_rep_atom: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del token_to_rep_atom, atom_to_token, atom_token_index
        self.pred_requires_grad = pred_coords.requires_grad
        self.embedding_requires_grad = (
            single.requires_grad,
            pair.requires_grad,
            token_embedding.requires_grad,
        )
        self.pair_values = pair.detach().clone()
        self.embedding_dtypes = (single.dtype, pair.dtype, token_embedding.dtype)
        self.pred_dtype = pred_coords.dtype
        self.autocast_enabled = torch.is_autocast_enabled(single.device.type)
        batch_size, sample_count, atom_count = pred_coords.shape[:3]
        token_count = single.shape[-2]
        value = self.weight * torch.ones((), device=pred_coords.device)
        return (
            value.expand(batch_size, sample_count, atom_count, 2),
            value.expand(batch_size, sample_count, token_count, token_count, 3),
            value.expand(batch_size, sample_count, token_count, token_count, 4),
            value.expand(batch_size, sample_count, atom_count, 2),
        )


class _DistogramSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dtypes: list[torch.dtype] = []
        self.autocast_enabled: list[bool] = []

    def forward(self, pair: Tensor) -> Tensor:
        self.input_dtypes.append(pair.dtype)
        self.autocast_enabled.append(torch.is_autocast_enabled(pair.device.type))
        return pair.sum(dim=-1, keepdim=True)


class _BaselineTiny(ProtenixModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._recycling_steps = 4
        self._num_sampling_steps = 7
        self.coordinate_denoiser = _CountingDenoiser()
        self.coordinate_denoiser.add_module("mode_probe", nn.Dropout())
        self.coordinate_denoiser.mode_probe.eval()
        self.confidence_head = _ConfidenceSpy()
        self.distogram_head = _DistogramSpy()
        self.input_embeddings = nn.Sequential(nn.Dropout(), nn.Sequential(nn.Dropout()))
        self.template_embedder = nn.Sequential(nn.Dropout())
        self.msa_trunk = nn.Sequential(nn.Dropout(), nn.Dropout())
        self.pairformer_stack = nn.Sequential(nn.Dropout())
        self.trunk_scale = nn.Parameter(torch.tensor(0.75))
        self.trunk_output_dtype = torch.float32
        self.trunk_calls = 0
        self.trunk_kwargs: dict[str, Any] | None = None
        self.trunk_modes: list[tuple[bool, ...]] = []
        self.trunk_grad_enabled: list[bool] = []
        self.raise_in_trunk = False
        self.msa_draws = 0
        self.rollout_steps: list[int] = []
        self.rollout_num_samples: list[int] = []
        self.rollout_grad_enabled: list[bool] = []
        self.rollout_modes: list[tuple[bool, bool]] = []
        self.rollout_dtypes: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
        self.rollout_autocast_enabled: list[bool] = []
        self.raise_in_rollout = False
        self.conditioning_keep: Tensor | None = None

    def _sample_msa_indices(self, num_rows: int, device: torch.device, **kwargs: Any) -> Tensor:
        del kwargs
        index = self.msa_draws % num_rows
        self.msa_draws += 1
        return torch.tensor([index], device=device)

    def _run_trunk(self, input: Any, **kwargs: Any) -> Any:
        self.trunk_calls += 1
        self.trunk_kwargs = kwargs
        self.trunk_modes.append(_trunk_mode_snapshot(self))
        self.trunk_grad_enabled.append(torch.is_grad_enabled())
        if self.raise_in_trunk:
            raise RuntimeError("trunk failure")
        batch_size = input.atom_mask.shape[0]
        value = self.trunk_scale * torch.ones(batch_size, 1, 2, device=input.atom_mask.device)
        value = value.to(self.trunk_output_dtype)
        return SimpleNamespace(
            single=value,
            pair=value.unsqueeze(-2),
            token_embedding=value,
            relative_features=None,
        )

    def _predict_coords(self, input: Any, *, representations: Any, num_sampling_steps: int, **kwargs: Any) -> Tensor:
        num_samples = kwargs["num_samples"]
        self.rollout_steps.append(num_sampling_steps)
        self.rollout_num_samples.append(num_samples)
        self.rollout_grad_enabled.append(torch.is_grad_enabled())
        self.rollout_modes.append(
            (
                self.coordinate_denoiser.training,
                self.coordinate_denoiser.mode_probe.training,
            )
        )
        self.rollout_dtypes.append(
            (
                representations.single.dtype,
                representations.pair.dtype,
                representations.token_embedding.dtype,
            )
        )
        self.rollout_autocast_enabled.append(torch.is_autocast_enabled(input.reference_positions.device.type))
        if self.raise_in_rollout:
            raise RuntimeError("rollout failure")
        assert not representations.single.requires_grad
        assert not representations.pair.requires_grad
        assert not representations.token_embedding.requires_grad
        return representations.single.new_zeros(
            input.reference_positions.shape[0],
            num_samples,
            input.reference_positions.shape[-2],
            3,
        )

    def _denoise_context(self, input: Any, representations: Any, **kwargs: Any) -> _DenoiseContext:
        del representations
        if "use_conditioning" in kwargs:
            self.conditioning_keep = kwargs["use_conditioning"].detach().clone()
        return _DenoiseContext(input.atom_mask.shape[0], input.atom_mask.shape[1])


class _IdentityMsa(nn.Module):
    def forward(self, msa: Tensor, pair: Tensor, token_embedding: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        del msa, token_embedding, kwargs
        return pair.new_zeros(()), pair


class _IdentityPairformer(nn.Module):
    def forward(self, single: Tensor, pair: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        del kwargs
        return single, pair


class _IdentityStructureEmbedding(nn.Module):
    def forward(self, pair: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        del positions, mask
        return torch.zeros_like(pair)


class _RecycleTiny(ProtenixModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.pair_recycle_norm = nn.Identity()
        self.pair_recycle_projection = nn.Identity()
        self.template_embedder = None
        self.msa_trunk = _IdentityMsa()
        self.single_recycle_norm = nn.Identity()
        self.single_recycle_projection = nn.Identity()
        self.pairformer_stack = _IdentityPairformer()

    @staticmethod
    def _msa_features(input: Any, indices: Tensor | None) -> Tensor:
        del input, indices
        return torch.zeros(())


class _StructureBaselineTiny(_BaselineTiny, ProtenixStructureConditionModel):
    def __init__(self) -> None:
        super().__init__()
        self.noisy_structure_embedder = nn.Sequential(nn.Dropout(), nn.Sequential(nn.Dropout()))


def _trunk_mode_snapshot(model: ProtenixModel) -> tuple[bool, ...]:
    roots = model._confidence_only_trunk_modules()
    return tuple(module.training for root in roots for module in root.modules())


def _baseline_training_pair(
    batch_size: int = 2,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Any, Any]:
    model_input, target = _training_pair(batch_size=batch_size)
    atom_count = target.coordinates.shape[-2]
    model_input.reference_positions = model_input.reference_positions.to(device)
    model_input.atom_index = model_input.atom_index.to(device)
    model_input.atom_mask = model_input.atom_mask.to(device)
    target.coordinates = target.coordinates.to(device)
    target.coordinate_mask = target.coordinate_mask.to(device)
    model_input.msa_tokens = torch.zeros(batch_size, 8, 1, dtype=torch.long, device=device)
    model_input.token_to_rep_atom = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    model_input.atom_to_token = torch.zeros(batch_size, atom_count, dtype=torch.long, device=device)
    model_input.atom_token_index = torch.arange(atom_count, device=device).expand(batch_size, atom_count)
    model_input.tfg_guidance = None
    model_input.tfg_features = None
    target.coordinate_mask[:, -1] = False
    target.molecule_id = torch.zeros(batch_size, atom_count, dtype=torch.long, device=device)
    target.molecule_equivalence_index = torch.zeros(batch_size, atom_count, dtype=torch.long, device=device)
    target.molecule_atom_index = torch.arange(atom_count, device=device).expand(batch_size, atom_count)
    target.representative_atom_mask = torch.ones(batch_size, atom_count, dtype=torch.bool, device=device)
    target.permutation_pool_mask = torch.ones(batch_size, atom_count, dtype=torch.bool, device=device)
    target.atom_molecule_type = torch.zeros(batch_size, atom_count, dtype=torch.long, device=device)
    target.symmetry_atom_index = torch.zeros(batch_size, 0, 1, dtype=torch.long, device=device)
    target.symmetry_permutation = torch.zeros(batch_size, 0, 1, 1, dtype=torch.long, device=device)
    return model_input, target


def test_structure_target_legacy_two_field_constructor_remains_valid() -> None:
    coordinates = torch.zeros(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)

    target = ProtenixStructureTarget(
        coordinates=coordinates,
        coordinate_mask=mask,
        batch_size=[1],
    )

    assert target.molecule_id is None
    assert target.resolution_is_known is None


def test_baseline_recipe_runs_one_trunk_and_one_detached_twenty_step_rollout() -> None:
    model = _BaselineTiny()
    model.train()
    model.coordinate_denoiser.mode_probe.eval()
    model_input, target = _baseline_training_pair()

    output = model(
        model_input,
        target,
        training_recipe="protenix_baseline_v1",
        num_samples=3,
        sample_chunk_size=2,
        current_step=17,
        coordinate_augmentation=False,
        condition_embedding_drop_rate=1.0,
        confidence_embedding_drop_rate=1.0,
        generator=torch.Generator().manual_seed(31),
    )

    assert isinstance(output, ProtenixTrainingOutput)
    assert model.trunk_calls == 1
    assert model.trunk_kwargs is not None
    actual_recycles = model.trunk_kwargs["recycling_steps"]
    assert len(model.trunk_kwargs["msa_indices"]) == actual_recycles
    assert model.msa_draws == actual_recycles
    assert model.rollout_steps == [20]
    assert model.rollout_grad_enabled == [False]
    assert model.rollout_modes == [(False, False)]
    assert model.coordinate_denoiser.training
    assert not model.coordinate_denoiser.mode_probe.training
    assert output.mini_rollout_coordinates.shape == (2, 1, 4, 3)
    assert not output.mini_rollout_coordinates.requires_grad
    assert output.coordinates is not None and output.coordinates.shape == (2, 3, 4, 3)
    assert output.augmented_target_coordinates is not None
    assert output.augmented_target_mask is not None
    assert output.noise_level is not None and output.noise_level.dtype == torch.float32
    assert output.matched_target_coordinates.dtype == torch.float32
    assert not output.augmented_target_mask[:, :, -1].any()
    assert not output.augmented_target_coordinates[:, :, -1].any()
    assert model.conditioning_keep is not None and not model.conditioning_keep.any()
    assert model.confidence_head.pair_values is not None
    assert not model.confidence_head.pair_values.any()
    assert output.plddt_logits.shape[1] == 1
    assert output.pae_logits.shape[1] == 1
    assert output.pde_logits.shape[1] == 1
    assert output.resolved_logits.shape[1] == 1
    assert model.confidence_head.pred_requires_grad is False
    assert model.confidence_head.embedding_requires_grad == (False, False, False)
    assert output.distogram_logits is not None
    (output.coordinates.sum() + output.distogram_logits.sum() + output.plddt_logits.sum()).backward()
    assert model.coordinate_denoiser.scale.grad is not None
    assert model.confidence_head.weight.grad is not None
    assert model.trunk_scale.grad is not None


def test_baseline_mini_rollout_restores_exact_denoiser_modes_when_sampling_raises() -> None:
    model = _BaselineTiny()
    model.train()
    model.coordinate_denoiser.mode_probe.eval()
    original_modes = tuple((module, module.training) for module in model.coordinate_denoiser.modules())
    model.raise_in_rollout = True
    model_input, target = _baseline_training_pair()

    with pytest.raises(RuntimeError, match="rollout failure"):
        model(
            model_input,
            target,
            training_recipe="protenix_baseline_v1",
            recycling_steps=1,
        )

    assert model.rollout_modes == [(False, False)]
    assert tuple((module, module.training) for module in model.coordinate_denoiser.modules()) == original_modes


def test_baseline_confidence_only_is_explicit_and_does_not_backpropagate_to_trunk() -> None:
    model = _BaselineTiny()
    model_input, target = _baseline_training_pair()

    output = model(
        model_input,
        target,
        training_recipe="protenix_baseline_v1",
        train_confidence_only=True,
        recycling_steps=2,
    )
    loss = output.plddt_logits.sum() + output.pae_logits.sum() + output.pde_logits.sum()
    loss.backward()

    assert output.coordinates is None
    assert output.augmented_target_coordinates is None
    assert output.augmented_target_mask is None
    assert output.noise_level is None
    assert output.distogram_logits is None
    assert model.confidence_head.weight.grad is not None
    assert model.trunk_scale.grad is None
    assert model.trunk_calls == 1
    assert model.msa_draws == 2
    assert model.trunk_grad_enabled == [False]


def test_baseline_confidence_only_temporarily_evals_and_exactly_restores_trunk_modes() -> None:
    model = _BaselineTiny()
    model.train()
    model.input_embeddings[1][0].eval()
    model.template_embedder.eval()
    model.msa_trunk[1].eval()
    original_modes = _trunk_mode_snapshot(model)
    model_input, target = _baseline_training_pair()

    model(
        model_input,
        target,
        training_recipe="protenix_baseline_v1",
        train_confidence_only=True,
        recycling_steps=1,
    )

    assert model.trunk_modes == [(False,) * len(original_modes)]
    assert _trunk_mode_snapshot(model) == original_modes


def test_baseline_confidence_only_restores_trunk_modes_when_trunk_raises() -> None:
    model = _BaselineTiny()
    model.train()
    model.input_embeddings[0].eval()
    model.msa_trunk[1].eval()
    original_modes = _trunk_mode_snapshot(model)
    model.raise_in_trunk = True
    model_input, target = _baseline_training_pair()

    with pytest.raises(RuntimeError, match="trunk failure"):
        model(
            model_input,
            target,
            training_recipe="protenix_baseline_v1",
            train_confidence_only=True,
            recycling_steps=1,
        )

    assert model.trunk_modes == [(False,) * len(original_modes)]
    assert _trunk_mode_snapshot(model) == original_modes


@pytest.mark.parametrize("raise_in_trunk", [False, True])
def test_structure_conditioned_confidence_only_restores_noisy_embedder_modes(
    raise_in_trunk: bool,
) -> None:
    model = _StructureBaselineTiny()
    model.train()
    model.noisy_structure_embedder[1][0].eval()
    original_modes = _trunk_mode_snapshot(model)
    model.raise_in_trunk = raise_in_trunk
    model_input, target = _baseline_training_pair()

    if raise_in_trunk:
        with pytest.raises(RuntimeError, match="trunk failure"):
            model(
                model_input,
                target,
                training_recipe="protenix_baseline_v1",
                train_confidence_only=True,
                recycling_steps=1,
            )
    else:
        model(
            model_input,
            target,
            training_recipe="protenix_baseline_v1",
            train_confidence_only=True,
            recycling_steps=1,
        )

    assert model.trunk_modes == [(False,) * len(original_modes)]
    assert model.trunk_grad_enabled == [False]
    assert _trunk_mode_snapshot(model) == original_modes


def test_baseline_normal_training_does_not_change_trunk_modes() -> None:
    model = _BaselineTiny()
    model.train()
    model.input_embeddings[0].eval()
    model.pairformer_stack.eval()
    original_modes = _trunk_mode_snapshot(model)
    model_input, target = _baseline_training_pair()

    model(
        model_input,
        target,
        training_recipe="protenix_baseline_v1",
        coordinate_augmentation=False,
        recycling_steps=1,
    )

    assert model.trunk_modes == [original_modes]
    assert model.trunk_grad_enabled == [True]
    assert _trunk_mode_snapshot(model) == original_modes


def test_baseline_dropout_is_one_generator_driven_call_level_decision() -> None:
    first = _draw_keep_mask(
        7,
        0.5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(101),
    )
    second = _draw_keep_mask(
        7,
        0.5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(101),
    )

    torch.testing.assert_close(first, second)
    assert torch.equal(first, first[:1].expand_as(first))


def test_baseline_recycle_seed_seventeen_selects_exact_depth_for_four_cycles() -> None:
    model = _BaselineTiny()

    actual = model._training_recycling_steps(
        None,
        current_step=17,
        generator=None,
    )

    assert actual == 4


def test_real_msa_sampler_replays_range_without_replacement_and_caps_depth() -> None:
    model = _BaselineTiny()

    first = ProtenixModel._sample_msa_indices(
        model,
        257,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(211),
    )
    replay = ProtenixModel._sample_msa_indices(
        model,
        257,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(211),
    )

    torch.testing.assert_close(first, replay)
    assert 1 <= first.numel() <= 257
    assert int(first.min()) >= 0
    assert int(first.max()) < 257
    assert torch.unique(first).numel() == first.numel()

    cycle_generator = torch.Generator().manual_seed(223)
    per_cycle = tuple(
        ProtenixModel._sample_msa_indices(
            model,
            64,
            torch.device("cpu"),
            generator=cycle_generator,
        )
        for _ in range(4)
    )
    assert len({tuple(indices.tolist()) for indices in per_cycle}) == 4

    cap_generator = torch.Generator().manual_seed(227)
    for _ in range(200):
        capped = ProtenixModel._sample_msa_indices(
            model,
            20_000,
            torch.device("cpu"),
            generator=cap_generator,
        )
        if capped.numel() == 16_384:
            break
    assert capped.numel() == 16_384
    assert torch.unique(capped).numel() == capped.numel()


def test_sample_omitted_evaluation_options_preserve_legacy_defaults() -> None:
    model = _BaselineTiny()
    model.trunk_output_dtype = torch.bfloat16
    model_input, _ = _baseline_training_pair()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model.sample(model_input)

    assert output.coordinates.shape == (2, 5, 4, 3)
    assert output.coordinates.dtype == torch.bfloat16
    assert model.rollout_num_samples == [5]
    assert model.rollout_steps == [7]
    assert model.rollout_dtypes == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]
    assert model.rollout_autocast_enabled == [True]
    assert model.trunk_kwargs is not None
    assert model.trunk_kwargs["recycling_steps"] == 4
    assert model.trunk_kwargs["mc_dropout"] is False
    assert model.trunk_kwargs["mc_dropout_rate"] == pytest.approx(0.4)
    assert model.confidence_head.embedding_dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert model.confidence_head.autocast_enabled is True
    assert model.distogram_head.input_dtypes == [torch.bfloat16]
    assert model.distogram_head.autocast_enabled == [True]


def test_baseline_sample_defaults_to_five_samples_two_hundred_steps_and_full_recycling() -> None:
    model = _BaselineTiny()
    model.trunk_output_dtype = torch.bfloat16
    model_input, _ = _baseline_training_pair()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model.sample(model_input, evaluation_recipe="protenix_baseline_v1")
        diffusion_loss = output.coordinates.square().mean()

    assert output.coordinates.shape == (2, 5, 4, 3)
    assert output.coordinates.dtype == torch.float32
    assert diffusion_loss.dtype == torch.float32
    assert model.rollout_num_samples == [5]
    assert model.rollout_steps == [200]
    assert model.rollout_dtypes == [(torch.float32, torch.float32, torch.float32)]
    assert model.rollout_autocast_enabled == [False]
    assert model.trunk_kwargs is not None
    assert model.trunk_kwargs["recycling_steps"] == 4
    assert len(model.trunk_kwargs["msa_indices"]) == 4
    assert model.trunk_kwargs["mc_dropout"] is False
    assert model.confidence_head.embedding_dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert model.confidence_head.pred_dtype == torch.float32
    assert model.confidence_head.autocast_enabled is True
    assert output.distogram_logits.dtype == torch.float32
    assert model.distogram_head.input_dtypes == [torch.float32]
    assert model.distogram_head.autocast_enabled == [False]


def test_baseline_sample_accepts_explicit_counts_and_caller_decided_mc_dropout() -> None:
    model = _BaselineTiny()
    model_input, _ = _baseline_training_pair()

    output = model.sample(
        model_input,
        evaluation_recipe="protenix_baseline_v1",
        num_samples=3,
        num_sampling_steps=11,
        recycling_steps=2,
        mc_dropout=True,
        mc_dropout_rate=0.25,
    )

    assert output.coordinates.shape == (2, 3, 4, 3)
    assert model.rollout_num_samples == [3]
    assert model.rollout_steps == [11]
    assert model.trunk_kwargs is not None
    assert model.trunk_kwargs["recycling_steps"] == 2
    assert len(model.trunk_kwargs["msa_indices"]) == 2
    assert model.trunk_kwargs["mc_dropout"] is True
    assert model.trunk_kwargs["mc_dropout_rate"] == pytest.approx(0.25)


def test_structure_conditioned_subclass_inherits_legacy_and_baseline_sample_contracts() -> None:
    model = _StructureBaselineTiny()
    model.trunk_output_dtype = torch.bfloat16
    model_input, _ = _baseline_training_pair()

    legacy = model.sample(
        model_input,
        num_samples=2,
        num_sampling_steps=3,
        recycling_steps=1,
    )
    baseline = model.sample(
        model_input,
        evaluation_recipe="protenix_baseline_v1",
        num_samples=3,
        num_sampling_steps=4,
        recycling_steps=2,
        mc_dropout=True,
        mc_dropout_rate=0.2,
    )

    assert legacy.coordinates.shape == (2, 2, 4, 3)
    assert legacy.coordinates.dtype == torch.bfloat16
    assert baseline.coordinates.shape == (2, 3, 4, 3)
    assert baseline.coordinates.dtype == torch.float32
    assert model.rollout_num_samples == [2, 3]
    assert model.rollout_steps == [3, 4]
    assert model.trunk_kwargs is not None
    assert model.trunk_kwargs["mc_dropout"] is True
    assert model.trunk_kwargs["mc_dropout_rate"] == pytest.approx(0.2)


def test_mc_dropout_is_forced_only_on_each_pair_recycling_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _RecycleTiny().eval()
    dropout_calls: list[tuple[float, bool]] = []

    def dropout(value: Tensor, *, p: float, training: bool) -> Tensor:
        dropout_calls.append((p, training))
        return value

    monkeypatch.setattr(torch.nn.functional, "dropout", dropout)
    original_modes = tuple((module, module.training) for module in model.modules())
    single_init = torch.zeros(1, 1, 2)
    pair_init = torch.zeros(1, 1, 1, 2)

    model._run_recycling(
        SimpleNamespace(),
        torch.zeros_like(single_init),
        single_init,
        pair_init,
        torch.ones(1, 1, 1, dtype=torch.bool),
        3,
        msa_indices=None,
        mc_dropout=True,
        mc_dropout_rate=0.25,
    )

    assert dropout_calls == [(0.25, True)] * 3
    assert tuple((module, module.training) for module in model.modules()) == original_modes

    dropout_calls.clear()
    model._run_recycling(
        SimpleNamespace(),
        torch.zeros_like(single_init),
        single_init,
        pair_init,
        torch.ones(1, 1, 1, dtype=torch.bool),
        2,
        msa_indices=None,
        mc_dropout=False,
        mc_dropout_rate=0.25,
    )
    assert dropout_calls == []

    model.noisy_structure_embedder = _IdentityStructureEmbedding()
    ProtenixStructureConditionModel._run_recycling(
        model,
        SimpleNamespace(
            structure_rep_positions=torch.zeros(1, 1, 3),
            structure_rep_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
        torch.zeros_like(single_init),
        single_init,
        pair_init,
        torch.ones(1, 1, 1, dtype=torch.bool),
        1,
        msa_indices=None,
        mc_dropout=True,
        mc_dropout_rate=0.3,
    )
    assert dropout_calls == [(0.3, True)]


def test_baseline_sample_mc_dropout_preserves_every_module_mode() -> None:
    model = _BaselineTiny()
    model.train()
    model.input_embeddings[0].eval()
    model.template_embedder.eval()
    model.msa_trunk[1].eval()
    model.coordinate_denoiser.mode_probe.eval()
    original_modes = tuple((module, module.training) for module in model.modules())
    model_input, _ = _baseline_training_pair()

    model.sample(
        model_input,
        evaluation_recipe="protenix_baseline_v1",
        recycling_steps=1,
        num_sampling_steps=1,
        mc_dropout=True,
        mc_dropout_rate=0.4,
    )

    assert tuple((module, module.training) for module in model.modules()) == original_modes


def test_baseline_fp32_boundaries_under_cpu_bfloat16_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _BaselineTiny()
    model.trunk_output_dtype = torch.bfloat16
    model_input, target = _baseline_training_pair()
    monkeypatch.setattr(
        "ofoldx.models.protenix.model.match_protenix_truth",
        lambda prediction, truth, inputs: (truth.coordinates, truth.coordinate_mask),
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(
            model_input,
            target,
            training_recipe="protenix_baseline_v1",
            coordinate_augmentation=False,
            recycling_steps=1,
        )

    assert output.coordinates is not None and output.coordinates.dtype == torch.float32
    assert output.distogram_logits is not None and output.distogram_logits.dtype == torch.float32
    assert model.rollout_dtypes == [(torch.float32, torch.float32, torch.float32)]
    assert model.rollout_autocast_enabled == [False]
    assert model.coordinate_denoiser.input_dtypes == [(torch.float32, torch.float32)]
    assert model.coordinate_denoiser.autocast_enabled == [False]
    assert model.distogram_head.input_dtypes == [torch.float32]
    assert model.distogram_head.autocast_enabled == [False]
    assert model.confidence_head.embedding_dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16)


def test_baseline_fp32_context_conversion_preserves_integer_and_boolean_metadata() -> None:
    batch_size, atom_count, token_count = 2, 4, 1
    relative_features = RelativePositionBuckets(
        residue_bucket=torch.zeros(batch_size, token_count, token_count, dtype=torch.long),
        token_bucket=torch.zeros(batch_size, token_count, token_count, dtype=torch.long),
        same_entity=torch.ones(batch_size, token_count, token_count, dtype=torch.bool),
        chain_bucket=torch.zeros(batch_size, token_count, token_count, dtype=torch.long),
        residue_size=3,
        chain_size=3,
        dtype=torch.bfloat16,
    )
    window_info = LocalAtomWindowInfo(
        attention_mask=torch.ones(batch_size, 1, atom_count, atom_count, dtype=torch.bool),
        key_index=torch.arange(atom_count).expand(batch_size, 1, atom_count),
    )
    context = CoordinateDenoiseContext(
        atom_features=torch.zeros(batch_size, atom_count, 2, dtype=torch.bfloat16),
        atom_to_token=torch.zeros(batch_size, atom_count, dtype=torch.long),
        reference_positions=torch.zeros(batch_size, atom_count, 3, dtype=torch.bfloat16),
        reference_charges=torch.zeros(batch_size, atom_count, 1, dtype=torch.bfloat16),
        reference_mask=torch.ones(batch_size, atom_count, dtype=torch.bool),
        atom_mask=torch.ones(batch_size, atom_count, dtype=torch.bool),
        token_embedding=torch.zeros(batch_size, token_count, 2, dtype=torch.bfloat16),
        single=torch.zeros(batch_size, token_count, 2, dtype=torch.bfloat16),
        conditioning_single=torch.zeros(batch_size, token_count, 2, dtype=torch.bfloat16),
        conditioning_pair=torch.zeros(batch_size, token_count, token_count, 2, dtype=torch.bfloat16),
        relative_features=relative_features,
        local_geometry=LocalAtomGeometry(
            distance_vectors=torch.zeros(batch_size, 1, atom_count, atom_count, 3, dtype=torch.bfloat16),
            distance_mask=torch.ones(batch_size, 1, atom_count, atom_count, 1, dtype=torch.bfloat16),
            window_info=window_info,
        ),
    )

    converted = _float32_coordinate_denoise_context(context)

    assert converted.dtype == torch.float32
    assert converted.atom_features.dtype == torch.float32
    assert converted.reference_positions.dtype == torch.float32
    assert converted.reference_charges.dtype == torch.float32
    assert converted.single.dtype == torch.float32
    assert converted.conditioning_single.dtype == torch.float32
    assert converted.conditioning_pair.dtype == torch.float32
    assert converted.local_geometry.distance_vectors.dtype == torch.float32
    assert converted.local_geometry.distance_mask.dtype == torch.float32
    assert converted.relative_features.dtype == torch.float32
    assert converted.atom_to_token.dtype == torch.long
    assert converted.atom_mask.dtype == torch.bool
    assert converted.relative_features.same_entity.dtype == torch.bool
    assert converted.local_geometry.window_info is window_info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real bf16 autocast validation")
def test_baseline_fp32_boundaries_under_cuda_bfloat16_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    model = _BaselineTiny().to(device)
    model.trunk_output_dtype = torch.bfloat16
    model_input, target = _baseline_training_pair(device=device)
    monkeypatch.setattr(
        "ofoldx.models.protenix.model.match_protenix_truth",
        lambda prediction, truth, inputs: (truth.coordinates, truth.coordinate_mask),
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            model_input,
            target,
            training_recipe="protenix_baseline_v1",
            coordinate_augmentation=False,
            recycling_steps=1,
            generator=torch.Generator(device=device).manual_seed(113),
        )

    assert output.coordinates is not None and output.coordinates.dtype == torch.float32
    assert output.distogram_logits is not None and output.distogram_logits.dtype == torch.float32
    assert model.rollout_dtypes == [(torch.float32, torch.float32, torch.float32)]
    assert model.rollout_autocast_enabled == [False]
    assert model.coordinate_denoiser.input_dtypes == [(torch.float32, torch.float32)]
    assert model.coordinate_denoiser.autocast_enabled == [False]
    assert model.distogram_head.input_dtypes == [torch.float32]
    assert model.distogram_head.autocast_enabled == [False]
    assert model.confidence_head.embedding_dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real bf16 autocast validation")
def test_baseline_sample_keeps_only_reverse_diffusion_fp32_under_cuda_bfloat16_autocast() -> None:
    device = torch.device("cuda")
    model = _BaselineTiny().to(device)
    model.trunk_output_dtype = torch.bfloat16
    model_input, _ = _baseline_training_pair(device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.sample(
            model_input,
            evaluation_recipe="protenix_baseline_v1",
            num_samples=2,
            num_sampling_steps=3,
            recycling_steps=2,
            generator=torch.Generator(device=device).manual_seed(127),
        )
        diffusion_loss = output.coordinates.square().mean()

    assert output.coordinates.shape == (2, 2, 4, 3)
    assert output.coordinates.dtype == torch.float32
    assert diffusion_loss.dtype == torch.float32
    assert model.rollout_dtypes == [(torch.float32, torch.float32, torch.float32)]
    assert model.rollout_autocast_enabled == [False]
    assert model.confidence_head.embedding_dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert model.confidence_head.pred_dtype == torch.float32
    assert model.confidence_head.autocast_enabled is True
    assert output.distogram_logits.dtype == torch.float32
    assert model.distogram_head.input_dtypes == [torch.float32]
    assert model.distogram_head.autocast_enabled == [False]


def test_baseline_recipe_requires_permutation_metadata_before_trunk() -> None:
    model = _BaselineTiny()
    model_input, target = _training_pair()
    model_input.msa_tokens = torch.zeros(2, 2, 1, dtype=torch.long)

    with pytest.raises(ValueError, match="requires target permutation metadata"):
        model(model_input, target, training_recipe="protenix_baseline_v1")

    assert model.trunk_calls == 0


def test_baseline_recipe_requires_atom_molecule_type_before_trunk() -> None:
    model = _BaselineTiny()
    model_input, target = _baseline_training_pair()
    target.atom_molecule_type = None

    with pytest.raises(ValueError, match="atom_molecule_type"):
        model(model_input, target, training_recipe="protenix_baseline_v1")

    assert model.trunk_calls == 0
