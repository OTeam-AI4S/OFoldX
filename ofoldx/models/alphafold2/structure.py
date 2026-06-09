r"""
AlphaFold 2 structure module geometry.

This module ports the inference-side geometry of DeepMind AlphaFold 2's
``folding.py`` / ``all_atom.py`` into PyTorch: invariant point attention,
backbone rigid updates, torsion-angle side-chain frames, and atom14-to-atom37
projection. The loss/violation helpers are intentionally not part of this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from oops import LayerNorm
from torch import Tensor, nn

from ...data import residue_constants


def _restype_atom37_to_atom14_tables() -> tuple[np.ndarray, np.ndarray]:
    atom37_to_atom14 = np.zeros((21, residue_constants.atom_type_num), dtype=np.int64)
    atom37_exists = np.zeros((21, residue_constants.atom_type_num), dtype=np.float32)
    for restype_index in range(21):
        resname = "UNK"
        if restype_index < residue_constants.restype_num:
            resname = residue_constants.restype_1to3[residue_constants.restypes[restype_index]]
        atom14_names = residue_constants.restype_name_to_atom14_names[resname]
        for atom37_index, atom_name in enumerate(residue_constants.atom_types):
            if atom_name in atom14_names:
                atom14_index = atom14_names.index(atom_name)
                if atom14_names[atom14_index]:
                    atom37_to_atom14[restype_index, atom37_index] = atom14_index
                    atom37_exists[restype_index, atom37_index] = 1.0
    return atom37_to_atom14, atom37_exists


_RESTYPE_ATOM37_TO_ATOM14, _RESTYPE_ATOM37_MASK = _restype_atom37_to_atom14_tables()


@dataclass(slots=True)
class AlphaFold2StructureModuleOutput:
    r"""
    Output tensors from the native AF2 structure module.
    """

    final_atom_positions: Tensor
    final_atom_mask: Tensor
    structure_module: Tensor
    final_atom14_positions: Tensor
    final_atom14_mask: Tensor
    final_affines: Tensor
    traj: Tensor
    sidechain_angles: Tensor
    sidechain_unnormalized_angles: Tensor


@dataclass(slots=True)
class Rigid:
    r"""
    Batched rigid transform represented by rotation matrices and translations.
    """

    rotation: Tensor
    translation: Tensor

    @classmethod
    def from_tensor4x4(cls, tensor: Tensor) -> Rigid:
        return cls(rotation=tensor[..., :3, :3], translation=tensor[..., :3, 3])

    def apply(self, points: Tensor) -> Tensor:
        return torch.matmul(self.rotation, points.unsqueeze(-1)).squeeze(-1) + self.translation

    def compose(self, other: Rigid) -> Rigid:
        rotation = self.rotation @ other.rotation
        translation = self.apply(other.translation)
        return Rigid(rotation=rotation, translation=translation)

    def compose_rotation(self, rotation: Tensor) -> Rigid:
        return Rigid(rotation=self.rotation @ rotation, translation=self.translation)

    def with_group_axis(self) -> Rigid:
        return Rigid(rotation=self.rotation.unsqueeze(-3), translation=self.translation.unsqueeze(-2))


@dataclass(slots=True)
class QuatAffine:
    r"""
    AF2 backbone affine represented by a quaternion, rotation, and translation.
    """

    quaternion: Tensor
    rotation: Tensor
    translation: Tensor

    @classmethod
    def identity(cls, shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> QuatAffine:
        quaternion = torch.zeros(*shape, 4, device=device, dtype=dtype)
        quaternion[..., 0] = 1.0
        rotation = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3).clone()
        translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
        return cls(quaternion=quaternion, rotation=rotation, translation=translation)

    @classmethod
    def from_tensor(cls, tensor: Tensor, *, normalize: bool = False) -> QuatAffine:
        quaternion = tensor[..., :4]
        if normalize:
            quaternion = F.normalize(quaternion, dim=-1)
        return cls(quaternion=quaternion, rotation=quat_to_rot(quaternion), translation=tensor[..., 4:7])

    def to_tensor(self) -> Tensor:
        return torch.cat([self.quaternion, self.translation], dim=-1)

    def to_rigid(self) -> Rigid:
        return Rigid(rotation=self.rotation, translation=self.translation)

    def scale_translation(self, position_scale: float) -> QuatAffine:
        return QuatAffine(
            quaternion=self.quaternion,
            rotation=self.rotation,
            translation=self.translation * position_scale,
        )

    def detach_rotation(self) -> QuatAffine:
        return QuatAffine(
            quaternion=self.quaternion.detach(),
            rotation=self.rotation.detach(),
            translation=self.translation,
        )

    def pre_compose(self, update: Tensor) -> QuatAffine:
        vector_quaternion_update = update[..., :3]
        translation_update = update[..., 3:6]
        quaternion = self.quaternion + quat_multiply_by_vec(self.quaternion, vector_quaternion_update)
        translation = self.translation + torch.matmul(self.rotation, translation_update.unsqueeze(-1)).squeeze(-1)
        quaternion = F.normalize(quaternion, dim=-1)
        return QuatAffine(quaternion=quaternion, rotation=quat_to_rot(quaternion), translation=translation)

    def apply_to_point(self, point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        return torch.matmul(rotation, point.unsqueeze(-1)).squeeze(-1) + translation

    def invert_point(self, transformed_point: Tensor, *, extra_dims: int = 0) -> Tensor:
        rotation = self.rotation
        translation = self.translation
        for _ in range(extra_dims):
            rotation = rotation.unsqueeze(-3)
            translation = translation.unsqueeze(-2)
        point = transformed_point - translation
        return torch.matmul(rotation.transpose(-1, -2), point.unsqueeze(-1)).squeeze(-1)


class AlphaFold2StructureModule(nn.Module):
    r"""
    Native AlphaFold 2 StructureModule inference path.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_layer: int,
        num_layer_in_transition: int,
        num_channel: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_num_channel: int,
        sidechain_num_residual_block: int,
        position_scale: float,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.position_scale = position_scale
        self.single_layer_norm = LayerNorm(single_size)
        self.initial_projection = nn.Linear(single_size, num_channel)
        self.pair_layer_norm = LayerNorm(pair_size)
        self.fold_iteration = FoldIteration(
            num_channel=num_channel,
            pair_size=pair_size,
            num_layer_in_transition=num_layer_in_transition,
            num_head=num_head,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            sidechain_num_channel=sidechain_num_channel,
            sidechain_num_residual_block=sidechain_num_residual_block,
            position_scale=position_scale,
            dropout=dropout,
            zero_init=zero_init,
        )

    def reset_parameters(self) -> None:
        self.fold_iteration.reset_parameters()

    def forward(self, single: Tensor, pair: Tensor, batch: Any) -> AlphaFold2StructureModuleOutput:
        token_mask = batch.token_mask.to(device=single.device, dtype=single.dtype)
        aatype = _aatype_from_batch(batch, token_mask)
        act = self.single_layer_norm(single)
        initial_act = act
        act = self.initial_projection(act)
        pair = self.pair_layer_norm(pair)
        affine = QuatAffine.identity(token_mask.shape, device=single.device, dtype=single.dtype)

        affines = []
        final_sidechain: SidechainOutput | None = None
        for _ in range(self.num_layer):
            act, affine, sidechain, raw_affine = self.fold_iteration(
                act=act,
                pair=pair,
                mask=token_mask,
                affine=affine,
                initial_act=initial_act,
                aatype=aatype,
            )
            affines.append(raw_affine.to_tensor())
            final_sidechain = sidechain

        if final_sidechain is None:
            raise RuntimeError("AF2 structure module produced no fold iterations")

        traj = torch.stack(affines, dim=0)
        traj_scale = traj.new_tensor(
            [1.0, 1.0, 1.0, 1.0, self.position_scale, self.position_scale, self.position_scale],
        )
        traj = traj * traj_scale
        atom14_mask = restype_atom14_mask(aatype, dtype=single.dtype) * token_mask.unsqueeze(-1)
        atom37_positions, atom37_mask = atom14_to_atom37(final_sidechain.atom_pos, aatype, token_mask)
        return AlphaFold2StructureModuleOutput(
            final_atom_positions=atom37_positions,
            final_atom_mask=atom37_mask,
            structure_module=act,
            final_atom14_positions=final_sidechain.atom_pos,
            final_atom14_mask=atom14_mask,
            final_affines=traj[-1],
            traj=traj,
            sidechain_angles=final_sidechain.angles_sin_cos,
            sidechain_unnormalized_angles=final_sidechain.unnormalized_angles_sin_cos,
        )


class FoldIteration(nn.Module):
    r"""
    One shared-weight AF2 structure-module iteration.
    """

    def __init__(
        self,
        *,
        num_channel: int,
        pair_size: int,
        num_layer_in_transition: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        sidechain_num_channel: int,
        sidechain_num_residual_block: int,
        position_scale: float,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.position_scale = position_scale
        self.invariant_point_attention = InvariantPointAttention(
            num_channel=num_channel,
            pair_size=pair_size,
            num_head=num_head,
            num_scalar_qk=num_scalar_qk,
            num_scalar_v=num_scalar_v,
            num_point_qk=num_point_qk,
            num_point_v=num_point_v,
            zero_init=zero_init,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_layer_norm = LayerNorm(num_channel)
        self.transition = StructureTransition(
            num_channel=num_channel,
            num_layer_in_transition=num_layer_in_transition,
            zero_init=zero_init,
        )
        self.transition_layer_norm = LayerNorm(num_channel)
        self.affine_update = nn.Linear(num_channel, 6)
        self.sidechain = MultiRigidSidechain(
            num_channel=num_channel,
            sidechain_num_channel=sidechain_num_channel,
            num_residual_block=sidechain_num_residual_block,
            zero_init=zero_init,
        )

    def reset_parameters(self) -> None:
        self.invariant_point_attention.reset_parameters()
        self.transition.reset_parameters()
        nn.init.zeros_(self.affine_update.weight)
        nn.init.zeros_(self.affine_update.bias)
        self.sidechain.reset_parameters()

    def forward(
        self,
        *,
        act: Tensor,
        pair: Tensor,
        mask: Tensor,
        affine: QuatAffine,
        initial_act: Tensor,
        aatype: Tensor,
    ) -> tuple[Tensor, QuatAffine, SidechainOutput, QuatAffine]:
        attn = self.invariant_point_attention(inputs_1d=act, inputs_2d=pair, mask=mask, affine=affine)
        act = self.attention_layer_norm(act + self.dropout(attn))
        act = self.transition_layer_norm(act + self.dropout(self.transition(act)))
        affine = affine.pre_compose(self.affine_update(act))
        sidechain = self.sidechain(affine.scale_translation(self.position_scale), (act, initial_act), aatype)
        return act, affine.detach_rotation(), sidechain, affine


class InvariantPointAttention(nn.Module):
    r"""
    AF2 invariant point attention.
    """

    def __init__(
        self,
        *,
        num_channel: int,
        pair_size: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
        zero_init: bool,
        dist_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.dist_epsilon = dist_epsilon
        self.zero_init = zero_init
        self.q_scalar = nn.Linear(num_channel, num_head * num_scalar_qk)
        self.kv_scalar = nn.Linear(num_channel, num_head * (num_scalar_v + num_scalar_qk))
        self.q_point_local = nn.Linear(num_channel, num_head * 3 * num_point_qk)
        self.kv_point_local = nn.Linear(num_channel, num_head * 3 * (num_point_qk + num_point_v))
        self.attention_2d = nn.Linear(pair_size, num_head)
        output_size = num_head * num_scalar_v + 4 * num_head * num_point_v + num_head * pair_size
        self.output_projection = nn.Linear(output_size, num_channel)
        self.trainable_point_weights = nn.Parameter(torch.empty(num_head))
        self._reset_point_weights()

    def reset_parameters(self) -> None:
        self._reset_point_weights()
        if self.zero_init:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, *, inputs_1d: Tensor, inputs_2d: Tensor, mask: Tensor, affine: QuatAffine) -> Tensor:
        num_head = self.num_head
        q_scalar = self.q_scalar(inputs_1d).unflatten(-1, (num_head, self.num_scalar_qk))
        kv_scalar = self.kv_scalar(inputs_1d).unflatten(-1, (num_head, self.num_scalar_v + self.num_scalar_qk))
        k_scalar, v_scalar = kv_scalar.split((self.num_scalar_qk, self.num_scalar_v), dim=-1)

        q_point_local = self.q_point_local(inputs_1d).unflatten(-1, (3, num_head, self.num_point_qk)).movedim(-3, -1)
        q_point = affine.apply_to_point(q_point_local, extra_dims=2)
        kv_point_local = (
            self.kv_point_local(inputs_1d)
            .unflatten(-1, (3, num_head, self.num_point_qk + self.num_point_v))
            .movedim(-3, -1)
        )
        kv_point = affine.apply_to_point(kv_point_local, extra_dims=2)
        k_point, v_point = kv_point.split((self.num_point_qk, self.num_point_v), dim=-2)

        scalar_weights = math.sqrt(1.0 / (3.0 * max(self.num_scalar_qk, 1)))
        q_scalar = scalar_weights * q_scalar
        attn_qk_scalar = torch.einsum("...qhd,...khd->...hqk", q_scalar, k_scalar)

        point_weights = math.sqrt(1.0 / (3.0 * max(self.num_point_qk, 1) * 9.0 / 2.0))
        point_weights = point_weights * F.softplus(self.trainable_point_weights)
        q_point = q_point.transpose(-4, -3)
        k_point = k_point.transpose(-4, -3)
        dist2 = (q_point.unsqueeze(-3) - k_point.unsqueeze(-4)).square().sum(dim=-1)
        attn_qk_point = -0.5 * (dist2 * _expand_head_weights(point_weights, dist2)).sum(dim=-1)

        attention_2d_weights = math.sqrt(1.0 / 3.0)
        attention_2d = attention_2d_weights * self.attention_2d(inputs_2d).movedim(-1, -3)
        attn_logits = attn_qk_scalar + attn_qk_point + attention_2d
        pair_mask = mask.to(dtype=torch.bool).unsqueeze(-1) & mask.to(dtype=torch.bool).unsqueeze(-2)
        attn_logits = attn_logits.masked_fill(~pair_mask.unsqueeze(-3), -1e5)
        attn = torch.softmax(attn_logits, dim=-1)

        v = v_scalar.transpose(-3, -2)
        result_scalar = torch.einsum("...hqk,...hkd->...hqd", attn, v).transpose(-3, -2).flatten(-2)
        v_point = v_point.transpose(-4, -3)
        result_point = torch.einsum("...hqk,...hkpc->...hqpc", attn, v_point).transpose(-4, -3)
        result_point = affine.invert_point(result_point, extra_dims=2)
        result_point_norm = torch.sqrt(self.dist_epsilon + result_point.square().sum(dim=-1))
        result_attention_over_2d = torch.einsum("...hqk,...qkc->...qhc", attn, inputs_2d).flatten(-2)
        output_features = torch.cat(
            [
                result_scalar,
                result_point[..., 0].flatten(-2),
                result_point[..., 1].flatten(-2),
                result_point[..., 2].flatten(-2),
                result_point_norm.flatten(-2),
                result_attention_over_2d,
            ],
            dim=-1,
        )
        return self.output_projection(output_features)

    def _reset_point_weights(self) -> None:
        nn.init.constant_(self.trainable_point_weights, math.log(math.exp(1.0) - 1.0))


class StructureTransition(nn.Module):
    def __init__(self, *, num_channel: int, num_layer_in_transition: int, zero_init: bool) -> None:
        super().__init__()
        self.zero_init = zero_init
        self.layers = nn.ModuleList(nn.Linear(num_channel, num_channel) for _ in range(num_layer_in_transition))

    def reset_parameters(self) -> None:
        if self.zero_init and self.layers:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, act: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            act = layer(act)
            if index != len(self.layers) - 1:
                act = F.relu(act)
        return act


@dataclass(slots=True)
class SidechainOutput:
    angles_sin_cos: Tensor
    unnormalized_angles_sin_cos: Tensor
    atom_pos: Tensor
    frames: Rigid


class MultiRigidSidechain(nn.Module):
    r"""
    AF2 side-chain torsion and atom14 coordinate generator.
    """

    def __init__(
        self,
        *,
        num_channel: int,
        sidechain_num_channel: int,
        num_residual_block: int,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.zero_init = zero_init
        self.input_projection = nn.ModuleList(nn.Linear(num_channel, sidechain_num_channel) for _ in range(2))
        self.resblocks = nn.ModuleList(
            SidechainResidualBlock(sidechain_num_channel, zero_init=zero_init) for _ in range(num_residual_block)
        )
        self.unnormalized_angles = nn.Linear(sidechain_num_channel, 14)

    def reset_parameters(self) -> None:
        for block in self.resblocks:
            block.reset_parameters()
        if self.zero_init:
            nn.init.zeros_(self.unnormalized_angles.weight)
            nn.init.zeros_(self.unnormalized_angles.bias)

    def forward(self, affine: QuatAffine, representations: tuple[Tensor, Tensor], aatype: Tensor) -> SidechainOutput:
        act = sum(proj(F.relu(rep)) for proj, rep in zip(self.input_projection, representations, strict=True))
        for block in self.resblocks:
            act = block(act)
        unnormalized_angles = self.unnormalized_angles(F.relu(act)).unflatten(-1, (7, 2))
        angles = F.normalize(unnormalized_angles, dim=-1)
        frames = torsion_angles_to_frames(aatype, affine.to_rigid(), angles)
        atom_pos = frames_and_literature_positions_to_atom14_pos(aatype, frames)
        return SidechainOutput(
            angles_sin_cos=angles,
            unnormalized_angles_sin_cos=unnormalized_angles,
            atom_pos=atom_pos,
            frames=frames,
        )


class SidechainResidualBlock(nn.Module):
    def __init__(self, num_channel: int, *, zero_init: bool) -> None:
        super().__init__()
        self.zero_init = zero_init
        self.resblock1 = nn.Linear(num_channel, num_channel)
        self.resblock2 = nn.Linear(num_channel, num_channel)

    def reset_parameters(self) -> None:
        if self.zero_init:
            nn.init.zeros_(self.resblock2.weight)
            nn.init.zeros_(self.resblock2.bias)

    def forward(self, act: Tensor) -> Tensor:
        old_act = act
        act = self.resblock1(F.relu(act))
        act = self.resblock2(F.relu(act))
        return act + old_act


def quat_to_rot(normalized_quat: Tensor) -> Tensor:
    w, x, y, z = normalized_quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def quat_multiply_by_vec(quat: Tensor, vec: Tensor) -> Tensor:
    r, i, j, k = quat.unbind(dim=-1)
    x, y, z = vec.unbind(dim=-1)
    return torch.stack(
        [
            -i * x - j * y - k * z,
            r * x + j * z - k * y,
            r * y + k * x - i * z,
            r * z + i * y - j * x,
        ],
        dim=-1,
    )


def torsion_angles_to_frames(aatype: Tensor, backb_to_global: Rigid, torsion_angles_sin_cos: Tensor) -> Rigid:
    default_frames = Rigid.from_tensor4x4(
        _gather_restype(residue_constants.restype_rigid_group_default_frame, aatype, dtype=torsion_angles_sin_cos.dtype)
    )
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]
    sin_angles = torch.cat([torch.zeros_like(sin_angles[..., :1]), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones_like(cos_angles[..., :1]), cos_angles], dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)
    all_rots = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_angles, -sin_angles], dim=-1),
            torch.stack([zeros, sin_angles, cos_angles], dim=-1),
        ],
        dim=-2,
    )
    all_frames = default_frames.compose_rotation(all_rots)
    chi1 = _frame_index(all_frames, 4)
    chi2 = chi1.compose(_frame_index(all_frames, 5))
    chi3 = chi2.compose(_frame_index(all_frames, 6))
    chi4 = chi3.compose(_frame_index(all_frames, 7))
    frames_to_backbone = Rigid(
        rotation=torch.cat(
            [
                all_frames.rotation[..., :5, :, :],
                chi2.rotation.unsqueeze(-3),
                chi3.rotation.unsqueeze(-3),
                chi4.rotation.unsqueeze(-3),
            ],
            dim=-3,
        ),
        translation=torch.cat(
            [
                all_frames.translation[..., :5, :],
                chi2.translation.unsqueeze(-2),
                chi3.translation.unsqueeze(-2),
                chi4.translation.unsqueeze(-2),
            ],
            dim=-2,
        ),
    )
    return backb_to_global.with_group_axis().compose(frames_to_backbone)


def frames_and_literature_positions_to_atom14_pos(aatype: Tensor, all_frames_to_global: Rigid) -> Tensor:
    group_index = _gather_restype(residue_constants.restype_atom14_to_rigid_group, aatype, dtype=torch.long)
    group_mask = F.one_hot(group_index, num_classes=8).to(dtype=all_frames_to_global.rotation.dtype)
    atom_frames = Rigid(
        rotation=torch.einsum("...nag,...ngij->...naij", group_mask, all_frames_to_global.rotation),
        translation=torch.einsum("...nag,...ngc->...nac", group_mask, all_frames_to_global.translation),
    )
    lit_positions = _gather_restype(
        residue_constants.restype_atom14_rigid_group_positions,
        aatype,
        dtype=all_frames_to_global.rotation.dtype,
    )
    pred_positions = atom_frames.apply(lit_positions)
    return pred_positions * restype_atom14_mask(aatype, dtype=pred_positions.dtype).unsqueeze(-1)


def atom14_to_atom37(atom14_positions: Tensor, aatype: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor]:
    atom37_to_atom14 = _gather_restype(_RESTYPE_ATOM37_TO_ATOM14, aatype, dtype=torch.long)
    gather_index = atom37_to_atom14.unsqueeze(-1).expand(*atom37_to_atom14.shape, atom14_positions.shape[-1])
    atom37_positions = torch.gather(atom14_positions, dim=-2, index=gather_index)
    atom37_mask = _gather_restype(_RESTYPE_ATOM37_MASK, aatype, dtype=atom14_positions.dtype)
    atom37_mask = atom37_mask * token_mask.unsqueeze(-1).to(dtype=atom37_mask.dtype)
    return atom37_positions * atom37_mask.unsqueeze(-1), atom37_mask


def restype_atom14_mask(aatype: Tensor, *, dtype: torch.dtype) -> Tensor:
    return _gather_restype(residue_constants.restype_atom14_mask, aatype, dtype=dtype)


def _frame_index(rigid: Rigid, index: int) -> Rigid:
    return Rigid(rotation=rigid.rotation[..., index, :, :], translation=rigid.translation[..., index, :])


def _gather_restype(table: np.ndarray, aatype: Tensor, *, dtype: torch.dtype) -> Tensor:
    source = torch.as_tensor(table, device=aatype.device, dtype=dtype)
    index = aatype.clamp(min=0, max=20).long()
    gathered = source.index_select(0, index.reshape(-1))
    return gathered.reshape(*index.shape, *source.shape[1:])


def _aatype_from_batch(batch: Any, token_mask: Tensor) -> Tensor:
    if getattr(batch, "aatype", None) is not None:
        return batch.aatype.to(device=token_mask.device, dtype=torch.long).clamp(min=0, max=20)
    raise ValueError("AlphaFold2StructureModule requires batch.aatype for official AF2 residue geometry")


def _expand_head_weights(weights: Tensor, target: Tensor) -> Tensor:
    view_shape = [1] * target.ndim
    view_shape[-4] = weights.shape[0]
    return weights.view(*view_shape)
