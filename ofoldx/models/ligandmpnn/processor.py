r"""
LigandMPNN side-chain featurization.

The LigandMPNN backbone consumes only N/CA/C/O coordinates, but the side-chain Packer
([`packer`][ofoldx.models.ligandmpnn.packer]) is a Von Mises torsion-diffusion model that
reconstructs full atom14 side chains from predicted chi angles. That reconstruction needs
AlphaFold2-style rigid torsion frames, which this module builds from backbone coordinates.

The frame math ([`Frame`][], [`torsion_angles_to_frames`][],
[`frames_and_literature_positions_to_atom14_pos`][], [`atom37_to_torsion_angles`][]) is a
compact rotation-matrix reimplementation of the OpenFold ``Rigid``/``feats`` routines used by
the reference LigandMPNN ``sc_utils``; it reproduces their numerics without vendoring the full
``Rigid``/``Rotation`` class hierarchy. Reference amino-acid geometry lives in
[`residue_constants`][ofoldx.data.residue_constants].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from ...data import residue_constants as rc
from ...processors.mixin import ProcessorMixin
from ...processors.registry import PROCESSORS
from ...processors.utils import StructureInput, structure_from_input
from .model import LigandMPNNInput

if TYPE_CHECKING:
    from ...data.structure import Structure

# LigandMPNN restype integers (alphabetical 3-letter order) remapped to the AlphaFold2 restype
# integers used by ``residue_constants``: ``MPNN_TO_AF2_RESTYPE[i]`` is the AF2 restype index for
# LigandMPNN token ``i``.
_MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_AF2_RESTYPE_ORDER = [*rc.restypes, "X"]
MPNN_TO_AF2_RESTYPE = torch.tensor([_AF2_RESTYPE_ORDER.index(letter) for letter in _MPNN_ALPHABET], dtype=torch.long)


@PROCESSORS.register("ligandmpnn")
class LigandMPNNProcessor(ProcessorMixin):
    r"""
    Convert structure inputs into the native LigandMPNN tensor contract.

    Args:
        atom_context_num: Per-residue ligand/context atom budget. Structures with no ligand context emit a
            zero-filled context tensor with this width and an all-false context mask.
        temperature: Default sampling temperature attached to the model input.
        num_samples: Optional default number of autoregressive samples.
    """

    input_class = LigandMPNNInput

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> LigandMPNNProcessor:
        r"""
        Build a processor whose context-atom budget matches a LigandMPNN model config.

        Args:
            config: Model config exposing ``atom_context_num``.
            **kwargs: Constructor overrides.

        Returns:
            A configured [`LigandMPNNProcessor`][ofoldx.models.ligandmpnn.processor.LigandMPNNProcessor].
        """

        atom_context_num = _config_scalar(config, "atom_context_num", 25)
        if not isinstance(atom_context_num, int | float | str):
            atom_context_num = 25
        parameters: dict[str, Any] = {"atom_context_num": int(atom_context_num)}
        parameters.update(kwargs)
        return cls(**parameters)

    def __init__(
        self,
        *,
        atom_context_num: int = 25,
        temperature: float = 0.1,
        num_samples: int | None = None,
    ) -> None:
        if atom_context_num <= 0:
            raise ValueError("atom_context_num must be positive")
        self.atom_context_num = int(atom_context_num)
        self.temperature = float(temperature)
        self.num_samples = None if num_samples is None else int(num_samples)

    def __call__(self, structure: StructureInput) -> LigandMPNNInput:
        r"""
        Featurize one structure for LigandMPNN sequence design.

        Args:
            structure: Parsed structure or structure file path.

        Returns:
            A single-structure [`LigandMPNNInput`][ofoldx.models.ligandmpnn.model.LigandMPNNInput].
        """

        features = structure_to_packer_features(structure_from_input(structure))
        backbone = features["X"]
        token_mask = features["mask"]
        batch_size, residue_count = token_mask.shape
        ligand_atom_coords = torch.zeros(
            batch_size,
            residue_count,
            self.atom_context_num,
            3,
            dtype=backbone.dtype,
            device=backbone.device,
        )
        ligand_atom_types = torch.zeros(
            batch_size,
            residue_count,
            self.atom_context_num,
            dtype=torch.long,
            device=backbone.device,
        )
        ligand_atom_mask = torch.zeros(
            batch_size,
            residue_count,
            self.atom_context_num,
            dtype=token_mask.dtype,
            device=backbone.device,
        )
        return LigandMPNNInput(
            backbone_coords=backbone,
            sequence_tokens=features["S"],
            token_mask=token_mask,
            residue_index=features["R_idx"],
            chain_index=features["chain_labels"],
            ligand_atom_coords=ligand_atom_coords,
            ligand_atom_types=ligand_atom_types,
            ligand_atom_mask=ligand_atom_mask,
            design_mask=torch.ones_like(token_mask),
            temperature=self.temperature,
            num_samples=self.num_samples,
            all_atom_coords=features["xyz_37"],
            all_atom_mask=features["xyz_37_m"],
        )

    def _processor_config(self) -> dict[str, object]:
        r"""Return the JSON-serializable processor configuration."""

        return {
            "atom_context_num": self.atom_context_num,
            "temperature": self.temperature,
            "num_samples": self.num_samples,
        }


def _config_scalar(config: Any, key: str, default: object) -> object:
    r"""Read ``key`` from either a mapping or an attribute-bearing config."""

    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class Frame:
    r"""
    Rigid frame stored as a rotation matrix and a translation.

    A minimal rotation-matrix analogue of the OpenFold ``Rigid`` transform, exposing only the
    operations the Packer needs. Batched over any number of leading dimensions.

    Args:
        rotation: Rotation matrices with shape ``(*, 3, 3)``.
        translation: Translations with shape ``(*, 3)``.

    Attributes:
        rotation: Rotation matrices with shape ``(*, 3, 3)``.
        translation: Translations with shape ``(*, 3)``.

    Examples:
        >>> import torch
        >>> frame = Frame(torch.eye(3), torch.tensor([1.0, 2.0, 3.0]))
        >>> frame.apply(torch.zeros(3)).tolist()
        [1.0, 2.0, 3.0]
    """

    __slots__ = ("rotation", "translation")

    def __init__(self, rotation: Tensor, translation: Tensor) -> None:
        self.rotation = rotation
        self.translation = translation

    def compose(self, other: Frame) -> Frame:
        r"""
        Compose two frames, applying ``other`` in ``self``'s local frame.

        Args:
            other: Frame to apply after ``self``.

        Returns:
            The composed frame ``self ∘ other``.
        """

        rotation = self.rotation @ other.rotation
        translation = _rotate(self.rotation, other.translation) + self.translation
        return Frame(rotation, translation)

    def apply(self, points: Tensor) -> Tensor:
        r"""
        Map local points into the global frame (rotate then translate).

        Args:
            points: Local coordinates with shape ``(*, 3)``.

        Returns:
            Global coordinates with shape ``(*, 3)``.
        """

        return _rotate(self.rotation, points) + self.translation

    def invert_apply(self, points: Tensor) -> Tensor:
        r"""
        Map global points back into the local frame (inverse of `apply`).

        Args:
            points: Global coordinates with shape ``(*, 3)``.

        Returns:
            Local coordinates with shape ``(*, 3)``.
        """

        return _rotate(self.rotation.transpose(-1, -2), points - self.translation)

    def unsqueeze(self, dim: int) -> Frame:
        r"""
        Insert a size-one dimension into the shared leading axes.

        Args:
            dim: Position of the new axis among the leading (batch) dimensions.

        Returns:
            A frame with the extra leading axis inserted into both rotation and translation.
        """

        rotation_dim = dim - 2 if dim < 0 else dim
        translation_dim = dim - 1 if dim < 0 else dim
        return Frame(self.rotation.unsqueeze(rotation_dim), self.translation.unsqueeze(translation_dim))

    def __getitem__(self, index) -> Frame:
        r"""Index the shared leading (batch) axes, returning the sub-frame at ``index``."""

        if not isinstance(index, tuple):
            index = (index,)
        return Frame(self.rotation[index + (slice(None), slice(None))], self.translation[index + (slice(None),)])

    @staticmethod
    def from_tensor_4x4(transform: Tensor) -> Frame:
        r"""
        Build a frame from a homogeneous 4x4 transform.

        Args:
            transform: Homogeneous transforms with shape ``(*, 4, 4)``.

        Returns:
            A frame taking the rotation from the top-left 3x3 block and translation from the last column.
        """

        return Frame(transform[..., :3, :3], transform[..., :3, 3])

    @staticmethod
    def from_3_points(neg_x_axis: Tensor, origin: Tensor, xy_plane: Tensor, eps: float = 1e-8) -> Frame:
        r"""
        Gram-Schmidt frame from three points (AF2 algorithm 21).

        Args:
            neg_x_axis: Point defining the negative x-axis direction, shape ``(*, 3)``.
            origin: Frame origin, shape ``(*, 3)``.
            xy_plane: Point placed in the xy-plane, shape ``(*, 3)``.
            eps: Numerical floor for the normalizations.

        Returns:
            The Gram-Schmidt frame with origin at ``origin``.
        """

        e0 = origin - neg_x_axis
        e1 = xy_plane - origin
        e0 = e0 / torch.sqrt((e0 * e0).sum(-1, keepdim=True) + eps)
        e1 = e1 - e0 * (e0 * e1).sum(-1, keepdim=True)
        e1 = e1 / torch.sqrt((e1 * e1).sum(-1, keepdim=True) + eps)
        e2 = torch.cross(e0, e1, dim=-1)
        rotation = torch.stack([e0, e1, e2], dim=-1)
        return Frame(rotation, origin)

    @staticmethod
    def make_transform_from_reference(n_xyz: Tensor, ca_xyz: Tensor, c_xyz: Tensor, eps: float = 1e-20) -> Frame:
        r"""
        Backbone frame from N/CA/C coordinates (AF2 reference-frame construction).

        Args:
            n_xyz: Nitrogen coordinates, shape ``(*, 3)``.
            ca_xyz: Alpha-carbon coordinates, shape ``(*, 3)``.
            c_xyz: Carbon coordinates, shape ``(*, 3)``.
            eps: Numerical floor for the normalizations.

        Returns:
            The backbone frame with origin at the alpha carbon.
        """

        translation = -ca_xyz
        n_local = n_xyz + translation
        c_local = c_xyz + translation

        c_x, c_y, c_z = c_local[..., 0], c_local[..., 1], c_local[..., 2]
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1, cos_c1 = -c_y / norm, c_x / norm
        c1 = _stack_rot([[cos_c1, -sin_c1, None], [sin_c1, cos_c1, None], [None, None, 1.0]], c_x)

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2, cos_c2 = c_z / norm, torch.sqrt(c_x**2 + c_y**2) / norm
        c2 = _stack_rot([[cos_c2, None, sin_c2], [None, 1.0, None], [-sin_c2, None, cos_c2]], c_x)

        c_rots = c2 @ c1
        n_rotated = _rotate(c_rots, n_local)
        n_y, n_z = n_rotated[..., 1], n_rotated[..., 2]
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n, cos_n = -n_z / norm, n_y / norm
        n_rot = _stack_rot([[1.0, None, None], [None, cos_n, -sin_n], [None, sin_n, cos_n]], c_x)

        rotation = (n_rot @ c_rots).transpose(-1, -2)
        return Frame(rotation, -translation)


def _rotate(rotation: Tensor, points: Tensor) -> Tensor:
    r"""Apply rotation matrices ``(*, 3, 3)`` to points ``(*, 3)``."""

    return (rotation @ points.unsqueeze(-1)).squeeze(-1)


def _stack_rot(rows: list[list], like: Tensor) -> Tensor:
    r"""Assemble a batched ``(*, 3, 3)`` rotation from per-entry tensors/scalars/``None``."""

    zeros = torch.zeros_like(like)
    filled = [
        [zeros if entry is None else (entry + zeros if not torch.is_tensor(entry) else entry) for entry in row]
        for row in rows
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in filled], dim=-2)


def make_atom14_masks(aatype: Tensor) -> dict[str, Tensor]:
    r"""
    Build atom14/atom37 existence masks for AF2 restype integers.

    Args:
        aatype: AF2 restype integers with shape ``(B, L)``.

    Returns:
        Mapping with ``atom14_atom_exists`` (``(B, L, 14)``) and ``atom37_atom_exists`` (``(B, L, 37)``).

    Examples:
        >>> import torch
        >>> masks = make_atom14_masks(torch.zeros(1, 3, dtype=torch.long))
        >>> masks["atom14_atom_exists"].shape
        torch.Size([1, 3, 14])
        >>> masks["atom37_atom_exists"].shape
        torch.Size([1, 3, 37])
    """

    device = aatype.device
    restype_atom14_mask = []
    for restype_letter in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[restype_letter]]
        restype_atom14_mask.append([1.0 if name else 0.0 for name in atom_names])
    restype_atom14_mask.append([0.0] * 14)  # UNK
    restype_atom14_mask = torch.tensor(restype_atom14_mask, dtype=torch.float32, device=device)

    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=device)
    for restype, restype_letter in enumerate(rc.restypes):
        for atom_name in rc.rigid_group_atom_positions[rc.restype_1to3[restype_letter]]:
            restype_atom37_mask[restype, rc.atom_order[atom_name[0]]] = 1

    clamped = aatype.clamp(max=20).long()
    return {
        "atom14_atom_exists": restype_atom14_mask[clamped],
        "atom37_atom_exists": restype_atom37_mask[clamped],
    }


def _get_chi_atom_indices() -> np.ndarray:
    r"""Per-restype atom37 indices for the four chi angles, shape ``(21, 4, 4)``."""

    chi_atom_indices = []
    for residue_name in [rc.restype_1to3[letter] for letter in rc.restypes]:
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        indices = [[rc.atom_order[atom] for atom in chi] for chi in residue_chi_angles]
        indices += [[0, 0, 0, 0]] * (4 - len(indices))
        chi_atom_indices.append(indices)
    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # UNK
    return np.array(chi_atom_indices, dtype=np.int64)


def atom37_to_torsion_angles(aatype: Tensor, all_atom_positions: Tensor, all_atom_mask: Tensor) -> dict[str, Tensor]:
    r"""
    Compute the seven backbone/side-chain torsion angles as sin/cos pairs.

    Compact reimplementation of OpenFold ``atom37_to_torsion_angles`` restricted to what the
    Packer consumes: the sin/cos torsion tensor and its validity mask.

    Args:
        aatype: AF2 restype integers with shape ``(B, L)``.
        all_atom_positions: Atom37 coordinates with shape ``(B, L, 37, 3)``.
        all_atom_mask: Atom37 existence mask with shape ``(B, L, 37)``.

    Returns:
        Mapping with ``torsion_angles_sin_cos`` (``(B, L, 7, 2)``) and ``torsion_angles_mask`` (``(B, L, 7)``).

    Examples:
        >>> torsions = atom37_to_torsion_angles(aatype, all_atom_positions, all_atom_mask)  # doctest: +SKIP
    """

    aatype = aatype.clamp(max=20).long()
    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)
    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_pos = torch.cat([prev_positions[..., 1:3, :], all_atom_positions[..., :2, :]], dim=-2)
    phi_pos = torch.cat([prev_positions[..., 2:3, :], all_atom_positions[..., :3, :]], dim=-2)
    psi_pos = torch.cat([all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]], dim=-2)

    pre_omega_mask = torch.prod(prev_mask[..., 1:3], dim=-1) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_mask[..., 2] * torch.prod(all_atom_mask[..., :3], dim=-1)
    psi_mask = torch.prod(all_atom_mask[..., :3], dim=-1) * all_atom_mask[..., 4]

    chi_atom_indices = torch.as_tensor(_get_chi_atom_indices(), device=aatype.device)[aatype]
    chis_pos = _batched_gather(all_atom_positions, chi_atom_indices, dim=-2)

    chi_angles_mask = all_atom_mask.new_tensor([*rc.chi_angles_mask, [0.0, 0.0, 0.0, 0.0]])
    chis_mask = chi_angles_mask[aatype]
    chi_atoms_mask = torch.prod(_batched_gather(all_atom_mask, chi_atom_indices, dim=-1), dim=-1)
    chis_mask = chis_mask * chi_atoms_mask

    torsions_pos = torch.cat(
        [pre_omega_pos[..., None, :, :], phi_pos[..., None, :, :], psi_pos[..., None, :, :], chis_pos], dim=-3
    )
    torsion_angles_mask = torch.cat(
        [pre_omega_mask[..., None], phi_mask[..., None], psi_mask[..., None], chis_mask], dim=-1
    )

    frames = Frame.from_3_points(torsions_pos[..., 1, :], torsions_pos[..., 2, :], torsions_pos[..., 0, :])
    fourth_atom_rel = frames.invert_apply(torsions_pos[..., 3, :])
    torsion_angles_sin_cos = torch.stack([fourth_atom_rel[..., 2], fourth_atom_rel[..., 1]], dim=-1)
    denom = torch.sqrt(torch.sum(torsion_angles_sin_cos**2, dim=-1, keepdim=True) + 1e-8)
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom
    torsion_angles_sin_cos = (
        torsion_angles_sin_cos * all_atom_mask.new_tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])[..., None]
    )
    return {"torsion_angles_sin_cos": torsion_angles_sin_cos, "torsion_angles_mask": torsion_angles_mask}


def torsion_angles_to_frames(backbone_frames: Frame, torsions: Tensor, aatype: Tensor) -> Frame:
    r"""
    Turn per-residue torsion sin/cos into the eight global rigid-group frames.

    Args:
        backbone_frames: Per-residue backbone frames with batch shape ``(B, L)``.
        torsions: Torsion sin/cos with shape ``(B, L, 7, 2)`` (three backbone + four chi).
        aatype: AF2 restype integers with shape ``(B, L)``.

    Returns:
        Eight global rigid-group frames with batch shape ``(B, L, 8)``.

    Examples:
        >>> frames = torsion_angles_to_frames(backbone_frames, torsions, aatype)  # doctest: +SKIP
    """

    device = torsions.device
    default_frames = torch.tensor(rc.restype_rigid_group_default_frame, device=device)[aatype.long()]
    default = Frame.from_tensor_4x4(default_frames)

    backbone_rotation = torsions.new_zeros((*((1,) * len(torsions.shape[:-1])), 2))
    backbone_rotation[..., 1] = 1
    alpha = torch.cat([backbone_rotation.expand(*torsions.shape[:-2], -1, -1), torsions], dim=-2)

    all_rots = alpha.new_zeros(default.rotation.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1] = alpha[..., 0]
    all_rots[..., 2, 2] = alpha[..., 1]
    all_frames = default.compose(Frame(all_rots, torch.zeros_like(default.translation)))

    chi1 = all_frames[..., 4]
    chi2 = chi1.compose(all_frames[..., 5])
    chi3 = chi2.compose(all_frames[..., 6])
    chi4 = chi3.compose(all_frames[..., 7])
    frames_to_bb = _frame_cat([all_frames[..., :5], chi2.unsqueeze(-1), chi3.unsqueeze(-1), chi4.unsqueeze(-1)])
    return backbone_frames.unsqueeze(-1).compose(frames_to_bb)


def frames_and_literature_positions_to_atom14_pos(frames: Frame, aatype: Tensor) -> Tensor:
    r"""
    Place literature atom14 positions using the global rigid-group frames.

    Args:
        frames: Eight global rigid-group frames with batch shape ``(B, L, 8)``.
        aatype: AF2 restype integers with shape ``(B, L)``.

    Returns:
        Atom14 coordinates with shape ``(B, L, 14, 3)``.

    Examples:
        >>> coords = frames_and_literature_positions_to_atom14_pos(frames, aatype)  # doctest: +SKIP
    """

    device = aatype.device
    aatype = aatype.long()
    group_idx = torch.tensor(rc.restype_atom14_to_rigid_group, device=device)[aatype]
    group_mask = torch.nn.functional.one_hot(group_idx, num_classes=8).to(frames.rotation.dtype)

    rotation = (frames.rotation[..., None, :, :, :] * group_mask[..., None, None]).sum(dim=-3)
    translation = (frames.translation[..., None, :, :] * group_mask[..., None]).sum(dim=-2)
    atom_frames = Frame(rotation, translation)

    lit_positions = torch.tensor(rc.restype_atom14_rigid_group_positions, device=device)[aatype]
    atom_mask = torch.tensor(rc.restype_atom14_mask, device=device)[aatype].unsqueeze(-1)
    return atom_frames.apply(lit_positions) * atom_mask


def _frame_cat(frames: list[Frame]) -> Frame:
    r"""Concatenate frames along their last shared dimension."""

    rotation = torch.cat([frame.rotation for frame in frames], dim=-3)
    translation = torch.cat([frame.translation for frame in frames], dim=-2)
    return Frame(rotation, translation)


def _batched_gather(data: Tensor, indices: Tensor, dim: int) -> Tensor:
    r"""Gather ``data`` along ``dim`` using per-residue ``indices`` (OpenFold ``batched_gather``)."""

    ranges = []
    for index, size in enumerate(data.shape[: len(indices.shape) - (1 if dim < 0 else 0)]):
        if index == (dim % data.ndim):
            break
        ranges.append(torch.arange(size, device=data.device).view(-1, *((1,) * (indices.ndim - index - 1))))
    remaining = [slice(None)] * (data.ndim - len(ranges) - 1)
    ranges.append(indices)
    ranges.extend(remaining)
    return data[tuple(ranges)]


def structure_to_packer_features(
    structure: Structure,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    r"""
    Featurize an OFoldX [`Structure`][ofoldx.data.structure.Structure] for the Packer.

    Extracts per-residue backbone coordinates (N/CA/C/O), the atom37 side-chain coordinates and
    mask, sequence tokens, and residue/chain indices. This is the OFoldX-native replacement
    for the reference LigandMPNN ``parse_PDB``: it reuses the existing ``read_pdb``/``read_mmcif``
    parsers ([`structure`][ofoldx.io.structure]) instead of ProDy.

    Args:
        structure: Parsed structure with protein atom records.
        device: Torch device for the returned tensors.

    Returns:
        Mapping with keys ``X`` (``(1, L, 4, 3)``), ``mask`` (``(1, L)``), ``S`` (``(1, L)``),
        ``xyz_37`` (``(1, L, 37, 3)``), ``xyz_37_m`` (``(1, L, 37)``), ``R_idx`` (``(1, L)``),
        ``chain_labels`` (``(1, L)``), and ``chain_letters`` (list of ``L`` chain ids).

    Raises:
        ValueError: If the structure has no protein CA atoms.

    Examples:
        >>> from ofoldx.io.structure import read_pdb  # doctest: +SKIP
        >>> features = structure_to_packer_features(read_pdb("complex.pdb"))  # doctest: +SKIP
    """

    residue_names: dict[tuple[str, int, str], str] = {}
    residue_coords: dict[tuple[str, int, str], dict[str, tuple[float, float, float]]] = {}
    order: list[tuple[str, int, str]] = []
    for atom in structure.atoms:
        if atom.mol_type not in (None, "protein"):
            continue
        key = (atom.residue.chain_id, atom.residue.residue_number, atom.residue.insertion_code)
        if key not in residue_coords:
            residue_names[key] = atom.residue_name
            residue_coords[key] = {}
            order.append(key)
        if atom.name in rc.atom_order:
            residue_coords[key][atom.name] = (atom.x, atom.y, atom.z)

    order = [key for key in order if "CA" in residue_coords[key]]
    if not order:
        raise ValueError("structure has no protein CA atoms to featurize for the Packer")

    length = len(order)
    xyz_37 = np.zeros([length, 37, 3], dtype=np.float32)
    xyz_37_m = np.zeros([length, 37], dtype=np.float32)
    tokens = np.zeros([length], dtype=np.int64)
    residue_index = np.zeros([length], dtype=np.int64)
    chain_letters = [key[0] for key in order]
    chain_ids = sorted(set(chain_letters))
    chain_labels = np.array([chain_ids.index(letter) for letter in chain_letters], dtype=np.int64)

    mpnn_index = {letter: index for index, letter in enumerate(_MPNN_ALPHABET)}
    for position, key in enumerate(order):
        residue_index[position] = key[1]
        one_letter = rc.restype_3to1.get(residue_names[key], "X")
        tokens[position] = mpnn_index.get(one_letter, mpnn_index["X"])
        for atom_name, coords in residue_coords[key].items():
            atom_type = rc.atom_order[atom_name]
            xyz_37[position, atom_type] = coords
            xyz_37_m[position, atom_type] = 1.0

    backbone_mask = xyz_37_m[:, rc.atom_order["N"]] * xyz_37_m[:, rc.atom_order["CA"]]
    backbone_mask = backbone_mask * xyz_37_m[:, rc.atom_order["C"]] * xyz_37_m[:, rc.atom_order["O"]]
    backbone = np.stack([xyz_37[:, rc.atom_order[name]] for name in ("N", "CA", "C", "O")], axis=1)

    def _to(array: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device).unsqueeze(0)

    return {
        "X": _to(backbone, torch.float32),
        "mask": _to(backbone_mask, torch.float32),
        "S": _to(tokens, torch.long),
        "xyz_37": _to(xyz_37, torch.float32),
        "xyz_37_m": _to(xyz_37_m, torch.float32),
        "R_idx": _to(residue_index, torch.long),
        "chain_labels": _to(chain_labels, torch.long),
        "chain_letters": chain_letters,
    }
