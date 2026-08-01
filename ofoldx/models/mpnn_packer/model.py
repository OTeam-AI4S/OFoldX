# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
MpnnPacker side-chain packing model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributions as distributions
import torch.nn.functional as F
from tensordict import TensorClass, tensorclass
from torch import Tensor, nn

from ...data import residue_constants as rc
from ...data.molecular import GROUP_BY_ATOMIC_NUMBER, PERIOD_BY_ATOMIC_NUMBER
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...model import Model, ShadowTensorCollection
from ...modules.backend import LayerNorm, gather_nodes, masked_knn, resolve_config_backend
from ...modules.functional import radial_basis
from ...modules.geometry import (
    atom37_to_torsion_angles,
    frames_to_atom14_positions,
    local_backbone_frame_features,
    neighbor_atom_distances,
    smoothed_vector_norm,
    torsion_angles_to_frames,
    virtual_beta_carbon,
)
from ...modules.mpnn import (
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnRelativePositionEncoding,
    cat_neighbors_nodes,
)
from ..processing import model_frame_rotation, project_model_categories, project_model_coordinates
from .config import MpnnPackerConfig

_ATOM14 = 14
_ELEMENT_CLASSES = 120
#: The checkpoint's residue classes in its upstream order (canonical spelling); every other canonical class is the
#: packer's unknown residue (index 20), a context descriptor without a geometry template.
MPNN_PACKER_VOCABULARY: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWYX")
_LETTER_IDS = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ],
    )
)
AMINO_ACID_COMPONENT_IDS = tuple(_LETTER_IDS[letter] for letter in MPNN_PACKER_VOCABULARY[:-1])
_UNKNOWN = len(AMINO_ACID_COMPONENT_IDS)
#: Per canonical class, its packer residue index (the 20 amino acids in checkpoint order, every other class unknown).
PACKER_RESIDUE = torch.full((len(TOKENS),), _UNKNOWN, dtype=torch.long)
PACKER_RESIDUE[[TOKEN_INDEX[letter] for letter in MPNN_PACKER_VOCABULARY[:-1]]] = torch.arange(_UNKNOWN)


def _packing_sequence_tokens(sequence_tokens: Tensor, residue_mask: Tensor) -> Tensor:
    """Select available integer classes before vocabulary indexing.

    The unknown class is a supported context descriptor. It does not provide
    a geometry template for a residue selected for side-chain generation.
    """
    if (
        sequence_tokens.dtype != torch.int64
        or residue_mask.dtype != torch.bool
        or sequence_tokens.shape != residue_mask.shape
        or sequence_tokens.device != residue_mask.device
    ):
        fail("value_invalid", RequestRef("sequence_tokens", "int64 classes with a matching boolean residue mask"))
    return project_model_categories(
        sequence_tokens,
        residue_mask,
        num_classes=len(TOKENS),
        feature="packing residue classes",
    )


def normalize_mpnn_packer_input(input: MpnnPackerInput, *, dtype: torch.dtype | None = None) -> MpnnPackerInput:
    """Select available coordinates and classes before arithmetic or indexing."""
    result = cast(MpnnPackerInput, cast(TensorClass, input).clone(recurse=False))
    result.sequence_tokens = _packing_sequence_tokens(input.sequence_tokens, input.residue_mask)
    result.backbone_coordinates = project_model_coordinates(
        input.backbone_coordinates,
        input.residue_mask[:, :, None].expand(-1, -1, 4),
        dtype=dtype,
        feature="packing backbone",
    )
    result.context_mask = input.context_mask & input.residue_mask[:, :, None]
    result.context_atomic_numbers = project_model_categories(
        input.context_atomic_numbers,
        result.context_mask,
        num_classes=_ELEMENT_CLASSES,
        feature="packing context elements",
    )
    result.context_coordinates = project_model_coordinates(
        input.context_coordinates,
        result.context_mask,
        dtype=dtype,
        feature="packing context",
    )
    result.atom37_mask = input.atom37_mask & input.residue_mask[:, :, None]
    result.atom37_coordinates = project_model_coordinates(
        input.atom37_coordinates,
        result.atom37_mask,
        dtype=dtype,
        feature="packing atom37 observations",
    )
    return result


class MpnnPackerModel(Model):
    r"""
    Pack protein side chains with MPNN Packer.

    Accepts [`MpnnPackerInput`][ofoldx.models.mpnn_packer.model.MpnnPackerInput] and returns
    [`MpnnPackerOutput`][ofoldx.models.mpnn_packer.model.MpnnPackerOutput].

    Args:
        config: Model architecture and inference configuration.
    """

    config_class = MpnnPackerConfig
    config: MpnnPackerConfig
    amino_acid_atom14_mask: Tensor
    amino_acid_atom37_mask: Tensor
    amino_acid_atom14_to_atom37: Tensor
    amino_acid_chi_atom_indices: Tensor
    amino_acid_chi_angles_mask: Tensor
    amino_acid_chi_pi_periodic: Tensor
    amino_acid_rigid_group_default_frame: Tensor
    amino_acid_atom14_to_rigid_group: Tensor
    amino_acid_atom14_rigid_group_positions: Tensor

    def __init__(self, config: MpnnPackerConfig | None = None) -> None:
        config = MpnnPackerConfig() if config is None else config
        super().__init__(config)
        self.num_mix = config.num_mix
        self.feature_encoder = MpnnPackerFeatureEncoder(
            node_size=config.hidden_size,
            edge_size=config.edge_size,
            positional_embedding_size=config.positional_embedding_size,
            num_rbf=config.num_rbf,
            num_neighbors=config.num_neighbors,
            num_context_atoms=config.num_context_atoms,
            coordinate_noise=config.coordinate_noise,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
        )
        self.context_edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.node_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.edge_embedding = nn.Linear(config.edge_size, config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            [
                MpnnEncoderBlock(config.hidden_size, config.hidden_size * 2, dropout=config.dropout)
                for _ in range(config.num_encoder_blocks)
            ]
        )
        self.residue_context_input = nn.Linear(config.hidden_size, config.hidden_size)
        self.atom_node_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.atom_edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.atom_context_blocks = nn.ModuleList(
            [MpnnDecoderBlock(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(2)]
        )
        self.context_blocks = nn.ModuleList(
            [MpnnDecoderBlock(config.hidden_size, config.hidden_size * 2, dropout=config.dropout) for _ in range(2)]
        )
        self.context_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_dropout = nn.Dropout(config.dropout)
        self.context_norm = LayerNorm(config.hidden_size)
        self.decode_edge_embedding = nn.Linear(config.edge_size, config.hidden_size)
        self.sidechain_node_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder_node_projection = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            [
                MpnnDecoderBlock(config.hidden_size, config.hidden_size * 3, dropout=config.dropout)
                for _ in range(config.num_decoder_blocks)
            ]
        )
        self.torsion_projection = nn.Linear(config.hidden_size, 4 * 3 * config.num_mix)
        self.softplus = nn.Softplus(beta=1, threshold=20)
        self.gradient_checkpointing = False

        self.post_init()

        buffer_device = self.torsion_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        buffer_dtype = self.torsion_projection.weight.dtype
        self.register_buffer("packer_residue", PACKER_RESIDUE.to(buffer_device), persistent=False)
        component_ids = (*AMINO_ACID_COMPONENT_IDS, "UNK")
        standard_amino_acid_order = [
            rc.AMINO_ACID_ONE_LETTER_CODES.index(letter) for letter in MPNN_PACKER_VOCABULARY[:-1]
        ]
        amino_acid_order = [*standard_amino_acid_order, len(rc.AMINO_ACID_ONE_LETTER_CODES)]

        amino_acid_atom37_mask = []
        for component_id in AMINO_ACID_COMPONENT_IDS:
            atom_names = {atom[0] for atom in rc.rigid_group_atom_positions[component_id]}
            amino_acid_atom37_mask.append([atom_name in atom_names for atom_name in rc.atom_types])
        amino_acid_atom37_mask.append([False] * len(rc.atom_types))

        self.register_buffer(
            "amino_acid_atom14_mask",
            torch.tensor(
                rc.AMINO_ACID_ATOM14_MASK[amino_acid_order],
                dtype=torch.bool,
                device=buffer_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_atom37_mask",
            torch.tensor(amino_acid_atom37_mask, dtype=torch.bool, device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_atom14_to_atom37",
            torch.tensor(
                [
                    [
                        rc.atom_order[atom_name] if atom_name else 0
                        for atom_name in rc.AMINO_ACID_ATOM14_NAMES[component_id]
                    ]
                    for component_id in component_ids
                ],
                dtype=torch.long,
                device=buffer_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_chi_atom_indices",
            torch.tensor(rc.AMINO_ACID_CHI_ATOM_INDICES[amino_acid_order], dtype=torch.long, device=buffer_device),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_chi_angles_mask",
            torch.tensor(
                [
                    *[rc.chi_angles_mask[amino_acid_index] for amino_acid_index in standard_amino_acid_order],
                    [False, False, False, False],
                ],
                dtype=torch.bool,
                device=buffer_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_chi_pi_periodic",
            torch.tensor(
                [
                    *[rc.chi_pi_periodic[amino_acid_index] for amino_acid_index in standard_amino_acid_order],
                    [False, False, False, False],
                ],
                dtype=torch.bool,
                device=buffer_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_rigid_group_default_frame",
            torch.tensor(
                rc.AMINO_ACID_RIGID_GROUP_DEFAULT_FRAME[amino_acid_order],
                dtype=buffer_dtype,
                device=buffer_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_atom14_to_rigid_group",
            torch.tensor(
                rc.AMINO_ACID_ATOM14_TO_RIGID_GROUP[amino_acid_order],
                dtype=torch.long,
                device=buffer_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "amino_acid_atom14_rigid_group_positions",
            torch.tensor(
                rc.AMINO_ACID_ATOM14_RIGID_GROUP_POSITIONS[amino_acid_order],
                dtype=buffer_dtype,
                device=buffer_device,
            ),
            persistent=False,
        )

    _gradient_checkpointing_func: Callable[..., Any]

    def initialize_parameters(self) -> None:
        r"""
        Initialize matrix parameters with Xavier-uniform values.
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        input: MpnnPackerInput,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Encode an atom14 state as residue graph representations.

        Args:
            input: MPNN Packer tensors with batch size ``B`` and ``L`` residues.
            atom14_coordinates: Current atom14 coordinates with shape ``(B, L, 14, 3)``.
            atom14_mask: Availability of the current atom14 coordinates ``(B, L, 14)``.
            generator: Random generator used for configured training-time
                coordinate noise.

        Returns:
            node_states: Node states with shape ``(B, L, H)``.
            edge_states: Edge states with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, L, K)``.
        """

        input = normalize_mpnn_packer_input(input, dtype=self.edge_embedding.weight.dtype)
        residue_mask = input.residue_mask
        node_features, edges, neighbor_indices, neighbor_valid, atom_nodes, atom_edges, context_nodes, context_mask = (
            self.feature_encoder.encode(
                atom14_coordinates=atom14_coordinates,
                atom14_mask=atom14_mask,
                sequence_tokens=input.sequence_tokens,
                residue_mask=residue_mask,
                residue_index=input.residue_index,
                chain_index=input.chain_index,
                context_coordinates=input.context_coordinates,
                context_atomic_numbers=input.context_atomic_numbers,
                context_mask=input.context_mask,
                generator=generator,
            )
        )
        context = self.context_edge_embedding(context_nodes)
        node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edges)
        attend_mask = residue_mask.unsqueeze(-1) & neighbor_valid
        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node_states, edge_states = self._gradient_checkpointing_func(
                    block.__call__,
                    node_states,
                    edge_states,
                    neighbor_indices,
                    residue_mask,
                    attend_mask,
                )
            else:
                node_states, edge_states = block(
                    node_states,
                    edge_states,
                    neighbor_indices,
                    residue_mask,
                    attend_mask,
                )

        residue_context = self.residue_context_input(node_states)
        atom_pair_mask = context_mask[:, :, :, None] & context_mask[:, :, None, :]
        atom_nodes = self.atom_node_embedding(atom_nodes)
        atom_edges = self.atom_edge_embedding(atom_edges)
        for block_index, block in enumerate(self.context_blocks):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                atom_nodes = self._gradient_checkpointing_func(
                    self.atom_context_blocks[block_index].__call__,
                    atom_nodes,
                    atom_edges,
                    context_mask,
                    atom_pair_mask,
                )
                residue_context = self._gradient_checkpointing_func(
                    block.__call__,
                    residue_context,
                    torch.cat([context, atom_nodes], dim=-1),
                    residue_mask,
                    context_mask,
                )
            else:
                atom_nodes = self.atom_context_blocks[block_index](
                    atom_nodes,
                    atom_edges,
                    context_mask,
                    atom_pair_mask,
                )
                residue_context = block(
                    residue_context,
                    torch.cat([context, atom_nodes], dim=-1),
                    residue_mask,
                    context_mask,
                )
        residue_context = self.context_output(residue_context)
        node_states = node_states + self.context_norm(self.context_dropout(residue_context))
        return node_states, edge_states, neighbor_indices, neighbor_valid

    def decode(
        self,
        input: MpnnPackerInput,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
        neighbor_valid: Tensor,
    ) -> MpnnPackerOutput:
        r"""
        Predict side-chain torsion mixtures from encoded graph states.

        Args:
            input: MPNN Packer tensors with batch size ``B`` and ``L`` residues.
            atom14_coordinates: Current atom14 coordinates with shape ``(B, L, 14, 3)``.
            atom14_mask: Valid-atom mask with shape ``(B, L, 14)``.
            node_states: Encoded residue states with shape ``(B, L, H)``.
            edge_states: Encoded edge states with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, L, K)``.

        Returns:
            Four torsion-mixture distributions per residue.
        """

        input = normalize_mpnn_packer_input(input, dtype=self.edge_embedding.weight.dtype)
        residue_mask = input.residue_mask
        sidechain_nodes, decode_edges = self.feature_encoder.decode(
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            sequence_tokens=input.sequence_tokens,
            residue_mask=residue_mask,
            neighbor_indices=neighbor_indices,
            context_coordinates=input.context_coordinates,
            context_atomic_numbers=input.context_atomic_numbers,
            context_mask=input.context_mask,
        )
        decode_edges = self.decode_edge_embedding(decode_edges)
        edge_context = torch.cat([edge_states, decode_edges], dim=-1)
        sidechain_states = self.sidechain_node_embedding(sidechain_nodes)
        node_states = self.decoder_node_projection(torch.cat([node_states, sidechain_states], dim=-1))
        attend_mask = residue_mask.unsqueeze(-1) & neighbor_valid

        for block in self.decoder_blocks:
            neighbor_states = cat_neighbors_nodes(node_states, edge_context, neighbor_indices)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                node_states = self._gradient_checkpointing_func(
                    block.__call__,
                    node_states,
                    neighbor_states,
                    residue_mask,
                    attend_mask,
                )
            else:
                node_states = block(node_states, neighbor_states, residue_mask, attend_mask)

        torsions = self.torsion_projection(node_states)
        torsions = torsions.unflatten(-1, (4, self.num_mix, 3))
        return MpnnPackerOutput(
            mean=torsions[..., 0],
            concentration=0.1 + self.softplus(torsions[..., 1]),
            mix_logits=torsions[..., 2],
            batch_size=[torsions.shape[0]],
        )

    def forward(
        self,
        input: MpnnPackerInput,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> MpnnPackerOutput:
        r"""
        Predict side-chain torsion mixtures for an atom14 state.

        Args:
            input: MPNN Packer tensors with batch size ``B`` and ``L`` residues.
            atom14_coordinates: Current atom14 coordinates with shape ``(B, L, 14, 3)``.
            atom14_mask: Valid-atom mask with shape ``(B, L, 14)``.
            generator: Random generator used for configured training-time
                coordinate noise.

        Returns:
            Mixture means, concentrations, and logits with shape
                ``(B, L, 4, num_mixtures)``.

        Notes:
            The returned torsion-mixture parameters support caller-defined differentiable losses. This family
            provides no aggregate packing-loss function. ``encode`` and ``forward`` use the supplied generator for
            configured coordinate noise; inference sampling advances the generator before torsion draws.
        """

        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            generator=generator,
        )
        return self.decode(
            input,
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            node_states=node_states,
            edge_states=edge_states,
            neighbor_indices=neighbor_indices,
            neighbor_valid=neighbor_valid,
        )

    @torch.inference_mode()
    def sample(
        self,
        input: MpnnPackerInput,
        *,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        generator: torch.Generator | None = None,
    ) -> MpnnPackerSamplingOutput:
        r"""
        Sample one packed atom14 state for each input batch item.

        Args:
            input: MPNN Packer tensors with batch size ``B`` and ``L`` residues.
            num_denoising_steps: Number of iterative side-chain updates.
            num_torsion_samples: Number of torsion proposals evaluated per update.
            generator: Random generator used for configured coordinate noise and
                torsion sampling.

        Returns:
            Packed atom14 coordinates and masks, sampled chi angles, their log
                densities, and the final torsion-mixture parameters.
        """

        if num_denoising_steps <= 0:
            fail("value_invalid", RequestRef("num_denoising_steps", str(num_denoising_steps)))
        if num_torsion_samples <= 0:
            fail("value_invalid", RequestRef("num_torsion_samples", str(num_torsion_samples)))

        input = normalize_mpnn_packer_input(input, dtype=self.edge_embedding.weight.dtype)
        device = input.sequence_tokens.device
        batch_size, num_residues = input.residue_mask.shape
        amino_acid_index = self.packer_residue[input.sequence_tokens]
        supported_geometry = ~(input.residue_mask & input.pack_mask & (amino_acid_index == _UNKNOWN)).any()
        if torch.compiler.is_compiling():
            torch._assert_async(supported_geometry, "selected packing residues require a supported geometry template")
        elif not bool(supported_geometry):
            fail("model_capacity", RequestRef("pack_mask", "a packed residue without a geometry template"))
        backbone_atom37_coordinates = torch.zeros(
            batch_size,
            num_residues,
            37,
            3,
            device=device,
            dtype=input.backbone_coordinates.dtype,
        )
        backbone_atom37_coordinates[:, :, :3] = input.backbone_coordinates[:, :, :3]
        backbone_atom37_coordinates[:, :, 4] = input.backbone_coordinates[:, :, 3]

        atom14_exists = self.amino_acid_atom14_mask[amino_acid_index]
        atom37_exists = self.amino_acid_atom37_mask[amino_acid_index]
        atom14_to_atom37 = self.amino_acid_atom14_to_atom37[amino_acid_index]
        default_frames = self.amino_acid_rigid_group_default_frame[amino_acid_index]
        atom14_tables = {
            "atom14_to_rigid_group": self.amino_acid_atom14_to_rigid_group[amino_acid_index],
            "atom14_rigid_group_positions": self.amino_acid_atom14_rigid_group_positions[amino_acid_index],
            "atom14_mask": atom14_exists,
        }
        torsion_sin_cos, _, _ = atom37_to_torsion_angles(
            backbone_atom37_coordinates,
            atom37_exists,
            chi_atom_indices=self.amino_acid_chi_atom_indices[amino_acid_index],
            chi_angles_mask=self.amino_acid_chi_angles_mask[amino_acid_index],
            chi_pi_periodic=self.amino_acid_chi_pi_periodic[amino_acid_index],
            backend=self.backend,
        )
        nitrogen, alpha_carbon, carbon = backbone_atom37_coordinates[:, :, :3].unbind(dim=-2)
        x_direction = carbon.double() - alpha_carbon.double()
        xy_direction = nitrogen.double() - alpha_carbon.double()
        finite_directions = torch.isfinite(x_direction).all() & torch.isfinite(xy_direction).all()
        if torch.compiler.is_compiling():
            torch._assert_async(finite_directions, "packing backbone directions exceed the numeric range")
        elif not bool(finite_directions):
            fail("model_numeric", RequestRef("backbone", "directions exceed the numeric range"))
        backbone_rotation = model_frame_rotation(
            x_direction,
            xy_direction,
            input.residue_mask,
            feature="packing backbone frames",
        ).to(alpha_carbon.dtype)

        input_torsion_sin_cos, _, _ = atom37_to_torsion_angles(
            input.atom37_coordinates,
            input.atom37_mask,
            chi_atom_indices=self.amino_acid_chi_atom_indices[amino_acid_index],
            chi_angles_mask=self.amino_acid_chi_angles_mask[amino_acid_index],
            chi_pi_periodic=self.amino_acid_chi_pi_periodic[amino_acid_index],
            backend=self.backend,
        )
        input_chi_sin_cos = input_torsion_sin_cos[:, :, 3:].clone()
        pack_mask = input.pack_mask[:, :, None, None]
        random_angles = (
            2
            * torch.pi
            * torch.rand(
                batch_size,
                num_residues,
                4,
                device=device,
                dtype=input.backbone_coordinates.dtype,
                generator=generator,
            )
        )
        random_sin_cos = torch.stack(
            [torch.sin(random_angles), torch.cos(random_angles)],
            dim=-1,
        )
        torsion_sin_cos = torsion_sin_cos.clone()
        torsion_sin_cos[:, :, 3:] = torch.where(pack_mask, random_sin_cos, input_chi_sin_cos)

        frames = torsion_angles_to_frames(
            backbone_rotation,
            alpha_carbon,
            torsion_sin_cos,
            default_frames=default_frames,
        )
        atom14_coordinates = frames_to_atom14_positions(*frames, **atom14_tables)
        observed_atom14_mask = torch.gather(input.atom37_mask, 2, atom14_to_atom37) & atom14_exists
        observed_atom14_coordinates = torch.gather(
            input.atom37_coordinates,
            2,
            atom14_to_atom37[:, :, :, None].expand(-1, -1, -1, 3),
        )
        fixed_unknown = ~input.pack_mask & (amino_acid_index == _UNKNOWN)
        observed_atom14_mask[:, :, :4] = torch.where(
            fixed_unknown[:, :, None],
            input.atom37_mask[:, :, [0, 1, 2, 4]],
            observed_atom14_mask[:, :, :4],
        )
        observed_atom14_coordinates[:, :, :4] = torch.where(
            fixed_unknown[:, :, None, None],
            input.atom37_coordinates[:, :, [0, 1, 2, 4]],
            observed_atom14_coordinates[:, :, :4],
        )
        atom14_mask = torch.where(
            input.pack_mask[:, :, None],
            atom14_exists,
            observed_atom14_mask,
        )
        atom14_mask = atom14_mask & input.residue_mask[:, :, None]
        partial_fixed = ~input.pack_mask & (observed_atom14_mask != atom14_exists).any(dim=-1)
        atom14_coordinates = torch.where(
            partial_fixed[:, :, None, None],
            observed_atom14_coordinates,
            atom14_coordinates,
        )
        atom14_coordinates = atom14_coordinates * atom14_mask[:, :, :, None]

        node_states, edge_states, neighbor_indices, neighbor_valid = self.encode(
            input,
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            generator=generator,
        )
        for _ in range(num_denoising_steps):
            output = self.decode(
                input,
                atom14_coordinates=atom14_coordinates,
                atom14_mask=atom14_mask,
                node_states=node_states,
                edge_states=edge_states,
                neighbor_indices=neighbor_indices,
                neighbor_valid=neighbor_valid,
            )
            predicted_distribution = distributions.MixtureSameFamily(
                distributions.Categorical(logits=output.mix_logits),
                distributions.VonMises(output.mean, output.concentration),
            )
            candidates = _sample_von_mises_mixture(
                output.mean,
                output.concentration,
                output.mix_logits,
                num_samples=num_torsion_samples,
                generator=generator,
            )
            candidate_log_densities = predicted_distribution.log_prob(candidates)
            best = torch.argmax(candidate_log_densities, dim=0)[None]
            sampled_chi_angles = torch.gather(candidates, dim=0, index=best)[0]

            sampled_chi_sin_cos = torch.stack(
                [torch.sin(sampled_chi_angles), torch.cos(sampled_chi_angles)],
                dim=-1,
            )
            torsion_sin_cos[:, :, 3:] = torch.where(pack_mask, sampled_chi_sin_cos, input_chi_sin_cos)
            frames = torsion_angles_to_frames(
                backbone_rotation,
                alpha_carbon,
                torsion_sin_cos,
                default_frames=default_frames,
            )
            atom14_coordinates = frames_to_atom14_positions(*frames, **atom14_tables)
            atom14_coordinates = torch.where(
                partial_fixed[:, :, None, None],
                observed_atom14_coordinates,
                atom14_coordinates,
            )
            atom14_coordinates = atom14_coordinates * atom14_mask[:, :, :, None]

        return MpnnPackerSamplingOutput(
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            sampled_chi_angles=sampled_chi_angles,
            chi_log_densities=predicted_distribution.log_prob(sampled_chi_angles),
            mean=output.mean,
            concentration=output.concentration,
            mix_logits=output.mix_logits,
            input_chi_sin_cos=input_chi_sin_cos,
            batch_size=[batch_size],
        )


class MpnnPackerFeatureEncoder(nn.Module):
    r"""
    Build residue-graph and atom-context features for MPNN Packer.

    Args:
        node_size: Width of the per-node features.
        edge_size: Width of the per-edge features.
        positional_embedding_size: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
        num_neighbors: Number of nearest-neighbor residues in the graph.
        num_context_atoms: Number of ligand context atoms per residue used by the decoder.
        coordinate_noise: Training-time Gaussian coordinate-noise standard deviation.
        lower_bound: Lower distance bound for the context RBF expansion.
        upper_bound: Upper distance bound for the context RBF expansion.
        backend: Backend used by neighbour selection and geometry.
    """

    periodic_group: Tensor
    periodic_period: Tensor

    def __init__(
        self,
        *,
        node_size: int,
        edge_size: int,
        positional_embedding_size: int,
        num_rbf: int,
        num_neighbors: int,
        num_context_atoms: int,
        coordinate_noise: float,
        lower_bound: float,
        upper_bound: float,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = resolve_config_backend(backend, supported=("oops", "eager"), public_default="eager")
        self.num_rbf = num_rbf
        self.num_neighbors = num_neighbors
        self.num_context_atoms = num_context_atoms
        self.coordinate_noise = coordinate_noise
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.relative_position = MpnnRelativePositionEncoding(positional_embedding_size)

        self.edge_projection = nn.Linear(positional_embedding_size + num_rbf * 25, edge_size, bias=False)
        self.edge_norm = LayerNorm(edge_size, backend=backend)
        self.node_projection = nn.Linear(len(TOKENS), node_size, bias=False)
        self.node_norm = LayerNorm(node_size, backend=backend)

        buffer_device = self.edge_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        group = torch.tensor(GROUP_BY_ATOMIC_NUMBER, dtype=torch.long, device=buffer_device)
        period = torch.tensor(PERIOD_BY_ATOMIC_NUMBER, dtype=torch.long, device=buffer_device)
        self.register_buffer("periodic_group", group, persistent=False)
        self.register_buffer("periodic_period", period, persistent=False)

        self.element_projection = nn.Linear(147, 64)
        self.context_node_projection = nn.Linear(5 * num_rbf + 64 + 4, node_size)
        self.context_node_norm = LayerNorm(node_size, backend=backend)
        self.atom_edge_projection = nn.Linear(num_rbf, node_size, bias=False)
        self.atom_edge_norm = LayerNorm(node_size, backend=backend)
        self.atom_node_projection = nn.Linear(147, node_size, bias=False)
        self.atom_node_norm = LayerNorm(node_size, backend=backend)

        self.context_down_projection = nn.Linear(num_rbf + 120, num_rbf)
        self.decode_node_projection = nn.Linear(_ATOM14 * num_context_atoms * num_rbf, node_size, bias=False)
        self.decode_node_norm = LayerNorm(node_size, backend=backend)
        self.decode_edge_projection = nn.Linear(
            num_rbf * _ATOM14 * _ATOM14 + 2 * len(TOKENS),
            edge_size,
            bias=False,
        )
        self.decode_edge_norm = LayerNorm(edge_size, backend=backend)

    def _context_rbf(self, distances: Tensor) -> Tensor:
        r"""
        RBF-expand distances over the configured ``[lower_bound, upper_bound]`` range.
        """

        return radial_basis(distances, self.num_rbf, self.lower_bound, self.upper_bound)

    def _element_features(self, atomic_numbers: Tensor, *, dtype: torch.dtype) -> Tensor:
        r"""
        One-hot-encode atoms by element, periodic group, and period `[..., 147]`.
        """

        atomic_numbers = atomic_numbers.long()
        group = self.periodic_group[atomic_numbers]
        period = self.periodic_period[atomic_numbers]
        return torch.cat([F.one_hot(atomic_numbers, 120), F.one_hot(group, 19), F.one_hot(period, 8)], dim=-1).to(
            dtype=dtype
        )

    def encode(
        self,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
        sequence_tokens: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build residue-graph and atom-context encoder features.

        Args:
            atom14_coordinates: Current atom14 coordinates ``[B, L, 14, 3]``.
            atom14_mask: Current coordinate availability ``[B, L, 14]``.
            sequence_tokens: Residue tokens ``[B, L]``.
            residue_mask: Valid-residue mask ``[B, L]``.
            residue_index: Residue positions used by relative-position encoding ``[B, L]``.
            chain_index: Chain indices ``[B, L]``.
            context_coordinates: Per-residue non-protein context coordinates ``[B, L, A, 3]``.
            context_atomic_numbers: Atomic numbers for the context coordinates ``[B, L, A]``.
            context_mask: Valid-context mask ``[B, L, A]``.
            generator: Random generator used for configured training-time
                coordinate noise.

        Returns:
            nodes: Residue nodes with shape ``(B, L, H)``.
            edges: Residue edges with shape ``(B, L, K, H)``.
            neighbor_indices: Neighbor indices with shape ``(B, L, K)``.
            neighbor_valid: Real-neighbor mask with shape ``(B, L, K)``.
            atom_nodes: Atom nodes with shape ``(B, L, A, H)``.
            atom_edges: Atom edges with shape ``(B, L, A, A, H)``.
            context_nodes: Residue-to-context features with shape ``(B, L, A, H)``.
            context_mask: Context mask with shape ``(B, L, A)``.
        """

        sequence_tokens = _packing_sequence_tokens(sequence_tokens, residue_mask)
        atom_mask = atom14_mask & residue_mask[:, :, None]
        complete_backbone = atom_mask[..., :4].all(dim=-1) | ~residue_mask
        if torch.compiler.is_compiling():
            torch._assert_async(complete_backbone.all(), "packing encoder requires available N, CA, C and O")
        elif not bool(complete_backbone.all()):
            fail("value_invalid", RequestRef("atom14_mask", "an active residue without N, CA, C and O"))
        coordinates = project_model_coordinates(
            atom14_coordinates,
            atom_mask,
            dtype=self.edge_projection.weight.dtype,
            feature="packing atom14 state",
        )
        context_mask = context_mask & residue_mask[:, :, None]
        context_coordinates = project_model_coordinates(
            context_coordinates,
            context_mask,
            dtype=coordinates.dtype,
            feature="packing context",
        )
        if self.training and self.coordinate_noise > 0:
            noise = torch.empty_like(coordinates).normal_(generator=generator)
            coordinates = coordinates + self.coordinate_noise * noise
        context_atomic_numbers = project_model_categories(
            context_atomic_numbers,
            context_mask,
            num_classes=min(_ELEMENT_CLASSES, self.periodic_group.numel(), self.periodic_period.numel()),
            feature="packing context elements",
        )

        n, ca, c, o = (
            coordinates[:, :, 0],
            coordinates[:, :, 1],
            coordinates[:, :, 2],
            coordinates[:, :, 3],
        )
        cb = virtual_beta_carbon(n, ca, c)
        _, neighbor_indices, neighbor_valid = masked_knn(ca, residue_mask, self.num_neighbors, backend=self.backend)

        backbone_atoms = torch.stack((n, ca, c, o, cb), dim=2)
        pair_distances = neighbor_atom_distances(backbone_atoms, neighbor_indices, eps=1e-6, backend=self.backend)
        pair_rbf = radial_basis(
            pair_distances,
            self.num_rbf,
            self.lower_bound,
            self.upper_bound,
        ).flatten(start_dim=-3)
        neighbor_residue_index = gather_nodes(residue_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        offset = residue_index[:, :, None] - neighbor_residue_index
        neighbor_chain = gather_nodes(chain_index.unsqueeze(-1), neighbor_indices).squeeze(-1)
        same_chain = (chain_index[:, :, None] == neighbor_chain).long()
        edges = self.edge_norm(
            self.edge_projection(
                torch.cat([self.relative_position(offset.long(), same_chain), pair_rbf], dim=-1).to(
                    self.edge_projection.weight.dtype
                )
            )
        )
        nodes = self.node_norm(
            self.node_projection(F.one_hot(sequence_tokens.long(), len(TOKENS)).to(self.node_projection.weight.dtype))
        )

        element_features = self._element_features(context_atomic_numbers, dtype=coordinates.dtype)
        element_projection = self.element_projection(element_features.to(self.element_projection.weight.dtype))
        context_distances = smoothed_vector_norm(
            backbone_atoms[:, :, :, None] - context_coordinates[:, :, None],
            eps=1e-6,
            backend=self.backend,
        )
        context_features = self._context_rbf(context_distances.movedim(2, 3)).flatten(start_dim=-2)
        angle_features = local_backbone_frame_features(n, ca, c, context_coordinates, backend=self.backend)
        context = torch.cat([context_features, element_projection, angle_features], dim=-1)
        context_nodes = self.context_node_norm(
            self.context_node_projection(context.to(self.context_node_projection.weight.dtype))
        )

        atom_pair_distances = smoothed_vector_norm(
            context_coordinates[:, :, :, None] - context_coordinates[:, :, None, :],
            eps=1e-6,
            backend=self.backend,
        )
        atom_edges = self.atom_edge_norm(
            self.atom_edge_projection(self._context_rbf(atom_pair_distances).to(self.atom_edge_projection.weight.dtype))
        )
        atom_nodes = self.atom_node_norm(
            self.atom_node_projection(element_features.to(self.atom_node_projection.weight.dtype))
        )
        return nodes, edges, neighbor_indices, neighbor_valid, atom_nodes, atom_edges, context_nodes, context_mask

    def decode(
        self,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
        sequence_tokens: Tensor,
        residue_mask: Tensor,
        neighbor_indices: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Build atom14-aware decoder features.

        Args:
            atom14_coordinates: Current atom14 coordinates ``[B, L, 14, 3]``.
            atom14_mask: Current atom14 mask ``[B, L, 14]``.
            sequence_tokens: Residue tokens ``[B, L]``.
            residue_mask: Valid-residue mask ``[B, L]``.
            neighbor_indices: Encoded residue-neighbor indices ``[B, L, K]``.
            context_coordinates: Per-residue non-protein context coordinates ``[B, L, A, 3]``.
            context_atomic_numbers: Atomic numbers for the context coordinates ``[B, L, A]``.
            context_mask: Valid-context mask ``[B, L, A]``.

        Returns:
            sidechain_nodes: Side-chain node features with shape ``(B, L, H)``.
            decode_edges: Decoder edge features with shape ``(B, L, K, H)``.
        """

        sequence_tokens = _packing_sequence_tokens(sequence_tokens, residue_mask)
        atom_mask = atom14_mask & residue_mask[:, :, None]
        coordinates = project_model_coordinates(
            atom14_coordinates,
            atom_mask,
            dtype=self.decode_edge_projection.weight.dtype,
            feature="packing atom14 state",
        )
        context_coordinates = context_coordinates[:, :, : self.num_context_atoms]
        context_mask = context_mask[:, :, : self.num_context_atoms]
        context_mask = context_mask & residue_mask[:, :, None]
        context_coordinates = project_model_coordinates(
            context_coordinates,
            context_mask,
            dtype=coordinates.dtype,
            feature="packing context",
        )
        context_atomic_numbers = project_model_categories(
            context_atomic_numbers[:, :, : self.num_context_atoms],
            context_mask,
            num_classes=_ELEMENT_CLASSES,
            feature="packing context elements",
        )

        gathered_mask = gather_nodes(atom_mask, neighbor_indices)
        sidechain_distances = neighbor_atom_distances(coordinates, neighbor_indices, eps=1e-6, backend=self.backend)
        pair_mask = atom_mask[:, :, None, :, None] & gathered_mask[:, :, :, None, :]
        sidechain_rbf = radial_basis(
            sidechain_distances,
            self.num_rbf,
            self.lower_bound,
            self.upper_bound,
        )
        sidechain_rbf = (sidechain_rbf * pair_mask[..., None]).flatten(start_dim=-3)

        atom_context_distances = smoothed_vector_norm(
            coordinates[:, :, :, None, :] - context_coordinates[:, :, None, :, :],
            eps=1e-6,
            backend=self.backend,
        )
        atom_context_features = self._context_rbf(atom_context_distances)
        atom_context_features = (
            atom_context_features * atom_mask[:, :, :, None, None] * context_mask[:, :, None, :, None]
        )
        context_types_1hot = F.one_hot(context_atomic_numbers, 120).to(dtype=coordinates.dtype)
        combined = torch.cat(
            [atom_context_features, context_types_1hot[:, :, None].expand(-1, -1, _ATOM14, -1, -1)], dim=-1
        )
        node_features = self.context_down_projection(combined.to(self.context_down_projection.weight.dtype)).flatten(
            start_dim=2
        )
        sidechain_nodes = self.decode_node_norm(self.decode_node_projection(node_features))

        sequence_1hot = F.one_hot(sequence_tokens.long(), len(TOKENS)).to(dtype=coordinates.dtype)
        gathered_sequence = gather_nodes(sequence_1hot, neighbor_indices)
        sequence_features = torch.cat(
            [
                sequence_1hot[:, :, None].expand(-1, -1, neighbor_indices.shape[2], -1),
                gathered_sequence,
            ],
            dim=-1,
        )
        decode_edges = torch.cat([sidechain_rbf, sequence_features], dim=-1)
        decode_edges = self.decode_edge_norm(
            self.decode_edge_projection(decode_edges.to(self.decode_edge_projection.weight.dtype))
        )
        return sidechain_nodes, decode_edges


def _sample_von_mises_mixture(
    mean: Tensor,
    concentration: Tensor,
    mix_logits: Tensor,
    *,
    num_samples: int,
    generator: torch.Generator | None,
) -> Tensor:
    r"""
    Match PyTorch's mixture/Von-Mises sampling while accepting a Generator.
    """

    probabilities = mix_logits.softmax(dim=-1).reshape(-1, mix_logits.shape[-1])
    mixture_indices = torch.multinomial(
        probabilities,
        num_samples,
        replacement=True,
        generator=generator,
    ).T.reshape(num_samples, *mix_logits.shape[:-1])

    loc = mean.to(torch.double)
    kappa = concentration.to(torch.double)
    tau = 1 + torch.sqrt(1 + 4 * kappa**2)
    rho = (tau - torch.sqrt(2 * tau)) / (2 * kappa)
    proposal = (1 + rho**2) / (2 * rho)
    proposal = torch.where(kappa < 1e-5, 1 / kappa + kappa, proposal)
    shape = (num_samples, *mean.shape)
    component_samples = torch.empty(shape, dtype=torch.double, device=mean.device)
    done = torch.zeros(shape, dtype=torch.bool, device=mean.device)
    while not bool(done.all()):
        uniform = torch.rand((3, *shape), dtype=torch.double, device=mean.device, generator=generator)
        first, second, third = uniform.unbind()
        z = torch.cos(math.pi * first)
        f = (1 + proposal * z) / (proposal + z)
        c = kappa * (proposal - f)
        accept = ((c * (2 - c) - second) > 0) | ((c / second).log() + 1 - c >= 0)
        component_samples = torch.where(accept, (third - 0.5).sign() * f.acos(), component_samples)
        done = done | accept
    component_samples = ((component_samples + math.pi + loc) % (2 * math.pi) - math.pi).to(mean.dtype)
    return torch.gather(component_samples, -1, mixture_indices.unsqueeze(-1)).squeeze(-1)


# ===========================================================================
# MpnnPacker tensor inputs and outputs.
# ===========================================================================


@tensorclass
class MpnnPackerInput:
    r"""
    Inputs for side-chain packing.

    Attributes:
        backbone_coordinates: N, CA, C, and O coordinates with shape
            ``(batch_size, num_residues, 4, 3)``.
        sequence_tokens: Canonical classes ``(batch_size, num_residues)``; a class outside the 20 amino acids is
            unknown context with no packing geometry template. Inactive payload is uninterpreted.
        residue_mask: Valid-residue mask with shape ``(batch_size, num_residues)``.
        residue_index: Residue indices with shape ``(batch_size, num_residues)``.
        chain_index: Chain indices with shape ``(batch_size, num_residues)``.
        context_coordinates: Nearby context-atom coordinates with shape
            ``(batch_size, num_residues, num_context_atoms, 3)``.
        context_atomic_numbers: Context atomic numbers with shape
            ``(batch_size, num_residues, num_context_atoms)``.
        context_mask: Context-atom mask with shape
            ``(batch_size, num_residues, num_context_atoms)``.
        atom37_coordinates: Observed atom37 coordinates with shape
            ``(batch_size, num_residues, 37, 3)``.
        atom37_mask: Observed atom37 mask with shape
            ``(batch_size, num_residues, 37)``.
        pack_mask: Residues selected for packing with shape
            ``(batch_size, num_residues)``.
        atom14_atom_index: Prepared canonical atom rows, or -1, ``[B, L, 14]``.
        atom14_chi_index: Prepared chi-group indices, or -1, ``[B, L, 14]``.
        chi_mask: Supported torsions ``[B, L, 4]``.
        component_index: Canonical component indices aligned to residue rows with
            shape ``(batch_size, num_residues)``.
    """

    backbone_coordinates: Tensor
    sequence_tokens: Tensor
    residue_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    context_coordinates: Tensor
    context_atomic_numbers: Tensor
    context_mask: Tensor
    atom37_coordinates: Tensor
    atom37_mask: Tensor
    pack_mask: Tensor
    atom14_atom_index: Tensor
    atom14_chi_index: Tensor
    chi_mask: Tensor
    component_index: Tensor


class MpnnPackerOutput(ShadowTensorCollection):
    r"""
    Predicted side-chain torsion mixtures.

    Attributes:
        mean: Von Mises means with shape
            ``(batch_size, num_residues, 4, num_mixtures)``.
        concentration: Positive concentrations with the same shape as ``mean``.
        mix_logits: Mixture logits with the same shape as ``mean``.
    """

    mean: Tensor
    concentration: Tensor
    mix_logits: Tensor


class MpnnPackerSamplingOutput(ShadowTensorCollection):
    r"""
    Outputs from iterative side-chain sampling.

    Attributes:
        atom14_coordinates: Packed atom14 coordinates with shape
            ``(batch_size, num_residues, 14, 3)``.
        atom14_mask: Packed atom14 mask with shape
            ``(batch_size, num_residues, 14)``.
        sampled_chi_angles: Selected chi angles with shape
            ``(batch_size, num_residues, 4)``.
        chi_log_densities: Selected-angle log densities with shape
            ``(batch_size, num_residues, 4)``.
        mean: Final von Mises means with shape
            ``(batch_size, num_residues, 4, num_mixtures)``.
        concentration: Final von Mises concentrations with the same shape as ``mean``.
        mix_logits: Final mixture logits with the same shape as ``mean``.
        input_chi_sin_cos: Input chi-angle sine and cosine values with shape
            ``(batch_size, num_residues, 4, 2)``.
    """

    atom14_coordinates: Tensor
    atom14_mask: Tensor
    sampled_chi_angles: Tensor
    chi_log_densities: Tensor
    mean: Tensor
    concentration: Tensor
    mix_logits: Tensor
    input_chi_sin_cos: Tensor
