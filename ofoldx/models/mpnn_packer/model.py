r"""
MpnnPacker side-chain packing model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn.functional as F
from torch import Tensor, nn

from ...data.molecular import AMINO_ACID_THREE_TO_ONE, AtomSite, ComponentReference
from ...data.structure import Structure
from ...model import Model
from ...modules import residue_constants as rc
from ...modules.backend import LayerNorm
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    MpnnDecoderBlock,
    MpnnEncoderBlock,
    MpnnRelativePositionEncoding,
    cat_neighbors_nodes,
    gather_nodes,
    local_backbone_frame_features,
    periodic_group_period,
    radial_basis,
    virtual_beta_carbon,
)
from ...results import Metric, PackPrediction, PackResult
from .config import MpnnPackerConfig

_ATOM14 = 14


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

    def __init__(self, config: MpnnPackerConfig | None = None) -> None:
        config = MpnnPackerConfig() if config is None else config
        super().__init__(config)
        self.num_mix = config.num_mix
        self.feature_encoder = MpnnPackerFeatures(
            node_size=config.hidden_size,
            edge_size=config.edge_size,
            num_positional_embeddings=config.num_positional_embeddings,
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

        self.post_init()

        buffer_device = self.torsion_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        buffer_dtype = self.torsion_projection.weight.dtype
        component_ids = (*AMINO_ACID_COMPONENT_IDS, "UNK")
        standard_amino_acid_order = [
            rc.AMINO_ACID_ONE_LETTER_CODES.index(AMINO_ACID_THREE_TO_ONE[component_id])
            for component_id in AMINO_ACID_COMPONENT_IDS
        ]
        amino_acid_order = [*standard_amino_acid_order, len(rc.AMINO_ACID_ONE_LETTER_CODES)]

        amino_acid_atom37_mask = []
        amino_acid_chi_atom_indices = []
        for component_id in AMINO_ACID_COMPONENT_IDS:
            atom_names = {atom[0] for atom in rc.rigid_group_atom_positions[component_id]}
            amino_acid_atom37_mask.append([atom_name in atom_names for atom_name in rc.atom_types])
            chi_atom_indices = [
                [rc.atom_order[atom_name] for atom_name in chi] for chi in rc.chi_angles_atoms[component_id]
            ]
            amino_acid_chi_atom_indices.append(chi_atom_indices + [[0, 0, 0, 0]] * (4 - len(chi_atom_indices)))
        amino_acid_atom37_mask.append([False] * len(rc.atom_types))
        amino_acid_chi_atom_indices.append([[0, 0, 0, 0]] * 4)

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
            torch.tensor(amino_acid_chi_atom_indices, dtype=torch.long, device=buffer_device),
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Encode an atom14 state as residue graph representations.

        Args:
            input: MPNN Packer tensors with batch size ``B`` and ``L`` residues.
            atom14_coordinates: Current atom14 coordinates with shape ``(B, L, 14, 3)``.

        Returns:
            Node states with shape ``(B, L, H)``, edge states with shape
            ``(B, L, K, H)``, and neighbor indices with shape ``(B, L, K)``.
        """

        residue_mask = input.residue_mask
        node_features, edges, neighbor_indices, atom_nodes, atom_edges, context_nodes, context_mask = (
            self.feature_encoder.encode(
                atom14_coordinates=atom14_coordinates,
                sequence_tokens=input.sequence_tokens,
                residue_mask=residue_mask,
                residue_index=input.residue_index,
                chain_index=input.chain_index,
                context_coordinates=input.context_coordinates,
                context_atomic_numbers=input.context_atomic_numbers,
                context_mask=input.context_mask,
            )
        )
        context = self.context_edge_embedding(context_nodes)
        node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edges)
        attend_mask = gather_nodes(residue_mask.unsqueeze(-1), neighbor_indices).squeeze(-1)
        attend_mask = residue_mask.unsqueeze(-1) & attend_mask
        for block in self.encoder_blocks:
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
        return node_states, edge_states, neighbor_indices

    def decode(
        self,
        input: MpnnPackerInput,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
        node_states: Tensor,
        edge_states: Tensor,
        neighbor_indices: Tensor,
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

        Returns:
            Four torsion-mixture distributions per residue.
        """

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
        attend_mask = gather_nodes(residue_mask.unsqueeze(-1), neighbor_indices).squeeze(-1)
        attend_mask = residue_mask.unsqueeze(-1) & attend_mask

        for block in self.decoder_blocks:
            neighbor_states = cat_neighbors_nodes(node_states, edge_context, neighbor_indices)
            node_states = block(node_states, neighbor_states, residue_mask, attend_mask)

        torsions = self.torsion_projection(node_states)
        torsions = torsions.reshape(node_states.shape[0], node_states.shape[1], 4, self.num_mix, 3)
        return MpnnPackerOutput(
            mean=torsions[..., 0],
            concentration=0.1 + self.softplus(torsions[..., 1]),
            mix_logits=torsions[..., 2],
        )

    def forward(
        self,
        input: MpnnPackerInput,
        *,
        atom14_coordinates: Tensor,
        atom14_mask: Tensor,
    ) -> MpnnPackerOutput:
        r"""
        Predict side-chain torsion mixtures for an atom14 state.

        Args:
            input: MPNN Packer tensors with batch size ``B`` and ``L`` residues.
            atom14_coordinates: Current atom14 coordinates with shape ``(B, L, 14, 3)``.
            atom14_mask: Valid-atom mask with shape ``(B, L, 14)``.

        Returns:
            Mixture means, concentrations, and logits with shape
            ``(B, L, 4, num_mixtures)``.
        """

        node_states, edge_states, neighbor_indices = self.encode(input, atom14_coordinates=atom14_coordinates)
        return self.decode(
            input,
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            node_states=node_states,
            edge_states=edge_states,
            neighbor_indices=neighbor_indices,
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
            generator: Random generator used for torsion sampling.

        Returns:
            Packed atom14 coordinates and masks, sampled chi angles, their log
            densities, and the final torsion-mixture parameters.
        """

        if num_denoising_steps <= 0:
            raise ValueError("num_denoising_steps must be positive")
        if num_torsion_samples <= 0:
            raise ValueError("num_torsion_samples must be positive")

        device = input.sequence_tokens.device
        batch_size, length = input.residue_mask.shape
        backbone_atom37_coordinates = torch.zeros(
            batch_size,
            length,
            37,
            3,
            device=device,
            dtype=input.backbone_coordinates.dtype,
        )
        backbone_atom37_coordinates[:, :, :3] = input.backbone_coordinates[:, :, :3]
        backbone_atom37_coordinates[:, :, 4] = input.backbone_coordinates[:, :, 3]

        amino_acid_index = input.sequence_tokens.long()
        atom14_exists = self.amino_acid_atom14_mask[amino_acid_index]
        atom37_exists = self.amino_acid_atom37_mask[amino_acid_index]
        atom14_to_atom37 = self.amino_acid_atom14_to_atom37[amino_acid_index]
        chi_atom_indices = self.amino_acid_chi_atom_indices[amino_acid_index]
        chi_angles_mask = self.amino_acid_chi_angles_mask[amino_acid_index]
        default_frames = self.amino_acid_rigid_group_default_frame[amino_acid_index]
        atom14_to_rigid_group = self.amino_acid_atom14_to_rigid_group[amino_acid_index]
        atom14_rigid_group_positions = self.amino_acid_atom14_rigid_group_positions[amino_acid_index]
        torsion_sin_cos, _ = _atom37_to_torsion_angles(
            backbone_atom37_coordinates,
            atom37_exists,
            chi_atom_indices=chi_atom_indices,
            chi_angles_mask=chi_angles_mask,
        )
        backbone_frames = MpnnPackerFrame.make_transform_from_reference(
            backbone_atom37_coordinates[:, :, 0],
            backbone_atom37_coordinates[:, :, 1],
            backbone_atom37_coordinates[:, :, 2],
            eps=1e-9,
        )

        input_torsion_sin_cos, _ = _atom37_to_torsion_angles(
            input.atom37_coordinates,
            input.atom37_mask,
            chi_atom_indices=chi_atom_indices,
            chi_angles_mask=chi_angles_mask,
        )
        input_chi_sin_cos = input_torsion_sin_cos[:, :, 3:].clone()
        pack_positions = input.pack_positions[:, :, None, None]
        random_angles = (
            2
            * torch.pi
            * torch.rand(
                batch_size,
                length,
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
        torsion_sin_cos[:, :, 3:] = torch.where(pack_positions, random_sin_cos, input_chi_sin_cos)

        frames = _torsion_angles_to_frames(
            backbone_frames,
            torsion_sin_cos,
            default_frames=default_frames,
        )
        atom14_coordinates = _frames_and_literature_positions_to_atom14_pos(
            frames,
            atom14_to_rigid_group=atom14_to_rigid_group,
            atom14_rigid_group_positions=atom14_rigid_group_positions,
            atom14_mask=atom14_exists,
        )
        observed_atom14_mask = torch.gather(input.atom37_mask, 2, atom14_to_atom37) & atom14_exists
        observed_atom14_coordinates = torch.gather(
            input.atom37_coordinates,
            2,
            atom14_to_atom37[:, :, :, None].expand(-1, -1, -1, 3),
        )
        fixed_unknown = ~input.pack_positions & (amino_acid_index == len(AMINO_ACID_COMPONENT_IDS))
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
            input.pack_positions[:, :, None],
            atom14_exists,
            observed_atom14_mask,
        )
        atom14_mask = atom14_mask & input.residue_mask[:, :, None]
        partial_fixed = ~input.pack_positions & (observed_atom14_mask != atom14_exists).any(dim=-1)
        atom14_coordinates = torch.where(
            partial_fixed[:, :, None, None],
            observed_atom14_coordinates,
            atom14_coordinates,
        )
        atom14_coordinates = atom14_coordinates * atom14_mask[:, :, :, None]

        node_states, edge_states, neighbor_indices = self.encode(
            input,
            atom14_coordinates=atom14_coordinates,
        )
        for _ in range(num_denoising_steps):
            output = self.decode(
                input,
                atom14_coordinates=atom14_coordinates,
                atom14_mask=atom14_mask,
                node_states=node_states,
                edge_states=edge_states,
                neighbor_indices=neighbor_indices,
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
            torsion_sin_cos[:, :, 3:] = torch.where(pack_positions, sampled_chi_sin_cos, input_chi_sin_cos)
            frames = _torsion_angles_to_frames(
                backbone_frames,
                torsion_sin_cos,
                default_frames=default_frames,
            )
            atom14_coordinates = _frames_and_literature_positions_to_atom14_pos(
                frames,
                atom14_to_rigid_group=atom14_to_rigid_group,
                atom14_rigid_group_positions=atom14_rigid_group_positions,
                atom14_mask=atom14_exists,
            )
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
        )

    @torch.inference_mode()
    def pack(
        self,
        input: MpnnPackerInput,
        *,
        num_samples: int = 1,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        generator: torch.Generator | None = None,
    ) -> PackResult:
        r"""
        Generate independently packed structures for one processed target.

        Args:
            input: One processed side-chain packing target.
            num_samples: Number of packed structures to generate.
            num_denoising_steps: Number of iterative side-chain updates per sample.
            num_torsion_samples: Number of torsion proposals evaluated per update.
            generator: Random generator used for torsion sampling.

        Returns:
            Packed structure predictions in sampling order.
        """

        if input.sequence_tokens.shape[0] != 1:
            raise ValueError("pack requires batch size one")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_denoising_steps <= 0:
            raise ValueError("num_denoising_steps must be positive")
        if num_torsion_samples <= 0:
            raise ValueError("num_torsion_samples must be positive")
        if not bool(input.pack_positions.any()):
            return PackResult(predictions=tuple(PackPrediction(input.structure) for _ in range(num_samples)))

        predictions = tuple(
            _pack_candidate(
                input,
                self.sample(
                    input,
                    num_denoising_steps=num_denoising_steps,
                    num_torsion_samples=num_torsion_samples,
                    generator=generator,
                ),
            )
            for _ in range(num_samples)
        )
        return PackResult(predictions=predictions)


class MpnnPackerFeatures(nn.Module):
    r"""
    Build residue-graph and atom-context features for MPNN Packer.

    Args:
        node_size: Width of the per-node features.
        edge_size: Width of the per-edge features.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
        num_neighbors: Number of nearest-neighbor residues in the graph.
        num_context_atoms: Number of ligand context atoms per residue used by the decoder.
        coordinate_noise: Training-time Gaussian coordinate-noise standard deviation.
        lower_bound: Lower distance bound for the context RBF expansion.
        upper_bound: Upper distance bound for the context RBF expansion.
    """

    def __init__(
        self,
        *,
        node_size: int,
        edge_size: int,
        num_positional_embeddings: int,
        num_rbf: int,
        num_neighbors: int,
        num_context_atoms: int,
        coordinate_noise: float,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.num_neighbors = num_neighbors
        self.num_context_atoms = num_context_atoms
        self.coordinate_noise = coordinate_noise
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.relative_position = MpnnRelativePositionEncoding(num_positional_embeddings)

        self.edge_projection = nn.Linear(num_positional_embeddings + num_rbf * 25, edge_size, bias=False)
        self.edge_norm = LayerNorm(edge_size)
        self.node_projection = nn.Linear(len(AMINO_ACID_COMPONENT_IDS) + 1, node_size, bias=False)
        self.node_norm = LayerNorm(node_size)

        self.context_down_projection = nn.Linear(num_rbf + 120, num_rbf)
        self.decode_edge_projection = nn.Linear(
            num_rbf * _ATOM14 * _ATOM14 + 2 * (len(AMINO_ACID_COMPONENT_IDS) + 1),
            edge_size,
            bias=False,
        )
        self.decode_edge_norm = LayerNorm(edge_size)
        self.decode_node_projection = nn.Linear(_ATOM14 * num_context_atoms * num_rbf, node_size, bias=False)
        self.decode_node_norm = LayerNorm(node_size)

        self.context_node_projection = nn.Linear(5 * num_rbf + 64 + 4, node_size)
        self.context_node_norm = LayerNorm(node_size)
        self.element_projection = nn.Linear(147, 64)
        self.atom_node_projection = nn.Linear(147, node_size, bias=False)
        self.atom_edge_projection = nn.Linear(num_rbf, node_size, bias=False)
        self.atom_edge_norm = LayerNorm(node_size)
        self.atom_node_norm = LayerNorm(node_size)

        buffer_device = self.element_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        group, period = periodic_group_period(device=buffer_device)
        self.register_buffer("periodic_group", group, persistent=False)
        self.register_buffer("periodic_period", period, persistent=False)

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
        sequence_tokens: Tensor,
        residue_mask: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build residue-graph and atom-context encoder features.

        Args:
            atom14_coordinates: Current atom14 coordinates ``[B, L, 14, 3]``.
            sequence_tokens: Residue tokens ``[B, L]``.
            residue_mask: Valid-residue mask ``[B, L]``.
            residue_index: Residue positions used by relative-position encoding ``[B, L]``.
            chain_index: Chain indices ``[B, L]``.
            context_coordinates: Per-residue non-protein context coordinates ``[B, L, A, 3]``.
            context_atomic_numbers: Atomic numbers for the context coordinates ``[B, L, A]``.
            context_mask: Valid-context mask ``[B, L, A]``.

        Returns:
            Residue nodes ``(B, L, H)``, residue edges ``(B, L, K, H)``,
            neighbor indices ``(B, L, K)``, atom nodes ``(B, L, A, H)``, atom
            edges ``(B, L, A, A, H)``, residue-to-context features
            ``(B, L, A, H)``, and context mask ``(B, L, A)``.
        """

        coordinates = atom14_coordinates
        if self.training and self.coordinate_noise > 0:
            coordinates = coordinates + self.coordinate_noise * torch.randn_like(coordinates)
        context_atomic_numbers = context_atomic_numbers.long()

        n, ca, c, o = (
            coordinates[:, :, 0],
            coordinates[:, :, 1],
            coordinates[:, :, 2],
            coordinates[:, :, 3],
        )
        cb = virtual_beta_carbon(n, ca, c)
        _, neighbor_indices = _knn(ca, residue_mask, self.num_neighbors)

        backbone_atoms = torch.stack((n, ca, c, o, cb), dim=2)
        pair_distances = _neighbor_atom_distances(backbone_atoms, neighbor_indices)
        pair_rbf = radial_basis(
            pair_distances,
            self.num_rbf,
            self.lower_bound,
            self.upper_bound,
        ).flatten(start_dim=-3)
        batch = torch.arange(coordinates.shape[0], device=coordinates.device)[:, None, None]
        neighbor_residue_index = residue_index[batch, neighbor_indices]
        offset = residue_index[:, :, None] - neighbor_residue_index
        neighbor_chain = chain_index[batch, neighbor_indices]
        same_chain = (chain_index[:, :, None] == neighbor_chain).long()
        edges = self.edge_norm(
            self.edge_projection(torch.cat([self.relative_position(offset.long(), same_chain), pair_rbf], dim=-1))
        )
        nodes = self.node_norm(
            self.node_projection(
                F.one_hot(sequence_tokens.long(), len(AMINO_ACID_COMPONENT_IDS) + 1).to(dtype=coordinates.dtype)
            )
        )

        element_features = self._element_features(context_atomic_numbers, dtype=coordinates.dtype)
        element_projection = self.element_projection(element_features)
        context_distances = torch.sqrt(
            torch.sum(
                (backbone_atoms[:, :, :, None] - context_coordinates[:, :, None]) ** 2,
                dim=-1,
            )
            + 1e-6
        )
        context_features = self._context_rbf(context_distances.movedim(2, 3)).flatten(start_dim=-2)
        angle_features = local_backbone_frame_features(n, ca, c, context_coordinates)
        context = torch.cat([context_features, element_projection, angle_features], dim=-1)
        context_nodes = self.context_node_norm(self.context_node_projection(context))

        atom_pair_distances = torch.sqrt(
            torch.sum(
                (context_coordinates[:, :, :, None] - context_coordinates[:, :, None, :]) ** 2,
                dim=-1,
            )
            + 1e-6
        )
        atom_edges = self.atom_edge_norm(self.atom_edge_projection(self._context_rbf(atom_pair_distances)))
        atom_nodes = self.atom_node_norm(self.atom_node_projection(element_features))
        return nodes, edges, neighbor_indices, atom_nodes, atom_edges, context_nodes, context_mask

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
            Side-chain node features with shape ``(B, L, H)`` and decoder edge
            features with shape ``(B, L, K, H)``.
        """

        coordinates = atom14_coordinates
        atom_mask = atom14_mask & residue_mask[:, :, None]
        context_coordinates = context_coordinates[:, :, : self.num_context_atoms]
        context_mask = context_mask[:, :, : self.num_context_atoms]
        context_atomic_numbers = context_atomic_numbers[:, :, : self.num_context_atoms].long()
        batch, length = coordinates.shape[:2]

        gathered_mask = gather_nodes(atom_mask, neighbor_indices)
        sidechain_distances = _neighbor_atom_distances(coordinates, neighbor_indices)
        pair_mask = atom_mask[:, :, None, :, None] & gathered_mask[:, :, :, None, :]
        sidechain_rbf = radial_basis(
            sidechain_distances,
            self.num_rbf,
            self.lower_bound,
            self.upper_bound,
        )
        sidechain_rbf = (sidechain_rbf * pair_mask[..., None]).flatten(start_dim=-3)

        atom_context_distances = torch.sqrt(
            torch.sum(
                (coordinates[:, :, :, None, :] - context_coordinates[:, :, None, :, :]) ** 2,
                dim=-1,
            )
            + 1e-6
        )
        atom_context_features = self._context_rbf(atom_context_distances)
        atom_context_features = (
            atom_context_features * atom_mask[:, :, :, None, None] * context_mask[:, :, None, :, None]
        )
        context_types_1hot = F.one_hot(context_atomic_numbers, 120).to(dtype=coordinates.dtype)
        combined = torch.cat(
            [atom_context_features, context_types_1hot[:, :, None].expand(-1, -1, _ATOM14, -1, -1)], dim=-1
        )
        node_features = self.context_down_projection(combined).view(batch, length, -1)
        sidechain_nodes = self.decode_node_norm(self.decode_node_projection(node_features))

        sequence_1hot = F.one_hot(sequence_tokens.long(), len(AMINO_ACID_COMPONENT_IDS) + 1).to(dtype=coordinates.dtype)
        gathered_sequence = gather_nodes(sequence_1hot, neighbor_indices)
        sequence_features = torch.cat(
            [
                sequence_1hot[:, :, None].expand(-1, -1, neighbor_indices.shape[2], -1),
                gathered_sequence,
            ],
            dim=-1,
        )
        decode_edges = torch.cat([sidechain_rbf, sequence_features], dim=-1)
        decode_edges = self.decode_edge_norm(self.decode_edge_projection(decode_edges))
        return sidechain_nodes, decode_edges


def _knn(points: Tensor, mask: Tensor, num_neighbors: int, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    r"""
    Find the ``num_neighbors`` nearest residues by CA distance, after every valid pair.

    Args:
        points: Reference (CA) coordinates `[B, L, 3]`.
        mask: Valid-residue mask `[B, L]`.
        num_neighbors: Number of neighbors to keep (capped at ``L``).
        eps: Numerical floor under the distance square root.

    Returns:
        Tuple ``(distances, neighbor_idx)`` of the top-k nearest neighbors, each `[B, L, k]`.
    """

    pair_mask = mask[:, None, :] & mask[:, :, None]
    distances = torch.sqrt(torch.sum((points[:, :, None] - points[:, None, :]) ** 2, dim=-1) + eps)
    adjusted = torch.where(pair_mask, distances, torch.inf)
    return torch.topk(adjusted, min(num_neighbors, points.shape[1]), dim=-1, largest=False)


def _neighbor_atom_distances(atoms: Tensor, neighbor_idx: Tensor) -> Tensor:
    r"""
    Compute every ordered atom-pair distance along the residue neighbor graph.

    Args:
        atoms: Per-residue atom coordinates `[B, L, A, 3]`.
        neighbor_idx: Neighbor indices `[B, L, K]`.

    Returns:
        Distances `[B, L, K, A, A]`, with source atom before target atom.
    """

    batch = torch.arange(atoms.shape[0], device=atoms.device)[:, None, None]
    neighbors = atoms[batch, neighbor_idx]
    distances = torch.sqrt(torch.sum((atoms[:, :, None, :, None] - neighbors[:, :, :, None]) ** 2, dim=-1) + 1e-6)
    return distances


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


def _pack_candidate(
    input: MpnnPackerInput,
    packed: MpnnPackerSamplingOutput,
) -> PackPrediction:
    r"""
    Decode one sampled atom14 state into a packed structure.

    Args:
        input: Processor output describing the source structure and selected residues.
        packed: Sampled atom14 coordinates, masks, and torsions.

    Returns:
        A packed structure containing sampled atoms only at selected residues.
    """

    structure = input.structure
    source_index = {site: index for index, site in enumerate(structure.sites)}
    selected = input.pack_positions[0].detach().cpu().tolist()
    selected_references = {
        reference for reference, is_selected in zip(input.component_references, selected) if is_selected
    }
    component_ids = {
        reference: AMINO_ACID_COMPONENT_IDS[int(token)]
        for reference, token, is_selected in zip(
            input.component_references,
            input.sequence_tokens[0].detach().cpu().tolist(),
            selected,
        )
        if is_selected
    }
    expected_atoms = {
        reference: frozenset(filter(None, rc.AMINO_ACID_ATOM14_NAMES[component_ids[reference]]))
        for reference in selected_references
    }
    site_list = [
        site
        for site in structure.sites
        if (site.instance_id, site.component_index) not in selected_references
        or (
            (site.instance_id, site.component_index) in selected_references
            and (site.atom_id in expected_atoms[(site.instance_id, site.component_index)] or site.atom_id == "OXT")
        )
    ]
    retained_sites = set(site_list)
    for reference, is_selected in zip(input.component_references, selected):
        if not is_selected:
            continue
        for atom_name in rc.AMINO_ACID_ATOM14_NAMES[component_ids[reference]][4:]:
            if not atom_name:
                continue
            site = AtomSite(reference[0], reference[1], atom_name)
            if site not in retained_sites:
                site_list.append(site)
                retained_sites.add(site)
    sites = tuple(site_list)
    output_index = {site: index for index, site in enumerate(sites)}
    atom14 = packed.atom14_coordinates[0].detach().cpu().numpy()
    coordinate_dtype = np.result_type(structure.coordinates.dtype, atom14.dtype)
    coordinates = np.zeros((len(sites), 3), dtype=coordinate_dtype)
    coordinate_mask = np.zeros(len(sites), dtype=np.bool_)
    source_occupancy = structure.occupancy
    occupancy = (
        None
        if source_occupancy is None
        else np.full(
            len(sites),
            np.nan,
            dtype=np.result_type(source_occupancy.dtype, np.float32),
        )
    )
    for index, site in enumerate(sites):
        source = source_index.get(site)
        if source is None:
            continue
        coordinates[index] = structure.coordinates[source]
        coordinate_mask[index] = structure.coordinate_mask[source]
        if occupancy is not None and source_occupancy is not None:
            occupancy[index] = source_occupancy[source]

    atom14_mask = packed.atom14_mask[0].detach().cpu().numpy()
    for row, (reference, is_selected) in enumerate(zip(input.component_references, selected)):
        if not is_selected:
            continue
        for atom, atom_name in enumerate(rc.AMINO_ACID_ATOM14_NAMES[component_ids[reference]][4:], start=4):
            if not atom_name:
                continue
            index = output_index[AtomSite(reference[0], reference[1], atom_name)]
            coordinates[index] = atom14[row, atom]
            coordinate_mask[index] = atom14_mask[row, atom]
            if occupancy is not None:
                occupancy[index] = np.nan

    realized = Structure(
        structure.system,
        sites,
        coordinates,
        coordinate_mask,
        occupancy,
        structure.residue_numbers,
    )
    metrics: dict[str, Metric] = {}
    chi_log_densities = packed.chi_log_densities[0].detach().cpu().numpy()
    for chi_index in range(4):
        rows = tuple(
            row
            for row, (reference, is_selected) in enumerate(zip(input.component_references, selected))
            if is_selected and len(rc.chi_angles_atoms[component_ids[reference]]) > chi_index
        )
        if rows:
            metrics[f"chi{chi_index + 1}_log_density"] = Metric(
                values=chi_log_densities[rows, chi_index],
                axis=tuple(input.component_references[row] for row in rows),
                unit="nat/radian",
            )
    return PackPrediction(structure=realized, metrics=metrics)


class MpnnPackerFrame:
    r"""
    Rigid frame stored as a rotation matrix and translation.

    Any number of leading batch dimensions is supported.

    Args:
        rotation: Rotation matrices with shape ``(*, 3, 3)``.
        translation: Translations with shape ``(*, 3)``.
    """

    def __init__(self, rotation: Tensor, translation: Tensor) -> None:
        self.rotation = rotation
        self.translation = translation

    def compose(self, other: MpnnPackerFrame) -> MpnnPackerFrame:
        r"""
        Compose ``self`` with a frame expressed in its local coordinates.

        Args:
            other: Frame expressed in ``self``'s local coordinates.

        Returns:
            The composed transform ``self(other(points))``.
        """

        rotation = self.rotation @ other.rotation
        translation = _rotate(self.rotation, other.translation) + self.translation
        return MpnnPackerFrame(rotation, translation)

    def apply(self, points: Tensor) -> Tensor:
        r"""
        Map local points into the global frame.

        Args:
            points: Local coordinates with shape ``(*, 3)``.

        Returns:
            Rotated and translated coordinates with shape ``(*, 3)``.
        """

        return _rotate(self.rotation, points) + self.translation

    def invert_apply(self, points: Tensor) -> Tensor:
        r"""
        Map global points back into the local frame.

        Args:
            points: Global coordinates with shape ``(*, 3)``.

        Returns:
            Local coordinates with shape ``(*, 3)``.
        """

        inverse_rotation = self.rotation.transpose(-1, -2)
        inverse_translation = -_rotate(inverse_rotation, self.translation)
        return _rotate(inverse_rotation, points) + inverse_translation

    def unsqueeze(self, dim: int) -> MpnnPackerFrame:
        r"""
        Insert a size-one dimension into the leading axes.

        Args:
            dim: Position of the new leading axis.

        Returns:
            A frame with the axis inserted into both rotation and translation.
        """

        rotation_dim = dim - 2 if dim < 0 else dim
        translation_dim = dim - 1 if dim < 0 else dim
        return MpnnPackerFrame(self.rotation.unsqueeze(rotation_dim), self.translation.unsqueeze(translation_dim))

    def __getitem__(self, index) -> MpnnPackerFrame:
        r"""
        Index the shared leading (batch) axes, returning the sub-frame at ``index``.
        """

        if not isinstance(index, tuple):
            index = (index,)
        return MpnnPackerFrame(
            self.rotation[index + (slice(None), slice(None))],
            self.translation[index + (slice(None),)],
        )

    @staticmethod
    def from_tensor_4x4(transform: Tensor) -> MpnnPackerFrame:
        r"""
        Build a frame from homogeneous transforms.

        Args:
            transform: Homogeneous transforms with shape ``(*, 4, 4)``.

        Returns:
            Frames with rotation from ``transform[..., :3, :3]`` and translation
            from ``transform[..., :3, 3]``.
        """

        return MpnnPackerFrame(transform[..., :3, :3], transform[..., :3, 3])

    @staticmethod
    def from_3_points(
        neg_x_axis: Tensor,
        origin: Tensor,
        xy_plane: Tensor,
        eps: float = 1e-8,
    ) -> MpnnPackerFrame:
        r"""
        Construct right-handed frames from three points with Gram-Schmidt.

        Args:
            neg_x_axis: Point defining the negative x-axis direction, shape ``(*, 3)``.
            origin: Frame origin, shape ``(*, 3)``.
            xy_plane: Point placed in the xy-plane, shape ``(*, 3)``.
            eps: Numerical floor for the normalizations.

        Returns:
            The Gram-Schmidt frame with origin at ``origin``.
        """

        neg_x = neg_x_axis.unbind(dim=-1)
        origin_xyz = origin.unbind(dim=-1)
        xy_plane_xyz = xy_plane.unbind(dim=-1)

        e0 = tuple(origin_axis - negative_axis for origin_axis, negative_axis in zip(origin_xyz, neg_x))
        e1 = tuple(xy_axis - origin_axis for xy_axis, origin_axis in zip(xy_plane_xyz, origin_xyz))
        denominator = torch.sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2] + eps)
        e0 = tuple(axis / denominator for axis in e0)
        dot = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2]
        e1 = tuple(axis - e0_axis * dot for e0_axis, axis in zip(e0, e1))
        denominator = torch.sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] + eps)
        e1 = tuple(axis / denominator for axis in e1)
        e2 = (
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        )
        rotation = torch.stack(tuple(axis for frame_axis in zip(e0, e1, e2) for axis in frame_axis), dim=-1)
        return MpnnPackerFrame(
            rotation.reshape(rotation.shape[:-1] + (3, 3)),
            torch.stack(origin_xyz, dim=-1),
        )

    @staticmethod
    def make_transform_from_reference(
        n_xyz: Tensor,
        ca_xyz: Tensor,
        c_xyz: Tensor,
        eps: float = 1e-20,
    ) -> MpnnPackerFrame:
        r"""
        Construct backbone frames from N, C-alpha, and C coordinates.

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
        return MpnnPackerFrame(rotation, -translation)


def _rotate(rotation: Tensor, points: Tensor) -> Tensor:
    r"""
    Apply rotation matrices ``(*, 3, 3)`` to points ``(*, 3)``.
    """

    return torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1)


def _stack_rot(rows: list[list], like: Tensor) -> Tensor:
    r"""
    Assemble a batched ``(*, 3, 3)`` rotation from per-entry tensors/scalars/``None``.
    """

    zeros = torch.zeros_like(like)
    filled = [
        [zeros if entry is None else (entry + zeros if not torch.is_tensor(entry) else entry) for entry in row]
        for row in rows
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in filled], dim=-2)


def _atom37_to_torsion_angles(
    all_atom_positions: Tensor,
    all_atom_mask: Tensor,
    *,
    chi_atom_indices: Tensor,
    chi_angles_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute the seven backbone/side-chain torsion angles as sin/cos pairs.

    The packing model consumes only normalized sine/cosine torsions and their
    validity mask, so this routine returns that bounded representation directly.

    Args:
        all_atom_positions: Atom37 coordinates with shape ``(B, L, 37, 3)``.
        all_atom_mask: Atom37 existence mask with shape ``(B, L, 37)``.
        chi_atom_indices: Pre-indexed atom37 indices for the four chi angles, shape
            ``(B, L, 4, 4)``.
        chi_angles_mask: Pre-indexed chi existence mask, shape ``(B, L, 4)``.

    Returns:
        Tuple ``(torsion_sin_cos, torsion_mask)`` with shapes ``(B, L, 7, 2)``
        and ``(B, L, 7)``.
    """

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)
    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_pos = torch.cat([prev_positions[..., 1:3, :], all_atom_positions[..., :2, :]], dim=-2)
    phi_pos = torch.cat([prev_positions[..., 2:3, :], all_atom_positions[..., :3, :]], dim=-2)
    psi_pos = torch.cat([all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]], dim=-2)

    pre_omega_mask = prev_mask[..., 1:3].all(dim=-1) & all_atom_mask[..., :2].all(dim=-1)
    phi_mask = prev_mask[..., 2] & all_atom_mask[..., :3].all(dim=-1)
    psi_mask = all_atom_mask[..., :3].all(dim=-1) & all_atom_mask[..., 4]

    batch_indices = torch.arange(all_atom_positions.shape[0], device=all_atom_positions.device)[:, None, None, None]
    residue_index = torch.arange(all_atom_positions.shape[1], device=all_atom_positions.device)[None, :, None, None]
    chis_pos = all_atom_positions[batch_indices, residue_index, chi_atom_indices]
    chi_atoms_mask = all_atom_mask[batch_indices, residue_index, chi_atom_indices].all(dim=-1)
    chis_mask = chi_angles_mask & chi_atoms_mask

    torsions_pos = torch.cat(
        [pre_omega_pos[..., None, :, :], phi_pos[..., None, :, :], psi_pos[..., None, :, :], chis_pos], dim=-3
    )
    torsion_angles_mask = torch.cat(
        [pre_omega_mask[..., None], phi_mask[..., None], psi_mask[..., None], chis_mask], dim=-1
    )

    frames = MpnnPackerFrame.from_3_points(
        torsions_pos[..., 1, :],
        torsions_pos[..., 2, :],
        torsions_pos[..., 0, :],
    )
    fourth_atom_rel = frames.invert_apply(torsions_pos[..., 3, :])
    torsion_angles_sin_cos = torch.stack([fourth_atom_rel[..., 2], fourth_atom_rel[..., 1]], dim=-1)
    denom = torch.sqrt(torch.sum(torsion_angles_sin_cos**2, dim=-1, keepdim=True) + 1e-8)
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom
    torsion_angles_sin_cos[..., 2, :] = -torsion_angles_sin_cos[..., 2, :]
    return torsion_angles_sin_cos, torsion_angles_mask


def _torsion_angles_to_frames(
    backbone_frames: MpnnPackerFrame,
    torsions: Tensor,
    *,
    default_frames: Tensor,
) -> MpnnPackerFrame:
    r"""
    Turn per-residue torsion sin/cos into the eight global rigid-group frames.

    Args:
        backbone_frames: Per-residue backbone frames with batch shape ``(B, L)``.
        torsions: Torsion sin/cos with shape ``(B, L, 7, 2)`` (three backbone + four chi).
        default_frames: Pre-indexed rigid-group transforms with shape ``(B, L, 8, 4, 4)``.

    Returns:
        Eight global rigid-group frames with batch shape ``(B, L, 8)``.
    """

    default = MpnnPackerFrame.from_tensor_4x4(default_frames)

    backbone_rotation = torsions.new_zeros((*((1,) * len(torsions.shape[:-1])), 2))
    backbone_rotation[..., 1] = 1
    alpha = torch.cat([backbone_rotation.expand(*torsions.shape[:-2], -1, -1), torsions], dim=-2)

    all_rots = alpha.new_zeros(default.rotation.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1] = alpha[..., 0]
    all_rots[..., 2, 2] = alpha[..., 1]
    all_frames = default.compose(MpnnPackerFrame(all_rots, torch.zeros_like(default.translation)))

    chi1 = all_frames[..., 4]
    chi2 = chi1.compose(all_frames[..., 5])
    chi3 = chi2.compose(all_frames[..., 6])
    chi4 = chi3.compose(all_frames[..., 7])
    frames_to_bb = _frame_cat([all_frames[..., :5], chi2.unsqueeze(-1), chi3.unsqueeze(-1), chi4.unsqueeze(-1)])
    return backbone_frames.unsqueeze(-1).compose(frames_to_bb)


def _frames_and_literature_positions_to_atom14_pos(
    frames: MpnnPackerFrame,
    *,
    atom14_to_rigid_group: Tensor,
    atom14_rigid_group_positions: Tensor,
    atom14_mask: Tensor,
) -> Tensor:
    r"""
    Place literature atom14 positions using the global rigid-group frames.

    Args:
        frames: Eight global rigid-group frames with batch shape ``(B, L, 8)``.
        atom14_to_rigid_group: Pre-indexed rigid-group indices with shape ``(B, L, 14)``.
        atom14_rigid_group_positions: Pre-indexed literature positions with shape ``(B, L, 14, 3)``.
        atom14_mask: Pre-indexed atom14 existence mask with shape ``(B, L, 14)``.

    Returns:
        Atom14 coordinates with shape ``(B, L, 14, 3)``.
    """

    group_mask = torch.nn.functional.one_hot(atom14_to_rigid_group, num_classes=8).to(frames.rotation.dtype)

    rotation = (frames.rotation[..., None, :, :, :] * group_mask[..., None, None]).sum(dim=-3)
    translation = (frames.translation[..., None, :, :] * group_mask[..., None]).sum(dim=-2)
    atom_frames = MpnnPackerFrame(rotation, translation)

    return atom_frames.apply(atom14_rigid_group_positions) * atom14_mask.unsqueeze(-1)


def _frame_cat(frames: list[MpnnPackerFrame]) -> MpnnPackerFrame:
    r"""
    Concatenate frames along their last shared dimension.
    """

    rotation = torch.cat([frame.rotation for frame in frames], dim=-3)
    translation = torch.cat([frame.translation for frame in frames], dim=-2)
    return MpnnPackerFrame(rotation, translation)


# ---------------------------------------------------------------------------
# MpnnPacker tensor inputs and outputs.
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class MpnnPackerInput:
    r"""
    Inputs for side-chain packing.

    Args:
        backbone_coordinates: N, CA, C, and O coordinates with shape
            ``(batch_size, num_residues, 4, 3)``.
        sequence_tokens: Residue token IDs with shape ``(batch_size, num_residues)``.
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
        pack_positions: Residues selected for packing with shape
            ``(batch_size, num_residues)``.
        structure: Source structure.
        component_references: Component occurrences aligned to residue rows.
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
    pack_positions: Tensor
    structure: Structure
    component_references: tuple[ComponentReference, ...]


@dataclass(eq=False)
class MpnnPackerOutput:
    r"""
    Predicted side-chain torsion mixtures.

    Args:
        mean: Von Mises means with shape
            ``(batch_size, num_residues, 4, num_mixtures)``.
        concentration: Positive concentrations with the same shape as ``mean``.
        mix_logits: Mixture logits with the same shape as ``mean``.
    """

    mean: Tensor
    concentration: Tensor
    mix_logits: Tensor


@dataclass(eq=False)
class MpnnPackerSamplingOutput:
    r"""
    Outputs from iterative side-chain sampling.

    Args:
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
