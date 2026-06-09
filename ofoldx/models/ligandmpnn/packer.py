r"""
Predicts and reconstructs protein side-chain atoms from a designed backbone and sequence.

An optional companion to [`LigandMPNNDesignGenerator`][ofoldx.models.ligandmpnn.model.LigandMPNNDesignGenerator]
that predicts side-chain chi angles as a Von Mises mixture and denoises them into atom14
coordinates. The backbone design generator emits sequence only; constructing a [`Packer`][]
adds side-chain packing as a separate component with its own weights (it does not change the
backbone parameter count).

Faithful port of the reference LigandMPNN ``sc_utils`` Packer. The message-passing layers reuse
the backbone layer classes; the torsion-frame reconstruction lives in
[`ligandmpnn`][ofoldx.processors.ligandmpnn].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributions as distributions
import torch.nn.functional as F
from torch import Tensor, nn

from ...data import residue_constants as rc
from ..mixin import ModelMixin
from ..registry import MODELS
from .chemical import periodic_group_period
from .config import PackerConfig
from .model import (
    LigandAtomLayer,
    LigandDecoderLayer,
    LigandEncoderLayer,
    RelativePositionEncoding,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)
from .processor import (
    MPNN_TO_AF2_RESTYPE,
    Frame,
    atom37_to_torsion_angles,
    frames_and_literature_positions_to_atom14_pos,
    make_atom14_masks,
    torsion_angles_to_frames,
)

_ATOM14 = 14


@MODELS.register("ligandmpnn_packer")
class Packer(ModelMixin):
    r"""
    Predicts side-chain chi angles as a Von Mises mixture and denoises them into atom coordinates.

    Encodes the backbone graph and ligand context, then decodes a mixture-of-Von-Mises
    distribution over the four side-chain chi angles per residue. Sampling denoises random
    torsions into atom14 coordinates (see [`pack_side_chains`][]).

    Args:
        config: Side-chain packer configuration; a default
            [`PackerConfig`][ofoldx.models.ligandmpnn.config.PackerConfig] is used when ``None``.
    """

    config_class = PackerConfig

    def __init__(self, config: PackerConfig | None = None) -> None:
        config = PackerConfig() if config is None else config
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_mix = config.num_mix
        self.atom_context_num = config.atom_context_num
        self.features = PackerFeatures(
            node_size=config.hidden_size,
            edge_size=config.hidden_size,
            num_positional_embeddings=config.num_positional_embeddings,
            num_rbf=config.num_rbf,
            k_neighbors=config.k_neighbors,
            atom_context_num=config.atom_context_num,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
        )
        self.softplus = nn.Softplus(beta=1, threshold=20)
        self.dropout = nn.Dropout(config.dropout)

        self.edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.node_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.decode_edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.side_chain_node_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_down = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.torsion_projection = nn.Linear(config.hidden_size, 4 * 3 * config.num_mix)

        self.encoder_layers = nn.ModuleList(
            [
                LigandEncoderLayer(config.hidden_size, config.hidden_size * 2, dropout=config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.residue_context_input = nn.Linear(config.hidden_size, config.hidden_size)
        self.context_edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.atom_node_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.atom_edge_embedding = nn.Linear(config.hidden_size, config.hidden_size)
        self.context_layers = nn.ModuleList(
            [LigandDecoderLayer(config.hidden_size, config.hidden_size * 2, dropout=config.dropout) for _ in range(2)]
        )
        self.context_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_norm = nn.LayerNorm(config.hidden_size)
        self.atom_context_layers = nn.ModuleList(
            [LigandAtomLayer(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(2)]
        )
        self.context_dropout = nn.Dropout(config.dropout)
        self.decoder_layers = nn.ModuleList(
            [
                LigandDecoderLayer(config.hidden_size, config.hidden_size * 3, dropout=config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )

        self.post_init()

    def _init_module_weights(self, module: nn.Module) -> None:
        return None

    def reset_parameters(self) -> None:
        r"""
        Re-initialize Packer weights to match the released LigandMPNN scheme.
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Encode the backbone graph and ligand context into node/edge states.

        Runs the encoder message-passing stack over the k-nearest-neighbor residue graph, then folds the
        ligand-context atom graph into each residue node. The result feeds
        [`decode`][ofoldx.models.ligandmpnn.packer.Packer.decode] via the cached ``h_V``/``h_E``/``E_idx``
        features.

        Args:
            features: Feature mapping with backbone coords ``X``, sequence ``S``, masks, and ligand context.

        Returns:
            Tuple ``(node_states, edge_states, neighbor_idx)`` of per-residue node states, per-edge states,
            and neighbor indices.
        """

        mask = features["mask"]
        node_features, edges, neighbor_idx, atom_nodes, atom_edges, context_nodes, atom_mask = (
            self.features.features_encode(features)
        )
        context = self.context_edge_embedding(context_nodes)
        node_states = self.node_embedding(node_features)
        edge_states = self.edge_embedding(edges)
        attend_mask = gather_nodes(mask.unsqueeze(-1), neighbor_idx).squeeze(-1)
        attend_mask = mask.unsqueeze(-1) * attend_mask
        for layer in self.encoder_layers:
            node_states, edge_states = layer(node_states, edge_states, neighbor_idx, mask, attend_mask)

        residue_context = self.residue_context_input(node_states)
        atom_pair_mask = atom_mask[:, :, :, None] * atom_mask[:, :, None, :]
        atom_nodes = self.atom_node_embedding(atom_nodes)
        atom_edges = self.atom_edge_embedding(atom_edges)
        for layer_idx, layer in enumerate(self.context_layers):
            atom_nodes = self.atom_context_layers[layer_idx](atom_nodes, atom_edges, atom_mask, atom_pair_mask)
            residue_context = layer(residue_context, torch.cat([context, atom_nodes], dim=-1), mask, atom_mask)
        residue_context = self.context_output(residue_context)
        node_states = node_states + self.context_norm(self.context_dropout(residue_context))
        return node_states, edge_states, neighbor_idx

    def decode(self, features: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Decode the Von Mises torsion mixture parameters from the encoded graph.

        Adds explicit atom14-atom and atom14-ligand distance features (from
        [`features_decode`][ofoldx.models.ligandmpnn.packer.PackerFeatures.features_decode]), runs the
        decoder layers, and projects to per-chi mixture parameters. The concentration is soft-plus-clamped to
        stay positive.

        Args:
            features: Feature mapping carrying the cached encoder outputs ``h_V``/``h_E``/``E_idx`` plus the
                per-residue atom coordinates and masks.

        Returns:
            Tuple of ``(mean, concentration, mix_logits)``, each with shape ``(B, L, 4, num_mix)``.
        """

        node_states = features["h_V"]
        edge_states = features["h_E"]
        neighbor_idx = features["E_idx"]
        mask = features["mask"]
        side_chain_nodes, decode_edges = self.features.features_decode(features)

        decode_edges = self.decode_edge_embedding(decode_edges)
        edge_context = torch.cat([edge_states, decode_edges], dim=-1)
        side_chain_states = self.side_chain_node_embedding(side_chain_nodes)
        node_states = self.linear_down(torch.cat([node_states, side_chain_states], dim=-1))

        for layer in self.decoder_layers:
            neighbor_states = cat_neighbors_nodes(node_states, edge_context, neighbor_idx)
            node_states = layer(node_states, neighbor_states, mask)

        torsions = self.torsion_projection(node_states)
        torsions = torsions.reshape(node_states.shape[0], node_states.shape[1], 4, self.num_mix, 3)
        mean = torsions[:, :, :, :, 0].float()
        concentration = 0.1 + self.softplus(torsions[:, :, :, :, 1]).float()
        mix_logits = torsions[:, :, :, :, 2].float()
        return mean, concentration, mix_logits


class PackerFeatures(nn.Module):
    r"""
    Turns backbone coordinates and ligand context into the graph and distance features the Packer consumes.

    Builds the backbone graph, ligand-context, and side-chain atom-distance features used by the
    Packer encoder and decoder. Mirrors the reference LigandMPNN ``ProteinFeatures`` used by the
    Packer (note: distinct from the sequence-model feature encoder).

    Args:
        node_size: Width of the per-node features.
        edge_size: Width of the per-edge features.
        num_positional_embeddings: Width of the relative-position embedding.
        num_rbf: Number of radial basis functions per distance feature.
        k_neighbors: Number of nearest-neighbor residues in the graph.
        atom_context_num: Number of ligand context atoms per residue used by the decoder.
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
        k_neighbors: int,
        atom_context_num: int,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.k_neighbors = k_neighbors
        self.atom_context_num = atom_context_num
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.relative_position = RelativePositionEncoding(num_positional_embeddings)

        self.edge_projection = nn.Linear(num_positional_embeddings + num_rbf * 25, edge_size, bias=False)
        self.edge_norm = nn.LayerNorm(edge_size)
        self.node_projection = nn.Linear(21, node_size, bias=False)
        self.node_norm = nn.LayerNorm(node_size)

        self.context_down_projection = nn.Linear(num_rbf + 120, num_rbf)
        self.decode_edge_projection = nn.Linear(num_rbf * _ATOM14 * _ATOM14 + 42, edge_size, bias=False)
        self.decode_edge_norm = nn.LayerNorm(edge_size)
        self.decode_node_projection = nn.Linear(_ATOM14 * atom_context_num * num_rbf, node_size, bias=False)
        self.decode_node_norm = nn.LayerNorm(node_size)

        self.context_node_projection = nn.Linear(5 * num_rbf + 64 + 4, node_size)
        self.context_node_norm = nn.LayerNorm(node_size)
        self.element_projection = nn.Linear(147, 64)
        self.atom_node_projection = nn.Linear(147, node_size, bias=False)
        self.atom_edge_projection = nn.Linear(num_rbf, node_size, bias=False)
        self.atom_edge_norm = nn.LayerNorm(node_size)
        self.atom_node_norm = nn.LayerNorm(node_size)

        group, period = periodic_group_period()
        self.register_buffer("periodic_group", group, persistent=False)
        self.register_buffer("periodic_period", period, persistent=False)

    def _context_rbf(self, distances: Tensor) -> Tensor:
        r"""RBF-expand distances over the configured ``[lower_bound, upper_bound]`` range."""

        return _rbf_bounds(distances, self.num_rbf, self.lower_bound, self.upper_bound)

    def _graph_rbf(self, source: Tensor, target: Tensor, neighbor_idx: Tensor) -> Tensor:
        r"""RBF-expand the neighbor distances between two atom sets.

        Args:
            source: Source atom coordinates `[B, L, 3]`.
            target: Target atom coordinates `[B, L, 3]`.
            neighbor_idx: Neighbor indices `[B, L, K]`.

        Returns:
            Per-edge RBF distance features `[B, L, K, num_rbf]`.
        """

        distances = torch.sqrt(torch.sum((source[:, :, None] - target[:, None, :]) ** 2, dim=-1) + 1e-6)
        neighbor_distances = gather_edges(distances[:, :, :, None], neighbor_idx)[:, :, :, 0]
        return _rbf_bounds(neighbor_distances, self.num_rbf, self.lower_bound, self.upper_bound)

    def _element_features(self, atomic_numbers: Tensor) -> Tensor:
        r"""One-hot-encode atoms by element, periodic group, and period `[..., 147]`."""

        atomic_numbers = atomic_numbers.long()
        group = self.periodic_group[atomic_numbers]
        period = self.periodic_period[atomic_numbers]
        return torch.cat([F.one_hot(atomic_numbers, 120), F.one_hot(group, 19), F.one_hot(period, 8)], dim=-1).float()

    def features_encode(
        self, features: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build the encoder inputs: backbone-graph edges, sequence nodes, and ligand-context features.

        Args:
            features: Feature mapping with backbone coords ``X``, sequence ``S``, residue/chain indices,
                validity ``mask``, and ligand context ``Y``/``Y_t``/``Y_m``.

        Returns:
            Tuple ``(nodes, edges, neighbor_idx, atom_nodes, atom_edges, context_nodes, context_mask)``.
        """

        sequence = features["S"]
        coords = features["X"]
        context_coords = features["Y"]
        context_mask = features["Y_m"]
        context_types = features["Y_t"].long()
        mask = features["mask"]
        residue_index = features["R_idx"]
        chain_labels = features["chain_labels"]

        n, ca, c, o = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], coords[:, :, 3]
        b, c_vec = ca - n, c - ca
        cb = -0.58273431 * torch.cross(b, c_vec, dim=-1) + 0.56802827 * b - 0.54067466 * c_vec + ca
        neighbor_idx = _knn(ca, mask, self.k_neighbors)

        backbone_atoms = [n, ca, c, o, cb]
        pair_rbf = torch.cat(
            [self._graph_rbf(a, b_atom, neighbor_idx) for a in backbone_atoms for b_atom in backbone_atoms], dim=-1
        )
        offset = residue_index[:, :, None] - residue_index[:, None, :]
        offset = gather_edges(offset[:, :, :, None], neighbor_idx)[:, :, :, 0]
        same_chain = (chain_labels[:, :, None] == chain_labels[:, None, :]).long()
        same_chain = gather_edges(same_chain[:, :, :, None], neighbor_idx)[:, :, :, 0]
        edges = self.edge_norm(
            self.edge_projection(torch.cat([self.relative_position(offset.long(), same_chain), pair_rbf], dim=-1))
        )
        nodes = self.node_norm(self.node_projection(F.one_hot(sequence.long(), 21).float()))

        element_features = self._element_features(context_types)
        element_projection = self.element_projection(element_features)
        context_distances = [
            self._context_rbf(torch.sqrt(torch.sum((atom[:, :, None] - context_coords) ** 2, dim=-1) + 1e-6))
            for atom in backbone_atoms
        ]
        angle_features = _local_angle_features(n, ca, c, context_coords)
        context = torch.cat([*context_distances, element_projection, angle_features], dim=-1)
        context_nodes = self.context_node_norm(self.context_node_projection(context))

        atom_pair_distances = torch.sqrt(
            torch.sum((context_coords[:, :, :, None] - context_coords[:, :, None, :]) ** 2, dim=-1) + 1e-6
        )
        atom_edges = self.atom_edge_norm(self.atom_edge_projection(self._context_rbf(atom_pair_distances)))
        atom_nodes = self.atom_node_norm(self.atom_node_projection(element_features))
        return nodes, edges, neighbor_idx, atom_nodes, atom_edges, context_nodes, context_mask

    def features_decode(self, features: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        r"""
        Build the decoder inputs: explicit atom14-atom and atom14-ligand distance features.

        Args:
            features: Feature mapping carrying the current atom14 coords ``X``, atom mask ``X_m``, cached
                neighbor indices ``E_idx``, sequence ``S``, and ligand context.

        Returns:
            Tuple ``(side_chain_nodes, decode_edges)`` of per-residue side-chain node features and the
            decoder edge context.
        """

        sequence = features["S"]
        coords = features["X"]
        atom_mask = features["X_m"] * features["mask"][:, :, None]
        neighbor_idx = features["E_idx"]
        context_coords = features["Y"][:, :, : self.atom_context_num]
        context_mask = features["Y_m"][:, :, : self.atom_context_num]
        context_types = features["Y_t"][:, :, : self.atom_context_num].long()
        batch, length = coords.shape[:2]

        gathered_mask = gather_nodes(atom_mask, neighbor_idx)
        side_chain_rbf = []
        for i in range(_ATOM14):
            for j in range(_ATOM14):
                pair = self._graph_rbf(coords[:, :, i], coords[:, :, j], neighbor_idx)
                pair = pair * atom_mask[:, :, i, None, None] * gathered_mask[:, :, :, j, None]
                side_chain_rbf.append(pair)

        atom_context_distances = torch.sqrt(
            torch.sum((coords[:, :, :, None, :] - context_coords[:, :, None, :, :]) ** 2, dim=-1) + 1e-6
        )
        atom_context_features = self._context_rbf(atom_context_distances)
        atom_context_features = (
            atom_context_features * atom_mask[:, :, :, None, None] * context_mask[:, :, None, :, None]
        )
        context_types_1hot = F.one_hot(context_types, 120).float()
        combined = torch.cat(
            [atom_context_features, context_types_1hot[:, :, None].repeat(1, 1, _ATOM14, 1, 1)], dim=-1
        )
        node_features = self.context_down_projection(combined).view(batch, length, -1)
        side_chain_nodes = self.decode_node_norm(self.decode_node_projection(node_features))

        sequence_1hot = F.one_hot(sequence.long(), 21).float()
        gathered_sequence = gather_nodes(sequence_1hot, neighbor_idx)
        sequence_features = torch.cat(
            [sequence_1hot[:, :, None].repeat(1, 1, neighbor_idx.shape[2], 1), gathered_sequence], dim=-1
        )
        decode_edges = torch.cat([*side_chain_rbf, sequence_features], dim=-1)
        decode_edges = self.decode_edge_norm(self.decode_edge_projection(decode_edges))
        return side_chain_nodes, decode_edges


@torch.no_grad()
def pack_side_chains(
    features: dict[str, Any],
    packer: Packer,
    *,
    num_denoising_steps: int = 3,
    num_samples: int = 10,
    repack_everything: bool = True,
    num_context_atoms: int = 16,
) -> dict[str, Any]:
    r"""
    Denoise random side-chain torsions into atom14 coordinates with the Packer.

    Faithful port of the reference LigandMPNN ``sc_utils.pack_side_chains``. Requires the input
    ``features`` to carry ``X`` (backbone N/CA/C/O, shape ``(B, L, 4, 3)``), ``S`` (LigandMPNN
    sequence tokens), and ``mask``; ligand context (``Y``/``Y_t``/``Y_m``) is optional and zeroed
    when absent. When ``repack_everything`` is ``False`` the true side chains at fixed positions
    (``chain_mask`` == 0) are retained and ``xyz_37`` must be present.

    Args:
        features: Mutable feature dict; updated in place and returned.
        packer: The Packer model.
        num_denoising_steps: Number of Von Mises denoising iterations.
        num_samples: Number of Von Mises samples drawn per step (best log-prob kept).
        repack_everything: Repack all side chains if ``True``; otherwise keep fixed ones.
        num_context_atoms: Number of ligand context atoms to allocate when ``Y`` is absent.

    Returns:
        The feature dict updated with ``X`` (atom14 coords), ``X_m`` (atom14 mask),
        ``b_factors``, and the raw mixture parameters ``mean``/``concentration``/``mix_logits``.

    Raises:
        ValueError: If ``num_denoising_steps`` is less than 1.

    Examples:
        >>> from ofoldx.models.ligandmpnn import Packer, pack_side_chains  # doctest: +SKIP
        >>> packed = pack_side_chains(features, Packer())  # doctest: +SKIP
    """

    device = features["X"].device
    torsion_dict = _make_torsion_features(features, repack_everything=repack_everything)
    features["X"] = torsion_dict["xyz14_noised"]
    features["X_m"] = torsion_dict["xyz14_m"]
    if "Y" not in features:
        batch, length = features["X"].shape[:2]
        features["Y"] = torch.zeros(batch, length, num_context_atoms, 3, device=device)
        features["Y_t"] = torch.zeros(batch, length, num_context_atoms, device=device)
        features["Y_m"] = torch.zeros(batch, length, num_context_atoms, device=device)

    if num_denoising_steps < 1:
        raise ValueError("num_denoising_steps must be at least 1")

    node_states, edge_states, neighbor_idx = packer.encode(features)
    features["h_V"], features["h_E"], features["E_idx"] = node_states, edge_states, neighbor_idx

    sample = features["X"].new_zeros(features["X"].shape[:2] + (4,))
    predicted_distribution: distributions.MixtureSameFamily | None = None
    mean = concentration = mix_logits = features["X"].new_zeros(0)
    xyz14_noised = features["X"]
    for _ in range(num_denoising_steps):
        mean, concentration, mix_logits = packer.decode(features)
        mixture = distributions.Categorical(logits=mix_logits)
        component = distributions.VonMises(mean, concentration)
        predicted_distribution = distributions.MixtureSameFamily(mixture, component)
        candidates = predicted_distribution.sample([num_samples])
        candidate_log_probs = predicted_distribution.log_prob(candidates)
        best = torch.argmax(candidate_log_probs, dim=0)[None]
        sample = torch.gather(candidates, dim=0, index=best)[0]

        torsions_unit = torch.cat([torch.sin(sample[:, :, :, None]), torch.cos(sample[:, :, :, None])], dim=-1)
        torsion_dict["torsions_noised"][:, :, 3:] = torsions_unit * torsion_dict["mask_fix_sc"] + torsion_dict[
            "torsions_true"
        ] * (1 - torsion_dict["mask_fix_sc"])
        frames = torsion_angles_to_frames(
            torsion_dict["backbone_frames"], torsion_dict["torsions_noised"], torsion_dict["aatype"]
        )
        xyz14_noised = frames_and_literature_positions_to_atom14_pos(frames, torsion_dict["aatype"])
        xyz14_noised = xyz14_noised * features["X_m"][:, :, :, None]
        features["X"] = xyz14_noised

    features["X"] = xyz14_noised
    aatype = torsion_dict["aatype"]
    assert predicted_distribution is not None  # num_denoising_steps >= 1 guarantees at least one step
    log_prob = predicted_distribution.log_prob(sample) * torsion_dict["mask_fix_sc"][..., 0] + 2.0 * (
        1 - torsion_dict["mask_fix_sc"][..., 0]
    )

    group = torch.tensor(rc.restype_atom14_to_rigid_group, device=device)[aatype].clamp(min=4) - 4
    atom_types_for_b_factor = F.one_hot(group, 4)
    features["b_factors"] = (log_prob[:, :, None, :] * atom_types_for_b_factor).sum(-1)
    features["mean"], features["concentration"], features["mix_logits"] = mean, concentration, mix_logits
    features["log_prob"], features["sample"] = log_prob, sample
    features["true_torsion_sin_cos"] = torsion_dict["torsions_true"]
    return features


def _make_torsion_features(features: Mapping[str, Any], *, repack_everything: bool) -> dict[str, Tensor]:
    r"""Build noised/true torsion angles and initial atom14 coordinates for packing.

    Reconstructs backbone frames from the input coordinates, converts them to torsion sin/cos, and seeds the
    side-chain chi angles with random values. When ``repack_everything`` is ``False`` the true torsions at
    fixed positions (from ``xyz_37``) are retained instead of noise.

    Args:
        features: Feature mapping with backbone coords ``X``, sequence ``S``, and (when not repacking
            everything) the observed side-chain coords ``xyz_37`` and the ``chain_mask``.
        repack_everything: Repack all side chains from random torsions if ``True``; otherwise keep fixed ones.

    Returns:
        Mapping with the atom14 mask/coords, backbone frames, noised/true torsion sin/cos, the fixed-side-chain
        mask, and the AF2 restype integers.
    """

    device = features["mask"].device
    mask = features["mask"]
    batch, length = mask.shape

    xyz37 = torch.zeros(batch, length, 37, 3, device=device, dtype=torch.float32)
    xyz37[:, :, :3] = features["X"][:, :, :3]
    xyz37[:, :, 4] = features["X"][:, :, 3]

    aatype = MPNN_TO_AF2_RESTYPE.to(device)[features["S"].long()]
    atom_masks = make_atom14_masks(aatype)
    torsion = atom37_to_torsion_angles(aatype, xyz37, atom_masks["atom37_atom_exists"])
    backbone_frames = Frame.make_transform_from_reference(xyz37[:, :, 0], xyz37[:, :, 1], xyz37[:, :, 2], eps=1e-9)

    if not repack_everything:
        true_torsion = atom37_to_torsion_angles(aatype, features["xyz_37"], atom_masks["atom37_atom_exists"])
        torsions_true = true_torsion["torsion_angles_sin_cos"].clone()[:, :, 3:]
        mask_fix_sc = features["chain_mask"][:, :, None, None]
    else:
        torsions_true = torch.zeros(batch, length, 4, 2, device=device)
        mask_fix_sc = torch.ones(batch, length, 1, 1, device=device)

    random_angle = 2 * torch.pi * torch.rand(batch, length, 4, device=device)
    random_sin_cos = torch.cat([torch.sin(random_angle)[..., None], torch.cos(random_angle)[..., None]], dim=-1)
    torsions_noised = torsion["torsion_angles_sin_cos"].clone()
    torsions_noised[:, :, 3:] = random_sin_cos * mask_fix_sc + torsions_true * (1 - mask_fix_sc)

    frames = torsion_angles_to_frames(backbone_frames, torsions_noised, aatype)
    xyz14_noised = frames_and_literature_positions_to_atom14_pos(frames, aatype)
    xyz14_m = atom_masks["atom14_atom_exists"] * mask[:, :, None]
    xyz14_noised = xyz14_noised * xyz14_m[:, :, :, None]

    return {
        "xyz14_m": xyz14_m,
        "xyz14_noised": xyz14_noised,
        "backbone_frames": backbone_frames,
        "torsions_noised": torsions_noised,
        "mask_fix_sc": mask_fix_sc,
        "torsions_true": torsions_true,
        "aatype": aatype,
    }


def _knn(points: Tensor, mask: Tensor, k_neighbors: int, eps: float = 1e-6) -> Tensor:
    r"""K nearest neighbors by CA distance, masking invalid residues to the far end."""

    pair_mask = mask[:, None, :] * mask[:, :, None]
    distances = torch.sqrt(torch.sum((points[:, :, None] - points[:, None, :]) ** 2, dim=-1) + eps)
    distances = pair_mask * distances
    max_distance = distances.max(dim=-1, keepdim=True).values
    adjusted = distances + (1.0 - pair_mask) * max_distance
    return torch.topk(adjusted, min(k_neighbors, points.shape[1]), dim=-1, largest=False).indices


def _rbf_bounds(distances: Tensor, num_bins: int, lower_bound: float, upper_bound: float) -> Tensor:
    r"""Radial basis expansion with explicit bounds (Packer uses [0, 20] for context features)."""

    centers = torch.linspace(lower_bound, upper_bound, num_bins, device=distances.device, dtype=distances.dtype)
    centers = centers.view(*((1,) * distances.ndim), num_bins)
    width = (upper_bound - lower_bound) / num_bins
    return torch.exp(-(((distances.unsqueeze(-1) - centers) / width) ** 2))


def _local_angle_features(n: Tensor, ca: Tensor, c: Tensor, points: Tensor) -> Tensor:
    r"""Direction features of ``points`` in each residue's local backbone frame."""

    first_axis = F.normalize(n - ca, dim=-1)
    c_vector = c - ca
    projected = torch.einsum("bli,bli->bl", first_axis, c_vector)[..., None]
    second_axis = F.normalize(c_vector - first_axis * projected, dim=-1)
    third_axis = torch.cross(first_axis, second_axis, dim=-1)
    frame = torch.stack([first_axis, second_axis, third_axis], dim=-1)
    local = torch.einsum("blqp,blyq->blyp", frame, points - ca[:, :, None])
    planar_radius = torch.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2 + 1e-8)
    spatial_radius = torch.norm(local, dim=-1) + 1e-8
    return torch.stack(
        [
            local[..., 0] / planar_radius,
            local[..., 1] / planar_radius,
            planar_radius / spatial_radius,
            local[..., 2] / spatial_radius,
        ],
        dim=-1,
    )
