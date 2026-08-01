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
Learned confidence prediction heads.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..atom import TokenToAtomBroadcast
from ..atom.coordinates import gather_atoms_by_index
from ..backend import LayerNorm
from ..backend.normalization import layer_norm_projection
from ..batch import insert_sample_axis
from ..functional import pairwise_distance


def token_confidence_logits(
    single: Tensor,
    pair: Tensor,
    *,
    pae_projection: nn.Linear | None,
    pde_projection: nn.Linear,
    plddt_projection: nn.Linear,
    resolved_projection: nn.Linear,
) -> dict[str, Tensor]:
    r"""Apply the complete learned token/pair confidence readout."""

    logits = {
        "pde_logits": pde_projection(pair + pair.transpose(-3, -2)),
        "plddt_logits": plddt_projection(single),
        "resolved_logits": resolved_projection(single),
    }
    if pae_projection is not None:
        logits["pae_logits"] = pae_projection(pair)
    return logits


class TokenConfidenceOutputHead(nn.Module):
    r"""
    Project token and pair states to raw pLDDT, resolved, PAE, and symmetric PDE logits.
    """

    def __init__(
        self,
        *,
        single_size: int,
        pair_size: int,
        num_plddt_bins: int = 50,
        num_pae_bins: int | None = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)
        self.plddt_projection = nn.Linear(single_size, num_plddt_bins, bias=False)
        self.resolved_projection = nn.Linear(single_size, 2, bias=False)
        self.pae_projection: nn.Linear | None = None
        if num_pae_bins is not None:
            self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)

    def forward(self, single: Tensor, pair: Tensor) -> dict[str, Tensor]:
        return token_confidence_logits(
            single,
            pair,
            pae_projection=self.pae_projection,
            pde_projection=self.pde_projection,
            plddt_projection=self.plddt_projection,
            resolved_projection=self.resolved_projection,
        )


class AtomConfidenceProjector(nn.Module):
    r"""Project token representations to per-atom pLDDT and resolved logits."""

    def __init__(
        self,
        single_size: int = 384,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.plddt_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_plddt_bins))
        self.resolved_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_resolved_bins))
        self.plddt_norm = LayerNorm(single_size, backend=backend)
        self.resolved_norm = LayerNorm(single_size, backend=backend)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.plddt_weight)
        nn.init.zeros_(self.resolved_weight)

    def forward(self, single: Tensor, atom_to_token: Tensor, atom_token_index: Tensor) -> tuple[Tensor, Tensor]:
        atom_single = self.token_to_atom_broadcast(single, atom_to_token)
        atom_token_index = atom_token_index.to(device=single.device, dtype=torch.long)
        plddt_weight = self._gather_atom_weight(self.plddt_weight, atom_token_index)
        resolved_weight = self._gather_atom_weight(self.resolved_weight, atom_token_index)
        plddt_hidden = self.plddt_norm(atom_single)
        resolved_hidden = self.resolved_norm(atom_single)
        plddt_logits = torch.einsum("...nc,...ncb->...nb", plddt_hidden, plddt_weight.to(plddt_hidden))
        resolved_logits = torch.einsum("...nc,...ncb->...nb", resolved_hidden, resolved_weight.to(resolved_hidden))
        return plddt_logits, resolved_logits

    def _gather_atom_weight(self, weight: Tensor, atom_token_index: Tensor) -> Tensor:
        num_offsets, single_size, num_bins = weight.shape
        if torch.compiler.is_compiling():
            torch._check_tensor_all(atom_token_index < num_offsets)
        elif atom_token_index.numel():
            max_index = int(atom_token_index.max())
            if max_index >= num_offsets:
                raise ValueError(
                    f"a token carries {max_index + 1} atoms but this confidence head was "
                    f"built for max_atoms_per_token={num_offsets}; raise it to cover the largest component"
                )
        gathered = F.embedding(atom_token_index, weight.reshape(num_offsets, single_size * num_bins))
        return gathered.reshape(*gathered.shape[:-1], single_size, num_bins)


class ConfidencePairEmbedder(nn.Module):
    r"""
    Add token features and binned representative-atom distances to a confidence pair state.

    The representative-atom distance enters through
    [`embed_distance`][ofoldx.modules.head.confidence.ConfidencePairEmbedder.embed_distance], which a
    family whose released embedder adds further distance terms overrides.

    Args:
        token_feature_size: Width of the incoming per-token feature vector.
        pair_size: Width of the pair representation.
        distance_bin_start: Lower edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_step: Width of each representative-distance bin in Angstrom.
    """

    def __init__(
        self,
        token_feature_size: int,
        pair_size: int = 128,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        self.pair_left_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.pair_right_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        buffer_device = self.pair_left_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        lower_bins = torch.arange(
            distance_bin_start,
            distance_bin_end,
            distance_bin_step,
            device=buffer_device,
            dtype=self.pair_left_projection.weight.dtype,
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([torch.inf])], dim=-1)
        self.lower_bins: Tensor
        self.upper_bins: Tensor
        self.register_buffer("lower_bins", lower_bins, persistent=False)
        self.register_buffer("upper_bins", upper_bins, persistent=False)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        pair = pair + self.pair_left_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.pair_right_projection(token_embedding).unsqueeze(-3)
        distance = pairwise_distance(representative_coords)
        return self.embed_distance(pair, distance)

    def embed_distance(self, pair: Tensor, distance: Tensor) -> Tensor:
        r"""
        Add the representative-atom distance contribution to the pair state.

        Args:
            pair: Pair state already carrying the left and right token projections.
            distance: Representative-atom distances in Angstrom, in the coordinate dtype.

        Returns:
            The pair state with the one-hot binned distance projection added.
            Bins exclude their lower edge and include their upper edge; distances
            at or below the first lower edge contribute no binned feature.
        """

        distance_bins = ((distance[..., None] > self.lower_bins) & (distance[..., None] <= self.upper_bins)).to(
            pair.dtype
        )
        return pair + self.distance_one_hot_projection(distance_bins)


class PairConfidenceProjector(nn.Module):
    r"""
    Project pair representations to PAE and symmetric PDE logits.

    The PDE branch lives in
    [`project_pde`][ofoldx.modules.head.confidence.PairConfidenceProjector.project_pde], which
    projects first and symmetrizes the resulting logits afterward. A family whose released head
    symmetrizes before the projection overrides it: LayerNorm does not commute with the sum, so
    the two orders are different computations, not two spellings of one.

    Args:
        pair_size: Width of the pair representation.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size, backend=backend)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_norm = LayerNorm(pair_size, backend=backend)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        pae_pair = pair.to(dtype=self.pae_projection.weight.dtype)
        pde_pair = pair.to(dtype=self.pde_projection.weight.dtype)
        return layer_norm_projection(pae_pair, norm=self.pae_norm, projection=self.pae_projection), self.project_pde(
            pde_pair
        )

    def project_pde(self, pair: Tensor) -> Tensor:
        r"""
        Project the pair representation to symmetric predicted-distance-error logits.

        Args:
            pair: Pair representation cast to the PDE projection dtype.

        Returns:
            PDE logits, projected per ordered pair and symmetrized afterward.
        """

        left_distance_logits = layer_norm_projection(pair, norm=self.pde_norm, projection=self.pde_projection)
        return left_distance_logits + left_distance_logits.transpose(-2, -3)


class ConfidenceHead(nn.Module):
    r"""
    Predict pLDDT, PAE, PDE, and resolved logits from trunk representations.

    A family whose released head differs overrides the pieces that differ:
    [`build_pair_embedder`][ofoldx.modules.head.confidence.ConfidenceHead.build_pair_embedder] and
    [`build_pair_projector`][ofoldx.modules.head.confidence.ConfidenceHead.build_pair_projector] to
    supply its own components, and
    [`prepare_single`][ofoldx.modules.head.confidence.ConfidenceHead.prepare_single] to change how
    the single representation is conditioned before scoring.

    Args:
        token_feature_size: Width of the incoming per-token feature vector consumed by the pair
            embedder.
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        distance_bin_start: Lower edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_step: Width of each representative-distance bin in Angstrom.
        num_plddt_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        single_max_abs: Optional clamp on the single representation before scoring; ``None``
            disables it.
        backend: Public backend request forwarded to every dispatched normalization.
    """

    def __init__(
        self,
        token_feature_size: int,
        single_size: int = 384,
        pair_size: int = 128,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        single_max_abs: float | None = 512.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_max_abs = single_max_abs
        self.pair_embedder = self.build_pair_embedder(
            token_feature_size=token_feature_size,
            pair_size=pair_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
            backend=backend,
        )
        self.pair_projector = self.build_pair_projector(
            pair_size=pair_size,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            backend=backend,
        )

    def build_pair_embedder(
        self,
        *,
        token_feature_size: int,
        pair_size: int,
        distance_bin_start: float,
        distance_bin_end: float,
        distance_bin_step: float,
    ) -> ConfidencePairEmbedder:
        r"""
        Build the pair embedder this head runs once per sample.

        Returns:
            A [`ConfidencePairEmbedder`][ofoldx.modules.head.confidence.ConfidencePairEmbedder].
        """

        return ConfidencePairEmbedder(
            token_feature_size=token_feature_size,
            pair_size=pair_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )

    def build_pair_projector(
        self, *, pair_size: int, num_pae_bins: int, num_pde_bins: int, backend: str = "auto"
    ) -> PairConfidenceProjector:
        r"""
        Build the projector turning the refined pair state into PAE and PDE logits.

        Returns:
            A [`PairConfidenceProjector`][ofoldx.modules.head.confidence.PairConfidenceProjector].
        """

        return PairConfidenceProjector(
            pair_size=pair_size,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
            backend=backend,
        )

    def prepare_single(self, single: Tensor) -> Tensor:
        r"""
        Condition the trunk single representation before it enters the Pairformer stack.

        Args:
            single: Trunk single representation.

        Returns:
            The single representation clamped to ``single_max_abs`` when that is set.
        """

        if self.single_max_abs is None:
            return single
        return torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pred_coords: Tensor,
        token_representative_atom_index: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
        pairformer_stack: nn.Module,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Return atom-level and token-pair confidence logits for every predicted sample.

        Args:
            single: Trunk single representation.
            pair: Trunk pair representation.
            token_embedding: Per-token feature vector consumed by the pair embedder.
            pred_coords: Predicted atom coordinates for every sample.
            token_representative_atom_index: Index of the representative atom for each token.
            atom_to_token: Token index for each atom.
            atom_token_index: Per-token atom offset for each atom.
            pairformer_stack: Caller-owned Pairformer composition run once per sample, called as
                ``pairformer_stack(single, pair, pair_mask=pair_mask) -> (single, pair)``. The
                confidence head runs whichever concrete stack its family trunk uses; it does not
                own that composition.
            pair_mask: Optional token-pair mask forwarded to ``pairformer_stack``.
        """

        single = self.prepare_single(single)
        representative_coords = gather_atoms_by_index(
            pred_coords,
            insert_sample_axis(token_representative_atom_index, pred_coords.shape[-3]),
        )

        plddt_logits = []
        pae_logits = []
        pde_logits = []
        resolved_logits = []
        for representative_coords_sample in representative_coords.unbind(dim=-3):
            sample_pair = self.pair_embedder(pair, token_embedding, representative_coords_sample)
            sample_single, sample_pair = pairformer_stack(single, sample_pair, pair_mask=pair_mask)
            sample_plddt, sample_resolved = self.atom_projector(sample_single, atom_to_token, atom_token_index)
            sample_pae, sample_pde = self.pair_projector(sample_pair)
            plddt_logits.append(sample_plddt)
            pae_logits.append(sample_pae)
            pde_logits.append(sample_pde)
            resolved_logits.append(sample_resolved)

        return (
            torch.stack(plddt_logits, dim=-3),
            torch.stack(pae_logits, dim=-4),
            torch.stack(pde_logits, dim=-4),
            torch.stack(resolved_logits, dim=-3),
        )
