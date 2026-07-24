r"""
Confidence prediction from trunk representations and sampled coordinates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..atom import TokenToAtomBroadcast
from ..backend import LayerNorm
from ..trunk import PairformerStack
from .frames import gather_atoms_by_index, insert_sample_axis


def expected_plddt(logits: Tensor) -> Tensor:
    r"""
    Compute expected pLDDT from categorical logits.

    Args:
        logits: Confidence-bin logits with shape ``(..., num_bins)``.

    Returns:
        Expected pLDDT with shape ``(...)`` on the 0--100 scale.
    """

    centers = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype) + 0.5
    return (logits.softmax(dim=-1) * centers * (100.0 / logits.shape[-1])).sum(dim=-1)


class ConfidenceHead(nn.Module):
    r"""
    Predict pLDDT, PAE, PDE, and resolved logits.

    Args:
        single_size: Width of the single (token) representation.
        pair_size: Width of the pair representation.
        token_feature_size: Width of the packed per-token feature vector.
        distance_bin_start: Lower edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the representative-distance one-hot bins in Angstrom.
        distance_bin_step: Width of each representative-distance bin in Angstrom.
        num_plddt_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
        num_pae_bins: Number of predicted-aligned-error bins.
        num_pde_bins: Number of predicted-distance-error bins.
        num_blocks: Number of Pairformer blocks in the confidence stack.
        num_pair_attention_heads: Number of pair-attention heads in the Pairformer stack.
        triangle_multiplication_hidden_size: Hidden width of the triangle-multiplication updates.
        triangle_attention_hidden_size: Per-head width of the triangle-attention updates.
        num_triangle_attention_heads: Number of triangle-attention heads.
        expansion: Feed-forward expansion factor in the Pairformer transitions.
        dropout: Pairformer dropout probability.
        single_max_abs: Optional clamp on the single representation before scoring; ``None`` disables it.
        backend: Public backend request forwarded to the Pairformer and atom projector.
    """

    def __init__(
        self,
        single_size: int = 384,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
        num_blocks: int = 4,
        num_pair_attention_heads: int = 16,
        triangle_multiplication_hidden_size: int = 128,
        triangle_attention_hidden_size: int = 32,
        num_triangle_attention_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        single_max_abs: float | None = 512.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.single_max_abs = single_max_abs
        self.single_norm = LayerNorm(single_size, backend=backend)
        self.pair_embedder = ConfidencePairEmbedder(
            pair_size=pair_size,
            token_feature_size=token_feature_size,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
        )
        self.pairformer_stack = PairformerStack(
            num_blocks=num_blocks,
            single_size=single_size,
            pair_size=pair_size,
            triangle_multiplication_hidden_size=triangle_multiplication_hidden_size,
            triangle_attention_hidden_size=triangle_attention_hidden_size,
            num_triangle_attention_heads=num_triangle_attention_heads,
            num_pair_attention_heads=num_pair_attention_heads,
            expansion=expansion,
            dropout=dropout,
            backend=backend,
        )
        self.atom_projector = AtomConfidenceProjector(
            single_size=single_size,
            num_plddt_bins=num_plddt_bins,
            num_resolved_bins=num_resolved_bins,
            max_atoms_per_token=max_atoms_per_token,
            backend=backend,
        )
        self.pair_projector = PairConfidenceProjector(
            pair_size=pair_size,
            num_pae_bins=num_pae_bins,
            num_pde_bins=num_pde_bins,
        )

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        token_embedding: Tensor,
        pred_coords: Tensor,
        token_to_rep_atom: Tensor,
        atom_to_token: Tensor,
        atom_token_index: Tensor,
        pair_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Return atom-level and token-pair confidence logits for every predicted sample.

        Args:
            single: Token single representation ``(..., N_token, single_size)``.
            pair: Token pair representation ``(..., N_token, N_token, pair_size)``.
            token_embedding: Token embedding ``(..., N_token, token_feature_size)``.
            pred_coords: Predicted atom coords ``(..., N_sample, N_atom, 3)``.
            token_to_rep_atom: Representative atom index for each token ``(..., N_token)``.
            atom_to_token: Token index for each atom ``(..., N_atom)``.
            atom_token_index: Within-token atom offset for each atom ``(..., N_atom)``.
            pair_mask: Optional boolean token-pair validity mask ``(..., N_token, N_token)``.

        Returns:
            pLDDT, PAE, PDE, and resolved logits, each with a leading per-sample axis.
        """

        if self.single_max_abs is not None:
            single = torch.clamp(single, min=-self.single_max_abs, max=self.single_max_abs)
        single = self.single_norm(single)
        # Select each token's representative-atom coordinate by gathering on the per-token global atom index
        # (replaces a boolean mask over the atom axis; supports batched/sampled leading dims). pred_coords
        # is ``[..., N_sample, N_atom, 3]``; insert the sample axis into the per-token index before gathering.
        rep_index = token_to_rep_atom
        num_samples = pred_coords.shape[-3]
        sample_rep_index = insert_sample_axis(rep_index, num_samples)
        representative_coords = gather_atoms_by_index(pred_coords, sample_rep_index)

        plddt_logits = []
        pae_logits = []
        pde_logits = []
        resolved_logits = []
        for representative_coords_sample in representative_coords.unbind(dim=-3):
            sample_pair = self.pair_embedder(pair, token_embedding, representative_coords_sample)
            sample_single, sample_pair = self.pairformer_stack(single, sample_pair, pair_mask=pair_mask)
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


class ConfidencePairEmbedder(nn.Module):
    r"""
    Add token and representative-atom distance features to a pair state.

    Args:
        pair_size: Width of the pair representation.
        token_feature_size: Width of the packed per-token feature vector.
        distance_bin_start: Lower edge of the distance one-hot bins in Angstrom.
        distance_bin_end: Upper edge of the distance one-hot bins in Angstrom.
        distance_bin_step: Width of each distance bin in Angstrom.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        self.left_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.right_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        buffer_device = self.left_token_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        lower_bins = torch.arange(
            distance_bin_start,
            distance_bin_end,
            distance_bin_step,
            device=buffer_device,
            dtype=self.left_token_projection.weight.dtype,
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([torch.inf])], dim=-1)
        self.lower_bins: Tensor
        self.upper_bins: Tensor
        self.register_buffer("lower_bins", lower_bins)
        self.register_buffer("upper_bins", upper_bins)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)
        self.distance_projection = nn.Linear(1, pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        r"""
        Add token-projection and representative-distance channels to the pair state.

        Args:
            pair: Token pair state ``(..., N_token, N_token, pair_size)``.
            token_embedding: Token features ``(..., N_token, token_feature_size)``.
            representative_coords: Representative-atom coords ``(..., N_token, 3)``.

        Returns:
            Updated pair state ``(..., N_token, N_token, pair_size)``.
        """

        pair = pair + self.left_token_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.right_token_projection(token_embedding).unsqueeze(-3)
        # torch.cdist does not implement CUDA bf16; keep the cast local to this distance calculation.
        distance = torch.cdist(representative_coords.to(torch.float32), representative_coords.to(torch.float32))
        distance_bins = ((distance[..., None] > self.lower_bins) & (distance[..., None] < self.upper_bins)).to(
            pair.dtype
        )
        pair = pair + self.distance_one_hot_projection(distance_bins)
        pair = pair + self.distance_projection(distance.to(pair.dtype).unsqueeze(-1))
        return pair


class AtomConfidenceProjector(nn.Module):
    r"""
    Project token representations to per-atom pLDDT and resolved logits.

    Args:
        single_size: Width of the single (token) representation.
        num_plddt_bins: Number of pLDDT bins.
        num_resolved_bins: Number of resolved-classification bins.
        max_atoms_per_token: Maximum atoms per token, sizing the per-offset weight tables.
        backend: Public backend request forwarded to token-to-atom broadcast.
    """

    def __init__(
        self,
        single_size: int = 384,
        num_plddt_bins: int = 50,
        num_resolved_bins: int = 2,
        max_atoms_per_token: int = 24,
        backend: str = "auto",
    ):
        super().__init__()
        self.plddt_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_plddt_bins))
        self.resolved_weight = nn.Parameter(torch.empty(max_atoms_per_token, single_size, num_resolved_bins))
        self.plddt_norm = LayerNorm(single_size, backend=backend)
        self.resolved_norm = LayerNorm(single_size, backend=backend)
        self.token_to_atom_broadcast = TokenToAtomBroadcast(backend=backend)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        Zero-initialize the per-atom-offset pLDDT and resolved weight tensors.
        """

        nn.init.zeros_(self.plddt_weight)
        nn.init.zeros_(self.resolved_weight)

    def forward(self, single: Tensor, atom_to_token: Tensor, atom_token_index: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Broadcast token singles to atoms and project to per-atom pLDDT/resolved logits.

        Args:
            single: Token single representation ``(..., N_token, single_size)``.
            atom_to_token: Token index per atom ``(..., N_atom)``.
            atom_token_index: Within-token atom offset per atom ``(..., N_atom)``.

        Returns:
            ``(plddt_logits, resolved_logits)`` with shapes ``(..., N_atom, num_plddt_bins)`` and
            ``(..., N_atom, num_resolved_bins)``.
        """

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
        num_offsets, channels, bins = weight.shape
        gathered = F.embedding(atom_token_index, weight.reshape(num_offsets, channels * bins))
        return gathered.reshape(*gathered.shape[:-1], channels, bins)


class PairConfidenceProjector(nn.Module):
    r"""
    Project pair representations to PAE and symmetric PDE logits.

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
    ) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size)
        self.pde_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Project the pair state to PAE and symmetric PDE logits.

        Args:
            pair: Token pair state ``(..., N_token, N_token, pair_size)``.

        Returns:
            ``(pae_logits, pde_logits)`` with shapes ``(..., N_token, N_token, num_pae_bins)`` and
            ``(..., N_token, N_token, num_pde_bins)``.
        """

        pae_pair = pair.to(dtype=self.pae_projection.weight.dtype)
        pde_pair = pair.to(dtype=self.pde_projection.weight.dtype)
        pae_logits = self.pae_projection(self.pae_norm(pae_pair))
        pde_logits = self.pde_projection(self.pde_norm(pde_pair + pde_pair.transpose(-2, -3)))
        return pae_logits, pde_logits


class BinnedDistancePairEmbedder(nn.Module):
    r"""
    Add token features and binned representative-atom distances to an AlphaFold 3 pair state.
    """

    def __init__(
        self,
        pair_size: int = 128,
        token_feature_size: int = 449,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
    ) -> None:
        super().__init__()
        self.left_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        self.right_token_projection = nn.Linear(token_feature_size, pair_size, bias=False)
        buffer_device = self.left_token_projection.weight.device
        if buffer_device.type == "meta":
            buffer_device = torch.device("cpu")
        lower_bins = torch.arange(
            distance_bin_start,
            distance_bin_end,
            distance_bin_step,
            device=buffer_device,
            dtype=self.left_token_projection.weight.dtype,
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([torch.inf])], dim=-1)
        self.lower_bins: Tensor
        self.upper_bins: Tensor
        self.register_buffer("lower_bins", lower_bins, persistent=False)
        self.register_buffer("upper_bins", upper_bins, persistent=False)
        self.distance_one_hot_projection = nn.Linear(len(lower_bins), pair_size, bias=False)

    def forward(self, pair: Tensor, token_embedding: Tensor, representative_coords: Tensor) -> Tensor:
        pair = pair + self.left_token_projection(token_embedding).unsqueeze(-2)
        pair = pair + self.right_token_projection(token_embedding).unsqueeze(-3)
        distance = torch.cdist(representative_coords.to(torch.float32), representative_coords.to(torch.float32))
        distance_bins = ((distance[..., None] > self.lower_bins) & (distance[..., None] < self.upper_bins)).to(
            pair.dtype
        )
        return pair + self.distance_one_hot_projection(distance_bins)


class PostProjectionSymmetricPairConfidenceProjector(nn.Module):
    r"""
    Project AlphaFold 3 pair states and symmetrize PDE after nonlinear normalization.
    """

    def __init__(
        self,
        pair_size: int = 128,
        num_pae_bins: int = 64,
        num_pde_bins: int = 64,
    ) -> None:
        super().__init__()
        self.pae_norm = LayerNorm(pair_size)
        self.pde_norm = LayerNorm(pair_size)
        self.pae_projection = nn.Linear(pair_size, num_pae_bins, bias=False)
        self.pde_projection = nn.Linear(pair_size, num_pde_bins, bias=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        pae_pair = pair.to(dtype=self.pae_projection.weight.dtype)
        pde_pair = pair.to(dtype=self.pde_projection.weight.dtype)
        pae_logits = self.pae_projection(self.pae_norm(pae_pair))
        left_distance_logits = self.pde_projection(self.pde_norm(pde_pair))
        return pae_logits, left_distance_logits + left_distance_logits.transpose(-2, -3)
