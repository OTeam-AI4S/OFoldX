r"""
Reusable confidence mathematics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..atom import TokenToAtomBroadcast
from ..backend import LayerNorm


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
