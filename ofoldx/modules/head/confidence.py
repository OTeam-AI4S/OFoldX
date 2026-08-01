r"""
Confidence-score mathematics.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..atom import TokenToAtomBroadcast
from ..backend import LayerNorm


def expected_plddt(logits: Tensor) -> Tensor:
    r"""Return expected pLDDT on the 0--100 scale from confidence-bin logits."""

    centers = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype) + 0.5
    return (logits.softmax(dim=-1) * centers * (100.0 / logits.shape[-1])).sum(dim=-1)


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
        num_offsets, channels, bins = weight.shape
        if torch.compiler.is_compiling():
            torch._check_tensor_all(atom_token_index < num_offsets)
        elif atom_token_index.numel():
            max_index = int(atom_token_index.max())
            if max_index >= num_offsets:
                raise ValueError(
                    f"a token carries {max_index + 1} atoms but this confidence head was "
                    f"built for max_atoms_per_token={num_offsets}; raise it to cover the largest component"
                )
        gathered = F.embedding(atom_token_index, weight.reshape(num_offsets, channels * bins))
        return gathered.reshape(*gathered.shape[:-1], channels, bins)
