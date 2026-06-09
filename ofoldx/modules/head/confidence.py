r"""
Confidence cross-entropy losses.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .distogram import distance_bin_indices, masked_cross_entropy
from .frames import frame_aligned_error_squared, gather_atoms_by_index, gather_frame_atoms, insert_sample_axis
from .lddt import compute_per_atom_lddt, lddt_bin_labels

_MOLECULE_RNA, _MOLECULE_DNA, _MOLECULE_LIGAND = 1, 2, 3


class ConfidenceLoss(nn.Module):
    r"""
    Cross-entropy losses for the confidence logits.

    Bundles the pLDDT, resolved-atom, predicted-aligned-error (PAE) and predicted-distance-error
    (PDE) heads: [forward][ofoldx.modules.head.confidence.ConfidenceLoss.forward] derives the
    per-head binned targets from predicted and ground-truth coords and averages the available
    cross-entropy terms. Only pLDDT is mandatory; the other three are computed when their logits are
    supplied.

    Args:
        distance_error_max: Maximum error distance used to bin the aligned and pair distance errors.

    Examples:
        >>> loss = ConfidenceLoss(distance_error_max=32.0)
        >>> loss.distance_error_max
        32.0
    """

    def __init__(self, distance_error_max: float = 32.0) -> None:
        super().__init__()
        self.distance_error_max = distance_error_max

    def forward(
        self,
        *,
        pred_coords: Tensor,
        target_positions: Tensor,
        target_mask: Tensor,
        representative_atom_mask: Tensor,
        token_to_rep_atom: Tensor,
        frame_atom_index: Tensor,
        frame_mask: Tensor,
        molecule_type: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        plddt_logits: Tensor,
        resolved_logits: Tensor | None = None,
        pae_logits: Tensor | None = None,
        pde_logits: Tensor | None = None,
    ) -> Tensor:
        r"""
        Return the average confidence loss over available logits.

        ``representative_atom_mask`` is the per-atom polymer-neighbour mask used multiplicatively by
        the pLDDT target; ``token_to_rep_atom`` is the per-token global rep-atom gather index used to
        select representative coords for the frame/pair losses (a per-element gather that supports
        batched/sampled leading dims, replacing a boolean mask over the atom axis).

        Args:
            pred_coords: `[..., num_samples, N_atom, 3]` predicted coords.
            target_positions: `[..., N_atom, 3]` target coords.
            target_mask: `[..., N_atom]` valid-target mask.
            representative_atom_mask: `[..., N_atom]` neighbour-eligibility mask for LDDT.
            token_to_rep_atom: `[..., N_token]` representative atom indices.
            frame_atom_index: `[..., N_token, 3]` frame atom indices.
            frame_mask: `[..., N_token]` frame-validity mask.
            molecule_type: `[..., N_token]` molecule type ids.
            atom_to_token: `[..., N_atom]` token index for each atom.
            atom_mask: `[..., N_atom]` valid-atom mask.
            plddt_logits: pLDDT logits (mandatory).
            resolved_logits: Optional resolved-atom logits.
            pae_logits: Optional predicted aligned-error logits.
            pde_logits: Optional predicted distance-error logits.

        Returns:
            Scalar average over the available confidence loss terms.

        Examples:
            >>> import torch  # doctest: +SKIP
            >>> loss = ConfidenceLoss()  # doctest: +SKIP
            >>> loss(  # doctest: +SKIP
            ...     pred_coords=pred_coords,
            ...     target_positions=target_positions,
            ...     target_mask=target_mask,
            ...     representative_atom_mask=representative_atom_mask,
            ...     token_to_rep_atom=token_to_rep_atom,
            ...     frame_atom_index=frame_atom_index,
            ...     frame_mask=frame_mask,
            ...     molecule_type=molecule_type,
            ...     atom_to_token=atom_to_token,
            ...     atom_mask=atom_mask,
            ...     plddt_logits=plddt_logits,
            ... )
        """

        losses = [
            self._plddt_loss(
                plddt_logits,
                pred_coords=pred_coords,
                target_positions=target_positions,
                target_mask=target_mask,
                representative_atom_mask=representative_atom_mask,
                token_to_rep_atom=token_to_rep_atom,
                molecule_type=molecule_type,
                atom_to_token=atom_to_token,
            )
        ]
        if resolved_logits is not None:
            losses.append(
                self._resolved_loss(
                    resolved_logits,
                    target_mask=target_mask,
                    atom_mask=atom_mask,
                    token_to_rep_atom=token_to_rep_atom,
                )
            )
        if pae_logits is not None:
            losses.append(
                self._frame_aligned_error_loss(
                    pae_logits,
                    pred_coords=pred_coords,
                    target_positions=target_positions,
                    target_mask=target_mask,
                    token_to_rep_atom=token_to_rep_atom,
                    frame_atom_index=frame_atom_index,
                    frame_mask=frame_mask,
                )
            )
        if pde_logits is not None:
            losses.append(
                self._pair_error_loss(
                    pde_logits,
                    pred_coords=pred_coords,
                    target_positions=target_positions,
                    target_mask=target_mask,
                    token_to_rep_atom=token_to_rep_atom,
                )
            )
        return torch.stack(losses).mean()

    def _plddt_loss(
        self,
        logits: Tensor,
        *,
        pred_coords: Tensor,
        target_positions: Tensor,
        target_mask: Tensor,
        representative_atom_mask: Tensor,
        token_to_rep_atom: Tensor,
        molecule_type: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        r"""
        Per-atom LDDT target binned for the pLDDT cross-entropy.

        The label is the relative-distance-preservation score against polymer representative-atom
        neighbours within the locality radius, NOT an absolute positional error.

        Args:
            logits: pLDDT logits with the bin axis last, targeting either atoms or tokens.
            pred_coords: `[..., num_samples, N_atom, 3]` predicted coords.
            target_positions: `[..., N_atom, 3]` target coords.
            target_mask: `[..., N_atom]` valid-target mask.
            representative_atom_mask: `[..., N_atom]` neighbour-eligibility mask.
            token_to_rep_atom: `[..., N_token]` representative atom indices.
            molecule_type: `[..., N_token]` molecule type ids.
            atom_to_token: `[..., N_atom]` token index for each atom.

        Returns:
            Scalar masked cross-entropy loss for the pLDDT head.
        """

        target_positions = target_positions.to(device=logits.device, dtype=logits.dtype)
        pred_coords = pred_coords.to(dtype=logits.dtype)
        atom_molecule_type = torch.gather(
            molecule_type.to(device=logits.device), -1, atom_to_token.to(device=logits.device)
        )  # [..., N_atom]
        is_nucleotide = (atom_molecule_type == _MOLECULE_RNA) | (atom_molecule_type == _MOLECULE_DNA)
        is_polymer = atom_molecule_type != _MOLECULE_LIGAND
        neighbor_mask = representative_atom_mask.to(device=logits.device).bool() & is_polymer

        per_atom_lddt, per_atom_weight = compute_per_atom_lddt(
            pred_coords,
            target_positions,
            neighbor_mask=neighbor_mask,
            is_nucleotide=is_nucleotide,
        )
        labels = lddt_bin_labels(per_atom_lddt, per_atom_weight, num_bins=logits.shape[-1])
        labels, mask = _match_atom_or_token_axis(
            logits,
            atom_labels=labels,
            atom_mask=target_mask.to(device=logits.device).bool(),
            token_to_rep_atom=token_to_rep_atom,
        )
        return masked_cross_entropy(logits, labels, mask)

    @staticmethod
    def _resolved_loss(
        logits: Tensor,
        *,
        target_mask: Tensor,
        atom_mask: Tensor,
        token_to_rep_atom: Tensor,
    ) -> Tensor:
        r"""
        Binary resolved-atom cross-entropy over atoms or tokens.

        Args:
            logits: Resolved-atom logits with a 2-class axis last, targeting atoms or tokens.
            target_mask: `[..., N_atom]` whether a resolved coordinate exists (the 2-class target).
            atom_mask: `[..., N_atom]` whether the atom physically exists (the averaging mask).
            token_to_rep_atom: `[..., N_token]` representative atom indices for a token-level head.

        Returns:
            Scalar masked cross-entropy loss for the resolved-atom head.

        Raises:
            ValueError: If ``logits`` targets neither the atom nor the token axis.
        """

        # Per-atom 2-class target: whether a resolved coordinate exists (``target_mask``). The loss is masked by
        # ``atom_mask`` (whether the atom physically exists) so padded atoms do not pollute the mean.
        target_mask = target_mask.to(device=logits.device).bool()
        atom_mask = atom_mask.to(device=logits.device).bool()
        num_samples = logits.shape[-3]
        if logits.shape[-2] == target_mask.shape[-1]:
            labels = insert_sample_axis(target_mask.long(), num_samples, dim=-2)
            mask = insert_sample_axis(atom_mask, num_samples, dim=-2)
        elif logits.shape[-2] == token_to_rep_atom.shape[-1]:
            rep_index = token_to_rep_atom.to(device=logits.device, dtype=torch.long)
            labels = insert_sample_axis(torch.gather(target_mask.long(), -1, rep_index), num_samples, dim=-2)
            mask = insert_sample_axis(torch.gather(atom_mask, -1, rep_index), num_samples, dim=-2)
        else:
            raise ValueError(
                "resolved logits must target either atoms or tokens: "
                f"logits axis={logits.shape[-2]}, atoms={target_mask.shape[-1]}, "
                f"tokens={token_to_rep_atom.shape[-1]}"
            )
        return masked_cross_entropy(logits, labels, mask)

    def _frame_aligned_error_loss(
        self,
        logits: Tensor,
        *,
        pred_coords: Tensor,
        target_positions: Tensor,
        target_mask: Tensor,
        token_to_rep_atom: Tensor,
        frame_atom_index: Tensor,
        frame_mask: Tensor,
    ) -> Tensor:
        r"""
        Frame-based predicted aligned error binned into PAE classes.

        Each token's representative atom is expressed in every token's ``(N, CA, C)`` frame for both
        the prediction and ground truth; the per-frame/per-token distance between the two becomes the
        binned PAE label, masked to valid frames and resolved tokens.

        Args:
            logits: PAE logits with the bin axis last.
            pred_coords: `[..., num_samples, N_atom, 3]` predicted coords.
            target_positions: `[..., N_atom, 3]` target coords.
            target_mask: `[..., N_atom]` valid-target mask.
            token_to_rep_atom: `[..., N_token]` representative atom indices.
            frame_atom_index: `[..., N_token, 3]` frame atom indices.
            frame_mask: `[..., N_token]` frame-validity mask.

        Returns:
            Scalar masked cross-entropy loss for the PAE head.
        """

        target_positions = target_positions.to(device=logits.device, dtype=logits.dtype)
        pred_coords = pred_coords.to(dtype=logits.dtype)
        frame_atom_index = frame_atom_index.to(device=logits.device)
        rep_index = token_to_rep_atom.to(device=logits.device, dtype=torch.long)  # [..., N_token]
        num_samples = pred_coords.shape[-3]

        # Insert the sample axis (frame index has a trailing static 3, so the sample axis goes at -3)
        # and expand only that axis so the token axis is left as-is.
        sample_frame_index = insert_sample_axis(frame_atom_index, num_samples, dim=-3)
        pred_frames = gather_frame_atoms(pred_coords, sample_frame_index)  # [..., S, N_token, 3, 3]
        true_frames = gather_frame_atoms(target_positions, frame_atom_index)  # [..., N_token, 3, 3]

        # Gather each token's representative-atom centre by per-token global atom index.
        sample_rep_index = insert_sample_axis(rep_index, num_samples)
        pred_centers = gather_atoms_by_index(pred_coords, sample_rep_index)  # [..., S, N_token, 3]
        true_centers = gather_atoms_by_index(target_positions, rep_index)  # [..., N_token, 3]
        squared_error = frame_aligned_error_squared(pred_centers, true_centers, pred_frames, true_frames)
        errors = squared_error.clamp_min(0.0).sqrt()  # [..., S, N_frame, N_token]

        labels = distance_bin_indices(
            errors,
            num_bins=logits.shape[-1],
            min_distance=0.0,
            max_distance=self.distance_error_max,
        )
        frame_mask = frame_mask.to(device=logits.device).bool()  # [..., N_token]
        token_mask = torch.gather(target_mask.to(device=logits.device).bool(), -1, rep_index)  # [..., N_token]
        pair_mask = frame_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)  # [..., N_frame, N_token]
        # Add the sample axis (-1 on the other dims) so the mask aligns with ``labels`` for the cross entropy.
        pair_mask = insert_sample_axis(pair_mask, labels.shape[-3], dim=-3)
        return masked_cross_entropy(logits, labels, pair_mask)

    def _pair_error_loss(
        self,
        logits: Tensor,
        *,
        pred_coords: Tensor,
        target_positions: Tensor,
        target_mask: Tensor,
        token_to_rep_atom: Tensor,
    ) -> Tensor:
        r"""
        Predicted-distance-error cross-entropy over token-pair representative distances.

        The absolute difference between predicted and target pairwise representative-atom distances
        is binned into PDE classes and compared against ``logits``, masked to token pairs whose both
        tokens have a resolved target.

        Args:
            logits: PDE logits with the bin axis last.
            pred_coords: `[..., num_samples, N_atom, 3]` predicted coords.
            target_positions: `[..., N_atom, 3]` target coords.
            target_mask: `[..., N_atom]` valid-target mask.
            token_to_rep_atom: `[..., N_token]` representative atom indices.

        Returns:
            Scalar masked cross-entropy loss for the PDE head.
        """

        target_positions = target_positions.to(device=logits.device, dtype=logits.dtype)
        rep_index = token_to_rep_atom.to(device=logits.device, dtype=torch.long)  # [..., N_token]
        num_samples = pred_coords.shape[-3]

        # Gather each token's representative-atom coordinate by per-token global atom index.
        sample_rep_index = insert_sample_axis(rep_index, num_samples)
        predicted_representatives = gather_atoms_by_index(pred_coords, sample_rep_index).to(logits.dtype)
        target_representatives = gather_atoms_by_index(target_positions, rep_index)
        target_representatives = target_representatives.unsqueeze(-3)
        token_mask = torch.gather(target_mask.to(device=logits.device).bool(), -1, rep_index)  # [..., N_token]
        # Pairwise representative-atom distances.
        pred = predicted_representatives.float()
        predicted_distances = torch.cdist(pred, pred).to(logits.dtype)
        tgt = target_representatives.float()
        target_distances = torch.cdist(tgt, tgt).to(logits.dtype)
        distance_errors = (predicted_distances - target_distances).abs()
        labels = distance_bin_indices(
            distance_errors,
            num_bins=logits.shape[-1],
            min_distance=0.0,
            max_distance=self.distance_error_max,
        )
        pair_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        # Add the sample axis (-1 on the other dims) so the mask aligns with ``labels``.
        pair_mask = insert_sample_axis(pair_mask, labels.shape[-3], dim=-3)
        return masked_cross_entropy(logits, labels, pair_mask)


def _match_atom_or_token_axis(
    logits: Tensor,
    *,
    atom_labels: Tensor,
    atom_mask: Tensor,
    token_to_rep_atom: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Return labels/mask aligned to either an atom-level or token-level confidence head.

    The head is inferred from ``logits.shape[-2]``: when it matches the atom count the atom-level
    labels/mask pass through, and when it matches the token count they are gathered down to the
    per-token representative atoms.

    Args:
        logits: Confidence logits whose ``-2`` axis is either the atom or token count.
        atom_labels: `[..., num_samples, N_atom]` atom-level integer bin labels.
        atom_mask: `[..., N_atom]` atom-level averaging mask.
        token_to_rep_atom: `[..., N_token]` representative atom indices.

    Returns:
        ``(labels, mask)`` aligned to the inferred head, each with a materialized sample axis.

    Raises:
        ValueError: If ``logits`` targets neither the atom nor the token axis.
    """

    num_samples = atom_labels.shape[-2]
    if logits.shape[-2] == atom_labels.shape[-1]:
        return atom_labels, insert_sample_axis(atom_mask, num_samples, dim=-2)
    if logits.shape[-2] == token_to_rep_atom.shape[-1]:
        rep_index = token_to_rep_atom.to(device=logits.device, dtype=torch.long)
        sample_rep_index = insert_sample_axis(rep_index, num_samples, dim=-2)
        token_labels = torch.gather(atom_labels, -1, sample_rep_index)
        token_mask = torch.gather(atom_mask, -1, rep_index)
        return token_labels, insert_sample_axis(token_mask, num_samples, dim=-2)
    raise ValueError(
        "pLDDT logits must target either atoms or tokens: "
        f"logits axis={logits.shape[-2]}, atoms={atom_labels.shape[-1]}, tokens={token_to_rep_atom.shape[-1]}"
    )
