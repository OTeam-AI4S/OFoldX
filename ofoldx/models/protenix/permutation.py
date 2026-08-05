r"""
Prediction-dependent truth permutation for Protenix training.

This module deliberately owns the Protenix RMSD matching rule.  The shared
``modules.symmetry`` matcher implements the lDDT rule used by other model
families and is therefore not interchangeable with the upstream Protenix
training contract.
"""

from __future__ import annotations

import operator
import random
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ...data import ENTITY_MOLECULE


class ProtenixPermutationTarget(Protocol):
    r"""Structural target fields consumed by [`match_protenix_truth`][]."""

    coordinates: Tensor
    coordinate_mask: Tensor
    molecule_id: Tensor
    molecule_equivalence_index: Tensor
    molecule_atom_index: Tensor
    representative_atom_mask: Tensor
    permutation_pool_mask: Tensor
    atom_molecule_type: Tensor
    symmetry_atom_index: Tensor
    symmetry_permutation: Tensor


class ProtenixPermutationInput(Protocol):
    r"""Model-axis to full-axis mapping consumed by [`match_protenix_truth`][]."""

    atom_index: Tensor
    atom_to_token: Tensor


@dataclass(frozen=True)
class ProtenixPredictionPermutation:
    r"""Canonicalized labeled-evaluation prediction and its gather indices.

    ``atom_permutation`` and ``token_permutation`` use output-to-input gather
    semantics.  The output positions remain on the current model axes: atom
    position ``j`` represents full atom ``input.atom_index[b, j]`` rather than
    numeric full atom ``j``.  Consequently, labeled truth must also be gathered
    through ``input.atom_index``, and this object can keep confidence heads in
    lockstep with the independently matched diffusion samples.
    """

    coordinates: Tensor
    atom_permutation: Tensor
    token_permutation: Tensor

    def gather_atom_axis(self, values: Tensor) -> Tensor:
        r"""Gather a ``(B, S, A, ...)`` pLDDT/resolved-style atom head."""

        if values.ndim < 3 or values.shape[:3] != self.atom_permutation.shape:
            raise ValueError("atom heads must have shape (B, S, A, ...)")
        permutation = self.atom_permutation.to(device=values.device)
        gather_index = permutation.reshape(*permutation.shape, *((1,) * (values.ndim - 3)))
        return values.gather(2, gather_index.expand_as(values))

    def gather_token_pair_axes(self, values: Tensor) -> Tensor:
        r"""Gather both token axes of a ``(B, S, N, N, ...)`` PAE/PDE head."""

        batch_size, sample_count, token_count = self.token_permutation.shape
        if values.ndim < 4 or values.shape[:4] != (
            batch_size,
            sample_count,
            token_count,
            token_count,
        ):
            raise ValueError("token-pair heads must have shape (B, S, N, N, ...)")
        permutation = self.token_permutation.to(device=values.device)
        trailing_shape = values.shape[4:]
        trailing_singletons = (1,) * len(trailing_shape)
        row_index = permutation.reshape(
            batch_size,
            sample_count,
            token_count,
            1,
            *trailing_singletons,
        ).expand(batch_size, sample_count, token_count, token_count, *trailing_shape)
        gathered = values.gather(2, row_index)
        column_index = permutation.reshape(
            batch_size,
            sample_count,
            1,
            token_count,
            *trailing_singletons,
        ).expand(batch_size, sample_count, token_count, token_count, *trailing_shape)
        return gathered.gather(3, column_index)

    def gather_shared_token_pair_axes(self, values: Tensor) -> Tensor:
        r"""Broadcast and gather both axes of a shared ``(B, N, N, ...)`` value.

        The returned tensor has shape ``(B, S, N, N, ...)``. Every sample uses
        its own ``token_permutation`` for both the row and column axis, while
        trailing value axes are retained exactly.
        """

        batch_size, sample_count, token_count = self.token_permutation.shape
        if values.ndim < 3 or values.shape[:3] != (
            batch_size,
            token_count,
            token_count,
        ):
            raise ValueError("shared token-pair values must have shape (B, N, N, ...)")
        trailing_shape = values.shape[3:]
        shared_values = values.unsqueeze(1).expand(
            batch_size,
            sample_count,
            token_count,
            token_count,
            *trailing_shape,
        )
        return self.gather_token_pair_axes(shared_values)

    def gather_heads(
        self,
        *,
        plddt_logits: Tensor | None = None,
        resolved_logits: Tensor | None = None,
        pae_logits: Tensor | None = None,
        pde_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        r"""Return supplied Protenix confidence heads in canonical order."""

        output = {}
        if plddt_logits is not None:
            output["plddt_logits"] = self.gather_atom_axis(plddt_logits)
        if resolved_logits is not None:
            output["resolved_logits"] = self.gather_atom_axis(resolved_logits)
        if pae_logits is not None:
            output["pae_logits"] = self.gather_token_pair_axes(pae_logits)
        if pde_logits is not None:
            output["pde_logits"] = self.gather_token_pair_axes(pde_logits)
        return output


@torch.no_grad()
def match_protenix_truth(
    predicted_coordinates: Tensor,
    target: ProtenixPermutationTarget,
    input: ProtenixPermutationInput,
    *,
    max_molecules: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Match full-axis Protenix truth to one predicted model atom axis.

    The molecule stage is the released Protenix random-anchor heuristic.  It
    chooses an anchor among valid representative-atom molecules, enumerates
    equivalent truth anchors, greedily assigns the remaining molecules by
    aligned center distance, and retains the heuristic assignment only when its
    globally aligned RMSD improves on identity by at least ``1e-12``.  Released
    training is unlimited; callers may opt into a positive ``max_molecules``
    safety boundary.

    Once a molecule assignment has been selected, the prediction is globally
    aligned to its resolved truth.  Each carried local atom permutation is then
    compared by RMSD in that single fixed frame.  Coordinates and resolved masks
    are permuted together.  Local ties retain the identity-first candidate.

    Ground-truth coordinates remain float32 throughout.  Lower-precision
    predictions are converted to float32 only inside alignment and scoring.

    Args:
        predicted_coordinates: Coordinates with shape ``(B, A, 3)`` or the
            single-sample shape ``(B, 1, A, 3)``.
        target: Full-axis coordinates and molecule/local-symmetry metadata.
        input: Current model-axis mapping through ``atom_index``.
        max_molecules: Optional maximum number of truth-pool molecules. ``None``
            preserves released Protenix's unlimited training behavior.

    Returns:
        Float32 matched coordinates with shape ``(B, A, 3)`` and their resolved
        mask with shape ``(B, A)``.  Both tensors are detached by construction.
    """

    if max_molecules is not None:
        if isinstance(max_molecules, bool):
            raise TypeError("max_molecules must be an integer or None, not bool")
        max_molecules = operator.index(max_molecules)
        if max_molecules < 1:
            raise ValueError("max_molecules must be positive or None")

    predicted = _single_prediction(predicted_coordinates)
    batch_size, atom_count = predicted.shape[:2]
    device = predicted.device
    # Baseline matching and all downstream matched labels are FP32 even when
    # the mini-rollout prediction was produced under bf16 autocast.
    target_coordinates = target.coordinates.to(device=device, dtype=torch.float32)
    target_mask = target.coordinate_mask.to(device=device, dtype=torch.bool)
    molecule_id = target.molecule_id.to(device=device, dtype=torch.long)
    molecule_equivalence = target.molecule_equivalence_index.to(device=device, dtype=torch.long)
    molecule_atom_index = target.molecule_atom_index.to(device=device, dtype=torch.long)
    representative_mask = target.representative_atom_mask.to(device=device, dtype=torch.bool)
    pool_mask = target.permutation_pool_mask.to(device=device, dtype=torch.bool)
    molecule_type = target.atom_molecule_type.to(device=device, dtype=torch.long)
    symmetry_atom_index = target.symmetry_atom_index.to(device=device, dtype=torch.long)
    symmetry_permutation = target.symmetry_permutation.to(device=device, dtype=torch.long)
    atom_index = input.atom_index.to(device=device, dtype=torch.long)

    _validate_shapes(
        predicted,
        target_coordinates,
        target_mask,
        molecule_id,
        molecule_equivalence,
        molecule_atom_index,
        representative_mask,
        pool_mask,
        molecule_type,
        symmetry_atom_index,
        symmetry_permutation,
        atom_index,
    )
    if atom_index.shape != (batch_size, atom_count):
        raise ValueError("input.atom_index must have shape (B, A) matching predicted coordinates")

    matched_coordinates = torch.empty(
        (batch_size, atom_count, 3),
        dtype=torch.float32,
        device=device,
    )
    matched_mask = torch.empty((batch_size, atom_count), dtype=torch.bool, device=device)
    full_atom_count = target_coordinates.shape[1]
    for batch_index in range(batch_size):
        current_atom_index = atom_index[batch_index]
        valid_atom = (current_atom_index >= 0) & (current_atom_index < full_atom_count)
        safe_atom_index = current_atom_index.clamp(0, full_atom_count - 1)
        identity_coordinates = target_coordinates[batch_index].index_select(0, safe_atom_index)
        identity_coordinates = identity_coordinates.masked_fill(~valid_atom.unsqueeze(-1), 0)
        identity_mask = target_mask[batch_index].index_select(0, safe_atom_index) & valid_atom

        matched_atom_index = _match_molecule_indices(
            predicted[batch_index],
            current_atom_index,
            valid_atom,
            safe_atom_index,
            target_coordinates[batch_index],
            target_mask[batch_index],
            molecule_id[batch_index],
            molecule_equivalence[batch_index],
            molecule_atom_index[batch_index],
            representative_mask[batch_index],
            pool_mask[batch_index],
            molecule_type[batch_index],
            max_molecules=max_molecules,
        )
        current_coordinates = target_coordinates[batch_index].index_select(0, matched_atom_index)
        current_coordinates = current_coordinates.masked_fill(~valid_atom.unsqueeze(-1), 0)
        current_mask = target_mask[batch_index].index_select(0, matched_atom_index) & valid_atom

        # Atom matching keeps the selected chain assignment but falls back to
        # its identity local permutation when fewer than three atoms resolve the
        # global frame.
        aligned_prediction = _align_predicted(
            predicted[batch_index],
            current_coordinates,
            current_mask,
            minimum_atoms=3,
        )
        if aligned_prediction is not None:
            current_coordinates, current_mask = _match_local_symmetry(
                aligned_prediction,
                current_coordinates,
                current_mask,
                symmetry_atom_index[batch_index],
                symmetry_permutation[batch_index],
            )
        matched_coordinates[batch_index] = current_coordinates
        matched_mask[batch_index] = current_mask

    return matched_coordinates, matched_mask


@torch.no_grad()
def match_protenix_prediction(
    predicted_coordinates: Tensor,
    target: ProtenixPermutationTarget,
    input: ProtenixPermutationInput,
    *,
    max_molecules: int | None = None,
) -> ProtenixPredictionPermutation:
    r"""Canonicalize labeled-evaluation diffusion samples prediction-side.

    Chain and local-atom matching are performed independently for every sample
    in ``(B, S, A, 3)``.  Unlike the training mini-rollout API, evaluation is
    unlimited by default.  Prediction-side inversion is defined only for the
    full canonical atom set: cropped, padded, duplicated, or incomplete
    ``input.atom_index`` rows are rejected rather than silently producing a
    partial permutation.

    Args:
        predicted_coordinates: Diffusion samples with shape ``(B, S, A, 3)``.
        target: Full-axis labeled structure and permutation metadata.
        input: Current-to-canonical atom mapping and atom-to-token mapping.
        max_molecules: Optional molecule-count guard. ``None`` means unlimited.

    Returns:
        Coordinates ordered on the current model axis, with each position bound
        to its canonical full atom through ``input.atom_index``, plus atom/token
        gather permutations.
    """

    if max_molecules is not None:
        if isinstance(max_molecules, bool):
            raise TypeError("max_molecules must be an integer or None, not bool")
        max_molecules = operator.index(max_molecules)
        if max_molecules < 1:
            raise ValueError("max_molecules must be positive or None")
    if predicted_coordinates.ndim != 4 or predicted_coordinates.shape[-1] != 3:
        raise ValueError("predicted_coordinates must have shape (B, S, A, 3)")

    batch_size, sample_count, atom_count = predicted_coordinates.shape[:3]
    if sample_count < 1:
        raise ValueError("predicted_coordinates must contain at least one sample")
    device = predicted_coordinates.device
    target_coordinates = target.coordinates.to(device=device, dtype=torch.float32)
    target_mask = target.coordinate_mask.to(device=device, dtype=torch.bool)
    molecule_id = target.molecule_id.to(device=device, dtype=torch.long)
    molecule_equivalence = target.molecule_equivalence_index.to(device=device, dtype=torch.long)
    molecule_atom_index = target.molecule_atom_index.to(device=device, dtype=torch.long)
    representative_mask = target.representative_atom_mask.to(device=device, dtype=torch.bool)
    pool_mask = target.permutation_pool_mask.to(device=device, dtype=torch.bool)
    molecule_type = target.atom_molecule_type.to(device=device, dtype=torch.long)
    symmetry_atom_index = target.symmetry_atom_index.to(device=device, dtype=torch.long)
    symmetry_permutation = target.symmetry_permutation.to(device=device, dtype=torch.long)
    atom_index = input.atom_index.to(device=device, dtype=torch.long)
    atom_to_token_value = getattr(input, "atom_to_token", None)
    if not isinstance(atom_to_token_value, Tensor):
        raise ValueError("prediction-side matching requires input.atom_to_token")
    atom_to_token = atom_to_token_value.to(device=device, dtype=torch.long)

    _validate_shapes(
        predicted_coordinates[:, 0],
        target_coordinates,
        target_mask,
        molecule_id,
        molecule_equivalence,
        molecule_atom_index,
        representative_mask,
        pool_mask,
        molecule_type,
        symmetry_atom_index,
        symmetry_permutation,
        atom_index,
    )
    if atom_index.shape != (batch_size, atom_count):
        raise ValueError("input.atom_index must have shape (B, A) matching predicted coordinates")
    if atom_to_token.shape != (batch_size, atom_count):
        raise ValueError("input.atom_to_token must have shape (B, A) matching predicted coordinates")
    _validate_full_atom_axis(atom_index, target_coordinates.shape[1])
    token_count = _validate_token_axis(atom_to_token, representative_mask, atom_index)

    atom_permutation = torch.empty(
        (batch_size, sample_count, atom_count),
        dtype=torch.long,
        device=device,
    )
    token_permutation = torch.empty(
        (batch_size, sample_count, token_count),
        dtype=torch.long,
        device=device,
    )
    valid_atom = torch.ones(atom_count, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        current_atom_index = atom_index[batch_index]
        for sample_index in range(sample_count):
            predicted = predicted_coordinates[batch_index, sample_index]
            matched_atom_index = _match_molecule_indices(
                predicted,
                current_atom_index,
                valid_atom,
                current_atom_index,
                target_coordinates[batch_index],
                target_mask[batch_index],
                molecule_id[batch_index],
                molecule_equivalence[batch_index],
                molecule_atom_index[batch_index],
                representative_mask[batch_index],
                pool_mask[batch_index],
                molecule_type[batch_index],
                max_molecules=max_molecules,
            )
            current_coordinates = target_coordinates[batch_index].index_select(0, matched_atom_index)
            current_mask = target_mask[batch_index].index_select(0, matched_atom_index)
            aligned_prediction = _align_predicted(
                predicted,
                current_coordinates,
                current_mask,
                minimum_atoms=3,
            )
            if aligned_prediction is not None:
                _, _, matched_atom_index = _match_local_symmetry_with_indices(
                    aligned_prediction,
                    current_coordinates,
                    current_mask,
                    symmetry_atom_index[batch_index],
                    symmetry_permutation[batch_index],
                    matched_atom_index,
                )
            current_permutation = _invert_full_atom_match(matched_atom_index, current_atom_index)
            atom_permutation[batch_index, sample_index] = current_permutation
            token_permutation[batch_index, sample_index] = _atom_to_token_permutation(
                current_permutation,
                atom_to_token[batch_index],
                representative_mask[batch_index].index_select(0, current_atom_index),
                token_count,
            )

    gather_index = atom_permutation.unsqueeze(-1).expand(batch_size, sample_count, atom_count, 3)
    coordinates = predicted_coordinates.gather(2, gather_index)
    return ProtenixPredictionPermutation(
        coordinates=coordinates,
        atom_permutation=atom_permutation,
        token_permutation=token_permutation,
    )


def _single_prediction(predicted_coordinates: Tensor) -> Tensor:
    if predicted_coordinates.ndim == 3 and predicted_coordinates.shape[-1] == 3:
        return predicted_coordinates
    if (
        predicted_coordinates.ndim == 4
        and predicted_coordinates.shape[1] == 1
        and predicted_coordinates.shape[-1] == 3
    ):
        return predicted_coordinates[:, 0]
    raise ValueError("predicted_coordinates must have shape (B, A, 3) or (B, 1, A, 3)")


def _validate_shapes(
    predicted: Tensor,
    coordinates: Tensor,
    coordinate_mask: Tensor,
    molecule_id: Tensor,
    molecule_equivalence: Tensor,
    molecule_atom_index: Tensor,
    representative_mask: Tensor,
    pool_mask: Tensor,
    molecule_type: Tensor,
    symmetry_atom_index: Tensor,
    symmetry_permutation: Tensor,
    atom_index: Tensor,
) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("target.coordinates must have shape (B, T, 3)")
    full_shape = coordinates.shape[:2]
    if predicted.shape[0] != full_shape[0]:
        raise ValueError("predicted coordinates and target must have the same batch size")
    if full_shape[1] < 1:
        raise ValueError("target full atom axis must not be empty")
    full_fields = {
        "coordinate_mask": coordinate_mask,
        "molecule_id": molecule_id,
        "molecule_equivalence_index": molecule_equivalence,
        "molecule_atom_index": molecule_atom_index,
        "representative_atom_mask": representative_mask,
        "permutation_pool_mask": pool_mask,
        "atom_molecule_type": molecule_type,
    }
    for name, value in full_fields.items():
        if value.shape != full_shape:
            raise ValueError(f"target.{name} must have shape (B, T)")
    if atom_index.ndim != 2 or atom_index.shape[0] != full_shape[0]:
        raise ValueError("input.atom_index must have shape (B, A)")
    if symmetry_atom_index.ndim != 3 or symmetry_atom_index.shape[0] != full_shape[0]:
        raise ValueError("target.symmetry_atom_index must have shape (B, G, W)")
    if symmetry_permutation.ndim != 4 or symmetry_permutation.shape[0] != full_shape[0]:
        raise ValueError("target.symmetry_permutation must have shape (B, G, P, W)")
    if symmetry_atom_index.shape[1] != symmetry_permutation.shape[1]:
        raise ValueError("local symmetry group counts must agree")
    if symmetry_atom_index.shape[2] != symmetry_permutation.shape[3]:
        raise ValueError("local symmetry widths must agree")


def _validate_full_atom_axis(atom_index: Tensor, full_atom_count: int) -> None:
    r"""Require every current row to be a bijection of the full atom axis."""

    if atom_index.shape[1] != full_atom_count:
        raise ValueError(
            "prediction-side permutation requires the full/no-crop canonical atom set; "
            f"received {atom_index.shape[1]} of {full_atom_count} atoms"
        )
    expected = torch.arange(full_atom_count, dtype=torch.long, device=atom_index.device)
    for batch_index, current in enumerate(atom_index):
        if not torch.equal(torch.sort(current).values, expected):
            raise ValueError(
                "prediction-side permutation requires a one-to-one full/no-crop input.atom_index row; "
                f"batch row {batch_index} is incomplete, duplicated, or out of range"
            )


def _validate_token_axis(
    atom_to_token: Tensor,
    representative_mask: Tensor,
    atom_index: Tensor,
) -> int:
    r"""Validate contiguous tokens with exactly one PAE representative each."""

    if bool((atom_to_token < 0).any()):
        raise ValueError("prediction-side permutation requires nonnegative input.atom_to_token values")
    token_count = int(atom_to_token.max()) + 1
    expected = torch.arange(token_count, dtype=torch.long, device=atom_to_token.device)
    for batch_index in range(atom_to_token.shape[0]):
        current_tokens = atom_to_token[batch_index]
        if not torch.equal(torch.unique(current_tokens, sorted=True), expected):
            raise ValueError("each input.atom_to_token row must contain the same contiguous token set")
        current_representative = representative_mask[batch_index].index_select(0, atom_index[batch_index])
        representative_tokens = current_tokens[current_representative]
        if not torch.equal(torch.sort(representative_tokens).values, expected):
            raise ValueError("each current token must have exactly one representative atom")
    return token_count


def _invert_full_atom_match(matched_atom_index: Tensor, current_atom_index: Tensor) -> Tensor:
    r"""Invert prediction-to-truth full indices into output-to-input gathers."""

    if not torch.equal(torch.sort(matched_atom_index).values, torch.sort(current_atom_index).values):
        raise ValueError("matched full atom indices are not a one-to-one canonical atom permutation")
    atom_count = current_atom_index.numel()
    matched_position_by_full = torch.empty(atom_count, dtype=torch.long, device=current_atom_index.device)
    matched_position_by_full[matched_atom_index] = torch.arange(
        atom_count,
        dtype=torch.long,
        device=current_atom_index.device,
    )
    return matched_position_by_full.index_select(0, current_atom_index)


def _atom_to_token_permutation(
    atom_permutation: Tensor,
    atom_to_token: Tensor,
    representative_mask: Tensor,
    token_count: int,
) -> Tensor:
    r"""Lift an atom gather to the corresponding output-to-input token gather."""

    permuted_atom_to_token = atom_to_token[atom_permutation]
    permuted_representative_mask = representative_mask[atom_permutation]
    token_permutation = permuted_atom_to_token[permuted_representative_mask]
    expected = torch.arange(token_count, dtype=torch.long, device=atom_permutation.device)
    if not torch.equal(torch.sort(token_permutation).values, expected):
        raise ValueError("atom permutation does not induce a one-to-one token permutation")
    return token_permutation


def _match_molecule_indices(
    predicted: Tensor,
    atom_index: Tensor,
    valid_atom: Tensor,
    identity_index: Tensor,
    target_coordinates: Tensor,
    target_mask: Tensor,
    molecule_id: Tensor,
    molecule_equivalence: Tensor,
    molecule_atom_index: Tensor,
    representative_mask: Tensor,
    pool_mask: Tensor,
    molecule_type: Tensor,
    *,
    max_molecules: int | None,
) -> Tensor:
    r"""Run the released random-anchor/greedy molecule assignment heuristic."""

    model_positions = torch.nonzero(valid_atom, as_tuple=False).flatten().tolist()
    identity_match: dict[int, int] = {}
    pred_rep_positions: dict[int, list[int]] = {}
    pred_rep_local_atoms: dict[int, list[int]] = {}
    source_equivalence: dict[int, int] = {}
    for model_position in model_positions:
        full_index = int(atom_index[model_position])
        source = int(molecule_id[full_index])
        identity_match[source] = source
        source_equivalence.setdefault(source, int(molecule_equivalence[full_index]))
        if bool(representative_mask[full_index]):
            pred_rep_positions.setdefault(source, []).append(model_position)
            pred_rep_local_atoms.setdefault(source, []).append(int(molecule_atom_index[full_index]))

    pool_indices = torch.nonzero(pool_mask, as_tuple=False).flatten().tolist()
    pool_lookup: dict[tuple[int, int], int] = {}
    target_rep_indices: dict[int, list[int]] = {}
    target_equivalence: dict[int, int] = {}
    for full_index in pool_indices:
        molecule = int(molecule_id[full_index])
        local_atom = int(molecule_atom_index[full_index])
        pool_lookup.setdefault((molecule, local_atom), full_index)
        target_equivalence.setdefault(molecule, int(molecule_equivalence[full_index]))
        if bool(representative_mask[full_index]):
            target_rep_indices.setdefault(molecule, []).append(full_index)

    target_molecules = sorted(target_equivalence)
    if (
        not pred_rep_positions
        or (max_molecules is not None and len(target_molecules) > max_molecules)
        or len(set(target_equivalence.values())) == len(target_molecules)
        or not set(identity_match).issubset(target_equivalence)
    ):
        return identity_index

    valid_equivalence = {
        equivalence
        for equivalence in set(target_equivalence.values())
        if any(
            int(target_mask[target_rep_indices.get(molecule, [])].sum()) >= 4
            for molecule in target_molecules
            if target_equivalence[molecule] == equivalence
        )
    }
    valid_entity_molecule = [
        (source_equivalence[molecule], molecule)
        for molecule in sorted(pred_rep_positions)
        if len(pred_rep_positions[molecule]) >= 4 and source_equivalence[molecule] in valid_equivalence
    ]
    candidate_equivalence = {equivalence for equivalence, _ in valid_entity_molecule}
    if not candidate_equivalence:
        return identity_index

    polymer_equivalence = []
    for equivalence in candidate_equivalence:
        representative_full_index = [
            int(atom_index[position])
            for molecule, positions in pred_rep_positions.items()
            if source_equivalence[molecule] == equivalence
            for position in positions
        ]
        is_ligand = molecule_type[representative_full_index] == ENTITY_MOLECULE
        if int(is_ligand.sum()) <= len(representative_full_index) / 2 and len(representative_full_index) >= 12:
            polymer_equivalence.append(equivalence)
    if polymer_equivalence:
        candidate_equivalence = set(polymer_equivalence)

    target_count = {
        equivalence: sum(value == equivalence for value in target_equivalence.values())
        for equivalence in candidate_equivalence
    }
    minimum_count = min(target_count.values())
    candidate_equivalence = {
        equivalence for equivalence, count in target_count.items() if count == minimum_count
    }
    candidate_source = [
        molecule
        for equivalence, molecule in valid_entity_molecule
        if equivalence in candidate_equivalence
    ]
    maximum_length = max(len(pred_rep_positions[molecule]) for molecule in candidate_source)
    candidate_source = [
        molecule for molecule in candidate_source if len(pred_rep_positions[molecule]) == maximum_length
    ]
    anchor_source = random.choice(candidate_source)
    anchor_equivalence = source_equivalence[anchor_source]

    best_rmsd = float("inf")
    best_match: dict[int, int] | None = None
    anchor_candidates = [
        molecule for molecule in target_molecules if target_equivalence[molecule] == anchor_equivalence
    ]
    for anchor_target in anchor_candidates:
        anchor_truth_index = _truth_indices_for_local_atoms(
            anchor_target,
            pred_rep_local_atoms[anchor_source],
            pool_lookup,
            device=atom_index.device,
        )
        if anchor_truth_index is None:
            continue
        anchor_mask = target_mask.index_select(0, anchor_truth_index)
        transform = _alignment_transform(
            target_coordinates.index_select(0, anchor_truth_index),
            predicted.index_select(
                0,
                torch.tensor(pred_rep_positions[anchor_source], device=predicted.device),
            ),
            anchor_mask,
            minimum_atoms=1,
        )
        if transform is None:
            continue
        aligned_truth = _apply_alignment(target_coordinates, transform)

        current_match = {anchor_source: anchor_target}
        remaining_source = sorted(
            (molecule for molecule in pred_rep_positions if molecule != anchor_source),
            key=lambda molecule: (-len(pred_rep_positions[molecule]), molecule),
        )
        available_target = [molecule for molecule in target_molecules if molecule != anchor_target]
        assignment_failed = False
        for source in remaining_source:
            candidates = [
                molecule
                for molecule in available_target
                if target_equivalence[molecule] == source_equivalence[source]
            ]
            selected = _nearest_target_molecule(
                source,
                candidates,
                predicted,
                aligned_truth,
                target_mask,
                pred_rep_positions,
                pred_rep_local_atoms,
                pool_lookup,
            )
            if selected is None:
                assignment_failed = True
                break
            current_match[source] = selected
            available_target.remove(selected)
        if assignment_failed:
            continue

        total_rmsd = _heuristic_match_rmsd(
            current_match,
            predicted,
            aligned_truth,
            target_mask,
            pred_rep_positions,
            pred_rep_local_atoms,
            pool_lookup,
        )
        if total_rmsd < best_rmsd:
            best_rmsd = total_rmsd
            best_match = current_match

    if best_match is None:
        return identity_index
    complete_match = {**identity_match, **best_match}
    if complete_match == identity_match:
        return identity_index
    matched_index = _assignment_indices(
        atom_index,
        valid_atom,
        molecule_id,
        molecule_atom_index,
        complete_match,
        pool_lookup,
    )
    if matched_index is None:
        return identity_index

    # Upstream accepts a nonidentity heuristic assignment only when the global
    # aligned-RMSD improvement reaches 1e-12; equal and worse assignments revert.
    permuted_rmsd = _global_aligned_rmsd(
        predicted,
        target_coordinates.index_select(0, matched_index),
        target_mask.index_select(0, matched_index) & valid_atom,
    )
    identity_rmsd = _global_aligned_rmsd(
        predicted,
        target_coordinates.index_select(0, identity_index),
        target_mask.index_select(0, identity_index) & valid_atom,
    )
    if permuted_rmsd is None or identity_rmsd is None or identity_rmsd - permuted_rmsd < 1e-12:
        return identity_index
    return matched_index


def _truth_indices_for_local_atoms(
    molecule: int,
    local_atoms: list[int],
    pool_lookup: dict[tuple[int, int], int],
    *,
    device: torch.device,
) -> Tensor | None:
    indices = [pool_lookup.get((molecule, local_atom)) for local_atom in local_atoms]
    if any(index is None for index in indices):
        return None
    return torch.tensor(indices, dtype=torch.long, device=device)


def _nearest_target_molecule(
    source: int,
    candidates: list[int],
    predicted: Tensor,
    aligned_truth: Tensor,
    target_mask: Tensor,
    pred_rep_positions: dict[int, list[int]],
    pred_rep_local_atoms: dict[int, list[int]],
    pool_lookup: dict[tuple[int, int], int],
) -> int | None:
    best_error = float("inf")
    best_target: int | None = None
    unresolved: list[int] = []
    pred_position = torch.tensor(pred_rep_positions[source], dtype=torch.long, device=predicted.device)
    for target in candidates:
        truth_index = _truth_indices_for_local_atoms(
            target,
            pred_rep_local_atoms[source],
            pool_lookup,
            device=predicted.device,
        )
        if truth_index is None:
            continue
        mask = target_mask.index_select(0, truth_index)
        if not bool(mask.any()):
            unresolved.append(target)
            continue
        truth_center = aligned_truth.index_select(0, truth_index)[mask].mean(dim=0)
        pred_center = predicted.index_select(0, pred_position).to(dtype=torch.float32)[mask].mean(dim=0)
        error = float(torch.linalg.vector_norm(truth_center - pred_center))
        if error < best_error:
            best_error = error
            best_target = target
    if best_target is not None:
        return best_target
    return unresolved[0] if unresolved else None


def _heuristic_match_rmsd(
    match: dict[int, int],
    predicted: Tensor,
    aligned_truth: Tensor,
    target_mask: Tensor,
    pred_rep_positions: dict[int, list[int]],
    pred_rep_local_atoms: dict[int, list[int]],
    pool_lookup: dict[tuple[int, int], int],
) -> float:
    scores = []
    for source, target in match.items():
        truth_index = _truth_indices_for_local_atoms(
            target,
            pred_rep_local_atoms[source],
            pool_lookup,
            device=predicted.device,
        )
        if truth_index is None:
            return float("inf")
        mask = target_mask.index_select(0, truth_index)
        if not bool(mask.any()):
            scores.append(0.0)
            continue
        pred_position = torch.tensor(pred_rep_positions[source], dtype=torch.long, device=predicted.device)
        score = _fixed_frame_rmsd(
            predicted.index_select(0, pred_position),
            aligned_truth.index_select(0, truth_index),
            mask,
            eps=0.0,
        )
        scores.append(float(score) if score is not None else 0.0)
    return sum(scores) / len(scores)


def _assignment_indices(
    atom_index: Tensor,
    valid_atom: Tensor,
    molecule_id: Tensor,
    molecule_atom_index: Tensor,
    assignment: dict[int, int],
    pool_lookup: dict[tuple[int, int], int],
) -> Tensor | None:
    assigned = torch.zeros_like(atom_index)
    for model_position in torch.nonzero(valid_atom, as_tuple=False).flatten().tolist():
        full_index = int(atom_index[model_position])
        source = int(molecule_id[full_index])
        destination = assignment.get(source, source)
        local_atom = int(molecule_atom_index[full_index])
        destination_index = pool_lookup.get((destination, local_atom))
        if destination_index is None:
            return None
        assigned[model_position] = destination_index
    return assigned


def _global_aligned_rmsd(predicted: Tensor, truth: Tensor, mask: Tensor) -> float | None:
    aligned = _align_predicted(predicted, truth, mask, minimum_atoms=1)
    if aligned is None:
        return None
    score = _fixed_frame_rmsd(aligned, truth, mask, eps=1e-8)
    return None if score is None else float(score)


def _match_local_symmetry(
    aligned_prediction: Tensor,
    coordinates: Tensor,
    coordinate_mask: Tensor,
    symmetry_atom_index: Tensor,
    symmetry_permutation: Tensor,
) -> tuple[Tensor, Tensor]:
    matched_coordinates, matched_mask, _ = _match_local_symmetry_with_indices(
        aligned_prediction,
        coordinates,
        coordinate_mask,
        symmetry_atom_index,
        symmetry_permutation,
        torch.arange(coordinates.shape[0], dtype=torch.long, device=coordinates.device),
    )
    return matched_coordinates, matched_mask


def _match_local_symmetry_with_indices(
    aligned_prediction: Tensor,
    coordinates: Tensor,
    coordinate_mask: Tensor,
    symmetry_atom_index: Tensor,
    symmetry_permutation: Tensor,
    matched_atom_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Match local atoms while carrying the selected full atom indices."""

    current_coordinates = coordinates
    current_mask = coordinate_mask
    current_atom_index = matched_atom_index
    atom_count = coordinates.shape[0]
    for group_index in range(symmetry_atom_index.shape[0]):
        group = symmetry_atom_index[group_index]
        group = group[(group >= 0) & (group < atom_count)]
        if group.numel() == 0:
            continue
        # Baseline atom matching operates per residue and drops residues with
        # fewer than three resolved atoms even when the global frame is valid.
        if int(current_mask.index_select(0, group).sum()) < 3:
            continue
        best_coordinates = current_coordinates
        best_mask = current_mask
        best_atom_index = current_atom_index
        best_score = _fixed_frame_rmsd(
            aligned_prediction.index_select(0, group),
            current_coordinates.index_select(0, group),
            current_mask.index_select(0, group),
            eps=1e-8,
        )
        for permutation in symmetry_permutation[group_index]:
            permutation = permutation[(permutation >= 0) & (permutation < atom_count)]
            if permutation.numel() != group.numel():
                continue
            candidate_coordinates = current_coordinates.clone()
            candidate_mask = current_mask.clone()
            candidate_atom_index = current_atom_index.clone()
            candidate_coordinates[group] = current_coordinates.index_select(0, permutation)
            candidate_mask[group] = current_mask.index_select(0, permutation)
            candidate_atom_index[group] = current_atom_index.index_select(0, permutation)
            score = _fixed_frame_rmsd(
                aligned_prediction.index_select(0, group),
                candidate_coordinates.index_select(0, group),
                candidate_mask.index_select(0, group),
                eps=1e-8,
            )
            if score is not None and (best_score is None or score < best_score):
                best_coordinates = candidate_coordinates
                best_mask = candidate_mask
                best_atom_index = candidate_atom_index
                best_score = score
        current_coordinates = best_coordinates
        current_mask = best_mask
        current_atom_index = best_atom_index
    return current_coordinates, current_mask, current_atom_index


def _alignment_transform(
    source: Tensor,
    destination: Tensor,
    mask: Tensor,
    *,
    minimum_atoms: int,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if int(mask.sum()) < minimum_atoms:
        return None
    # Explicit casts alone do not form an FP32 boundary: an enclosing CPU/CUDA
    # autocast region can still lower the matrix multiplies below.  Protenix
    # performs all alignment math outside autocast.
    with torch.autocast(device_type=source.device.type, enabled=False):
        source = source[mask].to(dtype=torch.float32)
        destination = destination[mask].to(dtype=torch.float32)
        if not bool(torch.isfinite(source).all() & torch.isfinite(destination).all()):
            return None
        source_center = source.mean(dim=0)
        destination_center = destination.mean(dim=0)
        covariance = (source - source_center).transpose(0, 1) @ (destination - destination_center)
        left, _, right_transpose = torch.linalg.svd(covariance)
        correction = torch.ones(3, dtype=torch.float32, device=source.device)
        correction[-1] = torch.where(torch.det(left @ right_transpose) < 0, -1.0, 1.0)
        rotation = (left * correction.unsqueeze(0)) @ right_transpose
        return rotation, source_center, destination_center


def _apply_alignment(coordinates: Tensor, transform: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    rotation, source_center, destination_center = transform
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        return (coordinates.to(dtype=torch.float32) - source_center) @ rotation + destination_center


def _align_predicted(
    predicted: Tensor,
    truth: Tensor,
    mask: Tensor,
    *,
    minimum_atoms: int,
) -> Tensor | None:
    transform = _alignment_transform(
        predicted,
        truth,
        mask,
        minimum_atoms=minimum_atoms,
    )
    return None if transform is None else _apply_alignment(predicted, transform)


def _fixed_frame_rmsd(
    predicted: Tensor,
    truth: Tensor,
    mask: Tensor,
    *,
    eps: float,
) -> Tensor | None:
    if not bool(mask.any()):
        return None
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        difference = predicted[mask].to(dtype=torch.float32) - truth[mask].to(dtype=torch.float32)
        if not bool(torch.isfinite(difference).all()):
            return None
        return torch.sqrt(torch.mean(torch.sum(difference.square(), dim=-1)) + eps)
