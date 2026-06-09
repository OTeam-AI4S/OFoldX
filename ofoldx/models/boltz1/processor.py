r"""
Boltz-1 structure processing.

Turns a parsed structure into the Boltz-1 input contract: residue-token features from the shared
residue-token processor, a default all-``UNSPECIFIED`` pocket one-hot, and the Boltz-1x physical-potential
constraint feature surface (bond/angle/chirality/contact index tensors).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F

from ...processors.registry import PROCESSORS
from ...processors.residue_token import ResidueTokenStructureProcessor
from ...processors.utils import StructuralFields, StructureInput
from .model import Boltz1Input

_BOLTZ1_FIELDS = frozenset(Boltz1Input.__dataclass_fields__)


@PROCESSORS.register("boltz1")
class Boltz1Processor(ResidueTokenStructureProcessor):
    r"""
    Turn a parsed structure into a batched [`Boltz1Input`][ofoldx.models.boltz1.model.Boltz1Input].

    Builds on the shared residue-token structure processor, then adds the Boltz-1-specific token pocket
    one-hot and the Boltz-1x constraint feature surface, and finally keeps only the fields declared by the
    Boltz-1 input contract.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("boltz-community/boltz-1")  # doctest: +SKIP
        >>> batch = processor("complex.cif")  # doctest: +SKIP
    """

    input_class = Boltz1Input

    def __call__(self, structure: StructureInput) -> Boltz1Input:  # type: ignore[override]
        r"""
        Process one structure and return its batched Boltz-1 input.

        Args:
            structure: One parsed structure input accepted by the shared residue-token processor.

        Returns:
            A batched [`Boltz1Input`][ofoldx.models.boltz1.model.Boltz1Input] carrying the Boltz-1
            token/atom features, pocket one-hot, and constraint feature surface.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("boltz-community/boltz-1")  # doctest: +SKIP
            >>> batch = processor("complex.cif")  # doctest: +SKIP
        """

        return cast(Boltz1Input, super().__call__(structure))

    def _process_structure(self, structure: Any) -> tuple[dict[str, Any], StructuralFields]:
        values, fields = super()._process_structure(structure)
        num_tokens = values["token_mask"].shape[-1]
        values["pocket_feature"] = (
            F.one_hot(torch.zeros(num_tokens, dtype=torch.long), num_classes=4).float().unsqueeze(0)
        )
        values.update(_boltz1_constraint_features(fields))
        return {key: value for key, value in values.items() if key in _BOLTZ1_FIELDS}, fields


def _boltz1_constraint_features(fields: StructuralFields) -> dict[str, torch.Tensor]:
    r"""
    Build the Boltz-1x physical-potential constraint feature surface for one processed structure.

    Most constraints (RDKit bounds, chirality, stereo, planarity, contacts) are emitted empty; only the
    cross-chain connected-atom bonds and same-entity symmetric-chain pairs are populated from the parsed
    structure so that the potential steering path has valid index tensors to consume.

    Args:
        fields: The intermediate structural fields produced by the residue-token processor, exposing the
            parsed atoms/bonds, residues, and the chain/entity index maps.

    Returns:
        Mapping from constraint feature name to its (batched, leading dim 1) index/mask tensor.
    """

    connected_chain_index: list[tuple[int, int]] = []
    connected_atom_index: list[tuple[int, int]] = []
    atoms = fields.structure.atoms
    for first, second in fields.structure.bonds:
        first_chain = fields.chain_to_index[atoms[first].residue.chain_id]
        second_chain = fields.chain_to_index[atoms[second].residue.chain_id]
        if first_chain == second_chain:
            continue
        connected_chain_index.append((first_chain, second_chain))
        connected_atom_index.append((first, second))

    chain_to_entity: dict[int, int] = {}
    for residue, entity in zip(fields.residues, fields.entity_values):
        chain_index = fields.chain_to_index[residue.chain_id]
        chain_to_entity.setdefault(chain_index, entity)
    symmetric_chain_index = [
        (left, right)
        for offset, (left, left_entity) in enumerate(chain_to_entity.items())
        for right, right_entity in tuple(chain_to_entity.items())[offset + 1 :]
        if left_entity == right_entity
    ]

    return {
        "rdkit_bounds_index": torch.empty(1, 2, 0, dtype=torch.long),
        "rdkit_bounds_bond_mask": torch.empty(1, 0, dtype=torch.bool),
        "rdkit_bounds_angle_mask": torch.empty(1, 0, dtype=torch.bool),
        "rdkit_upper_bounds": torch.empty(1, 0, dtype=torch.float32),
        "rdkit_lower_bounds": torch.empty(1, 0, dtype=torch.float32),
        "chiral_atom_index": torch.empty(1, 4, 0, dtype=torch.long),
        "chiral_reference_mask": torch.empty(1, 0, dtype=torch.bool),
        "chiral_atom_orientations": torch.empty(1, 0, dtype=torch.bool),
        "stereo_bond_index": torch.empty(1, 4, 0, dtype=torch.long),
        "stereo_reference_mask": torch.empty(1, 0, dtype=torch.bool),
        "stereo_bond_orientations": torch.empty(1, 0, dtype=torch.bool),
        "planar_bond_index": torch.empty(1, 6, 0, dtype=torch.long),
        "connected_chain_index": _index_tensor(connected_chain_index),
        "connected_atom_index": _index_tensor(connected_atom_index),
        "symmetric_chain_index": _index_tensor(symmetric_chain_index),
        "contact_pair_index": torch.empty(1, 2, 0, dtype=torch.long),
        "contact_union_index": torch.empty(1, 0, dtype=torch.long),
        "contact_negation_mask": torch.empty(1, 0, dtype=torch.bool),
        "contact_thresholds": torch.empty(1, 0, dtype=torch.float32),
    }


def _index_tensor(pairs: list[tuple[int, int]]) -> torch.Tensor:
    r"""
    Stack a list of index pairs into a batched ``[1, 2, num_pairs]`` long tensor.

    Args:
        pairs: List of ``(first, second)`` index pairs; an empty list yields the empty ``[1, 2, 0]`` tensor.

    Returns:
        A ``[1, 2, num_pairs]`` long tensor with the two members of each pair along dim 1.
    """

    if not pairs:
        return torch.empty(1, 2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).T.unsqueeze(0)
