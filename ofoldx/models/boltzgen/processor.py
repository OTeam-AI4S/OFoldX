r"""
BoltzGen structure processing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ...processors.registry import PROCESSORS
from ...processors.residue_token import ResidueTokenStructureProcessor
from ...processors.utils import StructuralFields, StructureInput, structure_from_input
from .model import BoltzGenInput

_PROTEIN_BACKBONE4_ATOMS = ("N", "CA", "C", "O")

# The 20 canonical amino acids (three-letter CCD codes) in the model's residue-token order; column ``i`` of the
# per-position constraint mask corresponds to ``_CANONICAL_TOKENS[i]``. Mirrors the model's ``CANONICAL_TOKENS``.
_CANONICAL_TOKENS: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)  # fmt: skip
_CANONICAL_INDEX = {token: index for index, token in enumerate(_CANONICAL_TOKENS)}
_NUM_CANONICAL = len(_CANONICAL_TOKENS)


@PROCESSORS.register("boltzgen")
@PROCESSORS.register("boltzgen_diverse")
@PROCESSORS.register("boltzgen_adherence")
@PROCESSORS.register("boltzgen_structuretrained_small")
@PROCESSORS.register("boltzgen_ifold")
class BoltzGenProcessor(ResidueTokenStructureProcessor):
    r"""
    Process structure inputs into the BoltzGen design-generator input schema.

    Extends [`ResidueTokenStructureProcessor`][ofoldx.processors.residue_token.ResidueTokenStructureProcessor]
    with the structure-design conditioning defaults (binding, design-mask, secondary-structure) and the optional
    inverse-folding constraint tensors. Registered for every BoltzGen artifact key.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/boltzgen")  # doctest: +SKIP
        >>> batch = processor("complex.cif")  # doctest: +SKIP
    """

    input_class = BoltzGenInput

    def __call__(  # type: ignore[override]
        self,
        structure: StructureInput,
        *,
        residue_constraints: Mapping[int, Mapping[str, Sequence[str]]] | None = None,
        symmetric_group: Sequence[int] | None = None,
    ) -> BoltzGenInput:
        r"""
        Return a batched ``BoltzGenInput`` for one structure input.

        Args:
            structure: Parsed structure object or structure file path.
            residue_constraints: Optional per-token inverse-folding amino-acid constraints, keyed by token index.
                Each value is a mapping with an ``"allowed"`` and/or ``"disallowed"`` sequence of three-letter
                canonical amino-acid codes; ``allowed`` whitelists (all others blocked) and ``disallowed``
                blacklists. When given, builds the model's ``aa_constraint_mask`` (input-side, mirroring the
                upstream ``schema.parse_residue_constraints``).
            symmetric_group: Optional per-token symmetry-copy id (0 = no group) used to tie homomer sequences
                during inverse-folding decoding.

        Returns:
            Batched BoltzGen model input with structure-design conditioning defaults.

        Raises:
            ValueError: If target coords or per-token backbone atoms are unavailable, or a constraint blocks every
                amino acid at a position.
        """

        values, fields = self._process_structure(structure_from_input(structure))
        target_positions = values["target_positions"]
        if target_positions is None:
            raise ValueError("BoltzGen processing requires target_positions")

        token_to_rep_atom = values["token_to_rep_atom"]
        token_mask = values["token_mask"]
        rep_index = token_to_rep_atom.to(dtype=torch.long)
        center_coords = target_positions.gather(1, rep_index.unsqueeze(-1).expand(-1, -1, 3))
        token_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        token_to_bb4_atoms = _token_to_backbone4_atoms(fields)
        batch, num_tokens = token_mask.shape

        aa_constraint_mask = (
            build_aa_constraint_mask(residue_constraints, num_tokens).unsqueeze(0)
            if residue_constraints is not None
            else None
        )
        symmetric_group_tensor = (
            _symmetric_group_tensor(symmetric_group, num_tokens) if symmetric_group is not None else None
        )

        return BoltzGenInput(
            **values,
            center_coords=center_coords,
            token_to_bb4_atoms=token_to_bb4_atoms,
            coords=target_positions,
            token_distance_mask=token_distance_mask.to(dtype=target_positions.dtype),
            binding_type=torch.zeros(batch, num_tokens, dtype=torch.long),
            design_mask=torch.ones(batch, num_tokens, dtype=torch.long),
            ss_type=torch.zeros(batch, num_tokens, dtype=torch.long),
            aa_constraint_mask=aa_constraint_mask,
            symmetric_group=symmetric_group_tensor,
        )


def build_aa_constraint_mask(
    residue_constraints: Mapping[int, Mapping[str, Sequence[str]]], num_tokens: int
) -> torch.Tensor:
    r"""
    Build the per-token canonical-amino-acid block mask consumed by the inverse-folding decoder.

    Args:
        residue_constraints: Per-token constraints keyed by token index, each an ``{"allowed"?, "disallowed"?}``
            mapping of three-letter canonical amino-acid codes. ``allowed`` blocks every amino acid not listed;
            ``disallowed`` blocks the listed amino acids. Both may be combined (block = union).
        num_tokens: Number of tokens in the structure.

    Returns:
        Float mask ``(num_tokens, 20)`` where ``1`` marks a blocked amino acid and ``0`` an allowed one.

    Raises:
        ValueError: If a token index is out of range, an amino-acid code is not canonical, or a token ends up with
            every amino acid blocked.

    Examples:
        >>> mask = build_aa_constraint_mask({0: {"allowed": ["ALA"]}}, 2)
        >>> mask.shape
        torch.Size([2, 20])
        >>> mask[0].sum().item()  # only ALA allowed -> 19 blocked
        19.0
        >>> mask[1].sum().item()  # unconstrained token -> nothing blocked
        0.0
    """

    mask = torch.zeros(num_tokens, _NUM_CANONICAL, dtype=torch.float32)
    for token_index, spec in residue_constraints.items():
        if not 0 <= token_index < num_tokens:
            raise ValueError(f"residue_constraints token index {token_index} out of range [0, {num_tokens})")
        blocked = torch.zeros(_NUM_CANONICAL, dtype=torch.bool)
        allowed = spec.get("allowed")
        if allowed is not None:
            blocked |= ~_codes_to_mask(allowed)
        disallowed = spec.get("disallowed")
        if disallowed is not None:
            blocked |= _codes_to_mask(disallowed)
        if bool(blocked.all()):
            raise ValueError(f"residue_constraints block every amino acid at token {token_index}")
        mask[token_index] = blocked.float()
    return mask


def _codes_to_mask(codes: Sequence[str]) -> torch.Tensor:
    r"""
    Build a boolean mask over the 20 canonical amino acids, set at each listed three-letter code.

    Args:
        codes: Three-letter canonical amino-acid codes to mark.

    Returns:
        Boolean mask ``(20,)`` set at each listed amino acid.

    Raises:
        ValueError: If a code is not one of the 20 canonical amino acids.
    """

    mask = torch.zeros(_NUM_CANONICAL, dtype=torch.bool)
    for code in codes:
        if code not in _CANONICAL_INDEX:
            raise ValueError(f"unknown canonical amino acid '{code}'")
        mask[_CANONICAL_INDEX[code]] = True
    return mask


def _symmetric_group_tensor(symmetric_group: Sequence[int], num_tokens: int) -> torch.Tensor:
    r"""
    Build the batched per-token symmetry-copy id tensor from a per-token id sequence.

    Args:
        symmetric_group: Per-token symmetry-copy id (0 = no group), one entry per token.
        num_tokens: Number of tokens in the structure.

    Returns:
        Long tensor ``(1, num_tokens)`` of symmetry-copy ids.

    Raises:
        ValueError: If ``symmetric_group`` length does not match ``num_tokens``.
    """

    if len(symmetric_group) != num_tokens:
        raise ValueError(f"symmetric_group has length {len(symmetric_group)}, expected {num_tokens}")
    return torch.tensor(symmetric_group, dtype=torch.long).unsqueeze(0)


def _token_to_backbone4_atoms(fields: StructuralFields) -> torch.Tensor:
    r"""
    Return the per-token indices of the four backbone atoms ``(N, CA, C, O)``.

    Args:
        fields: Tokenized structural fields (atoms and per-token atom indices).

    Returns:
        Long tensor ``(1, num_tokens * 4)`` of atom indices, four consecutive per token in ``(N, CA, C, O)`` order.

    Raises:
        ValueError: If any token is missing one of the four backbone atoms.
    """

    atoms = fields.structure.atoms
    indices: list[int] = []
    for token_index, token in enumerate(fields.tokens):
        name_to_index = {atoms[atom_index].name: atom_index for atom_index in token.atom_indices}
        missing = [name for name in _PROTEIN_BACKBONE4_ATOMS if name not in name_to_index]
        if missing:
            residue = token.residue
            residue_name = atoms[token.atom_indices[0]].residue_name
            raise ValueError(
                "BoltzGen processing requires each token to contain backbone atoms N, CA, C, and O; "
                f"token {token_index} ({residue.chain_id}:{residue.residue_number}{residue.insertion_code} "
                f"{residue_name}) is missing {', '.join(missing)}"
            )
        indices.extend(name_to_index[name] for name in _PROTEIN_BACKBONE4_ATOMS)
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)
