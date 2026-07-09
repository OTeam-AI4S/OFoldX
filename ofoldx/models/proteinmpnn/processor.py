r"""
ProteinMPNN structure processing.

Turns a parsed [`Structure`][ofoldx.data.structure.Structure] (or a PDB/mmCIF path) into a batched
[`ProteinMPNNInput`][ofoldx.models.proteinmpnn.model.ProteinMPNNInput], matching the featurization semantics of the
upstream ``parse_PDB`` + ``tied_featurize`` pipeline: per-chain backbone atoms (N/CA/C/O), the ProteinMPNN
amino-acid alphabet, the 100-residue inter-chain gap in ``residue_index``, and the full conditioning surface
(fixed positions, tied positions, omitted amino acids, PSSM, and per-residue bias).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import torch

from ...data.structure import Structure
from ...io import read_structure
from ...processors.mixin import ProcessorMixin
from ...processors.registry import PROCESSORS
from ...processors.utils import StructureInput, structure_from_input
from ...results_registry import register_sequence_decoder
from .model import ProteinMPNNInput

# ProteinMPNN residue alphabet: index order used by the released checkpoints (``X`` is the unknown/gap token).
PROTEINMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_ALPHABET_INDEX = {letter: index for index, letter in enumerate(PROTEINMPNN_ALPHABET)}
_UNKNOWN_INDEX = _ALPHABET_INDEX["X"]

_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}
_BACKBONE_ATOMS = ("N", "CA", "C", "O")


@PROCESSORS.register("proteinmpnn")
class ProteinMPNNProcessor(ProcessorMixin):
    r"""
    Build a batched ProteinMPNN input from a structure.

    Args:
        designed_chains: Chain ids to redesign (upstream ``masked_chains``). ``None`` designs every parsed chain.
        fixed_positions: Per-chain 1-based residue positions to hold fixed (upstream ``fixed_positions_dict``).
        tied_positions: Groups of ``(chain_id, position)`` residues (1-based) whose identity is tied during
            ``tied_sample`` (upstream ``tied_positions_dict``). Optional per-group weights map to ``tied_weights``.
        tied_weights: Per-group ``tied_beta`` weights aligned one-to-one with ``tied_positions``; ``None`` weights
            every tied position equally.
        omit_amino_acids: Global amino acids forbidden at every designable position.
        ca_only: Accepted for API symmetry with the upstream Cα-only path; only the full backbone is supported.

    Raises:
        ValueError: If ``ca_only`` is set, or if ``tied_weights`` does not align with ``tied_positions``.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("proteinmpnn")  # doctest: +SKIP
        >>> batch = processor("complex.pdb")  # doctest: +SKIP
    """

    input_class = ProteinMPNNInput

    def __init__(
        self,
        *,
        designed_chains: Sequence[str] | None = None,
        fixed_positions: Mapping[str, Sequence[int]] | None = None,
        tied_positions: Sequence[Sequence[tuple[str, int]]] | None = None,
        tied_weights: Sequence[Sequence[float]] | None = None,
        omit_amino_acids: Sequence[str] = (),
        ca_only: bool = False,
    ) -> None:
        if ca_only:
            raise ValueError("ProteinMPNNProcessor supports only the full N/CA/C/O backbone")
        self.designed_chains = None if designed_chains is None else [str(chain) for chain in designed_chains]
        self.fixed_positions = {str(k): [int(p) for p in v] for k, v in (fixed_positions or {}).items()}
        normalized = [_normalize_tied_group(group) for group in (tied_positions or [])]
        self.tied_positions = [members for members, _ in normalized]
        embedded_weights = [weights for _, weights in normalized if weights is not None]
        if tied_weights is not None:
            self.tied_weights = [[float(w) for w in group] for group in tied_weights]
        elif embedded_weights and len(embedded_weights) == len(normalized):
            self.tied_weights = embedded_weights
        else:
            self.tied_weights = []
        self.omit_amino_acids = [aa for aa in omit_amino_acids if aa in _ALPHABET_INDEX]
        if self.tied_weights and len(self.tied_weights) != len(self.tied_positions):
            raise ValueError("tied_weights must align one-to-one with tied_positions groups")

    def __call__(self, structure: StructureInput) -> ProteinMPNNInput:
        r"""
        Featurize one structure into a single-structure batched ProteinMPNN input.

        Parses backbone atoms per chain, assembles coords/tokens/masks with the 100-residue inter-chain gap in
        ``residue_index``, and attaches the design/fixed masks plus any tied-position and omit-AA conditioning.

        Args:
            structure: A parsed structure or a PDB/mmCIF path.

        Returns:
            A batched [`ProteinMPNNInput`][ofoldx.models.proteinmpnn.model.ProteinMPNNInput] with batch axis
            size one.

        Raises:
            ValueError: If the structure has no protein residues, or a tied position names an unknown chain.

        Examples:
            >>> model_input = processor("complex.pdb")  # doctest: +SKIP
            >>> model_input.backbone_coords.shape  # doctest: +SKIP
            torch.Size([1, 217, 4, 3])
        """

        parsed = structure_from_input(structure)
        chains = parse_pdb_to_chains(parsed)
        if not chains:
            raise ValueError("ProteinMPNN processing requires at least one protein residue")
        designed = set(self.designed_chains) if self.designed_chains is not None else {c.chain_id for c in chains}

        coords: list[torch.Tensor] = []
        tokens: list[int] = []
        residue_index: list[int] = []
        chain_index: list[int] = []
        design_mask: list[float] = []
        fixed_mask: list[float] = []
        chain_offsets: dict[str, int] = {}
        offset = 0
        for chain_number, chain in enumerate(chains):
            chain_offsets[chain.chain_id] = offset
            is_designed = chain.chain_id in designed
            fixed = set(self.fixed_positions.get(chain.chain_id, ()))
            for position, (residue_coords, token) in enumerate(zip(chain.backbone_coords, chain.tokens), start=1):
                coords.append(residue_coords)
                tokens.append(token)
                residue_index.append(100 * chain_number + offset + position - 1)
                chain_index.append(chain_number)
                design_mask.append(1.0 if is_designed else 0.0)
                fixed_mask.append(0.0 if (is_designed and position in fixed) else 1.0)
            offset += len(chain.tokens)

        length = len(tokens)
        backbone = torch.stack(coords).unsqueeze(0)
        token_mask = torch.ones(1, length)
        design = torch.tensor(design_mask)[None, :]

        model_input = ProteinMPNNInput(
            backbone_coords=backbone,
            sequence_tokens=torch.tensor(tokens, dtype=torch.long)[None, :],
            token_mask=token_mask,
            residue_index=torch.tensor(residue_index, dtype=torch.long)[None, :],
            chain_index=torch.tensor(chain_index, dtype=torch.long)[None, :],
            design_mask=design,
            fixed_position_mask=torch.tensor(fixed_mask)[None, :],
        )

        tied_positions, tied_weights = self._resolve_tied(chain_offsets, length)
        if tied_positions:
            model_input.tied_positions = tied_positions
            model_input.tied_weights = tied_weights
        if self.omit_amino_acids:
            model_input.omit_amino_acid_mask = self._omit_mask(length, design)
        return model_input

    def _resolve_tied(
        self, chain_offsets: Mapping[str, int], length: int
    ) -> tuple[list[list[int]], torch.Tensor | None]:
        r"""
        Flatten ``(chain, 1-based position)`` tied groups into global 0-based indices and a ``tied_weights`` vector.
        """

        if not self.tied_positions:
            return [], None
        groups: list[list[int]] = []
        weights = torch.ones(length)
        for group_number, group in enumerate(self.tied_positions):
            indices: list[int] = []
            for member_number, (chain_id, position) in enumerate(group):
                if chain_id not in chain_offsets:
                    raise ValueError(f"tied position references unknown chain {chain_id!r}")
                index = chain_offsets[chain_id] + position - 1
                indices.append(index)
                if self.tied_weights:
                    weights[index] = self.tied_weights[group_number][member_number]
            groups.append(indices)
        return groups, weights

    def _omit_mask(self, length: int, design_mask: torch.Tensor) -> torch.Tensor:
        r"""
        Build the per-position omit-amino-acid mask restricted to designable positions.

        Args:
            length: Number of residues.
            design_mask: Design mask `[1, L]` gating which positions the omit restriction applies to.

        Returns:
            A `[1, L, len(alphabet)]` mask with ones at forbidden (amino-acid, designable-position) entries.
        """

        mask = torch.zeros(1, length, len(PROTEINMPNN_ALPHABET))
        for amino_acid in self.omit_amino_acids:
            mask[..., _ALPHABET_INDEX[amino_acid]] = 1.0
        return mask * design_mask[..., None]

    def _processor_config(self) -> dict[str, object]:
        r"""
        Serialize the processor's conditioning options into a config-round-trippable mapping.

        Returns:
            A mapping of the designed chains, fixed positions, tied groups (each as parallel chain/position/weight
            lists), and omitted amino acids.
        """

        # Tied groups are stored as mappings (chains/positions/weights kept as parallel lists): a JSON list of
        # ``[chain, position]`` pairs is ambiguous with a mapping under config round-trips, and a bare list of
        # weight lists collapses when it holds a single group, so everything for a group lives in one mapping.
        groups: list[dict[str, object]] = []
        for group_number, group in enumerate(self.tied_positions):
            entry: dict[str, object] = {
                "chains": [chain for chain, _ in group],
                "positions": [position for _, position in group],
            }
            if self.tied_weights:
                entry["weights"] = list(self.tied_weights[group_number])
            groups.append(entry)
        return {
            "designed_chains": self.designed_chains,
            "fixed_positions": {k: list(v) for k, v in self.fixed_positions.items()},
            "tied_positions": groups,
            "omit_amino_acids": list(self.omit_amino_acids),
        }


def _normalize_tied_group(
    group: Mapping[str, object] | Iterable[tuple[str, int]],
) -> tuple[list[tuple[str, int]], list[float] | None]:
    r"""
    Accept a tied group as ``[(chain, position), ...]`` or a ``{"chains": [...], "positions": [...], "weights"?}``
    mapping, returning its ``(chain, position)`` members plus any embedded per-member weights.

    The mapping form is what config round-trips produce; the pair form is the natural in-code call site.
    """

    if isinstance(group, Mapping):
        chains = cast("Iterable[object]", group["chains"])
        positions = cast("Iterable[object]", group["positions"])
        members = [(str(chain), int(str(position))) for chain, position in zip(chains, positions)]
        raw_weights = group.get("weights")
        weights = [float(str(weight)) for weight in cast("Iterable[object]", raw_weights)] if raw_weights else None
        return members, weights
    return [(str(chain), int(position)) for chain, position in group], None


@dataclass(frozen=True, slots=True)
class _ParsedResidue:
    r"""
    One parsed protein residue: its ProteinMPNN token id and the backbone atoms found for it.
    """

    token: int
    atoms: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def has_full_backbone(self) -> bool:
        r"""
        Whether all four N/CA/C/O backbone atoms were found for this residue.
        """

        return all(name in self.atoms for name in _BACKBONE_ATOMS)

    def resolved_token(self) -> int:
        r"""
        The residue token, downgraded to the unknown (``X``) token when the backbone is incomplete.

        Returns:
            The ProteinMPNN token id, or the unknown-token index for residues missing a backbone atom.
        """

        # Incomplete backbone: treat as unknown, matching the upstream NaN-then-mask path.
        return self.token if self.has_full_backbone else _UNKNOWN_INDEX

    def backbone_tensor(self) -> torch.Tensor:
        r"""
        Stack this residue's N/CA/C/O coordinates, zero-filling any missing atom.

        Returns:
            Backbone coordinates `[4, 3]` in N, CA, C, O order.
        """

        return torch.stack([self.atoms.get(name, torch.zeros(3)) for name in _BACKBONE_ATOMS])


@dataclass(frozen=True, slots=True)
class ParsedChain:
    r"""
    One parsed protein chain: its id, ProteinMPNN token ids, and per-residue N/CA/C/O coordinates.

    Attributes:
        chain_id: Source chain identifier from the structure.
        tokens: Per-residue ProteinMPNN token ids, in residue order (unknown/gap residues use the ``X`` token).
        backbone_coords: Per-residue backbone coordinates, one `[4, 3]` tensor per residue in N, CA, C, O order.
    """

    chain_id: str
    tokens: list[int]
    backbone_coords: list[torch.Tensor]


def parse_pdb_to_chains(structure: StructureInput | Structure) -> list[ParsedChain]:
    r"""
    Parse protein chains and their backbone atoms from a structure (faithful to upstream ``parse_PDB``).

    Args:
        structure: A parsed structure or a PDB/mmCIF path.

    Returns:
        Protein chains in first-seen order, each carrying ProteinMPNN token ids and per-residue N/CA/C/O
        coordinates. Residues missing a backbone atom get zero-filled coordinates and the unknown (``X``) token,
        mirroring the upstream NaN-fill-then-mask behavior.
    """

    parsed = structure if isinstance(structure, Structure) else read_structure(structure)
    residue_atoms: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}
    residue_order: list[tuple[str, int, str]] = []
    residue_names: dict[tuple[str, int, str], str] = {}
    for atom in parsed.atoms:
        residue = atom.residue
        key = (residue.chain_id, residue.residue_number, residue.insertion_code)
        if key not in residue_atoms:
            residue_atoms[key] = {}
            residue_order.append(key)
            residue_names[key] = atom.residue_name.upper()
        if atom.name in _BACKBONE_ATOMS:
            residue_atoms[key][atom.name] = torch.tensor([atom.x, atom.y, atom.z], dtype=torch.float32)

    chains: dict[str, list[_ParsedResidue]] = {}
    chain_order: list[str] = []
    for key in residue_order:
        residue_name = residue_names[key]
        if residue_name not in _THREE_TO_ONE and residue_name not in _ALPHABET_INDEX:
            continue  # skip non-protein residues (ligands, nucleotides, water)
        chain_id = key[0]
        if chain_id not in chains:
            chains[chain_id] = []
            chain_order.append(chain_id)
        one_letter = _THREE_TO_ONE.get(residue_name, "X")
        token = _ALPHABET_INDEX.get(one_letter, _UNKNOWN_INDEX)
        chains[chain_id].append(_ParsedResidue(token=token, atoms=residue_atoms[key]))

    return [
        ParsedChain(
            chain_id=chain_id,
            tokens=[residue.resolved_token() for residue in chains[chain_id]],
            backbone_coords=[residue.backbone_tensor() for residue in chains[chain_id]],
        )
        for chain_id in chain_order
    ]


def sequences_from_tokens(tokens: Iterable[int]) -> str:
    r"""
    Decode ProteinMPNN token ids back into a one-letter amino-acid string.

    Args:
        tokens: Iterable of token ids in the ProteinMPNN alphabet.

    Returns:
        The decoded one-letter amino-acid sequence (``X`` for the unknown/gap token).

    Examples:
        >>> sequences_from_tokens([0, 1, 2, 20])
        'ACDX'
    """

    return "".join(PROTEINMPNN_ALPHABET[int(token)] for token in tokens)


register_sequence_decoder("proteinmpnn", sequences_from_tokens)
