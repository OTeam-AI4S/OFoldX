r"""
NA-MPNN structure processing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ...data.structure import Structure
from ...processors.mixin import ProcessorMixin
from ...processors.registry import PROCESSORS
from ...processors.utils import StructureInput, structure_from_input
from .model import (
    NAMPNN_ATOM_NAMES,
    NAMPNN_LETTER_TO_TOKEN,
    NAMPNN_POLYMER_TYPE_TO_ID,
    NAMPNN_RESTYPE_TO_ID,
    NAMPNNInput,
)

_PROTEIN_RESTYPES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
)
_PROTEIN_ALIASES = {"MSE": "MET"}
_DNA_RESTYPES = ("DA", "DC", "DG", "DT", "DX")
_RNA_RESTYPES = ("A", "C", "G", "U", "RX")
_PROTEIN_BACKBONE_ATOMS = ("N", "CA", "C", "O")
_DNA_BACKBONE_ATOMS = ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
_RNA_BACKBONE_ATOMS = ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
_RNA_TO_SHARED_DNA_TOKEN = {"A": "DA", "C": "DC", "G": "DG", "U": "DT", "RX": "DX"}


@PROCESSORS.register("nampnn")
class NAMPNNProcessor(ProcessorMixin):
    r"""
    Build NA-MPNN input tensors from a protein, DNA, RNA, or mixed-polymer structure.

    Args:
        designed_chains: Chain ids to design. ``None`` designs every parsed chain unless ``design_na_only`` is set.
        design_na_only: Restrict design positions to DNA/RNA chains.
        fixed_positions: Per-chain 1-based residue numbers to keep fixed.
        redesigned_positions: Per-chain 1-based residue numbers to design exclusively.
        omit_residue_tokens: Global residue tokens forbidden during sampling; accepts one-letter NA-MPNN symbols or
            three-letter residue names.
        shared_nucleic_acid_tokens: Map RNA A/C/G/U/RX onto the shared DNA token ids used by released checkpoints.
        load_residues_with_missing_atoms: Keep polymer tokens designable even when their backbone atom set is
            incomplete.
        temperature: Default sampling temperature attached to the model input.
        num_samples: Optional default number of autoregressive samples.
    """

    input_class = NAMPNNInput

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> NAMPNNProcessor:
        r"""
        Build a processor whose token-sharing option matches a model config.

        Args:
            config: Model config exposing ``shared_nucleic_acid_tokens``.
            **kwargs: Constructor overrides.

        Returns:
            Configured processor.
        """

        parameters: dict[str, Any] = {
            "shared_nucleic_acid_tokens": bool(_config_scalar(config, "shared_nucleic_acid_tokens", True))
        }
        parameters.update(kwargs)
        return cls(**parameters)

    def __init__(
        self,
        *,
        designed_chains: Sequence[str] | None = None,
        design_na_only: bool = False,
        fixed_positions: Mapping[str, Sequence[int]] | None = None,
        redesigned_positions: Mapping[str, Sequence[int]] | None = None,
        omit_residue_tokens: Sequence[str] | str = ("X",),
        shared_nucleic_acid_tokens: bool = True,
        load_residues_with_missing_atoms: bool = False,
        temperature: float = 0.1,
        num_samples: int | None = None,
    ) -> None:
        self.designed_chains = None if designed_chains is None else [str(chain) for chain in designed_chains]
        self.design_na_only = bool(design_na_only)
        self.fixed_positions = _normalize_positions(fixed_positions)
        self.redesigned_positions = _normalize_positions(redesigned_positions)
        self.omit_residue_tokens = _normalize_omit_tokens(omit_residue_tokens, shared_nucleic_acid_tokens)
        self.shared_nucleic_acid_tokens = bool(shared_nucleic_acid_tokens)
        self.load_residues_with_missing_atoms = bool(load_residues_with_missing_atoms)
        self.temperature = float(temperature)
        self.num_samples = None if num_samples is None else int(num_samples)

    def __call__(self, structure: StructureInput) -> NAMPNNInput:
        r"""
        Featurize one structure into a single-backbone NA-MPNN input.

        Args:
            structure: Parsed structure or structure file path.

        Returns:
            Batched NA-MPNN input with batch axis size one.

        Raises:
            ValueError: If no protein/DNA/RNA residue with a reference atom is found.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/nampnn-design")  # doctest: +SKIP
            >>> batch = processor("complex.pdb")  # doctest: +SKIP
        """

        parsed = parse_structure_to_nampnn_residues(
            structure_from_input(structure),
            shared_nucleic_acid_tokens=self.shared_nucleic_acid_tokens,
            load_residues_with_missing_atoms=self.load_residues_with_missing_atoms,
        )
        if not parsed:
            raise ValueError("NA-MPNN processing requires at least one protein, DNA, or RNA residue")
        chain_to_index: dict[str, int] = {}
        chain_order: list[str] = []
        for residue in parsed:
            if residue.chain_id not in chain_to_index:
                chain_to_index[residue.chain_id] = len(chain_to_index)
                chain_order.append(residue.chain_id)

        designed_chains = set(self.designed_chains) if self.designed_chains is not None else set(chain_order)
        coords = torch.stack([residue.atom_coords for residue in parsed])[None, :]
        atom_mask = torch.stack([residue.atom_mask for residue in parsed])[None, :]
        sequence = torch.tensor([residue.token_id for residue in parsed], dtype=torch.long)[None, :]
        token_mask = torch.tensor([float(residue.valid) for residue in parsed])[None, :]
        residue_index = torch.tensor(_renumber_residue_indices([residue.residue_number for residue in parsed]))[None, :]
        chain_index = torch.tensor([chain_to_index[residue.chain_id] for residue in parsed], dtype=torch.long)[None, :]
        polymer_type_ids = torch.tensor([residue.polymer_type_id for residue in parsed], dtype=torch.long)[None, :]
        protein_mask = torch.tensor([float(residue.is_protein) for residue in parsed])[None, :]
        dna_mask = torch.tensor([float(residue.is_dna) for residue in parsed])[None, :]
        rna_mask = torch.tensor([float(residue.is_rna) for residue in parsed])[None, :]
        na_mask = torch.tensor([float(residue.is_dna or residue.is_rna) for residue in parsed])[None, :]
        chain_design_mask = torch.tensor(
            [float(residue.chain_id in designed_chains) for residue in parsed],
            dtype=torch.float32,
        )[None, :]
        if self.design_na_only:
            chain_design_mask = chain_design_mask * na_mask
        fixed_position_mask = self._fixed_position_mask(parsed)
        design_mask = token_mask * chain_design_mask * fixed_position_mask * self._redesigned_mask(parsed)

        return NAMPNNInput(
            atom_coords=coords,
            atom_mask=atom_mask,
            sequence_tokens=sequence,
            token_mask=token_mask,
            residue_index=residue_index.long(),
            chain_index=chain_index,
            polymer_type_ids=polymer_type_ids,
            protein_mask=protein_mask,
            dna_mask=dna_mask,
            rna_mask=rna_mask,
            design_mask=design_mask,
            amino_acid_bias=self._omit_bias(len(parsed)),
            temperature=self.temperature,
            num_samples=self.num_samples,
            fixed_position_mask=fixed_position_mask,
            rna_token_conversion_mask=torch.tensor([float(residue.is_rna) for residue in parsed])[None, :],
        )

    def _fixed_position_mask(self, residues: Sequence[ParsedNAMPNNResidue]) -> Tensor:
        r"""
        Build the official fixed-position mask: one means designable, zero means hard-fixed.
        """

        values = []
        for residue in residues:
            fixed = set(self.fixed_positions.get(residue.chain_id, ()))
            values.append(0.0 if residue.residue_number in fixed else 1.0)
        return torch.tensor(values, dtype=torch.float32)[None, :]

    def _redesigned_mask(self, residues: Sequence[ParsedNAMPNNResidue]) -> Tensor:
        r"""
        Return an exclusive-redesign mask when ``redesigned_positions`` was configured.
        """

        if not self.redesigned_positions:
            return torch.ones(1, len(residues), dtype=torch.float32)
        values = []
        for residue in residues:
            redesigned = set(self.redesigned_positions.get(residue.chain_id, ()))
            values.append(1.0 if residue.residue_number in redesigned else 0.0)
        return torch.tensor(values, dtype=torch.float32)[None, :]

    def _omit_bias(self, length: int) -> Tensor:
        r"""
        Build the global omitted-token bias tensor.
        """

        bias = torch.zeros(1, length, len(NAMPNN_RESTYPE_TO_ID), dtype=torch.float32)
        for token in self.omit_residue_tokens:
            bias[..., NAMPNN_RESTYPE_TO_ID[token]] = -1e8
        return bias

    def _processor_config(self) -> dict[str, object]:
        r"""
        Serialize constructor options into a processor config.
        """

        return {
            "designed_chains": self.designed_chains,
            "design_na_only": self.design_na_only,
            "fixed_positions": {chain: list(positions) for chain, positions in self.fixed_positions.items()},
            "redesigned_positions": {chain: list(positions) for chain, positions in self.redesigned_positions.items()},
            "omit_residue_tokens": list(self.omit_residue_tokens),
            "shared_nucleic_acid_tokens": self.shared_nucleic_acid_tokens,
            "load_residues_with_missing_atoms": self.load_residues_with_missing_atoms,
            "temperature": self.temperature,
            "num_samples": self.num_samples,
        }


@dataclass(slots=True)
class ParsedNAMPNNResidue:
    r"""
    Parsed polymer residue in NA-MPNN atom and token space.

    Attributes:
        chain_id: Source chain id.
        residue_number: Source residue number.
        residue_name: Source residue name.
        token_id: NA-MPNN residue token id.
        atom_coords: Atom coords in NA-MPNN atom order.
        atom_mask: Atom-validity mask in NA-MPNN atom order.
        polymer_type_id: NA-MPNN polymer-type id.
        is_protein: Whether the token is protein.
        is_dna: Whether the token is DNA.
        is_rna: Whether the token is RNA.
        valid: Whether the residue has the required backbone atoms for its polymer type.
    """

    chain_id: str
    residue_number: int
    residue_name: str
    token_id: int
    atom_coords: Tensor
    atom_mask: Tensor
    polymer_type_id: int
    is_protein: bool
    is_dna: bool
    is_rna: bool
    valid: bool


@dataclass(slots=True)
class _ResidueBuilder:
    r"""
    Mutable residue accumulator used while parsing a structure.
    """

    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    atoms: dict[str, Tensor] = field(default_factory=dict)


def parse_structure_to_nampnn_residues(
    structure: Structure,
    *,
    shared_nucleic_acid_tokens: bool = True,
    load_residues_with_missing_atoms: bool = False,
) -> list[ParsedNAMPNNResidue]:
    r"""
    Parse a structure into NA-MPNN residue records.

    Args:
        structure: Parsed structure.
        shared_nucleic_acid_tokens: Map RNA residues to shared DNA token ids.
        load_residues_with_missing_atoms: Keep residues valid even when required backbone atoms are missing.

    Returns:
        Parsed residues in first-reference-atom order.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> key = ResidueKey("A", 1, "")
        >>> atoms = (
        ...     AtomRecord("N", "N", "ALA", key, 0.0, 0.0, 0.0),
        ...     AtomRecord("CA", "C", "ALA", key, 1.0, 0.0, 0.0),
        ...     AtomRecord("C", "C", "ALA", key, 2.0, 0.0, 0.0),
        ...     AtomRecord("O", "O", "ALA", key, 3.0, 0.0, 0.0),
        ... )
        >>> residues = parse_structure_to_nampnn_residues(Structure(atoms))
        >>> residues[0].token_id == NAMPNN_RESTYPE_TO_ID["ALA"]
        True
    """

    builders: dict[tuple[str, int, str], _ResidueBuilder] = {}
    order: list[tuple[str, int, str]] = []
    for atom in structure.atoms:
        key = (atom.residue.chain_id, atom.residue.residue_number, atom.residue.insertion_code)
        if key not in builders:
            builders[key] = _ResidueBuilder(
                chain_id=atom.residue.chain_id,
                residue_number=atom.residue.residue_number,
                insertion_code=atom.residue.insertion_code,
                residue_name=atom.residue_name.upper(),
            )
            order.append(key)
        atom_name = atom.name.strip()
        if atom_name in NAMPNN_ATOM_NAMES:
            builders[key].atoms[atom_name] = torch.tensor([atom.x, atom.y, atom.z], dtype=torch.float32)

    parsed: list[ParsedNAMPNNResidue] = []
    for key in order:
        residue = builders[key]
        residue_name = _PROTEIN_ALIASES.get(residue.residue_name, residue.residue_name)
        kind = _polymer_kind(residue_name)
        if kind is None:
            continue
        reference_atom = "CA" if kind == "protein" else "C1'"
        if reference_atom not in residue.atoms:
            continue
        atom_coords, atom_mask = _atom_tensors(residue.atoms)
        valid = load_residues_with_missing_atoms or _has_required_backbone(residue.atoms, kind)
        token_name = _token_name(residue_name, kind, shared_nucleic_acid_tokens)
        parsed.append(
            ParsedNAMPNNResidue(
                chain_id=residue.chain_id,
                residue_number=residue.residue_number,
                residue_name=residue_name,
                token_id=NAMPNN_RESTYPE_TO_ID[token_name],
                atom_coords=atom_coords,
                atom_mask=atom_mask,
                polymer_type_id=NAMPNN_POLYMER_TYPE_TO_ID[_polymer_type_name(kind)],
                is_protein=valid and kind == "protein",
                is_dna=valid and kind == "dna",
                is_rna=valid and kind == "rna",
                valid=valid,
            )
        )
    return parsed


def _config_scalar(config: Any, key: str, default: object) -> object:
    r"""
    Read a scalar from either a mapping or an attribute-bearing config.
    """

    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_positions(values: Mapping[str, Sequence[int]] | None) -> dict[str, list[int]]:
    r"""
    Normalize chain-position mappings to ``str -> list[int]``.
    """

    return {str(chain): [int(position) for position in positions] for chain, positions in (values or {}).items()}


def _normalize_omit_tokens(tokens: Sequence[str] | str, shared_nucleic_acid_tokens: bool) -> list[str]:
    r"""
    Normalize omitted one-letter or three-letter tokens to NA-MPNN residue names.
    """

    raw_tokens: Iterable[str]
    if isinstance(tokens, str):
        raw_tokens = [tokens] if tokens in NAMPNN_RESTYPE_TO_ID else list(tokens)
    else:
        raw_tokens = tokens
    normalized: list[str] = []
    for token in raw_tokens:
        normalized_token = _token_from_symbol(str(token))
        if normalized_token not in normalized:
            normalized.append(normalized_token)
    if shared_nucleic_acid_tokens:
        for token in ("A", "C", "G", "U", "RX"):
            if token not in normalized:
                normalized.append(token)
    return normalized


def _token_from_symbol(symbol: str) -> str:
    r"""
    Resolve a one-letter or three-letter residue symbol to an NA-MPNN token name.
    """

    if symbol in NAMPNN_LETTER_TO_TOKEN:
        return NAMPNN_LETTER_TO_TOKEN[symbol]
    if symbol in NAMPNN_RESTYPE_TO_ID:
        return symbol
    raise ValueError(f"unknown NA-MPNN residue token {symbol!r}")


def _polymer_kind(residue_name: str) -> str | None:
    r"""
    Return ``protein``, ``dna``, ``rna``, or ``None`` for unsupported residues.
    """

    if residue_name in _PROTEIN_RESTYPES:
        return "protein"
    if residue_name in _DNA_RESTYPES:
        return "dna"
    if residue_name in _RNA_RESTYPES:
        return "rna"
    return None


def _polymer_type_name(kind: str) -> str:
    r"""
    Return the NA-MPNN polymer-type name for a parsed polymer kind.
    """

    if kind == "protein":
        return "PP"
    if kind == "dna":
        return "DNA"
    if kind == "rna":
        return "RNA"
    return "UNK"


def _token_name(residue_name: str, kind: str, shared_nucleic_acid_tokens: bool) -> str:
    r"""
    Return the model token name for a parsed residue.
    """

    if kind == "protein":
        return residue_name if residue_name in NAMPNN_RESTYPE_TO_ID else "UNK"
    if kind == "rna" and shared_nucleic_acid_tokens:
        return _RNA_TO_SHARED_DNA_TOKEN.get(residue_name, "DX")
    return residue_name if residue_name in NAMPNN_RESTYPE_TO_ID else ("DX" if kind == "dna" else "RX")


def _atom_tensors(atoms: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    r"""
    Stack residue atoms into NA-MPNN atom-order coords and masks.
    """

    coords = []
    mask = []
    for atom_name in NAMPNN_ATOM_NAMES:
        if atom_name in atoms:
            coords.append(atoms[atom_name])
            mask.append(1.0)
        else:
            coords.append(torch.zeros(3, dtype=torch.float32))
            mask.append(0.0)
    return torch.stack(coords), torch.tensor(mask, dtype=torch.float32)


def _has_required_backbone(atoms: Mapping[str, Tensor], kind: str) -> bool:
    r"""
    Return whether the residue has the required backbone atom set for its polymer kind.
    """

    if kind == "protein":
        return all(atom in atoms for atom in _PROTEIN_BACKBONE_ATOMS)
    if kind == "dna":
        return all(atom in atoms for atom in _DNA_BACKBONE_ATOMS)
    if kind == "rna":
        return all(atom in atoms for atom in _RNA_BACKBONE_ATOMS)
    return False


def _renumber_residue_indices(residue_numbers: Sequence[int]) -> list[int]:
    r"""
    Renumber duplicate consecutive residue numbers like the official NA-MPNN featurizer.
    """

    output: list[int] = []
    count = 0
    previous = -100000
    for residue_number in residue_numbers:
        if previous == residue_number:
            count += 1
        output.append(int(residue_number) + count)
        previous = int(residue_number)
    return output
