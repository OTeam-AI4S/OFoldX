r"""
AlphaFold 2 structure processing.
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data import residue_constants as rc
from ...data.structure import Structure
from ...processors.registry import PROCESSORS
from ...processors.structure import StructureProcessor
from ...processors.utils import _PROTEIN_RESIDUE_NAMES, StructureInput, structure_from_input
from .model import AlphaFold2Input


@PROCESSORS.register("alphafold2")
class AlphaFold2Processor(StructureProcessor):
    r"""
    Featurize one protein structure into the AlphaFold 2 input contract.

    The returned batch carries OFoldX structure-token fields required by the current
    PyTorch inference path. The same [`AlphaFold2Input`][] dataclass also
    accepts official processed AF2 fields (`target_feat`, `msa_feat`, templates, and
    extra-MSA tensors) for the full feature pipeline as it lands.

    Args:
        protein_only: Reject non-protein structures by default, matching official
            AlphaFold 2 monomer/pTM scope. Set to ``False`` only for the temporary
            shared scaffold or tests that intentionally exercise non-protein tokens.
        **kwargs: Shared [`StructureProcessor`][ofoldx.processors.structure.StructureProcessor]
            constructor arguments.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/alphafold2")  # doctest: +SKIP
        >>> batch = processor("protein.cif")  # doctest: +SKIP
    """

    input_class = AlphaFold2Input

    def __init__(self, *, protein_only: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.protein_only = protein_only

    def __call__(self, structure: StructureInput) -> AlphaFold2Input:
        r"""
        Return a batched ``AlphaFold2Input`` for one protein structure or sequence input.
        """

        if _looks_like_sequence_input(structure):
            sequence, msa = _read_sequence_input(structure)
            return self._sequence_batch(sequence, msa=msa)
        parsed = structure_from_input(structure)
        if self.protein_only:
            _validate_protein_only(parsed)
        batch = cast(AlphaFold2Input, super().__call__(parsed))
        batch.aatype = _aatype(parsed).unsqueeze(0)
        batch.target_feat = _target_feat_from_aatype(batch.aatype, dtype=batch.token_features.dtype)
        if batch.msa_feat is None:
            batch.msa_feat = _msa_feat_from_alignment(None, batch.aatype[0]).unsqueeze(0)
            batch.msa_mask = torch.ones(*batch.msa_feat.shape[:-1], dtype=batch.msa_feat.dtype)
        _set_dummy_extra_msa(batch, num_tokens=batch.aatype.shape[-1])
        _set_dummy_template(batch, num_tokens=batch.aatype.shape[-1])
        batch.atom37_index = _atom37_index(parsed).unsqueeze(0)
        return batch

    def _sequence_batch(self, sequence: str, *, msa: _ParsedAf2Msa | None = None) -> AlphaFold2Input:
        aatype = _sequence_aatype(sequence)
        num_tokens = int(aatype.numel())
        atom37_count = rc.atom_type_num
        num_atoms = num_tokens * atom37_count
        atom37_index = torch.arange(atom37_count, dtype=torch.long).repeat(num_tokens)
        atom_to_token = torch.arange(num_tokens, dtype=torch.long).repeat_interleave(atom37_count)
        token_to_rep_atom = torch.arange(num_tokens, dtype=torch.long) * atom37_count + rc.atom_order["CA"]
        atom_mask = _sequence_atom37_mask(aatype).reshape(num_atoms)
        token_mask = torch.ones(num_tokens, dtype=torch.float32)
        atom_features = torch.zeros(num_atoms, self.atom_feature_size)
        if self.atom_feature_size:
            atom_features[:, 0] = 1.0
        target_feat = _target_feat_from_aatype(aatype, dtype=torch.float32)
        msa_feat = _msa_feat_from_alignment(msa, aatype)
        num_msa = int(msa_feat.shape[0])
        zeros_atom_xyz = torch.zeros(num_atoms, 3)
        return AlphaFold2Input(
            token_mask=token_mask.unsqueeze(0),
            residue_index=torch.arange(num_tokens, dtype=torch.long).unsqueeze(0),
            chain_index=torch.zeros(1, num_tokens, dtype=torch.long),
            entity_index=torch.zeros(1, num_tokens, dtype=torch.long),
            symmetry_index=torch.zeros(1, num_tokens, dtype=torch.long),
            token_index=torch.arange(num_tokens, dtype=torch.long).unsqueeze(0),
            atom_features=atom_features.unsqueeze(0),
            atom_to_token=atom_to_token.unsqueeze(0),
            atom_mask=atom_mask.unsqueeze(0),
            reference_positions=zeros_atom_xyz.unsqueeze(0),
            reference_space_id=atom_to_token.unsqueeze(0),
            representative_atom_mask=torch.ones(1, num_tokens),
            atom_token_index=atom37_index.unsqueeze(0),
            token_to_rep_atom=token_to_rep_atom.unsqueeze(0),
            token_features=torch.zeros(1, num_tokens, self.token_feature_size),
            reference_mask=atom_mask.unsqueeze(0),
            reference_charges=torch.zeros(1, num_atoms),
            target_positions=zeros_atom_xyz.unsqueeze(0),
            target_mask=torch.zeros(1, num_atoms),
            molecule_type=torch.zeros(1, num_tokens, dtype=torch.long),
            frame_atom_index=_sequence_frame_atom_index(num_tokens).unsqueeze(0),
            frame_mask=torch.ones(1, num_tokens, dtype=torch.bool),
            aatype=aatype.unsqueeze(0),
            target_feat=target_feat.unsqueeze(0),
            msa_feat=msa_feat.unsqueeze(0),
            msa_mask=torch.ones(1, num_msa, num_tokens),
            extra_msa=torch.zeros(1, 1, num_tokens, dtype=torch.long),
            extra_has_deletion=torch.zeros(1, 1, num_tokens),
            extra_deletion_value=torch.zeros(1, 1, num_tokens),
            extra_msa_mask=torch.zeros(1, 1, num_tokens),
            atom37_index=atom37_index.unsqueeze(0),
            template_mask=torch.zeros(1, 1),
            template_aatype=torch.zeros(1, 1, num_tokens, dtype=torch.long),
            template_all_atom_positions=torch.zeros(1, 1, num_tokens, atom37_count, 3),
            template_all_atom_masks=torch.zeros(1, 1, num_tokens, atom37_count),
            template_pseudo_beta=torch.zeros(1, 1, num_tokens, 3),
            template_pseudo_beta_mask=torch.zeros(1, 1, num_tokens),
            template_torsion_angles_sin_cos=torch.zeros(1, 1, num_tokens, 7, 2),
            template_alt_torsion_angles_sin_cos=torch.zeros(1, 1, num_tokens, 7, 2),
            template_torsion_angles_mask=torch.zeros(1, 1, num_tokens, 7),
        )

    def _processor_config(self) -> dict[str, object]:
        config = super()._processor_config()
        config["protein_only"] = self.protein_only
        return config


def _validate_protein_only(structure: Structure) -> None:
    r"""
    Raise if ``structure`` contains residues outside the official AF2 protein scope.
    """

    for atom in structure.atoms:
        mol_type = atom.mol_type.lower() if atom.mol_type is not None else None
        is_protein = mol_type == "protein" or (mol_type is None and atom.residue_name in _PROTEIN_RESIDUE_NAMES)
        if atom.hetero or not is_protein:
            raise ValueError(
                "AlphaFold2Processor is protein-only by default; set protein_only=False to use the shared "
                f"structure-token adapter with residue {atom.residue_name!r}"
            )


def _aatype(structure: Structure) -> Tensor:
    residue_to_aatype: dict[tuple[str, int, str], int] = {}
    order: list[tuple[str, int, str]] = []
    restype_order = {letter: index for index, letter in enumerate(rc.restypes)}
    unknown = len(rc.restypes)
    for atom in structure.atoms:
        key = (atom.residue.chain_id, atom.residue.residue_number, atom.residue.insertion_code)
        if key not in residue_to_aatype:
            residue_to_aatype[key] = restype_order.get(rc.restype_3to1.get(atom.residue_name, "X"), unknown)
            order.append(key)
    return torch.tensor([residue_to_aatype[key] for key in order], dtype=torch.long)


def _atom37_index(structure: Structure) -> Tensor:
    return torch.tensor([rc.atom_order.get(atom.name, 0) for atom in structure.atoms], dtype=torch.long)


@dataclass(frozen=True, slots=True)
class _ParsedAf2Msa:
    token_ids: tuple[tuple[int, ...], ...]
    deletions: tuple[tuple[int, ...], ...]


def _looks_like_sequence_input(value: StructureInput) -> bool:
    if isinstance(value, Structure):
        return False
    if isinstance(value, PathLike):
        path = Path(value)
        return _is_sequence_path(path)
    text = str(value)
    path = Path(text)
    if _path_exists(path):
        return _is_sequence_path(path)
    stripped = "".join(line.strip() for line in text.splitlines() if not line.lstrip().startswith(">"))
    return bool(stripped) and all(character.upper() in _AA_TO_AF2_ID for character in stripped)


def _read_sequence_input(value: StructureInput) -> tuple[str, _ParsedAf2Msa | None]:
    if isinstance(value, Structure):
        raise TypeError("Structure objects are not sequence inputs")
    path = Path(value) if isinstance(value, PathLike) else Path(str(value))
    if isinstance(value, PathLike) or _path_exists(path):
        if _is_a3m_path(path):
            msa = _parse_a3m(path)
            return _sequence_from_msa(msa), msa
        text = path.read_text()
    else:
        text = str(value)
    sequence = "".join(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(">"))
    sequence = sequence.upper()
    if not sequence:
        raise ValueError("AlphaFold2Processor sequence input is empty")
    invalid = sorted({character for character in sequence if character not in _AA_TO_AF2_ID})
    if invalid:
        raise ValueError(f"AlphaFold2Processor sequence contains unsupported residue letters: {''.join(invalid)}")
    return sequence, None


_AA_TO_AF2_ID = {letter: index for index, letter in enumerate(rc.restypes)}
_AA_TO_AF2_ID["X"] = rc.restype_num
_GAP_AF2_ID = rc.restype_num + 1


def _sequence_aatype(sequence: str) -> Tensor:
    return torch.tensor([_AA_TO_AF2_ID[letter] for letter in sequence], dtype=torch.long)


def _target_feat_from_aatype(aatype: Tensor, *, dtype: torch.dtype) -> Tensor:
    aatype_1hot = F.one_hot(aatype.clamp(min=0, max=rc.restype_num), num_classes=rc.restype_num + 1)
    has_break = torch.zeros(*aatype.shape, 1, dtype=aatype_1hot.dtype, device=aatype.device)
    return torch.cat([has_break, aatype_1hot], dim=-1).to(dtype=dtype)


def _msa_feat_from_alignment(msa: _ParsedAf2Msa | None, aatype: Tensor) -> Tensor:
    if msa is None:
        msa_tokens = aatype.clamp(min=0, max=rc.restype_num).unsqueeze(0)
        deletions = torch.zeros_like(msa_tokens, dtype=torch.float32)
    else:
        msa_tokens = torch.tensor(msa.token_ids, dtype=torch.long)
        deletions = torch.tensor(msa.deletions, dtype=torch.float32)
        if msa_tokens.shape[-1] != aatype.numel():
            raise ValueError(f"MSA query length {msa_tokens.shape[-1]} does not match sequence length {aatype.numel()}")
    msa_one_hot = F.one_hot(msa_tokens.clamp(min=0, max=22), num_classes=23).float()
    has_deletion = (deletions > 0).float().unsqueeze(-1)
    deletion_value = ((2.0 / math.pi) * torch.atan(deletions / 3.0)).unsqueeze(-1)
    profile = msa_one_hot.mean(dim=0, keepdim=True).expand_as(msa_one_hot)
    deletion_mean = deletion_value.mean(dim=0, keepdim=True).expand_as(deletion_value)
    return torch.cat([msa_one_hot, has_deletion, deletion_value, profile, deletion_mean], dim=-1)


def _set_dummy_extra_msa(batch: AlphaFold2Input, *, num_tokens: int) -> None:
    batch.extra_msa = torch.zeros(1, 1, num_tokens, dtype=torch.long)
    batch.extra_has_deletion = torch.zeros(1, 1, num_tokens)
    batch.extra_deletion_value = torch.zeros(1, 1, num_tokens)
    batch.extra_msa_mask = torch.zeros(1, 1, num_tokens)


def _set_dummy_template(batch: AlphaFold2Input, *, num_tokens: int) -> None:
    batch.template_mask = torch.zeros(1, 1)
    batch.template_aatype = torch.zeros(1, 1, num_tokens, dtype=torch.long)
    batch.template_all_atom_positions = torch.zeros(1, 1, num_tokens, rc.atom_type_num, 3)
    batch.template_all_atom_masks = torch.zeros(1, 1, num_tokens, rc.atom_type_num)
    batch.template_pseudo_beta = torch.zeros(1, 1, num_tokens, 3)
    batch.template_pseudo_beta_mask = torch.zeros(1, 1, num_tokens)
    batch.template_torsion_angles_sin_cos = torch.zeros(1, 1, num_tokens, 7, 2)
    batch.template_alt_torsion_angles_sin_cos = torch.zeros(1, 1, num_tokens, 7, 2)
    batch.template_torsion_angles_mask = torch.zeros(1, 1, num_tokens, 7)


def _parse_a3m(path: Path) -> _ParsedAf2Msa:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt") as handle:
            rows = [line.strip() for line in handle if line.strip() and not line.startswith(("#", ">"))]
    else:
        rows = [
            line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith(("#", ">"))
        ]
    token_rows: list[tuple[int, ...]] = []
    deletion_rows: list[tuple[int, ...]] = []
    seen: set[str] = set()
    for row in rows:
        ungapped = row.replace("-", "").upper()
        if ungapped in seen:
            continue
        seen.add(ungapped)
        tokens: list[int] = []
        deletions: list[int] = []
        insertion_count = 0
        for character in row:
            if character != "-" and character.islower():
                insertion_count += 1
                continue
            tokens.append(_msa_token_id(character))
            deletions.append(insertion_count)
            insertion_count = 0
        token_rows.append(tuple(tokens))
        deletion_rows.append(tuple(deletions))
    if not token_rows:
        raise ValueError("A3M file contains no aligned sequences")
    query_length = len(token_rows[0])
    if any(len(row) != query_length for row in token_rows):
        raise ValueError("A3M rows have inconsistent alignment lengths")
    return _ParsedAf2Msa(token_ids=tuple(token_rows), deletions=tuple(deletion_rows))


def _sequence_from_msa(msa: _ParsedAf2Msa) -> str:
    letters = []
    for token_id in msa.token_ids[0]:
        if token_id == _GAP_AF2_ID:
            raise ValueError("A3M query row must not contain gap columns")
        if token_id >= rc.restype_num:
            letters.append("X")
        else:
            letters.append(rc.restypes[token_id])
    return "".join(letters)


def _msa_token_id(character: str) -> int:
    if character == "-":
        return _GAP_AF2_ID
    return _AA_TO_AF2_ID.get(character.upper(), rc.restype_num)


def _sequence_atom37_mask(aatype: Tensor) -> Tensor:
    masks: list[list[float]] = []
    for residue_id in aatype.tolist():
        residue_id = min(int(residue_id), rc.restype_num)
        if residue_id >= rc.restype_num:
            masks.append([0.0] * rc.atom_type_num)
            continue
        residue_name = rc.restype_1to3[rc.restypes[residue_id]]
        residue_atoms = {atom_name for atom_name in rc.restype_name_to_atom14_names[residue_name] if atom_name}
        masks.append([1.0 if atom_name in residue_atoms else 0.0 for atom_name in rc.atom_types])
    return torch.tensor(masks, dtype=torch.float32)


def _sequence_frame_atom_index(num_tokens: int) -> Tensor:
    residue_offset = torch.arange(num_tokens, dtype=torch.long) * rc.atom_type_num
    frame_atoms = torch.tensor([rc.atom_order["N"], rc.atom_order["CA"], rc.atom_order["C"]], dtype=torch.long)
    return residue_offset.unsqueeze(-1) + frame_atoms


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _is_sequence_path(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return bool(suffixes) and (
        suffixes[-1] in {".fa", ".faa", ".fasta", ".seq", ".a3m"} or suffixes[-2:] == [".a3m", ".gz"]
    )


def _is_a3m_path(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return suffixes[-1:] == [".a3m"] or suffixes[-2:] == [".a3m", ".gz"]
