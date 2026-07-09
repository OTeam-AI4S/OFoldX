r"""
Shared structure-to-model-input processing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from ..data.ccd import CcdProvider, ReferenceComponent, default_ccd_provider
from ..data.residue_constants import HYDROGEN_ELEMENTS, LEAVING_ATOMS
from ..data.structure import AtomRecord, ResidueKey, Structure
from .mixin import ProcessorMixin
from .utils import (
    _DNA_RESIDUE_NAMES,
    _PROTEIN_RESIDUE_NAMES,
    _RNA_RESIDUE_NAMES,
    _STANDARD_RESIDUE_NAMES,
    StructuralFields,
    StructureInput,
    StructureToken,
    _atom_to_token,
    _atom_token_fields,
    _index_values,
    _reference_space_ids,
    _token_residue_names,
    structure_from_input,
    tokenize_structure,
)


class StructureProcessor(ProcessorMixin):
    r"""
    Shared scaffold for processors that emit structure-derived atom and token fields.

    Args:
        single_size: Width of the single representation prefix reserved in token features.
        token_feature_size: Total token feature width emitted by the processor.
        atom_feature_size: Total atom feature width emitted by the processor.
        residue_vocabulary: Optional residue vocabulary override.
        element_vocabulary: Optional element vocabulary override.
        element_channels: Number of element one-hot channels.
        atom_name_characters_count: Number of atom-name characters to encode.
        atom_name_character_channels: Number of one-hot channels per atom-name character.
        ccd: Optional CCD provider override. Processors with custom providers cannot be saved.
        reference_conformer_augment: If ``True``, randomly rigid-transform reference conformers.
        augment_seed: Optional seed for reference-conformer augmentation.
        ligand_atom_rename: If ``True``, rename ligand atom-name features to ``ELEM+count``.
        materialize_missing_atoms: If ``True``, synthesize expected-but-unresolved heavy atoms.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/model")  # doctest: +SKIP
        >>> model_input = processor("complex.cif")  # doctest: +SKIP
    """

    input_class: type[Any]

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> Self:
        r"""
        Build a processor with dimensions taken from a model config.

        Args:
            config: Model config carrying feature dimensions.
            **kwargs: Processor constructor overrides.

        Returns:
            Configured processor instance.

        Examples:
            >>> processor = StructureProcessor.from_config(config)  # doctest: +SKIP
        """

        parameters: dict[str, Any] = {
            "single_size": int(getattr(config, "single_size", 384)),
            "token_feature_size": int(getattr(config, "token_feature_size", 449)),
            "atom_feature_size": int(getattr(config, "atom_feature_size", 385)),
        }
        parameters.update(kwargs)
        return cls(**parameters)

    def __init__(
        self,
        *,
        single_size: int = 384,
        token_feature_size: int = 449,
        atom_feature_size: int = 385,
        residue_vocabulary: Sequence[str] | None = None,
        element_vocabulary: Sequence[str] | None = None,
        element_channels: int = 128,
        atom_name_characters_count: int = 4,
        atom_name_character_channels: int = 64,
        ccd: CcdProvider | None = None,
        reference_conformer_augment: bool = False,
        augment_seed: int | None = None,
        ligand_atom_rename: bool = False,
        materialize_missing_atoms: bool = False,
    ) -> None:
        self._ccd = ccd
        self._augment_seed = augment_seed
        # Materialize expected-but-missing heavy atoms within present residues so target_mask becomes
        # a real resolved mask (target-resolved coords). Off by default (changes the atom set); training on.
        self.materialize_missing_atoms = materialize_missing_atoms
        # Per-residue random rigid augmentation of the reference conformer (training
        # regularization). Off by default so the processor is deterministic for eval/parity.
        self._reference_conformer_augment = reference_conformer_augment
        self._augment_rng = np.random.default_rng(augment_seed)
        # Rename ligand atoms to ELEM+count in the atom-name feature (atom-name anti-leakage augmentation).
        self._ligand_atom_rename = ligand_atom_rename
        self.single_size = single_size
        self.token_feature_size = token_feature_size
        self.atom_feature_size = atom_feature_size
        self.residue_vocabulary = tuple(
            default_residue_vocabulary() if residue_vocabulary is None else residue_vocabulary
        )
        self.element_vocabulary = tuple(
            default_element_vocabulary() if element_vocabulary is None else element_vocabulary
        )
        self._residue_indices = {value: index for index, value in enumerate(self.residue_vocabulary)}
        self._element_indices = {value: index for index, value in enumerate(self.element_vocabulary)}
        self.element_channels = element_channels
        self.atom_name_characters_count = atom_name_characters_count
        self.atom_name_character_channels = atom_name_character_channels
        # Unknown residues fall back to the ``UNK`` class; unknown elements fall back to the first
        # padding class above the real elements (the schema reserves indices 118-127 for unknown elements).
        self._residue_unknown_index = self._residue_indices.get("UNK", 0)
        self._element_unknown_index = min(len(self.element_vocabulary), element_channels - 1)

    def _processor_config(self) -> dict[str, object]:
        if self._ccd is not None:
            raise ValueError("processors with custom CCD providers cannot be saved")
        return {
            "single_size": self.single_size,
            "token_feature_size": self.token_feature_size,
            "atom_feature_size": self.atom_feature_size,
            "residue_vocabulary": list(self.residue_vocabulary),
            "element_vocabulary": list(self.element_vocabulary),
            "element_channels": self.element_channels,
            "atom_name_characters_count": self.atom_name_characters_count,
            "atom_name_character_channels": self.atom_name_character_channels,
            "reference_conformer_augment": self._reference_conformer_augment,
            "augment_seed": self._augment_seed,
            "ligand_atom_rename": self._ligand_atom_rename,
            "materialize_missing_atoms": self.materialize_missing_atoms,
        }

    def _structural_fields(self, structure: Structure, ccd: CcdProvider) -> StructuralFields:
        r"""
        Build the model-agnostic structural pieces shared by structure processors.

        Tokenizes the (optionally atom-materialized) structure, partitions atoms into tokens, and computes
        the identity indices, reference-conformer geometry/charges, per-token frames, and the
        representative-atom fields. The caller layers model-specific encodings (token/atom feature
        vectors, residue-type vocabulary, bond typing) on top.

        Args:
            structure: Parsed structure to featurize.
            ccd: CCD provider supplying reference conformers and bonds.

        Returns:
            The shared structural fields.
        """

        # Expand to the full expected heavy-atom set so target_mask becomes a real resolved mask;
        # the rest of the pipeline runs unchanged on the expanded structure.
        if self.materialize_missing_atoms:
            structure, resolved = materialize_missing_atoms(structure, ccd)
        else:
            resolved = None

        tokens = tokenize_structure(structure)
        residues = tuple(token.residue for token in tokens)
        residue_names = _token_residue_names(structure, tokens)
        chain_to_index = _index_values(residue.chain_id for residue in residues)
        num_tokens = len(tokens)
        num_atoms = len(structure.atoms)

        atom_to_token_values = _atom_to_token(tokens, num_atoms)
        atom_to_token = torch.tensor(atom_to_token_values, dtype=torch.long)
        # Experimental (target) coords — the training labels, NOT the reference conformer.
        target_positions = torch.tensor([[atom.x, atom.y, atom.z] for atom in structure.atoms], dtype=torch.float32)
        atom_mask = torch.ones(num_atoms, dtype=torch.bool)
        # target_mask = is_resolved (a true experimental coordinate exists); equals atom_mask unless
        # missing atoms were materialized.
        target_mask = atom_mask if resolved is None else torch.from_numpy(resolved)
        token_mask = torch.ones(num_tokens, dtype=torch.bool)
        representative_atom_mask, atom_token_index, token_to_rep_atom = _atom_token_fields(tokens, num_atoms)
        entity_values = _entity_indices(residues, residue_names)
        reference_space_values = _reference_space_ids(structure)
        molecule_type_values = [atom_molecule_type(structure.atoms[token.atom_indices[0]]) for token in tokens]
        # Reference conformer geometry / bonds from the CCD (independent of the experimental coords).
        reference_positions, reference_mask, reference_charges = _reference_features(
            structure,
            reference_space_values,
            ccd,
            augment=self._reference_conformer_augment,
            rng=self._augment_rng,
        )
        # Frames need the reference conformer: ligand/atom tokens use nearest-neighbour ref atoms.
        frame_atom_index_values, frame_mask_values = _token_frames(
            structure, tokens, reference_positions, reference_mask, reference_space_values
        )

        return StructuralFields(
            structure=structure,
            tokens=tokens,
            residues=residues,
            residue_names=residue_names,
            chain_to_index=chain_to_index,
            num_tokens=num_tokens,
            num_atoms=num_atoms,
            atom_to_token_values=atom_to_token_values,
            atom_to_token=atom_to_token,
            atom_mask=atom_mask,
            token_mask=token_mask,
            representative_atom_mask=representative_atom_mask,
            atom_token_index=atom_token_index,
            token_to_rep_atom=token_to_rep_atom,
            entity_values=entity_values,
            reference_space_values=reference_space_values,
            reference_positions=reference_positions,
            reference_mask=reference_mask,
            reference_charges=reference_charges,
            molecule_type_values=molecule_type_values,
            frame_atom_index_values=frame_atom_index_values,
            frame_mask_values=frame_mask_values,
            target_positions=target_positions,
            target_mask=target_mask,
        )

    def __call__(self, structure: StructureInput) -> Any:
        r"""
        Return a batched model input for one structure input.

        Args:
            structure: Parsed structure object or structure file path.

        Returns:
            Batched model-local input with leading batch size 1.

        Examples:
            >>> model_input = processor("complex.cif")  # doctest: +SKIP
        """

        structure = structure_from_input(structure)
        ccd = self._ccd if self._ccd is not None else default_ccd_provider()
        fields = self._structural_fields(structure, ccd)
        structure = fields.structure
        residues = fields.residues
        num_tokens = fields.num_tokens

        token_bond_matrix = _token_bonds(
            structure, fields.atom_to_token_values, fields.reference_space_values, num_tokens, ccd
        )

        return self.input_class(
            token_features=self._token_features(residues, fields.residue_names).unsqueeze(0),
            token_mask=fields.token_mask.unsqueeze(0),
            residue_index=torch.tensor([residue.residue_number for residue in residues], dtype=torch.long).unsqueeze(0),
            chain_index=torch.tensor(
                [fields.chain_to_index[residue.chain_id] for residue in residues], dtype=torch.long
            ).unsqueeze(0),
            atom_features=self._atom_features(structure, self._atom_names(structure)).unsqueeze(0),
            atom_to_token=fields.atom_to_token.unsqueeze(0),
            atom_mask=fields.atom_mask.unsqueeze(0),
            reference_positions=fields.reference_positions.unsqueeze(0),
            reference_mask=fields.reference_mask.unsqueeze(0),
            reference_charges=fields.reference_charges.unsqueeze(0),
            entity_index=torch.tensor(fields.entity_values, dtype=torch.long).unsqueeze(0),
            symmetry_index=torch.zeros(1, num_tokens, dtype=torch.long),
            token_index=torch.arange(num_tokens, dtype=torch.long).unsqueeze(0),
            reference_space_id=torch.tensor(fields.reference_space_values, dtype=torch.long).unsqueeze(0),
            representative_atom_mask=fields.representative_atom_mask.unsqueeze(0),
            atom_token_index=fields.atom_token_index.unsqueeze(0),
            token_to_rep_atom=fields.token_to_rep_atom.unsqueeze(0),
            molecule_type=torch.tensor(fields.molecule_type_values, dtype=torch.long).unsqueeze(0),
            frame_atom_index=torch.tensor(fields.frame_atom_index_values, dtype=torch.long)
            .reshape(num_tokens, 3)
            .unsqueeze(0),
            frame_mask=torch.tensor(fields.frame_mask_values, dtype=torch.bool).unsqueeze(0),
            token_bonds=token_bond_matrix.unsqueeze(0),
            target_positions=fields.target_positions.unsqueeze(0),
            target_mask=fields.target_mask.unsqueeze(0),
        )

    def source_structure(self, inputs: StructureInput) -> Structure | None:
        r"""
        Return the atom-ordered reference structure that this processor's predicted coordinates overlay onto.

        A structure processor emits one predicted coordinate per atom of the (optionally atom-materialized) input
        structure, in that atom order, so the reference for
        [`coords_to_structure`][ofoldx.data.structure.coords_to_structure]
        is exactly that structure -- materialized the same way ``__call__`` materializes it, so the atom counts
        line up. Inputs that are not a structure/path (e.g. a bare sequence) have no such reference and yield
        ``None``.

        Args:
            inputs: Parsed structure object or structure file path.

        Returns:
            The atom-ordered reference structure, or ``None`` for non-structure inputs.
        """

        try:
            structure = structure_from_input(inputs)
        except (OSError, ValueError, TypeError):
            return None  # a bare sequence / unreadable path is not a structure this processor can reference
        if self.materialize_missing_atoms:
            ccd = self._ccd if self._ccd is not None else default_ccd_provider()
            structure, _ = materialize_missing_atoms(structure, ccd)
        return structure

    def _atom_names(self, structure: Structure) -> list[str]:
        r"""
        Return the atom names used for the name-char feature (ligand-renamed when enabled).
        """

        if not self._ligand_atom_rename:
            return [atom.name for atom in structure.atoms]
        return _renamed_ligand_atom_names(structure)

    def _atom_features(self, structure: Structure, atom_names: list[str]) -> Tensor:
        r"""
        Build the per-atom feature matrix ``[flag(1), element one-hot, atom-name-char one-hots]``.

        When ``atom_feature_size`` matches the full layout, the element and name-character one-hots are
        both filled; otherwise only the leading flag + element block that fits is emitted.

        Args:
            structure: Parsed structure supplying atom elements.
            atom_names: Atom names used for the name-character block (possibly ligand-renamed).

        Returns:
            Atom feature matrix of shape ``[num_atoms, atom_feature_size]``.
        """

        expected_size = 1 + self.element_channels + self.atom_name_characters_count * self.atom_name_character_channels
        atoms = structure.atoms
        num_atoms = len(atoms)
        if self.atom_feature_size == expected_size:
            features = torch.zeros(num_atoms, expected_size)
            features[:, 0] = 1
            if num_atoms:
                atom_indices = torch.arange(num_atoms)
                element_indices = torch.tensor(
                    [self._element_index(atom.element) for atom in atoms],
                    dtype=torch.long,
                )
                features[atom_indices, 1 + element_indices] = 1
                character_indices = torch.tensor(
                    [
                        [
                            atom_name_char_index(character, self.atom_name_character_channels)
                            for character in name[: self.atom_name_characters_count].ljust(
                                self.atom_name_characters_count
                            )
                        ]
                        for name in atom_names
                    ],
                    dtype=torch.long,
                )
                character_offset_start = 1 + self.element_channels
                character_offsets = character_offset_start + (
                    torch.arange(self.atom_name_characters_count) * self.atom_name_character_channels
                )
                features[atom_indices.unsqueeze(1), character_offsets.unsqueeze(0) + character_indices] = 1
            return features

        features = torch.zeros(num_atoms, self.atom_feature_size)
        features[:, 0] = 1
        feature_indices = [1 + self._element_index(atom.element) for atom in atoms]
        valid_indices = [
            index for index, feature_index in enumerate(feature_indices) if feature_index < self.atom_feature_size
        ]
        if valid_indices:
            features[
                torch.tensor(valid_indices, dtype=torch.long),
                torch.tensor([feature_indices[index] for index in valid_indices], dtype=torch.long),
            ] = 1
        return features

    def _element_index(self, element: str) -> int:
        r"""
        Return the vocabulary index for an element, falling back to the unknown-element class.

        Args:
            element: Element symbol (case-insensitive).

        Returns:
            Element one-hot index.
        """

        return _lookup(element.upper(), self._element_indices, self._element_unknown_index)

    def _residue_index(self, residue_name: str) -> int:
        r"""
        Return the vocabulary index for a residue name, falling back to the ``UNK`` class.

        Args:
            residue_name: Residue name to look up.

        Returns:
            Residue-type one-hot index.
        """

        return _lookup(residue_name, self._residue_indices, self._residue_unknown_index)

    def _token_features(self, residues: tuple[ResidueKey, ...], residue_names: dict[ResidueKey, str]) -> Tensor:
        r"""
        Build the per-token feature matrix, filling only the residue-type block of the token tail.

        Args:
            residues: Residue key per token.
            residue_names: Residue name keyed by residue key.

        Returns:
            Token feature matrix of shape ``[num_tokens, token_feature_size]``.
        """

        features = torch.zeros(len(residues), self.token_feature_size)
        tail_size = max(self.token_feature_size - self.single_size, 0)
        if tail_size == 65:
            # Token tail = restype(32) + profile(32) + deletion_mean(1); only restype is filled
            # here (profile/deletion_mean are MSA-derived, deferred to the MSA milestone).
            residue_type_indices = torch.tensor(
                [self._residue_index(residue_names[residue]) for residue in residues],
                dtype=torch.long,
            )
            features[torch.arange(len(residues)), self.single_size + residue_type_indices] = 1
        elif tail_size > 0:
            for residue_index, residue in enumerate(residues):
                feature_index = self.single_size + self._residue_index(residue_names[residue]) % tail_size
                features[residue_index, feature_index] = 1
        return features


def default_residue_vocabulary() -> tuple[str, ...]:
    r"""
    Return the default 32-class residue-type vocabulary in reference ``restype`` order.

    Index by position: amino acids 0-19, ``UNK`` 20, RNA ``A/G/C/U/N`` 21-25, DNA
    ``DA/DG/DC/DT/DN`` 26-30, gap 31 (``STD_RESIDUES_WITH_GAP``). Unknown residues map to ``UNK``.

    Returns:
        Residue vocabulary tuple indexed by emitted residue-type ids.

    Examples:
        >>> vocabulary = default_residue_vocabulary()
        >>> len(vocabulary), vocabulary[0], vocabulary[20], vocabulary[31]
        (32, 'ALA', 'UNK', 'GAP')
    """

    return (
        # 20 standard amino acids (0-19)
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
        "UNK",  # 20
        "A",
        "G",
        "C",
        "U",
        "N",  # RNA 21-25
        "DA",
        "DG",
        "DC",
        "DT",
        "DN",  # DNA 26-30
        "GAP",  # 31
    )


# Per-token molecule type used by model-local input contracts and per-atom loss weighting.
MOLECULE_PROTEIN, MOLECULE_RNA, MOLECULE_DNA, MOLECULE_LIGAND = 0, 1, 2, 3
# Entity-level ``mol_type`` strings (from the parser, e.g. mmCIF _entity_poly.type) -> molecule type.
_MOLECULE_TYPE_NAMES = {
    "protein": MOLECULE_PROTEIN,
    "rna": MOLECULE_RNA,
    "dna": MOLECULE_DNA,
    "ligand": MOLECULE_LIGAND,
}
# Inverse of ``_MOLECULE_TYPE_NAMES``: molecule-type id -> name, used to key the shared ``LEAVING_ATOMS``
# reference table (keyed by name in ``data.residue_constants``).
_MOLECULE_TYPE_ID_TO_NAME = {value: key for key, value in _MOLECULE_TYPE_NAMES.items()}

# Element symbols in atomic-number order (H=1 -> index 0, ...), so the ``ref_element``
# one-hot index is ``atomic_number - 1``.
ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
    "BH",
    "HS",
    "MT",
    "DS",
    "RG",
    "CN",
    "NH",
    "FL",
    "MC",
    "LV",
    "TS",
    "OG",
)


def default_element_vocabulary() -> tuple[str, ...]:
    r"""
    Return the atom element vocabulary in atomic-number order.

    Returns:
        Element vocabulary tuple where index ``atomic_number - 1`` names the element.

    Examples:
        >>> vocabulary = default_element_vocabulary()
        >>> vocabulary[0], vocabulary[5]
        ('H', 'C')
    """

    return ELEMENT_SYMBOLS


def _entity_indices(residues: tuple[ResidueKey, ...], residue_names: dict[ResidueKey, str]) -> list[int]:
    r"""
    Group chains by token sequence: chains with identical sequences share one entity id.

    Args:
        residues: Residue key per token.
        residue_names: Residue name keyed by residue key.

    Returns:
        Entity id per token.
    """

    chain_sequences: dict[str, list[str]] = {}
    for residue in residues:
        chain_sequences.setdefault(residue.chain_id, []).append(residue_names[residue])
    sequence_to_entity: dict[tuple[str, ...], int] = {}
    chain_to_entity: dict[str, int] = {}
    for chain_id, sequence in chain_sequences.items():
        chain_to_entity[chain_id] = sequence_to_entity.setdefault(tuple(sequence), len(sequence_to_entity))
    return [chain_to_entity[residue.chain_id] for residue in residues]


def _reference_features(
    structure: Structure,
    reference_space_values: list[int],
    ccd: CcdProvider,
    *,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Return per-atom reference conformer positions, validity mask, and formal charges.

    Each atom's coordinate is looked up in its component's reference conformer by atom name, then
    the conformer is mean-centered per reference-space group (per residue). When ``augment`` is set,
    each group is additionally given an independent random rigid transform (training
    regularization, ``(points + U(-1,1)^3) @ R^T``). Atoms whose component or name is unknown get a
    zeroed, masked-out reference.

    Args:
        structure: Parsed structure.
        reference_space_values: Reference-space group id per atom.
        ccd: CCD provider supplying each component's reference conformer.
        augment: If ``True``, apply an independent random rigid transform per group.
        rng: Random generator used when ``augment`` is set.

    Returns:
        Per-atom ``(reference_positions [N, 3], reference_mask [N], reference_charges [N, 1])`` tensors.
    """

    num_atoms = len(structure.atoms)
    coords = np.zeros((num_atoms, 3), dtype=np.float32)
    mask = np.zeros(num_atoms, dtype=bool)
    charge = np.zeros(num_atoms, dtype=np.float32)
    components: dict[str, ReferenceComponent | None] = {}
    for index, atom in enumerate(structure.atoms):
        if atom.residue_name not in components:
            components[atom.residue_name] = ccd.get(atom.residue_name)
        component = components[atom.residue_name]
        if component is None:
            continue
        atom_index = component.atom_map.get(atom.name)
        if atom_index is None or atom_index >= len(component.coords):
            continue
        coords[index] = component.coords[atom_index]
        mask[index] = bool(component.mask[atom_index])
        charge[index] = component.charge[atom_index]

    space = np.asarray(reference_space_values)
    for uid in np.unique(space):
        selection = space == uid
        valid = selection & mask  # center on valid atoms only; masked/unknown atoms are zero and must not bias it
        center = coords[valid].mean(axis=0) if valid.any() else np.zeros(3, dtype=np.float32)
        group = coords[selection] - center
        if augment:
            group = _augment_conformer(group, rng if rng is not None else np.random.default_rng())
        coords[selection] = group.astype(np.float32)

    return (
        torch.from_numpy(coords),
        torch.from_numpy(mask),
        torch.from_numpy(charge).unsqueeze(-1),
    )


def _augment_conformer(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""
    Apply one random rigid transform ``(points + U(-1,1)^3) @ R^T`` to reference coords.

    Args:
        points: `[N, 3]` reference-conformer coordinates for one group.
        rng: Random generator supplying the translation and rotation.

    Returns:
        The transformed `[N, 3]` coordinates.
    """

    from scipy.spatial.transform import Rotation

    translation = rng.uniform(-1.0, 1.0, size=3)
    rotation = Rotation.random(random_state=rng).as_matrix()
    return (points + translation) @ rotation.T


def _token_bonds(
    structure: Structure,
    atom_to_token_values: list[int],
    reference_space_values: list[int],
    num_tokens: int,
    ccd: CcdProvider,
    max_bond_length: float = 2.4,
) -> Tensor:
    r"""
    Return the symmetric ``[N_token, N_token]`` token-bond adjacency.

    Covers intra-residue CCD bonds plus explicit inter-residue covalent bonds (``structure.bonds``,
    distance-filtered to ``max_bond_length`` Angstrom), then drops the bonds that the token-bond schema treats
    as implicit polymer connectivity (``get_bond_features``): standard-standard, standard-to-modified
    (unstd polymer), and inter-residue modified-to-modified bonds. Bonds involving a ligand, and
    intra-residue modified-residue bonds, are kept. So a plain protein yields all zeros while ligands
    and covalent attachments contribute their bonds across atom tokens.

    Args:
        structure: Parsed structure supplying explicit bonds and coordinates.
        atom_to_token_values: Token index per atom.
        reference_space_values: Reference-space group id per atom.
        num_tokens: Total token count.
        ccd: CCD provider supplying intra-residue reference bonds.
        max_bond_length: Distance cutoff (Angstrom) for keeping explicit inter-residue bonds.

    Returns:
        Symmetric `[num_tokens, num_tokens]` binary token-bond adjacency.
    """

    atoms = structure.atoms
    is_standard = [atom.residue_name in _STANDARD_RESIDUE_NAMES for atom in atoms]
    is_polymer = [atom_molecule_type(atom) != MOLECULE_LIGAND for atom in atoms]
    matrix = torch.zeros(num_tokens, num_tokens, dtype=torch.long)

    def add_bond(index_a: int, index_b: int) -> None:
        r"""
        Add one retained covalent bond to the token adjacency.

        Args:
            index_a: First atom index.
            index_b: Second atom index.
        """

        standard_a, standard_b = is_standard[index_a], is_standard[index_b]
        modified_a = is_polymer[index_a] and not standard_a  # unstd polymer (e.g. MSE)
        modified_b = is_polymer[index_b] and not standard_b
        if standard_a and standard_b:
            return  # standard polymer connectivity is implicit in residue_index
        if (standard_a and modified_b) or (modified_a and standard_b):
            return  # standard-to-modified polymer link is still polymer connectivity
        if modified_a and modified_b and reference_space_values[index_a] != reference_space_values[index_b]:
            return  # inter-residue modified-to-modified polymer link
        token_a = atom_to_token_values[index_a]
        token_b = atom_to_token_values[index_b]
        matrix[token_a, token_b] = 1
        matrix[token_b, token_a] = 1

    # Intra-residue bonds from the CCD reference component.
    residue_atom_indices: dict[ResidueKey, dict[str, int]] = {}
    residue_names: dict[ResidueKey, str] = {}
    for index, atom in enumerate(atoms):
        residue_atom_indices.setdefault(atom.residue, {})[atom.name] = index
        residue_names.setdefault(atom.residue, atom.residue_name)
    components: dict[str, ReferenceComponent | None] = {}
    for residue, name_to_index in residue_atom_indices.items():
        name = residue_names[residue]
        if name not in components:
            components[name] = ccd.get(name)
        component = components[name]
        if component is None:
            continue
        for name_a, name_b, _bond_order in component.bonds:
            index_a = name_to_index.get(name_a)
            index_b = name_to_index.get(name_b)
            if index_a is not None and index_b is not None:
                add_bond(index_a, index_b)

    # Explicit inter-residue covalent bonds (struct_conn), kept only within the bond-length cutoff.
    for index_a, index_b in structure.bonds:
        if _atom_distance(atoms[index_a], atoms[index_b]) <= max_bond_length:
            add_bond(index_a, index_b)
    return matrix


def _atom_distance(atom_a: AtomRecord, atom_b: AtomRecord) -> float:
    r"""
    Return the Euclidean distance in Angstrom between two atom records.

    Args:
        atom_a: First atom record.
        atom_b: Second atom record.

    Returns:
        Distance between the two atoms.
    """

    return float(((atom_a.x - atom_b.x) ** 2 + (atom_a.y - atom_b.y) ** 2 + (atom_a.z - atom_b.z) ** 2) ** 0.5)


def materialize_missing_atoms(structure: Structure, ccd: CcdProvider) -> tuple[Structure, np.ndarray]:
    r"""
    Expand each present residue to its full expected heavy-atom set and return the resolved-atom mask.

    Present atoms keep their coords (resolved); expected-but-absent heavy atoms are synthesized
    with zero coords (unresolved). Hydrogens and internal leaving atoms (OXT/OP3) are excluded;
    terminal residues keep their leaving atom. Residues whose component is unknown, lacks element
    data, or whose present atoms mostly do not match the component are left untouched (all resolved),
    so the change degrades gracefully. Returns the expanded structure and the per-atom resolved mask.

    Args:
        structure: Parsed structure to expand.
        ccd: CCD provider supplying each residue's expected heavy-atom set.

    Returns:
        The expanded structure and a boolean per-atom resolved mask (``True`` = real experimental atom).

    Examples:
        >>> from ofoldx.data.ccd import default_ccd_provider  # doctest: +SKIP
        >>> expanded, resolved = materialize_missing_atoms(structure, default_ccd_provider())  # doctest: +SKIP
    """

    atoms = structure.atoms
    residue_order: list[ResidueKey] = []
    residue_atoms: dict[ResidueKey, list[int]] = {}
    for index, atom in enumerate(atoms):
        if atom.residue not in residue_atoms:
            residue_atoms[atom.residue] = []
            residue_order.append(atom.residue)
        residue_atoms[atom.residue].append(index)

    terminal: set[ResidueKey] = set()
    chain_residues: dict[str, list[ResidueKey]] = {}
    for residue in residue_order:
        chain_residues.setdefault(residue.chain_id, []).append(residue)
    for residues in chain_residues.values():
        terminal.add(residues[0])
        terminal.add(residues[-1])

    expanded: list[AtomRecord] = []
    resolved: list[bool] = []
    index_map: dict[int, int] = {}

    def emit_present(old_index: int) -> None:
        r"""
        Copy one resolved input atom into the expanded atom list.

        Args:
            old_index: Original atom index.
        """

        index_map[old_index] = len(expanded)
        expanded.append(atoms[old_index])
        resolved.append(True)

    for residue in residue_order:
        present_indices = residue_atoms[residue]
        sibling = atoms[present_indices[0]]
        component = ccd.get(sibling.residue_name)
        present_by_name = {atoms[index].name: index for index in present_indices}
        if component is None or not component.elements:
            for index in present_indices:
                emit_present(index)
            continue

        heavy_names = [
            name
            for name in sorted(component.atom_map, key=lambda name: component.atom_map[name])
            if component.elements[component.atom_map[name]].upper() not in HYDROGEN_ELEMENTS
        ]
        heavy_set = set(heavy_names)
        matched = sum(1 for name in present_by_name if name in heavy_set)
        if matched * 2 < len(present_by_name):  # present atoms mostly do not match this component
            for index in present_indices:
                emit_present(index)
            continue

        leaving = LEAVING_ATOMS.get(_MOLECULE_TYPE_ID_TO_NAME[atom_molecule_type(sibling)], frozenset())
        keep_leaving = residue in terminal
        expected = [name for name in heavy_names if keep_leaving or name not in leaving]
        emitted: set[int] = set()
        for name in expected:
            if name in present_by_name:
                emitted.add(present_by_name[name])
                emit_present(present_by_name[name])
            else:
                atom_index = component.atom_map[name]
                expanded.append(
                    AtomRecord(
                        name=name,
                        element=component.elements[atom_index].upper(),
                        residue_name=sibling.residue_name,
                        residue=residue,
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        charge=float(component.charge[atom_index]) if atom_index < len(component.charge) else 0.0,
                        hetero=sibling.hetero,
                        mol_type=sibling.mol_type,
                    )
                )
                resolved.append(False)
        # Keep any present atom not already emitted (alt-named, or an excluded leaving atom).
        for index in present_indices:
            if index not in emitted:
                emit_present(index)

    bonds = tuple(
        (index_map[first], index_map[second])
        for first, second in structure.bonds
        if first in index_map and second in index_map
    )
    return Structure(tuple(expanded), bonds=bonds), np.asarray(resolved, dtype=bool)


def _renamed_ligand_atom_names(structure: Structure) -> list[str]:
    r"""
    Rename ligand-residue atoms to ``ELEM+count`` for the atom-name feature.

    Only the ``ref_atom_name_chars`` feature uses these names; reference-conformer and bond lookups
    keep the original CCD atom names. Non-ligand residues are unchanged.

    Args:
        structure: Parsed structure whose ligand atom names are rewritten.

    Returns:
        Atom names in atom order, with ligand atoms renamed to ``ELEM+count``.
    """

    names = [atom.name for atom in structure.atoms]
    residue_atoms: dict[ResidueKey, list[int]] = {}
    for index, atom in enumerate(structure.atoms):
        residue_atoms.setdefault(atom.residue, []).append(index)
    for indices in residue_atoms.values():
        if atom_molecule_type(structure.atoms[indices[0]]) != MOLECULE_LIGAND:
            continue
        counts: dict[str, int] = {}
        for index in indices:
            element = structure.atoms[index].element.upper()
            counts[element] = counts.get(element, 0) + 1
            names[index] = f"{element}{counts[element]}"
    return names


def atom_molecule_type(atom: AtomRecord) -> int:
    r"""
    Per-atom molecule type, preferring the parser's entity ``mol_type`` over the residue name.

    The entity ``mol_type`` (e.g. from mmCIF ``_entity_poly.type``) is what distinguishes a modified
    polymer residue (classified protein/rna/dna) from a true ligand; without it (e.g. plain PDB) the
    residue name is used.

    Args:
        atom: Atom record whose molecule type is classified.

    Returns:
        Molecule type id (``0`` protein, ``1`` RNA, ``2`` DNA, ``3`` ligand).

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey
        >>> atom = AtomRecord(name="CA", element="C", residue_name="ALA",
        ...                   residue=ResidueKey("A", 1, ""), x=0.0, y=0.0, z=0.0)
        >>> atom_molecule_type(atom)
        0
    """

    if atom.mol_type is not None:
        return _MOLECULE_TYPE_NAMES.get(atom.mol_type, MOLECULE_LIGAND)
    return _molecule_type(atom.residue_name)


def _molecule_type(residue_name: str) -> int:
    r"""
    Classify a token's molecule type by residue name.

    Args:
        residue_name: Residue name to classify.

    Returns:
        Molecule type id (``0`` protein, ``1`` RNA, ``2`` DNA, ``3`` ligand).
    """

    if residue_name in _PROTEIN_RESIDUE_NAMES:
        return MOLECULE_PROTEIN
    if residue_name in _RNA_RESIDUE_NAMES:
        return MOLECULE_RNA
    if residue_name in _DNA_RESIDUE_NAMES:
        return MOLECULE_DNA
    return MOLECULE_LIGAND


def _token_frames(
    structure: Structure,
    tokens: tuple[StructureToken, ...],
    reference_positions: Tensor,
    reference_mask: Tensor,
    reference_space_values: list[int],
) -> tuple[list[list[int]], list[bool]]:
    r"""
    Return per-token frame atom indices and a validity mask.

    Standard polymer residues use a backbone frame: protein ``(N, CA, C)`` and nucleic-acid
    ``(C1', C3', C4')``. Every other token (ligands, ions, modified residues) builds its frame from
    the reference conformer: the token's centre atom is ``b`` and its two nearest reference-conformer
    neighbours (within the same residue) are ``a`` and ``c``. Tokens without a valid, non-colinear
    frame get a masked-out frame so the frame-based PAE loss skips them.

    Args:
        structure: Parsed structure.
        tokens: Structure tokens.
        reference_positions: `[num_atoms, 3]` reference-conformer coordinates.
        reference_mask: `[num_atoms]` reference-conformer validity mask.
        reference_space_values: Reference-space group id per atom.

    Returns:
        Per-token ``[a, b, c]`` frame atom indices and a per-token validity mask.
    """

    ref_pos = reference_positions.numpy()
    ref_mask = reference_mask.numpy()
    space = np.asarray(reference_space_values)
    group_atoms: dict[int, np.ndarray] = {uid: np.where(space == uid)[0] for uid in np.unique(space)}

    frame_atom_index: list[list[int]] = []
    frame_mask: list[bool] = []
    for token in tokens:
        residue_name = structure.atoms[token.atom_indices[0]].residue_name
        if residue_name in _STANDARD_RESIDUE_NAMES and len(token.atom_indices) > 1:
            names = {structure.atoms[atom_index].name: atom_index for atom_index in token.atom_indices}
            triple = _backbone_frame_indices(names)
            frame_atom_index.append(triple if triple is not None else [0, 0, 0])
            frame_mask.append(triple is not None)
        else:
            centre = token.representative_atom_index
            triple, valid = _ligand_frame_indices(centre, group_atoms[int(space[centre])], ref_pos, ref_mask)
            frame_atom_index.append(triple)
            frame_mask.append(valid)
    return frame_atom_index, frame_mask


def _backbone_frame_indices(atom_name_to_index: dict[str, int]) -> list[int] | None:
    r"""
    Return the protein/nucleic backbone frame indices, or ``None`` if incomplete.

    Args:
        atom_name_to_index: Atom name to atom index within the token.

    Returns:
        The `[a, b, c]` backbone frame indices, or ``None`` when the backbone atoms are missing.
    """

    for frame_atoms in (("N", "CA", "C"), ("C1'", "C3'", "C4'")):
        if all(name in atom_name_to_index for name in frame_atoms):
            return [atom_name_to_index[name] for name in frame_atoms]
    return None


def _ligand_frame_indices(
    centre: int,
    group: np.ndarray,
    ref_pos: np.ndarray,
    ref_mask: np.ndarray,
    colinear_degrees: float = 25.0,
) -> tuple[list[int], bool]:
    r"""
    Build a ligand/atom-token frame from the two nearest reference-conformer neighbours.

    Args:
        centre: Atom index of the token's centre atom (frame ``b``).
        group: Atom indices in the same reference-space group as the centre.
        ref_pos: `[num_atoms, 3]` reference-conformer coordinates.
        ref_mask: `[num_atoms]` reference-conformer validity mask.
        colinear_degrees: Reject frames whose ``a-b-c`` angle is within this many degrees of 0 or 180.

    Returns:
        The `[a, b, c]` frame indices and whether the frame is valid (three masked-in, non-colinear atoms).
    """

    neighbours = group[group != centre]
    if neighbours.shape[0] < 2:
        return [0, 0, 0], False  # fewer than three atoms in the residue: no frame
    distances = np.linalg.norm(ref_pos[neighbours] - ref_pos[centre], axis=-1)
    nearest = neighbours[np.argsort(distances, kind="stable")]
    a_index, c_index = int(nearest[0]), int(nearest[1])
    frame = [a_index, int(centre), c_index]
    if not all(ref_mask[index] for index in frame):
        return frame, False
    first = ref_pos[a_index] - ref_pos[centre]
    second = ref_pos[c_index] - ref_pos[centre]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return frame, False
    cosine = float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    if angle <= colinear_degrees or angle >= 180.0 - colinear_degrees:
        return frame, False
    return frame, True


def _lookup(value: str, vocabulary: dict[str, int], default: int = 0) -> int:
    r"""
    Return a vocabulary index for a value, or the default when it is absent.

    Args:
        value: Key to look up.
        vocabulary: Value-to-index mapping.
        default: Index returned when the value is not present.

    Returns:
        The looked-up index or the default.
    """

    return vocabulary.get(value, default)


def atom_name_char_index(character: str, channels: int) -> int:
    r"""
    Encode one atom-name character as ``clip(ord(c) - 32, 0, channels - 1)``.

    Args:
        character: Single atom-name character to encode.
        channels: Number of one-hot channels (the encoded index is clipped to this range).

    Returns:
        The channel index for the character.

    Examples:
        >>> atom_name_char_index("C", 64), atom_name_char_index(" ", 64)
        (35, 0)
    """

    return min(max(ord(character) - 32, 0), channels - 1)
