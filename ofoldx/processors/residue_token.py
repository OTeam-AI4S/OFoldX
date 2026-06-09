r"""
Residue-token single-sequence structure processing.

[`ResidueTokenStructureProcessor`][ofoldx.processors.residue_token.ResidueTokenStructureProcessor] builds shared
structural fields through [`StructuralFields`][ofoldx.processors.utils.StructuralFields], then applies the
residue-token input contract: atomic-number element index, residue-token vocabulary, atom-feature concat, chain-local
residue index, distogram representative atom, bond-type matrix, and single-sequence / contact / method-conditioning
defaults.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self

from ..data.ccd import CcdProvider, ReferenceComponent, default_ccd_provider
from ..data.residue_constants import LEAVING_ATOMS
from ..data.structure import AtomRecord, ResidueKey, Structure
from .structure import (
    _MOLECULE_TYPE_ID_TO_NAME,
    ELEMENT_SYMBOLS,
    MOLECULE_DNA,
    MOLECULE_RNA,
    StructureProcessor,
    atom_molecule_type,
    atom_name_char_index,
)
from .utils import _STANDARD_RESIDUE_NAMES, StructuralFields, StructureInput, structure_from_input

# Residue-token vocabulary (reference ``boltz.data.const.tokens``): index 0=``<pad>``, 1=``-`` (gap),
# 2-22 the 20 canonical amino acids + ``UNK``, 23-27 RNA ``A G C U N``, 28-32 DNA ``DA DG DC DT DN``.
# Load-bearing for the loaded checkpoint's ``residue_type_projection``.
_RESIDUE_TOKEN_VOCABULARY: tuple[str, ...] = (
    "<pad>",
    "-",
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
    "A",
    "G",
    "C",
    "U",
    "N",
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
)
_RESIDUE_TOKEN_IDS = {token: index for index, token in enumerate(_RESIDUE_TOKEN_VOCABULARY)}
_NUM_RESIDUE_TOKENS = len(_RESIDUE_TOKEN_VOCABULARY)

_RNA_RESIDUE_NAMES = frozenset({"A", "G", "C", "U"})
_DNA_RESIDUE_NAMES = frozenset({"DA", "DG", "DC", "DT"})
# Unknown-residue fallbacks per molecule class (reference ``unk_token``): UNK/N/DN.
_UNK_PROTEIN_TOKEN = _RESIDUE_TOKEN_IDS["UNK"]
_UNK_RNA_TOKEN = _RESIDUE_TOKEN_IDS["N"]
_UNK_DNA_TOKEN = _RESIDUE_TOKEN_IDS["DN"]

# Reference ``boltz.data.const.bond_types`` order; ``type_bonds`` stores ``bond_type_id + 1`` so that
# class 0 is reserved for "no bond" (a real, zero-initialised, embedding row added to every unbonded pair).
_BOND_TYPE_IDS = {
    "OTHER": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
    "COVALENT": 5,
}
_BOND_TYPE_OTHER = _BOND_TYPE_IDS["OTHER"]
_BOND_TYPE_COVALENT = _BOND_TYPE_IDS["COVALENT"]
_CCD_BOND_TYPE_IDS = {
    "SINGLE": _BOND_TYPE_IDS["SINGLE"],
    "DOUBLE": _BOND_TYPE_IDS["DOUBLE"],
    "TRIPLE": _BOND_TYPE_IDS["TRIPLE"],
    "AROMATIC": _BOND_TYPE_IDS["AROMATIC"],
}

# Distogram representative atom (reference ``boltz.data.const.res_to_disto_atom``): CB for standard amino
# acids, CA for GLY, and the base atom for nucleotides. Other residues fall back to the center/CA atom.
_DISTO_ATOM_NAME = {
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "UNK": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'",
}

# Reference ``boltz.data.const.method_types_ids`` (lowercased keys); the experimental method maps to a
# per-token constant. Default to "x-ray diffraction" (id 1) when the method is absent / unrecognised.
_METHOD_TYPE_IDS = {
    "md": 0,
    "x-ray diffraction": 1,
    "electron microscopy": 2,
    "solution nmr": 3,
    "solid-state nmr": 4,
    "neutron diffraction": 4,
    "electron crystallography": 4,
    "fiber diffraction": 4,
    "powder diffraction": 4,
    "infrared spectroscopy": 4,
    "fluorescence transfer": 4,
    "epr": 4,
    "theoretical model": 4,
    "solution scattering": 4,
    "other": 4,
    "afdb": 5,
    "boltz-1": 6,
}
_DEFAULT_METHOD = "x-ray diffraction"

_NUM_ELEMENTS = 128  # reference ``const.num_elements`` (atomic number used directly as the index)
_NUM_CONTACT_CLASSES = 5  # reference ``const.contact_conditioning_info`` (UNSPECIFIED=0 default)


class ResidueTokenStructureProcessor(StructureProcessor):
    r"""
    Build residue-token single-sequence input tensors from structures.

    Composes the shared structural scaffold and overrides only model-specific encodings, so a loaded checkpoint
    sees the exact feature layout it expects. Full MSA, templates, and the binding/secondary-structure design
    conditioning are omitted (single-sequence inference): ``msa_profile`` equals the residue-type one-hot,
    ``deletion_mean`` is zero, and the contact conditioning is the all-``UNSPECIFIED`` default.

    Args:
        ccd: Optional CCD provider override. Processors with custom providers cannot be saved.
        reference_conformer_augment: If ``True``, randomly rigid-transform reference conformers.
        augment_seed: Optional seed for reference-conformer augmentation.
        materialize_missing_atoms: If ``True``, synthesize expected-but-unresolved heavy atoms.
        method: Method-conditioning label to encode.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/boltz1")  # doctest: +SKIP
        >>> model_input = processor("complex.cif")  # doctest: +SKIP
    """

    input_class: type[Any]

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> Self:
        r"""
        Build a processor for this contract; checkpoint dimensions are fixed by the contract.

        Args:
            config: Model config. Dimensions are schema-fixed and are not read from this object.
            **kwargs: Processor constructor overrides.

        Returns:
            Configured processor instance.

        Examples:
            >>> processor = ResidueTokenStructureProcessor.from_config(config)  # doctest: +SKIP
        """

        return cls(**kwargs)

    def __init__(
        self,
        *,
        ccd: CcdProvider | None = None,
        reference_conformer_augment: bool = False,
        augment_seed: int | None = None,
        materialize_missing_atoms: bool = False,
        method: str = _DEFAULT_METHOD,
    ) -> None:
        self._method = method
        super().__init__(
            ccd=ccd,
            reference_conformer_augment=reference_conformer_augment,
            augment_seed=augment_seed,
            materialize_missing_atoms=materialize_missing_atoms,
        )
        self._method_id = _METHOD_TYPE_IDS.get(method.lower(), _METHOD_TYPE_IDS[_DEFAULT_METHOD])

    def _processor_config(self) -> dict[str, object]:
        if self._ccd is not None:
            raise ValueError("processors with custom CCD providers cannot be saved")
        return {
            "reference_conformer_augment": self._reference_conformer_augment,
            "augment_seed": self._augment_seed,
            "materialize_missing_atoms": self.materialize_missing_atoms,
            "method": self._method,
        }

    def __call__(self, structure: StructureInput) -> Any:
        r"""
        Return a batched model-local input for one structure input.

        Args:
            structure: Parsed structure object or structure file path.

        Returns:
            Batched model-local input with leading batch size 1.

        Examples:
            >>> model_input = processor("complex.cif")  # doctest: +SKIP
        """

        values, _ = self._process_structure(structure_from_input(structure))
        return self.input_class(**values)

    def _process_structure(self, structure: Structure) -> tuple[dict[str, Any], StructuralFields]:
        r"""
        Return shared input fields together with the structural fields used to build them.

        Args:
            structure: Parsed structure to featurize.

        Returns:
            The model input field mapping and the shared structural fields it was built from.
        """

        ccd = self._ccd if self._ccd is not None else default_ccd_provider()
        # The reference tokenizer emits standard polymer residues with their canonical CCD atom set, which
        # excludes the terminal leaving atoms (protein OXT, nucleic OP3); drop them so the atom set matches.
        structure = _drop_terminal_leaving_atoms(structure)
        fields = self._structural_fields(structure, ccd)
        structure = fields.structure
        residues = fields.residues
        num_tokens = fields.num_tokens

        # ``reference_positions`` and the ``ref_pos`` slice of ``atom_features`` must be byte-identical, so
        # the centered reference coords are built once and sliced into both.
        reference_positions = fields.reference_positions
        atom_features = self._atom_reference_features(structure, reference_positions, fields.reference_charges)

        residue_type = self._residue_type_onehot(residues, fields.residue_names)
        token_bonds, type_bonds = self._token_bonds(structure, fields.atom_to_token_values, num_tokens, ccd)
        contact_conditioning = F.one_hot(
            torch.zeros(num_tokens, num_tokens, dtype=torch.long), _NUM_CONTACT_CLASSES
        ).float()

        return (
            {
                "token_mask": fields.token_mask.unsqueeze(0),
                "residue_index": self._residue_index_chain_local(residues).unsqueeze(0),
                "chain_index": torch.tensor(
                    [fields.chain_to_index[residue.chain_id] for residue in residues], dtype=torch.long
                ).unsqueeze(0),
                "entity_index": torch.tensor(fields.entity_values, dtype=torch.long).unsqueeze(0),
                "symmetry_index": torch.zeros(1, num_tokens, dtype=torch.long),
                "token_index": torch.arange(num_tokens, dtype=torch.long).unsqueeze(0),
                "atom_features": atom_features.unsqueeze(0),
                "atom_to_token": fields.atom_to_token.unsqueeze(0),
                "atom_mask": fields.atom_mask.unsqueeze(0),
                "reference_positions": reference_positions.unsqueeze(0),
                "reference_space_id": torch.tensor(fields.reference_space_values, dtype=torch.long).unsqueeze(0),
                "representative_atom_mask": self._representative_atom_mask_disto(structure, fields).unsqueeze(0),
                "atom_token_index": fields.atom_token_index.unsqueeze(0),
                "token_to_rep_atom": self._token_to_rep_atom_disto(structure, fields).unsqueeze(0),
                "molecule_type": torch.tensor(fields.molecule_type_values, dtype=torch.long).unsqueeze(0),
                "chain_type": self._chain_type(fields.molecule_type_values).unsqueeze(0),
                "frame_atom_index": torch.tensor(fields.frame_atom_index_values, dtype=torch.long)
                .reshape(num_tokens, 3)
                .unsqueeze(0),
                "frame_mask": torch.tensor(fields.frame_mask_values, dtype=torch.bool).unsqueeze(0),
                "target_positions": fields.target_positions.unsqueeze(0),
                "target_mask": fields.target_mask.unsqueeze(0),
                "residue_type": residue_type.unsqueeze(0),
                # Single-sequence MSA closed forms: the dummy single-row MSA equals the query, so the per-token
                # profile is the residue-type one-hot and there are no deletions.
                "msa_profile": residue_type.unsqueeze(0),
                "deletion_mean": torch.zeros(1, num_tokens),
                "token_bonds": token_bonds.unsqueeze(0),
                "type_bonds": type_bonds.unsqueeze(0),
                "contact_conditioning": contact_conditioning.unsqueeze(0),
                "contact_threshold": torch.zeros(1, num_tokens, num_tokens),
                "method_feature": torch.full((1, num_tokens), self._method_id, dtype=torch.long),
                "modified": torch.zeros(1, num_tokens, dtype=torch.long),
                "cyclic_period": torch.zeros(1, num_tokens),
            },
            fields,
        )

    def _atom_reference_features(
        self, structure: Structure, reference_positions: Tensor, reference_charges: Tensor
    ) -> Tensor:
        r"""
        Build the 388-d atom feature ``[ref_pos(3), ref_charge(1), ref_element(128), name_chars(256)]``.

        ``reference_positions`` and ``reference_charges`` come from the shared reference-conformer geometry,
        so the ``ref_pos`` slice is byte-identical to the standalone ``reference_positions`` field.

        Args:
            structure: Parsed structure supplying atom elements and names.
            reference_positions: `[num_atoms, 3]` centered reference-conformer coordinates.
            reference_charges: `[num_atoms, 1]` reference formal charges.

        Returns:
            Atom feature matrix of shape `[num_atoms, 388]`.
        """

        atoms = structure.atoms
        num_atoms = len(atoms)
        element_indices = torch.tensor([self._element_index(atom.element) for atom in atoms], dtype=torch.long)
        element_onehot = F.one_hot(element_indices, _NUM_ELEMENTS).float()
        char_channels = self.atom_name_character_channels
        char_count = self.atom_name_characters_count
        character_indices = torch.tensor(
            [
                [
                    atom_name_char_index(character, char_channels)
                    for character in atom.name[:char_count].ljust(char_count)
                ]
                for atom in atoms
            ],
            dtype=torch.long,
        ).reshape(num_atoms, char_count)
        name_chars = F.one_hot(character_indices, char_channels).float().reshape(num_atoms, char_count * char_channels)
        return torch.cat(
            [reference_positions, reference_charges.to(reference_positions.dtype), element_onehot, name_chars],
            dim=-1,
        )

    @staticmethod
    def _element_index(element: str) -> int:
        r"""
        Return the element index = atomic number (one-hot over 128; class 0 reserved, H=1, C=6).

        Args:
            element: Element symbol (case-insensitive); unknown elements map to the reserved class 0.

        Returns:
            Element one-hot index.

        Examples:
            >>> ResidueTokenStructureProcessor._element_index("C")
            6
            >>> ResidueTokenStructureProcessor._element_index("XX")
            0
        """

        symbol = element.upper()
        try:
            atomic_number = _ELEMENT_TO_ATOMIC_NUMBER[symbol]
        except KeyError:
            return 0  # unknown element -> the reserved empty class
        return min(atomic_number, _NUM_ELEMENTS - 1)

    @staticmethod
    def _residue_type_onehot(residues: tuple[ResidueKey, ...], residue_names: dict[ResidueKey, str]) -> Tensor:
        r"""
        Return the ``[N, 33]`` one-hot over the residue-token vocabulary (UNK/N/DN unknown fallbacks).

        Args:
            residues: Residue key per token.
            residue_names: Residue name keyed by residue key.

        Returns:
            Residue-type one-hot of shape `[num_tokens, 33]`.
        """

        indices = torch.tensor([_residue_token_index(residue_names[residue]) for residue in residues], dtype=torch.long)
        return F.one_hot(indices, _NUM_RESIDUE_TOKENS).float()

    @staticmethod
    def _residue_index_chain_local(residues: tuple[ResidueKey, ...]) -> Tensor:
        r"""
        Return the chain-local 0-based residue index (reference ``enumerate(sequence)`` per chain).

        Args:
            residues: Residue key per token, in token order.

        Returns:
            Chain-local residue index per token.

        Examples:
            >>> from ofoldx.data.structure import ResidueKey
            >>> residues = (ResidueKey("A", 5, ""), ResidueKey("A", 6, ""), ResidueKey("B", 1, ""))
            >>> ResidueTokenStructureProcessor._residue_index_chain_local(residues).tolist()
            [0, 1, 0]
        """

        counters: dict[str, int] = {}
        indices: list[int] = []
        for residue in residues:
            index = counters.get(residue.chain_id, 0)
            indices.append(index)
            counters[residue.chain_id] = index + 1
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _chain_type(molecule_type_values: list[int]) -> Tensor:
        r"""
        Remap structural molecule types (PROTEIN0/RNA1/DNA2/LIGAND3) to model chain-type ids
        (PROTEIN0/DNA1/RNA2/NONPOLYMER3), i.e. swap RNA<->DNA (1<->2).

        Args:
            molecule_type_values: Structural molecule type id per token.

        Returns:
            Model chain-type id per token.

        Examples:
            >>> ResidueTokenStructureProcessor._chain_type([0, 1, 2, 3]).tolist()
            [0, 2, 1, 3]
        """

        remap = {MOLECULE_RNA: 2, MOLECULE_DNA: 1}
        return torch.tensor([remap.get(value, value) for value in molecule_type_values], dtype=torch.long)

    def _disto_atom_indices(self, structure: Structure, fields: StructuralFields) -> list[int]:
        r"""
        Return the per-token global atom index of the distogram representative atom (disto/CB).

        Args:
            structure: Parsed structure.
            fields: Shared structural fields providing the tokens.

        Returns:
            Distogram representative atom index per token.
        """

        atoms = structure.atoms
        rep_indices: list[int] = []
        for token in fields.tokens:
            residue_name = atoms[token.atom_indices[0]].residue_name
            disto_name = _DISTO_ATOM_NAME.get(residue_name)
            chosen = token.representative_atom_index
            if disto_name is not None:
                for atom_index in token.atom_indices:
                    if atoms[atom_index].name == disto_name:
                        chosen = atom_index
                        break
            rep_indices.append(chosen)
        return rep_indices

    def _token_to_rep_atom_disto(self, structure: Structure, fields: StructuralFields) -> Tensor:
        r"""
        Return the per-token distogram representative atom index as a tensor.

        Args:
            structure: Parsed structure.
            fields: Shared structural fields.

        Returns:
            Distogram representative atom index per token, shape `[num_tokens]`.
        """

        return torch.tensor(self._disto_atom_indices(structure, fields), dtype=torch.long)

    def _representative_atom_mask_disto(self, structure: Structure, fields: StructuralFields) -> Tensor:
        r"""
        Return the boolean atom mask selecting each token's distogram representative atom.

        Args:
            structure: Parsed structure.
            fields: Shared structural fields.

        Returns:
            Boolean mask of shape `[num_atoms]`.
        """

        mask = torch.zeros(fields.num_atoms, dtype=torch.bool)
        for atom_index in self._disto_atom_indices(structure, fields):
            mask[atom_index] = True
        return mask

    def _token_bonds(
        self, structure: Structure, atom_to_token_values: list[int], num_tokens: int, ccd: CcdProvider
    ) -> tuple[Tensor, Tensor]:
        r"""
        Return the symmetric ``token_bonds`` (binary ``[N, N]`` float) and ``type_bonds`` (``[N, N]`` long).

        Intra-residue CCD bonds are typed by their bond order (``OTHER`` fallback) and explicit
        ``struct_conn`` covalent links are typed ``COVALENT``; ``type_bonds`` stores ``bond_type_id + 1``
        (class 0 = no bond). The implicit polymer backbone — inter-residue bonds between two standard
        polymer residues, which the reference parser never stores in ``struct.bonds`` — is dropped, so a
        plain protein yields all zeros for both matrices (the relative-position encoding carries chain
        connectivity instead).

        Args:
            structure: Parsed structure supplying explicit bonds.
            atom_to_token_values: Token index per atom.
            num_tokens: Total token count.
            ccd: CCD provider supplying intra-residue reference bonds.

        Returns:
            The binary `token_bonds` `[N, N]` float matrix and the typed `type_bonds` `[N, N]` long matrix.
        """

        token_bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float32)
        type_bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        atoms = structure.atoms
        is_standard = [not atom.hetero and atom.residue_name in _STANDARD_RESIDUE_NAMES for atom in atoms]

        def add_bond(index_a: int, index_b: int, bond_type_id: int) -> None:
            r"""
            Add one typed inter-token bond to the output matrices.

            Args:
                index_a: First atom index.
                index_b: Second atom index.
                bond_type_id: Zero-based bond type id.
            """

            token_a = atom_to_token_values[index_a]
            token_b = atom_to_token_values[index_b]
            if token_a == token_b:
                return  # intra-token bonds are not token-level edges
            token_bonds[token_a, token_b] = token_bonds[token_b, token_a] = 1.0
            type_class = bond_type_id + 1
            type_bonds[token_a, token_b] = type_bonds[token_b, token_a] = type_class

        # Intra-residue bonds from the CCD reference component (typed by their bond order).
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
            for name_a, name_b, bond_order in component.bonds:
                bond_type_id = _CCD_BOND_TYPE_IDS.get(bond_order, _BOND_TYPE_OTHER)
                index_a = name_to_index.get(name_a)
                index_b = name_to_index.get(name_b)
                if index_a is not None and index_b is not None:
                    add_bond(index_a, index_b, bond_type_id)

        # Explicit inter-residue covalent bonds (struct_conn) — typed COVALENT, no distance filter. The
        # implicit polymer backbone (both endpoints standard polymer residues) is connectivity, not a
        # struct_conn record, so it is dropped to match the reference ``struct.bonds`` contents.
        for index_a, index_b in structure.bonds:
            if is_standard[index_a] and is_standard[index_b]:
                continue
            add_bond(index_a, index_b, _BOND_TYPE_COVALENT)
        return token_bonds, type_bonds


def _drop_terminal_leaving_atoms(structure: Structure) -> Structure:
    r"""
    Return the structure with terminal leaving atoms (protein OXT, nucleic OP3) of standard polymer
    residues removed, matching the reference tokenizer's canonical-atom-set tokenization. Ligand / modified
    residues and the explicit bond graph (reindexed) are preserved; a no-op when no such atom is present.

    Args:
        structure: Parsed structure to filter.

    Returns:
        The structure with standard-polymer terminal leaving atoms dropped and bonds reindexed.
    """

    atoms = structure.atoms
    keep_mask: list[bool] = []
    for atom in atoms:
        leaving = LEAVING_ATOMS.get(_MOLECULE_TYPE_ID_TO_NAME[atom_molecule_type(atom)], frozenset())
        is_standard = not atom.hetero and atom.residue_name in _STANDARD_RESIDUE_NAMES
        keep_mask.append(not (is_standard and atom.name in leaving))
    if all(keep_mask):
        return structure
    index_map: dict[int, int] = {}
    kept: list[AtomRecord] = []
    for old_index, (atom, keep) in enumerate(zip(atoms, keep_mask)):
        if keep:
            index_map[old_index] = len(kept)
            kept.append(atom)
    bonds = tuple(
        (index_map[first], index_map[second])
        for first, second in structure.bonds
        if first in index_map and second in index_map
    )
    return Structure(tuple(kept), bonds=bonds)


def _residue_token_index(residue_name: str) -> int:
    r"""
    Map a residue name to its token index, with per-class unknown fallbacks (UNK/N/DN).

    Unknown protein residues fall back to ``UNK``, unknown RNA to ``N``, and unknown DNA to ``DN``.

    Args:
        residue_name: Residue name to map.

    Returns:
        The residue-token index.

    Examples:
        >>> _residue_token_index("ALA"), _residue_token_index("A"), _residue_token_index("DA")
        (2, 23, 28)
        >>> _residue_token_index("XYZ")
        22
    """

    index = _RESIDUE_TOKEN_IDS.get(residue_name)
    if index is not None:
        return index
    if residue_name in _RNA_RESIDUE_NAMES:
        return _UNK_RNA_TOKEN
    if residue_name in _DNA_RESIDUE_NAMES:
        return _UNK_DNA_TOKEN
    return _UNK_PROTEIN_TOKEN


_ELEMENT_TO_ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(ELEMENT_SYMBOLS, start=1)}
