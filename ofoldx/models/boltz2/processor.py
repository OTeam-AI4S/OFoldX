r"""
Input processing for Boltz-2 folding and affinity scoring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout, resolved_token_mask
from ...data.evidence import (
    MSA,
    MSA_GAP_INDEX,
    MSA_VOCAB_SIZE,
    MsaCollection,
    Template,
    msa_residue_index,
    resolve_msa_bindings,
)
from ...data.molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    component_boundary_atoms,
)
from ...data.structure import Structure
from ...data.tokenization import (
    TOKEN_OTHER_INDEX,
    TOKEN_VOCAB_SIZE,
    TokenLayout,
    component_token_index,
    tokenize_molecular_system,
)
from ...processor import Processor
from .model import (
    Boltz2Contact,
    Boltz2Input,
    Boltz2Pocket,
    Boltz2StructureTarget,
    Boltz2Template,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

_STANDARD_COMPONENTS = {
    "protein": frozenset(
        {
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
        }
    ),
    "rna": frozenset({"A", "G", "C", "U", "N"}),
    "dna": frozenset({"DA", "DG", "DC", "DT", "DN"}),
}
_STANDARD_BY_ENTITY_TYPE = {
    ENTITY_PROTEIN: _STANDARD_COMPONENTS["protein"],
    ENTITY_RNA: _STANDARD_COMPONENTS["rna"],
    ENTITY_DNA: _STANDARD_COMPONENTS["dna"],
    ENTITY_DNA_RNA_HYBRID: _STANDARD_COMPONENTS["rna"] | _STANDARD_COMPONENTS["dna"],
    ENTITY_PNA: frozenset(),
}
_CONSTRAINT_FREE_COMPONENTS = {
    "protein": _STANDARD_COMPONENTS["protein"] | {"UNK", "MSE"},
    "rna": _STANDARD_COMPONENTS["rna"],
    "dna": _STANDARD_COMPONENTS["dna"],
}
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_POLYMER_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
}
_MSA_GAP = MSA_GAP_INDEX
_METHOD_TYPES = {
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
    "future1": 7,
    "future2": 8,
    "future3": 9,
    "future4": 10,
    "future5": 11,
}
_STANDARD_LOCAL_SYMMETRY = {
    "ASP": (("OD1", "OD2"), ("OD2", "OD1")),
    "GLU": (("OE1", "OE2"), ("OE2", "OE1")),
    "PHE": (
        ("CD1", "CD2", "CE1", "CE2"),
        ("CD2", "CD1", "CE2", "CE1"),
    ),
    "TYR": (
        ("CD1", "CD2", "CE1", "CE2"),
        ("CD2", "CD1", "CE2", "CE1"),
    ),
    "A": (("OP1", "OP2"), ("OP2", "OP1")),
    "G": (("OP1", "OP2"), ("OP2", "OP1")),
    "C": (("OP1", "OP2"), ("OP2", "OP1")),
    "U": (("OP1", "OP2"), ("OP2", "OP1")),
    "DA": (("OP1", "OP2"), ("OP2", "OP1")),
    "DG": (("OP1", "OP2"), ("OP2", "OP1")),
    "DC": (("OP1", "OP2"), ("OP2", "OP1")),
    "DT": (("OP1", "OP2"), ("OP2", "OP1")),
}


_MAX_TOKENS = 256
_MAX_POLYMER_TOKENS = 200
_MAX_ATOMS = 2048
_CROP_NEIGHBORHOOD = 10
_CONTACT_UNSPECIFIED = 0
_CONTACT_UNSELECTED = 1
_CONTACT_POCKET_TO_BINDER = 2
_CONTACT_BINDER_TO_POCKET = 3
_CONTACT_PAIR = 4


class Boltz2AffinityProcessor(Processor):
    r"""
    Convert protein-ligand systems into `Boltz2Input` for affinity scoring.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure,
        *,
        ligand: str | None = None,
        msa: MSA | MsaCollection | None = None,
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "other",
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""
        Prepare affinity features for one protein-ligand system.

        Args:
            input: Molecular system or structure.
            ligand: Molecule instance scored by the affinity head. May be omitted for a system
                containing exactly one molecule instance.
            msa: Optional MSA value or collection.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Ligand-centered token, atom, MSA, and affinity tensors. Reference
            coordinates are in angstroms; canonical integer indices bind the
            family axes to ``system``.
        """

        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        if system.system_entity_ptr.numel() != 2:
            raise ValueError("Boltz-2 affinity Processor accepts one canonical molecular system")
        instance_entities = system.instance_entity.detach().cpu().tolist()
        entity_types = system.entity_type.detach().cpu().tolist()
        protein_instances = [
            instance_index
            for instance_index, entity_index in enumerate(instance_entities)
            if entity_types[entity_index] == ENTITY_PROTEIN
        ]
        if not protein_instances:
            raise ValueError("Boltz-2 affinity requires a protein receptor")
        molecule_instances = [
            instance_index
            for instance_index, entity_index in enumerate(instance_entities)
            if entity_types[entity_index] == ENTITY_MOLECULE
        ]
        if ligand is None:
            if len(molecule_instances) != 1:
                raise ValueError("ligand is required unless the system contains exactly one Molecule instance")
            ligand_instance_index = molecule_instances[0]
        else:
            ligand_instance_index = system.instance_index(ligand)
        ligand_entity_index = instance_entities[ligand_instance_index]
        if entity_types[ligand_entity_index] != ENTITY_MOLECULE:
            raise ValueError(f"ligand instance {ligand!r} does not reference a Molecule")
        if instance_entities.count(ligand_entity_index) != 1:
            raise ValueError("affinity scoring does not support copied ligand entities")
        ligand_atom_count, affinity_mw = _affinity_ligand_properties(
            system,
            ligand_instance_index,
        )
        if ligand_atom_count > 128:
            raise ValueError("the affinity head supports at most 128 ligand atoms after removing hydrogens")

        resolved_msa = () if msa is None else resolve_msa_bindings(msa, system)
        layout = _tokenize(system)
        return _prepare_boltz2_features(
            system,
            tokens=_lower_token_layout(
                system,
                layout,
                ligand_instance_index=ligand_instance_index,
            ),
            structure=structure,
            ligand_instance_index=ligand_instance_index,
            affinity_mw=affinity_mw,
            msa=msa,
            msa_bindings=resolved_msa,
            templates=(),
            design=(),
            self_template=(),
            contacts=contacts,
            pockets=pockets,
            method=method,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )


class Boltz2Processor(Processor):
    r"""
    Convert supported biomolecular inputs into `Boltz2Input`.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.

    Examples:
        >>> from ofoldx.data import AtomSite, MolecularSystem, Molecule
        >>> molecule = Molecule.from_smiles("[H]OCC", atom_ids=("HO", "O", "C1", "C2"))
        >>> system = MolecularSystem.from_entities({"ligand": molecule}, {"L": "ligand"})
        >>> bool(system.reference_coordinate_mask.any())
        False
        >>> input = Boltz2Processor()(system)
        >>> bool(input.reference_positions[0, :3].abs().sum() > 0)
        True
        >>> bool(system.reference_coordinate_mask.any())
        False
        >>> Boltz2Processor()(  # doctest: +ELLIPSIS
        ...     system,
        ...     contacts=(Boltz2Contact(AtomSite("L", 0, "HO"), AtomSite("L", 0, "C1")),),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Boltz-2 tokenizes only heavy atoms, so contact endpoint ... has no token: ...
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        design: Sequence[ComponentReference] = (),
        self_template: Sequence[ComponentReference] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        r"""
        Prepare folding features for one molecular system.

        Args:
            input: Molecular system or structure.
            msa: Optional MSA value or collection.
            templates: Structural protein templates with canonical residue alignments.
            design: Component occurrences designed by the calling protocol.
            self_template: Component occurrences whose input coordinates form one
                direct self-template.
            contacts: Learned pairwise contact conditions.
            pockets: Learned binder-to-pocket conditions.
            method: Experimental method used by the learned method embedding.
            generator: Optional generator for independent rigid transforms of reference
                conformers.

        Returns:
            Batched token, atom, and MSA tensors. Reference coordinates are in
            angstroms; canonical integer indices bind the family axes to
            ``system``.
        """

        return self._lower(
            input,
            msa=msa,
            templates=templates,
            design=design,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )

    def _lower(
        self,
        input: MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None,
        templates: Sequence[Template | Boltz2Template],
        design: Sequence[ComponentReference],
        self_template: Sequence[ComponentReference],
        contacts: Sequence[Boltz2Contact],
        pockets: Sequence[Boltz2Pocket],
        method: str,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2Input:
        if isinstance(input, Structure):
            structure = input
            system = input.system
        else:
            structure = None
            system = input
        if system.system_entity_ptr.numel() != 2:
            raise ValueError("Boltz-2 Processor accepts one canonical molecular system")
        if not system.instance_symbol.numel():
            raise ValueError("Boltz-2 requires at least one molecular instance")

        resolved_msa = () if msa is None else resolve_msa_bindings(msa, system)
        layout = _tokenize(system)
        if cropper is not None:
            anchor = resolved_token_mask(cast(Structure, input), layout)
            if design:
                design_anchor = torch.zeros_like(anchor)
                for reference in design:
                    design_anchor |= layout.token_component_index == system.component_index(reference)
                anchor &= design_anchor
            layout = crop_token_layout(
                cast(Structure, input),
                layout,
                cropper,
                required_token_mask=_conditioning_required_token_mask(
                    system,
                    layout,
                    self_template=self_template,
                    contacts=contacts,
                    pockets=pockets,
                ),
                anchor_token_mask=anchor,
                generator=generator,
            )
        return _prepare_boltz2_features(
            system,
            tokens=_lower_token_layout(
                system,
                layout,
                ligand_instance_index=None,
            ),
            structure=structure,
            ligand_instance_index=None,
            affinity_mw=None,
            msa=msa,
            msa_bindings=resolved_msa,
            templates=templates,
            design=design,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        design: Sequence[ComponentReference] = (),
        self_template: Sequence[ComponentReference] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Boltz2Input, Boltz2StructureTarget]:
        r"""
        Lower one complete structure into model input and separate truth.
        """

        processed = self._lower(
            input,
            msa=msa,
            templates=templates,
            design=design,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            cropper=cropper,
            generator=generator,
        )
        return processed, _prepare_boltz2_target(
            input,
            processed.atom_index,
            device=self.device,
            dtype=self.dtype,
        )


def _tokenize(
    system: MolecularSystem,
) -> TokenLayout:
    r"""Build the Boltz-2 view of the canonical structural token axis."""

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atom_mask = torch.ones_like(system.atomic_number, dtype=torch.bool)

    for instance_index, entity_index in enumerate(instance_entities):
        if int(entity_types[entity_index]) == ENTITY_MOLECULE:
            continue
        for component_index in range(
            int(instance_components[instance_index]),
            int(instance_components[instance_index + 1]),
        ):
            for atom_index in range(
                int(component_atoms[component_index]),
                int(component_atoms[component_index + 1]),
            ):
                if system.symbols[int(atom_symbols[atom_index])] in {"OXT", "OP3"}:
                    atom_mask[atom_index] = False

    return tokenize_molecular_system(
        system,
        atomize_nonstandard_polymers=False,
        atom_mask=atom_mask,
    )


def _lower_token_layout(
    system: MolecularSystem,
    layout: TokenLayout,
    *,
    ligand_instance_index: int | None,
) -> list[dict[str, object]]:
    r"""Lower the shared structural partition into Boltz-2 feature rows."""

    token_atom_ptr = layout.token_atom_ptr.detach().cpu().tolist()
    atom_indices = layout.atom_index.detach().cpu().tolist()
    token_component_indices = layout.token_component_index.detach().cpu().tolist()
    token_atom_indices = layout.token_atom_index.detach().cpu().tolist()
    token_instance_indices = layout.token_instance_index.detach().cpu().tolist()
    token_entity_indices = layout.token_entity_index.detach().cpu().tolist()
    token_residue_indices = layout.token_residue_index.detach().cpu().tolist()
    token_symmetry_indices = layout.token_symmetry_index.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    tokens: list[dict[str, object]] = []

    for token_index, component_index in enumerate(token_component_indices):
        start = int(token_atom_ptr[token_index])
        stop = int(token_atom_ptr[token_index + 1])
        group = tuple(int(atom_index) for atom_index in atom_indices[start:stop])
        atom_index = int(token_atom_indices[token_index])
        instance_index = int(token_instance_indices[token_index])
        entity_index = int(token_entity_indices[token_index])
        entity_type = int(entity_types[entity_index])
        component_id = system.symbols[int(component_symbols[component_index])]
        polymer_type = _component_polymer_type(entity_type, component_id)

        if atom_index >= 0:
            tokens.append(
                {
                    "atoms": group,
                    "atom_index": atom_index,
                    "component_index": int(component_index),
                    "instance_index": instance_index,
                    "monomer_index": None,
                    "residue_index": 0,
                    "chain_index": instance_index,
                    "entity_index": entity_index,
                    "symmetry_index": int(token_symmetry_indices[token_index]),
                    "molecule_type": 3,
                    "polymer_type": None,
                    "residue_type": int(layout.token_type_index[token_index]),
                    "modified": False,
                    "center": atom_index,
                    "distogram": atom_index,
                    "frame": (atom_index, atom_index, atom_index),
                    "affinity": instance_index == ligand_instance_index,
                }
            )
            continue

        standard = component_id in _STANDARD_BY_ENTITY_TYPE.get(entity_type, frozenset())
        by_name = {system.symbols[int(atom_symbols[index])]: index for index in group}
        center_name = "CA" if polymer_type == "protein" else "C1'"
        if polymer_type == "protein":
            distogram_name = "CA" if component_id == "GLY" else "CB"
            frame_names = ("N", "CA", "C")
        else:
            distogram_name = (
                "C4"
                if component_id in {"A", "G", "DA", "DG"}
                else "C2" if component_id in {"C", "U", "DC", "DT"} else "C1'"
            )
            frame_names = ("C1'", "C3'", "C4'")
        center = by_name.get(center_name, group[0])
        tokens.append(
            {
                "atoms": group,
                "atom_index": None,
                "component_index": int(component_index),
                "instance_index": instance_index,
                "monomer_index": int(token_residue_indices[token_index]),
                "residue_index": int(token_residue_indices[token_index]),
                "chain_index": instance_index,
                "entity_index": entity_index,
                "symmetry_index": int(token_symmetry_indices[token_index]),
                "molecule_type": 3 if polymer_type is None else _MOLECULE_TYPES[polymer_type],
                "polymer_type": polymer_type,
                "residue_type": int(layout.token_type_index[token_index]),
                "modified": not standard,
                "center": center,
                "distogram": by_name.get(distogram_name, center),
                "frame": tuple(by_name.get(name, center) for name in frame_names),
                "affinity": False,
            }
        )
    return tokens


def _prepare_boltz2_features(
    system: MolecularSystem,
    *,
    tokens: list[dict[str, object]],
    structure: Structure | None,
    ligand_instance_index: int | None,
    affinity_mw: float | None,
    msa: MSA | MsaCollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    templates: Sequence[Template | Boltz2Template],
    design: Sequence[ComponentReference],
    self_template: Sequence[ComponentReference],
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Boltz2Input:
    r"""
    Build Boltz-2 tensors directly from the canonical molecular graph.
    """

    atom_component_index = _atom_component_index(system)

    if ligand_instance_index is not None:
        selected = _affinity_crop(tokens, structure=structure)
        tokens = [tokens[index] for index in selected]
        if not any(cast(bool, token["affinity"]) for token in tokens):
            raise ValueError("the affinity crop does not contain the selected ligand")

    atom_indices = tuple(atom_index for token in tokens for atom_index in cast(tuple[int, ...], token["atoms"]))
    system_to_model_atom = {atom_index: model_atom_index for model_atom_index, atom_index in enumerate(atom_indices)}
    system_atom_to_token = {
        atom_index: token_index
        for token_index, token in enumerate(tokens)
        for atom_index in cast(tuple[int, ...], token["atoms"])
    }
    atom_count = len(atom_indices)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    reference_coordinates = _reference_coordinates(system, atom_indices)
    reference_space_ids = (
        atom_component_index[torch.tensor(atom_indices, device=atom_component_index.device)].detach().cpu()
    )
    resolved = torch.zeros(atom_count, dtype=torch.bool)
    if structure is not None:
        resolved = (
            structure.coordinate_mask[
                torch.tensor(
                    atom_indices,
                    device=structure.coordinate_mask.device,
                )
            ]
            .detach()
            .cpu()
        )

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = reference_coordinates
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = reference_space_ids
    _augment_reference_positions(
        reference_positions[:atom_count],
        reference_space_id[:atom_count],
        resolved,
        generator=generator,
    )

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(
        padded_atom_count,
        4,
        dtype=torch.long,
    )
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_indices)
    canonical_atom_index = torch.tensor(
        atom_indices,
        device=system.atomic_number.device,
    )
    atomic_number[:atom_count] = system.atomic_number[canonical_atom_index].detach().cpu().long()
    reference_charges[:atom_count] = system.formal_charge[canonical_atom_index].detach().cpu().float()
    for model_atom_index, (system_atom_index, atom_name) in enumerate(zip(atom_indices, atom_names)):
        atom_name_characters[model_atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[model_atom_index] = True
        atom_to_token[model_atom_index] = system_atom_to_token[system_atom_index]

    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            torch.nn.functional.one_hot(
                atomic_number,
                128,
            ).float(),
            torch.nn.functional.one_hot(
                atom_name_characters,
                64,
            )
            .reshape(padded_atom_count, 256)
            .float(),
        ),
        dim=-1,
    )
    token_count = len(tokens)
    token_to_rep_atom = torch.tensor(
        [system_to_model_atom[cast(int, token["distogram"])] for token in tokens],
        dtype=torch.long,
    )
    token_bonds, type_bonds = _token_bond_features(
        system,
        system_atom_to_token,
        token_count,
    )
    design_mask = _component_token_mask(system, tokens, design)
    chain_design_mask = _chain_design_mask(
        design_mask,
        torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
        token_bonds,
    )
    physical_potential_values = _physical_potential_features(
        system,
        system_to_model_atom,
    )
    contact_values = _contact_features(
        system,
        tokens,
        system_to_model_atom,
        contacts=contacts,
        pockets=pockets,
    )
    contact_conditioning = contact_values.pop("contact_conditioning")
    contact_threshold = contact_values.pop("contact_threshold")
    template_values = _template_features(system, tokens, templates)
    if self_template:
        if structure is None:
            raise ValueError("Boltz-2 self-template components require a Structure")
        self_template_values = _self_template_features(
            structure,
            tokens,
            self_template,
        )
        template_values = (
            {
                name: torch.cat(
                    (template_values[name], value),
                    dim=0,
                )
                for name, value in self_template_values.items()
            }
            if template_values
            else self_template_values
        )
    msa_values = _msa_features(msa, msa_bindings, tokens, system)
    molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
    residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
    frame_atom_index = torch.tensor(
        [
            [
                system_to_model_atom[atom_index]
                for atom_index in cast(
                    tuple[int, int, int],
                    token["frame"],
                )
            ]
            for token in tokens
        ],
        dtype=torch.long,
    )
    cyclic_period_by_instance = _cyclic_periods(system)
    cyclic_period = torch.tensor(
        [cyclic_period_by_instance[cast(int, token["instance_index"])] for token in tokens],
        dtype=torch.float32,
    )

    tensor_values: dict[str, Tensor] = {
        "token_mask": torch.ones(token_count, dtype=torch.bool),
        "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
        "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
        "entity_index": torch.tensor([cast(int, token["entity_index"]) for token in tokens]),
        "symmetry_index": torch.tensor([cast(int, token["symmetry_index"]) for token in tokens]),
        "token_index": torch.arange(token_count),
        "atom_features": atom_features,
        "atom_to_token": atom_to_token,
        "atom_mask": atom_mask,
        "reference_positions": reference_positions,
        "reference_space_id": reference_space_id,
        "token_to_rep_atom": token_to_rep_atom,
        "residue_type_features": torch.nn.functional.one_hot(
            residue_types,
            TOKEN_VOCAB_SIZE,
        ).float(),
        "msa_profile": msa_values.pop("msa_profile"),
        "deletion_mean": msa_values.pop("deletion_mean"),
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        **contact_values,
        "molecule_type": molecule_types,
        "design_mask": design_mask,
        "chain_design_mask": chain_design_mask,
        **physical_potential_values,
        "frame_atom_index": frame_atom_index,
        "method_feature": torch.full(
            (token_count,),
            _METHOD_TYPES[method.lower()],
            dtype=torch.long,
        ),
        "modified": torch.tensor(
            [cast(bool, token["modified"]) for token in tokens],
            dtype=torch.long,
        ),
        "cyclic_period": cyclic_period,
        **msa_values,
        **template_values,
    }
    if ligand_instance_index is not None:
        affinity_msa_tokens = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
        tensor_values.update(
            {
                "affinity_token_mask": torch.tensor([cast(bool, token["affinity"]) for token in tokens]),
                "affinity_msa_profile": torch.nn.functional.one_hot(
                    affinity_msa_tokens,
                    MSA_VOCAB_SIZE,
                ).float(),
                "affinity_deletion_mean": torch.zeros(token_count),
            }
        )
    placed = {
        name: (value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device))
        for name, value in tensor_values.items()
    }
    return Boltz2Input(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        system=system,
        atom_index=torch.cat(
            (
                torch.tensor(
                    atom_indices,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (padded_atom_count - atom_count,),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
            )
        ).unsqueeze(0),
        token_component_index=torch.tensor(
            [cast(int, token["component_index"]) for token in tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        token_atom_index=torch.tensor(
            [-1 if token["atom_index"] is None else cast(int, token["atom_index"]) for token in tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        affinity_mw=(
            None
            if affinity_mw is None
            else torch.tensor(
                [[affinity_mw]],
                dtype=dtype,
                device=device,
            )
        ),
        batch_size=[],
    )


def _atom_component_index(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            device=system.component_atom_ptr.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


def _prepare_boltz2_target(
    structure: Structure,
    atom_index: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Boltz2StructureTarget:
    r"""
    Lower full-axis canonical truth and its discrete symmetry candidates.
    """

    system = structure.system
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    chain_span: list[tuple[int, int]] = []
    chain_equivalence: list[int] = []
    equivalence_by_layout: dict[
        tuple[int, tuple[tuple[int, int], ...]],
        int,
    ] = {}
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        atom_start = component_atoms[component_start]
        atom_stop = component_atoms[component_stop]
        chain_span.append((atom_start, atom_stop))
        layout_key = (
            entity_index,
            tuple(
                (
                    component_index - component_start,
                    atom_symbols[system_atom_index],
                )
                for component_index in range(
                    component_start,
                    component_stop,
                )
                for system_atom_index in range(
                    component_atoms[component_index],
                    component_atoms[component_index + 1],
                )
            ),
        )
        chain_equivalence.append(
            equivalence_by_layout.setdefault(
                layout_key,
                len(equivalence_by_layout),
            )
        )

    symmetry_atom_index, symmetry_permutation = _truth_symmetry_features(
        system,
        tuple(int(value) for value in atom_index[0].detach().cpu().tolist() if value >= 0),
    )
    return Boltz2StructureTarget(  # type: ignore[call-arg]
        coordinates=structure.coordinates.to(
            device=device,
            dtype=dtype,
        )[None, None],
        coordinate_mask=structure.coordinate_mask.to(
            device=device,
        ).unsqueeze(0),
        chain_span=torch.tensor(
            chain_span,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        chain_equivalence_index=torch.tensor(
            chain_equivalence,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        symmetry_atom_index=symmetry_atom_index.to(
            device=device,
        ).unsqueeze(0),
        symmetry_permutation=symmetry_permutation.to(
            device=device,
        ).unsqueeze(0),
        batch_size=[1],
    )


def _truth_symmetry_features(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    r"""
    Build complete local permutations in stable model-occurrence order.
    """

    system_to_model_atom = {atom_index: model_atom_index for model_atom_index, atom_index in enumerate(atom_indices)}
    component_instance = (
        torch.repeat_interleave(
            torch.arange(
                system.instance_symbol.numel(),
                device=system.instance_component_ptr.device,
            ),
            system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
        )
        .detach()
        .cpu()
        .tolist()
    )
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    component_boundaries = component_boundary_atoms(system)
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for component_index, instance_index in enumerate(component_instance):
        entity_index = instance_entities[instance_index]
        entity_type = entity_types[entity_index]
        component_id = system.symbols[component_symbols[component_index]]
        standard = _STANDARD_LOCAL_SYMMETRY.get(component_id)
        if component_id in _STANDARD_BY_ENTITY_TYPE.get(
            entity_type,
            frozenset(),
        ):
            if standard is not None:
                by_name = {
                    system.symbols[atom_symbols[atom_index]]: atom_index
                    for atom_index in range(
                        component_atoms[component_index],
                        component_atoms[component_index + 1],
                    )
                    if atom_index in system_to_model_atom
                }
                permutations = tuple(
                    tuple(system_to_model_atom[by_name[atom_id]] for atom_id in names)
                    for names in standard
                    if all(atom_id in by_name for atom_id in names)
                )
                if len(permutations) > 1:
                    groups.append((permutations[0], permutations))
            continue

        rdkit_group = _rdkit_symmetry_group(
            system,
            component_index,
            system_to_model_atom,
            component_boundaries[component_index],
        )
        if rdkit_group is not None:
            groups.append(rdkit_group)

    if not groups:
        return (
            torch.empty(0, 0, dtype=torch.long),
            torch.empty(0, 0, 0, dtype=torch.long),
        )

    max_width = max(len(group) for group, _ in groups)
    max_permutations = max(len(permutations) for _, permutations in groups)
    group_atom_index = torch.full(
        (len(groups), max_width),
        -1,
        dtype=torch.long,
    )
    permutation = torch.full(
        (len(groups), max_permutations, max_width),
        -1,
        dtype=torch.long,
    )
    for group_index, (atom_group, permutations) in enumerate(groups):
        group_atom_index[
            group_index,
            : len(atom_group),
        ] = torch.tensor(atom_group)
        for permutation_index, value in enumerate(permutations):
            permutation[
                group_index,
                permutation_index,
                : len(value),
            ] = torch.tensor(value)
    return group_atom_index, permutation


def _rdkit_symmetry_group(
    system: MolecularSystem,
    component_index: int,
    system_to_model_atom: dict[int, int],
    boundary_atoms: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    from rdkit import Chem

    if not bool(system.component_topology_complete[component_index]):
        return None
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    visible_local_index = [
        atom_index - atom_start for atom_index in range(atom_start, atom_stop) if atom_index in system_to_model_atom
    ]
    if len(visible_local_index) < 2:
        return None

    edge_indices = (
        (
            (system.edge_index[0] >= atom_start)
            & (system.edge_index[1] < atom_stop)
            & (system.bond_role == BOND_ROLE_COMPONENT)
        )
        .nonzero(as_tuple=False)
        .flatten()
    )
    if edge_indices.numel() and not bool(system.bond_order_known[edge_indices].all()):
        return None
    molecule = _canonical_rdkit_molecule(
        system,
        tuple(range(atom_start, atom_stop)),
        edge_indices.detach().cpu().tolist(),
    )
    for boundary_index, atom_index in enumerate(
        sorted(boundary_atoms),
        start=1,
    ):
        molecule.GetAtomWithIdx(atom_index - atom_start).SetIsotope(1000 + boundary_index)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    identity = tuple(system_to_model_atom[atom_start + local_index] for local_index in visible_local_index)
    permutations = {identity}
    for match in molecule.GetSubstructMatches(
        molecule,
        uniquify=False,
        useChirality=True,
        maxMatches=1000,
    ):
        mapped_local_index = tuple(match[index] for index in visible_local_index)
        if set(mapped_local_index) != set(visible_local_index):
            continue
        permutations.add(tuple(system_to_model_atom[atom_start + index] for index in mapped_local_index))
    if len(permutations) == 1:
        return None
    ordered = (identity, *sorted(permutations - {identity}))
    return identity, ordered


def _physical_potential_features(
    system: MolecularSystem,
    system_to_model_atom: dict[int, int],
) -> dict[str, Tensor]:
    r"""
    Lower molecular geometry, assembly connections, and entity copies.
    """

    geometry_pairs: list[tuple[int, int]] = []
    geometry_bond_mask: list[bool] = []
    geometry_angle_mask: list[bool] = []
    geometry_upper_bounds: list[float] = []
    geometry_lower_bounds: list[float] = []
    chiral_atoms: list[tuple[int, int, int, int]] = []
    chiral_is_r: list[bool] = []
    stereo_bonds: list[tuple[int, int, int, int]] = []
    stereo_is_e: list[bool] = []
    planar_bonds: list[tuple[int, int, int, int, int, int]] = []
    constraint_scopes: list[tuple[Mol, dict[int, int]]] = []

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        entity_type = entity_types[entity_index]
        if entity_type != ENTITY_MOLECULE:
            for component_index in range(
                component_start,
                component_stop,
            ):
                component_id = system.symbols[component_symbols[component_index]]
                polymer_type = _component_polymer_type(
                    entity_type,
                    component_id,
                )
                if polymer_type is not None and component_id in _CONSTRAINT_FREE_COMPONENTS[polymer_type]:
                    continue
                if not bool(system.component_topology_complete[component_index]):
                    continue
                atom_start = component_atoms[component_index]
                atom_stop = component_atoms[component_index + 1]
                edge_indices = (
                    (
                        (system.edge_index[0] >= atom_start)
                        & (system.edge_index[1] < atom_stop)
                        & (system.bond_role == BOND_ROLE_COMPONENT)
                    )
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                if edge_indices.numel() and not bool(system.bond_order_known[edge_indices].all()):
                    continue
                rdkit_molecule = _canonical_rdkit_molecule(
                    system,
                    tuple(range(atom_start, atom_stop)),
                    edge_indices.detach().cpu().tolist(),
                )
                component_atom_index = {
                    atom_index - atom_start: system_to_model_atom[atom_index]
                    for atom_index in range(atom_start, atom_stop)
                    if atom_index in system_to_model_atom
                }
                constraint_scopes.append((rdkit_molecule, component_atom_index))
        else:
            if not bool(system.component_topology_complete[component_start:component_stop].all()):
                continue
            atom_start = component_atoms[component_start]
            atom_stop = component_atoms[component_stop]
            edge_indices = (
                (
                    (system.edge_index[0] >= atom_start)
                    & (system.edge_index[1] < atom_stop)
                    & (system.bond_role <= BOND_ROLE_ENTITY)
                )
                .nonzero(as_tuple=False)
                .flatten()
            )
            if edge_indices.numel() and not bool(system.bond_order_known[edge_indices].all()):
                continue
            rdkit_molecule = _canonical_rdkit_molecule(
                system,
                tuple(range(atom_start, atom_stop)),
                edge_indices.detach().cpu().tolist(),
            )
            molecule_atom_index = {
                atom_index - atom_start: system_to_model_atom[atom_index]
                for atom_index in range(atom_start, atom_stop)
                if atom_index in system_to_model_atom
            }
            constraint_scopes.append((rdkit_molecule, molecule_atom_index))

    for rdkit_molecule, atom_index in constraint_scopes:
        (
            scope_geometry_pairs,
            scope_bond_mask,
            scope_angle_mask,
            scope_upper_bounds,
            scope_lower_bounds,
            scope_chiral_atoms,
            scope_chiral_is_r,
            scope_stereo_bonds,
            scope_stereo_is_e,
            scope_planar_bonds,
        ) = _rdkit_constraint_features(
            rdkit_molecule,
            atom_index,
        )
        geometry_pairs.extend(scope_geometry_pairs)
        geometry_bond_mask.extend(scope_bond_mask)
        geometry_angle_mask.extend(scope_angle_mask)
        geometry_upper_bounds.extend(scope_upper_bounds)
        geometry_lower_bounds.extend(scope_lower_bounds)
        chiral_atoms.extend(scope_chiral_atoms)
        chiral_is_r.extend(scope_chiral_is_r)
        stereo_bonds.extend(scope_stereo_bonds)
        stereo_is_e.extend(scope_stereo_is_e)
        planar_bonds.extend(scope_planar_bonds)

    atom_component = _atom_component_index(system)
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_components = atom_component.detach().cpu().tolist()
    component_instances = component_instance.detach().cpu().tolist()
    connection_indices = (
        (system.bond_role == BOND_ROLE_ASSEMBLY).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    )
    connection_edges = system.edge_index[:, connection_indices].detach().cpu().T.tolist() if connection_indices else []
    connected_chains = []
    connected_atoms = []
    for first, second in connection_edges:
        if first not in system_to_model_atom or second not in system_to_model_atom:
            continue
        connected_chains.append(
            (
                component_instances[atom_components[first]],
                component_instances[atom_components[second]],
            )
        )
        connected_atoms.append(
            (
                system_to_model_atom[first],
                system_to_model_atom[second],
            )
        )

    symmetric_chains = [
        (first_index, second_index)
        for first_index, first_entity in enumerate(instance_entities)
        for second_index, second_entity in enumerate(instance_entities)
        if first_index < second_index and first_entity == second_entity
    ]

    return {
        "geometry_pair_index": _index_columns(geometry_pairs, 2),
        "geometry_bond_mask": torch.tensor(
            geometry_bond_mask,
            dtype=torch.bool,
        ),
        "geometry_angle_mask": torch.tensor(
            geometry_angle_mask,
            dtype=torch.bool,
        ),
        "geometry_upper_bounds": torch.tensor(
            geometry_upper_bounds,
            dtype=torch.float32,
        ),
        "geometry_lower_bounds": torch.tensor(
            geometry_lower_bounds,
            dtype=torch.float32,
        ),
        "chiral_atom_index": _index_columns(chiral_atoms, 4),
        "chiral_atom_is_r": torch.tensor(
            chiral_is_r,
            dtype=torch.bool,
        ),
        "stereo_bond_index": _index_columns(stereo_bonds, 4),
        "stereo_bond_is_e": torch.tensor(
            stereo_is_e,
            dtype=torch.bool,
        ),
        "planar_bond_index": _index_columns(planar_bonds, 6),
        "connected_chain_index": _index_columns(
            connected_chains,
            2,
        ),
        "connected_atom_index": _index_columns(connected_atoms, 2),
        "symmetric_chain_index": _index_columns(
            symmetric_chains,
            2,
        ),
    }


def _index_columns(values: Sequence[tuple[int, ...]], width: int) -> Tensor:
    if not values:
        return torch.empty(width, 0, dtype=torch.long)
    return torch.tensor(values, dtype=torch.long).T.contiguous()


def _rdkit_constraint_features(
    molecule: Mol,
    atom_index: dict[int, int],
) -> tuple[
    list[tuple[int, int]],
    list[bool],
    list[bool],
    list[float],
    list[float],
    list[tuple[int, int, int, int]],
    list[bool],
    list[tuple[int, int, int, int]],
    list[bool],
    list[tuple[int, int, int, int, int, int]],
]:
    from rdkit import Chem
    from rdkit.Chem import HybridizationType
    from rdkit.Chem.rdchem import BondStereo
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    molecule = Chem.AddHs(molecule)
    molecule.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(molecule)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)

    bounds = GetMoleculeBoundsMatrix(
        molecule,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bond_pattern = Chem.MolFromSmarts("*~*")
    angle_pattern = Chem.MolFromSmarts("*~*~*")
    bond_pairs = {tuple(sorted(match)) for match in molecule.GetSubstructMatches(bond_pattern)}
    angle_pairs = {tuple(sorted((match[0], match[2]))) for match in molecule.GetSubstructMatches(angle_pattern)}

    geometry_pairs: list[tuple[int, int]] = []
    geometry_bond_mask: list[bool] = []
    geometry_angle_mask: list[bool] = []
    geometry_upper_bounds: list[float] = []
    geometry_lower_bounds: list[float] = []
    for first in range(molecule.GetNumAtoms()):
        for second in range(first + 1, molecule.GetNumAtoms()):
            if first not in atom_index or second not in atom_index:
                continue
            geometry_pairs.append((atom_index[first], atom_index[second]))
            geometry_bond_mask.append((first, second) in bond_pairs)
            geometry_angle_mask.append((first, second) in angle_pairs)
            geometry_upper_bounds.append(float(bounds[first, second]))
            geometry_lower_bounds.append(float(bounds[second, first]))

    chiral_atoms: list[tuple[int, int, int, int]] = []
    chiral_is_r: list[bool] = []
    if all(atom.HasProp("_CIPRank") for atom in molecule.GetAtoms()):
        for center_index, orientation in Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=False,
        ):
            center = molecule.GetAtomWithIdx(center_index)
            neighbors = tuple(
                index
                for index, _rank in sorted(
                    ((neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank"))) for neighbor in center.GetNeighbors()),
                    key=lambda value: value[1],
                    reverse=True,
                )
            )
            if len(neighbors) > 4 or center.GetHybridization() != HybridizationType.SP3:
                continue
            indices = (*neighbors[:3], center_index)
            if all(index in atom_index for index in indices):
                chiral_atoms.append(
                    cast(
                        tuple[int, int, int, int],
                        tuple(atom_index[index] for index in indices),
                    )
                )
                chiral_is_r.append(orientation == "R")
            if len(neighbors) == 4:
                for skipped in range(3):
                    neighbor_set = neighbors[:skipped] + neighbors[skipped + 1 :]
                    indices = (
                        neighbor_set[::-1] + (center_index,) if skipped % 2 == 0 else neighbor_set + (center_index,)
                    )
                    if all(index in atom_index for index in indices):
                        chiral_atoms.append(
                            cast(
                                tuple[int, int, int, int],
                                tuple(atom_index[index] for index in indices),
                            )
                        )
                        chiral_is_r.append(orientation == "R")

    stereo_bonds: list[tuple[int, int, int, int]] = []
    stereo_is_e: list[bool] = []
    if all(atom.HasProp("_CIPRank") for atom in molecule.GetAtoms()):
        for bond in molecule.GetBonds():
            stereo = bond.GetStereo()
            if stereo not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
                continue
            start_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            start_neighbors = [
                index
                for index, _rank in sorted(
                    (
                        (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                        for neighbor in molecule.GetAtomWithIdx(start_index).GetNeighbors()
                        if neighbor.GetIdx() != end_index
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
            ]
            end_neighbors = [
                index
                for index, _rank in sorted(
                    (
                        (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                        for neighbor in molecule.GetAtomWithIdx(end_index).GetNeighbors()
                        if neighbor.GetIdx() != start_index
                    ),
                    key=lambda value: value[1],
                    reverse=True,
                )
            ]
            indices = (start_neighbors[0], start_index, end_index, end_neighbors[0])
            if all(index in atom_index for index in indices):
                stereo_bonds.append(
                    cast(
                        tuple[int, int, int, int],
                        tuple(atom_index[index] for index in indices),
                    )
                )
                stereo_is_e.append(stereo == BondStereo.STEREOE)
            if len(start_neighbors) == 2 and len(end_neighbors) == 2:
                indices = (
                    start_neighbors[1],
                    start_index,
                    end_index,
                    end_neighbors[1],
                )
                if all(index in atom_index for index in indices):
                    stereo_bonds.append(
                        cast(
                            tuple[int, int, int, int],
                            tuple(atom_index[index] for index in indices),
                        )
                    )
                    stereo_is_e.append(stereo == BondStereo.STEREOE)

    planar_pattern = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    planar_bonds = [
        cast(
            tuple[int, int, int, int, int, int],
            tuple(atom_index[index] for index in match),
        )
        for match in molecule.GetSubstructMatches(planar_pattern)
        if all(index in atom_index for index in match)
    ]
    return (
        geometry_pairs,
        geometry_bond_mask,
        geometry_angle_mask,
        geometry_upper_bounds,
        geometry_lower_bounds,
        chiral_atoms,
        chiral_is_r,
        stereo_bonds,
        stereo_is_e,
        planar_bonds,
    )


def _conditioning_required_token_mask(
    system: MolecularSystem,
    layout: TokenLayout,
    *,
    self_template: Sequence[ComponentReference],
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
) -> Tensor | None:
    if not self_template and not contacts and not pockets:
        return None

    required = torch.zeros_like(layout.token_component_index, dtype=torch.bool)
    component_references = list(self_template)
    endpoints: list[ComponentReference | AtomSite] = []
    for contact in contacts:
        endpoints.extend((contact.endpoint_a, contact.endpoint_b))
    for pocket in pockets:
        try:
            binder_instance = system.instance_index(pocket.binder)
        except (KeyError, ValueError):
            pass
        else:
            required |= layout.token_instance_index == binder_instance
        endpoints.extend(pocket.contacts)

    for endpoint in endpoints:
        if isinstance(endpoint, AtomSite):
            try:
                atom_index = system.atom_index(endpoint)
            except (KeyError, IndexError, ValueError):
                continue
            required |= layout.token_atom_index == atom_index
        else:
            component_references.append(endpoint)

    for reference in component_references:
        try:
            component_index = system.component_index(reference)
        except (KeyError, IndexError, ValueError):
            continue
        required |= layout.token_component_index == component_index
    return required


def _contact_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    system_to_model_atom: dict[int, int],
    *,
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
) -> dict[str, Tensor]:
    token_count = len(tokens)
    token_by_component = {
        cast(int, token["component_index"]): token_index
        for token_index, token in enumerate(tokens)
        if token["atom_index"] is None
    }
    token_by_atom = {
        cast(int, token["atom_index"]): token_index
        for token_index, token in enumerate(tokens)
        if token["atom_index"] is not None
    }
    contact_class = torch.full(
        (token_count, token_count),
        _CONTACT_UNSELECTED,
        dtype=torch.long,
    )
    threshold = torch.zeros(token_count, token_count)
    token_atoms = [
        torch.tensor(
            [
                system_to_model_atom[atom_index]
                for atom_index in cast(
                    tuple[int, ...],
                    token["atoms"],
                )
            ],
            dtype=torch.long,
        )
        for token in tokens
    ]
    potential_pairs: list[Tensor] = []
    potential_union: list[Tensor] = []
    potential_thresholds: list[Tensor] = []
    union_index = 0

    for pocket in pockets:
        binder_instance = system.instance_index(pocket.binder)
        binder_tokens = [
            token_index for token_index, token in enumerate(tokens) if token["instance_index"] == binder_instance
        ]
        if not binder_tokens:
            raise ValueError(f"the affinity crop excludes pocket binder {pocket.binder!r}")
        for endpoint in pocket.contacts:
            pocket_token = _condition_endpoint_token(
                system,
                token_by_component,
                token_by_atom,
                endpoint,
            )
            contact_class[
                binder_tokens,
                pocket_token,
            ] = _CONTACT_BINDER_TO_POCKET
            contact_class[
                pocket_token,
                binder_tokens,
            ] = _CONTACT_POCKET_TO_BINDER
            threshold[
                binder_tokens,
                pocket_token,
            ] = pocket.max_distance
            threshold[
                pocket_token,
                binder_tokens,
            ] = pocket.max_distance
            if pocket.force:
                binder_atoms = torch.cat([token_atoms[index] for index in binder_tokens])
                pairs = torch.cartesian_prod(
                    binder_atoms,
                    token_atoms[pocket_token],
                ).T
                potential_pairs.append(pairs)
                potential_union.append(
                    torch.full(
                        (pairs.shape[1],),
                        union_index,
                    )
                )
                potential_thresholds.append(
                    torch.full(
                        (pairs.shape[1],),
                        pocket.max_distance,
                    )
                )
                union_index += 1

    for contact in contacts:
        token_a = _condition_endpoint_token(
            system,
            token_by_component,
            token_by_atom,
            contact.endpoint_a,
        )
        token_b = _condition_endpoint_token(
            system,
            token_by_component,
            token_by_atom,
            contact.endpoint_b,
        )
        contact_class[token_a, token_b] = _CONTACT_PAIR
        contact_class[token_b, token_a] = _CONTACT_PAIR
        threshold[token_a, token_b] = contact.max_distance
        threshold[token_b, token_a] = contact.max_distance
        if contact.force:
            pairs = torch.cartesian_prod(
                token_atoms[token_a],
                token_atoms[token_b],
            ).T
            potential_pairs.append(pairs)
            potential_union.append(torch.full((pairs.shape[1],), union_index))
            potential_thresholds.append(
                torch.full(
                    (pairs.shape[1],),
                    contact.max_distance,
                )
            )
            union_index += 1

    if bool((contact_class == _CONTACT_UNSELECTED).all()):
        contact_class.fill_(_CONTACT_UNSPECIFIED)
    if potential_pairs:
        pair_index = torch.cat(potential_pairs, dim=1)
        union = torch.cat(potential_union)
    else:
        pair_index = torch.empty(2, 0, dtype=torch.long)
        union = torch.empty(0, dtype=torch.long)
    return {
        "contact_conditioning": torch.nn.functional.one_hot(
            contact_class,
            5,
        ),
        "contact_threshold": threshold,
        "contact_pair_index": pair_index,
        "contact_union_index": union,
        "contact_thresholds": (torch.cat(potential_thresholds) if potential_thresholds else torch.empty(0)),
    }


def _condition_endpoint_token(
    system: MolecularSystem,
    token_by_component: dict[int, int],
    token_by_atom: dict[int, int],
    endpoint: ComponentReference | AtomSite,
) -> int:
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    if isinstance(endpoint, AtomSite):
        atom_index = system.atom_index(endpoint)
        component_index = int(_atom_component_index(system)[atom_index])
        instance_index = int(component_instance[component_index])
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) != ENTITY_MOLECULE:
            raise ValueError("Boltz-2 polymer contact endpoints must identify residues, not exact atoms")
        if int(system.atomic_number[atom_index]) == 1:
            raise ValueError(
                f"Boltz-2 tokenizes only heavy atoms, so contact endpoint {endpoint} has no token: "
                "name the heavy atom it is bonded to"
            )
        token_index = token_by_atom.get(atom_index)
        if token_index is None:
            raise ValueError(f"the affinity crop excludes contact endpoint {endpoint}")
        return token_index

    component_index = system.component_index(endpoint)
    instance_index = int(component_instance[component_index])
    entity_index = int(system.instance_entity[instance_index])
    if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
        raise ValueError("Boltz-2 non-polymer contact endpoints must identify exact atoms")
    token_index = token_by_component.get(component_index)
    if token_index is None:
        raise ValueError(f"the affinity crop excludes contact endpoint {endpoint}")
    return token_index


def _template_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    templates: Sequence[Template | Boltz2Template],
) -> dict[str, Tensor]:
    if not templates:
        return {}

    token_count = len(tokens)
    token_by_component = {
        cast(int, token["component_index"]): token_index
        for token_index, token in enumerate(tokens)
        if token["atom_index"] is None
    }
    tokens_by_instance: dict[int, list[int]] = {}
    for token_index, token in enumerate(tokens):
        tokens_by_instance.setdefault(
            cast(int, token["instance_index"]),
            [],
        ).append(token_index)

    instance_entities = system.instance_entity.detach().cpu().tolist()
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    residue_type_rows: list[Tensor] = []
    frame_rotation_rows: list[Tensor] = []
    frame_translation_rows: list[Tensor] = []
    frame_mask_rows: list[Tensor] = []
    beta_carbon_position_rows: list[Tensor] = []
    alpha_carbon_position_rows: list[Tensor] = []
    beta_carbon_mask_rows: list[Tensor] = []
    template_mask_rows: list[Tensor] = []
    visibility_rows: list[Tensor] = []
    force_rows: list[bool] = []
    force_threshold_rows: list[float] = []
    identity = torch.eye(3)

    for template_index, value in enumerate(templates):
        if isinstance(value, Boltz2Template):
            template = value.template
            force_rows.append(value.force)
            force_threshold_rows.append(float("inf") if value.threshold is None else value.threshold)
        else:
            template = value
            force_rows.append(False)
            force_threshold_rows.append(float("inf"))
        residue_type = torch.full((token_count,), TOKEN_OTHER_INDEX, dtype=torch.long)
        frame_rotations = torch.zeros(token_count, 3, 3)
        frame_translations = torch.zeros(token_count, 3)
        frame_mask = torch.zeros(token_count)
        beta_carbon_positions = torch.zeros(token_count, 3)
        alpha_carbon_positions = torch.zeros(token_count, 3)
        beta_carbon_mask = torch.zeros(token_count)
        template_mask = torch.zeros(token_count)
        visibility = torch.tensor([-1.0 - cast(int, token["chain_index"]) for token in tokens])
        covered_instances: set[int] = set()
        template_system = template.structure.system
        template_instance_components = template_system.instance_component_ptr.detach().cpu().tolist()
        template_component_symbols = template_system.component_symbol.detach().cpu().tolist()

        for alignment in template.alignments:
            target_entity_index = system.entity_index(alignment.target_entity_id)
            if int(system.entity_type[target_entity_index]) != ENTITY_PROTEIN:
                raise ValueError("Boltz-2 templates support only protein target alignments")
            template_instance_index = template_system.instance_index(alignment.template_instance_id)
            template_entity_index = int(template_system.instance_entity[template_instance_index])
            if int(template_system.entity_type[template_entity_index]) != ENTITY_PROTEIN:
                raise ValueError("Boltz-2 templates support only protein template alignments")

            if alignment.target_instance_id is None:
                target_instances = tuple(
                    instance_index
                    for instance_index, entity_index in enumerate(instance_entities)
                    if entity_index == target_entity_index
                )
            else:
                target_instance_index = system.instance_index(alignment.target_instance_id)
                if instance_entities[target_instance_index] != target_entity_index:
                    raise ValueError(
                        f"template target instance "
                        f"{alignment.target_instance_id!r} does not "
                        f"belong to entity "
                        f"{alignment.target_entity_id!r}"
                    )
                target_instances = (target_instance_index,)

            template_component_start = template_instance_components[template_instance_index]
            template_component_stop = template_instance_components[template_instance_index + 1]
            for (
                target_component_offset,
                template_component_offset,
            ) in alignment.residue_pairs:
                if not target_instances:
                    continue
                target_component_count = (
                    instance_components[target_instances[0] + 1] - instance_components[target_instances[0]]
                )
                if target_component_offset >= target_component_count:
                    raise IndexError(
                        f"template target component index "
                        f"{target_component_offset} is outside entity "
                        f"{alignment.target_entity_id!r}"
                    )
                template_component_index = template_component_start + template_component_offset
                if template_component_index >= template_component_stop:
                    raise IndexError(
                        f"template component index "
                        f"{template_component_offset} is outside instance "
                        f"{alignment.template_instance_id!r}"
                    )
                template_component_id = template_system.symbols[template_component_symbols[template_component_index]]
                template_residue_type = component_token_index(
                    template_component_id,
                    "protein",
                )
                atom_values = {
                    atom_id: _template_atom(
                        template.structure,
                        template_component_index,
                        atom_id,
                    )
                    for atom_id in ("N", "CA", "C", "CB")
                }
                beta_atom = "CA" if template_component_id == "GLY" else "CB"

                for target_instance in target_instances:
                    target_component_index = instance_components[target_instance] + target_component_offset
                    target_token = token_by_component.get(target_component_index)
                    if target_token is None:
                        continue
                    covered_instances.add(target_instance)
                    residue_type[target_token] = template_residue_type
                    template_mask[target_token] = 1
                    frame_rotations[target_token] = identity

                    alpha_carbon, alpha_present = atom_values["CA"]
                    if alpha_present:
                        alpha_carbon_positions[target_token] = alpha_carbon
                    beta_carbon, beta_present = atom_values[beta_atom]
                    if beta_present:
                        beta_carbon_positions[target_token] = beta_carbon
                        beta_carbon_mask[target_token] = 1

                    nitrogen, nitrogen_present = atom_values["N"]
                    carbon, carbon_present = atom_values["C"]
                    if nitrogen_present and alpha_present and carbon_present:
                        first = carbon - alpha_carbon
                        first_axis = first / (torch.linalg.vector_norm(first) + 1e-10)
                        second = nitrogen - alpha_carbon
                        second = second - first_axis * torch.dot(
                            first_axis,
                            second,
                        )
                        second_axis = second / (torch.linalg.vector_norm(second) + 1e-10)
                        third_axis = torch.linalg.cross(
                            first_axis,
                            second_axis,
                        )
                        frame_rotations[target_token] = torch.stack(
                            (
                                first_axis,
                                second_axis,
                                third_axis,
                            ),
                            dim=-1,
                        )
                        frame_translations[target_token] = alpha_carbon
                        frame_mask[target_token] = 1

        for instance_index in covered_instances:
            visibility[tokens_by_instance[instance_index]] = float(template_index)

        residue_type_rows.append(torch.nn.functional.one_hot(residue_type, TOKEN_VOCAB_SIZE))
        frame_rotation_rows.append(frame_rotations)
        frame_translation_rows.append(frame_translations)
        frame_mask_rows.append(frame_mask)
        beta_carbon_position_rows.append(beta_carbon_positions)
        alpha_carbon_position_rows.append(alpha_carbon_positions)
        beta_carbon_mask_rows.append(beta_carbon_mask)
        template_mask_rows.append(template_mask)
        visibility_rows.append(visibility)

    return {
        "template_residue_type_features": torch.stack(residue_type_rows),
        "template_frame_rotations": torch.stack(frame_rotation_rows),
        "template_frame_translations": torch.stack(frame_translation_rows),
        "template_frame_mask": torch.stack(frame_mask_rows),
        "template_beta_carbon_positions": torch.stack(beta_carbon_position_rows),
        "template_alpha_carbon_positions": torch.stack(alpha_carbon_position_rows),
        "template_beta_carbon_mask": torch.stack(beta_carbon_mask_rows),
        "template_mask": torch.stack(template_mask_rows),
        "template_visibility_index": torch.stack(visibility_rows),
        "template_force": torch.tensor(
            force_rows,
            dtype=torch.bool,
        ),
        "template_force_threshold": torch.tensor(force_threshold_rows),
    }


def _self_template_features(
    structure: Structure,
    tokens: list[dict[str, object]],
    references: Sequence[ComponentReference],
) -> dict[str, Tensor]:
    r"""
    Lower one direct coordinate template over canonical components.
    """

    selected = _component_token_mask(
        structure.system,
        tokens,
        references,
    )
    token_count = len(tokens)
    residue_type = torch.full((token_count,), TOKEN_OTHER_INDEX, dtype=torch.long)
    frame_rotations = torch.eye(3).expand(token_count, -1, -1).clone()
    frame_rotations[~selected] = 0
    frame_translations = torch.zeros(token_count, 3)
    frame_mask = torch.zeros(token_count)
    beta_carbon_positions = torch.zeros(token_count, 3)
    alpha_carbon_positions = torch.zeros(token_count, 3)
    beta_carbon_mask = torch.zeros(token_count)
    template_mask = selected.float()
    visibility = selected.float()

    for token_index, token in enumerate(tokens):
        if not bool(selected[token_index]):
            continue
        residue_type[token_index] = cast(
            int,
            token["residue_type"],
        )
        center, center_present = _structure_atom(
            structure,
            cast(int, token["center"]),
        )
        if center_present:
            alpha_carbon_positions[token_index] = center
        distogram, distogram_present = _structure_atom(
            structure,
            cast(int, token["distogram"]),
        )
        if distogram_present:
            beta_carbon_positions[token_index] = distogram
            beta_carbon_mask[token_index] = 1

        if cast(int, token["molecule_type"]) != _MOLECULE_TYPES["protein"]:
            continue
        nitrogen_index, alpha_index, carbon_index = cast(
            tuple[int, int, int],
            token["frame"],
        )
        if len({nitrogen_index, alpha_index, carbon_index}) != 3:
            continue
        nitrogen, nitrogen_present = _structure_atom(
            structure,
            nitrogen_index,
        )
        alpha_carbon, alpha_present = _structure_atom(
            structure,
            alpha_index,
        )
        carbon, carbon_present = _structure_atom(
            structure,
            carbon_index,
        )
        if nitrogen_present and alpha_present and carbon_present:
            first = carbon - alpha_carbon
            first_axis = first / (torch.linalg.vector_norm(first) + 1e-10)
            second = nitrogen - alpha_carbon
            second = second - first_axis * torch.dot(
                first_axis,
                second,
            )
            second_axis = second / (torch.linalg.vector_norm(second) + 1e-10)
            third_axis = torch.linalg.cross(
                first_axis,
                second_axis,
            )
            frame_rotations[token_index] = torch.stack(
                (first_axis, second_axis, third_axis),
                dim=-1,
            )
            frame_translations[token_index] = alpha_carbon
            frame_mask[token_index] = 1

    return {
        "template_residue_type_features": (torch.nn.functional.one_hot(residue_type, TOKEN_VOCAB_SIZE).unsqueeze(0)),
        "template_frame_rotations": frame_rotations.unsqueeze(0),
        "template_frame_translations": (frame_translations.unsqueeze(0)),
        "template_frame_mask": frame_mask.unsqueeze(0),
        "template_beta_carbon_positions": (beta_carbon_positions.unsqueeze(0)),
        "template_alpha_carbon_positions": (alpha_carbon_positions.unsqueeze(0)),
        "template_beta_carbon_mask": (beta_carbon_mask.unsqueeze(0)),
        "template_mask": template_mask.unsqueeze(0),
        "template_visibility_index": visibility.unsqueeze(0),
        "template_force": torch.zeros(1, dtype=torch.bool),
        "template_force_threshold": torch.full(
            (1,),
            float("inf"),
        ),
    }


def _template_atom(
    structure: Structure,
    component_index: int,
    atom_id: str,
) -> tuple[Tensor, bool]:
    system = structure.system
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    atom_symbols = system.atom_symbol[atom_start:atom_stop].detach().cpu().tolist()
    for offset, symbol_index in enumerate(atom_symbols):
        if system.symbols[symbol_index] == atom_id:
            return _structure_atom(structure, atom_start + offset)
    return torch.zeros(3), False


def _structure_atom(
    structure: Structure,
    atom_index: int,
) -> tuple[Tensor, bool]:
    if not bool(structure.coordinate_mask[atom_index]):
        return torch.zeros(3), False
    return (
        structure.coordinates[atom_index].detach().cpu().to(dtype=torch.float32),
        True,
    )


def _component_token_mask(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    references: Sequence[ComponentReference],
) -> Tensor:
    r"""
    Expand canonical component occurrences to their complete token rows.
    """

    selected = {system.component_index(reference) for reference in references}
    return torch.tensor(
        [cast(int, token["component_index"]) in selected for token in tokens],
        dtype=torch.bool,
    )


def _chain_design_mask(
    design_mask: Tensor,
    chain_index: Tensor,
    token_bonds: Tensor,
) -> Tensor:
    r"""
    Propagate design membership over whole chains and covalently linked chains.
    """

    connections: dict[int, set[int]] = {int(chain): set() for chain in torch.unique(chain_index).tolist()}
    for source, target in torch.nonzero(token_bonds, as_tuple=False).tolist():
        source_chain = int(chain_index[source])
        target_chain = int(chain_index[target])
        connections[source_chain].add(target_chain)
        connections[target_chain].add(source_chain)

    selected = {int(chain) for chain in chain_index[design_mask].tolist()}
    pending = list(selected)
    while pending:
        chain = pending.pop()
        for connected in connections[chain] - selected:
            selected.add(connected)
            pending.append(connected)
    return torch.tensor([int(chain) in selected for chain in chain_index], dtype=torch.bool)


def _affinity_crop(
    tokens: list[dict[str, object]],
    *,
    structure: Structure | None,
) -> list[int]:
    if structure is None:
        atom_count = sum(len(cast(tuple[int, ...], token["atoms"])) for token in tokens)
        polymer_count = sum(cast(int, token["molecule_type"]) != 3 for token in tokens)
        if len(tokens) > _MAX_TOKENS or atom_count > _MAX_ATOMS or polymer_count > _MAX_POLYMER_TOKENS:
            raise ValueError("large affinity systems require a Structure for ligand-centered cropping")
        return list(range(len(tokens)))

    valid: list[int] = []
    centers: list[Tensor] = []
    for token_index, token in enumerate(tokens):
        atom_index = cast(int, token["center"])
        if bool(structure.coordinate_mask[atom_index]):
            valid.append(token_index)
            centers.append(structure.coordinates[atom_index].detach().cpu())
    ligand_centers = [center for index, center in zip(valid, centers) if cast(bool, tokens[index]["affinity"])]
    if not ligand_centers:
        raise ValueError("the selected ligand has no resolved coordinates for affinity cropping")
    center_tensor = torch.stack(centers)
    ligand_tensor = torch.stack(ligand_centers)
    distances = torch.cdist(center_tensor, ligand_tensor).min(dim=1).values
    order = [valid[index] for index in distances.argsort().tolist()]
    chain_tokens: dict[int, list[int]] = {}
    for index, token in enumerate(tokens):
        chain_tokens.setdefault(
            cast(int, token["chain_index"]),
            [],
        ).append(index)
    ligand_tokens = {index for index in valid if cast(bool, tokens[index]["affinity"])}
    cropped: set[int] = set()
    cropped_polymer: set[int] = set()
    total_atoms = 0
    for query_index in order:
        query = tokens[query_index]
        chain = chain_tokens[cast(int, query["chain_index"])]
        if len(chain) <= _CROP_NEIGHBORHOOD:
            neighborhood = chain
        else:
            query_residue = cast(int, query["residue_index"])
            maximum = [
                index
                for index in chain
                if abs(cast(int, tokens[index]["residue_index"]) - query_residue) <= _CROP_NEIGHBORHOOD
            ]
            radius = 0
            neighborhood = [index for index in maximum if cast(int, tokens[index]["residue_index"]) == query_residue]
            while len(neighborhood) < _CROP_NEIGHBORHOOD:
                radius += 1
                neighborhood = [
                    index
                    for index in maximum
                    if abs(
                        cast(
                            int,
                            tokens[index]["residue_index"],
                        )
                        - query_residue
                    )
                    <= radius
                ]
        new = set(neighborhood) - cropped
        new_atoms = sum(len(cast(tuple[int, ...], tokens[index]["atoms"])) for index in new)
        if (
            len(new) > _MAX_TOKENS - len(cropped)
            or total_atoms + new_atoms > _MAX_ATOMS
            or len(cropped_polymer | (new - ligand_tokens)) > _MAX_POLYMER_TOKENS
        ):
            break
        cropped.update(new)
        cropped_polymer.update(new - ligand_tokens)
        total_atoms += new_atoms
    return sorted(cropped)


def _token_bond_features(
    system: MolecularSystem,
    system_atom_to_token: dict[int, int],
    token_count: int,
) -> tuple[Tensor, Tensor]:
    token_bonds = torch.zeros(token_count, token_count)
    type_bonds = torch.zeros(
        token_count,
        token_count,
        dtype=torch.long,
    )
    edge_rows = system.edge_index.detach().cpu().T.tolist()
    bond_order = system.bond_order.detach().cpu().tolist()
    bond_order_known = system.bond_order_known.detach().cpu().tolist()
    bond_aromatic = system.bond_aromatic.detach().cpu().tolist()
    bond_aromatic_known = system.bond_aromatic_known.detach().cpu().tolist()
    bond_role = system.bond_role.detach().cpu().tolist()
    for edge_index, (first, second) in enumerate(edge_rows):
        token_a = system_atom_to_token.get(first)
        token_b = system_atom_to_token.get(second)
        if token_a is None or token_b is None or token_a == token_b:
            continue
        if bond_role[edge_index] == BOND_ROLE_ASSEMBLY:
            bond_type = 6
        elif bond_aromatic_known[edge_index] and bond_aromatic[edge_index]:
            bond_type = 5
        elif bond_order_known[edge_index]:
            bond_type = {
                1.0: 2,
                2.0: 3,
                3.0: 4,
            }.get(bond_order[edge_index], 1)
        else:
            bond_type = 1
        token_bonds[token_a, token_b] = 1
        token_bonds[token_b, token_a] = 1
        type_bonds[token_a, token_b] = bond_type
        type_bonds[token_b, token_a] = bond_type
    return token_bonds, type_bonds


def _msa_features(
    msa: MSA | MsaCollection | None,
    resolved_bindings: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> dict[str, Tensor]:
    residue_types = [cast(int, token["residue_type"]) for token in tokens]
    if msa is None:
        rows = torch.tensor([residue_types])
        deletion_counts = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        columns = tuple(value for value, _, _ in resolved_bindings)
        if isinstance(msa, MSA):
            plan_rows: list[tuple[tuple[int, int, bool], ...]] = [
                ((0, row_index, False),) for row_index in range(len(msa.unpaired_sequences))
            ]
        else:
            plan_rows = [
                tuple(
                    (
                        column_index,
                        int(
                            msa.row_pairing[
                                plan_index,
                                column_index,
                            ]
                        ),
                        bool(is_paired),
                    )
                    for column_index, is_paired in enumerate(msa.row_is_paired[plan_index])
                    if msa.row_pairing[
                        plan_index,
                        column_index,
                    ]
                    != -1
                )
                for plan_index in range(msa.row_pairing.shape[0])
            ]
        row_values: list[Tensor] = []
        deletion_values: list[Tensor] = []
        paired_values: list[Tensor] = []
        for selections in plan_rows:
            row = torch.full(
                (len(tokens),),
                _MSA_GAP,
                dtype=torch.long,
            )
            deletion_row = torch.zeros(len(tokens))
            paired_row = torch.zeros(len(tokens))
            taxonomic_pair = (
                isinstance(msa, MsaCollection)
                and len(selections) > 1
                and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                (
                    _,
                    target_instance_index,
                    entity_index,
                ) = resolved_bindings[column_index]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if (
                        token["entity_index"] != entity_index
                        or (target_instance_index is not None and token["instance_index"] != target_instance_index)
                        or token["monomer_index"] is None
                    ):
                        continue
                    monomer_index = cast(
                        int,
                        token["monomer_index"],
                    )
                    polymer_type = cast(
                        str | None,
                        token["polymer_type"],
                    )
                    row[token_index] = msa_residue_index(
                        aligned[monomer_index],
                        polymer_type,
                    )
                    deletion_row[token_index] = float(deletion_source[monomer_index])
                    paired_row[token_index] = float(taxonomic_pair)
            row_values.append(row)
            deletion_values.append(deletion_row)
            paired_values.append(paired_row)
        rows = torch.stack(row_values)
        deletion_counts = torch.stack(deletion_values)
        paired = torch.stack(paired_values)
        rows[0] = torch.tensor(residue_types)
        paired[0] = 1
    has_deletion = deletion_counts > 0
    deletion_value = (math.pi / 2) * torch.atan(deletion_counts / 3)
    return {
        "msa_tokens": rows,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_paired": paired,
        "msa_mask": torch.ones_like(rows, dtype=torch.bool),
        "msa_profile": torch.nn.functional.one_hot(
            rows,
            MSA_VOCAB_SIZE,
        )
        .float()
        .mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _reference_coordinates(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> Tensor:
    index = torch.tensor(
        atom_indices,
        device=system.atom_symbol.device,
    )
    coordinates = system.reference_coordinates[index].detach().cpu().clone()
    reference_mask = system.reference_coordinate_mask[index].detach().cpu()
    if bool(reference_mask.all()):
        return coordinates

    atom_component = _atom_component_index(system)
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    model_components = atom_component[index].detach().cpu().tolist()
    component_instances = component_instance.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    grouped: dict[int, list[int]] = {}
    for model_atom_index, component_index in enumerate(model_components):
        grouped.setdefault(component_index, []).append(model_atom_index)

    generated: dict[int, dict[int, tuple[float, float, float]]] = {}
    for component_index, model_atom_indices in grouped.items():
        group_mask = reference_mask[model_atom_indices]
        if bool(group_mask.all()):
            continue
        missing_atoms = [
            atom_indices[model_atom_index]
            for model_atom_index in model_atom_indices
            if not bool(reference_mask[model_atom_index])
        ]
        instance_index = component_instances[component_index]
        entity_index = instance_entities[instance_index]
        if entity_types[entity_index] != ENTITY_MOLECULE:
            raise ValueError(f"Boltz-2 requires reference coordinates for canonical atoms {missing_atoms}")
        if bool(group_mask.any()):
            component_start = int(system.instance_component_ptr[instance_index])
            reference = (system.instance_id(instance_index), component_index - component_start)
            raise ValueError(
                "Boltz-2 cannot combine generated coordinates with partial reference "
                f"coordinates for component {reference!r}"
            )
        if instance_index not in generated:
            generated[instance_index] = _molecule_reference_coordinates(
                system,
                instance_index,
            )
        for model_atom_index in model_atom_indices:
            canonical_atom_index = atom_indices[model_atom_index]
            coordinates[model_atom_index] = coordinates.new_tensor(generated[instance_index][canonical_atom_index])
    return coordinates


def _molecule_reference_coordinates(
    system: MolecularSystem,
    instance_index: int,
) -> dict[int, tuple[float, float, float]]:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    component_start = int(system.instance_component_ptr[instance_index])
    component_stop = int(system.instance_component_ptr[instance_index + 1])
    atom_start = int(system.component_atom_ptr[component_start])
    atom_stop = int(system.component_atom_ptr[component_stop])
    atom_indices = tuple(range(atom_start, atom_stop))
    if not bool(system.component_topology_complete[component_start:component_stop].all()):
        raise ValueError("Boltz-2 cannot generate reference coordinates from unknown molecular connectivity")
    edge_indices = (
        (
            (system.edge_index[0] >= atom_start)
            & (system.edge_index[1] < atom_stop)
            & (system.bond_role <= BOND_ROLE_ENTITY)
        )
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    rdkit_molecule = Chem.AddHs(
        _canonical_rdkit_molecule(
            system,
            atom_indices,
            edge_indices,
        )
    )
    options: Any = rdDistGeom.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = 42
    conformer_id = rdDistGeom.EmbedMolecule(rdkit_molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = rdDistGeom.EmbedMolecule(rdkit_molecule, options)
    rdForceFieldHelpers.UFFOptimizeMolecule(rdkit_molecule, confId=conformer_id, maxIters=1000)
    conformer = rdkit_molecule.GetConformer(conformer_id)
    coordinates: dict[int, tuple[float, float, float]] = {}
    for local_index, atom_index in enumerate(atom_indices):
        position = conformer.GetAtomPosition(local_index)
        coordinates[atom_index] = (position.x, position.y, position.z)
    return coordinates


def _augment_reference_positions(
    positions: Tensor,
    space_ids: Tensor,
    resolved: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> None:
    r"""
    Randomly rotate and translate each resolved reference conformer.

    Random values are drawn from ``generator``; ``None`` uses the ambient PyTorch
    RNG.
    """

    if not len(space_ids):
        return
    active_space_ids = torch.unique(space_ids[resolved & (space_ids >= 0)], sorted=True)
    num_active_spaces = active_space_ids.numel()
    if not num_active_spaces:
        return

    random_device = positions.device if generator is None else generator.device
    random_values = torch.randn(
        num_active_spaces,
        7,
        dtype=positions.dtype,
        device=random_device,
        generator=generator,
    ).to(positions.device)
    quaternions = random_values[:, :4]
    norm = torch.sqrt((quaternions * quaternions).sum(dim=-1, keepdim=True))
    quaternions = quaternions / torch.copysign(norm, quaternions[:, :1])
    real, i, j, k = quaternions.unbind(dim=-1)
    two = 2.0 / (quaternions * quaternions).sum(dim=-1)
    rotations = torch.stack(
        (
            1 - two * (j * j + k * k),
            two * (i * j - k * real),
            two * (i * k + j * real),
            two * (i * j + k * real),
            1 - two * (i * i + k * k),
            two * (j * k - i * real),
            two * (i * k - j * real),
            two * (j * k + i * real),
            1 - two * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    translations = random_values[:, 4:]

    for space_id, rotation, translation in zip(active_space_ids, rotations, translations):
        included = space_ids == space_id
        mask = resolved[included]
        values = positions[included]
        center = (values * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
        positions[included] = (values - center) @ rotation + translation


def _model_atom_names(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
) -> list[str]:
    from rdkit.Chem import GetPeriodicTable

    atom_component = _atom_component_index(system)
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            device=system.instance_component_ptr.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    atom_instance = component_instance[atom_component].detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    periodic_table = GetPeriodicTable()
    names: list[str] = []
    counts: dict[tuple[int, str], int] = {}
    for atom_index in atom_indices:
        name = system.symbols[atom_symbols[atom_index]]
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = periodic_table.GetElementSymbol(atomic_numbers[atom_index]).upper()
            key = (atom_instance[atom_index], element)
            counts[key] = counts.get(key, 0) + 1
            name = f"{element}{counts[key]}"
        if len(name) > 4:
            raise ValueError(f"Boltz-2 atom name cannot be encoded in four characters: {name!r}")
        names.append(name)
    return names


def _affinity_ligand_properties(
    system: MolecularSystem,
    instance_index: int,
) -> tuple[int, float | None]:
    r"""
    Return the selected ligand heavy-atom count and reliable graph MW.
    """

    from rdkit.Chem import Descriptors

    component_start = int(system.instance_component_ptr[instance_index])
    component_stop = int(system.instance_component_ptr[instance_index + 1])
    atom_start = int(system.component_atom_ptr[component_start])
    atom_stop = int(system.component_atom_ptr[component_stop])
    atom_indices = tuple(
        atom_index for atom_index in range(atom_start, atom_stop) if int(system.atomic_number[atom_index]) != 1
    )
    if not bool(system.component_topology_complete[component_start:component_stop].all()):
        return len(atom_indices), None
    edge_indices = (
        (
            (system.edge_index[0] >= atom_start)
            & (system.edge_index[1] < atom_stop)
            & (system.bond_role <= BOND_ROLE_ENTITY)
        )
        .nonzero(as_tuple=False)
        .flatten()
    )
    if edge_indices.numel() and not bool(system.bond_order_known[edge_indices].all()):
        return len(atom_indices), None
    molecule = _canonical_rdkit_molecule(
        system,
        tuple(range(atom_start, atom_stop)),
        edge_indices.detach().cpu().tolist(),
    )
    return len(atom_indices), float(Descriptors.MolWt(molecule))  # type: ignore[attr-defined]


def _canonical_rdkit_molecule(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
    edge_indices: Sequence[int],
) -> Mol:
    r"""
    Project one family-owned canonical graph scope into RDKit.
    """

    from rdkit import Chem

    editable = Chem.RWMol()
    selected_atoms = torch.tensor(
        atom_indices,
        device=system.atom_symbol.device,
    )
    atomic_numbers = system.atomic_number[selected_atoms].detach().cpu().tolist()
    isotope = system.isotope[selected_atoms].detach().cpu().tolist()
    isotope_known = system.isotope_known[selected_atoms].detach().cpu().tolist()
    formal_charge = system.formal_charge[selected_atoms].detach().cpu().tolist()
    formal_charge_known = system.formal_charge_known[selected_atoms].detach().cpu().tolist()
    radical_electrons = system.radical_electrons[selected_atoms].detach().cpu().tolist()
    radical_electrons_known = system.radical_electrons_known[selected_atoms].detach().cpu().tolist()
    explicit_hydrogens = system.explicit_hydrogens[selected_atoms].detach().cpu().tolist()
    explicit_hydrogens_known = system.explicit_hydrogens_known[selected_atoms].detach().cpu().tolist()
    implicit_hydrogens_allowed = system.implicit_hydrogens_allowed[selected_atoms].detach().cpu().tolist()
    implicit_hydrogens_allowed_known = system.implicit_hydrogens_allowed_known[selected_atoms].detach().cpu().tolist()
    atom_aromatic = system.atom_aromatic[selected_atoms].detach().cpu().tolist()
    atom_aromatic_known = system.atom_aromatic_known[selected_atoms].detach().cpu().tolist()
    atom_stereo_values = system.atom_stereo[selected_atoms].detach().cpu().tolist()
    canonical_to_local: dict[int, int] = {}
    atom_stereo: dict[int, str] = {}
    for source_index, atom_index in enumerate(atom_indices):
        rdkit_atom = Chem.Atom(atomic_numbers[source_index])
        if isotope_known[source_index]:
            rdkit_atom.SetIsotope(isotope[source_index])
        if formal_charge_known[source_index]:
            rdkit_atom.SetFormalCharge(formal_charge[source_index])
        if radical_electrons_known[source_index]:
            rdkit_atom.SetNumRadicalElectrons(radical_electrons[source_index])
        if explicit_hydrogens_known[source_index]:
            rdkit_atom.SetNumExplicitHs(explicit_hydrogens[source_index])
        if implicit_hydrogens_allowed_known[source_index]:
            rdkit_atom.SetNoImplicit(not implicit_hydrogens_allowed[source_index])
        if atom_aromatic_known[source_index]:
            rdkit_atom.SetIsAromatic(atom_aromatic[source_index])
        local_index = editable.AddAtom(rdkit_atom)
        canonical_to_local[atom_index] = local_index
        stereo_index = atom_stereo_values[source_index]
        if stereo_index >= 0:
            atom_stereo[local_index] = system.symbols[stereo_index].upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    selected_edges = torch.tensor(
        edge_indices,
        dtype=torch.long,
        device=system.edge_index.device,
    )
    edge_rows = system.edge_index[:, selected_edges].detach().cpu().T.tolist()
    bond_order = system.bond_order[selected_edges].detach().cpu().tolist()
    bond_order_known = system.bond_order_known[selected_edges].detach().cpu().tolist()
    bond_aromatic = system.bond_aromatic[selected_edges].detach().cpu().tolist()
    bond_aromatic_known = system.bond_aromatic_known[selected_edges].detach().cpu().tolist()
    bond_kind = system.bond_kind[selected_edges].detach().cpu().tolist()
    bond_direction = system.bond_direction[selected_edges].detach().cpu().tolist()
    bond_stereo_values = system.bond_stereo[selected_edges].detach().cpu().tolist()
    bond_stereo_index = system.bond_stereo_index[:, selected_edges].detach().cpu().T.tolist()
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for source_index, (first, second) in enumerate(edge_rows):
        first_local = canonical_to_local[first]
        second_local = canonical_to_local[second]
        if bond_kind[source_index] == BOND_KIND_DATIVE:
            if bond_direction[source_index] == BOND_DIRECTION_ENDPOINT_0_DONOR:
                donor, acceptor = first_local, second_local
            elif bond_direction[source_index] == BOND_DIRECTION_ENDPOINT_1_DONOR:
                donor, acceptor = second_local, first_local
            else:
                raise ValueError("Boltz-2 RDKit geometry requires a donor direction for dative bonds")
            editable.AddBond(donor, acceptor, Chem.BondType.DATIVE)
        else:
            if not bond_order_known[source_index]:
                raise ValueError("Boltz-2 RDKit geometry requires known bond order")
            editable.AddBond(first_local, second_local, bond_types[bond_order[source_index]])
        bond_index = editable.GetNumBonds() - 1
        rdkit_bond = editable.GetBondWithIdx(bond_index)
        if bond_aromatic_known[source_index]:
            rdkit_bond.SetIsAromatic(bond_aromatic[source_index])
        stereo_index = bond_stereo_values[source_index]
        if stereo_index >= 0:
            first_stereo, second_stereo = bond_stereo_index[source_index]
            stereo_atoms = (
                (
                    canonical_to_local[first_stereo],
                    canonical_to_local[second_stereo],
                )
                if first_stereo >= 0 and second_stereo >= 0
                else None
            )
            bond_stereo.append(
                (
                    bond_index,
                    system.symbols[stereo_index],
                    stereo_atoms,
                )
            )

    value = editable.GetMol()
    Chem.SanitizeMol(value)
    for index in atom_stereo:
        value.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
    for index, desired in atom_stereo.items():
        atom = value.GetAtomWithIdx(index)
        assigned = atom.GetProp("_CIPCode").upper() if atom.HasProp("_CIPCode") else ""
        if assigned != desired:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)

    stereo_types = {
        "E": Chem.BondStereo.STEREOE,
        "TRANS": Chem.BondStereo.STEREOE,
        "Z": Chem.BondStereo.STEREOZ,
        "CIS": Chem.BondStereo.STEREOZ,
    }
    for index, stereo, stereo_atoms in bond_stereo:
        bond = value.GetBondWithIdx(index)
        if stereo_atoms is None:
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            begin_neighbors = [
                neighbor for neighbor in value.GetAtomWithIdx(begin).GetNeighbors() if neighbor.GetIdx() != end
            ]
            end_neighbors = [
                neighbor for neighbor in value.GetAtomWithIdx(end).GetNeighbors() if neighbor.GetIdx() != begin
            ]
            stereo_atoms = (
                max(begin_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
                max(end_neighbors, key=lambda atom: int(atom.GetProp("_CIPRank"))).GetIdx(),
            )
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo_types[stereo.upper()])
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
    return value


def _cyclic_periods(system: MolecularSystem) -> list[int]:
    atom_component = _atom_component_index(system).detach().cpu().tolist()
    edge_rows = system.edge_index.detach().cpu().T.tolist()
    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    periods: list[int] = []
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        component_count = component_stop - component_start
        if entity_types[entity_index] == ENTITY_MOLECULE or component_count <= 2:
            periods.append(0)
            continue
        cyclic = any(
            {
                atom_component[first],
                atom_component[second],
            }
            == {component_start, component_stop - 1}
            for first, second in edge_rows
        )
        periods.append(component_count if cyclic else 0)
    return periods


def _component_polymer_type(
    entity_type: int,
    component_id: str,
) -> str | None:
    polymer_type = _POLYMER_TYPE.get(entity_type)
    if polymer_type is not None:
        return polymer_type
    if entity_type == ENTITY_DNA_RNA_HYBRID:
        if component_id in _STANDARD_COMPONENTS["rna"]:
            return "rna"
        if component_id in _STANDARD_COMPONENTS["dna"]:
            return "dna"
    return None
