r"""
Input processing for BoltzGen structure design and inverse folding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Endpoint, Hotspot
from ...data.evidence import (
    MSA,
    MSA_GAP_INDEX,
    MSA_INDEX,
    MSA_VOCAB_SIZE,
    MsaCollection,
    Template,
    msa_bindings,
    msa_component_index,
    msa_residue_index,
    resolve_msa_bindings,
)
from ...data.molecular import (
    BOND_DIRECTION_ENDPOINT_0_DONOR,
    BOND_DIRECTION_ENDPOINT_1_DONOR,
    BOND_DIRECTION_UNDIRECTED,
    BOND_KIND_COVALENT,
    BOND_KIND_DATIVE,
    BOND_ROLE_ASSEMBLY,
    BOND_ROLE_COMPONENT,
    BOND_ROLE_ENTITY,
    ENTITY_DNA,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    component_boundary_atoms,
    disjoint_union,
    form_bond,
)
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    PlacementValue,
    SymbolicAtomSite,
    SymbolicResidueSite,
)
from ...data.placement import instance_of as _placement_instance_of
from ...data.placement import resolve
from ...data.structure import Structure
from ...processor import Processor
from .model import (
    CANONICAL_TOKENS,
    NUM_CANONICAL_TOKENS,
    BoltzGenBinding,
    BoltzGenContact,
    BoltzGenDesignMask,
    BoltzGenInput,
    BoltzGenInverseFoldingInput,
    BoltzGenResidueConstraint,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenStructureTarget,
    BoltzGenSymmetry,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


# Contact-conditioning one-hot classes consumed by the BoltzGen model:
# 0 UNSPECIFIED, 1 UNSELECTED, 2 POCKET>BINDER, 3 BINDER>POCKET, 4 CONTACT.
_CONTACT_CLASSES = 5
_CONTACT_KIND_TO_CLASS = {"pocket>binder": 2, "binder>pocket": 3, "contact": 4}

# Conditioning ids consumed by the BoltzGen input embeddings.
_BINDING_UNSPECIFIED = 0
_BINDING_BINDING = 1
_BINDING_NOT_BINDING = 2
_SS_UNSPECIFIED = 3  # num_ss_types == 4; class 3 taken as the masked/unspecified label.
_SS_LABEL_TO_ID = {"H": 0, "E": 1, "L": 2}


class BoltzGenProcessor(Processor):
    r"""
    Convert supported structure-design inputs into `BoltzGenInput`.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        generator: torch.Generator | None = None,
    ) -> BoltzGenInput:
        r"""
        Build model inputs for one structure-design request.

        Args:
            input: Optional target or context system or structure. ``None`` means a
                fully generated assembly defined by ``placement`` and ``total_len``.
            placement: Optional topology of existing and generated molecular segments.
            total_len: Total token budget used to realize generated segments. With an
                existing input and no generated placement, appends one generated protein
                chain of this length.
            conditions: BoltzGen conditioning records.
            msa: Optional canonical MSA evidence.
            templates: Optional structural templates.
            generator: Random generator used for reference-conformer augmentation.

        Returns:
            A [`BoltzGenInput`][] with one explicit model batch and aligned
            token, atom, evidence, condition, and canonical-graph axes.
        """

        structure, resolved_conditions, generated_references = self._plan(
            input,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
        )
        processed, _ = self._lower(
            structure,
            resolved_conditions,
            generated_references,
            msa,
            templates=templates,
            generator=generator,
        )
        return processed

    def process_training(
        self,
        input: Structure,
        *,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInput, BoltzGenStructureTarget]:
        r"""
        Lower one complete structure into masked Model input and separate truth.

        Structure training redesigns residues already present in ``input``; generated
        placements have no target coordinates and therefore belong to inference.
        """

        structure, resolved_conditions, generated_references = self._plan(
            input,
            placement=None,
            total_len=None,
            conditions=conditions,
        )
        processed, target = self._lower(
            structure,
            resolved_conditions,
            generated_references,
            msa,
            templates=templates,
            generator=generator,
        )
        return processed, target

    def _plan(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: PlacementSpec | None,
        total_len: int | None,
        conditions: Sequence[BoltzGenSchema],
    ) -> tuple[
        Structure,
        tuple[BoltzGenSchema, ...],
        tuple[ComponentReference, ...],
    ]:
        r"""
        Materialize the final design system and placement identities once.
        """

        condition_values = tuple(conditions)
        covalent_bonds = tuple(condition for condition in condition_values if isinstance(condition, SymbolicBond))
        spatial_conditions: tuple[BoltzGenSchema, ...] = tuple(
            condition for condition in condition_values if not isinstance(condition, SymbolicBond)
        )
        system, structure = _normalize_structure_input(input)

        placement_map: PlacementMap = {}
        instance_of: dict[str, str] = {}
        generated_residues: frozenset[ComponentReference] = frozenset()
        if system is None:
            # A fully generated assembly uses explicit placements or one default chain.
            if placement is None and total_len is None:
                raise ValueError("a fully-generated BoltzGen assembly requires placement or a total_len length budget")
            assembly_placements = placement if placement is not None else _default_assembly(total_len)
            system, placement_map, instance_of = _materialize_assembly(assembly_placements, total_len)
            generated_residues = frozenset(
                reference
                for group in placement_map.values()
                for occurrences in group.values()
                for reference in occurrences
            )
            structure = _empty_structure(system)
        elif placement is not None or total_len is not None:
            extension_placements = placement
            assert structure is not None
            structure, placement_map, instance_of, generated_residues = _materialize_extension(
                structure, extension_placements, total_len
            )
            system = structure.system
        else:
            assert system is not None and structure is not None

        generated_references = tuple(sorted(generated_residues, key=system.component_index))
        if generated_references:
            spatial_conditions = (
                *spatial_conditions,
                BoltzGenDesignMask(generated_references),
            )
        if covalent_bonds:
            structure = _materialize_covalent_bonds(
                structure,
                covalent_bonds,
                placement_map,
                instance_of,
            )
        return (
            structure,
            _resolve_conditions(
                spatial_conditions,
                placement_map,
                instance_of,
            ),
            generated_references,
        )

    def _lower(
        self,
        structure: Structure,
        conditions: tuple[BoltzGenSchema, ...],
        generated_references: tuple[ComponentReference, ...],
        msa: MSA | MsaCollection | None,
        *,
        templates: Sequence[Template] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInput, BoltzGenStructureTarget]:
        r"""
        Validate canonical evidence and lower one materialized plan to tensors.
        """

        system = structure.system
        if not system.instance_symbol.numel():
            raise ValueError("BoltzGen requires at least one molecular instance")
        msa = _validate_msa(msa, system)
        generated_components = frozenset(system.component_index(reference) for reference in generated_references)
        tokens, _ = _build_tokens(system, generated_components=generated_components)
        token_count = len(tokens)
        system_atom_to_token = {
            atom_index: token_index
            for token_index, token in enumerate(tokens)
            for atom_index in cast(tuple[int, ...], token["atoms"])
        }
        token_index_by_component = {
            cast(int, token["component_index"]): token_index
            for token_index, token in enumerate(tokens)
            if token["monomer_index"] is not None
        }
        token_bonds = _token_bond_indicator(system, system_atom_to_token, token_count)
        contact_conditioning, contact_threshold = _contact_channels(
            system,
            conditions,
            token_index_by_component,
            system_atom_to_token,
            token_count,
        )
        design_mask = _design_mask(
            system,
            conditions,
            token_index_by_component,
            token_count,
        )
        _validate_structure_design_rows(
            system,
            tokens,
            design_mask,
            generated_components,
        )
        (
            atom_values,
            model_atom_index,
            coordinate_atom_index,
            system_to_model_atom,
        ) = _build_structure_design_atom_features(
            system,
            tokens,
            structure,
            design_mask,
            generated_components=generated_components,
            generator=generator,
        )
        token_center_system_atom = [cast(int, token["center"]) for token in tokens]
        token_center_atom_index = torch.tensor(
            [system_to_model_atom[atom_index] for atom_index in token_center_system_atom],
            dtype=torch.long,
        )
        token_to_bb4 = torch.stack(
            [
                torch.tensor(
                    [system_to_model_atom[atom_index] for atom_index in cast(tuple[int, ...], token["bb4"])],
                    dtype=torch.long,
                )
                for token in tokens
            ]
        )
        frame_atom_index = _frame_atom_indices(tokens, system_to_model_atom)
        msa_values = _mask_structure_design_msa(
            _msa_features(msa, tokens, system),
            design_mask,
        )
        residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
        target_residue_type = torch.nn.functional.one_hot(residue_types, 33).float()
        residue_type_features = target_residue_type.clone()
        residue_type_features[design_mask] = 0
        residue_type_features[design_mask, _UNKNOWN_RESIDUE_TYPES["protein"]] = 1
        molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
        cyclic_period = torch.tensor(
            [_cyclic_period(system, cast(int, token["instance_index"])) for token in tokens],
            dtype=torch.float32,
        )
        chain_design_mask = _chain_design_mask(
            design_mask,
            torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
            token_bonds,
        )
        binding_type = _binding_ids(
            system,
            conditions,
            token_index_by_component,
            system_atom_to_token,
            token_count,
        )
        ss_type = _ss_ids(system, conditions, token_index_by_component, token_count)
        symmetry_index = _symmetry_index(
            system,
            conditions,
            token_index_by_component,
            tokens,
        )
        # Structure groups affect pairwise geometry through ``token_distance_mask``;
        # atom reference-space ids remain per-residue.
        structure_group = _structure_group_ids(
            system,
            conditions,
            token_index_by_component,
            token_count,
            design_mask,
        )
        token_distance_mask = _token_distance_mask(structure_group)
        # ``None`` leaves template fields unset so the trunk skips the template embedder.
        template_values = _template_channels(tuple(templates), tokens, token_count)

        tensor_values: dict[str, Tensor] = {
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
            "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
            "entity_index": torch.tensor([cast(int, token["entity_index"]) for token in tokens]),
            "symmetry_index": symmetry_index,
            "token_index": torch.arange(token_count),
            "atom_features": atom_values["atom_features"],
            "atom_to_token": atom_values["atom_to_token"],
            "atom_mask": atom_values["atom_mask"],
            "reference_positions": atom_values["reference_positions"],
            "reference_space_id": atom_values["reference_space_id"],
            "atom_token_index": atom_values["atom_token_index"],
            "token_center_atom_index": token_center_atom_index,
            "residue_type_features": residue_type_features,
            "msa_profile": msa_values.pop("msa_profile"),
            "deletion_mean": msa_values.pop("deletion_mean"),
            "token_bonds": token_bonds,
            "type_bonds": torch.zeros(token_count, token_count, dtype=torch.long),
            "contact_conditioning": contact_conditioning,
            "contact_threshold": contact_threshold,
            "center_coordinates": atom_values["center_coordinates"],
            "token_to_bb4_atoms": token_to_bb4,
            "atom_coordinates": atom_values["atom_coordinates"],
            "token_distance_mask": token_distance_mask,
            "frame_atom_index": frame_atom_index,
            "molecule_type": molecule_types,
            "cyclic_period": cyclic_period,
            "method_feature": torch.zeros(token_count, dtype=torch.long),
            "modified": torch.zeros(token_count, dtype=torch.long),
            "binding_type": binding_type,
            "design_mask": design_mask,
            "chain_design_mask": chain_design_mask,
            "ss_type": ss_type,
            **msa_values,
            **(template_values or {}),
        }
        placed = {
            name: (
                value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(device=self.device)
            )
            for name, value in tensor_values.items()
        }
        bond_atom_index = _inter_component_bond_atom_index(
            system=system,
            system_to_model_atom=system_to_model_atom,
        )
        processed = BoltzGenInput(  # type: ignore[call-arg]
            **{name: value.unsqueeze(0) for name, value in placed.items()},
            system=system,
            atom_index=model_atom_index.to(device=self.device).unsqueeze(0),
            token_component_index=torch.tensor(
                [cast(int, token["component_index"]) for token in tokens],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0),
            token_atom_index=torch.tensor(
                [-1 if token["atom_index"] is None else cast(int, token["atom_index"]) for token in tokens],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0),
            bond_atom_index=bond_atom_index.to(device=self.device).unsqueeze(0),
            bond_mask=torch.ones(
                1,
                bond_atom_index.shape[0],
                dtype=torch.bool,
                device=self.device,
            ),
            batch_size=[],
        )
        truth_values = _truth_pool_features(
            system,
            tokens,
            structure,
            coordinate_atom_index,
            system_to_model_atom,
        )
        # The BoltzGen B-factor head reads the trunk single representation, so its target is
        # the deposited B factor of each token's representative atom on the token axis.
        target_values: dict[str, Tensor] = {
            **truth_values,
            "residue_type": target_residue_type,
            "b_factor": structure.b_factor.detach().cpu()[token_center_system_atom],
            "b_factor_mask": structure.b_factor_mask.detach().cpu()[token_center_system_atom],
        }
        placed_target = {
            name: (
                value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(device=self.device)
            )
            for name, value in target_values.items()
        }
        target = BoltzGenStructureTarget(  # type: ignore[call-arg]
            coordinates=placed_target["coordinates"].unsqueeze(0).unsqueeze(0),
            coordinate_mask=placed_target["coordinate_mask"].unsqueeze(0),
            coordinate_atom_index=placed_target["coordinate_atom_index"].unsqueeze(0),
            chain_span=placed_target["chain_span"].unsqueeze(0),
            chain_equivalence_index=placed_target["chain_equivalence_index"].unsqueeze(0),
            symmetry_atom_index=placed_target["symmetry_atom_index"].unsqueeze(0),
            symmetry_permutation=placed_target["symmetry_permutation"].unsqueeze(0),
            residue_type=placed_target["residue_type"].unsqueeze(0),
            b_factor=placed_target["b_factor"].unsqueeze(0),
            b_factor_mask=placed_target["b_factor_mask"].unsqueeze(0),
            batch_size=[1],
        )
        return processed, target


class BoltzGenInverseFoldingProcessor(Processor):
    r"""
    Convert supported inverse-folding inputs into `BoltzGenInverseFoldingInput`.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __call__(
        self,
        input: Structure,
        *,
        design: Sequence[ComponentReference] = (),
        global_omit_tokens: Sequence[str] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MsaCollection | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenInverseFoldingInput:
        r"""
        Build inverse-folding inputs for one fixed structure.

        Args:
            input: Fixed canonical structure.
            design: Residue occurrences whose identities may change.
            global_omit_tokens: Three-letter canonical amino-acid component IDs
                excluded from every designed position.
            aa_constraints: Per-residue amino-acid constraints.
            symmetry: Residue groups sharing one sequence identity.
            msa: Optional canonical MSA evidence.
            generator: Random generator used for reference-conformer augmentation.

        Returns:
            A [`BoltzGenInverseFoldingInput`][] with batch size one and aligned
            token, atom, design-mask, amino-acid-constraint, and symmetry axes.
        """

        system = input.system
        if not system.instance_symbol.numel():
            raise ValueError("BoltzGen inverse folding requires at least one molecular instance")

        normalized_msa = _validate_msa(msa, system)
        processed, _ = self._prepare(
            input,
            msa=normalized_msa,
            design=tuple(design),
            global_omit_tokens=tuple(global_omit_tokens),
            aa_constraints=tuple(aa_constraints),
            symmetry=tuple(tuple(unit) for unit in symmetry),
            generator=generator,
        )
        return processed

    def process_training(
        self,
        input: Structure,
        *,
        design: Sequence[ComponentReference] = (),
        global_omit_tokens: Sequence[str] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MsaCollection | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInverseFoldingInput, Tensor]:
        r"""
        Lower one inverse-folding example and its unmasked residue target.

        The returned input masks every design position to the released unknown
        protein token. The aligned target remains separate for teacher forcing
        and the residue objective.
        """

        system = input.system
        if not system.instance_symbol.numel():
            raise ValueError("BoltzGen inverse folding requires at least one molecular instance")
        normalized_msa = _validate_msa(msa, system)
        return self._prepare(
            input,
            msa=normalized_msa,
            design=tuple(design),
            global_omit_tokens=tuple(global_omit_tokens),
            aa_constraints=tuple(aa_constraints),
            symmetry=tuple(tuple(unit) for unit in symmetry),
            generator=generator,
        )

    def _prepare(
        self,
        structure: Structure,
        *,
        msa: MSA | MsaCollection | None,
        design: tuple[ComponentReference, ...],
        global_omit_tokens: tuple[str, ...],
        aa_constraints: tuple[BoltzGenResidueConstraint, ...],
        symmetry: tuple[tuple[ComponentReference, ...], ...],
        generator: torch.Generator | None,
    ) -> tuple[BoltzGenInverseFoldingInput, Tensor]:
        system = structure.system
        # Generated ``UNK`` protein residues use backbone-only GLY reference coordinates.
        generated_components = frozenset(system.component_index(reference) for reference in design)
        tokens, atom_indices = _build_tokens(
            system,
            generated_components=generated_components,
        )
        token_count = len(tokens)
        token_index_by_component = {
            cast(int, token["component_index"]): token_index
            for token_index, token in enumerate(tokens)
            if token["monomer_index"] is not None
        }
        molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
        for reference in design:
            _require_protein_reference(
                system,
                reference,
                token_index_by_component,
                molecule_types,
                "design",
            )
        for constraint in aa_constraints:
            _require_protein_reference(
                system,
                constraint.position,
                token_index_by_component,
                molecule_types,
                "amino-acid constraint",
            )
        for group in symmetry:
            for reference in group:
                _require_protein_reference(
                    system,
                    reference,
                    token_index_by_component,
                    molecule_types,
                    "sequence-symmetry group",
                )

        (
            atom_features,
            reference_positions,
            reference_space_id,
            atom_mask,
            atom_to_token,
            atom_token_index,
            atom_coordinates,
            resolved,
            center_coordinates,
            model_atom_index,
            system_to_model_atom,
        ) = _build_atom_features(
            system,
            tokens,
            atom_indices,
            structure,
            generated_components=generated_components,
            generator=generator,
        )
        system_atom_to_token = {
            atom_index: token_index
            for token_index, token in enumerate(tokens)
            for atom_index in cast(tuple[int, ...], token["atoms"])
        }
        token_bonds = _token_bond_indicator(system, system_atom_to_token, token_count)
        msa_values = _msa_features(msa, tokens, system)
        residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
        cyclic_period = torch.tensor(
            [_cyclic_period(system, cast(int, token["instance_index"])) for token in tokens],
            dtype=torch.float32,
        )
        token_to_bb4 = torch.stack(
            [
                torch.tensor(
                    [system_to_model_atom[atom_index] for atom_index in cast(tuple[int, ...], token["bb4"])],
                    dtype=torch.long,
                )
                for token in tokens
            ]
        )
        design_mask = _design_mask_from_positions(
            system,
            design,
            token_index_by_component,
            token_count,
        )
        designed_atom_mask = atom_mask & design_mask[atom_to_token]
        reference_positions[designed_atom_mask] = 0
        atom_features[designed_atom_mask, :3] = 0
        target_residue_type = torch.nn.functional.one_hot(residue_types, 33).float()
        residue_type_features = target_residue_type.clone()
        residue_type_features[design_mask] = 0
        residue_type_features[design_mask, _UNKNOWN_RESIDUE_TYPES["protein"]] = 1
        token_center_atom_index = torch.tensor(
            [system_to_model_atom[cast(int, token["center"])] for token in tokens],
            dtype=torch.long,
        )
        token_resolved_mask = resolved[token_center_atom_index]
        aa_constraint_mask = _aa_constraint_mask(
            global_omit_tokens,
            aa_constraints,
            system,
            token_index_by_component,
            token_count,
        )
        symmetric_group = _sequence_symmetry_index(
            symmetry,
            system,
            token_index_by_component,
            token_count,
        )

        tensor_values: dict[str, Tensor] = {
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
            "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
            "atom_features": atom_features,
            "atom_to_token": atom_to_token,
            "atom_mask": atom_mask,
            "reference_positions": reference_positions,
            "reference_space_id": reference_space_id,
            "residue_type_features": residue_type_features,
            "msa_profile": msa_values["msa_profile"],
            "deletion_mean": msa_values["deletion_mean"],
            "token_bonds": token_bonds,
            "type_bonds": torch.zeros(token_count, token_count, dtype=torch.long),
            "molecule_type": molecule_types,
            "center_coordinates": center_coordinates,
            "token_to_bb4_atoms": token_to_bb4,
            "atom_coordinates": atom_coordinates,
            "token_resolved_mask": token_resolved_mask,
            "cyclic_period": cyclic_period,
            "method_feature": torch.zeros(token_count, dtype=torch.long),
            "modified": torch.zeros(token_count, dtype=torch.long),
            "design_mask": design_mask,
            "aa_constraint_mask": aa_constraint_mask,
            "symmetric_group": symmetric_group,
        }
        placed = {
            name: (
                value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(device=self.device)
            )
            for name, value in tensor_values.items()
        }
        return (
            BoltzGenInverseFoldingInput(  # type: ignore[call-arg]
                **{name: value.unsqueeze(0) for name, value in placed.items()},
                system=system,
                atom_index=model_atom_index.to(device=self.device).unsqueeze(0),
                token_component_index=torch.tensor(
                    [cast(int, token["component_index"]) for token in tokens],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                token_atom_index=torch.tensor(
                    [-1 if token["atom_index"] is None else cast(int, token["atom_index"]) for token in tokens],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                batch_size=[],
            ),
            target_residue_type.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        )


#: One designed polymer type or fixed source residue in a BoltzGen design mask.
DesignMaskResidue = PolymerType | ComponentReference


def from_design_mask(
    residues: Sequence[DesignMaskResidue],
    *,
    groups: Sequence[str],
    placement_ids: Sequence[str],
) -> PlacementSpec:
    r"""
    Convert BoltzGen residue design specifications to placements.
    """

    placement: PlacementSpec = {}
    seen_placement_ids: set[str] = set()
    run: list[DesignMaskResidue] = []
    run_group: str | None = None
    run_placement_id: str | None = None

    def flush() -> None:
        nonlocal run
        if not run:
            return
        assert run_group is not None and run_placement_id is not None
        if run_placement_id in seen_placement_ids:
            raise ValueError(
                f"placement_id {run_placement_id!r} is non-contiguous: it reappears after another "
                "placement; a placement_id must name one contiguous run"
            )
        seen_placement_ids.add(run_placement_id)
        placement.setdefault(run_group, {})[run_placement_id] = _placement_value_of_run(run_placement_id, run)
        run = []

    for residue, group, placement_id in zip(residues, groups, placement_ids, strict=True):
        if placement_id != run_placement_id or group != run_group:
            flush()
            run_group = group
            run_placement_id = placement_id
        run.append(residue)
    flush()
    return placement


def to_design_mask(
    placement: PlacementSpec,
    placement_map: PlacementMap,
) -> tuple[list[bool], list[str]]:
    r"""
    Recover BoltzGen designedness and declaration groups from realized placements.
    """

    design_mask: list[bool] = []
    row_groups: list[str] = []
    for group, placements in placement.items():
        for placement_id, value in placements.items():
            designed = isinstance(value, GeneratedSegment)
            for _ in placement_map[group][placement_id]:
                design_mask.append(designed)
                row_groups.append(group)
    return design_mask, row_groups


def _placement_value_of_run(placement_id: str, run: list[DesignMaskResidue]) -> PlacementValue:
    designed = isinstance(run[0], str)
    if any(isinstance(residue, str) != designed for residue in run):
        raise ValueError(
            f"placement {placement_id!r} mixes designed and fixed residues; a placement_id must be "
            "either wholly designed (GeneratedSegment) or wholly fixed (ExistingSpan)"
        )

    if designed:
        polymer_type = cast(PolymerType, run[0])
        if any(residue != polymer_type for residue in run):
            raise ValueError(
                f"designed placement {placement_id!r} spans more than one polymer type; a maximal "
                "designed run is one polymer type"
            )
        return GeneratedSegment(polymer_type, len(run), len(run))

    sources = [residue for residue in run if not isinstance(residue, str)]
    instance_id = sources[0][0]
    start = sources[0][1]
    for offset, source in enumerate(sources):
        if source[0] != instance_id or source[1] != start + offset:
            raise ValueError(
                f"fixed placement {placement_id!r} is not one contiguous source interval of a single "
                f"instance: {sources!r}"
            )
    return ExistingSpan(instance_id, start, start + len(run))


# BoltzGen processor implementation helpers.


def _normalize_structure_input(
    input: MolecularSystem | Structure | None,
) -> tuple[MolecularSystem | None, Structure | None]:
    r"""
    Resolve an input into (system, optional Structure).
    """

    if input is None:
        return None, None
    if isinstance(input, Structure):
        return input.system, input
    if isinstance(input, MolecularSystem):
        return input, _empty_structure(input)
    raise TypeError("BoltzGen input must be a MolecularSystem, Structure, or None")


def _component_reference_to_token(
    system: MolecularSystem,
    reference: ComponentReference,
    token_index: dict[int, int],
) -> int:
    r"""
    Resolve one ``(instance_id, component_index)`` reference to its token row.
    """

    component_index = system.component_index(reference)
    if component_index not in token_index:
        raise ValueError(f"condition references residue {reference!r} not present in the lowered system")
    return token_index[component_index]


def _empty_structure(system: MolecularSystem) -> Structure:
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    device = system.atom_symbol.device
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=torch.zeros(atom_count, 3, device=device),
        coordinate_mask=torch.zeros(atom_count, dtype=torch.bool, device=device),
        occupancy=torch.zeros(atom_count, device=device),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool, device=device),
        b_factor=torch.zeros(atom_count, device=device),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool, device=device),
        source_symbols=(),
        residue_id=torch.full(
            (component_count,),
            -1,
            dtype=torch.long,
            device=device,
        ),
        insertion_code=torch.full(
            (component_count,),
            -1,
            dtype=torch.long,
            device=device,
        ),
        standard_component_id=torch.full(
            (component_count,),
            -1,
            dtype=torch.long,
            device=device,
        ),
        batch_size=[],
    )


def _transfer_structure(
    structure: Structure,
    system: MolecularSystem,
    atom_to_new: Tensor,
    component_to_new: Tensor | None = None,
) -> Structure:
    atom_count = system.atom_symbol.numel()
    retained_atom = atom_to_new >= 0

    def atoms(value: Tensor, fill: float | bool = 0) -> Tensor:
        output = value.new_full((atom_count, *value.shape[1:]), fill)
        output[atom_to_new[retained_atom]] = value[retained_atom]
        return output

    if component_to_new is None:
        residue_id = structure.residue_id
        insertion_code = structure.insertion_code
        standard_component_id = structure.standard_component_id
    else:
        component_count = system.component_symbol.numel()
        retained_component = component_to_new >= 0
        residue_id = structure.residue_id.new_full((component_count,), -1)
        insertion_code = structure.insertion_code.new_full((component_count,), -1)
        standard_component_id = structure.standard_component_id.new_full((component_count,), -1)
        residue_id[component_to_new[retained_component]] = structure.residue_id[retained_component]
        insertion_code[component_to_new[retained_component]] = structure.insertion_code[retained_component]
        standard_component_id[component_to_new[retained_component]] = structure.standard_component_id[
            retained_component
        ]

    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=atoms(structure.coordinates),
        coordinate_mask=atoms(structure.coordinate_mask, False),
        occupancy=atoms(structure.occupancy),
        occupancy_mask=atoms(structure.occupancy_mask, False),
        b_factor=atoms(structure.b_factor),
        b_factor_mask=atoms(structure.b_factor_mask, False),
        source_symbols=structure.source_symbols,
        residue_id=residue_id,
        insertion_code=insertion_code,
        standard_component_id=standard_component_id,
        batch_size=[],
    )


# ===========================================================================
# Whole-assembly ``total_len`` realization.
# ===========================================================================

# Placeholder component identities used for generated polymer residues.
_GENERATED_COMPONENT = {"protein": "UNK", "rna": "N", "dna": "DN"}


def _default_assembly(total_len: int | None) -> PlacementSpec:
    r"""
    The degenerate whole-assembly PlacementSpec when only a ``total_len`` budget is given.

    One declaration group ``"assembly"`` with one generated protein placement ``"chain_A"``
    spanning the whole ``total_len`` budget.
    """

    if total_len is None or total_len <= 0:
        raise ValueError("a BoltzGen assembly without a PlacementSpec requires a positive total_len")
    return {"assembly": {"chain_A": GeneratedSegment("protein", total_len, total_len)}}


def _resolve_generated_lengths(placements: PlacementSpec, total_len: int | None) -> dict[str, int]:
    r"""
    Resolve each generated placement within the ``total_len`` budget.

    ``total_len`` constrains the sum of generated lengths. Every placement starts
    at ``min_length`` and receives remaining capacity in declaration order up to
    ``max_length``. ``None`` selects every ``max_length``.
    """

    segments: list[tuple[str, GeneratedSegment]] = []
    for chain in placements.values():
        for placement_id, part in chain.items():
            if not isinstance(part, GeneratedSegment):
                continue
            if part.min_length < 0 or part.max_length < part.min_length:
                raise ValueError(
                    f"placement {placement_id!r} has an invalid length range [{part.min_length}, {part.max_length}]"
                )
            segments.append((placement_id, part))

    if total_len is None:
        return {placement_id: part.max_length for placement_id, part in segments}

    total_min = sum(part.min_length for _, part in segments)
    total_max = sum(part.max_length for _, part in segments)
    if not total_min <= total_len <= total_max:
        raise ValueError(
            f"total_len={total_len} is outside the assembly's generated-length range [{total_min}, {total_max}]"
        )
    lengths = {placement_id: part.min_length for placement_id, part in segments}
    remaining = total_len - total_min
    for placement_id, part in segments:
        added = min(remaining, part.max_length - part.min_length)
        lengths[placement_id] += added
        remaining -= added
    return lengths


def _materialize_assembly(
    placements: PlacementSpec,
    total_len: int | None,
    *,
    reserved_entity_ids: frozenset[str] = frozenset(),
    reserved_instance_ids: frozenset[str] = frozenset(),
) -> tuple[MolecularSystem, PlacementMap, dict[str, str]]:
    r"""
    Realize a fully generated placement specification.

    Every value must be a [`GeneratedSegment`][]. Non-empty segments mint one
    ``chain_{n}`` instance in declaration order; zero-length segments map to an
    empty occurrence tuple. Lengths are resolved by [`_resolve_generated_lengths`][].
    """

    instance_of = _placement_instance_of(placements)  # validates global placement_id uniqueness
    lengths = _resolve_generated_lengths(placements, total_len)

    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    placement_map: PlacementMap = {}
    next_instance = 0
    entity_index = 0
    for group, chain in placements.items():
        realized_group: dict[str, tuple[ComponentReference, ...]] = {}
        for placement_id, part in chain.items():
            if not isinstance(part, GeneratedSegment):
                raise ValueError(
                    "a fully-generated BoltzGen assembly (input=None) accepts only GeneratedSegment "
                    f"placements; placement {placement_id!r} is an ExistingSpan, which requires a "
                    "non-None input system (use the extend-existing path)"
                )
            polymer_type = part.polymer_type
            if polymer_type not in _GENERATED_COMPONENT:
                raise ValueError(
                    f"BoltzGen generates protein/rna/dna placements only; placement {placement_id!r} "
                    f"requests {polymer_type!r}"
                )
            length = lengths[placement_id]
            if length == 0:
                realized_group[placement_id] = ()
                continue
            while True:
                instance_id = f"chain_{next_instance}"
                next_instance += 1
                if instance_id not in reserved_instance_ids:
                    break
            while True:
                entity_id = f"boltzgen_entity_{entity_index}"
                entity_index += 1
                if entity_id not in reserved_entity_ids:
                    break
            component = get_component(_GENERATED_COMPONENT[polymer_type])
            entities[entity_id] = PolymerSequence(polymer_type, (component,) * length)
            instances[instance_id] = entity_id
            realized_group[placement_id] = tuple((instance_id, index) for index in range(length))
        placement_map[group] = realized_group

    if not instances:
        raise ValueError("a BoltzGen assembly must realize at least one non-empty generated placement")
    system = MolecularSystem.from_entities(entities, instances)
    return system, placement_map, instance_of


def _default_extension(total_len: int | None) -> PlacementSpec:
    r"""
    Build the default extension for a ``total_len`` budget.

    The input system is retained and one generated protein chain of length
    ``total_len`` is appended.
    """

    if total_len is None or total_len <= 0:
        raise ValueError("total_len extension of an input system requires a positive total_len")
    return {"assembly": {"chain_A": GeneratedSegment("protein", total_len, total_len)}}


def _materialize_extension(
    structure: Structure,
    placements: PlacementSpec | None,
    total_len: int | None,
) -> tuple[Structure, PlacementMap, dict[str, str], frozenset[ComponentReference]]:
    r"""
    Extend an existing input system with generated rows.

    Existing instances retain their ids, entities, coordinates, and bonds.
    [`ExistingSpan`][] occurrences address those source rows directly.
    [`GeneratedSegment`][] values append fresh non-colliding ``chain_{n}``
    instances in declaration order.

    ``placements is None`` is the ``total_len`` extension: one generated protein chain of length
    ``total_len`` is appended (see [`_default_extension`][]). A ``total_len`` given alongside an
    explicit [`PlacementSpec`][ofoldx.data.placement.PlacementSpec] constrains
    the sum of generated lengths.

    Returns the extended system, the realized ``placement_map`` (generated occurrences under minted
    ``chain_{n}`` ids, ExistingSpan occurrences under reused source ids), the derived
    ``placement_id -> declaration_group`` index, and the frozenset of GENERATED output residues
    (the design positions — an ExistingSpan occurrence is NOT in this set).
    """

    system = structure.system
    placements = _default_extension(total_len) if placements is None else placements
    instance_of = _placement_instance_of(placements)
    source_instances = {system.instance_id(index) for index in range(system.instance_symbol.numel())}
    generated_placements: PlacementSpec = {}
    placement_map: PlacementMap = {}
    for group, placement_values in placements.items():
        generated_values: dict[str, PlacementValue] = {}
        realized_values: dict[str, tuple[ComponentReference, ...]] = {}
        for placement_id, value in placement_values.items():
            if isinstance(value, ExistingSpan):
                instance_index = system.instance_index(value.instance_id)
                entity_index = int(system.instance_entity[instance_index])
                if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
                    raise ValueError(f"ExistingSpan placement {placement_id!r} names a non-polymer instance")
                component_count = int(
                    system.instance_component_ptr[instance_index + 1] - system.instance_component_ptr[instance_index]
                )
                if not 0 <= value.start <= value.stop <= component_count:
                    raise ValueError(
                        f"ExistingSpan placement {placement_id!r} is outside instance {value.instance_id!r}"
                    )
                realized_values[placement_id] = tuple(
                    (value.instance_id, index) for index in range(value.start, value.stop)
                )
            else:
                generated_values[placement_id] = value
        generated_placements[group] = generated_values
        placement_map[group] = realized_values

    generated_system, generated_map, _ = _materialize_assembly(
        generated_placements,
        total_len,
        reserved_entity_ids=frozenset(system.entity_id(index) for index in range(system.entity_symbol.numel())),
        reserved_instance_ids=frozenset(source_instances),
    )
    for group, realized_group in generated_map.items():
        placement_map[group].update(realized_group)
    generated = frozenset(
        reference
        for realized_group in generated_map.values()
        for references in realized_group.values()
        for reference in references
    )
    if not generated:
        raise ValueError(
            "extending an input system must realize at least one non-empty generated placement (nothing to design)"
        )
    extended, component_maps, atom_maps = disjoint_union((system, generated_system))
    return (
        _transfer_structure(
            structure,
            extended,
            atom_maps[0],
            component_maps[0],
        ),
        placement_map,
        instance_of,
        generated,
    )


def _materialize_covalent_bonds(
    structure: Structure,
    covalent_bonds: tuple[SymbolicBond, ...],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> Structure:
    r"""
    Resolve requested [`SymbolicBond`][] links into canonical graph edits.

    Each [`SymbolicBond`][ofoldx.data.bonds.SymbolicBond] has endpoints lowered to realized
    [`AtomSite`][ofoldx.data.molecular.AtomSite] occurrences (an existing ``AtomSite`` passes through;
    a generated [`SymbolicAtomSite`][ofoldx.data.placement.SymbolicAtomSite] resolves through
    ``placement_map`` / ``instance_of``). Each realized edge is applied through
    [`form_bond`][ofoldx.data.molecular.form_bond], and every coordinate-aligned
    Structure field follows the returned old-to-new atom map.
    """

    system = structure.system
    for bond in covalent_bonds:
        realized = bond.resolve(placement_map, instance_of)
        atom_a = system.atom_index(realized.site_a)
        atom_b = system.atom_index(realized.site_b)
        endpoints = torch.tensor(
            sorted((atom_a, atom_b)),
            dtype=torch.long,
            device=system.edge_index.device,
        )
        component_a = system.component_index((realized.site_a.instance_id, realized.site_a.component_index))
        component_b = system.component_index((realized.site_b.instance_id, realized.site_b.component_index))
        existing = torch.nonzero(
            (system.edge_index.T == endpoints).all(dim=-1),
            as_tuple=False,
        ).flatten()
        if existing.numel():
            edge = int(existing[0].item())
            component_atom_ptr = system.component_atom_ptr.tolist()
            component_a_atoms = {
                system.symbols[symbol]
                for symbol in system.atom_symbol[
                    component_atom_ptr[component_a] : component_atom_ptr[component_a + 1]
                ].tolist()
            }
            component_b_atoms = {
                system.symbols[symbol]
                for symbol in system.atom_symbol[
                    component_atom_ptr[component_b] : component_atom_ptr[component_b + 1]
                ].tolist()
            }
            expected_direction = {
                "undirected": BOND_DIRECTION_UNDIRECTED,
                "a_to_b": BOND_DIRECTION_ENDPOINT_0_DONOR,
                "b_to_a": BOND_DIRECTION_ENDPOINT_1_DONOR,
            }[realized.direction]
            if atom_b < atom_a:
                expected_direction = -expected_direction
            existing_stereo_index = int(system.bond_stereo[edge].item())
            existing_stereo = None if existing_stereo_index < 0 else str(system.symbols[existing_stereo_index])
            exact = (
                bool(system.bond_order_known[edge].item()) == (realized.order is not None)
                and (
                    realized.order is None
                    or bool((system.bond_order[edge] == system.bond_order.new_tensor(realized.order)).item())
                )
                and bool(system.bond_aromatic_known[edge].item()) == (realized.aromatic is not None)
                and (realized.aromatic is None or bool(system.bond_aromatic[edge].item()) == realized.aromatic)
                and int(system.bond_kind[edge].item())
                == {
                    "covalent": BOND_KIND_COVALENT,
                    "dative": BOND_KIND_DATIVE,
                }[realized.kind]
                and int(system.bond_direction[edge].item()) == expected_direction
                and int(system.bond_role[edge].item()) == BOND_ROLE_ASSEMBLY
                and existing_stereo == realized.stereo
                and bool((system.bond_stereo_index[:, edge] < 0).all().item())
                and realized.leaving_a.isdisjoint(component_a_atoms)
                and realized.leaving_b.isdisjoint(component_b_atoms)
            )
            if exact:
                continue
            raise ValueError("BoltzGen covalent condition conflicts with an existing bond")
        leaving_a = torch.tensor(
            [
                system.atom_index(AtomSite(realized.site_a.instance_id, realized.site_a.component_index, atom_id))
                for atom_id in realized.leaving_a
            ],
            dtype=torch.long,
            device=system.edge_index.device,
        )
        leaving_b = torch.tensor(
            [
                system.atom_index(AtomSite(realized.site_b.instance_id, realized.site_b.component_index, atom_id))
                for atom_id in realized.leaving_b
            ],
            dtype=torch.long,
            device=system.edge_index.device,
        )
        if component_a == component_b:
            raise ValueError("a BoltzGen covalent condition must join distinct components")
        system, old_to_new = form_bond(
            system,
            atom_a,
            atom_b,
            leaving_a,
            leaving_b,
            order=realized.order,
            aromatic=realized.aromatic,
            kind=realized.kind,
            direction=realized.direction,
            stereo=realized.stereo,
            role="assembly",
        )
        structure = _transfer_structure(structure, system, old_to_new)
    return structure


def _resolve_conditions(
    conditions: tuple[BoltzGenSchema, ...],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> tuple[BoltzGenSchema, ...]:
    def residue(reference: ComponentReference | SymbolicResidueSite) -> ComponentReference:
        return (
            resolve(reference, placement_map, instance_of) if isinstance(reference, SymbolicResidueSite) else reference
        )

    def endpoints(values: tuple[Endpoint, ...]) -> tuple[Endpoint, ...]:
        resolved_values: list[Endpoint] = []
        for value in values:
            if isinstance(value, str):
                resolved_values.extend(placement_map[instance_of[value]][value])
            elif isinstance(value, SymbolicResidueSite):
                resolved_values.append(residue(value))
            elif isinstance(value, SymbolicAtomSite):
                reference = residue(value.residue)
                resolved_values.append(AtomSite(reference[0], reference[1], value.atom_id))
            else:
                resolved_values.append(value)
        return tuple(resolved_values)

    output: list[BoltzGenSchema] = []
    for condition in conditions:
        if isinstance(condition, BoltzGenDesignMask):
            output.append(BoltzGenDesignMask(tuple(residue(value) for value in condition.positions)))
        elif isinstance(condition, BoltzGenBinding):
            output.append(
                BoltzGenBinding(
                    binding=Hotspot(endpoints(condition.binding.residues)),
                    not_binding=tuple(residue(value) for value in condition.not_binding),
                )
            )
        elif isinstance(condition, BoltzGenSecondaryStructure):
            output.append(
                BoltzGenSecondaryStructure(tuple((residue(value), label) for value, label in condition.assignments))
            )
        elif isinstance(condition, BoltzGenStructureGroup):
            output.append(
                BoltzGenStructureGroup(
                    tuple(residue(value) for value in condition.members),
                    condition.group_id,
                )
            )
        elif isinstance(condition, BoltzGenContact):
            output.append(
                BoltzGenContact(
                    endpoints(condition.side_a),
                    endpoints(condition.side_b),
                    condition.cutoff_angstrom,
                    condition.kind,
                )
            )
        elif isinstance(condition, BoltzGenSymmetry):
            output.append(BoltzGenSymmetry(tuple(tuple(residue(value) for value in unit) for unit in condition.units)))
        else:
            output.append(condition)
    return tuple(output)


def _build_tokens(
    system: MolecularSystem,
    *,
    generated_components: frozenset[int] = frozenset(),
) -> tuple[list[dict[str, object]], tuple[int, ...]]:
    r"""Build BoltzGen token rows directly from canonical graph indices."""

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    entity_types = system.entity_type.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    polymer_type_by_entity = {
        ENTITY_PROTEIN: "protein",
        ENTITY_RNA: "rna",
        ENTITY_DNA: "dna",
    }
    instance_count_by_entity: dict[int, int] = {}
    tokens: list[dict[str, object]] = []

    for instance_index, entity_index in enumerate(instance_entities):
        entity_type = entity_types[entity_index]
        instance_id = system.instance_id(instance_index)
        entity_id = system.entity_id(entity_index)
        symmetry_index = instance_count_by_entity.get(entity_index, 0)
        instance_count_by_entity[entity_index] = symmetry_index + 1
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        if entity_type != ENTITY_MOLECULE:
            polymer_type = polymer_type_by_entity.get(entity_type)
            if polymer_type is None:
                raise ValueError("BoltzGen supports protein, RNA, and DNA polymers")
            standards = _STANDARD_COMPONENTS[polymer_type]
            for component_index in range(component_start, component_stop):
                monomer_index = component_index - component_start
                component_id = system.symbols[component_symbols[component_index]]
                atom_indices = tuple(
                    atom_index
                    for atom_index in range(
                        component_atoms[component_index],
                        component_atoms[component_index + 1],
                    )
                    if atomic_numbers[atom_index] != 1
                    and system.symbols[atom_symbols[atom_index]] not in {"OXT", "OP3"}
                    and (
                        component_index not in generated_components
                        or polymer_type != "protein"
                        or system.symbols[atom_symbols[atom_index]] in {"N", "CA", "C", "O"}
                    )
                )
                if not atom_indices:
                    raise ValueError(f"component {component_id!r} has no model-visible heavy atoms")
                by_name = {system.symbols[atom_symbols[atom_index]]: atom_index for atom_index in atom_indices}
                if polymer_type == "protein":
                    center_name = "CA"
                    bb4_names = ("N", "CA", "C", "O")
                    frame_names = ("N", "CA", "C")
                else:
                    center_name = "C1'"
                    bb4_names = ("C1'", "C3'", "C4'", "C1'")
                    frame_names = ("C1'", "C3'", "C4'")
                center = by_name.get(center_name, atom_indices[0])
                tokens.append(
                    {
                        "atoms": atom_indices,
                        "atom_index": None,
                        "component_index": component_index,
                        "component_id": component_id,
                        "instance_index": instance_index,
                        "instance_id": instance_id,
                        "entity_index": entity_index,
                        "entity_id": entity_id,
                        "monomer_index": monomer_index,
                        "residue_index": monomer_index,
                        "chain_index": instance_index,
                        "symmetry_index": symmetry_index,
                        "polymer_type": polymer_type,
                        "molecule_type": _MOLECULE_TYPES[polymer_type],
                        "residue_type": (
                            _residue_type(component_id, polymer_type)
                            if component_id in standards
                            else _UNKNOWN_RESIDUE_TYPES[polymer_type]
                        ),
                        "center": center,
                        "bb4": tuple(by_name.get(name, center) for name in bb4_names),
                        "frame": tuple(by_name.get(name, center) for name in frame_names),
                    }
                )
        else:
            atom_indices = tuple(
                atom_index
                for component_index in range(component_start, component_stop)
                for atom_index in range(
                    component_atoms[component_index],
                    component_atoms[component_index + 1],
                )
                if atomic_numbers[atom_index] != 1
            )
            if not atom_indices:
                raise ValueError(f"molecule {instance_id!r} has no model-visible heavy atoms")
            atom_component = _atom_component_index(system).detach().cpu().tolist()
            for atom_index in atom_indices:
                component_index = atom_component[atom_index]
                tokens.append(
                    {
                        "atoms": (atom_index,),
                        "atom_index": atom_index,
                        "component_index": component_index,
                        "component_id": system.symbols[component_symbols[component_index]],
                        "instance_index": instance_index,
                        "instance_id": instance_id,
                        "entity_index": entity_index,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": component_index - component_start,
                        "chain_index": instance_index,
                        "symmetry_index": symmetry_index,
                        "polymer_type": None,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "center": atom_index,
                        "bb4": (atom_index,) * 4,
                        "frame": (atom_index,) * 3,
                    }
                )

    atom_indices = tuple(atom_index for token in tokens for atom_index in cast(tuple[int, ...], token["atoms"]))
    return tokens, atom_indices


def _atom_component_index(system: MolecularSystem) -> Tensor:
    return torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            device=system.component_atom_ptr.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )


_PROTEIN_ATOM14_NAMES = {
    "GLY": ("N", "CA", "C", "O"),
    "ALA": ("N", "CA", "C", "O", "CB"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

_PROTEIN_ATOM14_FAKE_SOURCE = {
    "GLY": ("O",) * 10,
    "ALA": ("O",) * 9,
    "CYS": ("O",) * 8,
    "SER": ("N",) * 8,
    "PRO": ("O",) * 7,
    "THR": ("N", "N", "N", "O", "O", "O", "O"),
    "VAL": ("N",) * 7,
    "ASN": ("N", "O", "O", "O", "O", "O"),
    "ASP": ("N", "N", "O", "O", "O", "O"),
    "ILE": ("O",) * 6,
    "LEU": ("N", "N", "N", "N", "O", "O"),
    "MET": ("N",) * 6,
    "GLN": ("O",) * 5,
    "GLU": ("N", "N", "O", "O", "O"),
    "LYS": ("N",) * 5,
    "HIS": ("O",) * 4,
    "PHE": ("O",) * 3,
    "ARG": ("N",) * 3,
    "TYR": ("O", "O"),
    "TRP": (),
}

_STANDARD_LOCAL_SYMMETRY_ATOMS = {
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
    "C": (("OP1", "OP2"), ("OP2", "OP1")),
    "G": (("OP1", "OP2"), ("OP2", "OP1")),
    "U": (("OP1", "OP2"), ("OP2", "OP1")),
    "DA": (("OP1", "OP2"), ("OP2", "OP1")),
    "DC": (("OP1", "OP2"), ("OP2", "OP1")),
    "DG": (("OP1", "OP2"), ("OP2", "OP1")),
    "DT": (("OP1", "OP2"), ("OP2", "OP1")),
}


def _validate_structure_design_rows(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    design_mask: Tensor,
    generated_components: frozenset[int],
) -> None:
    r"""
    Require the released structure-design domain: standard protein rows only.
    """

    for token, designed in zip(tokens, design_mask.tolist(), strict=True):
        if not designed:
            continue
        component_index = cast(int, token["component_index"])
        instance_index = cast(int, token["instance_index"])
        entity_index = int(system.instance_entity[instance_index])
        if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
            raise ValueError(
                "BoltzGen structure design supports protein rows only; DNA and RNA may be supplied as fixed context"
            )
        component_id = cast(str, token["component_id"])
        if component_index not in generated_components and component_id not in _PROTEIN_ATOM14_NAMES:
            raise ValueError(
                "BoltzGen structure design requires a standard amino-acid row, "
                f"got {component_id!r} at {system.component_reference(component_index)!r}"
            )


def _build_structure_design_atom_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    structure: Structure,
    design_mask: Tensor,
    *,
    generated_components: frozenset[int] = frozenset(),
    generator: torch.Generator | None = None,
) -> tuple[
    dict[str, Tensor],
    Tensor,
    Tensor,
    dict[int, int],
]:
    r"""
    Lower the released protein-atom14 design rows and natural fixed-context rows.

    Synthetic rows have no canonical `AtomSite`. Their target coordinates copy
    the residue-specific N/O source, while every design row is already masked to
    the exact features visible to the Model.
    """

    model_atom_indices: list[int] = []
    target_source_indices: list[int] = []
    reference_source_indices: list[int] = []
    atom_to_token_rows: list[int] = []
    atom_token_indices: list[int] = []
    design_atom_rows: list[bool] = []
    design_backbone_rows: list[bool] = []
    reference_space_ids: list[int] = []

    for token_index, (token, designed) in enumerate(zip(tokens, design_mask.tolist(), strict=True)):
        natural_atoms = cast(tuple[int, ...], token["atoms"])
        component_index = cast(int, token["component_index"])
        if designed:
            component_id = "GLY" if component_index in generated_components else cast(str, token["component_id"])
            atom_names = _PROTEIN_ATOM14_NAMES[component_id]
            by_name = {system.symbols[int(system.atom_symbol[atom_index])]: atom_index for atom_index in natural_atoms}
            missing = [name for name in atom_names if name not in by_name]
            if missing:
                raise ValueError(
                    f"BoltzGen atom14 row {system.component_reference(component_index)!r} "
                    f"is missing canonical atoms {missing!r}"
                )
            real_atoms = tuple(by_name[name] for name in atom_names)
            ca_atom = by_name["CA"]
            for slot, atom_index in enumerate(real_atoms):
                model_atom_indices.append(atom_index)
                target_source_indices.append(atom_index)
                reference_source_indices.append(atom_index)
                atom_to_token_rows.append(token_index)
                atom_token_indices.append(slot)
                design_atom_rows.append(True)
                design_backbone_rows.append(slot < 4)
                reference_space_ids.append(component_index)
            for slot, source_name in enumerate(
                _PROTEIN_ATOM14_FAKE_SOURCE[component_id],
                start=len(real_atoms),
            ):
                model_atom_indices.append(-1)
                target_source_indices.append(by_name[source_name])
                reference_source_indices.append(ca_atom)
                atom_to_token_rows.append(token_index)
                atom_token_indices.append(slot)
                design_atom_rows.append(True)
                design_backbone_rows.append(False)
                reference_space_ids.append(component_index)
            continue

        for slot, atom_index in enumerate(natural_atoms):
            model_atom_indices.append(atom_index)
            target_source_indices.append(atom_index)
            reference_source_indices.append(atom_index)
            atom_to_token_rows.append(token_index)
            atom_token_indices.append(slot)
            design_atom_rows.append(False)
            design_backbone_rows.append(False)
            reference_space_ids.append(component_index)

    atom_count = len(model_atom_indices)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    system_to_model_atom = {
        system_atom_index: model_atom_index
        for model_atom_index, system_atom_index in enumerate(model_atom_indices)
        if system_atom_index >= 0
    }
    if len(system_to_model_atom) != sum(atom_index >= 0 for atom_index in model_atom_indices):
        raise ValueError("BoltzGen model atom rows contain a duplicate canonical atom occurrence")

    reference_coordinates = _reference_coordinates(
        system,
        tuple(reference_source_indices),
        generated_components=generated_components,
    )
    target_coordinates = torch.zeros(padded_atom_count, 3)
    coordinate_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    coordinate_source = torch.tensor(
        target_source_indices,
        dtype=torch.long,
        device=structure.coordinates.device,
    )
    target_coordinates[:atom_count] = structure.coordinates[coordinate_source].detach().cpu()
    coordinate_mask[:atom_count] = structure.coordinate_mask[coordinate_source].detach().cpu()

    atom_coordinates = target_coordinates.clone()
    valid_coordinates = coordinate_mask[:atom_count]
    center = (atom_coordinates[:atom_count] * valid_coordinates[:, None]).sum(dim=0) / valid_coordinates.sum().clamp(
        min=1
    )
    atom_coordinates[:atom_count] -= center
    center_atom_indices = torch.tensor(
        [system_to_model_atom[cast(int, token["center"])] for token in tokens],
        dtype=torch.long,
    )
    center_coordinates = target_coordinates[center_atom_indices]

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = reference_coordinates
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = torch.tensor(reference_space_ids)
    _augment_reference_positions(
        reference_positions[:atom_count],
        reference_space_id[:atom_count],
        valid_coordinates,
        generator=generator,
    )
    design_atom_mask = torch.tensor(design_atom_rows, dtype=torch.bool)
    reference_positions[:atom_count] = reference_positions[:atom_count].masked_fill(
        design_atom_mask[:, None],
        0,
    )

    real_atoms = tuple(atom_index for atom_index in model_atom_indices if atom_index >= 0)
    model_names = dict(zip(real_atoms, _model_atom_names(system, real_atoms), strict=True))
    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    for atom_index, system_atom_index in enumerate(model_atom_indices):
        token_slot = atom_token_indices[atom_index]
        if design_atom_rows[atom_index] and not design_backbone_rows[atom_index]:
            atom_name = f"FL{token_slot}"
            atomic_number[atom_index] = 114
        else:
            assert system_atom_index >= 0
            atom_name = model_names[system_atom_index]
            atomic_number[atom_index] = int(system.atomic_number[system_atom_index])
            if bool(system.formal_charge_known[system_atom_index]):
                reference_charges[atom_index] = int(system.formal_charge[system_atom_index])
        atom_name_characters[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        atom_to_token[atom_index] = atom_to_token_rows[atom_index]
        atom_token_index[atom_index] = token_slot

    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    model_atom_index = torch.full((padded_atom_count,), -1, dtype=torch.long)
    model_atom_index[:atom_count] = torch.tensor(model_atom_indices)
    coordinate_atom_index = torch.full(
        (padded_atom_count,),
        -1,
        dtype=torch.long,
    )
    coordinate_atom_index[:atom_count] = torch.tensor(target_source_indices)
    return (
        {
            "atom_features": atom_features,
            "reference_positions": reference_positions,
            "reference_space_id": reference_space_id,
            "atom_mask": atom_mask,
            "atom_to_token": atom_to_token,
            "atom_token_index": atom_token_index,
            "atom_coordinates": atom_coordinates,
            "center_coordinates": center_coordinates,
        },
        model_atom_index,
        coordinate_atom_index,
        system_to_model_atom,
    )


def _truth_pool_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    structure: Structure,
    coordinate_atom_index: Tensor,
    system_to_model_atom: dict[int, int],
) -> dict[str, Tensor]:
    r"""
    Lower complete-assembly truth and its current-atom/symmetry mappings.

    The truth pool is the complete canonical atom axis. Unresolved atoms retain
    their rows.
    """

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    chain_spans: list[tuple[int, int]] = []
    chain_equivalence: list[int] = []
    equivalence_index: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
    for instance_index, entity_index in enumerate(instance_entities):
        component_start = instance_components[instance_index]
        component_stop = instance_components[instance_index + 1]
        atom_start = component_atoms[component_start]
        atom_stop = component_atoms[component_stop]
        chain_spans.append((atom_start, atom_stop))
        layout = tuple(
            (
                component_index - component_start,
                atom_symbols[atom_index],
            )
            for component_index in range(component_start, component_stop)
            for atom_index in range(
                component_atoms[component_index],
                component_atoms[component_index + 1],
            )
        )
        key = (entity_index, layout)
        chain_equivalence.append(equivalence_index.setdefault(key, len(equivalence_index)))

    symmetry_atom_index, symmetry_permutation = _local_symmetry_features(
        system,
        system_to_model_atom,
    )
    return {
        "coordinates": structure.coordinates.detach().cpu(),
        "coordinate_mask": structure.coordinate_mask.detach().cpu(),
        "coordinate_atom_index": coordinate_atom_index,
        "chain_span": torch.tensor(chain_spans, dtype=torch.long),
        "chain_equivalence_index": torch.tensor(
            chain_equivalence,
            dtype=torch.long,
        ),
        "symmetry_atom_index": symmetry_atom_index,
        "symmetry_permutation": symmetry_permutation,
    }


def _local_symmetry_features(
    system: MolecularSystem,
    system_to_model_atom: dict[int, int],
) -> tuple[Tensor, Tensor]:
    r"""
    Carry exact standard swaps and the available stereochemical self-matches.
    """

    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    component_symbols = system.component_symbol.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    component_boundaries = component_boundary_atoms(system)
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for component_index in range(system.component_symbol.numel()):
        component_id = system.symbols[component_symbols[component_index]]
        atom_permutations = _STANDARD_LOCAL_SYMMETRY_ATOMS.get(component_id)
        is_standard = any(component_id in standards for standards in _STANDARD_COMPONENTS.values())
        if atom_permutations is not None:
            by_name = {
                system.symbols[atom_symbols[atom_index]]: atom_index
                for atom_index in range(
                    component_atoms[component_index],
                    component_atoms[component_index + 1],
                )
                if atom_index in system_to_model_atom
            }
            permutations = tuple(
                tuple(system_to_model_atom[by_name[atom_id]] for atom_id in atom_names)
                for atom_names in atom_permutations
                if all(atom_id in by_name for atom_id in atom_names)
            )
            if len(permutations) > 1:
                groups.append((permutations[0], permutations))
            continue
        if is_standard:
            continue
        group = _component_self_match_group(
            system,
            component_index,
            system_to_model_atom,
            component_boundaries[component_index],
        )
        if group is not None:
            groups.append(group)

    if not groups:
        return (
            torch.empty(0, 0, dtype=torch.long),
            torch.empty(0, 0, 0, dtype=torch.long),
        )

    max_width = max(len(atom_indices) for atom_indices, _ in groups)
    max_permutations = max(len(permutations) for _, permutations in groups)
    atom_index = torch.full((len(groups), max_width), -1, dtype=torch.long)
    permutation = torch.full(
        (len(groups), max_permutations, max_width),
        -1,
        dtype=torch.long,
    )
    for group_index, (atom_indices, permutations) in enumerate(groups):
        width = len(atom_indices)
        atom_index[group_index, :width] = torch.tensor(atom_indices)
        for permutation_index, values in enumerate(permutations):
            permutation[group_index, permutation_index, :width] = torch.tensor(values)
    return atom_index, permutation


def _component_self_match_group(
    system: MolecularSystem,
    component_index: int,
    system_to_model_atom: dict[int, int],
    boundary_atoms: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    r"""
    Return the carried RDKit automorphism candidates for one occurrence.
    """

    from rdkit import Chem

    if not bool(system.component_topology_complete[component_index]):
        return None
    atom_start = int(system.component_atom_ptr[component_index])
    atom_stop = int(system.component_atom_ptr[component_index + 1])
    visible_local = [
        atom_index - atom_start for atom_index in range(atom_start, atom_stop) if atom_index in system_to_model_atom
    ]
    if len(visible_local) < 2:
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
    for boundary_index, atom_index in enumerate(sorted(boundary_atoms), start=1):
        molecule.GetAtomWithIdx(atom_index - atom_start).SetIsotope(1000 + boundary_index)
    Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    identity = tuple(system_to_model_atom[atom_start + local_index] for local_index in visible_local)
    candidates = {identity}
    for match in molecule.GetSubstructMatches(
        molecule,
        uniquify=False,
        useChirality=True,
        maxMatches=1000,
    ):
        mapped = tuple(match[index] for index in visible_local)
        if set(mapped) != set(visible_local):
            continue
        candidates.add(tuple(system_to_model_atom[atom_start + local_index] for local_index in mapped))
    if len(candidates) == 1:
        return None
    return identity, (identity, *sorted(candidates - {identity}))


def _frame_atom_indices(
    tokens: list[dict[str, object]],
    system_to_model_atom: dict[int, int],
) -> Tensor:
    r"""
    Lower prediction-independent polymer confidence-frame atom indices.

    Non-polymer entries are placeholders: confidence equations replace them
    from each prediction or matched target because their frames depend on the
    corresponding coordinates and resolved mask.
    """

    return torch.tensor(
        [
            [system_to_model_atom[atom_index] for atom_index in cast(tuple[int, ...], token["frame"])]
            for token in tokens
        ],
        dtype=torch.long,
    )


def _build_atom_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    atom_indices: tuple[int, ...],
    structure: Structure,
    generated_components: frozenset[int] = frozenset(),
    generator: torch.Generator | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    dict[int, int],
]:
    r"""
    Build the 388-wide BoltzGen/Boltz-2 atom reference feature vector and index maps.

    The 388-wide layout contains positions (3), charge (1), element (128), and
    atom-name characters (4x64), with no atom-mask channel.

    ``generated_components`` selects protein placeholders whose backbone atoms
    use GLY-template coordinates. An empty set uses component reference
    coordinates.
    """

    system_to_model_atom = {
        system_atom_index: model_atom_index for model_atom_index, system_atom_index in enumerate(atom_indices)
    }
    system_atom_to_token = {
        system_atom_index: token_index
        for token_index, token in enumerate(tokens)
        for system_atom_index in cast(tuple[int, ...], token["atoms"])
    }
    atom_count = len(atom_indices)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)

    reference_coordinates = _reference_coordinates(
        system,
        atom_indices,
        generated_components=generated_components,
    )
    atom_component = _atom_component_index(system)
    canonical_atom_index = torch.tensor(
        atom_indices,
        device=system.atom_symbol.device,
    )
    reference_space_ids = atom_component[canonical_atom_index].detach().cpu()
    raw_atom_coordinates = torch.zeros(padded_atom_count, 3)
    resolved = torch.zeros(padded_atom_count, dtype=torch.bool)
    raw_atom_coordinates[:atom_count] = structure.coordinates[canonical_atom_index].detach().cpu()
    resolved[:atom_count] = structure.coordinate_mask[canonical_atom_index].detach().cpu()

    atom_coordinates = raw_atom_coordinates.clone()
    resolved_atoms = resolved[:atom_count]
    center = (atom_coordinates[:atom_count] * resolved_atoms[:, None]).sum(dim=0)
    center = center / resolved_atoms.sum().clamp(min=1)
    atom_coordinates[:atom_count] -= center
    center_atom_indices = torch.tensor(
        [system_to_model_atom[cast(int, token["center"])] for token in tokens],
        dtype=torch.long,
    )
    center_coordinates = raw_atom_coordinates[center_atom_indices]

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = reference_coordinates
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = reference_space_ids
    _augment_reference_positions(
        reference_positions[:atom_count],
        reference_space_id[:atom_count],
        resolved_atoms,
        generator=generator,
    )

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_indices)
    atomic_number[:atom_count] = system.atomic_number[canonical_atom_index].detach().cpu().long()
    known_charge = system.formal_charge_known[canonical_atom_index].detach().cpu()
    reference_charges[:atom_count] = torch.where(
        known_charge,
        system.formal_charge[canonical_atom_index].detach().cpu().float(),
        torch.zeros(atom_count),
    )
    token_slot_by_atom = {
        system_atom_index: slot
        for token in tokens
        for slot, system_atom_index in enumerate(cast(tuple[int, ...], token["atoms"]))
    }
    for atom_index, (system_atom_index, atom_name) in enumerate(zip(atom_indices, atom_names, strict=True)):
        atom_name_characters[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = system_atom_to_token[system_atom_index]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = token_slot_by_atom[system_atom_index]

    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    model_atom_index = torch.full((padded_atom_count,), -1, dtype=torch.long)
    model_atom_index[:atom_count] = torch.tensor(atom_indices)
    return (
        atom_features,
        reference_positions,
        reference_space_id,
        atom_mask,
        atom_to_token,
        atom_token_index,
        atom_coordinates,
        resolved,
        center_coordinates,
        model_atom_index,
        system_to_model_atom,
    )


def _contact_channels(
    system: MolecularSystem,
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_component: dict[int, int],
    token_index_by_atom: dict[int, int],
    token_count: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Realize contact records into the ``[N, N, 5]`` one-hot + ``[N, N]`` threshold.
    """

    selected: list[tuple[BoltzGenContact, tuple[int, ...], tuple[int, ...]]] = []
    for condition in conditions:
        if not isinstance(condition, BoltzGenContact):
            continue
        rows = _endpoint_tokens(
            system,
            condition.side_a,
            token_index_by_component,
            token_index_by_atom,
        )
        columns = _endpoint_tokens(
            system,
            condition.side_b,
            token_index_by_component,
            token_index_by_atom,
        )
        if not rows or not columns:
            raise ValueError("a BoltzGen contact must resolve to at least one token on each side")
        selected.append((condition, rows, columns))

    # An absent contact request is UNSPECIFIED; background pairs in a request are UNSELECTED.
    classes = torch.full((token_count, token_count), int(bool(selected)), dtype=torch.long)
    threshold = torch.zeros(token_count, token_count)
    for contact, rows, columns in selected:
        forward_class = _CONTACT_KIND_TO_CLASS[contact.kind]
        reverse_class = {2: 3, 3: 2}.get(forward_class, forward_class)
        for row in rows:
            for column in columns:
                classes[row, column] = forward_class
                classes[column, row] = reverse_class
                threshold[row, column] = contact.cutoff_angstrom
                threshold[column, row] = contact.cutoff_angstrom
    contact_conditioning = torch.nn.functional.one_hot(classes, _CONTACT_CLASSES).float()
    return contact_conditioning, threshold


def _endpoint_tokens(
    system: MolecularSystem,
    endpoints: tuple[Endpoint, ...],
    token_index_by_component: dict[int, int],
    token_index_by_atom: dict[int, int],
) -> tuple[int, ...]:
    r"""
    Resolve contact endpoints to token rows, preserving declaration order.
    """

    rows: list[int] = []
    for endpoint in endpoints:
        if isinstance(endpoint, AtomSite):
            rows.append(token_index_by_atom[system.atom_index(endpoint)])
        else:
            rows.extend(
                _component_tokens(
                    system,
                    system.component_index(cast(ComponentReference, endpoint)),
                    token_index_by_component,
                    token_index_by_atom,
                )
            )
    return tuple(dict.fromkeys(rows))


def _component_tokens(
    system: MolecularSystem,
    component_index: int,
    token_index_by_component: dict[int, int],
    token_index_by_atom: dict[int, int],
) -> tuple[int, ...]:
    r"""
    Map one polymer residue or ligand component to all token rows it owns.
    """

    if component_index in token_index_by_component:
        return (token_index_by_component[component_index],)
    rows = tuple(
        dict.fromkeys(
            token_index_by_atom[atom_index]
            for atom_index in range(
                int(system.component_atom_ptr[component_index]),
                int(system.component_atom_ptr[component_index + 1]),
            )
            if atom_index in token_index_by_atom
        )
    )
    if not rows:
        raise KeyError(system.component_reference(component_index))
    return rows


def _design_mask(
    system: MolecularSystem,
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_component: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Realize the ``[N]`` boolean design mask.

    Generated placement residues are designable by default. Explicit
    [`BoltzGenDesignMask`][ofoldx.models.boltzgen.model.BoltzGenDesignMask] records add existing
    residues to that set.
    """

    positions: list[ComponentReference] = []
    for condition in conditions:
        if isinstance(condition, BoltzGenDesignMask):
            positions.extend(condition.positions)
    return _design_mask_from_positions(
        system,
        tuple(positions),
        token_index_by_component,
        token_count,
    )


def _chain_design_mask(
    design_mask: Tensor,
    chain_index: Tensor,
    token_bonds: Tensor,
) -> Tensor:
    r"""
    Propagate design membership over whole chains and covalently linked chains.

    This is a connected-component lowering over two edge kinds: tokens in the
    same molecular instance and explicit inter-token covalent bonds.
    """

    connections: dict[int, set[int]] = {int(chain): set() for chain in torch.unique(chain_index).tolist()}
    for source, target in torch.nonzero(token_bonds[..., 0], as_tuple=False).tolist():
        source_chain = int(chain_index[source])
        target_chain = int(chain_index[target])
        connections[source_chain].add(target_chain)
        connections[target_chain].add(source_chain)

    selected = {int(chain) for chain in chain_index[design_mask.bool()].tolist()}
    pending = list(selected)
    while pending:
        chain = pending.pop()
        for connected in connections[chain] - selected:
            selected.add(connected)
            pending.append(connected)
    return torch.tensor([int(chain) in selected for chain in chain_index], dtype=torch.bool)


def _structure_group_ids(
    system: MolecularSystem,
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_component: dict[int, int],
    token_count: int,
    design_mask: Tensor,
) -> Tensor:
    r"""
    Realize ``BoltzGenStructureGroup`` records into the per-token ``[N]`` structure-group ids.

    Group ``0`` is free and ungrouped; each record assigns its ``group_id`` to its
    members. Designed residues must remain in group ``0`` because their backbone
    geometry cannot anchor distance conditioning.
    """

    ids = torch.zeros(token_count, dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenStructureGroup):
            continue
        for reference in condition.members:
            token = _component_reference_to_token(
                system,
                reference,
                token_index_by_component,
            )
            if condition.group_id != 0 and bool(design_mask[token]):
                raise ValueError(
                    "a designed/generated residue cannot be placed in a nonzero BoltzGenStructureGroup: "
                    "its backbone structure is not fixed, so it cannot anchor distance conditioning"
                )
            ids[token] = condition.group_id
    return ids


def _token_distance_mask(structure_group: Tensor) -> Tensor:
    r"""
    Realize structure groups into the ``[N, N]`` ``token_distance_mask``.

    The whole matrix is ``1`` when every token shares group ``1``. Otherwise cell ``(i, j)`` is
    ``1`` iff ``structure_group[i] > 0`` and ``structure_group[i] == structure_group[j]`` (distinct
    nonzero groups move independently; group ``0`` is never tied). The general formula already yields
    all-ones for the all-group-``1`` case, so the two branches agree by construction.
    """

    token_count = int(structure_group.shape[0])
    mask = torch.zeros(token_count, token_count)
    if token_count == 0:
        return mask
    if bool((structure_group == 1).all()):
        return mask + 1
    same = structure_group.unsqueeze(1) == structure_group.unsqueeze(0)
    nonzero_row = (structure_group > 0).unsqueeze(1)
    mask[same & nonzero_row] = 1
    return mask


def _design_mask_from_positions(
    system: MolecularSystem,
    positions: tuple[ComponentReference, ...],
    token_index_by_component: dict[int, int],
    token_count: int,
) -> Tensor:
    mask = torch.zeros(token_count, dtype=torch.bool)
    for reference in positions:
        mask[
            _component_reference_to_token(
                system,
                reference,
                token_index_by_component,
            )
        ] = True
    return mask


def _binding_ids(
    system: MolecularSystem,
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_component: dict[int, int],
    token_index_by_atom: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Realize ``BoltzGenBinding`` records into the ``[N]`` binding-specification ids.
    """

    ids = torch.full((token_count,), _BINDING_UNSPECIFIED, dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenBinding):
            continue
        for token_index in _endpoint_tokens(
            system,
            condition.binding.residues,
            token_index_by_component,
            token_index_by_atom,
        ):
            ids[token_index] = _BINDING_BINDING
        for reference in condition.not_binding:
            ids[
                _component_reference_to_token(
                    system,
                    reference,
                    token_index_by_component,
                )
            ] = _BINDING_NOT_BINDING
    return ids


def _ss_ids(
    system: MolecularSystem,
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_component: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Realize ``BoltzGenSecondaryStructure`` records into the ``[N]`` ss ids.
    """

    ids = torch.full((token_count,), _SS_UNSPECIFIED, dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenSecondaryStructure):
            continue
        for reference, label in condition.assignments:
            ids[
                _component_reference_to_token(
                    system,
                    reference,
                    token_index_by_component,
                )
            ] = _SS_LABEL_TO_ID[label]
    return ids


def _symmetry_index(
    system: MolecularSystem,
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_component: dict[int, int],
    tokens: list[dict[str, object]],
) -> Tensor:
    r"""
    Realize ``BoltzGenSymmetry`` records into the ``[N]`` symmetry-copy index.

    Defaults to the tokenizer's per-entity symmetry-copy index; an explicit
    ``BoltzGenSymmetry`` overrides matched positions with the unit ordinal.
    """

    ids = torch.tensor([cast(int, token["symmetry_index"]) for token in tokens], dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenSymmetry):
            continue
        for unit_index, unit in enumerate(condition.units):
            for reference in unit:
                ids[
                    _component_reference_to_token(
                        system,
                        reference,
                        token_index_by_component,
                    )
                ] = unit_index
    return ids


def _sequence_symmetry_index(
    groups: tuple[tuple[ComponentReference, ...], ...],
    system: MolecularSystem,
    token_index_by_component: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Realize inverse-folding identity groups into positive ``[N]`` group IDs.

    Unlike structure-design ``symmetry_index``, whose zero-based values identify
    copy offsets for relative-position features, inverse folding reserves zero for
    an untied position.
    """

    ids = torch.zeros(token_count, dtype=torch.long)
    parents: dict[int, int] = {}

    def find(token: int) -> int:
        root = token
        while parents[root] != root:
            root = parents[root]
        while parents[token] != token:
            parent = parents[token]
            parents[token] = root
            token = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for group in groups:
        tokens = [
            _component_reference_to_token(
                system,
                reference,
                token_index_by_component,
            )
            for reference in group
        ]
        if not tokens:
            continue
        for token in tokens:
            parents.setdefault(token, token)
        for token in tokens[1:]:
            union(tokens[0], token)

    root_to_group: dict[int, int] = {}
    for token in parents:
        root = find(token)
        ids[token] = root_to_group.setdefault(root, len(root_to_group) + 1)
    return ids


def _require_protein_reference(
    system: MolecularSystem,
    reference: ComponentReference,
    token_index_by_component: dict[int, int],
    molecule_types: Tensor,
    role: str,
) -> int:
    r"""
    Resolve an inverse-folding residue reference and require a protein token.
    """

    token = _component_reference_to_token(
        system,
        reference,
        token_index_by_component,
    )
    if int(molecule_types[token].item()) != _MOLECULE_TYPES["protein"]:
        raise ValueError(f"BoltzGen inverse-folding {role} must reference a protein residue: {reference!r}")
    return token


def _aa_constraint_mask(
    global_omit_tokens: tuple[str, ...],
    aa_constraints: tuple[BoltzGenResidueConstraint, ...],
    system: MolecularSystem,
    token_index_by_component: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Realize global and per-residue constraints into the ``[N, 20]`` canonical-AA block mask.

    The ``[N, 20]`` mask follows ``CANONICAL_TOKENS`` order, where ``0`` is allowed and ``1`` disallowed;
    the default is all-zero (unconstrained). Global omissions block one vocabulary
    column everywhere. Each ``allowed`` whitelist blocks every canonical AA except
    the listed codes, accumulated with ``max`` so overlapping constraints intersect
    (only AAs allowed by every covering constraint survive).
    """

    mask = torch.zeros(token_count, NUM_CANONICAL_TOKENS, dtype=torch.float32)
    for code in global_omit_tokens:
        mask[:, CANONICAL_TOKENS.index(code)] = 1.0
    for constraint in aa_constraints:
        token = _component_reference_to_token(
            system,
            constraint.position,
            token_index_by_component,
        )
        block = torch.ones(NUM_CANONICAL_TOKENS, dtype=torch.float32)
        for code in constraint.allowed:
            block[CANONICAL_TOKENS.index(code)] = 0.0
        mask[token] = torch.maximum(mask[token], block)
    return mask


# Template features contain residue identity, backbone frames, CB/CA coordinates,
# and visibility for aligned fixed tokens. A self-template is the identity-aligned case.


def _template_frame(
    nitrogen: Tensor,
    alpha: Tensor,
    carbon: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Build one residue's backbone frame.

    Gram-Schmidt over ``(C - CA, N - CA)``: ``e1`` along ``C - CA``, ``e2`` the ``N - CA``
    component orthogonal to ``e1``, ``e3 = e1 x e2``; the rotation columns are ``[e1, e2, e3]``
    and the translation is ``CA``. Norms use a ``1e-10`` epsilon.
    """

    v1 = carbon - alpha
    v2 = nitrogen - alpha
    e1 = v1 / (torch.linalg.vector_norm(v1) + 1e-10)
    u2 = v2 - e1 * torch.dot(e1, v2)
    e2 = u2 / (torch.linalg.vector_norm(u2) + 1e-10)
    e3 = torch.linalg.cross(e1, e2)
    rotation = torch.stack((e1, e2, e3), dim=-1)
    return rotation, alpha


def _template_center_atom(polymer_type: str) -> str:
    r"""
    Return CA for protein tokens and C1' for nucleic tokens.
    """

    return "CA" if polymer_type == "protein" else "C1'"


def _template_disto_atom(component_id: str, polymer_type: str) -> str:
    r"""
    Return the token atom used for template distograms.
    """

    if polymer_type == "protein":
        return "CA" if component_id == "GLY" else "CB"
    if component_id in {"A", "G", "DA", "DG"}:
        return "C4"
    if component_id in {"C", "U", "DC", "DT"}:
        return "C2"
    return "C1'"


def _template_atom(
    structure: Structure,
    component_index: int,
    atom_id: str,
) -> tuple[Tensor, bool]:
    r"""
    Look up one template atom's coordinate + presence, or ``(None, False)`` if absent.
    """

    system = structure.system
    for atom_index in range(
        int(system.component_atom_ptr[component_index]),
        int(system.component_atom_ptr[component_index + 1]),
    ):
        if system.symbols[int(system.atom_symbol[atom_index])] == atom_id:
            return (
                structure.coordinates[atom_index].detach().cpu().float(),
                bool(structure.coordinate_mask[atom_index]),
            )
    return torch.zeros(3), False


def _template_channels(
    templates: tuple[Template, ...],
    tokens: list[dict[str, object]],
    token_count: int,
) -> dict[str, Tensor] | None:
    r"""
    Lower aligned [`Template`][ofoldx.data.evidence.Template] evidence into the BoltzGen template tensors.

    Returns ``None`` when no templates are supplied, so the model skips the template embedder.
    Otherwise returns the nine per-template ``[T, N, ...]`` tensors consumed by
    ``BoltzGenTemplateEmbedder``:
    residue-type one-hot
    (``num_tokens == 33``), ``compute_frame`` backbone frame (protein, all of N/CA/C resolved),
    beta-carbon positions and mask, alpha-carbon positions, template mask, and visibility groups
    where the template covers a token, zeros everywhere else.
    """

    if not templates:
        return None

    instance_residue: dict[tuple[int, int], int] = {}
    entity_residue: dict[tuple[int, int], list[int]] = {}
    instance_entities: dict[int, int] = {}
    for token_index, token in enumerate(tokens):
        monomer_index = cast("int | None", token["monomer_index"])
        if monomer_index is None:
            continue
        instance_index = cast(int, token["instance_index"])
        entity_index = cast(int, token["entity_index"])
        instance_entities[instance_index] = entity_index
        instance_residue[(instance_index, monomer_index)] = token_index
        entity_residue.setdefault((entity_index, monomer_index), []).append(token_index)

    tdim = len(templates)
    residue_type_ids = torch.zeros(tdim, token_count, dtype=torch.long)
    frame_rotations = torch.zeros(tdim, token_count, 3, 3)
    frame_translations = torch.zeros(tdim, token_count, 3)
    beta_carbon_positions = torch.zeros(tdim, token_count, 3)
    alpha_carbon_positions = torch.zeros(tdim, token_count, 3)
    beta_carbon_mask = torch.zeros(tdim, token_count)
    frame_mask = torch.zeros(tdim, token_count)
    template_mask = torch.zeros(tdim, token_count)
    visibility_index = torch.zeros(tdim, token_count)

    for template_index, template in enumerate(templates):
        template_system = template.structure.system
        template_instance_components = template_system.instance_component_ptr.detach().cpu().tolist()
        template_component_symbols = template_system.component_symbol.detach().cpu().tolist()
        for alignment in template.alignments:
            target_entity_index = next(
                (
                    cast(int, token["entity_index"])
                    for token in tokens
                    if token["entity_id"] == alignment.target_entity_id
                ),
                -1,
            )
            if target_entity_index < 0:
                raise ValueError(f"BoltzGen template targets unknown entity {alignment.target_entity_id!r}")
            template_instance_index = template_system.instance_index(alignment.template_instance_id)
            template_entity_index = int(template_system.instance_entity[template_instance_index])
            template_entity_type = int(template_system.entity_type[template_entity_index])
            polymer_type = {
                ENTITY_PROTEIN: "protein",
                ENTITY_RNA: "rna",
                ENTITY_DNA: "dna",
            }.get(template_entity_type)
            if polymer_type is None:
                raise ValueError("a BoltzGen template alignment must reference a polymer instance")
            if alignment.target_instance_id is not None:
                target_instance_index = next(
                    (
                        cast(int, token["instance_index"])
                        for token in tokens
                        if token["instance_id"] == alignment.target_instance_id
                    ),
                    -1,
                )
                if target_instance_index < 0 or instance_entities[target_instance_index] != target_entity_index:
                    raise ValueError(
                        f"BoltzGen template alignment instance {alignment.target_instance_id!r} belongs to entity "
                        f"other than {alignment.target_entity_id!r}"
                    )
            template_component_start = template_instance_components[template_instance_index]
            template_component_stop = template_instance_components[template_instance_index + 1]
            for target_component, template_component in alignment.residue_pairs:
                targets: tuple[int, ...]
                if alignment.target_instance_id is not None:
                    hit = instance_residue.get((target_instance_index, target_component))
                    if hit is None:
                        raise ValueError(
                            f"BoltzGen template alignment target {alignment.target_instance_id!r}:"
                            f"{target_component} does not resolve to a token"
                        )
                    targets = (hit,)
                else:
                    targets = tuple(
                        entity_residue.get(
                            (target_entity_index, target_component),
                            (),
                        )
                    )
                    if not targets:
                        raise ValueError(
                            f"BoltzGen template alignment target {alignment.target_entity_id!r}:"
                            f"{target_component} does not resolve to a token"
                        )
                template_component_index = template_component_start + template_component
                if template_component_index >= template_component_stop:
                    raise IndexError(
                        f"template component index {template_component} is outside "
                        f"instance {alignment.template_instance_id!r}"
                    )
                component_id = template_system.symbols[template_component_symbols[template_component_index]]

                center_coord, _ = _template_atom(
                    template.structure,
                    template_component_index,
                    _template_center_atom(polymer_type),
                )
                disto_coord, disto_present = _template_atom(
                    template.structure,
                    template_component_index,
                    _template_disto_atom(component_id, polymer_type),
                )
                nitrogen, nitrogen_present = _template_atom(
                    template.structure,
                    template_component_index,
                    "N",
                )
                alpha, alpha_present = _template_atom(
                    template.structure,
                    template_component_index,
                    "CA",
                )
                carbon, carbon_present = _template_atom(
                    template.structure,
                    template_component_index,
                    "C",
                )
                is_protein = polymer_type == "protein"
                has_frame = is_protein and nitrogen_present and alpha_present and carbon_present
                rotation = translation = None
                if has_frame:
                    rotation, translation = _template_frame(nitrogen, alpha, carbon)

                residue_type = _residue_type(component_id, polymer_type)
                for target_token in targets:
                    residue_type_ids[template_index, target_token] = residue_type
                    alpha_carbon_positions[template_index, target_token] = center_coord
                    beta_carbon_positions[template_index, target_token] = disto_coord
                    beta_carbon_mask[template_index, target_token] = 1.0 if disto_present else 0.0
                    if has_frame:
                        assert rotation is not None and translation is not None
                        frame_rotations[template_index, target_token] = rotation
                        frame_translations[template_index, target_token] = translation
                        frame_mask[template_index, target_token] = 1.0
                    template_mask[template_index, target_token] = 1.0
                    visibility_index[template_index, target_token] = 1.0

        if not bool(template_mask[template_index].any()):
            raise ValueError(f"BoltzGen template {template_index} does not cover any lowered target token")

    return {
        "template_residue_type_features": torch.nn.functional.one_hot(residue_type_ids, 33),
        "template_frame_rotations": frame_rotations,
        "template_frame_translations": frame_translations,
        "template_beta_carbon_positions": beta_carbon_positions,
        "template_alpha_carbon_positions": alpha_carbon_positions,
        "template_beta_carbon_mask": beta_carbon_mask,
        "template_frame_mask": frame_mask,
        "template_mask": template_mask,
        "template_visibility_index": visibility_index,
    }


# Token, atom, and MSA features shared by the two BoltzGen processors.

# The 33-token vocabulary uses protein tokens 2..21, unknown tokens 22/27/32,
# RNA tokens 23..27, DNA tokens 28..32, and token 1 for the MSA gap.
_PROTEIN_RESIDUE_TYPES = {
    component_id: index
    for index, component_id in enumerate(
        (
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
        ),
        start=2,
    )
}
_RNA_RESIDUE_TYPES = {"A": 23, "G": 24, "C": 25, "U": 26, "N": 27}
_DNA_RESIDUE_TYPES = {"DA": 28, "DG": 29, "DC": 30, "DT": 31, "DN": 32}
_STANDARD_COMPONENTS = {
    "protein": frozenset(_PROTEIN_RESIDUE_TYPES),
    "rna": frozenset(_RNA_RESIDUE_TYPES),
    "dna": frozenset(_DNA_RESIDUE_TYPES),
}
_MOLECULE_TYPES = {"protein": 0, "dna": 1, "rna": 2}
_UNKNOWN_RESIDUE_TYPES = {"protein": 22, "rna": 27, "dna": 32}
_MSA_GAP = MSA_GAP_INDEX


def _msa_symbol_restype(symbol: str, polymer_type: str) -> int:
    r"""
    Map one UPPERCASE one-letter MSA symbol (or ``-``) to a BoltzGen token.
    """

    return msa_residue_index(symbol, polymer_type)


def _token_msa_restype(token: dict[str, object], system: MolecularSystem) -> int:
    polymer_type = cast(str | None, token["polymer_type"])
    if polymer_type is None:
        return msa_residue_index("X", None)
    component_index = cast(int, token["component_index"])
    component_id = system.symbols[int(system.component_symbol[component_index])]
    return msa_component_index(component_id, polymer_type)


def _token_bond_indicator(
    system: MolecularSystem,
    atom_to_token: dict[int, int],
    token_count: int,
) -> Tensor:
    r"""
    Build the symmetric inter-token bond indicator ``[N, N, 1]`` consumed by ``token_bond_projection``.

    The trailing singleton feeds ``nn.Linear(1, ...)`` in the design trunk; inverse
    folding indexes ``token_bonds[..., 0]``.
    """

    token_bonds = torch.zeros(token_count, token_count)

    def add(atom_a: int, atom_b: int) -> None:
        token_a = atom_to_token.get(atom_a)
        token_b = atom_to_token.get(atom_b)
        if token_a is None or token_b is None or token_a == token_b:
            return
        token_bonds[token_a, token_b] = token_bonds[token_b, token_a] = 1

    for atom_a, atom_b in system.edge_index.detach().cpu().T.tolist():
        add(atom_a, atom_b)
    return token_bonds.unsqueeze(-1)


def _inter_component_bond_atom_index(
    system: MolecularSystem,
    system_to_model_atom: dict[int, int],
) -> Tensor:
    r"""
    Return unique known covalent edges whose endpoints belong to different residues.
    """

    atom_component = _atom_component_index(system).detach().cpu().tolist()
    edges: list[tuple[int, int]] = []
    for left, right in system.edge_index.detach().cpu().T.tolist():
        if atom_component[left] == atom_component[right]:
            continue
        if left not in system_to_model_atom or right not in system_to_model_atom:
            continue
        model_left = system_to_model_atom[left]
        model_right = system_to_model_atom[right]
        edges.append((min(model_left, model_right), max(model_left, model_right)))
    if not edges:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


def _msa_features(
    msa: MSA | MsaCollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> dict[str, Tensor]:
    residue_types = [_token_msa_restype(token, system) for token in tokens]
    if msa is None:
        rows = torch.tensor([residue_types])
        deletion_counts = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        # The MSA holds no PolymerSequence: column ``c`` binds ``columns[c].entity_id`` to a system entity,
        # a token's monomer index is directly the aligned-row column, and cells are UPPERCASE one-letter
        # symbols (or ``-``). A bare MSA is a single-chain plan over its unpaired pool; a collection replays
        # its dense ``row_pairing`` and ``row_is_paired`` plan across the paired and unpaired pools.
        bindings = msa_bindings(msa)
        columns = tuple(value for value, _target_instance_id in bindings)
        if isinstance(msa, MSA):
            plan_rows: list[tuple[tuple[int, int, bool], ...]] = [
                ((0, row_index, False),) for row_index in range(len(msa.unpaired_sequences))
            ]
        else:
            plan_rows = [
                tuple(
                    (column_index, int(msa.row_pairing[plan_index, column_index]), bool(is_paired))
                    for column_index, is_paired in enumerate(msa.row_is_paired[plan_index])
                    if msa.row_pairing[plan_index, column_index] != -1
                )
                for plan_index in range(msa.row_pairing.shape[0])
            ]
        entity_type_to_polymer = {
            ENTITY_PROTEIN: "protein",
            ENTITY_RNA: "rna",
            ENTITY_DNA: "dna",
        }
        polymer_types = {
            value.entity_id: entity_type_to_polymer[int(system.entity_type[system.entity_index(value.entity_id)])]
            for value in columns
        }
        row_values: list[Tensor] = []
        deletion_values: list[Tensor] = []
        paired_values: list[Tensor] = []
        for selections in plan_rows:
            row = torch.full((len(tokens),), _MSA_GAP, dtype=torch.long)
            deletion_row = torch.zeros(len(tokens))
            paired_row = torch.zeros(len(tokens))
            present = len(selections)
            # A taxonomy-paired plan row draws every populated cell from the paired pool; the dense
            # ``row_is_paired`` flag records exactly that, so ``present > 1`` paired-pool rows reproduce the
            # a taxonomy-paired row.
            taxonomic_pair = (
                isinstance(msa, MsaCollection) and present > 1 and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                target_instance_id = bindings[column_index][1]
                polymer_type = polymer_types[value.entity_id]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if (
                        token["entity_id"] != value.entity_id
                        or (target_instance_id is not None and token["instance_id"] != target_instance_id)
                        or token["monomer_index"] is None
                    ):
                        continue
                    monomer_index = cast(int, token["monomer_index"])
                    row[token_index] = _msa_symbol_restype(aligned[monomer_index], polymer_type)
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
        "msa_profile": torch.nn.functional.one_hot(rows, MSA_VOCAB_SIZE).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _mask_structure_design_msa(values: dict[str, Tensor], design_mask: Tensor) -> dict[str, Tensor]:
    r"""
    Remove designed residue identity from every model-visible MSA channel.
    """

    if not bool(design_mask.any()):
        return values
    unknown = MSA_INDEX["X"]
    values["msa_tokens"][:, design_mask] = unknown
    values["has_deletion"][:, design_mask] = False
    values["deletion_value"][:, design_mask] = 0
    values["msa_paired"][:, design_mask] = 0
    values["msa_profile"][design_mask] = 0
    values["msa_profile"][design_mask, unknown] = 1
    values["deletion_mean"][design_mask] = 0
    return values


def _reference_coordinates(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
    *,
    generated_components: frozenset[int] = frozenset(),
) -> Tensor:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    index = torch.tensor(atom_indices, device=system.atom_symbol.device)
    coordinates = system.reference_coordinates[index].detach().cpu().clone()
    present = system.reference_coordinate_mask[index].detach().cpu().clone()
    atom_component = _atom_component_index(system).detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()

    if generated_components:
        glycine = get_component("GLY")
        glycine_coordinates = {
            atom.atom_id: glycine.reference_coordinates[local_index]
            for local_index, atom in enumerate(glycine.atoms)
            if glycine.reference_coordinates[local_index] is not None
        }
        for row, atom_index in enumerate(atom_indices):
            if atom_component[atom_index] not in generated_components:
                continue
            atom_id = system.symbols[atom_symbols[atom_index]]
            coordinate = glycine_coordinates.get(atom_id)
            if coordinate is not None:
                coordinates[row] = coordinates.new_tensor(coordinate)
                present[row] = True

    missing = {atom_index for atom_index, value in zip(atom_indices, present.tolist(), strict=True) if not value}
    if not missing:
        return coordinates

    instance_components = system.instance_component_ptr.detach().cpu().tolist()
    instance_entities = system.instance_entity.detach().cpu().tolist()
    component_atoms = system.component_atom_ptr.detach().cpu().tolist()
    for instance_index, entity_index in enumerate(instance_entities):
        if int(system.entity_type[entity_index]) != ENTITY_MOLECULE:
            continue
        atom_start = component_atoms[instance_components[instance_index]]
        atom_stop = component_atoms[instance_components[instance_index + 1]]
        if not any(atom_index in missing for atom_index in range(atom_start, atom_stop)):
            continue
        scope_atoms = tuple(range(atom_start, atom_stop))
        edge_indices = (
            (
                (system.edge_index[0] >= atom_start)
                & (system.edge_index[1] < atom_stop)
                & (system.bond_role <= BOND_ROLE_ENTITY)
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        molecule = Chem.AddHs(
            _canonical_rdkit_molecule(
                system,
                scope_atoms,
                edge_indices.detach().cpu().tolist(),
            )
        )
        options: Any = rdDistGeom.ETKDGv3()
        options.randomSeed = 42
        conformer_id = rdDistGeom.EmbedMolecule(molecule, options)
        if conformer_id == -1:
            options.useRandomCoords = True
            conformer_id = rdDistGeom.EmbedMolecule(molecule, options)
        rdForceFieldHelpers.UFFOptimizeMolecule(molecule, confId=conformer_id, maxIters=1000)
        conformer = molecule.GetConformer(conformer_id)
        generated = {atom_index: conformer.GetAtomPosition(atom_index - atom_start) for atom_index in scope_atoms}
        for row, atom_index in enumerate(atom_indices):
            position = generated.get(atom_index)
            if position is None or atom_index not in missing:
                continue
            coordinates[row] = coordinates.new_tensor((position.x, position.y, position.z))
            present[row] = True

    if not bool(present.all()):
        unresolved = [atom_index for atom_index, value in zip(atom_indices, present.tolist(), strict=True) if not value]
        raise ValueError(f"BoltzGen requires reference coordinates for canonical atoms {unresolved}")
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
    for space_id in range(int(space_ids.max().item()) + 1):
        included = space_ids == space_id
        mask = resolved[included]
        if not bool(mask.any()):
            continue
        values = positions[included]
        center = (values * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
        random_device = generator.device if generator is not None else values.device
        quaternion = torch.randn(
            4,
            dtype=values.dtype,
            device=random_device,
            generator=generator,
        ).to(values.device)
        norm = torch.sqrt((quaternion * quaternion).sum())
        quaternion = quaternion / torch.copysign(norm, quaternion[0])
        real, i, j, k = quaternion
        two = 2.0 / (quaternion * quaternion).sum()
        rotation = torch.stack(
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
            )
        ).reshape(3, 3)
        translation = torch.randn(
            1,
            3,
            dtype=values.dtype,
            device=random_device,
            generator=generator,
        ).to(values.device)
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
            raise ValueError(f"Boltz-1 atom name cannot be encoded in four characters: {name!r}")
        names.append(name)
    return names


def _canonical_rdkit_molecule(
    system: MolecularSystem,
    atom_indices: tuple[int, ...],
    edge_indices: Sequence[int],
) -> Mol:
    r"""Project one canonical graph scope into RDKit."""

    from rdkit import Chem

    editable = Chem.RWMol()
    selected = torch.tensor(atom_indices, device=system.atom_symbol.device)
    canonical_to_local: dict[int, int] = {}
    atom_stereo: dict[int, str] = {}
    for source_row, atom_index in enumerate(atom_indices):
        atomic_number = int(system.atomic_number[selected[source_row]])
        atom = Chem.Atom(atomic_number)
        if bool(system.isotope_known[atom_index]):
            atom.SetIsotope(int(system.isotope[atom_index]))
        if bool(system.formal_charge_known[atom_index]):
            atom.SetFormalCharge(int(system.formal_charge[atom_index]))
        if bool(system.radical_electrons_known[atom_index]):
            atom.SetNumRadicalElectrons(int(system.radical_electrons[atom_index]))
        if bool(system.explicit_hydrogens_known[atom_index]):
            atom.SetNumExplicitHs(int(system.explicit_hydrogens[atom_index]))
        if bool(system.implicit_hydrogens_allowed_known[atom_index]):
            atom.SetNoImplicit(not bool(system.implicit_hydrogens_allowed[atom_index]))
        if bool(system.atom_aromatic_known[atom_index]):
            atom.SetIsAromatic(bool(system.atom_aromatic[atom_index]))
        local_index = editable.AddAtom(atom)
        canonical_to_local[atom_index] = local_index
        stereo_index = int(system.atom_stereo[atom_index])
        if stereo_index >= 0:
            atom_stereo[local_index] = system.symbols[stereo_index].upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for edge_index in edge_indices:
        first = int(system.edge_index[0, edge_index])
        second = int(system.edge_index[1, edge_index])
        if not bool(system.bond_order_known[edge_index]):
            raise ValueError("BoltzGen RDKit projection requires known bond order")
        editable.AddBond(
            canonical_to_local[first],
            canonical_to_local[second],
            bond_types[float(system.bond_order[edge_index])],
        )
        local_bond_index = editable.GetNumBonds() - 1
        bond = editable.GetBondWithIdx(local_bond_index)
        if bool(system.bond_aromatic_known[edge_index]):
            bond.SetIsAromatic(bool(system.bond_aromatic[edge_index]))
        stereo_index = int(system.bond_stereo[edge_index])
        if stereo_index >= 0:
            stereo_rows = system.bond_stereo_index[:, edge_index]
            stereo_atoms = (
                (
                    canonical_to_local[int(stereo_rows[0])],
                    canonical_to_local[int(stereo_rows[1])],
                )
                if bool((stereo_rows >= 0).all())
                else None
            )
            bond_stereo.append(
                (
                    local_bond_index,
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
                max(
                    begin_neighbors,
                    key=lambda atom: int(atom.GetProp("_CIPRank")),
                ).GetIdx(),
                max(
                    end_neighbors,
                    key=lambda atom: int(atom.GetProp("_CIPRank")),
                ).GetIdx(),
            )
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo_types[stereo.upper()])
    Chem.AssignStereochemistry(value, force=True, cleanIt=True)
    return value


def _cyclic_period(system: MolecularSystem, instance_index: int) -> int:
    entity_index = int(system.instance_entity[instance_index])
    if int(system.entity_type[entity_index]) == ENTITY_MOLECULE:
        return 0
    component_start = int(system.instance_component_ptr[instance_index])
    component_stop = int(system.instance_component_ptr[instance_index + 1])
    component_count = component_stop - component_start
    if component_count <= 2:
        return 0
    atom_component = _atom_component_index(system)
    edge_components = atom_component[system.edge_index]
    if bool(
        (
            ((edge_components[0] == component_start) & (edge_components[1] == component_stop - 1))
            | ((edge_components[1] == component_start) & (edge_components[0] == component_stop - 1))
        ).any()
    ):
        return component_count
    return 0


def _residue_type(component_id: str, polymer_type: str) -> int:
    if polymer_type == "protein":
        return _PROTEIN_RESIDUE_TYPES.get(component_id, 22)
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(component_id, 27)
    return _DNA_RESIDUE_TYPES.get(component_id, 32)


def _validate_msa(
    msa: MSA | MsaCollection | None,
    system: MolecularSystem,
) -> MSA | MsaCollection | None:
    if msa is None:
        return None
    # Evidence holds no PolymerSequence: each column names one entity by ``entity_id`` (and an optional exact
    # target instance) and closes over the primary system supplied here. The shared resolver binds the entity,
    # verifies any exact target instance, and checks the query width and query chemistry against the target -- so
    # an MSA bound to an unknown instance raises here instead of silently lowering to an all-gap row.
    resolve_msa_bindings(msa, system)
    return msa
