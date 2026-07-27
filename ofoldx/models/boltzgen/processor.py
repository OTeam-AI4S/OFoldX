r"""
Input processing for BoltzGen structure design and inverse folding.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from ...data.components import get_component
from ...data.conditions import Contact, Endpoint
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import (
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
)
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    InstanceIdAllocator,
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
    BoltzGenCovalentBond,
    BoltzGenDesignMask,
    BoltzGenDirectionalContact,
    BoltzGenInput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingLayout,
    BoltzGenLayout,
    BoltzGenPlacements,
    BoltzGenResidueConstraint,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenStructureTarget,
    BoltzGenSymmetry,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


# Contact-conditioning one-hot classes consumed by BoltzContactConditioning:
# 0 UNSPECIFIED, 1 UNSELECTED, 2 POCKET>BINDER, 3 BINDER>POCKET, 4 CONTACT.
_CONTACT_CLASSES = 5
_CONTACT_KIND_TO_CLASS = {"pocket>binder": 2, "binder>pocket": 3, "contact": 4}

# Conditioning ids consumed by the BoltzGen input embeddings.
_BINDING_UNSPECIFIED = 0
_BINDING_BINDING = 1
_BINDING_NOT_BINDING = 2
_SS_UNSPECIFIED = 3  # num_ss_types == 4; class 3 taken as the masked/unspecified label.
_SS_LABEL_TO_ID = {"H": 0, "E": 1, "L": 2}


@dataclass(eq=False)
class _BoltzGenDesignPlan:
    r"""
    Materialized BoltzGen structure-design domain state.
    """

    system: MolecularSystem
    source_structure: Structure | None
    spatial_conditions: tuple[BoltzGenSchema, ...]
    placement_map: PlacementMap
    instance_of: dict[str, str]
    generated_residues: frozenset[ComponentReference]


class BoltzGenProcessor(Processor):
    r"""
    Convert supported structure-design inputs into `BoltzGenInput`.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: BoltzGenPlacements | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MSACollection | None = None,
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
            A [`BoltzGenInput`][] with batch size one and token, atom, evidence,
            condition, and molecular-layout axes aligned.
        """

        plan = self._plan(
            input,
            placement=placement,
            total_len=total_len,
            conditions=conditions,
        )
        processed, _ = self._lower(plan, msa, templates=templates, generator=generator)
        return processed

    def process_training(
        self,
        input: Structure,
        *,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MSACollection | None = None,
        templates: Sequence[Template] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInput, BoltzGenStructureTarget]:
        r"""
        Lower one complete structure into masked Model input and separate truth.

        Structure training redesigns residues already present in ``input``; generated
        placements have no target coordinates and therefore belong to inference.
        """

        plan = self._plan(
            input,
            placement=None,
            total_len=None,
            conditions=conditions,
        )
        processed, target = self._lower(
            plan,
            msa,
            templates=templates,
            generator=generator,
        )
        assert target is not None
        return processed, target

    def _plan(
        self,
        input: MolecularSystem | Structure | None,
        *,
        placement: BoltzGenPlacements | None,
        total_len: int | None,
        conditions: Sequence[BoltzGenSchema],
    ) -> _BoltzGenDesignPlan:
        r"""
        Materialize the final design system and placement identities once.
        """

        condition_values = tuple(conditions)
        for condition in condition_values:
            if not isinstance(condition, BoltzGenSchema):
                raise TypeError(f"BoltzGen cannot consume {type(condition).__name__}")
        covalent_bonds = tuple(
            condition for condition in condition_values if isinstance(condition, BoltzGenCovalentBond)
        )
        spatial_conditions = tuple(
            condition for condition in condition_values if not isinstance(condition, BoltzGenCovalentBond)
        )
        system, structure = _normalize_structure_input(input)

        placement_map: PlacementMap = {}
        instance_of: dict[str, str] = {}
        generated_residues: frozenset[ComponentReference] = frozenset()
        if system is None:
            # A fully generated assembly uses explicit placements or one default chain.
            if placement is None and total_len is None:
                raise ValueError("a fully-generated BoltzGen assembly requires placement or a total_len length budget")
            assembly_placements = placement.placements if placement is not None else _default_assembly(total_len)
            system, placement_map, instance_of = _materialize_assembly(assembly_placements, total_len)
            generated_residues = frozenset(
                reference
                for group in placement_map.values()
                for occurrences in group.values()
                for reference in occurrences
            )
            structure = None
        elif placement is not None or total_len is not None:
            # Existing instances retain their identities and coordinates; generated
            # chains are appended with zero coordinates before global centering.
            extension_placements = None if placement is None else placement.placements
            system, placement_map, instance_of, generated_residues = _materialize_extension(
                system, extension_placements, total_len
            )
        # With no placement or length budget, the input system is used unchanged.
        if covalent_bonds:
            system = _materialize_covalent_bonds(system, covalent_bonds, placement_map, instance_of)

        return _BoltzGenDesignPlan(
            system=system,
            source_structure=structure,
            spatial_conditions=spatial_conditions,
            placement_map=placement_map,
            instance_of=instance_of,
            generated_residues=generated_residues,
        )

    def _lower(
        self,
        plan: _BoltzGenDesignPlan,
        msa: MSA | MSACollection | None,
        *,
        templates: Sequence[Template] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInput, BoltzGenStructureTarget | None]:
        r"""
        Validate canonical evidence and lower one materialized plan to tensors.
        """

        system = plan.system
        if not system.instances:
            raise ValueError("BoltzGen requires at least one molecular instance")
        msa = _validate_msa(msa, system)
        # Generated protein placeholders use backbone-only GLY reference coordinates.
        tokens, _ = _build_tokens(system, generated_residues=plan.generated_residues)
        token_count = len(tokens)
        site_to_token = {
            site: token_index
            for token_index, token in enumerate(tokens)
            for site in cast(tuple[AtomSite, ...], token["atom_sites"])
        }
        token_index_by_residue = {
            (cast(str, token["instance_id"]), cast(int, token["residue_index"])): token_index
            for token_index, token in enumerate(tokens)
            if token["monomer_index"] is not None
        }
        token_bonds = _token_bond_indicator(system, site_to_token, token_count)
        contact_conditioning, contact_threshold = _contact_channels(
            plan.spatial_conditions,
            token_index_by_residue,
            site_to_token,
            plan.placement_map,
            plan.instance_of,
            token_count,
        )
        design_mask = _design_mask(
            plan.spatial_conditions,
            token_index_by_residue,
            token_count,
            plan.generated_residues,
        )
        _validate_structure_design_rows(
            system,
            tokens,
            design_mask,
            plan.generated_residues,
        )
        (
            atom_values,
            model_atom_sites,
            target_source_sites,
            site_to_atom,
        ) = _build_structure_design_atom_features(
            system,
            tokens,
            plan.source_structure,
            design_mask,
            generated_residues=plan.generated_residues,
            generator=generator,
        )
        token_center_atom_index = torch.tensor(
            [site_to_atom[cast(AtomSite, token["center_site"])] for token in tokens],
            dtype=torch.long,
        )
        token_to_bb4 = torch.stack(
            [
                torch.tensor(
                    [site_to_atom[site] for site in cast(tuple[AtomSite, ...], token["bb4_sites"])],
                    dtype=torch.long,
                )
                for token in tokens
            ]
        )
        frame_atom_index = _frame_atom_indices(tokens, site_to_atom)
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
            [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens],
            dtype=torch.float32,
        )
        chain_design_mask = _chain_design_mask(
            design_mask,
            torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
            token_bonds,
        )
        binding_type = _binding_ids(
            plan.spatial_conditions,
            token_index_by_residue,
            site_to_token,
            plan.placement_map,
            plan.instance_of,
            token_count,
        )
        ss_type = _ss_ids(plan.spatial_conditions, token_index_by_residue, token_count)
        symmetry_index = _symmetry_index(plan.spatial_conditions, token_index_by_residue, tokens)
        # Structure groups affect pairwise geometry through ``token_distance_mask``;
        # atom reference-space ids remain per-residue.
        structure_group = _structure_group_ids(
            plan.spatial_conditions,
            token_index_by_residue,
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
        layout = BoltzGenLayout(
            system=system,
            atom_sites=model_atom_sites,
            token_sites=tuple(cast("AtomSite | tuple[str, int]", token["token_site"]) for token in tokens),
            output_residues=tuple(cast("ComponentReference", token["reference"]) for token in tokens),
            output_instance_ids=tuple(cast(str, token["instance_id"]) for token in tokens),
            placement_map=plan.placement_map,
            instance_of=plan.instance_of,
        )
        processed = BoltzGenInput(
            **{name: value.unsqueeze(0) for name, value in placed.items()},
            layout=layout,
            batch_size=[1],
        )
        if plan.source_structure is None:
            return processed, None

        source = plan.source_structure
        truth_values = _truth_pool_features(
            system,
            tokens,
            source,
            target_source_sites,
            atom_values["atom_mask"].shape[0],
            site_to_atom,
        )
        target_b_factor = _token_b_factors(source, tokens)
        bond_atom_index = _inter_residue_bond_atom_index(system, site_to_atom)
        center_pool_index = truth_values["atom_index"][token_center_atom_index]
        target_values: dict[str, Tensor] = {
            **truth_values,
            "distogram_coordinates": truth_values["coordinates"][center_pool_index],
            "distogram_mask": truth_values["coordinate_mask"][center_pool_index],
            "residue_type": target_residue_type,
            "fake_atom_mask": atom_values["fake_atom_mask"],
        }
        if target_b_factor is not None:
            target_values["b_factor"] = target_b_factor
        if bond_atom_index is not None:
            target_values["bond_atom_index"] = bond_atom_index
            atom_index = truth_values["atom_index"]
            valid_atom_index = atom_index >= 0
            atom_coordinate_mask = truth_values["coordinate_mask"].gather(
                0,
                atom_index.clamp_min(0),
            )
            atom_coordinate_mask = atom_coordinate_mask & valid_atom_index
            target_values["bond_mask"] = atom_coordinate_mask[bond_atom_index].all(dim=-1)
        placed_target = {
            name: (
                value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(device=self.device)
            )
            for name, value in target_values.items()
        }
        target = BoltzGenStructureTarget(
            coordinates=placed_target["coordinates"].unsqueeze(0).unsqueeze(0),
            coordinate_mask=placed_target["coordinate_mask"].unsqueeze(0),
            atom_index=placed_target["atom_index"].unsqueeze(0),
            chain_span=placed_target["chain_span"].unsqueeze(0),
            chain_equivalence_index=placed_target["chain_equivalence_index"].unsqueeze(0),
            symmetry_atom_index=placed_target["symmetry_atom_index"].unsqueeze(0),
            symmetry_permutation=placed_target["symmetry_permutation"].unsqueeze(0),
            distogram_coordinates=placed_target["distogram_coordinates"].unsqueeze(0).unsqueeze(0),
            distogram_mask=placed_target["distogram_mask"].unsqueeze(0),
            residue_type=placed_target["residue_type"].unsqueeze(0),
            fake_atom_mask=placed_target["fake_atom_mask"].unsqueeze(0),
            b_factor=(placed_target["b_factor"].unsqueeze(0) if "b_factor" in placed_target else None),
            bond_atom_index=(
                placed_target["bond_atom_index"].unsqueeze(0) if "bond_atom_index" in placed_target else None
            ),
            bond_mask=(placed_target["bond_mask"].unsqueeze(0) if "bond_mask" in placed_target else None),
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

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: Structure,
        *,
        design: Sequence[ComponentReference] = (),
        global_omit_tokens: Sequence[str] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MSACollection | None = None,
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

        if not isinstance(input, Structure):
            raise TypeError("BoltzGenInverseFoldingProcessor requires a fixed backbone Structure")
        system = input.system
        if not system.instances:
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
        msa: MSA | MSACollection | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[BoltzGenInverseFoldingInput, Tensor]:
        r"""
        Lower one inverse-folding example and its unmasked residue target.

        The returned input masks every design position to the released unknown
        protein token. The aligned target remains separate for teacher forcing
        and the residue objective.
        """

        if not isinstance(input, Structure):
            raise TypeError("BoltzGenInverseFoldingProcessor requires a fixed backbone Structure")
        system = input.system
        if not system.instances:
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
        msa: MSA | MSACollection | None,
        design: tuple[ComponentReference, ...],
        global_omit_tokens: tuple[str, ...],
        aa_constraints: tuple[BoltzGenResidueConstraint, ...],
        symmetry: tuple[tuple[ComponentReference, ...], ...],
        generator: torch.Generator | None,
    ) -> tuple[BoltzGenInverseFoldingInput, Tensor]:
        system = structure.system
        # Generated ``UNK`` protein residues use backbone-only GLY reference coordinates.
        generated_residues = _generated_placeholder_residues(system)
        tokens, atom_sites = _build_tokens(system, generated_residues=generated_residues)
        token_count = len(tokens)
        token_index_by_residue = {
            (cast(str, token["instance_id"]), cast(int, token["residue_index"])): token_index
            for token_index, token in enumerate(tokens)
            if token["monomer_index"] is not None
        }
        molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
        for reference in design:
            _require_protein_reference(reference, token_index_by_residue, molecule_types, "design")
        for constraint in aa_constraints:
            _require_protein_reference(
                constraint.position,
                token_index_by_residue,
                molecule_types,
                "amino-acid constraint",
            )
        for group in symmetry:
            for reference in group:
                _require_protein_reference(
                    reference,
                    token_index_by_residue,
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
            site_to_atom,
        ) = _build_atom_features(
            system,
            tokens,
            atom_sites,
            structure,
            generated_residues=generated_residues,
            generator=generator,
        )
        site_to_token = {
            site: token_index
            for token_index, token in enumerate(tokens)
            for site in cast(tuple[AtomSite, ...], token["atom_sites"])
        }
        token_bonds = _token_bond_indicator(system, site_to_token, token_count)
        msa_values = _msa_features(msa, tokens, system)
        residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
        cyclic_period = torch.tensor(
            [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens], dtype=torch.float32
        )
        token_to_bb4 = torch.stack(
            [
                torch.tensor(
                    [site_to_atom[site] for site in cast(tuple[AtomSite, ...], token["bb4_sites"])], dtype=torch.long
                )
                for token in tokens
            ]
        )
        design_mask = _design_mask_from_positions(design, token_index_by_residue, token_count)
        target_residue_type = torch.nn.functional.one_hot(residue_types, 33).float()
        residue_type_features = target_residue_type.clone()
        residue_type_features[design_mask] = 0
        residue_type_features[design_mask, _UNKNOWN_RESIDUE_TYPES["protein"]] = 1
        token_center_atom_index = torch.tensor(
            [site_to_atom[cast(AtomSite, token["center_site"])] for token in tokens],
            dtype=torch.long,
        )
        token_resolved_mask = resolved[token_center_atom_index]
        aa_constraint_mask = _aa_constraint_mask(
            global_omit_tokens,
            aa_constraints,
            token_index_by_residue,
            token_count,
        )
        symmetric_group = _sequence_symmetry_index(symmetry, token_index_by_residue, token_count)

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
        layout = BoltzGenInverseFoldingLayout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(cast("AtomSite | tuple[str, int]", token["token_site"]) for token in tokens),
            output_residues=tuple(cast("ComponentReference", token["reference"]) for token in tokens),
            output_instance_ids=tuple(cast(str, token["instance_id"]) for token in tokens),
        )
        return (
            BoltzGenInverseFoldingInput(
                **{name: value.unsqueeze(0) for name, value in placed.items()},
                layout=layout,
                batch_size=[1],
            ),
            target_residue_type.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        )


@dataclass(frozen=True)
class DesignMaskResidue:
    r"""
    Describe one generated or retained residue in a BoltzGen design mask.
    """

    designed: bool
    polymer_type: PolymerType | None = None
    source: ComponentReference | None = None


def from_design_mask(
    residues: Sequence[DesignMaskResidue],
    *,
    group_of: Callable[[int], str],
    placement_id_of: Callable[[int], str],
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

    for index, residue in enumerate(residues):
        group = group_of(index)
        placement_id = placement_id_of(index)
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
    designed = run[0].designed
    if any(residue.designed != designed for residue in run):
        raise ValueError(
            f"placement {placement_id!r} mixes designed and fixed residues; a placement_id must be "
            "either wholly designed (GeneratedSegment) or wholly fixed (ExistingSpan)"
        )

    if designed:
        polymer_type = run[0].polymer_type
        if polymer_type is None:
            raise ValueError(
                f"designed placement {placement_id!r} has no polymer_type: a designed non-polymer "
                "(GAP-1 de-novo ligand) has no PlacementValue and must be routed out-of-band"
            )
        if any(residue.polymer_type != polymer_type for residue in run):
            raise ValueError(
                f"designed placement {placement_id!r} spans more than one polymer type; a maximal "
                "designed run is one polymer type"
            )
        if any(residue.source is not None for residue in run):
            raise ValueError(f"designed placement {placement_id!r} carries a source reference")
        return GeneratedSegment(polymer_type, len(run), len(run))

    sources: list[ComponentReference] = []
    for residue in run:
        if residue.source is None:
            raise ValueError(f"fixed placement {placement_id!r} has a residue with no source reference")
        sources.append(residue.source)
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
        return input, None
    raise TypeError("BoltzGen input must be a MolecularSystem, Structure, or None")


def _component_reference_to_token(
    reference: ComponentReference,
    token_index: dict[tuple[str, int], int],
) -> int:
    r"""
    Resolve one ``(instance_id, component_index)`` reference to its token row.
    """

    key = (reference[0], reference[1])
    if key not in token_index:
        raise ValueError(f"condition references residue {reference!r} not present in the lowered system")
    return token_index[key]


# --------------------------------------------------------------------------------------
# Whole-assembly ``total_len`` realization.
# --------------------------------------------------------------------------------------

# Placeholder component identities used for generated polymer residues.
_GENERATED_COMPONENT = {"protein": "UNK", "rna": "N", "dna": "DN"}


def _default_assembly(total_len: int | None) -> PlacementSpec:
    r"""
    The degenerate whole-assembly PlacementSpec when only a ``total_len`` budget is given.

    One declaration group ``"assembly"`` with one generated protein placement ``"chain_A"``
    spanning the whole ``total_len`` budget.
    """

    if total_len is None or total_len <= 0:
        raise ValueError("a BoltzGen assembly without a BoltzGenPlacements requires a positive total_len")
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
                    f"placement {placement_id!r} has an invalid length range " f"[{part.min_length}, {part.max_length}]"
                )
            segments.append((placement_id, part))

    if total_len is None:
        return {placement_id: part.max_length for placement_id, part in segments}

    total_min = sum(part.min_length for _, part in segments)
    total_max = sum(part.max_length for _, part in segments)
    if not total_min <= total_len <= total_max:
        raise ValueError(
            f"total_len={total_len} is outside the assembly's generated-length range " f"[{total_min}, {total_max}]"
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
    allocator = InstanceIdAllocator()
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
            instance_id = allocator.allocate()
            entity_id = f"boltzgen_entity_{entity_index}"
            entity_index += 1
            component = get_component(_GENERATED_COMPONENT[polymer_type])
            entities[entity_id] = PolymerSequence(polymer_type, (component,) * length)
            instances[instance_id] = entity_id
            realized_group[placement_id] = tuple((instance_id, index) for index in range(length))
        placement_map[group] = realized_group

    if not instances:
        raise ValueError("a BoltzGen assembly must realize at least one non-empty generated placement")
    system = MolecularSystem(entities, instances)
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
    system: MolecularSystem,
    placements: PlacementSpec | None,
    total_len: int | None,
) -> tuple[MolecularSystem, PlacementMap, dict[str, str], frozenset[ComponentReference]]:
    r"""
    Extend an existing input system with generated rows.

    Existing instances retain their ids, entities, coordinates, and bonds.
    [`ExistingSpan`][] occurrences address those source rows directly.
    [`GeneratedSegment`][] values append fresh non-colliding ``chain_{n}``
    instances in declaration order.

    ``placements is None`` is the ``total_len`` extension: one generated protein chain of length
    ``total_len`` is appended (see [`_default_extension`][]). A ``total_len`` given alongside an
    explicit [`BoltzGenPlacements`][ofoldx.models.boltzgen.model.BoltzGenPlacements] constrains
    the sum of generated lengths.

    Returns the extended system, the realized ``placement_map`` (generated occurrences under minted
    ``chain_{n}`` ids, ExistingSpan occurrences under reused source ids), the derived
    ``placement_id -> declaration_group`` index, and the frozenset of GENERATED output residues
    (the design positions — an ExistingSpan occurrence is NOT in this set).
    """

    if placements is None:
        placements = _default_extension(total_len)

    instance_of = _placement_instance_of(placements)  # validates global placement_id uniqueness
    lengths = _resolve_generated_lengths(placements, total_len)

    source_ids = set(system.instances)
    entities: dict[str, PolymerSequence | Molecule] = dict(system.entities)
    instances: dict[str, str] = dict(system.instances)  # source instances remain first
    generated: set[ComponentReference] = set()
    placement_map: PlacementMap = {}

    allocator = InstanceIdAllocator()

    def _mint_instance() -> str:
        while True:
            candidate = allocator.allocate()
            if candidate not in source_ids and candidate not in instances:
                return candidate

    entity_index = 0
    for group, chain in placements.items():
        realized_group: dict[str, tuple[ComponentReference, ...]] = {}
        for placement_id, part in chain.items():
            if isinstance(part, ExistingSpan):
                if part.instance_id not in source_ids:
                    raise ValueError(
                        f"ExistingSpan placement {placement_id!r} names instance {part.instance_id!r}, "
                        "which is not an instance of the input system"
                    )
                source_entity = system.entities[system.instances[part.instance_id]]
                if not isinstance(source_entity, PolymerSequence):
                    raise ValueError(
                        f"ExistingSpan placement {placement_id!r} names non-polymer instance " f"{part.instance_id!r}"
                    )
                source_length = len(source_entity)
                if not (0 <= part.start <= part.stop <= source_length):
                    raise ValueError(
                        f"ExistingSpan placement {placement_id!r} interval [{part.start}, {part.stop}) is "
                        f"outside source instance {part.instance_id!r} of length {source_length}"
                    )
                # Existing spans retain their source residue identities.
                realized_group[placement_id] = tuple(
                    (part.instance_id, index) for index in range(part.start, part.stop)
                )
                continue
            if not isinstance(part, GeneratedSegment):
                raise ValueError(f"unsupported placement value for {placement_id!r}: {part!r}")
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
            instance_id = _mint_instance()
            entity_id = f"boltzgen_entity_{entity_index}"
            while entity_id in entities:
                entity_index += 1
                entity_id = f"boltzgen_entity_{entity_index}"
            entity_index += 1
            component = get_component(_GENERATED_COMPONENT[polymer_type])
            entities[entity_id] = PolymerSequence(polymer_type, (component,) * length)
            instances[instance_id] = entity_id
            occurrences = tuple((instance_id, index) for index in range(length))
            realized_group[placement_id] = occurrences
            generated.update(occurrences)
        placement_map[group] = realized_group

    if not generated:
        raise ValueError(
            "extending an input system must realize at least one non-empty generated placement " "(nothing to design)"
        )
    extended = MolecularSystem(entities, instances, system.bonds)
    return extended, placement_map, instance_of, frozenset(generated)


def _materialize_covalent_bonds(
    system: MolecularSystem,
    covalent_bonds: tuple[BoltzGenCovalentBond, ...],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> MolecularSystem:
    r"""
    Resolve requested [`SymbolicBond`][] links and inject them into ``system.bonds``.

    Each [`BoltzGenCovalentBond`][ofoldx.models.boltzgen.model.BoltzGenCovalentBond] wraps a
    [`SymbolicBond`][ofoldx.data.bonds.SymbolicBond] whose endpoints are lowered to realized
    [`AtomSite`][ofoldx.data.molecular.AtomSite] occurrences (an existing ``AtomSite`` passes through;
    a generated [`SymbolicAtomSite`][ofoldx.data.placement.SymbolicAtomSite] resolves through
    ``placement_map`` / ``instance_of``). The realized [`SystemBond`][ofoldx.data.molecular.SystemBond]
    is added to the system so ``_cyclic_period`` sees it like any input bond; a bond whose
    unordered endpoint set is already present is not duplicated. Endpoint validity is enforced by
    ``MolecularSystem`` construction (each endpoint atom must resolve).
    """

    existing = set(system.bonds)
    present = {frozenset((bond.site_a, bond.site_b)) for bond in existing}
    added = set(existing)
    for condition in covalent_bonds:
        realized = condition.bond.resolve(placement_map, instance_of)
        if frozenset((realized.site_a, realized.site_b)) in present:
            continue
        added.add(realized)
    if added == existing:
        return system
    return MolecularSystem(system.entities, system.instances, frozenset(added))


def _generated_placeholder_residues(system: MolecularSystem) -> frozenset[ComponentReference]:
    r"""
    Return polymer residues carrying a generated placeholder identity.

    Protein ``UNK`` placeholders have no reference conformer and therefore use the
    GLY backbone template during inverse folding. Fixed-context residues are excluded.
    """

    placeholders: set[ComponentReference] = set()
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if not isinstance(entity, PolymerSequence):
            continue
        placeholder = _GENERATED_COMPONENT.get(entity.polymer_type)
        if placeholder is None:
            continue
        for index, component in enumerate(entity.components):
            if component.component_id == placeholder:
                placeholders.add((instance_id, index))
    return frozenset(placeholders)


def _generated_reference_coordinates(
    system: MolecularSystem,
    atom_sites: tuple[AtomSite, ...],
    generated_residues: frozenset[ComponentReference],
) -> list[tuple[float, float, float]]:
    r"""
    Reference coordinates for an assembly containing generated placement residues.

    Generated protein backbones use GLY template coordinates. Other residues use
    their component conformer; molecule components without one use an RDKit conformer.
    """

    glycine = get_component("GLY")
    gly_backbone = {
        atom.atom_id: glycine.reference_coordinates[index]
        for index, atom in enumerate(glycine.atoms)
        if glycine.reference_coordinates and glycine.reference_coordinates[index] is not None
    }
    generated_molecules: dict[str, dict[tuple[int, str], tuple[float, float, float]]] = {}
    coordinates: list[tuple[float, float, float]] = []
    for site in atom_sites:
        entity_id = system.instances[site.instance_id]
        entity = system.entities[entity_id]
        residue = (site.instance_id, site.component_index)
        is_generated_protein = (
            isinstance(entity, PolymerSequence) and entity.polymer_type == "protein" and residue in generated_residues
        )
        if is_generated_protein:
            coordinate = gly_backbone.get(site.atom_id)
            if coordinate is None:
                raise ValueError(f"generated protein backbone has no GLY template coordinate for {site!r}")
            coordinates.append(coordinate)
            continue
        # Molecules without component coordinates use an RDKit conformer.
        component = entity.components[site.component_index]
        atom_index = next(index for index, atom in enumerate(component.atoms) if atom.atom_id == site.atom_id)
        coordinate = component.reference_coordinates[atom_index] if component.reference_coordinates else None
        if coordinate is None and isinstance(entity, Molecule):
            if entity_id not in generated_molecules:
                generated_molecules[entity_id] = _molecule_reference_coordinates(entity)
            coordinate = generated_molecules[entity_id][(site.component_index, site.atom_id)]
        if coordinate is None:
            raise ValueError(f"BoltzGen requires a reference coordinate for {site!r}")
        coordinates.append(coordinate)
    return coordinates


def _build_tokens(
    system: MolecularSystem,
    generated_residues: frozenset[ComponentReference] = frozenset(),
) -> tuple[list[dict[str, object]], tuple[AtomSite, ...]]:
    r"""
    Build per-token records and their natural heavy-atom order.

    Polymer residues each occupy one token; non-polymer atoms each occupy one token, matching the
    Boltz tokenization scheme.

    ``generated_residues`` names whole-assembly generated placement residues;
    a generated PROTEIN residue is restricted to its backbone ``{N, CA, C, O}`` atoms (it has
    a placeholder ``UNK`` identity and no side-chain conformer). An empty set leaves
    existing-input tokenization unchanged.
    """

    available_sites = {site for site, atom in system.iter_atoms() if atom.element != "H"}
    entity_index_by_id: dict[str, int] = {}
    instance_count_by_entity: dict[str, int] = {}
    tokens: list[dict[str, object]] = []

    for chain_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        entity_index = entity_index_by_id.setdefault(entity_id, len(entity_index_by_id))
        symmetry_index = instance_count_by_entity.get(entity_id, 0)
        instance_count_by_entity[entity_id] = symmetry_index + 1
        if isinstance(entity, PolymerSequence):
            molecule_type = _MOLECULE_TYPES[entity.polymer_type]
            standards = _STANDARD_COMPONENTS[entity.polymer_type]
            for monomer_index, component in enumerate(entity.components):
                component_id = component.component_id
                is_generated_protein = (
                    entity.polymer_type == "protein" and (instance_id, monomer_index) in generated_residues
                )
                atom_sites = tuple(
                    site
                    for atom in component.atoms
                    if atom.element != "H"
                    and atom.atom_id not in {"OXT", "OP3"}
                    and not (is_generated_protein and atom.atom_id not in {"N", "CA", "C", "O"})
                    and (site := AtomSite(instance_id, monomer_index, atom.atom_id)) in available_sites
                )
                if not atom_sites:
                    raise ValueError(f"component {component_id!r} has no model-visible heavy atoms")
                is_standard = component_id in standards
                residue_type = (
                    _residue_type(component_id, entity.polymer_type)
                    if is_standard
                    else _UNKNOWN_RESIDUE_TYPES[entity.polymer_type]
                )
                by_name = {site.atom_id: site for site in atom_sites}
                if entity.polymer_type == "protein":
                    center_name = "CA"
                    bb4_names = ("N", "CA", "C", "O")
                    frame_names = ("N", "CA", "C")
                else:
                    center_name = "C1'"
                    bb4_names = ("C1'", "C3'", "C4'", "C1'")
                    frame_names = ("C1'", "C3'", "C4'")
                center_site = by_name.get(center_name, atom_sites[0])
                tokens.append(
                    {
                        "atom_sites": atom_sites,
                        "token_site": (instance_id, monomer_index),
                        "reference": (instance_id, monomer_index),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": monomer_index,
                        "residue_index": monomer_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": molecule_type,
                        "residue_type": residue_type,
                        "center_site": center_site,
                        "bb4_sites": tuple(by_name.get(name, center_site) for name in bb4_names),
                        "frame_sites": tuple(by_name.get(name, atom_sites[0]) for name in frame_names),
                    }
                )
        else:
            atom_sites = tuple(
                site
                for component_index, component in enumerate(entity.components)
                for atom in component.atoms
                if atom.element != "H"
                and (site := AtomSite(instance_id, component_index, atom.atom_id)) in available_sites
            )
            if not atom_sites:
                raise ValueError(f"molecule {instance_id!r} has no model-visible heavy atoms")
            for site in atom_sites:
                tokens.append(
                    {
                        "atom_sites": (site,),
                        "token_site": site,
                        "reference": (instance_id, site.component_index),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": site.component_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "center_site": site,
                        "bb4_sites": (site, site, site, site),
                        "frame_sites": (site, site, site),
                    }
                )

    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    return tokens, atom_sites


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
    generated_residues: frozenset[ComponentReference],
) -> None:
    r"""
    Require the released structure-design domain: standard protein rows only.
    """

    for token, designed in zip(tokens, design_mask.tolist(), strict=True):
        if not designed:
            continue
        reference = cast(ComponentReference, token["reference"])
        entity = system.entities[system.instances[reference[0]]]
        if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
            raise ValueError(
                "BoltzGen structure design supports protein rows only; " "DNA and RNA may be supplied as fixed context"
            )
        component_id = system.component(reference).component_id
        if reference not in generated_residues and component_id not in _PROTEIN_ATOM14_NAMES:
            raise ValueError(
                "BoltzGen structure design requires a standard amino-acid row, "
                f"got {component_id!r} at {reference!r}"
            )


def _build_structure_design_atom_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    structure: Structure | None,
    design_mask: Tensor,
    *,
    generated_residues: frozenset[ComponentReference] = frozenset(),
    generator: torch.Generator | None = None,
) -> tuple[
    dict[str, Tensor],
    tuple[AtomSite | None, ...],
    tuple[AtomSite, ...],
    dict[AtomSite, int],
]:
    r"""
    Lower the released protein-atom14 design rows and natural fixed-context rows.

    Synthetic rows have no canonical `AtomSite`. Their target coordinates copy
    the residue-specific N/O source, while every design row is already masked to
    the exact features visible to the Model.
    """

    from rdkit.Chem import GetPeriodicTable

    model_atom_sites: list[AtomSite | None] = []
    target_source_sites: list[AtomSite] = []
    reference_source_sites: list[AtomSite] = []
    atom_to_token_rows: list[int] = []
    atom_token_indices: list[int] = []
    design_atom_rows: list[bool] = []
    design_backbone_rows: list[bool] = []
    reference_space_ids: list[int] = []
    fake_rows: list[bool] = []
    reference_space_by_component: dict[ComponentReference, int] = {}

    for token_index, (token, designed) in enumerate(zip(tokens, design_mask.tolist(), strict=True)):
        natural_sites = cast(tuple[AtomSite, ...], token["atom_sites"])
        reference = cast(ComponentReference, token["reference"])
        space_id = reference_space_by_component.setdefault(reference, len(reference_space_by_component))
        if designed:
            component_id = "GLY" if reference in generated_residues else system.component(reference).component_id
            atom_names = _PROTEIN_ATOM14_NAMES[component_id]
            by_name = {site.atom_id: site for site in natural_sites}
            missing = [name for name in atom_names if name not in by_name]
            if missing:
                raise ValueError(f"BoltzGen atom14 row {reference!r} is missing canonical atoms {missing!r}")
            real_sites = tuple(by_name[name] for name in atom_names)
            ca_site = by_name["CA"]
            for slot, site in enumerate(real_sites):
                model_atom_sites.append(site)
                target_source_sites.append(site)
                reference_source_sites.append(site)
                atom_to_token_rows.append(token_index)
                atom_token_indices.append(slot)
                design_atom_rows.append(True)
                design_backbone_rows.append(slot < 4)
                reference_space_ids.append(space_id)
                fake_rows.append(False)
            for slot, source_name in enumerate(
                _PROTEIN_ATOM14_FAKE_SOURCE[component_id],
                start=len(real_sites),
            ):
                model_atom_sites.append(None)
                target_source_sites.append(by_name[source_name])
                reference_source_sites.append(ca_site)
                atom_to_token_rows.append(token_index)
                atom_token_indices.append(slot)
                design_atom_rows.append(True)
                design_backbone_rows.append(False)
                reference_space_ids.append(space_id)
                fake_rows.append(True)
            continue

        for slot, site in enumerate(natural_sites):
            model_atom_sites.append(site)
            target_source_sites.append(site)
            reference_source_sites.append(site)
            atom_to_token_rows.append(token_index)
            atom_token_indices.append(slot)
            design_atom_rows.append(False)
            design_backbone_rows.append(False)
            reference_space_ids.append(space_id)
            fake_rows.append(False)

    atom_count = len(model_atom_sites)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)
    site_to_atom = {site: atom_index for atom_index, site in enumerate(model_atom_sites) if site is not None}
    if len(site_to_atom) != sum(site is not None for site in model_atom_sites):
        raise ValueError("BoltzGen model atom rows contain a duplicate canonical atom occurrence")

    reference_coordinates = _generated_reference_coordinates(
        system,
        tuple(reference_source_sites),
        generated_residues,
    )
    coordinate_index = {site: index for index, site in enumerate(structure.sites)} if structure is not None else {}
    target_coordinates = torch.zeros(padded_atom_count, 3)
    coordinate_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    for atom_index, source_site in enumerate(target_source_sites):
        source_index = coordinate_index.get(source_site)
        if source_index is not None:
            assert structure is not None
            target_coordinates[atom_index] = torch.as_tensor(
                structure.coordinates[source_index],
                dtype=target_coordinates.dtype,
            )
            coordinate_mask[atom_index] = bool(structure.coordinate_mask[source_index])
        elif (source_site.instance_id, source_site.component_index) in generated_residues:
            coordinate_mask[atom_index] = True

    atom_coordinates = target_coordinates.clone()
    valid_coordinates = coordinate_mask[:atom_count]
    center = (atom_coordinates[:atom_count] * valid_coordinates[:, None]).sum(dim=0) / valid_coordinates.sum().clamp(
        min=1
    )
    atom_coordinates[:atom_count] -= center
    center_atom_indices = torch.tensor(
        [site_to_atom[cast(AtomSite, token["center_site"])] for token in tokens],
        dtype=torch.long,
    )
    center_coordinates = target_coordinates[center_atom_indices]

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = torch.tensor(reference_coordinates)
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

    real_sites = tuple(site for site in model_atom_sites if site is not None)
    model_names = dict(zip(real_sites, _model_atom_names(system, real_sites), strict=True))
    periodic_table = GetPeriodicTable()
    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_charges = torch.zeros(padded_atom_count)
    atom_name_characters = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    for atom_index, model_site in enumerate(model_atom_sites):
        token_slot = atom_token_indices[atom_index]
        if design_atom_rows[atom_index] and not design_backbone_rows[atom_index]:
            atom_name = f"FL{token_slot}"
            atomic_number[atom_index] = 114
        else:
            assert model_site is not None
            atom = system.atom(model_site)
            atom_name = model_names[model_site]
            atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
            reference_charges[atom_index] = 0 if atom.formal_charge is None else atom.formal_charge
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
    fake_atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    fake_atom_mask[:atom_count] = torch.tensor(fake_rows, dtype=torch.bool)
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
            "fake_atom_mask": fake_atom_mask,
        },
        tuple(model_atom_sites),
        tuple(target_source_sites),
        site_to_atom,
    )


def _truth_pool_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    structure: Structure,
    target_source_sites: tuple[AtomSite, ...],
    padded_atom_count: int,
    site_to_atom: dict[AtomSite, int],
) -> dict[str, Tensor]:
    r"""
    Lower complete-assembly truth and its current-atom/symmetry mappings.

    The pool follows the natural model-visible atom order before synthetic
    atom14 rows are added. Unresolved atoms retain their pool rows.
    """

    pool_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    pool_index = {site: index for index, site in enumerate(pool_sites)}
    coordinate_index = {site: index for index, site in enumerate(structure.sites)}
    coordinates = torch.zeros(len(pool_sites), 3)
    coordinate_mask = torch.zeros(len(pool_sites), dtype=torch.bool)
    for index, site in enumerate(pool_sites):
        source_index = coordinate_index.get(site)
        if source_index is None or not structure.coordinate_mask[source_index]:
            continue
        coordinates[index] = torch.as_tensor(
            structure.coordinates[source_index],
            dtype=coordinates.dtype,
        )
        coordinate_mask[index] = True

    atom_index = torch.full((padded_atom_count,), -1, dtype=torch.long)
    atom_index[: len(target_source_sites)] = torch.tensor(
        [pool_index[site] for site in target_source_sites],
        dtype=torch.long,
    )

    chain_spans: list[tuple[int, int]] = []
    chain_equivalence: list[int] = []
    equivalence_index: dict[tuple[str, tuple[tuple[int, str], ...]], int] = {}
    start = 0
    for instance_id, entity_id in system.instances.items():
        stop = start
        while stop < len(pool_sites) and pool_sites[stop].instance_id == instance_id:
            stop += 1
        chain_spans.append((start, stop))
        layout = tuple((site.component_index, site.atom_id) for site in pool_sites[start:stop])
        key = (entity_id, layout)
        chain_equivalence.append(equivalence_index.setdefault(key, len(equivalence_index)))
        start = stop

    symmetry_atom_index, symmetry_permutation = _local_symmetry_features(
        system,
        tokens,
        site_to_atom,
    )
    return {
        "coordinates": coordinates,
        "coordinate_mask": coordinate_mask,
        "atom_index": atom_index,
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
    tokens: list[dict[str, object]],
    site_to_atom: dict[AtomSite, int],
) -> tuple[Tensor, Tensor]:
    r"""
    Carry exact standard swaps and the available stereochemical self-matches.
    """

    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    for token in tokens:
        reference = cast(ComponentReference, token["reference"])
        component_id = system.component(reference).component_id
        atom_permutations = _STANDARD_LOCAL_SYMMETRY_ATOMS.get(component_id)
        if atom_permutations is None:
            continue
        permutations = tuple(
            tuple(site_to_atom[AtomSite(reference[0], reference[1], atom_id)] for atom_id in atom_names)
            for atom_names in atom_permutations
        )
        groups.append((permutations[0], permutations))

    boundary_sites: set[AtomSite] = set()
    for bond in system.iter_bonds():
        reference_a = (bond.site_a.instance_id, bond.site_a.component_index)
        reference_b = (bond.site_b.instance_id, bond.site_b.component_index)
        if reference_a != reference_b:
            boundary_sites.add(bond.site_a)
            boundary_sites.add(bond.site_b)

    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        for component_index, component in enumerate(entity.components):
            if any(component.component_id in standards for standards in _STANDARD_COMPONENTS.values()):
                continue
            group = _component_self_match_group(
                component,
                instance_id,
                component_index,
                site_to_atom,
                boundary_sites,
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
    component: ChemicalComponent,
    instance_id: str,
    component_index: int,
    site_to_atom: dict[AtomSite, int],
    boundary_sites: set[AtomSite],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    r"""
    Return the carried RDKit automorphism candidates for one occurrence.
    """

    visible = tuple(
        (
            local_index,
            site_to_atom[site],
            site,
        )
        for local_index, atom in enumerate(component.atoms)
        if (site := AtomSite(instance_id, component_index, atom.atom_id)) in site_to_atom
    )
    if len(visible) < 2:
        return None

    molecule = _canonical_rdkit_molecule(Molecule((component,)))
    model_index = {local_index: atom_index for local_index, atom_index, _ in visible}
    fixed = {local_index for local_index, _atom_index, site in visible if site in boundary_sites}
    candidates = {
        tuple(model_index[match[local_index]] for local_index in model_index)
        for match in molecule.GetSubstructMatches(
            molecule,
            uniquify=False,
            useChirality=True,
            maxMatches=1000,
        )
        if all(match[local_index] in model_index for local_index in model_index)
        and all(match[local_index] == local_index for local_index in fixed)
    }
    identity = tuple(atom_index for _local_index, atom_index, _site in visible)
    candidates.add(identity)
    if len(candidates) == 1:
        return None
    return identity, (identity, *sorted(candidates - {identity}))


def _frame_atom_indices(
    tokens: list[dict[str, object]],
    site_to_atom: dict[AtomSite, int],
) -> Tensor:
    r"""
    Lower prediction-independent polymer confidence-frame atom indices.

    Non-polymer entries are placeholders: confidence equations replace them
    from each prediction or matched target because their frames depend on the
    corresponding coordinates and resolved mask.
    """

    return torch.tensor(
        [[site_to_atom[site] for site in cast(tuple[AtomSite, ...], token["frame_sites"])] for token in tokens],
        dtype=torch.long,
    )


def _build_atom_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    atom_sites: tuple[AtomSite, ...],
    structure: Structure | None,
    generated_residues: frozenset[ComponentReference] = frozenset(),
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[AtomSite, int]]:
    r"""
    Build the 388-wide BoltzGen/Boltz-2 atom reference feature vector and index maps.

    The 388-wide layout contains positions (3), charge (1), element (128), and
    atom-name characters (4x64), with no atom-mask channel.

    ``generated_residues`` selects protein placeholders whose backbone atoms use
    GLY-template coordinates. An empty set uses component reference coordinates.
    """

    from rdkit.Chem import GetPeriodicTable

    site_to_atom = {site: index for index, site in enumerate(atom_sites)}
    site_to_token = {
        site: token_index
        for token_index, token in enumerate(tokens)
        for site in cast(tuple[AtomSite, ...], token["atom_sites"])
    }
    atom_count = len(atom_sites)
    padded_atom_count = max(32, math.ceil(atom_count / 32) * 32)

    reference_coordinates = (
        _generated_reference_coordinates(system, atom_sites, generated_residues)
        if generated_residues
        else _reference_coordinates(system, atom_sites)
    )
    reference_space_by_residue: dict[tuple[str, int], int] = {}
    reference_space_ids = [
        reference_space_by_residue.setdefault((site.instance_id, site.component_index), len(reference_space_by_residue))
        for site in atom_sites
    ]
    raw_atom_coordinates = torch.zeros(padded_atom_count, 3)
    resolved = torch.zeros(padded_atom_count, dtype=torch.bool)
    coordinate_index: dict[AtomSite, int] = {}
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
    for atom_index, site in enumerate(atom_sites):
        source_index = coordinate_index.get(site)
        if source_index is not None:
            assert structure is not None
            raw_atom_coordinates[atom_index] = torch.as_tensor(
                structure.coordinates[source_index], dtype=raw_atom_coordinates.dtype
            )
            resolved[atom_index] = bool(structure.coordinate_mask[source_index])
        elif (site.instance_id, site.component_index) in generated_residues:
            # Generated atoms are present with zero coordinates before centering.
            resolved[atom_index] = True

    atom_coordinates = raw_atom_coordinates.clone()
    resolved_atoms = resolved[:atom_count]
    center = (atom_coordinates[:atom_count] * resolved_atoms[:, None]).sum(dim=0)
    center = center / resolved_atoms.sum().clamp(min=1)
    atom_coordinates[:atom_count] -= center
    center_atom_indices = torch.tensor(
        [site_to_atom[cast(AtomSite, token["center_site"])] for token in tokens], dtype=torch.long
    )
    center_coordinates = raw_atom_coordinates[center_atom_indices]

    reference_positions = torch.zeros(padded_atom_count, 3)
    reference_positions[:atom_count] = torch.tensor(reference_coordinates)
    reference_space_id = torch.zeros(padded_atom_count, dtype=torch.long)
    reference_space_id[:atom_count] = torch.tensor(reference_space_ids)
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
    atom_names = _model_atom_names(system, atom_sites)
    periodic_table = GetPeriodicTable()
    for atom_index, (site, atom_name) in enumerate(zip(atom_sites, atom_names)):
        atom = system.atom(site)
        atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
        reference_charges[atom_index] = atom.formal_charge
        atom_name_characters[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = site_to_token[site]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = cast(tuple[AtomSite, ...], tokens[token_index]["atom_sites"]).index(site)

    atom_features = torch.cat(
        (
            reference_positions,
            reference_charges.unsqueeze(-1),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_characters, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
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
        site_to_atom,
    )


def _contact_channels(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    site_to_token: dict[AtomSite, int],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
    token_count: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Realize contact records into the ``[N, N, 5]`` one-hot + ``[N, N]`` threshold.
    """

    selected: list[tuple[Contact, str, tuple[int, ...], tuple[int, ...]]] = []
    for condition in conditions:
        if isinstance(condition, Contact):
            contact = condition
            kind = "contact"
        elif isinstance(condition, BoltzGenDirectionalContact):
            contact = condition.contact
            kind = condition.direction
        else:
            continue
        rows = _endpoint_tokens(
            contact.side_a,
            token_index_by_residue,
            site_to_token,
            placement_map,
            instance_of,
        )
        columns = _endpoint_tokens(
            contact.side_b,
            token_index_by_residue,
            site_to_token,
            placement_map,
            instance_of,
        )
        if not rows or not columns:
            raise ValueError("a BoltzGen contact must resolve to at least one token on each side")
        selected.append((contact, kind, rows, columns))

    # An absent contact request is UNSPECIFIED; background pairs in a request are UNSELECTED.
    classes = torch.full((token_count, token_count), int(bool(selected)), dtype=torch.long)
    threshold = torch.zeros(token_count, token_count)
    for contact, kind, rows, columns in selected:
        forward_class = _CONTACT_KIND_TO_CLASS[kind]
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
    endpoints: tuple[Endpoint, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    site_to_token: dict[AtomSite, int],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> tuple[int, ...]:
    r"""
    Resolve contact endpoints to token rows, preserving declaration order.
    """

    rows: list[int] = []
    for endpoint in endpoints:
        if isinstance(endpoint, str):
            references = placement_map[instance_of[endpoint]][endpoint]
            for reference in references:
                rows.extend(_component_tokens(reference, token_index_by_residue, site_to_token))
        elif isinstance(endpoint, SymbolicResidueSite):
            reference = resolve(endpoint, placement_map, instance_of)
            rows.extend(_component_tokens(reference, token_index_by_residue, site_to_token))
        elif isinstance(endpoint, SymbolicAtomSite):
            reference = resolve(endpoint.residue, placement_map, instance_of)
            rows.append(site_to_token[AtomSite(reference[0], reference[1], endpoint.atom_id)])
        elif isinstance(endpoint, AtomSite):
            rows.append(site_to_token[endpoint])
        else:
            rows.extend(_component_tokens(cast(ComponentReference, endpoint), token_index_by_residue, site_to_token))
    return tuple(dict.fromkeys(rows))


def _component_tokens(
    reference: ComponentReference,
    token_index_by_residue: dict[tuple[str, int], int],
    site_to_token: dict[AtomSite, int],
) -> tuple[int, ...]:
    r"""
    Map one polymer residue or ligand component to all token rows it owns.
    """

    key = (reference[0], reference[1])
    if key in token_index_by_residue:
        return (token_index_by_residue[key],)
    rows = tuple(
        dict.fromkeys(
            token_index
            for site, token_index in site_to_token.items()
            if (site.instance_id, site.component_index) == key
        )
    )
    if not rows:
        raise KeyError(reference)
    return rows


def _design_mask(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
    generated_residues: frozenset[ComponentReference] = frozenset(),
) -> Tensor:
    r"""
    Realize the ``[N]`` boolean design mask.

    Generated placement residues are designable by default. Explicit
    [`BoltzGenDesignMask`][ofoldx.models.boltzgen.model.BoltzGenDesignMask] records add existing
    residues to that set.
    """

    positions: list[ComponentReference] = list(generated_residues)
    for condition in conditions:
        if isinstance(condition, BoltzGenDesignMask):
            positions.extend(condition.positions)
    return _design_mask_from_positions(tuple(positions), token_index_by_residue, token_count)


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
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
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
        if condition.group_id < 0:
            raise ValueError(
                f"BoltzGenStructureGroup group_id must be >= 0 (0 is the free/ungrouped class), "
                f"got {condition.group_id}"
            )
        for reference in condition.members:
            token = _component_reference_to_token(reference, token_index_by_residue)
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
    positions: tuple[ComponentReference, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
) -> Tensor:
    mask = torch.zeros(token_count, dtype=torch.bool)
    for reference in positions:
        mask[_component_reference_to_token(reference, token_index_by_residue)] = True
    return mask


def _binding_ids(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    site_to_token: dict[AtomSite, int],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
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
            condition.binding.residues,
            token_index_by_residue,
            site_to_token,
            placement_map,
            instance_of,
        ):
            ids[token_index] = _BINDING_BINDING
        for reference in condition.not_binding:
            ids[_component_reference_to_token(reference, token_index_by_residue)] = _BINDING_NOT_BINDING
    return ids


def _ss_ids(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
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
            ids[_component_reference_to_token(reference, token_index_by_residue)] = _SS_LABEL_TO_ID[label]
    return ids


def _symmetry_index(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
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
                ids[_component_reference_to_token(reference, token_index_by_residue)] = unit_index
    return ids


def _sequence_symmetry_index(
    groups: tuple[tuple[ComponentReference, ...], ...],
    token_index_by_residue: dict[tuple[str, int], int],
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
        tokens = [_component_reference_to_token(reference, token_index_by_residue) for reference in group]
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
    reference: ComponentReference,
    token_index_by_residue: dict[tuple[str, int], int],
    molecule_types: Tensor,
    role: str,
) -> int:
    r"""
    Resolve an inverse-folding residue reference and require a protein token.
    """

    token = _component_reference_to_token(reference, token_index_by_residue)
    if int(molecule_types[token].item()) != _MOLECULE_TYPES["protein"]:
        raise ValueError(f"BoltzGen inverse-folding {role} must reference a protein residue: {reference!r}")
    return token


def _aa_constraint_mask(
    global_omit_tokens: tuple[str, ...],
    aa_constraints: tuple[BoltzGenResidueConstraint, ...],
    token_index_by_residue: dict[tuple[str, int], int],
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
        token = _component_reference_to_token(constraint.position, token_index_by_residue)
        block = torch.ones(NUM_CANONICAL_TOKENS, dtype=torch.float32)
        for code in constraint.allowed:
            block[CANONICAL_TOKENS.index(code)] = 0.0
        mask[token] = torch.maximum(mask[token], block)
    return mask


# Template features contain residue identity, backbone frames, CB/CA coordinates,
# and visibility for aligned fixed tokens. A self-template is the identity-aligned case.


def _template_frame(nitrogen: np.ndarray, alpha: np.ndarray, carbon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build one residue's backbone frame.

    Gram-Schmidt over ``(C - CA, N - CA)``: ``e1`` along ``C - CA``, ``e2`` the ``N - CA``
    component orthogonal to ``e1``, ``e3 = e1 x e2``; the rotation columns are ``[e1, e2, e3]``
    and the translation is ``CA``. Norms use a ``1e-10`` epsilon.
    """

    v1 = carbon - alpha
    v2 = nitrogen - alpha
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    u2 = v2 - e1 * np.dot(e1, v2)
    e2 = u2 / (np.linalg.norm(u2) + 1e-10)
    e3 = np.cross(e1, e2)
    rotation = np.column_stack([e1, e2, e3])
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
    coordinates: dict[tuple[str, int, str], tuple[np.ndarray, bool]],
    instance_id: str,
    component_index: int,
    atom_id: str,
) -> tuple[np.ndarray | None, bool]:
    r"""
    Look up one template atom's coordinate + presence, or ``(None, False)`` if absent.
    """

    entry = coordinates.get((instance_id, component_index, atom_id))
    if entry is None:
        return None, False
    return entry


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

    # Resolve a query token index for a polymer residue named by instance or by entity (a copy count).
    instance_residue: dict[tuple[str, int], int] = {}
    entity_residue: dict[tuple[str, int], list[int]] = {}
    for token_index, token in enumerate(tokens):
        monomer_index = cast("int | None", token["monomer_index"])
        if monomer_index is None:
            continue
        instance_residue[(cast(str, token["instance_id"]), monomer_index)] = token_index
        entity_residue.setdefault((cast(str, token["entity_id"]), monomer_index), []).append(token_index)

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
        coordinates = {
            (site.instance_id, site.component_index, site.atom_id): (
                template.structure.coordinates[index],
                bool(template.structure.coordinate_mask[index]),
            )
            for index, site in enumerate(template.structure.sites)
        }
        for alignment in template.alignments:
            template_entity_id = template.structure.system.instances[alignment.template_instance_id]
            template_entity = template.structure.system.entities[template_entity_id]
            if not isinstance(template_entity, PolymerSequence):
                raise ValueError("a BoltzGen template alignment must reference a polymer instance")
            polymer_type = template_entity.polymer_type
            for target_component, template_component in alignment.residue_pairs:
                targets: tuple[int, ...]
                if alignment.target_instance_id is not None:
                    hit = instance_residue.get((alignment.target_instance_id, target_component))
                    targets = () if hit is None else (hit,)
                else:
                    targets = tuple(entity_residue.get((alignment.target_entity_id, target_component), ()))
                if not targets:
                    continue
                component = template_entity.components[template_component]
                component_id = component.component_id
                instance_id = alignment.template_instance_id

                center_coord, center_present = _template_atom(
                    coordinates, instance_id, template_component, _template_center_atom(polymer_type)
                )
                disto_coord, disto_present = _template_atom(
                    coordinates, instance_id, template_component, _template_disto_atom(component_id, polymer_type)
                )
                nitrogen, nitrogen_present = _template_atom(coordinates, instance_id, template_component, "N")
                alpha, alpha_present = _template_atom(coordinates, instance_id, template_component, "CA")
                carbon, carbon_present = _template_atom(coordinates, instance_id, template_component, "C")
                is_protein = polymer_type == "protein"
                has_frame = is_protein and nitrogen_present and alpha_present and carbon_present
                rotation = translation = None
                if has_frame:
                    rotation, translation = _template_frame(
                        np.asarray(nitrogen, dtype=np.float32),
                        np.asarray(alpha, dtype=np.float32),
                        np.asarray(carbon, dtype=np.float32),
                    )

                residue_type = _residue_type(component_id, polymer_type)
                for target_token in targets:
                    residue_type_ids[template_index, target_token] = residue_type
                    if center_coord is not None:
                        alpha_carbon_positions[template_index, target_token] = torch.from_numpy(
                            np.asarray(center_coord, dtype=np.float32)
                        )
                    if disto_coord is not None:
                        beta_carbon_positions[template_index, target_token] = torch.from_numpy(
                            np.asarray(disto_coord, dtype=np.float32)
                        )
                    beta_carbon_mask[template_index, target_token] = 1.0 if disto_present else 0.0
                    if has_frame:
                        frame_rotations[template_index, target_token] = torch.from_numpy(
                            np.asarray(rotation, dtype=np.float32)
                        )
                        frame_translations[template_index, target_token] = torch.from_numpy(
                            np.asarray(translation, dtype=np.float32)
                        )
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
_MSA_GAP = 1
# MSA cells are UPPERCASE one-letter MSA symbols (or the ``-`` gap), not ChemicalComponent IDs. The
# processor maps each symbol through the 33-token BoltzGen vocabulary selected by the entity's resolved
# polymer type. Standard residues use their vocabulary ids; all other symbols use that polymer type's
# unknown id. RNA one-letter symbols equal their component IDs directly.
_PROTEIN_MSA_RESTYPE = {
    symbol: _PROTEIN_RESIDUE_TYPES[component_id]
    for symbol, component_id in {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }.items()
}
_DNA_MSA_RESTYPE = {
    symbol: _DNA_RESIDUE_TYPES[component_id]
    for symbol, component_id in {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"}.items()
}


def _msa_symbol_restype(symbol: str, polymer_type: str) -> int:
    r"""
    Map one UPPERCASE one-letter MSA symbol (or ``-``) to a Boltz-1 token.
    """

    if symbol == "-":
        return _MSA_GAP
    if polymer_type == "protein":
        return _PROTEIN_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["protein"])
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(symbol, _UNKNOWN_RESIDUE_TYPES["rna"])
    return _DNA_MSA_RESTYPE.get(symbol, _UNKNOWN_RESIDUE_TYPES["dna"])


def _token_bond_indicator(
    system: MolecularSystem,
    site_to_token: dict[AtomSite, int],
    token_count: int,
) -> Tensor:
    r"""
    Build the symmetric inter-token bond indicator ``[N, N, 1]`` consumed by ``token_bond_projection``.

    The trailing singleton feeds ``nn.Linear(1, ...)`` in the design trunk; inverse
    folding indexes ``token_bonds[..., 0]``.
    """

    token_bonds = torch.zeros(token_count, token_count)

    def add(site_a: AtomSite, site_b: AtomSite) -> None:
        token_a = site_to_token.get(site_a)
        token_b = site_to_token.get(site_b)
        if token_a is None or token_b is None or token_a == token_b:
            return
        token_bonds[token_a, token_b] = token_bonds[token_b, token_a] = 1

    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if not isinstance(entity, Molecule):
            continue
        for molecule_bond in entity.bonds:
            add(
                AtomSite(instance_id, molecule_bond.component_a, molecule_bond.atom_a),
                AtomSite(instance_id, molecule_bond.component_b, molecule_bond.atom_b),
            )
    for system_bond in system.bonds:
        add(system_bond.site_a, system_bond.site_b)
    return token_bonds.unsqueeze(-1)


def _inter_residue_bond_atom_index(
    system: MolecularSystem,
    site_to_atom: dict[AtomSite, int],
) -> Tensor | None:
    r"""
    Return unique known covalent edges whose endpoints belong to different residues.
    """

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for bond in system.iter_bonds():
        left, right = bond.site_a, bond.site_b
        if left.instance_id == right.instance_id and left.component_index == right.component_index:
            continue
        if left not in site_to_atom or right not in site_to_atom:
            continue
        left_index, right_index = sorted((site_to_atom[left], site_to_atom[right]))
        endpoints = (left_index, right_index)
        if endpoints in seen:
            continue
        seen.add(endpoints)
        edges.append(endpoints)
    if not edges:
        return None
    return torch.tensor(edges, dtype=torch.long)


def _msa_features(
    msa: MSA | MSACollection | None,
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> dict[str, Tensor]:
    residue_types = [cast(int, token["residue_type"]) for token in tokens]
    if msa is None:
        rows = torch.tensor([residue_types])
        deletion_counts = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        # The MSA holds no PolymerSequence: column ``c`` binds ``columns[c].entity_id`` to a system entity,
        # a token's monomer index is directly the aligned-row column, and cells are UPPERCASE one-letter
        # symbols (or ``-``). A bare MSA is a single-chain plan over its unpaired pool; a collection replays
        # its dense ``row_pairing`` and ``row_is_paired`` plan across the paired and unpaired pools.
        columns = (msa,) if isinstance(msa, MSA) else msa.msas
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
        polymer_types = {
            value.entity_id: cast(PolymerSequence, system.entities[value.entity_id]).polymer_type for value in columns
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
                isinstance(msa, MSACollection) and present > 1 and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                polymer_type = polymer_types[value.entity_id]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts)[
                    source_row
                ]
                for token_index, token in enumerate(tokens):
                    if token["entity_id"] != value.entity_id or token["monomer_index"] is None:
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
        "msa_profile": torch.nn.functional.one_hot(rows, 33).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _mask_structure_design_msa(values: dict[str, Tensor], design_mask: Tensor) -> dict[str, Tensor]:
    r"""
    Remove designed residue identity from every model-visible MSA channel.
    """

    if not bool(design_mask.any()):
        return values
    unknown = _UNKNOWN_RESIDUE_TYPES["protein"]
    values["msa_tokens"][:, design_mask] = unknown
    values["has_deletion"][:, design_mask] = False
    values["deletion_value"][:, design_mask] = 0
    values["msa_paired"][:, design_mask] = 0
    values["msa_profile"][design_mask] = 0
    values["msa_profile"][design_mask, unknown] = 1
    values["deletion_mean"][design_mask] = 0
    return values


def _token_b_factors(
    structure: Structure,
    tokens: list[dict[str, object]],
) -> Tensor | None:
    r"""
    Gather optional per-site B factors at each token center.
    """

    if structure.b_factor is None:
        return None
    source_index = {site: index for index, site in enumerate(structure.sites)}
    values = torch.full((len(tokens),), float("nan"))
    for token_index, token in enumerate(tokens):
        index = source_index.get(cast(AtomSite, token["center_site"]))
        if index is not None:
            values[token_index] = float(structure.b_factor[index])
    return values


def _reference_coordinates(system: MolecularSystem, sites: tuple[AtomSite, ...]) -> list[tuple[float, float, float]]:
    generated: dict[str, dict[tuple[int, str], tuple[float, float, float]]] = {}
    coordinates: list[tuple[float, float, float]] = []
    for site in sites:
        entity_id = system.instances[site.instance_id]
        entity = system.entities[entity_id]
        component = entity.components[site.component_index]
        atom_index = next(index for index, atom in enumerate(component.atoms) if atom.atom_id == site.atom_id)
        coordinate = component.reference_coordinates[atom_index] if component.reference_coordinates else None
        if coordinate is None and isinstance(entity, Molecule):
            if entity_id not in generated:
                generated[entity_id] = _molecule_reference_coordinates(entity)
            coordinate = generated[entity_id][(site.component_index, site.atom_id)]
        if coordinate is None:
            raise ValueError(f"Boltz-1 requires reference coordinates for {site!r}")
        coordinates.append(coordinate)
    return coordinates


def _molecule_reference_coordinates(molecule: Molecule) -> dict[tuple[int, str], tuple[float, float, float]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rdkit_molecule = Chem.AddHs(_canonical_rdkit_molecule(molecule))
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = 42
    conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(rdkit_molecule, options)
    AllChem.UFFOptimizeMolecule(rdkit_molecule, confId=conformer_id, maxIters=1000)
    conformer = rdkit_molecule.GetConformer(conformer_id)
    coordinates: dict[tuple[int, str], tuple[float, float, float]] = {}
    flat_index = 0
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            position = conformer.GetAtomPosition(flat_index)
            coordinates[(component_index, atom.atom_id)] = (position.x, position.y, position.z)
            flat_index += 1
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


def _model_atom_names(system: MolecularSystem, sites: tuple[AtomSite, ...]) -> list[str]:
    names: list[str] = []
    counts: dict[tuple[str, str], int] = {}
    for site in sites:
        name = site.atom_id
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = system.atom(site).element.upper()
            key = (site.instance_id, element)
            counts[key] = counts.get(key, 0) + 1
            name = f"{element}{counts[key]}"
        if len(name) > 4:
            raise ValueError(f"Boltz-1 atom name cannot be encoded in four characters: {name!r}")
        names.append(name)
    return names


def _canonical_rdkit_molecule(molecule: Molecule) -> Mol:
    r"""
    Materialize a canonical molecular graph as RDKit chemistry.
    """

    from rdkit import Chem

    editable = Chem.RWMol()
    atom_index: dict[tuple[int, str], int] = {}
    atom_stereo: dict[int, str] = {}
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            rdkit_atom = Chem.Atom(atom.element)
            if atom.isotope is not None:
                rdkit_atom.SetIsotope(atom.isotope)
            if atom.formal_charge is not None:
                rdkit_atom.SetFormalCharge(atom.formal_charge)
            if atom.radical_electrons is not None:
                rdkit_atom.SetNumRadicalElectrons(atom.radical_electrons)
            if atom.explicit_hydrogens is not None:
                rdkit_atom.SetNumExplicitHs(atom.explicit_hydrogens)
            if atom.implicit_hydrogens_allowed is not None:
                rdkit_atom.SetNoImplicit(not atom.implicit_hydrogens_allowed)
            if atom.aromatic is not None:
                rdkit_atom.SetIsAromatic(atom.aromatic)
            index = editable.AddAtom(rdkit_atom)
            atom_index[(component_index, atom.atom_id)] = index
            if atom.stereo is not None:
                atom_stereo[index] = atom.stereo.upper()

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    bond_stereo: list[tuple[int, str, tuple[int, int] | None]] = []
    for component_index, component in enumerate(molecule.components):
        component_bonds = sorted(
            component.bonds or (),
            key=lambda bond: (
                atom_index[(component_index, bond.atom_a)],
                atom_index[(component_index, bond.atom_b)],
            ),
        )
        for component_bond in component_bonds:
            editable.AddBond(
                atom_index[(component_index, component_bond.atom_a)],
                atom_index[(component_index, component_bond.atom_b)],
                bond_types[component_bond.order],
            )
            bond_index = editable.GetNumBonds() - 1
            editable.GetBondWithIdx(bond_index).SetIsAromatic(component_bond.aromatic)
            if component_bond.stereo is not None:
                stereo_atoms = (
                    tuple(atom_index[(component_index, atom_id)] for atom_id in component_bond.stereo_atoms)
                    if component_bond.stereo_atoms is not None
                    else None
                )
                bond_stereo.append(
                    (
                        bond_index,
                        component_bond.stereo,
                        cast(tuple[int, int] | None, stereo_atoms),
                    )
                )
    molecule_bonds = sorted(
        molecule.bonds,
        key=lambda bond: (
            atom_index[(bond.component_a, bond.atom_a)],
            atom_index[(bond.component_b, bond.atom_b)],
        ),
    )
    for molecule_bond in molecule_bonds:
        editable.AddBond(
            atom_index[(molecule_bond.component_a, molecule_bond.atom_a)],
            atom_index[(molecule_bond.component_b, molecule_bond.atom_b)],
            bond_types[molecule_bond.order],
        )
        bond_index = editable.GetNumBonds() - 1
        editable.GetBondWithIdx(bond_index).SetIsAromatic(molecule_bond.aromatic)
        if molecule_bond.stereo is not None:
            stereo_atoms = (
                tuple(atom_index[site] for site in molecule_bond.stereo_atoms)
                if molecule_bond.stereo_atoms is not None
                else None
            )
            bond_stereo.append(
                (
                    bond_index,
                    molecule_bond.stereo,
                    cast(tuple[int, int] | None, stereo_atoms),
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


def _cyclic_period(system: MolecularSystem, instance_id: str) -> int:
    entity = system.entities[system.instances[instance_id]]
    if not isinstance(entity, PolymerSequence) or len(entity) <= 2:
        return 0
    for bond in system.bonds:
        if bond.site_a.instance_id != instance_id or bond.site_b.instance_id != instance_id:
            continue
        if {bond.site_a.component_index, bond.site_b.component_index} == {0, len(entity) - 1}:
            return len(entity)
    return 0


def _residue_type(component_id: str, polymer_type: str) -> int:
    if polymer_type == "protein":
        return _PROTEIN_RESIDUE_TYPES.get(component_id, 22)
    if polymer_type == "rna":
        return _RNA_RESIDUE_TYPES.get(component_id, 27)
    return _DNA_RESIDUE_TYPES.get(component_id, 32)


def _validate_msa(
    msa: MSA | MSACollection | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    values = (msa,) if isinstance(msa, MSA) else msa.msas
    for value in values:
        # Evidence holds no PolymerSequence: it names one entity by ``entity_id`` and closes over the
        # primary system supplied here, so the Processor resolves the entity and checks its query width.
        entity = system.entities.get(value.entity_id)
        if not isinstance(entity, PolymerSequence):
            raise ValueError(f"MSA entity {value.entity_id!r} disagrees with the input system")
        width = len(value.unpaired_sequences[0])
        if width != len(entity.components):
            raise ValueError(
                f"MSA for entity {value.entity_id!r} spans {width} columns but the entity has "
                f"{len(entity.components)} components"
            )
    return msa
