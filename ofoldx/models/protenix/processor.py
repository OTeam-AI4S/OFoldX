r"""
Protenix structure processing and JSON constraint featurization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from os import PathLike
from typing import Any, Literal, cast, overload

import numpy as np
import torch
from torch import Tensor

from ...data.ccd import CcdProvider, ReferenceComponent, default_ccd_provider
from ...data.structure import ResidueKey, Structure
from ...processors.registry import PROCESSORS
from ...processors.structure import (
    MOLECULE_LIGAND,
    StructureProcessor,
    _entity_indices,
    _reference_features,
    _token_bonds,
    _token_frames,
    atom_molecule_type,
    materialize_missing_atoms,
)
from ...processors.utils import (
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
from .io import structure_from_protenix_json
from .model import ProtenixConstraintFeatures, ProtenixGuidanceFeatures, ProtenixInput


@dataclass(frozen=True, slots=True)
class ProtenixRequest:
    r"""
    Per-call input bundling a structure with its official Protenix constraint JSON for one processor call.

    Attributes:
        structure: Structure object, structure file path, or an official Protenix job JSON. The job JSON may
            be a decoded ``sequences``/``constraint`` dict or a path to such a ``.json`` file; when supplied
            this way the structure is materialized from the ``sequences`` section (protein/DNA/RNA/ligand
            entities) and, unless overridden below, its ``constraint`` and ``sequences`` sections are used.
        constraint: Official JSON ``constraint`` section with ``contact`` and/or ``pocket`` entries. Defaults
            to the ``constraint`` section of a job-JSON ``structure`` when omitted.
        sequences: Optional official JSON ``sequences`` list, only needed when atom ids are integer
            atom-map labels rather than atom-name strings. Defaults to the ``sequences`` section of a
            job-JSON ``structure`` when omitted.
        constraint_feature: Optional already-featurized official tensor bundle. Mutually exclusive
            with ``constraint``.
        guidance_features: Optional official TFG geometry feature tensors.
        constraint_seed: Optional seed for the official random selection inside token-contact and pocket
            constraints.
    """

    structure: StructureInput | Mapping[str, Any]
    constraint: Mapping[str, Any] | None = None
    sequences: Sequence[Mapping[str, Any]] | None = None
    constraint_feature: ProtenixConstraintFeatures | Mapping[str, Tensor] | None = None
    guidance_features: ProtenixGuidanceFeatures | Mapping[str, Tensor] | None = None
    constraint_seed: int | None = None


@PROCESSORS.register("protenix")
class ProtenixProcessor(StructureProcessor):
    r"""
    Convert structure inputs plus official Protenix JSON constraints into model input tensors.

    The emitted ``constraint_feature`` bundle uses the official keys and tensor shapes:
    ``pocket`` ``[N,N,1]``, ``contact`` ``[N,N,2]``, ``contact_atom`` ``[N,N,2]``, and
    ``substructure`` ``[N,N,4]``. Atom-contact constraints follow the reference behavior by
    expanding selected standard-residue tokens into atom-level tokens before all structural
    fields are built.

    Defaults set on the processor apply to every call; a per-call
    [`ProtenixRequest`][ofoldx.models.protenix.processor.ProtenixRequest]
    overrides them.

    Args:
        constraint_spec: Default official JSON ``constraint`` section applied when a call does not
            supply one.
        sequences: Default official JSON ``sequences`` list, needed only for integer atom-map ids.
        constraint_feature: Default prebuilt constraint feature bundle; mutually exclusive with
            ``constraint_spec``.
        guidance_features: Default training-free-guidance geometry features.
        constraint_seed: Default seed for the random contact/pocket token selection.
        extract_guidance_features: Whether to derive guidance features from the structure when none
            are supplied.
        **kwargs: Additional arguments forwarded to
            [`StructureProcessor`][ofoldx.processors.structure.StructureProcessor].
    """

    input_class = ProtenixInput

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> ProtenixProcessor:
        r"""
        Build a processor from a Protenix model config.

        Reads only the scalar size fields, ignoring nested config blocks so they are not mistaken for
        dimensions.

        Args:
            config: Model config (mapping or object) exposing ``single_size``, ``token_feature_size``,
                and ``atom_feature_size``.
            **kwargs: Additional processor arguments overriding the config-derived defaults.

        Returns:
            The configured processor.
        """

        parameters: dict[str, Any] = {
            "single_size": int(_config_scalar(config, "single_size", 384)),
            "token_feature_size": int(_config_scalar(config, "token_feature_size", 449)),
            "atom_feature_size": int(_config_scalar(config, "atom_feature_size", 385)),
        }
        parameters.update(kwargs)
        return cls(**parameters)

    def __init__(
        self,
        *,
        constraint_spec: Mapping[str, Any] | None = None,
        sequences: Sequence[Mapping[str, Any]] | None = None,
        constraint_feature: ProtenixConstraintFeatures | Mapping[str, Tensor] | None = None,
        guidance_features: ProtenixGuidanceFeatures | Mapping[str, Tensor] | None = None,
        constraint_seed: int | None = None,
        extract_guidance_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.constraint_spec = constraint_spec
        self.sequences = sequences
        self.constraint_feature = constraint_feature
        self.guidance_features = guidance_features
        self.constraint_seed = constraint_seed
        self.extract_guidance_features = extract_guidance_features

    def __call__(  # type: ignore[override]
        self,
        value: StructureInput | Mapping[str, Any] | ProtenixRequest,
    ) -> ProtenixInput:
        r"""
        Featurize a structure (and its constraints) into a batched model input.

        Resolves the structure and constraint/guidance sources (per-call request fields override the
        processor defaults), tokenizes, builds the structural fields, and packs everything into a
        single-example (batch-of-one) input.

        Args:
            value: A parsed [`Structure`][ofoldx.data.structure.Structure], a structure file path,
                an official Protenix job JSON (decoded dict or ``.json`` path, whose ``sequences``
                section builds the structure and whose ``constraint`` section is featurized), or a
                [`ProtenixRequest`][ofoldx.models.protenix.processor.ProtenixRequest].

        Returns:
            The batched
            [`ProtenixInput`][ofoldx.models.protenix.model.ProtenixInput].

        Raises:
            ValueError: If both official JSON constraints and prebuilt constraint tensors are supplied.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/protenix-base")  # doctest: +SKIP
            >>> batch = processor("complex.cif")  # doctest: +SKIP
        """

        if isinstance(value, ProtenixRequest):
            structure_input = value.structure
            constraint_spec = value.constraint
            sequences = value.sequences
            raw_features = value.constraint_feature
            raw_guidance_features = value.guidance_features
            seed = value.constraint_seed
        else:
            structure_input = value
            constraint_spec = self.constraint_spec
            sequences = self.sequences
            raw_features = self.constraint_feature
            raw_guidance_features = self.guidance_features
            seed = self.constraint_seed

        if constraint_spec is not None and raw_features is not None:
            raise ValueError(
                "provide either official JSON constraints or prebuilt constraint_feature tensors, not both"
            )

        ccd = self._ccd if self._ccd is not None else default_ccd_provider()
        job = _load_protenix_job(structure_input)
        if job is not None:
            structure = structure_from_protenix_json(job, ccd=ccd)
            if constraint_spec is None and raw_features is None:
                constraint_spec = job.get("constraint")
            if sequences is None:
                sequences = job.get("sequences")
        else:
            structure = structure_from_input(cast("StructureInput", structure_input))
        if self.materialize_missing_atoms:
            structure, resolved = materialize_missing_atoms(structure, ccd)
        else:
            resolved = None

        tokens = tokenize_structure(structure)
        constraint_feature: ProtenixConstraintFeatures | Mapping[str, Tensor] | None
        if raw_features is not None:
            constraint_feature = (
                raw_features
                if isinstance(raw_features, ProtenixConstraintFeatures)
                else ProtenixConstraintFeatures.from_mapping(raw_features)
            )
        elif constraint_spec is not None:
            tokens, constraint_feature = self._features_from_json(
                structure,
                tokens=tokens,
                constraint=constraint_spec,
                sequences=sequences,
                seed=seed,
            )
        else:
            constraint_feature = None

        fields = self._structural_fields_from_tokens(structure, tokens=tokens, ccd=ccd, resolved=resolved)
        guidance_features = None
        if self.extract_guidance_features:
            guidance_features = self._guidance_features_from_fields(fields)
        if raw_guidance_features is not None:
            guidance_features = (
                raw_guidance_features
                if isinstance(raw_guidance_features, ProtenixGuidanceFeatures)
                else ProtenixGuidanceFeatures.from_mapping(raw_guidance_features)
            )
        token_bond_matrix = _token_bonds(
            fields.structure,
            fields.atom_to_token_values,
            fields.reference_space_values,
            fields.num_tokens,
            ccd,
        )
        return self.input_class(
            token_features=self._token_features(fields.residues, fields.residue_names).unsqueeze(0),
            token_mask=fields.token_mask.unsqueeze(0),
            residue_index=torch.tensor(
                [residue.residue_number for residue in fields.residues], dtype=torch.long
            ).unsqueeze(0),
            chain_index=torch.tensor(
                [fields.chain_to_index[residue.chain_id] for residue in fields.residues], dtype=torch.long
            ).unsqueeze(0),
            atom_features=self._atom_features(fields.structure, self._atom_names(fields.structure)).unsqueeze(0),
            atom_to_token=fields.atom_to_token.unsqueeze(0),
            atom_mask=fields.atom_mask.unsqueeze(0),
            reference_positions=fields.reference_positions.unsqueeze(0),
            reference_mask=fields.reference_mask.unsqueeze(0),
            reference_charges=fields.reference_charges.unsqueeze(0),
            entity_index=torch.tensor(fields.entity_values, dtype=torch.long).unsqueeze(0),
            symmetry_index=torch.tensor(
                _symmetry_values(fields.structure, fields.tokens),
                dtype=torch.long,
            ).unsqueeze(0),
            token_index=torch.arange(fields.num_tokens, dtype=torch.long).unsqueeze(0),
            reference_space_id=torch.tensor(fields.reference_space_values, dtype=torch.long).unsqueeze(0),
            representative_atom_mask=fields.representative_atom_mask.unsqueeze(0),
            atom_token_index=fields.atom_token_index.unsqueeze(0),
            token_to_rep_atom=fields.token_to_rep_atom.unsqueeze(0),
            molecule_type=torch.tensor(fields.molecule_type_values, dtype=torch.long).unsqueeze(0),
            frame_atom_index=torch.tensor(fields.frame_atom_index_values, dtype=torch.long)
            .reshape(fields.num_tokens, 3)
            .unsqueeze(0),
            frame_mask=torch.tensor(fields.frame_mask_values, dtype=torch.bool).unsqueeze(0),
            token_bonds=token_bond_matrix.unsqueeze(0),
            target_positions=fields.target_positions.unsqueeze(0),
            target_mask=fields.target_mask.unsqueeze(0),
            constraint_feature=_batched_constraint_feature(constraint_feature),
            guidance_features=guidance_features,
        )

    def _structural_fields_from_tokens(
        self,
        structure: Structure,
        *,
        tokens: tuple[StructureToken, ...],
        ccd: CcdProvider,
        resolved: np.ndarray | None,
    ) -> StructuralFields:
        r"""
        Build the full structural-field bundle from a tokenized structure.

        Derives the atom/token index maps, masks, entity/symmetry indices, reference conformer
        features, and token frames used to pack the model input.

        Args:
            structure: The parsed structure.
            tokens: The tokenized structure.
            ccd: Chemical Component Dictionary provider for reference features.
            resolved: Optional per-atom resolved mask from missing-atom materialization.

        Returns:
            The assembled structural fields.
        """

        residues = tuple(token.residue for token in tokens)
        residue_names = _token_residue_names(structure, tokens)
        chain_to_index = _index_values(residue.chain_id for residue in residues)
        num_tokens = len(tokens)
        num_atoms = len(structure.atoms)

        atom_to_token_values = _atom_to_token(tokens, num_atoms)
        atom_to_token = torch.tensor(atom_to_token_values, dtype=torch.long)
        target_positions = torch.tensor([[atom.x, atom.y, atom.z] for atom in structure.atoms], dtype=torch.float32)
        atom_mask = torch.ones(num_atoms, dtype=torch.bool)
        target_mask = atom_mask if resolved is None else torch.from_numpy(resolved)
        token_mask = torch.ones(num_tokens, dtype=torch.bool)
        representative_atom_mask, atom_token_index, token_to_rep_atom = _atom_token_fields(tokens, num_atoms)
        entity_values = _entity_values_for_tokens(structure, tokens)
        reference_space_values = _reference_space_ids(structure)
        molecule_type_values = [atom_molecule_type(structure.atoms[token.atom_indices[0]]) for token in tokens]
        reference_positions, reference_mask, reference_charges = _reference_features(
            structure,
            reference_space_values,
            ccd,
            augment=self._reference_conformer_augment,
            rng=self._augment_rng,
        )
        frame_atom_index_values, frame_mask_values = _token_frames(
            structure,
            tokens,
            reference_positions,
            reference_mask,
            reference_space_values,
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

    def _guidance_features_from_fields(self, fields: StructuralFields) -> ProtenixGuidanceFeatures:
        r"""
        Derive the training-free-guidance geometry features from the structural fields.

        Builds the element one-hot, chain/atom index maps, interchain-bond and symmetric-chain
        indices, and the RDKit-derived per-ligand geometry tensors.

        Args:
            fields: The structural fields for the tokenized structure.

        Returns:
            The guidance feature bundle.
        """

        ref_element = torch.zeros(fields.num_atoms, 128, dtype=torch.float32)
        if fields.num_atoms:
            element_indices = torch.tensor(
                [min(self._element_index(atom.element), 127) for atom in fields.structure.atoms],
                dtype=torch.long,
            )
            ref_element[torch.arange(fields.num_atoms), element_indices] = 1.0

        interchain_bonds = [
            (atom_i, atom_j)
            for atom_i, atom_j in fields.structure.bonds
            if fields.structure.atoms[atom_i].residue.chain_id != fields.structure.atoms[atom_j].residue.chain_id
        ]
        tensors = _empty_guidance_feature_tensors(dtype=ref_element.dtype)
        tensors.update(
            {
                "asym_id": torch.tensor(
                    [fields.chain_to_index[token.residue.chain_id] for token in fields.tokens],
                    dtype=torch.long,
                ),
                "atom_to_token_idx": fields.atom_to_token.to(dtype=torch.long),
                "ref_element": ref_element,
                "interchain_bond_index": _index_tensor(interchain_bonds, width=2),
                "symmetric_chain_index": _symmetric_chain_index(fields),
            }
        )
        ccd = self._ccd if self._ccd is not None else default_ccd_provider()
        tensors.update(_rdkit_guidance_feature_tensors(fields, ccd))
        return ProtenixGuidanceFeatures(tensors=tensors)

    def _features_from_json(
        self,
        structure: Structure,
        *,
        tokens: tuple[StructureToken, ...],
        constraint: Mapping[str, Any],
        sequences: Sequence[Mapping[str, Any]] | None,
        seed: int | None,
    ) -> tuple[tuple[StructureToken, ...], ProtenixConstraintFeatures]:
        r"""
        Featurize an official JSON ``constraint`` section into a constraint feature bundle.

        Canonicalizes the contacts, expands standard-residue tokens for atom contacts, and builds the
        pocket/contact/contact-atom/substructure feature tensors. Token expansion for atom contacts can
        change the token list, which is returned alongside the features.

        Args:
            structure: The parsed structure.
            tokens: The current tokenized structure.
            constraint: The official JSON ``constraint`` section.
            sequences: Optional official JSON ``sequences`` list for integer atom-map ids.
            seed: Optional seed for the random contact/pocket token selection.

        Returns:
            ``(tokens, features)`` where ``tokens`` reflects any atom-contact expansion.
        """

        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        canonical_contacts = [_canonicalize_contact(sequences, contact) for contact in constraint.get("contact", [])]
        atom_contact_inputs = [contact for contact in canonical_contacts if contact["contact_type"] == "atom_contact"]
        token_contact_inputs = [contact for contact in canonical_contacts if contact["contact_type"] == "token_contact"]

        atom_to_token_values = _atom_to_token(tokens, len(structure.atoms))
        atom_contact_pairs: list[tuple[int, int, float, float]] = []
        for contact in atom_contact_inputs:
            atoms_1 = _select_atoms(structure, contact["id1"])
            atoms_2 = _select_atoms(structure, contact["id2"])
            if len(atoms_1) != 1 or len(atoms_2) != 1:
                continue
            atom_contact_pairs.append((atoms_1[0], atoms_2[0], contact["max_distance"], contact["min_distance"]))

        if atom_contact_pairs:
            tokens, atom_contact_token_pairs = _expand_tokens_for_atom_contacts(
                structure,
                tokens=tokens,
                atom_contact_pairs=atom_contact_pairs,
                atom_to_token_values=atom_to_token_values,
            )
        else:
            atom_contact_token_pairs = []

        atom_to_token_values = _atom_to_token(tokens, len(structure.atoms))
        num_tokens = len(tokens)
        contact_atom = _contact_atom_feature(num_tokens, atom_contact_token_pairs)

        token_contact_specifics: list[tuple[Tensor, Tensor, float, float]] = []
        for contact in token_contact_inputs:
            atoms_1 = _select_atoms(structure, contact["id1"])
            atoms_2 = _select_atoms(structure, contact["id2"])
            tokens_1 = _atom_indices_to_tokens(atoms_1, atom_to_token_values)
            tokens_2 = _atom_indices_to_tokens(atoms_2, atom_to_token_values)
            if tokens_1.numel() == 0 or tokens_2.numel() == 0:
                continue
            token_contact_specifics.append((tokens_1, tokens_2, contact["max_distance"], contact["min_distance"]))
        contact = _contact_feature(num_tokens, token_contact_specifics, generator)

        pocket = _pocket_feature(
            structure,
            num_tokens=num_tokens,
            atom_to_token_values=atom_to_token_values,
            constraint=constraint,
            generator=generator,
        )
        substructure = _substructure_feature(
            structure,
            tokens=tokens,
            constraint=constraint,
        )
        return tokens, ProtenixConstraintFeatures(
            pocket=pocket,
            contact=contact,
            contact_atom=contact_atom,
            substructure=substructure,
        )


def _canonicalize_contact(
    sequences: Sequence[Mapping[str, Any]] | None,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    r"""
    Normalize one official JSON contact entry into a canonical two-sided selector dict.

    Resolves each side's entity/copy/position/atom identifiers (mapping integer atom ids to names via
    the sequences list), validates the distance band, and classifies the contact as atom- or
    token-level.

    Args:
        sequences: Optional official JSON ``sequences`` list for integer atom-map ids.
        pair: One official JSON ``contact`` entry.

    Returns:
        Canonical contact dict with ``id1``/``id2`` selectors, distances, and ``contact_type``.

    Raises:
        ValueError: If side identifiers are missing, the two sides share a chain, or the distance
            band is inverted.
    """

    canonical: dict[str, Any] = {}
    for side in ("1", "2"):
        identifiers = []
        for key in ("entity", "copy", "position", "atom"):
            value = pair.get(f"{key}{side}")
            if value is None and key != "atom":
                raise ValueError("contact should have entity/copy/position identifiers on both sides")
            if key == "atom" and isinstance(value, int):
                value = _atom_map_to_name(sequences, int(pair[f"entity{side}"]), value)
            identifiers.append(value)
        canonical[f"id{side}"] = tuple(identifiers)

    if canonical["id1"][:2] == canonical["id2"][:2]:
        raise ValueError("A contact pair can not be specified on the same chain")
    min_distance = float(pair.get("min_distance", 0.0))
    max_distance = float(pair["max_distance"])
    if max_distance < min_distance:
        raise ValueError("max_distance must be greater than min_distance")
    canonical["min_distance"] = min_distance
    canonical["max_distance"] = max_distance
    canonical["contact_type"] = "atom_contact" if "atom1" in pair and "atom2" in pair else "token_contact"
    return canonical


def _atom_map_to_name(sequences: Sequence[Mapping[str, Any]] | None, entity: int, atom_id: int) -> str:
    r"""
    Resolve an integer atom-map id to its atom name via the official sequences list.

    Args:
        sequences: Official JSON ``sequences`` list.
        entity: One-based entity number selecting the sequence entry.
        atom_id: Integer atom-map id to resolve.

    Returns:
        The atom name string.

    Raises:
        ValueError: If ``sequences`` is missing or the entity has no ``atom_map_to_atom_name``.
    """

    if sequences is None:
        raise ValueError("integer atom identifiers require the official JSON sequences list")
    entity_entry = list(sequences[entity - 1].values())[0]
    atom_map = entity_entry.get("atom_map_to_atom_name")
    if atom_map is None:
        raise ValueError(f"entity {entity} does not define atom_map_to_atom_name")
    if atom_id in atom_map:
        return str(atom_map[atom_id])
    return str(atom_map[str(atom_id)])


def _select_atoms(structure: Structure, selector: tuple[int, int, int | None, str | None]) -> list[int]:
    r"""
    Return the atom indices matching an ``(entity, copy, position, atom_name)`` selector.

    A ``None`` position matches any position (either chain-local or absolute residue number); a
    ``None`` atom name matches every atom in the selected residues.

    Args:
        structure: The parsed structure.
        selector: The ``(entity, copy, position, atom_name)`` tuple to match.

    Returns:
        The matching global atom indices.
    """

    entity, copy, position, atom_name = selector
    chain_info = _chain_info(structure)
    positions = _residue_positions(structure)
    atom_indices: list[int] = []
    for atom_index, atom in enumerate(structure.atoms):
        chain_entity, chain_copy = chain_info[atom.residue.chain_id]
        chain_position = positions[atom.residue]
        if chain_entity != int(entity) or chain_copy != int(copy):
            continue
        if position is not None and chain_position != int(position) and atom.residue.residue_number != int(position):
            continue
        if atom_name is not None and atom.name != str(atom_name):
            continue
        atom_indices.append(atom_index)
    return atom_indices


def _expand_tokens_for_atom_contacts(
    structure: Structure,
    *,
    tokens: tuple[StructureToken, ...],
    atom_contact_pairs: list[tuple[int, int, float, float]],
    atom_to_token_values: list[int],
) -> tuple[tuple[StructureToken, ...], list[tuple[int, int, float, float]]]:
    r"""
    Split constrained standard-residue tokens into per-atom tokens for atom-contact constraints.

    Mirrors the reference behavior: tokens involved in an atom contact, that are standard non-ligand
    residues, are expanded so each atom becomes its own token; the contact atom indices are remapped
    to the new token layout.

    Args:
        structure: The parsed structure.
        tokens: The current tokenized structure.
        atom_contact_pairs: Atom-contact pairs ``(atom_i, atom_j, max_distance, min_distance)``.
        atom_to_token_values: Token index per atom for the current tokenization.

    Returns:
        ``(tokens, token_pairs)`` with the (possibly expanded) tokens and remapped token-index pairs.
    """

    tokens_to_expand = {atom_to_token_values[atom_i] for atom_i, _, _, _ in atom_contact_pairs}
    tokens_to_expand.update(atom_to_token_values[atom_j] for _, atom_j, _, _ in atom_contact_pairs)
    if not tokens_to_expand:
        return tokens, []

    new_tokens: list[StructureToken] = []
    for token_index, token in enumerate(tokens):
        atom = structure.atoms[token.representative_atom_index]
        if (
            token_index in tokens_to_expand
            and atom.residue_name in _STANDARD_RESIDUE_NAMES
            and atom_molecule_type(atom) != MOLECULE_LIGAND
        ):
            new_tokens.extend(
                StructureToken(
                    residue=token.residue,
                    atom_indices=(atom_index,),
                    representative_atom_index=atom_index,
                )
                for atom_index in token.atom_indices
            )
        else:
            new_tokens.append(token)

    atom_to_new_token = _atom_to_token(tuple(new_tokens), len(structure.atoms))
    transformed = [
        (atom_to_new_token[atom_i], atom_to_new_token[atom_j], max_distance, min_distance)
        for atom_i, atom_j, max_distance, min_distance in atom_contact_pairs
    ]
    return tuple(new_tokens), transformed


def _contact_atom_feature(num_tokens: int, pairs: list[tuple[int, int, float, float]]) -> Tensor:
    r"""
    Build the symmetric atom-contact feature tensor from token-index min/max distance pairs.

    Args:
        num_tokens: Number of tokens.
        pairs: Contact pairs ``(token_i, token_j, max_distance, min_distance)``.

    Returns:
        Atom-contact feature ``[N_token, N_token, 2]`` (channel 0 min, channel 1 max distance).
    """

    feature = torch.zeros(num_tokens, num_tokens, 2, dtype=torch.float32)
    for token_i, token_j, max_distance, min_distance in pairs:
        feature[token_i, token_j, 1] = max_distance
        feature[token_j, token_i, 1] = max_distance
        feature[token_i, token_j, 0] = min_distance
        feature[token_j, token_i, 0] = min_distance
    return feature


def _contact_feature(
    num_tokens: int,
    specifics: list[tuple[Tensor, Tensor, float, float]],
    generator: torch.Generator | None,
) -> Tensor:
    r"""
    Build the symmetric token-contact feature by randomly picking one token per side.

    Follows the reference behavior of drawing a single representative token from each side's candidate
    token set for every contact.

    Args:
        num_tokens: Number of tokens.
        specifics: Per-contact ``(tokens_1, tokens_2, max_distance, min_distance)`` candidate sets.
        generator: Optional RNG for the random token selection.

    Returns:
        Token-contact feature ``[N_token, N_token, 2]`` (channel 0 min, channel 1 max distance).
    """

    feature = torch.zeros(num_tokens, num_tokens, 2, dtype=torch.float32)
    for tokens_1, tokens_2, max_distance, min_distance in specifics:
        token_i = tokens_1[_randint(tokens_1.shape[0], generator)]
        token_j = tokens_2[_randint(tokens_2.shape[0], generator)]
        feature[token_i, token_j, 1] = max_distance
        feature[token_j, token_i, 1] = max_distance
        feature[token_i, token_j, 0] = min_distance
        feature[token_j, token_i, 0] = min_distance
    return feature


def _pocket_feature(
    structure: Structure,
    *,
    num_tokens: int,
    atom_to_token_values: list[int],
    constraint: Mapping[str, Any],
    generator: torch.Generator | None,
) -> Tensor:
    r"""
    Build the pocket feature marking binder tokens against a random pocket contact token.

    For each configured contact residue, one representative pocket token is drawn and the max distance
    is written from every binder token to that pocket token.

    Args:
        structure: The parsed structure.
        num_tokens: Number of tokens.
        atom_to_token_values: Token index per atom.
        constraint: The official JSON ``constraint`` section (``pocket`` sub-section).
        generator: Optional RNG for the random pocket-token selection.

    Returns:
        Pocket feature ``[N_token, N_token, 1]``.

    Raises:
        ValueError: If a pocket contact residue shares the binder chain.
    """

    feature = torch.zeros(num_tokens, num_tokens, 1, dtype=torch.float32)
    pocket = constraint.get("pocket", {})
    if not pocket:
        return feature
    binder = pocket["binder_chain"]
    binder_tokens = _atom_indices_to_tokens(
        _select_atoms(structure, (binder["entity"], binder["copy"], None, None)), atom_to_token_values
    )
    if binder_tokens.numel() == 0:
        return feature
    for pocket_residue in pocket["contact_residues"]:
        if (binder["entity"], binder["copy"]) == (pocket_residue["entity"], pocket_residue["copy"]):
            raise ValueError("Pockets can not be the same chain with the binder")
        pocket_tokens = _atom_indices_to_tokens(
            _select_atoms(
                structure,
                (pocket_residue["entity"], pocket_residue["copy"], pocket_residue["position"], None),
            ),
            atom_to_token_values,
        )
        if pocket_tokens.numel() == 0:
            continue
        pocket_token = pocket_tokens[_randint(pocket_tokens.shape[0], generator)]
        feature[binder_tokens, pocket_token, 0] = float(pocket["max_distance"])
    return feature


def _substructure_feature(
    structure: Structure,
    *,
    tokens: tuple[StructureToken, ...],
    constraint: Mapping[str, Any],
) -> Tensor:
    r"""
    Build the one-hot binned substructure-distance feature over the selected token subset.

    Resolves the constrained token indices and their coordinates (explicit, or from representative
    atoms), bins the pairwise distances into four classes, and one-hot-encodes with padding for
    unconstrained pairs.

    Args:
        structure: The parsed structure.
        tokens: The tokenized structure.
        constraint: The official JSON ``constraint`` section (``structure`` sub-section).

    Returns:
        Substructure feature ``[N_token, N_token, 4]``.

    Raises:
        ValueError: If token indices and coordinates disagree in length, or an index is out of range.
    """

    num_tokens = len(tokens)
    substructure = constraint.get("structure", {})
    if not substructure:
        return torch.zeros(num_tokens, num_tokens, 4, dtype=torch.float32)

    token_indices = list(substructure.get("token_indices", []))
    token_coords = substructure.get("token_coords")
    if "tokens" in substructure:
        for selector in substructure["tokens"]:
            selected_atoms = _select_atoms(
                structure,
                (selector["entity"], selector["copy"], selector["position"], selector.get("atom")),
            )
            if not selected_atoms:
                continue
            atom_to_token_values = _atom_to_token(tokens, len(structure.atoms))
            token_indices.extend(_atom_indices_to_tokens(selected_atoms, atom_to_token_values).tolist())
        token_indices = sorted({int(index) for index in token_indices})
    if token_coords is None:
        token_coords = [
            [
                structure.atoms[tokens[int(token_index)].representative_atom_index].x,
                structure.atoms[tokens[int(token_index)].representative_atom_index].y,
                structure.atoms[tokens[int(token_index)].representative_atom_index].z,
            ]
            for token_index in token_indices
        ]

    if len(token_indices) != len(token_coords):
        raise ValueError("substructure token_indices and token_coords must have the same length")
    labels = torch.full((num_tokens, num_tokens), -1, dtype=torch.long)
    if token_indices:
        indices = torch.tensor(token_indices, dtype=torch.long)
        if torch.any((indices < 0) | (indices >= num_tokens)):
            raise ValueError("substructure token_indices must be valid zero-based token indices")
        coords = torch.tensor(token_coords, dtype=torch.float32)
        distances = torch.cdist(coords, coords)
        bins = torch.tensor([0.0, 4.0, 8.0, 16.0, torch.inf])
        labels[indices[:, None], indices[None, :]] = torch.bucketize(distances, bins, right=True) - 1
    return _one_hot_with_padding(labels, num_classes=4)


def _atom_indices_to_tokens(atom_indices: Sequence[int], atom_to_token_values: list[int]) -> Tensor:
    r"""
    Map atom indices to their unique owning token indices, preserving first-seen order.

    Args:
        atom_indices: Global atom indices.
        atom_to_token_values: Token index per atom.

    Returns:
        Deduplicated token indices as a long tensor.
    """

    seen: set[int] = set()
    tokens: list[int] = []
    for atom_index in atom_indices:
        token_index = atom_to_token_values[atom_index]
        if token_index not in seen:
            tokens.append(token_index)
            seen.add(token_index)
    return torch.tensor(tokens, dtype=torch.long)


def _one_hot_with_padding(labels: Tensor, *, num_classes: int) -> Tensor:
    r"""
    One-hot-encode integer labels, emitting all-zero vectors for padding (``-1``) entries.

    Args:
        labels: Integer labels ``[...]`` with ``-1`` marking padding.
        num_classes: Number of one-hot classes.

    Returns:
        One-hot tensor ``[..., num_classes]`` with zero rows where ``labels == -1``.
    """

    pad_mask = labels == -1
    safe = torch.where(pad_mask, torch.zeros_like(labels), labels)
    encoded = torch.nn.functional.one_hot(safe, num_classes=num_classes).float()
    encoded[pad_mask] = 0.0
    return encoded


def _batched_constraint_feature(
    feature: ProtenixConstraintFeatures | Mapping[str, Tensor] | None,
) -> ProtenixConstraintFeatures | None:
    r"""
    Add a leading batch axis to every constraint feature tensor.

    Args:
        feature: A constraint feature bundle or mapping, or ``None``.

    Returns:
        The batched feature bundle, or ``None`` when ``feature`` is ``None``.
    """

    if feature is None:
        return None
    if not isinstance(feature, ProtenixConstraintFeatures):
        feature = ProtenixConstraintFeatures.from_mapping(feature)
    return ProtenixConstraintFeatures(
        pocket=_with_batch_axis(feature.pocket, "pocket"),
        contact=_with_batch_axis(feature.contact, "contact"),
        contact_atom=_with_batch_axis(feature.contact_atom, "contact_atom"),
        substructure=_with_batch_axis(feature.substructure, "substructure"),
    )


def _empty_guidance_feature_tensors(*, dtype: torch.dtype) -> dict[str, Tensor]:
    r"""
    Return the empty (zero-length) guidance geometry feature tensors at their canonical widths.

    Args:
        dtype: Floating dtype for the value/bound tensors.

    Returns:
        Mapping from official Protenix feature name to its empty tensor.
    """

    return {
        "interchain_bond_index": torch.empty(2, 0, dtype=torch.long),
        "symmetric_chain_index": torch.empty(2, 0, dtype=torch.long),
        "pairwise_distance_index": torch.empty(2, 0, dtype=torch.long),
        "pairwise_distance_upper_bound": torch.empty(0, dtype=dtype),
        "pairwise_distance_lower_bound": torch.empty(0, dtype=dtype),
        "pairwise_distance_is_bond": torch.empty(0, dtype=torch.long),
        "pairwise_distance_is_angle": torch.empty(0, dtype=torch.long),
        "experimental_torsion_index": torch.empty(4, 0, dtype=torch.long),
        "experimental_torsion_force_constant": torch.empty(0, 6, dtype=dtype),
        "experimental_torsion_sign": torch.empty(0, 6, dtype=dtype),
        "linear_triple_bond_index": torch.empty(3, 0, dtype=torch.long),
        "chiral_index": torch.empty(4, 0, dtype=torch.long),
        "chiral_orientation": torch.empty(0, dtype=dtype),
        "stereo_bond_index": torch.empty(4, 0, dtype=torch.long),
        "stereo_bond_orientation": torch.empty(0, dtype=dtype),
        "planar_improper_index": torch.empty(4, 0, dtype=torch.long),
        "planar_improper_is_carbonyl": torch.empty(0, dtype=dtype),
    }


def _index_tensor(values: Sequence[tuple[int, ...]], *, width: int) -> Tensor:
    r"""
    Stack fixed-width index tuples into a transposed ``[width, N]`` long index tensor.

    Args:
        values: Index tuples, each of length ``width``.
        width: Expected tuple width (number of endpoints).

    Returns:
        Index tensor ``[width, N]`` (``[width, 0]`` when ``values`` is empty).

    Raises:
        ValueError: If the stacked tensor does not have the expected width.
    """

    if not values:
        return torch.empty(width, 0, dtype=torch.long)
    tensor = torch.tensor(values, dtype=torch.long)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"index tensor entries must have width {width}, got {tuple(tensor.shape)}")
    return tensor.T


def _symmetric_chain_index(fields: StructuralFields) -> Tensor:
    r"""
    Build the index of chain pairs sharing an entity (symmetry-equivalent chains).

    Args:
        fields: The structural fields for the tokenized structure.

    Returns:
        Symmetric-chain index ``[2, P]`` over ordered same-entity chain pairs.
    """

    chain_to_entity: dict[int, int] = {}
    for token, entity in zip(fields.tokens, fields.entity_values, strict=True):
        chain = fields.chain_to_index[token.residue.chain_id]
        chain_to_entity.setdefault(chain, entity)
    pairs = [
        (chain_i, chain_j)
        for chain_i, entity_i in chain_to_entity.items()
        for chain_j, entity_j in chain_to_entity.items()
        if chain_i < chain_j and entity_i == entity_j
    ]
    return _index_tensor(pairs, width=2)


def _rdkit_guidance_feature_tensors(fields: StructuralFields, ccd: CcdProvider) -> dict[str, Tensor]:
    r"""
    Build RDKit-derived per-ligand geometry features for the guidance potentials.

    For each non-standard residue with a CCD component, constructs an RDKit molecule and extracts
    distance bounds, experimental torsions, and chiral/planar/stereo/linear-bond indices, mapping
    local atom indices to global ones. Returns an empty dict when RDKit is unavailable.

    Args:
        fields: The structural fields for the tokenized structure.
        ccd: Chemical Component Dictionary provider.

    Returns:
        Mapping from official Protenix geometry feature name to its tensor.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix
        from rdkit.Chem.rdMolTransforms import GetDihedralRad
    except ImportError:
        return {}

    pair_index: list[tuple[int, int]] = []
    pair_lower: list[float] = []
    pair_upper: list[float] = []
    pair_is_bond: list[int] = []
    pair_is_angle: list[int] = []
    torsion_index: list[tuple[int, int, int, int]] = []
    torsion_force: list[list[float]] = []
    torsion_sign: list[list[float]] = []
    linear_index: list[tuple[int, int, int]] = []
    chiral_index: list[tuple[int, int, int, int]] = []
    chiral_orientation: list[float] = []
    stereo_index: list[tuple[int, int, int, int]] = []
    stereo_orientation: list[float] = []
    planar_index: list[tuple[int, int, int, int]] = []
    planar_is_carbonyl: list[float] = []

    for atom_indices in _residue_atom_groups_for_guidance(fields.structure):
        first_atom = fields.structure.atoms[atom_indices[0]]
        if first_atom.residue_name in _STANDARD_RESIDUE_NAMES and not first_atom.hetero:
            continue
        component = ccd.get(first_atom.residue_name)
        if component is None:
            continue
        mol = _component_to_rdkit_mol(component, Chem)
        if mol is None:
            continue
        local_to_global = _local_to_global_atom_map(fields.structure, atom_indices, component)
        if not local_to_global:
            continue

        bounds = GetMoleculeBoundsMatrix(mol)
        bond_pairs = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()}
        angle_pairs = {tuple(sorted((triple[0], triple[2]))) for triple in _angle_triples_from_bonds(bond_pairs)}
        for local_i in range(mol.GetNumAtoms() - 1):
            for local_j in range(local_i + 1, mol.GetNumAtoms()):
                if local_i not in local_to_global or local_j not in local_to_global:
                    continue
                pair_index.append((local_to_global[local_i], local_to_global[local_j]))
                pair_upper.append(float(bounds[local_i, local_j]))
                pair_lower.append(float(bounds[local_j, local_i]))
                pair = (local_i, local_j)
                pair_is_bond.append(int(pair in bond_pairs))
                pair_is_angle.append(int(pair in angle_pairs))

        for torsion in GetExperimentalTorsions(mol, useSmallRingTorsions=True):
            mapped = _map_rdkit_indices(torsion["atomIndices"], local_to_global, width=4)
            if mapped is None:
                continue
            torsion_index.append(mapped)
            torsion_force.append([float(value) for value in torsion["V"]])
            torsion_sign.append([float(value) for value in torsion["signs"]])

        conf = mol.GetConformer(0) if mol.GetNumConformers() else None
        for atom in mol.GetAtoms():
            center = atom.GetIdx()
            neighbors = sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
            if atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_TETRAHEDRAL_CW}:
                if conf is None or not (3 <= len(neighbors) <= 4):
                    continue
                for chiral_local_indices in combinations(neighbors, 3):
                    mapped = _map_rdkit_indices((*chiral_local_indices, center), local_to_global, width=4)
                    if mapped is None:
                        continue
                    chiral_index.append(mapped)
                    phi = GetDihedralRad(conf, *chiral_local_indices, center)
                    chiral_orientation.append(1.0 if phi >= 0.0 else -1.0)

            if atom.GetSymbol() in {"C", "N", "O"} and len(neighbors) == 3:
                if atom.GetHybridization() != Chem.HybridizationType.SP2:
                    continue
                nb1, nb2, nb3 = neighbors
                for local_indices in ((nb1, nb2, center, nb3), (nb3, nb1, center, nb2), (nb2, nb3, center, nb1)):
                    planar_mapped = _map_rdkit_indices(local_indices, local_to_global, width=4)
                    if planar_mapped is None:
                        continue
                    planar_index.append(planar_mapped)
                    planar_is_carbonyl.append(float(_is_carbonyl_center(atom)))

        for bond in mol.GetBonds():
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                linear_index.extend(_linear_bond_triples(begin, end, local_to_global, Chem))
            if bond.GetStereo() in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE}:
                mapped = _stereo_bond_quadruple(bond, local_to_global)
                if mapped is not None:
                    stereo_index.append(mapped)
                    stereo_orientation.append(float(bond.GetStereo() == Chem.BondStereo.STEREOE))

    return {
        "pairwise_distance_index": _index_tensor(pair_index, width=2),
        "pairwise_distance_upper_bound": torch.tensor(pair_upper, dtype=torch.float32),
        "pairwise_distance_lower_bound": torch.tensor(pair_lower, dtype=torch.float32),
        "pairwise_distance_is_bond": torch.tensor(pair_is_bond, dtype=torch.long),
        "pairwise_distance_is_angle": torch.tensor(pair_is_angle, dtype=torch.long),
        "experimental_torsion_index": _index_tensor(torsion_index, width=4),
        "experimental_torsion_force_constant": _matrix_tensor(torsion_force, width=6),
        "experimental_torsion_sign": _matrix_tensor(torsion_sign, width=6),
        "linear_triple_bond_index": _index_tensor(linear_index, width=3),
        "chiral_index": _index_tensor(chiral_index, width=4),
        "chiral_orientation": torch.tensor(chiral_orientation, dtype=torch.float32),
        "stereo_bond_index": _index_tensor(stereo_index, width=4),
        "stereo_bond_orientation": torch.tensor(stereo_orientation, dtype=torch.float32),
        "planar_improper_index": _index_tensor(planar_index, width=4),
        "planar_improper_is_carbonyl": torch.tensor(planar_is_carbonyl, dtype=torch.float32),
    }


def _component_to_rdkit_mol(component: ReferenceComponent, chem: Any) -> Any | None:
    r"""
    Build an RDKit molecule (with a reference conformer) from a CCD component.

    Args:
        component: The reference CCD component (elements, charges, bonds, coords).
        chem: The imported ``rdkit.Chem`` module.

    Returns:
        The sanitized RDKit molecule, or ``None`` if sanitization fails.
    """

    rw_mol = chem.RWMol()
    for element, charge in zip(component.elements, component.charge, strict=True):
        atom = chem.Atom(str(element).title())
        atom.SetFormalCharge(int(round(float(charge))))
        rw_mol.AddAtom(atom)
    bond_types = {
        "SINGLE": chem.BondType.SINGLE,
        "DOUBLE": chem.BondType.DOUBLE,
        "TRIPLE": chem.BondType.TRIPLE,
        "AROMATIC": chem.BondType.AROMATIC,
    }
    for name_i, name_j, bond_type in component.bonds:
        if name_i not in component.atom_map or name_j not in component.atom_map:
            continue
        rw_mol.AddBond(
            component.atom_map[name_i],
            component.atom_map[name_j],
            bond_types.get(bond_type, chem.BondType.SINGLE),
        )
    mol = rw_mol.GetMol()
    try:
        chem.SanitizeMol(mol)
    except Exception:
        return None
    conformer = chem.Conformer(mol.GetNumAtoms())
    for atom_index, (x, y, z) in enumerate(component.coords):
        conformer.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
    mol.AddConformer(conformer, assignId=True)
    return mol


def _local_to_global_atom_map(
    structure: Structure,
    atom_indices: Sequence[int],
    component: ReferenceComponent,
) -> dict[int, int]:
    r"""
    Map a component's local atom indices to global structure atom indices by atom name.

    Args:
        structure: The parsed structure.
        atom_indices: Global atom indices belonging to the residue.
        component: The reference CCD component with its atom-name map.

    Returns:
        Mapping from local (component) atom index to global atom index.
    """

    local_to_global = {}
    for atom_index in atom_indices:
        atom = structure.atoms[atom_index]
        local_index = component.atom_map.get(atom.name)
        if local_index is not None:
            local_to_global[int(local_index)] = atom_index
    return local_to_global


def _residue_atom_groups_for_guidance(structure: Structure) -> list[tuple[int, ...]]:
    r"""
    Group global atom indices by ``(residue, residue_name, hetero)`` for per-residue RDKit builds.

    Args:
        structure: The parsed structure.

    Returns:
        One tuple of global atom indices per residue group.
    """

    groups: dict[tuple[ResidueKey, str, bool], list[int]] = {}
    for atom_index, atom in enumerate(structure.atoms):
        groups.setdefault((atom.residue, atom.residue_name, atom.hetero), []).append(atom_index)
    return [tuple(atom_indices) for atom_indices in groups.values()]


def _angle_triples_from_bonds(bond_pairs: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    r"""
    Enumerate ``(i, center, j)`` angle triples from an undirected bond set.

    Args:
        bond_pairs: Set of undirected bonded atom-index pairs.

    Returns:
        Angle triples, one per pair of neighbors sharing a center atom.
    """

    neighbors: dict[int, list[int]] = {}
    for atom_i, atom_j in bond_pairs:
        neighbors.setdefault(atom_i, []).append(atom_j)
        neighbors.setdefault(atom_j, []).append(atom_i)
    triples = []
    for center, center_neighbors in neighbors.items():
        for atom_i, atom_j in combinations(sorted(center_neighbors), 2):
            triples.append((atom_i, center, atom_j))
    return triples


@overload
def _map_rdkit_indices(
    indices: Sequence[int],
    local_to_global: Mapping[int, int],
    *,
    width: Literal[3],
) -> tuple[int, int, int] | None: ...


@overload
def _map_rdkit_indices(
    indices: Sequence[int],
    local_to_global: Mapping[int, int],
    *,
    width: Literal[4],
) -> tuple[int, int, int, int] | None: ...


@overload
def _map_rdkit_indices(
    indices: Sequence[int],
    local_to_global: Mapping[int, int],
    *,
    width: int,
) -> tuple[int, ...] | None: ...


def _map_rdkit_indices(
    indices: Sequence[int],
    local_to_global: Mapping[int, int],
    *,
    width: int,
) -> tuple[int, ...] | None:
    r"""
    Map a fixed-width tuple of local RDKit atom indices to global indices.

    Args:
        indices: Local (component) atom indices.
        local_to_global: Mapping from local to global atom index.
        width: Expected number of indices.

    Returns:
        The mapped global-index tuple, or ``None`` if the width is wrong or any index is unmapped.
    """

    if len(indices) != width:
        return None
    mapped = []
    for index in indices:
        global_index = local_to_global.get(int(index))
        if global_index is None:
            return None
        mapped.append(global_index)
    return tuple(mapped)


def _matrix_tensor(values: Sequence[Sequence[float]], *, width: int) -> Tensor:
    r"""
    Stack fixed-width float rows into an ``[N, width]`` tensor.

    Args:
        values: Float rows, each of length ``width``.
        width: Expected row width.

    Returns:
        Float tensor ``[N, width]`` (``[0, width]`` when ``values`` is empty).

    Raises:
        ValueError: If the stacked tensor does not have the expected width.
    """

    if not values:
        return torch.empty(0, width, dtype=torch.float32)
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"matrix tensor entries must have width {width}, got {tuple(tensor.shape)}")
    return tensor


def _linear_bond_triples(
    begin: Any,
    end: Any,
    local_to_global: Mapping[int, int],
    chem: Any,
) -> list[tuple[int, int, int]]:
    r"""
    Enumerate ``(neighbor, begin, end)`` triples for an sp-sp triple bond (linearity constraint).

    Args:
        begin: RDKit begin atom of the triple bond.
        end: RDKit end atom of the triple bond.
        local_to_global: Mapping from local to global atom index.
        chem: The imported ``rdkit.Chem`` module.

    Returns:
        Mapped global-index triples, empty for aromatic or non-sp bonds.
    """

    if begin.GetIsAromatic() or end.GetIsAromatic():
        return []
    if begin.GetHybridization() != chem.HybridizationType.SP or end.GetHybridization() != chem.HybridizationType.SP:
        return []
    begin_index = begin.GetIdx()
    end_index = end.GetIdx()
    triples = []
    for neighbor in sorted(atom.GetIdx() for atom in begin.GetNeighbors() if atom.GetIdx() != end_index):
        mapped = _map_rdkit_indices((neighbor, begin_index, end_index), local_to_global, width=3)
        if mapped is not None:
            triples.append(mapped)
    for neighbor in sorted(atom.GetIdx() for atom in end.GetNeighbors() if atom.GetIdx() != begin_index):
        mapped = _map_rdkit_indices((begin_index, end_index, neighbor), local_to_global, width=3)
        if mapped is not None:
            triples.append(mapped)
    return triples


def _stereo_bond_quadruple(
    bond: Any,
    local_to_global: Mapping[int, int],
) -> tuple[int, int, int, int] | None:
    r"""
    Build the ``(stereo0, begin, end, stereo1)`` global-index quadruple for a stereo double bond.

    Args:
        bond: RDKit stereo double bond.
        local_to_global: Mapping from local to global atom index.

    Returns:
        The mapped global-index quadruple, or ``None`` if it lacks two stereo atoms or is unmapped.
    """

    stereo_atoms = list(bond.GetStereoAtoms())
    if len(stereo_atoms) != 2:
        return None
    return _map_rdkit_indices(
        (stereo_atoms[0], bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), stereo_atoms[1]),
        local_to_global,
        width=4,
    )


def _is_carbonyl_center(atom: Any) -> bool:
    r"""
    Return whether an atom is a carbonyl center (double-bonded to an oxygen).

    Args:
        atom: RDKit atom.

    Returns:
        ``True`` if the atom has a double bond to an oxygen.
    """

    return any(
        bond.GetBondTypeAsDouble() == 2.0 and bond.GetOtherAtom(atom).GetSymbol() == "O" for bond in atom.GetBonds()
    )


def _load_protenix_job(value: Any) -> Mapping[str, Any] | None:
    r"""
    Return an official Protenix job-JSON dict when ``value`` is one (decoded dict or ``.json`` path).

    A dict carrying a ``sequences`` section, or a path to such a ``.json`` file, is treated as an official
    job JSON. Parsed structures, structure file paths, and other inputs return ``None`` so the caller falls
    back to [`structure_from_input`][].
    """

    if isinstance(value, Mapping) and "sequences" in value:
        return value
    if isinstance(value, (str, PathLike)) and str(value).lower().endswith(".json"):
        from chanfig import load

        loaded = load(value)
        if not isinstance(loaded, Mapping) or "sequences" not in loaded:
            raise ValueError("official Protenix job JSON must decode to an object with a 'sequences' section")
        return loaded
    return None


def _config_scalar(config: Any, name: str, default: Any) -> Any:
    r"""
    Read a scalar field from a mapping- or object-style config, ignoring empty nested blocks.

    Args:
        config: The config mapping or object.
        name: Field name to read.
        default: Value returned when the field is absent or an empty mapping.

    Returns:
        The field value, or ``default``.
    """

    if isinstance(config, Mapping):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    if isinstance(value, Mapping) and not value:
        return default
    return value


def _randint(high: int, generator: torch.Generator | None) -> int:
    r"""
    Draw a single random integer in ``[0, high)``.

    Args:
        high: Exclusive upper bound.
        generator: Optional RNG for reproducibility.

    Returns:
        The sampled integer.
    """

    return int(torch.randint(high=high, size=(1,), generator=generator).item())


def _with_batch_axis(tensor: Tensor, name: str) -> Tensor:
    r"""
    Ensure a constraint pair feature has a leading batch axis.

    Args:
        tensor: An unbatched ``[N, N, C]`` or already-batched ``[B, N, N, C]`` feature.
        name: Feature name, used in error messages.

    Returns:
        The batched feature ``[B, N, N, C]``.

    Raises:
        ValueError: If the tensor is neither 3- nor 4-dimensional.
    """

    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(
        f"constraint_feature['{name}'] must be unbatched or batched pair features, got {tuple(tensor.shape)}"
    )


def _chain_info(structure: Structure) -> dict[str, tuple[int, int]]:
    r"""
    Map each chain id to its ``(entity, copy)`` numbers (both one-based).

    The copy number counts occurrences of the same entity in chain order.

    Args:
        structure: The parsed structure.

    Returns:
        Mapping from chain id to ``(entity_number, copy_number)``.
    """

    residue_names = {residue: _first_residue_name(structure, residue) for residue in structure.residues}
    entity_values = _entity_indices(structure.residues, residue_names)
    chain_entity: dict[str, int] = {}
    copy_counts: dict[int, int] = {}
    chain_info: dict[str, tuple[int, int]] = {}
    for residue, entity_value in zip(structure.residues, entity_values, strict=True):
        if residue.chain_id in chain_entity:
            continue
        entity = int(entity_value) + 1
        copy_counts[entity] = copy_counts.get(entity, 0) + 1
        chain_entity[residue.chain_id] = entity
        chain_info[residue.chain_id] = (entity, copy_counts[entity])
    return chain_info


def _symmetry_values(structure: Structure, tokens: tuple[StructureToken, ...]) -> list[int]:
    r"""
    Return the zero-based symmetry-copy index for each token.

    Args:
        structure: The parsed structure.
        tokens: The tokenized structure.

    Returns:
        Per-token zero-based copy index.
    """

    chain_info = _chain_info(structure)
    return [chain_info[token.residue.chain_id][1] - 1 for token in tokens]


def _entity_values_for_tokens(structure: Structure, tokens: tuple[StructureToken, ...]) -> list[int]:
    r"""
    Return the zero-based entity index for each token.

    Args:
        structure: The parsed structure.
        tokens: The tokenized structure.

    Returns:
        Per-token zero-based entity index.
    """

    chain_info = _chain_info(structure)
    return [chain_info[token.residue.chain_id][0] - 1 for token in tokens]


def _residue_positions(structure: Structure) -> dict[ResidueKey, int]:
    r"""
    Map each residue to its one-based position within its chain.

    Args:
        structure: The parsed structure.

    Returns:
        Mapping from residue key to chain-local one-based position.
    """

    positions: dict[ResidueKey, int] = {}
    chain_counts: dict[str, int] = {}
    for residue in structure.residues:
        chain_counts[residue.chain_id] = chain_counts.get(residue.chain_id, 0) + 1
        positions[residue] = chain_counts[residue.chain_id]
    return positions


def _first_residue_name(structure: Structure, residue: ResidueKey) -> str:
    r"""
    Return the residue name of the first atom belonging to a residue.

    Args:
        structure: The parsed structure.
        residue: The residue key to look up.

    Returns:
        The residue name.

    Raises:
        ValueError: If the residue has no atoms.
    """

    for atom in structure.atoms:
        if atom.residue == residue:
            return atom.residue_name
    raise ValueError(f"residue {residue} has no atoms")
