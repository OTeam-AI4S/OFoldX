r"""
Prepare monomer protein, MSA, and template features for AlphaFold 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.crop import Cropper
from ...data.evidence import (
    MSA,
    MSA_VOCAB_SIZE,
    MsaCollection,
    Template,
    TemplateAlignment,
    msa_component_index,
    msa_residue_index,
    resolve_msa_bindings,
)
from ...data.molecular import ENTITY_PROTEIN, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...data.tokenization import select_tokens, tokenize_molecular_system
from ...modules import residue_constants
from ...modules.geometry import virtual_beta_carbon
from ...processor import Processor
from .config import AlphaFold2ProcessorConfig
from .model import AlphaFold2InitialGuessInput, AlphaFold2Input

RESIDUE_TYPE_INDEX_BY_COMPONENT = {
    residue_constants.AMINO_ACID_ONE_TO_THREE[letter]: index
    for index, letter in enumerate(residue_constants.AMINO_ACID_ONE_LETTER_CODES)
}
RESIDUE_TYPE_INDEX_BY_COMPONENT["UNK"] = residue_constants.NUM_AMINO_ACID_TYPES
UNKNOWN_RESIDUE_TYPE_INDEX = residue_constants.NUM_AMINO_ACID_TYPES
GAP_RESIDUE_TYPE_INDEX = residue_constants.NUM_AMINO_ACID_TYPES + 1
_CHI_PI_PERIODIC = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
)


class AlphaFold2Processor(Processor):
    r"""
    Convert supported protein inputs into `AlphaFold2Input`.

    Args:
        config: Feature limits and preprocessing options. Defaults to
            ``AlphaFold2ProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = AlphaFold2ProcessorConfig
    config: AlphaFold2ProcessorConfig

    def __init__(
        self,
        config: AlphaFold2ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2ProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold2Input:
        r"""
        Prepare model features for one protein monomer.

        Args:
            input: Protein sequence, molecular system, or structure topology
                containing exactly one protein instance.
            msa: Optional monomer MSA value.
            templates: Template structures with explicit target alignments.
            cropper: Optional training crop over the bound target residue axis.
            generator: Optional random generator used by ``cropper``.

        Returns:
            Batched target, MSA, and template tensors. Atom coordinates in template
            features are in angstroms. Without a cropper, this method performs no
            random sampling.
        """

        topology = input.system if isinstance(input, Structure) else input
        system = topology if isinstance(topology, MolecularSystem) else monomer_system_from_sequences(topology)
        entity_index, instance_index, component_index = protein_monomer(system)
        if isinstance(msa, MsaCollection):
            raise ValueError("AlphaFold 2 monomer accepts an entity-bound MSA, not a multimer MsaCollection")
        if msa is not None:
            ((_value, target_instance_index, msa_entity_index),) = resolve_msa_bindings(msa, system)
            if msa_entity_index != entity_index or target_instance_index not in (None, instance_index):
                raise ValueError("MSA entity disagrees with the AlphaFold 2 target")
        normalized_templates = _normalize_templates(
            templates,
            system=system,
            target_entity_index=entity_index,
            target_instance_index=instance_index,
            target_length=component_index.numel(),
            max_templates=self.config.max_templates,
        )
        selection = _target_crop_selection(
            input,
            system,
            component_index,
            cropper,
            generator=generator,
        )
        return _prepare_alphafold2_features(
            system,
            component_index=component_index,
            target_entity_index=entity_index,
            target_instance_index=instance_index,
            msa=msa,
            templates=normalized_templates,
            selection=selection,
            device=self.device,
            dtype=self.dtype,
        )


class AlphaFold2InitialGuessProcessor(Processor):
    r"""
    Lower an in-memory target--binder structure to AF2 initial-guess features.

    The released path uses the pTM monomer graph over concatenated protein
    chains, query-only MSA features, input atom37 coordinates as the first
    recycled ``prev_pos`` state, and one all-GAP backbone/CB template whose
    target--binder pair features are masked by the Model. Residue indices retain
    every physical chain break, while the asymmetric-unit index groups all
    target chains separately from the binder. It does not initialize Structure
    Module frames.

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
        input: Structure,
        *,
        binder_instance_id: str,
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold2InitialGuessInput:
        r"""
        Prepare the released target--binder pTM evaluation input.

        Args:
            input: In-memory structure containing the target and designed binder.
            binder_instance_id: Exact protein instance to treat as the binder.
            cropper: Optional training crop. One target residue and one binder residue
                are required before applying the configured policy.
            generator: Optional random generator used by ``cropper`` and role anchors.

        Returns:
            Concatenated monomer/pTM features in target-then-binder order.
        """

        system = input.system
        binder_instance_index = system.instance_index(binder_instance_id)
        binder_entity_index = int(system.instance_entity[binder_instance_index])
        if int(system.entity_type[binder_entity_index]) != ENTITY_PROTEIN:
            raise ValueError("AlphaFold 2 initial guess requires a protein binder instance")
        target_instance_indices = tuple(
            instance_index
            for instance_index, entity_index in enumerate(system.instance_entity.tolist())
            if instance_index != binder_instance_index and int(system.entity_type[entity_index]) == ENTITY_PROTEIN
        )
        if not target_instance_indices:
            raise ValueError("AlphaFold 2 initial guess requires at least one protein target instance")
        chain_instance_indices = (*target_instance_indices, binder_instance_index)
        chain_components = tuple(
            torch.arange(
                int(system.instance_component_ptr[instance_index]),
                int(system.instance_component_ptr[instance_index + 1]),
                device=system.instance_component_ptr.device,
            )
            for instance_index in chain_instance_indices
        )
        if any(component_index.numel() == 0 for component_index in chain_components):
            raise ValueError("AlphaFold 2 initial guess requires nonempty protein chains")
        component_index = torch.cat(chain_components)
        residue_indices: list[int] = []
        next_index = 0
        for chain_component_index in chain_components:
            residue_indices.extend(range(next_index, next_index + chain_component_index.numel()))
            next_index = residue_indices[-1] + 50
        residue_index = torch.tensor(residue_indices, dtype=torch.long)
        binder_mask = torch.cat(
            [
                torch.full(
                    (chain_component_index.numel(),),
                    instance_index == binder_instance_index,
                    dtype=torch.bool,
                )
                for instance_index, chain_component_index in zip(
                    chain_instance_indices,
                    chain_components,
                    strict=True,
                )
            ]
        )
        required_token_mask = None
        if cropper is not None:
            required_token_mask = torch.zeros_like(binder_mask)
            for role_mask in (~binder_mask, binder_mask):
                candidates = torch.nonzero(role_mask, as_tuple=False).flatten()
                device = candidates.device if generator is None else generator.device
                offset = torch.randint(candidates.numel(), (), device=device, generator=generator)
                required_token_mask[candidates[offset.to(candidates.device)]] = True
        selection = _target_crop_selection(
            input,
            system,
            component_index,
            cropper,
            required_token_mask=required_token_mask,
            generator=generator,
        )
        base = _prepare_alphafold2_features(
            system,
            component_index=component_index,
            residue_index=residue_index,
            target_entity_index=int(system.instance_entity[chain_instance_indices[0]]),
            target_instance_index=chain_instance_indices[0],
            msa=None,
            templates=(),
            selection=selection,
            device=self.device,
            dtype=self.dtype,
        )
        if selection is not None:
            component_index = component_index[selection.to(component_index.device)]
            residue_index = residue_index[selection]
            binder_mask = binder_mask[selection]

        atom37_index = _atom37_indices(system, component_index)
        safe_atom_index = atom37_index.clamp_min(0)
        atom_mask = (atom37_index >= 0) & input.coordinate_mask[safe_atom_index]
        positions = input.coordinates[safe_atom_index].to(dtype=torch.float32)
        positions = torch.where(atom_mask.unsqueeze(-1), positions, torch.zeros_like(positions))
        atom_mask = atom_mask.to(torch.float32)

        n = residue_constants.atom_order["N"]
        ca = residue_constants.atom_order["CA"]
        c = residue_constants.atom_order["C"]
        cb = residue_constants.atom_order["CB"]
        reconstruct_cb = (
            atom_mask[:, n].bool() & atom_mask[:, ca].bool() & atom_mask[:, c].bool() & ~atom_mask[:, cb].bool()
        )
        positions[reconstruct_cb, cb] = virtual_beta_carbon(
            positions[reconstruct_cb, n],
            positions[reconstruct_cb, ca],
            positions[reconstruct_cb, c],
        )
        atom_mask[reconstruct_cb, cb] = 1.0

        template_residue_type_ids = torch.full(
            (1, component_index.numel()),
            GAP_RESIDUE_TYPE_INDEX,
            dtype=torch.long,
        )
        template_positions = positions.unsqueeze(0)
        template_atom_mask = atom_mask.clone()
        template_atom_mask[:, 5:] = 0.0
        template_atom_mask = template_atom_mask.unsqueeze(0)
        torsion = torch.zeros(1, component_index.numel(), 7, 2)
        alternative_torsion = torch.zeros_like(torsion)
        torsion_mask = torch.zeros(1, component_index.numel(), 7)

        values = {field.name: getattr(base, field.name) for field in fields(base)}  # type: ignore[arg-type]
        values.update(
            residue_index=residue_index.to(device=self.device).unsqueeze(0),
            template_mask=torch.ones(
                1,
                1,
                dtype=self.dtype,
                device=self.device,
            ),
            template_residue_type_ids=template_residue_type_ids.to(self.device).unsqueeze(0),
            template_all_atom_positions=template_positions.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            template_all_atom_mask=template_atom_mask.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            template_torsion_angles_sin_cos=torsion.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            template_alternative_torsion_angles_sin_cos=alternative_torsion.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            template_torsion_angles_mask=torsion_mask.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            system=system,
            component_index=component_index.to(device=self.device).unsqueeze(0),
            atom37_index=atom37_index.to(device=self.device).unsqueeze(0),
            initial_recycling_positions=positions.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            chain_index=binder_mask.to(device=self.device, dtype=torch.long).unsqueeze(0),
            binder_mask=binder_mask.to(device=self.device).unsqueeze(0),
        )
        return AlphaFold2InitialGuessInput(**values, batch_size=[])  # type: ignore[call-arg]


def monomer_system_from_sequences(
    input: PolymerSequence | Sequence[PolymerSequence],
) -> MolecularSystem:
    r"""
    Construct the one-polymer convenience input accepted by monomer AlphaFold 2.
    """

    if isinstance(input, PolymerSequence):
        named = [("protein", input)]
    else:
        named = [("protein", sequence) for sequence in input]
    if len(named) != 1:
        raise ValueError("AlphaFold 2 monomer requires exactly one PolymerSequence")
    entity_id, sequence = named[0]
    return MolecularSystem.from_entities(
        entities={entity_id: sequence},
        instances={entity_id: entity_id},
    )


def protein_monomer(system: MolecularSystem) -> tuple[int, int, Tensor]:
    r"""
    Return the single protein entity, instance, and component axes.
    """

    if system.instance_entity.numel() != 1 or system.entity_type.numel() != 1:
        raise ValueError("AlphaFold 2 monomer requires one protein instance")
    instance_index = 0
    entity_index = int(system.instance_entity[instance_index])
    if int(system.entity_type[entity_index]) != ENTITY_PROTEIN:
        raise ValueError("AlphaFold 2 monomer requires a protein PolymerSequence")
    component_index = torch.arange(
        int(system.instance_component_ptr[instance_index]),
        int(system.instance_component_ptr[instance_index + 1]),
        device=system.instance_component_ptr.device,
    )
    for component_id in _component_ids(system, component_index):
        if component_id not in RESIDUE_TYPE_INDEX_BY_COMPONENT:
            raise ValueError(f"AlphaFold 2 monomer does not support protein component {component_id!r}")
    return entity_index, instance_index, component_index


def _component_ids(system: MolecularSystem, component_index: Tensor) -> tuple[str, ...]:
    return tuple(system.symbols[symbol] for symbol in system.component_symbol[component_index].detach().cpu().tolist())


def _atom37_indices(system: MolecularSystem, component_index: Tensor) -> Tensor:
    indices = torch.full(
        (component_index.numel(), residue_constants.atom_type_num),
        -1,
        dtype=torch.long,
        device=system.component_atom_ptr.device,
    )
    for residue_index, component in enumerate(component_index.detach().cpu().tolist()):
        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        for atom in range(start, stop):
            atom_name = system.symbols[int(system.atom_symbol[atom])]
            atom37 = residue_constants.atom_order.get(atom_name)
            if atom37 is not None:
                indices[residue_index, atom37] = atom
    return indices


def _target_crop_selection(
    input: object,
    system: MolecularSystem,
    component_index: Tensor,
    cropper: Cropper | None,
    *,
    required_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor | None:
    r"""Crop a bound component-token target and return its original axis indices."""

    if cropper is None:
        return None
    if not isinstance(input, Structure):
        raise TypeError("AlphaFold 2 cropping requires a Structure input")

    component_mask = torch.zeros_like(system.component_symbol, dtype=torch.bool)
    component_mask[component_index] = True
    layout = tokenize_molecular_system(
        system,
        atomize_nonstandard_polymers=False,
        component_mask=component_mask,
    )
    layout_row_by_component = torch.full_like(system.component_symbol, -1)
    layout_row_by_component[layout.token_component_index] = torch.arange(
        layout.token_component_index.numel(),
        device=layout.token_component_index.device,
    )
    target_layout_rows = layout_row_by_component[component_index]
    layout = select_tokens(layout, target_layout_rows)
    selection = cropper(
        input,
        layout,
        required_token_mask=required_token_mask,
        generator=generator,
    )
    return selection.to(device="cpu", dtype=torch.long)


def _normalize_templates(
    templates: Sequence[Template],
    *,
    system: MolecularSystem,
    target_entity_index: int,
    target_instance_index: int,
    target_length: int,
    max_templates: int,
) -> tuple[Template, ...]:
    r"""
    Bind explicit template alignments to the AlphaFold 2 target.
    """

    if len(templates) > max_templates:
        raise ValueError(f"AlphaFold 2 accepts at most {max_templates} templates")
    normalized: list[Template] = []
    for template in templates:
        select_template_alignment(
            template,
            system=system,
            target_length=target_length,
            target_entity_index=target_entity_index,
            target_instance_index=target_instance_index,
        )
        normalized.append(template)
    return tuple(normalized)


def select_template_alignment(
    template: Template,
    *,
    system: MolecularSystem,
    target_length: int,
    target_entity_index: int,
    target_instance_index: int,
) -> TemplateAlignment:
    r"""
    Return the one Template alignment that binds to the single AlphaFold 2 target.

    AlphaFold 2 folds one monomer target, so exactly one of a Template's
    [`TemplateAlignment`][] values may name that target entity and, when it
    pins an instance, that exact target instance. The target-side upper bound is
    the Processor's job: [`TemplateAlignment`][] holds no target sequence.
    """

    matches = []
    for alignment in template.alignments:
        try:
            entity_index = system.entity_index(alignment.target_entity_id)
            instance_index = (
                None if alignment.target_instance_id is None else system.instance_index(alignment.target_instance_id)
            )
        except KeyError:
            continue
        if entity_index == target_entity_index and instance_index in (None, target_instance_index):
            matches.append(alignment)
    if not matches:
        raise ValueError("Template has no alignment for the AlphaFold 2 target")
    if len(matches) != 1:
        raise ValueError("Template has multiple alignments for the AlphaFold 2 target")
    alignment = matches[0]
    if alignment.residue_pairs[-1][0] >= target_length:
        raise ValueError("template alignment target index is outside the AlphaFold 2 target")
    return alignment


def _prepare_alphafold2_features(
    system: MolecularSystem,
    *,
    component_index: Tensor,
    target_entity_index: int,
    target_instance_index: int,
    msa: MSA | None,
    templates: tuple[Template, ...],
    device: torch.device,
    dtype: torch.dtype,
    residue_index: Tensor | None = None,
    selection: Tensor | None = None,
) -> AlphaFold2Input:
    r"""
    Build the monomer AlphaFold 2 raw MSA pool, profile, and semantic axes.

    The Model samples and masks MSA rows for each prediction. This function keeps
    every bound alignment row, its per-cell deletion counts, and the full-row-pool
    HHblits profile on the selected target columns.
    """

    component_ids = _component_ids(system, component_index)
    num_residues = len(component_ids)
    (
        template_residue_type_ids,
        template_mask,
        template_all_atom_positions,
        template_all_atom_mask,
        template_torsion_angles_sin_cos,
        template_alternative_torsion_angles_sin_cos,
        template_torsion_angles_mask,
    ) = _template_features(
        templates,
        target_system=system,
        target_length=num_residues,
        target_entity_index=target_entity_index,
        target_instance_index=target_instance_index,
    )
    if residue_index is None:
        residue_index = torch.arange(num_residues)
    msa_column_index = None
    if selection is not None:
        msa_column_index = selection.tolist()
        component_ids = tuple(component_ids[index] for index in msa_column_index)
        template_residue_type_ids = template_residue_type_ids[:, selection]
        template_all_atom_positions = template_all_atom_positions[:, selection]
        template_all_atom_mask = template_all_atom_mask[:, selection]
        # Torsions depend on sequence neighbours, so compute them on the complete
        # template axis before selecting possibly non-contiguous crop columns.
        template_torsion_angles_sin_cos = template_torsion_angles_sin_cos[:, selection]
        template_alternative_torsion_angles_sin_cos = template_alternative_torsion_angles_sin_cos[:, selection]
        template_torsion_angles_mask = template_torsion_angles_mask[:, selection]
        residue_index = residue_index[selection.to(residue_index.device)]
        component_index = component_index[selection.to(component_index.device)]
    residue_type_ids = torch.tensor(
        [RESIDUE_TYPE_INDEX_BY_COMPONENT[component_id] for component_id in component_ids],
        dtype=torch.long,
    )
    token_features = torch.cat(
        [
            torch.zeros(residue_type_ids.shape[0], 1),
            F.one_hot(residue_type_ids, num_classes=residue_constants.NUM_AMINO_ACID_TYPES + 1),
        ],
        dim=-1,
    ).to(torch.float32)
    msa_tokens, msa_deletions, msa_profile = _msa_pool_features(
        component_ids,
        msa,
        column_index=msa_column_index,
    )
    floating = {
        "token_features": token_features,
        "token_mask": torch.ones(residue_type_ids.shape[0]),
        "msa_deletions": msa_deletions,
        "msa_profile": msa_profile,
        "template_mask": template_mask,
        "template_all_atom_positions": template_all_atom_positions,
        "template_all_atom_mask": template_all_atom_mask,
        "template_torsion_angles_sin_cos": template_torsion_angles_sin_cos,
        "template_alternative_torsion_angles_sin_cos": template_alternative_torsion_angles_sin_cos,
        "template_torsion_angles_mask": template_torsion_angles_mask,
    }
    placed = {name: value.to(device=device, dtype=dtype).unsqueeze(0) for name, value in floating.items()}
    return AlphaFold2Input(  # type: ignore[call-arg]
        residue_type_ids=residue_type_ids.to(device=device).unsqueeze(0),
        residue_index=residue_index.to(device=device).unsqueeze(0),
        msa_tokens=msa_tokens.to(device=device).unsqueeze(0),
        system=system,
        component_index=component_index.to(device=device).unsqueeze(0),
        atom37_index=_atom37_indices(system, component_index).to(device=device).unsqueeze(0),
        template_residue_type_ids=template_residue_type_ids.to(device=device).unsqueeze(0),
        **placed,
        batch_size=[],
    )


def _msa_pool_features(
    component_ids: tuple[str, ...],
    msa: MSA | None,
    *,
    column_index: Sequence[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build the query-projected alignment pool and its HHblits profile.

    Returns ``msa_tokens [M, N]`` (canonical MSA indices, row 0 is the query),
    ``msa_deletions [M, N]`` (float per-cell insertion counts), and
    ``msa_profile [N, 38]`` -- the mean of the canonical one-hot tokens over the
    whole pool, computed before sampling or masking.
    """

    query = torch.tensor(
        [msa_component_index(component_id, "protein") for component_id in component_ids],
        dtype=torch.long,
    )
    if msa is None:
        tokens = query.unsqueeze(0)
        deletion_counts = torch.zeros_like(tokens, dtype=torch.float32)
    else:
        # The pool is query-width, row 0 is the ungapped query, and column ``i``
        # addresses the i-th target component. Deletions are per-cell insertion counts.
        if column_index is None:
            tokens = torch.tensor(
                [[msa_residue_index(symbol, "protein") for symbol in row] for row in msa.unpaired_sequences],
                dtype=torch.long,
            )
            deletion_counts = torch.tensor(msa.unpaired_deletion_counts, dtype=torch.float32)
        else:
            tokens = torch.tensor(
                [
                    [msa_residue_index(row[index], "protein") for index in column_index]
                    for row in msa.unpaired_sequences
                ],
                dtype=torch.long,
            )
            deletion_counts = torch.tensor(
                msa.unpaired_deletion_counts[:, list(column_index)],
                dtype=torch.float32,
            )
        tokens[0] = query
    msa_profile = F.one_hot(tokens, num_classes=MSA_VOCAB_SIZE).to(torch.float32).mean(dim=0)
    return tokens, deletion_counts, msa_profile


def _template_features(
    templates: tuple[Template, ...],
    *,
    target_system: MolecularSystem,
    target_length: int,
    target_entity_index: int,
    target_instance_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Project aligned template coordinates into atom37 and torsion features.
    """

    if not templates:
        return (
            torch.zeros(1, target_length, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, target_length, residue_constants.atom_type_num, 3),
            torch.zeros(1, target_length, residue_constants.atom_type_num),
            torch.zeros(1, target_length, 7, 2),
            torch.zeros(1, target_length, 7, 2),
            torch.zeros(1, target_length, 7),
        )
    template_residue_type_ids: list[Tensor] = []
    template_positions: list[Tensor] = []
    template_masks: list[Tensor] = []
    for template in templates:
        alignment = select_template_alignment(
            template,
            system=target_system,
            target_length=target_length,
            target_entity_index=target_entity_index,
            target_instance_index=target_instance_index,
        )
        structure = template.structure
        template_system = structure.system
        template_instance_index = template_system.instance_index(alignment.template_instance_id)
        template_entity_index = int(template_system.instance_entity[template_instance_index])
        if int(template_system.entity_type[template_entity_index]) != ENTITY_PROTEIN:
            raise ValueError("AlphaFold 2 templates must align a protein instance")
        template_component_start = int(template_system.instance_component_ptr[template_instance_index])
        residue_type_ids = torch.full((target_length,), GAP_RESIDUE_TYPE_INDEX, dtype=torch.long)
        positions = torch.zeros(target_length, residue_constants.atom_type_num, 3)
        masks = torch.zeros(target_length, residue_constants.atom_type_num)
        for target_index, template_index in alignment.residue_pairs:
            component = template_component_start + template_index
            component_id = template_system.symbols[int(template_system.component_symbol[component])]
            residue_type_ids[target_index] = RESIDUE_TYPE_INDEX_BY_COMPONENT.get(
                component_id, UNKNOWN_RESIDUE_TYPE_INDEX
            )
            atom_start = int(template_system.component_atom_ptr[component])
            atom_stop = int(template_system.component_atom_ptr[component + 1])
            for atom in range(atom_start, atom_stop):
                atom_index = residue_constants.atom_order.get(
                    template_system.symbols[int(template_system.atom_symbol[atom])]
                )
                if atom_index is None or not bool(structure.coordinate_mask[atom]):
                    continue
                positions[target_index, atom_index] = (
                    structure.coordinates[atom]
                    .detach()
                    .to(
                        device=positions.device,
                        dtype=positions.dtype,
                    )
                )
                masks[target_index, atom_index] = 1.0
        template_residue_type_ids.append(residue_type_ids)
        template_positions.append(positions)
        template_masks.append(masks)
    residue_type_ids = torch.stack(template_residue_type_ids)
    positions = torch.stack(template_positions)
    masks = torch.stack(template_masks)
    torsion, alternative_torsion, torsion_mask = _atom37_to_torsion_angles(
        residue_type_ids,
        positions,
        masks,
    )
    return (
        residue_type_ids,
        torch.ones(len(templates)),
        positions,
        masks,
        torsion,
        alternative_torsion,
        torsion_mask,
    )


def _atom37_to_torsion_angles(
    residue_type_ids: Tensor,
    positions: Tensor,
    atom_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute pre-omega, phi, psi, and chi sin/cos values from atom37 coordinates.
    """

    previous_positions = torch.cat([torch.zeros_like(positions[..., :1, :, :]), positions[..., :-1, :, :]], dim=-3)
    previous_mask = torch.cat([torch.zeros_like(atom_mask[..., :1, :]), atom_mask[..., :-1, :]], dim=-2)
    pre_omega = torch.cat([previous_positions[..., 1:3, :], positions[..., :2, :]], dim=-2)
    phi = torch.cat([previous_positions[..., 2:3, :], positions[..., :3, :]], dim=-2)
    psi = torch.cat([positions[..., :3, :], positions[..., 4:5, :]], dim=-2)
    pre_omega_mask = previous_mask[..., 1:3].prod(dim=-1) * atom_mask[..., :2].prod(dim=-1)
    phi_mask = previous_mask[..., 2] * atom_mask[..., :3].prod(dim=-1)
    psi_mask = atom_mask[..., :3].prod(dim=-1) * atom_mask[..., 4]

    chi_indices = torch.zeros(21, 4, 4, dtype=torch.long, device=residue_type_ids.device)
    for residue_type_index, letter in enumerate(residue_constants.AMINO_ACID_ONE_LETTER_CODES):
        for chi_index, atom_names in enumerate(
            residue_constants.chi_angles_atoms[residue_constants.AMINO_ACID_ONE_TO_THREE[letter]]
        ):
            chi_indices[residue_type_index, chi_index] = torch.tensor(
                [residue_constants.atom_order[name] for name in atom_names],
                device=residue_type_ids.device,
            )
    selected_indices = chi_indices[residue_type_ids.clamp(max=20)]
    expanded = selected_indices.unsqueeze(-1).expand(*selected_indices.shape, 3)
    chi_positions = positions.unsqueeze(-3).expand(*positions.shape[:-2], 4, *positions.shape[-2:]).gather(-2, expanded)
    chi_atom_mask = (
        atom_mask.unsqueeze(-2).expand(*atom_mask.shape[:-1], 4, atom_mask.shape[-1]).gather(-1, selected_indices)
    )
    chi_exists = atom_mask.new_tensor((*residue_constants.chi_angles_mask, (0.0, 0.0, 0.0, 0.0)))[
        residue_type_ids.clamp(max=20)
    ]
    chi_mask = chi_atom_mask.prod(dim=-1) * chi_exists
    torsion_atoms = torch.cat(
        [pre_omega.unsqueeze(-3), phi.unsqueeze(-3), psi.unsqueeze(-3), chi_positions],
        dim=-3,
    )
    torsion_mask = torch.cat(
        [pre_omega_mask.unsqueeze(-1), phi_mask.unsqueeze(-1), psi_mask.unsqueeze(-1), chi_mask],
        dim=-1,
    )
    p0, p1, p2, p3 = torsion_atoms.unbind(dim=-2)
    e0_x, e0_y, e0_z = (p2 - p1).unbind(dim=-1)
    e0_norm = torch.sqrt(e0_x.square() + e0_y.square() + e0_z.square() + 1e-8)
    e0_x, e0_y, e0_z = e0_x / e0_norm, e0_y / e0_norm, e0_z / e0_norm

    e1_x, e1_y, e1_z = (p0 - p2).unbind(dim=-1)
    projection = e1_x * e0_x + e1_y * e0_y + e1_z * e0_z
    e1_x = e1_x - projection * e0_x
    e1_y = e1_y - projection * e0_y
    e1_z = e1_z - projection * e0_z
    e1_norm = torch.sqrt(e1_x.square() + e1_y.square() + e1_z.square() + 1e-8)
    e1_x, e1_y, e1_z = e1_x / e1_norm, e1_y / e1_norm, e1_z / e1_norm

    e2_x = e0_y * e1_z - e0_z * e1_y
    e2_y = e0_z * e1_x - e0_x * e1_z
    e2_z = e0_x * e1_y - e0_y * e1_x
    fourth_x, fourth_y, fourth_z = (p3 - p2).unbind(dim=-1)
    sine = fourth_x * e2_x + fourth_y * e2_y + fourth_z * e2_z
    cosine = fourth_x * e1_x + fourth_y * e1_y + fourth_z * e1_z
    angle_norm = torch.sqrt(sine.square() + cosine.square() + 1e-8)
    angles = torch.stack((sine / angle_norm, cosine / angle_norm), dim=-1)
    angles = angles * angles.new_tensor((1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0))[..., None]
    periodic = angles.new_tensor(_CHI_PI_PERIODIC)[residue_type_ids.clamp(max=20)]
    mirror = torch.cat([torch.ones_like(periodic[..., :3]), 1.0 - 2.0 * periodic], dim=-1)
    return angles, angles * mirror.unsqueeze(-1), torsion_mask
