r"""
Prepare assembly, paired-MSA, and template features for AlphaFold-Multimer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.crop import Cropper
from ...data.evidence import (
    MSA,
    MSA_GAP_INDEX,
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
from ...modules import residue_constants
from ...processor import Processor
from .config import AlphaFold2MultimerProcessorConfig
from .model import AlphaFold2MultimerInput
from .processor import (
    GAP_RESIDUE_TYPE_INDEX,
    RESIDUE_TYPE_INDEX_BY_COMPONENT,
    UNKNOWN_RESIDUE_TYPE_INDEX,
    _atom37_indices,
    _component_ids,
    _target_crop_selection,
    select_template_alignment,
)


class AlphaFold2MultimerProcessor(Processor):
    r"""
    Convert supported protein assemblies into `AlphaFold2MultimerInput`.

    Args:
        config: MSA and template feature limits. Defaults to
            ``AlphaFold2MultimerProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = AlphaFold2MultimerProcessorConfig
    config: AlphaFold2MultimerProcessorConfig

    def __init__(
        self,
        config: AlphaFold2MultimerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2MultimerProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: Structure | MolecularSystem | Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
        *,
        msa: MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold2MultimerInput:
        r"""
        Prepare model features for one protein assembly.

        Args:
            input: Structure, molecular system, or named/ordered protein sequences
                forming one assembly.
            msa: Optional paired and unpaired MSA evidence.
            templates: Template structures with explicit target alignments.
            cropper: Optional training crop over the bound concatenated residue axis.
            generator: Optional random generator used by ``cropper``.

        Returns:
            Batched target, paired-MSA, and template tensors. Atom coordinates in
            template features are in angstroms. Without a cropper, this method
            performs no random sampling.
        """

        system = input.system if isinstance(input, Structure) else input
        if not isinstance(system, MolecularSystem):
            system = multimer_system(system)
        chains = multimer_chains(system)
        msa_bindings = () if msa is None else resolve_msa_bindings(msa, system)
        normalized_templates = tuple(templates)
        if len(normalized_templates) > self.config.max_templates:
            raise ValueError(f"AlphaFold-Multimer accepts at most {self.config.max_templates} templates")
        return prepare_multimer_features(
            system,
            chains=chains,
            msa=msa,
            msa_bindings=msa_bindings,
            templates=normalized_templates,
            crop_input=input,
            cropper=cropper,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )


class MultimerChain:
    r"""
    One resolved chain: its instance, entity identity indices, and token range.
    """

    __slots__ = (
        "instance_index",
        "system_entity_index",
        "chain_index",
        "model_entity_index",
        "symmetry_index",
        "component_index",
        "start",
        "stop",
    )

    def __init__(
        self,
        *,
        instance_index: int,
        system_entity_index: int,
        chain_index: int,
        model_entity_index: int,
        symmetry_index: int,
        component_index: Tensor,
        start: int,
        stop: int,
    ) -> None:
        self.instance_index = instance_index
        self.system_entity_index = system_entity_index
        self.chain_index = chain_index
        self.model_entity_index = model_entity_index
        self.symmetry_index = symmetry_index
        self.component_index = component_index
        self.start = start
        self.stop = stop


def multimer_system(
    input: Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
) -> MolecularSystem:
    r"""
    Lift a named/ordered ``PolymerSequence`` collection into one MolecularSystem.

    Distinct sequences become distinct entities; repeated identical sequences share
    one entity, so the symmetry-copy structure is preserved.
    """

    if isinstance(input, Mapping):
        named = list(input.items())
    else:
        named = [(f"chain_{index}", sequence) for index, sequence in enumerate(input)]
    if not named:
        raise ValueError("AlphaFold-Multimer requires at least one PolymerSequence")

    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    sequence_to_entity: dict[PolymerSequence, str] = {}
    for instance_id, sequence in named:
        entity_id = sequence_to_entity.get(sequence)
        if entity_id is None:
            entity_id = f"entity_{len(entities)}"
            sequence_to_entity[sequence] = entity_id
            entities[entity_id] = sequence
        instances[instance_id] = entity_id
    return MolecularSystem.from_entities(entities=entities, instances=instances)


def multimer_chains(system: MolecularSystem) -> tuple[MultimerChain, ...]:
    r"""
    Resolve chain identity and reject monomer/unsupported molecular meaning.
    """

    if system.instance_entity.numel() < 2:
        raise ValueError("AlphaFold-Multimer requires at least two chain instances; use alphafold2 for monomers")
    model_entity_index: dict[int, int] = {}
    symmetry_count: dict[int, int] = {}
    chains: list[MultimerChain] = []
    cursor = 0
    for instance_index, system_entity_index in enumerate(system.instance_entity.tolist()):
        if int(system.entity_type[system_entity_index]) != ENTITY_PROTEIN:
            raise ValueError("AlphaFold-Multimer requires protein PolymerSequence chains")
        component_index = torch.arange(
            int(system.instance_component_ptr[instance_index]),
            int(system.instance_component_ptr[instance_index + 1]),
            device=system.instance_component_ptr.device,
        )
        for component_id in _component_ids(system, component_index):
            if component_id not in RESIDUE_TYPE_INDEX_BY_COMPONENT:
                raise ValueError(f"AlphaFold-Multimer does not support protein component {component_id!r}")
        if system_entity_index not in model_entity_index:
            model_entity_index[system_entity_index] = len(model_entity_index) + 1
        symmetry_count[system_entity_index] = symmetry_count.get(system_entity_index, 0) + 1
        length = component_index.numel()
        chains.append(
            MultimerChain(
                instance_index=instance_index,
                system_entity_index=system_entity_index,
                chain_index=len(chains) + 1,
                model_entity_index=model_entity_index[system_entity_index],
                symmetry_index=symmetry_count[system_entity_index],
                component_index=component_index,
                start=cursor,
                stop=cursor + length,
            )
        )
        cursor += length
    return tuple(chains)


def prepare_multimer_features(
    system: MolecularSystem,
    *,
    chains: tuple[MultimerChain, ...],
    msa: MsaCollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    templates: tuple[Template, ...],
    crop_input: object,
    cropper: Cropper | None,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold2MultimerInput:
    r"""
    Build the concatenated multimer tensor and semantic axes.
    """

    total_length = chains[-1].stop
    component_index = torch.cat([chain.component_index for chain in chains])
    residue_index = torch.cat([torch.arange(chain.stop - chain.start) for chain in chains]).long()
    chain_index = torch.cat([torch.full((chain.stop - chain.start,), chain.chain_index) for chain in chains]).long()
    entity_index = torch.cat(
        [torch.full((chain.stop - chain.start,), chain.model_entity_index) for chain in chains]
    ).long()
    symmetry_index = torch.cat(
        [torch.full((chain.stop - chain.start,), chain.symmetry_index) for chain in chains]
    ).long()

    selection = _target_crop_selection(
        crop_input,
        system,
        component_index,
        cropper,
        generator=generator,
    )
    msa_column_index = None
    if selection is not None:
        msa_column_index = selection.tolist()
        residue_index = residue_index[selection]
        chain_index = chain_index[selection]
        entity_index = entity_index[selection]
        symmetry_index = symmetry_index[selection]
        component_index = component_index[selection.to(component_index.device)]
    component_ids = _component_ids(system, component_index)
    residue_type_ids = torch.tensor(
        [RESIDUE_TYPE_INDEX_BY_COMPONENT[component_id] for component_id in component_ids],
        dtype=torch.long,
    )
    token_features = F.one_hot(residue_type_ids, num_classes=residue_constants.NUM_AMINO_ACID_TYPES + 1).to(
        torch.float32
    )
    query_tokens = torch.tensor(
        [msa_component_index(component_id, "protein") for component_id in component_ids],
        dtype=torch.long,
    )
    tokens, deletions = materialize_collection(
        msa,
        msa_bindings,
        chains,
        total_length,
        query_tokens,
        column_index=msa_column_index,
    )
    # The Processor emits the raw paired alignment pool and its full-pool profile.
    # The Model samples clustered and extra rows once per recycle. ``msa_profile``
    # is the mean of the canonical one-hot tokens before sampling or masking.
    msa_profile = F.one_hot(tokens, num_classes=MSA_VOCAB_SIZE).to(torch.float32).mean(dim=0)
    template_values = multimer_template_features(
        templates,
        system=system,
        chains=chains,
        total_length=total_length,
        column_index=msa_column_index,
    )

    floating = {
        "token_features": token_features,
        "token_mask": torch.ones(residue_type_ids.shape[0]),
        "msa_deletions": deletions,
        "msa_profile": msa_profile,
        "template_mask": template_values["template_mask"],
        "template_all_atom_positions": template_values["template_all_atom_positions"],
        "template_all_atom_mask": template_values["template_all_atom_mask"],
        "template_pseudo_beta": template_values["template_pseudo_beta"],
        "template_pseudo_beta_mask": template_values["template_pseudo_beta_mask"],
        "template_backbone_frame": template_values["template_backbone_frame"],
        "template_backbone_frame_mask": template_values["template_backbone_frame_mask"],
    }
    placed = {name: value.to(device=device, dtype=dtype).unsqueeze(0) for name, value in floating.items()}
    integral = {
        "residue_type_ids": residue_type_ids,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "entity_index": entity_index,
        "symmetry_index": symmetry_index,
        "msa_tokens": tokens,
        "template_residue_type_ids": template_values["template_residue_type_ids"],
    }
    placed_integral = {name: value.to(device=device).unsqueeze(0) for name, value in integral.items()}
    return AlphaFold2MultimerInput(  # type: ignore[call-arg]
        system=system,
        component_index=component_index.to(device=device).unsqueeze(0),
        atom37_index=_atom37_indices(system, component_index).to(device=device).unsqueeze(0),
        **placed,
        **placed_integral,
        batch_size=[],
    )


def materialize_collection(
    msa: MsaCollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    chains: tuple[MultimerChain, ...],
    total_length: int,
    query_tokens: Tensor,
    *,
    column_index: Sequence[int] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Expand the paired MsaCollection row plan onto the concatenated token axis.

    Each plan row is materialized directly on ``column_index``, or on the complete
    target axis when no selection is supplied. Every chain's selected columns
    receive the symbols from its paired or unpaired pool; gap cells remain
    ``MSA_GAP_INDEX``.
    Row 0 is the concatenated query. Paired and unpaired rows retain the explicit
    order encoded by the collection's row plan.
    """

    if msa is None:
        tokens = query_tokens.unsqueeze(0)
        deletions = torch.zeros_like(tokens, dtype=torch.float32)
        return tokens, deletions

    columns_for_chain = _columns_for_chain(msa_bindings, chains)
    selected_columns = range(total_length) if column_index is None else column_index
    selected_by_chain = {
        chain.instance_index: tuple(
            (output_column, target_column - chain.start)
            for output_column, target_column in enumerate(selected_columns)
            if chain.start <= target_column < chain.stop
        )
        for chain in chains
    }
    row_count = msa.row_pairing.shape[0]
    token_rows: list[list[int]] = []
    deletion_rows: list[list[float]] = []
    for row in range(row_count):
        token_row = [MSA_GAP_INDEX] * query_tokens.numel()
        deletion_row = [0.0] * query_tokens.numel()
        for chain in chains:
            column = columns_for_chain.get(chain.instance_index)
            if column is None:
                continue
            source = int(msa.row_pairing[row, column])
            if source == -1:
                continue
            paired = bool(msa.row_is_paired[row, column])
            entry = msa.msas[column]
            sequence = entry.paired_sequences[source] if paired else entry.unpaired_sequences[source]
            counts = entry.paired_deletion_counts[source] if paired else entry.unpaired_deletion_counts[source]
            for output_column, chain_column in selected_by_chain[chain.instance_index]:
                token_row[output_column] = msa_residue_index(sequence[chain_column], "protein")
                deletion_row[output_column] = float(counts[chain_column])
        token_rows.append(token_row)
        deletion_rows.append(deletion_row)
    tokens = torch.tensor(token_rows, dtype=torch.long)
    deletions = torch.tensor(deletion_rows, dtype=torch.float32)
    tokens[0] = query_tokens
    return tokens, deletions


def _columns_for_chain(
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    chains: tuple[MultimerChain, ...],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for column, (_msa, target_instance_index, entity_index) in enumerate(msa_bindings):
        for chain in chains:
            if target_instance_index is None:
                if chain.system_entity_index == entity_index:
                    mapping[chain.instance_index] = column
            elif chain.instance_index == target_instance_index:
                mapping[chain.instance_index] = column
    return mapping


def multimer_template_features(
    templates: tuple[Template, ...],
    *,
    system: MolecularSystem,
    chains: tuple[MultimerChain, ...],
    total_length: int,
    column_index: Sequence[int] | None,
) -> dict[str, Tensor]:
    r"""
    Bind templates on the complete target and project the selected columns.

    Produces atom37 positions/masks plus the multimer pseudo-beta and backbone-frame
    projections. These geometries are residue-local, so unlike monomer template
    torsions they do not require unselected neighbours.
    """

    atom_type_num = residue_constants.atom_type_num
    output_length = total_length if column_index is None else len(column_index)
    if not templates:
        return {
            "template_residue_type_ids": torch.full((1, output_length), GAP_RESIDUE_TYPE_INDEX, dtype=torch.long),
            "template_mask": torch.zeros(1),
            "template_all_atom_positions": torch.zeros(1, output_length, atom_type_num, 3),
            "template_all_atom_mask": torch.zeros(1, output_length, atom_type_num),
            "template_pseudo_beta": torch.zeros(1, output_length, 3),
            "template_pseudo_beta_mask": torch.zeros(1, output_length),
            "template_backbone_frame": torch.zeros(1, output_length, 4, 4),
            "template_backbone_frame_mask": torch.zeros(1, output_length),
        }

    instance_to_chain = {chain.instance_index: chain for chain in chains}
    entity_to_chains: dict[int, list[MultimerChain]] = {}
    for chain in chains:
        entity_to_chains.setdefault(chain.system_entity_index, []).append(chain)

    output_column = None if column_index is None else {target: output for output, target in enumerate(column_index)}
    template_residue_type_ids: list[Tensor] = []
    positions_list: list[Tensor] = []
    masks_list: list[Tensor] = []
    for template in templates:
        residue_type_ids = torch.full((output_length,), GAP_RESIDUE_TYPE_INDEX, dtype=torch.long)
        positions = torch.zeros(output_length, atom_type_num, 3)
        masks = torch.zeros(output_length, atom_type_num)
        projected_instances: set[int] = set()
        for alignment in template.alignments:
            try:
                target_entity_index = system.entity_index(alignment.target_entity_id)
                target_instance_index = (
                    None
                    if alignment.target_instance_id is None
                    else system.instance_index(alignment.target_instance_id)
                )
            except KeyError:
                continue
            target_chains = (
                entity_to_chains.get(target_entity_index, [])
                if target_instance_index is None
                else [instance_to_chain[target_instance_index]] if target_instance_index in instance_to_chain else []
            )
            for chain in target_chains:
                if chain.instance_index in projected_instances:
                    continue
                selected_alignment = select_template_alignment(
                    template,
                    system=system,
                    target_length=chain.stop - chain.start,
                    target_entity_index=chain.system_entity_index,
                    target_instance_index=chain.instance_index,
                )
                _project_template_chain(
                    template,
                    selected_alignment,
                    chain,
                    residue_type_ids,
                    positions,
                    masks,
                    output_column=output_column,
                )
                projected_instances.add(chain.instance_index)
        template_residue_type_ids.append(residue_type_ids)
        positions_list.append(positions)
        masks_list.append(masks)

    residue_type_ids = torch.stack(template_residue_type_ids)
    positions = torch.stack(positions_list)
    masks = torch.stack(masks_list)
    pseudo_beta, pseudo_beta_mask = _template_pseudo_beta(residue_type_ids, positions, masks)
    backbone_frame, backbone_frame_mask = _template_backbone_frame(positions, masks)
    return {
        "template_residue_type_ids": residue_type_ids,
        "template_mask": torch.ones(len(templates)),
        "template_all_atom_positions": positions,
        "template_all_atom_mask": masks,
        "template_pseudo_beta": pseudo_beta,
        "template_pseudo_beta_mask": pseudo_beta_mask,
        "template_backbone_frame": backbone_frame,
        "template_backbone_frame_mask": backbone_frame_mask,
    }


def _project_template_chain(
    template: Template,
    alignment: TemplateAlignment,
    chain: MultimerChain,
    residue_type_ids: Tensor,
    positions: Tensor,
    masks: Tensor,
    *,
    output_column: dict[int, int] | None,
) -> None:
    structure = template.structure
    system = structure.system
    template_instance_index = system.instance_index(alignment.template_instance_id)
    template_entity_index = int(system.instance_entity[template_instance_index])
    if int(system.entity_type[template_entity_index]) != ENTITY_PROTEIN:
        raise ValueError("AlphaFold-Multimer templates must align a protein instance")
    component_start = int(system.instance_component_ptr[template_instance_index])
    for target_index, template_index in alignment.residue_pairs:
        target_column = chain.start + target_index
        if output_column is not None:
            selected_column = output_column.get(target_column)
            if selected_column is None:
                continue
            target_column = selected_column
        component = component_start + template_index
        component_id = system.symbols[int(system.component_symbol[component])]
        residue_type_ids[target_column] = RESIDUE_TYPE_INDEX_BY_COMPONENT.get(
            component_id,
            UNKNOWN_RESIDUE_TYPE_INDEX,
        )
        atom_start = int(system.component_atom_ptr[component])
        atom_stop = int(system.component_atom_ptr[component + 1])
        for atom in range(atom_start, atom_stop):
            atom_index = residue_constants.atom_order.get(system.symbols[int(system.atom_symbol[atom])])
            if atom_index is None or not bool(structure.coordinate_mask[atom]):
                continue
            positions[target_column, atom_index] = (
                structure.coordinates[atom]
                .detach()
                .to(
                    device=positions.device,
                    dtype=positions.dtype,
                )
            )
            masks[target_column, atom_index] = 1.0


def _template_pseudo_beta(residue_type_ids: Tensor, positions: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    ca = residue_constants.atom_order["CA"]
    cb = residue_constants.atom_order["CB"]
    gly = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("G")
    atom_index = torch.where(
        residue_type_ids == gly,
        torch.full_like(residue_type_ids, ca),
        torch.full_like(residue_type_ids, cb),
    )
    gather = atom_index.unsqueeze(-1).unsqueeze(-1).expand(*atom_index.shape, 1, 3)
    pseudo_beta = positions.gather(-2, gather).squeeze(-2)
    mask_gather = atom_index.unsqueeze(-1).expand(*atom_index.shape, 1)
    pseudo_beta_mask = masks.gather(-1, mask_gather).squeeze(-1)
    return pseudo_beta, pseudo_beta_mask


def _template_backbone_frame(positions: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Construct the N/CA/C backbone rigid with Gram-Schmidt orthogonalization.

    The rotation is formed from ``C - CA`` and ``N - CA`` with translation ``CA``.
    ``from_two_vectors(e0_raw, e1_raw)`` sets the x-axis ``e0 = normalize(C - CA)``, makes
    ``e1 = normalize((N - CA) - (e1_raw·e0) e0)`` orthogonal to it, and ``e2 = e0 × e1``.
    The 3x3 block stores these as columns (``R[:, 0] = e0``), so the consumer's
    ``R_iᵀ (t_j - t_i)`` reproduces ``rigid.inverse().apply_to_point`` exactly.
    """

    n = residue_constants.atom_order["N"]
    ca = residue_constants.atom_order["CA"]
    c = residue_constants.atom_order["C"]
    backbone_mask = masks[..., n] * masks[..., ca] * masks[..., c]

    ca_xyz = positions[..., ca, :]
    e0_raw = positions[..., c, :] - ca_xyz
    e1_raw = positions[..., n, :] - ca_xyz

    e0 = e0_raw / e0_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (e1_raw * e0).sum(dim=-1, keepdim=True)
    e1 = e1_raw - dot * e0
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e2 = torch.linalg.cross(e0, e1, dim=-1)

    frame = torch.zeros(*positions.shape[:-2], 4, 4)
    frame[..., :3, 0] = e0
    frame[..., :3, 1] = e1
    frame[..., :3, 2] = e2
    frame[..., :3, 3] = ca_xyz
    frame[..., 3, 3] = 1.0
    return frame, backbone_mask
