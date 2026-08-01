r"""
Convert biomolecular inputs into AlphaFold 3 features.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.ccd import resolve_ccd_component
from ...data.components import complete_component
from ...data.crop import Cropper, crop_token_layout, resolved_token_mask
from ...data.evidence import (
    MSA,
    MSA_VOCAB_SIZE,
    MsaCollection,
    Template,
    TemplateAlignment,
    msa_residue_index,
    resolve_msa_bindings,
)
from ...data.molecular import (
    BOND_ROLE_ASSEMBLY,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    STANDARD_AMINO_ACID_IDS,
    ChemicalComponent,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...data.tokenization import (
    TOKEN_GAP_INDEX,
    TOKEN_VOCAB_SIZE,
    TokenLayout,
    component_token_index,
    tokenize_molecular_system,
)
from ...processor import Processor
from .config import AlphaFold3ProcessorConfig
from .model import AlphaFold3Input, AlphaFold3StructureTarget, TemplateFeatures

_STANDARD = {
    "protein": STANDARD_AMINO_ACID_IDS | {"UNK"},
    "rna": frozenset(("A", "G", "C", "U", "N")),
    "dna": frozenset(("DA", "DG", "DC", "DT", "DN")),
}
_STANDARD_BY_ENTITY_TYPE = {
    ENTITY_PROTEIN: _STANDARD["protein"],
    ENTITY_RNA: _STANDARD["rna"],
    ENTITY_DNA: _STANDARD["dna"],
    ENTITY_DNA_RNA_HYBRID: _STANDARD["rna"] | _STANDARD["dna"],
    ENTITY_PNA: frozenset(),
}
_POLYMER_TYPE_BY_ENTITY_TYPE = {
    ENTITY_PROTEIN: "protein",
    ENTITY_RNA: "rna",
    ENTITY_DNA: "dna",
    ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
}


# AlphaFold 3 template distogram bins: 39 lower breaks over [3.25, 50.75] (squared).
_TEMPLATE_DISTOGRAM_MIN = 3.25
_TEMPLATE_DISTOGRAM_MAX = 50.75


class AlphaFold3Processor(Processor):
    r"""
    Convert supported biomolecular inputs into `AlphaFold3Input`.

    Args:
        config: Feature limits and CCD resolution settings. Defaults to
            ``AlphaFold3ProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.

    Examples:
        >>> from ofoldx.data import MolecularSystem, Molecule, PolymerSequence, get_component
        >>> molecule = Molecule((get_component("GLY"), get_component("ALA")))
        >>> system = MolecularSystem.from_entities({"ligand": molecule}, {"L": "ligand"})
        >>> input = AlphaFold3Processor()(system)
        >>> input.residue_index.unique().tolist()
        [0, 1]
        >>> input.reference_space_id.unique().tolist()
        [0, 1]
        >>> bool((input.token_atom_index >= 0).all())
        True
        >>> protein = AlphaFold3Processor()(PolymerSequence.from_one_letter("GA"))
        >>> protein.token_atom_index.tolist()
        [[-1, -1]]
        >>> protein.token_features.shape[-1]
        461
        >>> protein.token_features[0, :, 384:422].argmax(dim=-1).tolist()
        [5, 0]
        >>> protein.msa_tokens.shape
        torch.Size([1, 1, 2])
        >>> protein.msa_deletion_counts.dtype
        torch.int32
    """

    config_class = AlphaFold3ProcessorConfig
    config: AlphaFold3ProcessorConfig

    def __init__(
        self,
        config: AlphaFold3ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold3ProcessorConfig() if config is None else config
        if not 1 <= self.config.max_msa_rows <= 16384:
            raise ValueError("max_msa_rows must be between 1 and 16384")
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
    ) -> AlphaFold3Input:
        r"""
        Prepare model features for one molecular assembly.

        Args:
            input: One or more ``PolymerSequence`` values, ``MolecularSystem``,
                or ``Structure``. A structure contributes topology, not observed
                coordinates.
            msa: Optional ``MSA`` or ``MsaCollection``.
            templates: Template evidence aligned to the target polymers.

        Returns:
            Batched token, atom, MSA, and template tensors. Reference and
            template coordinates are in angstroms. ``atom_index`` and the two
            token identity tensors map family-local axes to the canonical graph.
            This method performs no random sampling.
        """

        return self._lower(input, msa=msa, templates=tuple(templates))

    def _lower(
        self,
        input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | None,
        templates: tuple[Template, ...],
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> AlphaFold3Input:
        system = molecular_system_from_input(input, revision=self.config.ccd_revision)
        resolved_msa = () if msa is None else resolve_msa_bindings(msa, system)
        for column, (_value, _target_instance_index, entity_index) in enumerate(resolved_msa):
            entity_type = int(system.entity_type[entity_index])
            if entity_type not in {ENTITY_PROTEIN, ENTITY_RNA}:
                raise ValueError("AlphaFold 3 consumes supplied MSA evidence only for proteins and unpaired RNA")
            if (
                entity_type == ENTITY_RNA
                and isinstance(msa, MsaCollection)
                and bool(((msa.row_pairing[1:, column] >= 0) & msa.row_is_paired[1:, column]).any())
            ):
                raise ValueError("AlphaFold 3 accepts RNA only through unpaired MSA rows")
        layout = _tokenize(system)
        if cropper is not None:
            layout = crop_token_layout(
                cast(Structure, input),
                layout,
                cropper,
                anchor_token_mask=resolved_token_mask(cast(Structure, input), layout),
                generator=generator,
            )
        tokens = _lower_token_layout(system, layout)
        return _features(
            system,
            tokens=tokens,
            msa=msa,
            resolved_msa=resolved_msa,
            templates=templates,
            max_msa_rows=self.config.max_msa_rows,
            device=self.device,
            dtype=self.dtype,
        )

    def process_training(
        self,
        input: Structure,
        *,
        msa: MSA | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[AlphaFold3Input, AlphaFold3StructureTarget]:
        r"""
        Lower one structure into model input and separate coordinate truth.
        """

        processed = self._lower(
            input,
            msa=msa,
            templates=tuple(templates),
            cropper=cropper,
            generator=generator,
        )
        return processed, _structure_target(
            input,
            device=self.device,
            dtype=self.dtype,
        )


def _structure_target(
    structure: Structure,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold3StructureTarget:
    r"""
    Lower full-axis structure coordinates without introducing another atom layout.
    """

    return AlphaFold3StructureTarget(  # type: ignore[call-arg]
        coordinates=structure.coordinates.to(device=device, dtype=dtype).unsqueeze(0),
        coordinate_mask=structure.coordinate_mask.to(device=device).unsqueeze(0),
        batch_size=[1],
    )


def molecular_system_from_input(
    input: PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
    *,
    revision: str,
) -> MolecularSystem:
    r"""
    Normalize one AlphaFold 3 biological input to its molecular system.

    Args:
        input: One polymer, a polymer collection, a molecular system, or a
            structure whose topology supplies the system.
        revision: CCD snapshot used to resolve bare sequence construction values.

    Returns:
        The supplied system, or one constructed from bare polymer sequences.

    Examples:
        Existing molecular systems are already prepared inputs and are returned
        unchanged rather than resolved against a CCD snapshot:

        >>> from ofoldx.data import MolecularSystem, Structure
        >>> system = MolecularSystem.from_entities({}, {})
        >>> molecular_system_from_input(system, revision="unused") is system
        True
        >>> structure = Structure.from_sequence("A")
        >>> molecular_system_from_input(structure, revision="unused") is structure.system
        True
    """

    if isinstance(input, Structure):
        return input.system
    if isinstance(input, MolecularSystem):
        return input
    if isinstance(input, PolymerSequence):
        return molecular_system_from_named_sequences(
            (("A", input),),
            revision=revision,
        )
    if isinstance(input, Sequence):
        values = tuple(input)
        if not values or not all(isinstance(sequence, PolymerSequence) for sequence in values):
            raise TypeError("AlphaFold 3 sequence collections must contain PolymerSequence values")
        polymer_sequences = tuple(cast(PolymerSequence, sequence) for sequence in values)
        return molecular_system_from_named_sequences(
            tuple((f"chain_{index + 1}", sequence) for index, sequence in enumerate(polymer_sequences)),
            revision=revision,
        )
    raise TypeError(
        "AlphaFold 3 input must be a PolymerSequence, sequence of PolymerSequence "
        "values, MolecularSystem, or Structure"
    )


def molecular_system_from_named_sequences(
    named: tuple[tuple[str, PolymerSequence], ...],
    *,
    revision: str,
) -> MolecularSystem:
    r"""
    Construct an AlphaFold 3 molecular system from named polymer instances.

    Repeated names receive deterministic numeric suffixes while preserving
    declaration order.

    Args:
        named: Ordered ``(base_id, sequence)`` pairs.
        revision: CCD snapshot used to resolve bare component construction values
            before materialization.

    Returns:
        A molecular system with one entity and instance per sequence.
    """

    used: set[str] = set()
    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    cache: dict[str, ChemicalComponent] = {}
    topology_complete: list[bool] = []
    for base_id, sequence in named:
        entity_id = base_id
        suffix = 1
        while entity_id in used:
            suffix += 1
            entity_id = f"{base_id}_{suffix}"
        used.add(entity_id)
        components: list[ChemicalComponent] = []
        changed = False
        for component in sequence.components:
            if not component.atoms or not any(coordinate is not None for coordinate in component.reference_coordinates):
                resolved = cache.get(component.component_id)
                if resolved is None:
                    resolved = resolve_ccd_component(
                        component.component_id,
                        revision=revision,
                    )
                    cache[component.component_id] = resolved
                completed, complete = complete_component(
                    component,
                    resolved,
                    occurrence_topology_complete=component.bonds is not None,
                )
                components.append(completed)
                topology_complete.append(complete)
                changed = True
            else:
                components.append(component)
                topology_complete.append(component.bonds is not None)
        entities[entity_id] = (
            PolymerSequence(sequence.polymer_type, tuple(components), sequence.linked) if changed else sequence
        )
        instances[entity_id] = entity_id
    system = MolecularSystem.from_entities(entities, instances)
    return system.replace(  # type: ignore[attr-defined]
        component_topology_complete=system.component_topology_complete.new_tensor(topology_complete),
    )


def _is_standard(system: MolecularSystem, component_index: int, entity_type: int) -> bool:
    component_id = system.symbols[int(system.component_symbol[component_index])]
    return component_id in _STANDARD_BY_ENTITY_TYPE.get(entity_type, frozenset())


def _tokenize(system: MolecularSystem) -> TokenLayout:
    atom_mask = _af3_visible_atom_mask(system)
    return tokenize_molecular_system(system, atom_mask=atom_mask)


def _af3_visible_atom_mask(system: MolecularSystem) -> Tensor:
    r"""Drop the placeholder nitrogen carried by unknown nucleic components."""

    visible = torch.ones_like(system.atomic_number, dtype=torch.bool)
    component_atom_ptr = system.component_atom_ptr.detach().cpu().tolist()
    component_symbol = system.component_symbol.detach().cpu().tolist()
    atom_symbol = system.atom_symbol.detach().cpu().tolist()
    for component_index, component in enumerate(component_symbol):
        if system.symbols[int(component)] not in {"N", "DN"}:
            continue
        for atom_index in range(component_atom_ptr[component_index], component_atom_ptr[component_index + 1]):
            if system.symbols[int(atom_symbol[atom_index])] == "N":
                visible[atom_index] = False
    return visible


def _lower_token_layout(system: MolecularSystem, layout: TokenLayout) -> list[dict[str, object]]:
    r"""Lower the shared structural partition into AlphaFold 3 feature rows."""

    token_atom_ptr = layout.token_atom_ptr.detach().cpu().tolist()
    atom_index = layout.atom_index.detach().cpu().tolist()
    token_component_index = layout.token_component_index.detach().cpu().tolist()
    token_atom_index = layout.token_atom_index.detach().cpu().tolist()
    token_type_index = layout.token_type_index.detach().cpu().tolist()
    token_instance_index = layout.token_instance_index.detach().cpu().tolist()
    token_entity_index = layout.token_entity_index.detach().cpu().tolist()
    token_residue_index = layout.token_residue_index.detach().cpu().tolist()
    token_symmetry_index = layout.token_symmetry_index.detach().cpu().tolist()
    entity_type = system.entity_type.detach().cpu().tolist()
    component_symbol = system.component_symbol.detach().cpu().tolist()
    atom_symbol = system.atom_symbol.detach().cpu().tolist()
    tokens: list[dict[str, object]] = []

    for token_index, component_index in enumerate(token_component_index):
        group = tuple(atom_index[token_atom_ptr[token_index] : token_atom_ptr[token_index + 1]])
        instance_index = int(token_instance_index[token_index])
        entity_index = int(token_entity_index[token_index])
        current_entity_type = int(entity_type[entity_index])
        local_component = int(token_residue_index[token_index])
        component_id = system.symbols[int(component_symbol[component_index])]
        canonical_atom = int(token_atom_index[token_index])
        atom_token = canonical_atom >= 0
        standard = _is_standard(system, component_index, current_entity_type)
        if atom_token:
            center = representative = canonical_atom
        else:
            by_name = {system.symbols[int(atom_symbol[atom])]: atom for atom in group}
            center_name = "CA" if current_entity_type == ENTITY_PROTEIN else "C1'"
            center = by_name[center_name]
            if current_entity_type == ENTITY_PROTEIN:
                representative_name = "CA" if component_id == "GLY" else "CB"
            elif component_id in {"A", "G", "DA", "DG"}:
                representative_name = "C4"
            elif component_id in {"C", "U", "DC", "DT"}:
                representative_name = "C2"
            else:
                representative_name = "C1'"
            representative = by_name.get(representative_name, center)
        tokens.append(
            {
                "atoms": group,
                "component_index": component_index,
                "token_atom_index": canonical_atom,
                "instance_index": instance_index,
                "entity_index": entity_index,
                "monomer_index": None if current_entity_type == ENTITY_MOLECULE else local_component,
                "residue_index": local_component,
                "symmetry_index": int(token_symmetry_index[token_index]),
                "token_type_index": int(token_type_index[token_index]),
                "center": center,
                "representative": representative,
                "standard": standard,
            }
        )
    return tokens


def _features(
    system: MolecularSystem,
    *,
    tokens: list[dict[str, object]],
    msa: MSA | MsaCollection | None,
    resolved_msa: tuple[tuple[MSA, int | None, int], ...],
    templates: tuple[Template, ...],
    max_msa_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold3Input:
    atom_indices = tuple(atom_index for token in tokens for atom_index in cast(tuple[int, ...], token["atoms"]))
    canonical_to_model_atom = {atom_index: model_atom_index for model_atom_index, atom_index in enumerate(atom_indices)}
    atom_to_token_by_index = {
        atom_index: token_index
        for token_index, token in enumerate(tokens)
        for atom_index in cast(tuple[int, ...], token["atoms"])
    }
    atom_to_token = torch.tensor(
        [atom_to_token_by_index[atom_index] for atom_index in atom_indices],
        dtype=torch.long,
    )
    atom_component_index = torch.tensor(
        [cast(int, tokens[atom_to_token_by_index[atom_index]]["component_index"]) for atom_index in atom_indices],
        dtype=torch.long,
    )
    canonical_atom_index = torch.tensor(atom_indices, dtype=torch.long)
    reference_positions, reference_mask, reference_charges = _reference_features(
        system,
        canonical_atom_index,
        atom_component_index,
    )
    atom_names = tuple(system.symbols[int(system.atom_symbol[atom_index])] for atom_index in atom_indices)
    source_atom_index = canonical_atom_index.to(device=system.atomic_number.device)
    element_indices = system.atomic_number[source_atom_index].detach().cpu().long()
    name_indices = torch.tensor(
        [[min(63, max(0, ord(character) - 32)) for character in name.ljust(4)[:4]] for name in atom_names],
        dtype=torch.long,
    )
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            F.one_hot(element_indices, num_classes=128).float(),
            F.one_hot(name_indices, num_classes=64).reshape(len(atom_indices), 256).float(),
        ),
        dim=-1,
    )
    atom_token_index = _atom_token_offsets(
        tokens,
        system,
        canonical_to_model_atom,
        len(atom_indices),
    )

    profile, deletion_mean = _profile_features(resolved_msa, tokens, system)
    token_type = torch.tensor([cast(int, token["token_type_index"]) for token in tokens])
    token_features = torch.cat(
        (
            torch.zeros(len(tokens), 384),
            F.one_hot(token_type, num_classes=TOKEN_VOCAB_SIZE).float(),
            profile,
            deletion_mean[:, None],
        ),
        dim=-1,
    )

    msa_tokens, msa_deletion_counts = _msa_rows_and_deletions(
        msa,
        resolved_msa,
        tokens,
        system,
        max_rows=max_msa_rows,
    )
    msa_mask = torch.ones_like(msa_tokens, dtype=torch.bool)
    template_features = _template_features(templates, tokens, system)
    token_bonds = _token_bonds(system, atom_to_token_by_index, len(tokens))
    token_to_rep_atom = torch.tensor(
        [canonical_to_model_atom[cast(int, token["representative"])] for token in tokens],
        dtype=torch.long,
    )

    def floating(value: Tensor) -> Tensor:
        return value.to(device=device, dtype=dtype).unsqueeze(0)

    def discrete(value: Tensor) -> Tensor:
        return value.to(device=device).unsqueeze(0)

    return AlphaFold3Input(  # type: ignore[call-arg]
        token_mask=discrete(torch.ones(len(tokens), dtype=torch.bool)),
        residue_index=discrete(torch.tensor([cast(int, token["residue_index"]) for token in tokens])),
        chain_index=discrete(torch.tensor([cast(int, token["instance_index"]) for token in tokens])),
        entity_index=discrete(torch.tensor([cast(int, token["entity_index"]) for token in tokens])),
        symmetry_index=discrete(torch.tensor([cast(int, token["symmetry_index"]) for token in tokens])),
        token_index=discrete(torch.arange(len(tokens))),
        atom_features=floating(atom_features),
        atom_to_token=discrete(atom_to_token),
        atom_mask=discrete(torch.ones(len(atom_indices), dtype=torch.bool)),
        reference_positions=floating(reference_positions),
        reference_space_id=discrete(atom_component_index),
        atom_token_index=discrete(atom_token_index),
        token_to_rep_atom=discrete(token_to_rep_atom),
        token_features=floating(token_features),
        reference_mask=discrete(reference_mask),
        reference_charges=floating(reference_charges[:, None]),
        token_bonds=floating(token_bonds),
        msa_tokens=discrete(msa_tokens),
        msa_deletion_counts=discrete(msa_deletion_counts),
        msa_mask=discrete(msa_mask),
        template_features=(
            None if template_features is None else _template_to(template_features, device=device, dtype=dtype)
        ),
        system=system,
        atom_index=discrete(canonical_atom_index),
        token_component_index=discrete(
            torch.tensor(
                [cast(int, token["component_index"]) for token in tokens],
                dtype=torch.long,
            )
        ),
        token_atom_index=discrete(
            torch.tensor(
                [cast(int, token["token_atom_index"]) for token in tokens],
                dtype=torch.long,
            )
        ),
        batch_size=[],
    )


def _atom_token_offsets(
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    canonical_to_model_atom: dict[int, int],
    atom_count: int,
) -> Tensor:
    atom_token_index = torch.empty(atom_count, dtype=torch.long)
    component_atom_ptr = system.component_atom_ptr.detach().cpu().tolist()
    atom_symbols = system.atom_symbol.detach().cpu().tolist()
    atomic_numbers = system.atomic_number.detach().cpu().tolist()
    for token in tokens:
        group = cast(tuple[int, ...], token["atoms"])
        if not cast(bool, token["standard"]):
            for atom_index in group:
                atom_token_index[canonical_to_model_atom[atom_index]] = 0
            continue
        component_index = cast(int, token["component_index"])
        component_id = system.symbols[int(system.component_symbol[component_index])]
        if component_id == "N":
            heavy_order = {
                name: index
                for index, name in enumerate(
                    ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
                )
            }
        elif component_id == "DN":
            heavy_order = {
                name: index
                for index, name in enumerate(
                    ("OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
                )
            }
        else:
            heavy_order = {
                system.symbols[int(atom_symbols[atom_index])]: index
                for index, atom_index in enumerate(
                    atom_index
                    for atom_index in range(
                        int(component_atom_ptr[component_index]),
                        int(component_atom_ptr[component_index + 1]),
                    )
                    if int(atomic_numbers[atom_index]) != 1
                )
            }
        for atom_index in group:
            atom_name = system.symbols[int(atom_symbols[atom_index])]
            atom_token_index[canonical_to_model_atom[atom_index]] = heavy_order[atom_name]
    return atom_token_index


def _reference_features(
    system: MolecularSystem,
    atom_index: Tensor,
    component_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    source_index = atom_index.to(device=system.reference_coordinates.device)
    positions = system.reference_coordinates[source_index].detach().cpu().float()
    mask = system.reference_coordinate_mask[source_index].detach().cpu()
    charges = system.formal_charge[source_index].detach().cpu().float()
    for current_component in component_index.unique(sorted=True).tolist():
        indices = (component_index == current_component).nonzero(as_tuple=False).squeeze(-1)
        if not bool(mask[indices].any()):
            component_id = system.symbols[int(system.component_symbol[current_component])]
            raise ValueError(
                f"AlphaFold 3 requires explicit reference_coordinates for {component_id!r}; "
                "the Processor does not generate conformers implicitly"
            )
        positions[indices] -= positions[indices].mean(dim=0)
    return positions, mask, charges


def _msa_descriptors(
    msa: MSA | MsaCollection | None,
    resolved_msa: tuple[tuple[MSA, int | None, int], ...],
    *,
    max_rows: int,
) -> list[tuple[dict[tuple[int, int | None], tuple[int, bool]], bool]]:
    # Each descriptor maps an entity-broadcast or exact-instance column binding to
    # ``(source_row, is_paired)``. The bool is the query flag.
    if msa is None:
        return [({}, True)]
    if isinstance(msa, MSA):
        _value, target_instance_index, entity_index = resolved_msa[0]
        key = (entity_index, target_instance_index)
        return [({key: (row, False)}, row == 0) for row in range(min(len(msa.unpaired_sequences), max_rows))]

    row_pairing = msa.row_pairing
    row_is_paired = msa.row_is_paired
    plan_rows = int(row_pairing.shape[0])

    paired_cells = np.count_nonzero((row_pairing != -1) & row_is_paired, axis=1)
    paired = [row for row in range(1, plan_rows) if paired_cells[row] >= 2]
    paired_rows = set(paired)
    unpaired = [row for row in range(1, plan_rows) if row not in paired_rows]

    pair_limit = max_rows // 2
    ordered: list[int | None] = [None]
    ordered.extend(paired[: max(0, pair_limit - 1)])
    ordered.extend(unpaired[: max_rows - len(ordered)])
    ordered = ordered[:max_rows]

    descriptors: list[tuple[dict[tuple[int, int | None], tuple[int, bool]], bool]] = []
    for row in ordered:
        if row is None:
            descriptors.append(({}, True))
            continue
        mapping: dict[tuple[int, int | None], tuple[int, bool]] = {}
        for column, (_value, target_instance_index, entity_index) in enumerate(resolved_msa):
            source = int(row_pairing[row, column])
            if source == -1:
                continue
            mapping[(entity_index, target_instance_index)] = (source, bool(row_is_paired[row, column]))
        descriptors.append((mapping, False))
    return descriptors


def _msa_rows_and_deletions(
    msa: MSA | MsaCollection | None,
    resolved_msa: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
    *,
    max_rows: int,
) -> tuple[Tensor, Tensor]:
    bindings = {
        (entity_index, target_instance_index): value for value, target_instance_index, entity_index in resolved_msa
    }
    descriptors = _msa_descriptors(msa, resolved_msa, max_rows=max_rows)
    rows = torch.empty(len(descriptors), len(tokens), dtype=torch.long)
    deletions = torch.zeros(len(descriptors), len(tokens), dtype=torch.int32)

    for row_index, (descriptor, query) in enumerate(descriptors):
        for token_index, token in enumerate(tokens):
            entity_index = cast(int, token["entity_index"])
            instance_index = cast(int, token["instance_index"])
            monomer_index = cast(int | None, token["monomer_index"])
            key: tuple[int, int | None] = (entity_index, instance_index)
            if key not in bindings:
                key = (entity_index, None)
            selected = bindings.get(key)
            current_entity_type = int(system.entity_type[entity_index])
            if query:
                rows[row_index, token_index] = cast(int, token["token_type_index"])
            elif selected is None or current_entity_type not in {ENTITY_PROTEIN, ENTITY_RNA} or key not in descriptor:
                rows[row_index, token_index] = msa_residue_index("-", "protein")
            else:
                source_row, is_paired = descriptor[key]
                pool = selected.paired_sequences if is_paired else selected.unpaired_sequences
                deletion_pool = selected.paired_deletion_counts if is_paired else selected.unpaired_deletion_counts
                symbol = pool[source_row][cast(int, monomer_index)]
                polymer_type = "protein" if current_entity_type == ENTITY_PROTEIN else "rna"
                rows[row_index, token_index] = msa_residue_index(symbol, polymer_type)
                deletions[row_index, token_index] = int(deletion_pool[source_row, cast(int, monomer_index)])
    return rows, deletions


def _profile_features(
    resolved_msa: tuple[tuple[MSA, int | None, int], ...],
    tokens: list[dict[str, object]],
    system: MolecularSystem,
) -> tuple[Tensor, Tensor]:
    r"""
    The canonical 38-class MSA profile and per-token mean deletion over the unpaired pool.
    """

    bindings = {
        (entity_index, target_instance_index): value for value, target_instance_index, entity_index in resolved_msa
    }
    profile = torch.zeros(len(tokens), MSA_VOCAB_SIZE, dtype=torch.float32)
    deletion_mean = torch.zeros(len(tokens), dtype=torch.float32)
    for token_index, token in enumerate(tokens):
        entity_index = cast(int, token["entity_index"])
        instance_index = cast(int, token["instance_index"])
        monomer_index = cast(int | None, token["monomer_index"])
        key: tuple[int, int | None] = (entity_index, instance_index)
        if key not in bindings:
            key = (entity_index, None)
        selected = bindings.get(key)
        current_entity_type = int(system.entity_type[entity_index])
        if selected is None or current_entity_type not in {ENTITY_PROTEIN, ENTITY_RNA}:
            profile[token_index, cast(int, token["token_type_index"])] = 1
            continue
        column = cast(int, monomer_index)
        polymer_type = "protein" if current_entity_type == ENTITY_PROTEIN else "rna"
        encoded = [msa_residue_index(sequence[column], polymer_type) for sequence in selected.unpaired_sequences]
        profile[token_index] = F.one_hot(torch.tensor(encoded), num_classes=MSA_VOCAB_SIZE).float().mean(dim=0)
        deletion_mean[token_index] = torch.tensor(
            selected.unpaired_deletion_counts[:, column],
            dtype=torch.float32,
        ).mean()
    return profile, deletion_mean


def _token_bonds(
    system: MolecularSystem,
    atom_to_token: dict[int, int],
    token_count: int,
) -> Tensor:
    feature = torch.zeros(token_count, token_count)
    component_atom_ptr = system.component_atom_ptr.detach().cpu()
    instance_component_ptr = system.instance_component_ptr.detach().cpu()
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel()),
        component_atom_ptr[1:] - component_atom_ptr[:-1],
    )
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel()),
        instance_component_ptr[1:] - instance_component_ptr[:-1],
    )
    instance_entity = system.instance_entity.detach().cpu()
    entity_type = system.entity_type.detach().cpu()
    edge_index = system.edge_index.detach().cpu()
    bond_role = system.bond_role.detach().cpu()
    for edge in range(edge_index.shape[1]):
        first = int(edge_index[0, edge])
        second = int(edge_index[1, edge])
        if first not in atom_to_token or second not in atom_to_token:
            continue
        first_component = int(atom_component[first])
        second_component = int(atom_component[second])
        first_instance = int(component_instance[first_component])
        second_instance = int(component_instance[second_component])
        first_entity = int(instance_entity[first_instance])
        second_entity = int(instance_entity[second_instance])
        first_entity_type = int(entity_type[first_entity])
        second_entity_type = int(entity_type[second_entity])
        keep = first_entity_type == ENTITY_MOLECULE or second_entity_type == ENTITY_MOLECULE
        if first_entity_type != ENTITY_MOLECULE and second_entity_type != ENTITY_MOLECULE:
            if first_component == second_component:
                keep = not _is_standard(system, first_component, first_entity_type)
            elif int(bond_role[edge]) == BOND_ROLE_ASSEMBLY and (
                first_instance != second_instance or abs(first_component - second_component) > 1
            ):
                keep = True
            else:
                keep = False
        if keep:
            first_token, second_token = atom_to_token[first], atom_to_token[second]
            feature[first_token, second_token] = 1
    return feature


def _template_to(features: TemplateFeatures, *, device: torch.device, dtype: torch.dtype) -> TemplateFeatures:
    template_mask = features.template_mask
    return TemplateFeatures(
        distogram=features.distogram.to(device=device, dtype=dtype),
        pseudo_beta_mask=features.pseudo_beta_mask.to(device=device, dtype=dtype),
        residue_type_ids=features.residue_type_ids.to(device=device),
        unit_vector=features.unit_vector.to(device=device, dtype=dtype),
        backbone_frame_mask=features.backbone_frame_mask.to(device=device, dtype=dtype),
        **({"template_mask": template_mask.to(device=device, dtype=dtype)} if template_mask is not None else {}),
        batch_size=[features.distogram.shape[0]],
    )


def _template_backbone(positions: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Pseudo-beta positions/mask and per-pair unit vectors in each residue's backbone frame.

    ``positions`` / ``mask`` are ``[N, 4, *]`` over the dense backbone atoms (N, CA, C, CB); glycine
    carries no CB so its pseudo-beta falls back to CA (index 1). Each frame places the x-axis along
    ``C - CA`` with ``N - CA`` in the xy-plane.
    """

    nitrogen, alpha, carbon, beta = positions.unbind(dim=1)
    nitrogen_mask, alpha_mask, carbon_mask, beta_mask = mask.unbind(dim=1)

    is_glycine = beta_mask == 0
    pseudo_beta = torch.where(is_glycine[:, None], alpha, beta)
    pseudo_beta_mask = torch.where(is_glycine, alpha_mask, beta_mask)

    def _normalize(vector: Tensor) -> Tensor:
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    x_axis = _normalize(carbon - alpha)
    reference = nitrogen - alpha
    reference = reference - (reference * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = _normalize(reference)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)  # columns are the frame axes

    backbone_mask = nitrogen_mask * alpha_mask * carbon_mask
    difference = alpha[None, :, :] - alpha[:, None, :]  # CA_j - CA_i
    local = torch.einsum("inm,ijn->ijm", rotation, difference)  # R_i^T (CA_j - CA_i)
    unit_vector = local / local.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask


def _template_distogram(pseudo_beta: Tensor, pseudo_beta_mask_2d: Tensor, num_bins: int) -> Tensor:
    lower = torch.linspace(_TEMPLATE_DISTOGRAM_MIN, _TEMPLATE_DISTOGRAM_MAX, num_bins) ** 2
    upper = torch.cat((lower[1:], lower.new_tensor([1e8])))
    squared = ((pseudo_beta[:, None, :] - pseudo_beta[None, :, :]) ** 2).sum(dim=-1, keepdim=True)
    dgram = ((squared > lower) & (squared < upper)).float()
    return dgram * pseudo_beta_mask_2d[..., None]


def _template_features(
    templates: tuple[Template, ...],
    tokens: list[dict[str, object]],
    target_system: MolecularSystem,
    *,
    num_distogram_bins: int = 39,
) -> TemplateFeatures | None:
    if not templates:
        return None

    token_count = len(tokens)
    component_token: dict[int, int] = {}
    for token_index, token in enumerate(tokens):
        monomer_index = cast(int | None, token["monomer_index"])
        if monomer_index is None:
            continue
        component_token[cast(int, token["component_index"])] = token_index

    per_template_positions: list[Tensor] = []
    per_template_mask: list[Tensor] = []
    per_template_residue_type: list[Tensor] = []
    for template in templates:
        positions = torch.zeros(token_count, 4, 3)
        atom_mask = torch.zeros(token_count, 4)
        residue_type = torch.full((token_count,), TOKEN_GAP_INDEX, dtype=torch.long)
        template_system = template.structure.system
        for alignment in template.alignments:
            template_instance_index = template_system.instance_index(alignment.template_instance_id)
            template_entity_index = int(template_system.instance_entity[template_instance_index])
            template_entity_type = int(template_system.entity_type[template_entity_index])
            template_component_start = int(template_system.instance_component_ptr[template_instance_index])
            for target_component, template_component in alignment.residue_pairs:
                targets = _alignment_targets(
                    alignment,
                    target_component,
                    target_system,
                    component_token,
                )
                template_component_index = template_component_start + template_component
                backbone, backbone_mask = _residue_backbone(
                    template.structure,
                    template_component_index,
                )
                component_id = template_system.symbols[int(template_system.component_symbol[template_component_index])]
                residue_type_index = component_token_index(
                    component_id,
                    _POLYMER_TYPE_BY_ENTITY_TYPE.get(template_entity_type),
                )
                for target_token in targets:
                    positions[target_token] = backbone
                    atom_mask[target_token] = backbone_mask
                    residue_type[target_token] = residue_type_index
        per_template_positions.append(positions)
        per_template_mask.append(atom_mask)
        per_template_residue_type.append(residue_type)

    distograms: list[Tensor] = []
    pseudo_beta_masks: list[Tensor] = []
    unit_vectors: list[Tensor] = []
    backbone_masks: list[Tensor] = []
    for positions, atom_mask in zip(per_template_positions, per_template_mask):
        pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask = _template_backbone(positions, atom_mask)
        pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        distograms.append(_template_distogram(pseudo_beta, pseudo_beta_mask_2d, num_distogram_bins))
        pseudo_beta_masks.append(pseudo_beta_mask_2d)
        unit_vectors.append(unit_vector * backbone_mask_2d[..., None])
        backbone_masks.append(backbone_mask_2d)

    return TemplateFeatures(
        distogram=torch.stack(distograms).unsqueeze(0),
        pseudo_beta_mask=torch.stack(pseudo_beta_masks).unsqueeze(0),
        residue_type_ids=torch.stack(per_template_residue_type).unsqueeze(0),
        unit_vector=torch.stack(unit_vectors).unsqueeze(0),
        backbone_frame_mask=torch.stack(backbone_masks).unsqueeze(0),
        template_mask=torch.ones(1, len(templates)),
        batch_size=[1],
    )


def _alignment_targets(
    alignment: TemplateAlignment,
    target_component: int,
    system: MolecularSystem,
    component_token: dict[int, int],
) -> list[int]:
    try:
        target_entity_index = system.entity_index(alignment.target_entity_id)
    except KeyError:
        raise ValueError(f"template alignment targets unknown entity {alignment.target_entity_id!r}") from None
    if int(system.entity_type[target_entity_index]) == ENTITY_MOLECULE:
        raise ValueError(f"template alignment target {alignment.target_entity_id!r} is not a polymer entity")
    if alignment.target_instance_id is not None:
        try:
            target_instance_index = system.instance_index(alignment.target_instance_id)
        except KeyError:
            raise ValueError(f"template alignment targets unknown instance {alignment.target_instance_id!r}") from None
        actual_entity_index = int(system.instance_entity[target_instance_index])
        if actual_entity_index != target_entity_index:
            raise ValueError(
                f"template alignment instance {alignment.target_instance_id!r} belongs to entity "
                f"index {actual_entity_index}, not {alignment.target_entity_id!r}"
            )
        start = int(system.instance_component_ptr[target_instance_index])
        stop = int(system.instance_component_ptr[target_instance_index + 1])
        if target_component >= stop - start:
            raise ValueError(
                f"template alignment target {alignment.target_instance_id!r}:"
                f"{target_component} does not resolve to a component"
            )
        component_index = start + target_component
        token_index = component_token.get(component_index)
        if token_index is None:
            return []
        return [token_index]
    targets = []
    target_instances = (
        (system.instance_entity == target_entity_index).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    )
    if not target_instances:
        raise ValueError(
            f"template alignment target {alignment.target_entity_id!r}:{target_component} "
            "does not resolve to an instance"
        )
    for target_instance_index in target_instances:
        start = int(system.instance_component_ptr[target_instance_index])
        stop = int(system.instance_component_ptr[target_instance_index + 1])
        if target_component >= stop - start:
            raise ValueError(
                f"template alignment target {alignment.target_entity_id!r}:"
                f"{target_component} does not resolve to a component"
            )
        component_index = start + target_component
        if component_index in component_token:
            targets.append(component_token[component_index])
    return targets


def _residue_backbone(
    structure: Structure,
    component_index: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Return dense (N, CA, C, CB) positions and their validity mask for one template residue.
    """

    positions = torch.zeros(4, 3)
    mask = torch.zeros(4)
    system = structure.system
    start = int(system.component_atom_ptr[component_index])
    stop = int(system.component_atom_ptr[component_index + 1])
    atom_names = system.atom_symbol[start:stop].detach().cpu().tolist()
    coordinates = (
        structure.coordinates[start:stop]
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
    )
    coordinate_mask = structure.coordinate_mask[start:stop].detach().cpu()
    atoms = {
        system.symbols[int(symbol)]: (coordinates[index], bool(coordinate_mask[index]))
        for index, symbol in enumerate(atom_names)
    }
    for slot, atom_id in enumerate(("N", "CA", "C", "CB")):
        entry = atoms.get(atom_id)
        if entry is None:
            continue
        coordinate, present = entry
        if present:
            positions[slot] = coordinate
            mask[slot] = 1.0
    return positions, mask
