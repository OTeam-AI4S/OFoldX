r"""Canonical molecular lowering for the two BoltzGen architectures.

Two Processors, because the two architectures take different input types and produce
different tensor Inputs (§7 "separate arch only on signature/type difference"):

* ``BoltzGenProcessor`` lowers a design request (an optional target system plus the
  family-owned ``total_len`` topology and ``conditions`` schema) to ``BoltzGenInput``.
* ``BoltzGenIFoldProcessor`` lowers a fixed backbone plus family-owned inverse-fold
  constraints to ``BoltzGenIFoldInput``.

Parity (validated): the tensor featurization below is a faithful port, numerically validated
against released BoltzGen (denoiser max_abs 2.4e-4; extend_existing 5/5 exact 0; total_len
bit-exact). It reuses the field-compatible pure featurizer helpers from ``boltz1.processor``
read-only (the atom reference-conformer builder, MSA featurizer, token-bond indicator,
cyclic-period and residue-type helpers) and adds the BoltzGen-specific channels
(``contact_conditioning`` / ``contact_threshold`` / ``token_distance_mask`` / ``type_bonds``
/ ``chain_type`` / ``center_coords`` / ``token_to_bb4_atoms`` / ``coords`` and the design
conditioning ids). Conventions realized below (validated):
  - the 388-wide atom feature vector (Boltz-2 layout, no atom-pad-mask channel);
  - the neutral ``ss_type`` / ``binding_type`` / ``design_mask`` ids and their class integers;
  - the design pass's ``residue_type`` handling at generated positions;
  - the fully-generated ``total_len`` (no input system) path is implemented under the frozen
    §3.7 placement core (X1-B): the whole assembly is ONE declaration group whose inner
    placements are its chains, per-chain lengths are drawn from the family-owned ``total_len``
    budget, and the realized topology is exposed as ``BoltzGenLayout.placement_map``. Extending
    an EXISTING input system with generated rows (X1-A) is implemented faithfully to released
    BoltzGen (``task/predict/data_protein_binder.py`` / ``data_from_yaml.py``): the source
    instances are carried VERBATIM (whole, keeping their ids / coords / bonds so §3.7.3
    inpaint_sequence can address them by their source ``ComponentReference``) and the generated
    placements mint fresh non-colliding ``chain_{n}`` instances appended after the source, exactly
    as the released ``Structure.concatenate(str_native, empty_protein(seq_len))`` does. The
    :class:`~ofoldx.models.boltzgen.model.BoltzGenStructureGroup` conditioning lowers to the
    released ``token_distance_mask`` (§3.7.2, validated against the released featurizer).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from multimolecule.io import read_fasta_records
from torch import Tensor
from typing_extensions import Self

from ...data.components import get_component
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    InstanceIdAllocator,
    PlacementMap,
    PlacementSpec,
)
from ...data.placement import instance_of as _placement_instance_of
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem, Molecule, PolymerSequence
from ...data.structure import Structure
from ...io.structure import read_structure
from .model import (
    CANONICAL_TOKENS,
    NUM_CANONICAL_TOKENS,
    BoltzGenBinding,
    BoltzGenContact,
    BoltzGenCovalentBond,
    BoltzGenDesignMask,
    BoltzGenIFoldInput,
    BoltzGenIFoldLayout,
    BoltzGenInput,
    BoltzGenLayout,
    BoltzGenPlacements,
    BoltzGenResidueConstraint,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenSymmetry,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


# Contact-conditioning one-hot classes (must match model.CONTACT_CONDITIONING_CLASSES):
# 0 UNSPECIFIED, 1 UNSELECTED, 2 POCKET>BINDER, 3 BINDER>POCKET, 4 CONTACT.
_CONTACT_CLASSES = 5
_CONTACT_KIND_TO_CLASS = {"pocket>binder": 2, "binder>pocket": 3, "contact": 4}

# Neutral conditioning ids (FLAG: exact upstream vocab ordering unverified in this env).
_BINDING_UNSPECIFIED = 0
_BINDING_BINDING = 1
_BINDING_NOT_BINDING = 2
_SS_UNSPECIFIED = 3  # num_ss_types == 4; class 3 taken as the masked/unspecified label.
_SS_LABEL_TO_ID = {"H": 0, "E": 1, "L": 2}


def _normalize_structure_input(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> tuple[MolecularSystem | None, Structure | None]:
    r"""Resolve an input into (system, optional Structure)."""

    if input is None:
        return None, None
    if isinstance(input, str | PathLike):
        input = read_structure(Path(input).expanduser())
    if isinstance(input, Structure):
        return input.system, input
    if isinstance(input, MolecularSystem):
        return input, None
    raise TypeError("BoltzGen input must be a MolecularSystem, Structure, structure path, or None")


def _normalize_templates(
    templates: Sequence[Template | str | PathLike[str]],
) -> tuple[Template, ...]:
    r"""Validate the template evidence into aligned :class:`~ofoldx.data.evidence.Template` records.

    Released BoltzGen lowers templates only through an aligned self/target Structure
    (``template_from_tokens``); a bare structure path carries no target<->template alignment and so
    cannot be lowered into the template channel.
    """

    normalized: list[Template] = []
    for template in templates:
        if isinstance(template, Template):
            normalized.append(template)
            continue
        raise TypeError(
            "BoltzGen template evidence must be an aligned ofoldx.data.evidence.Template "
            "(a bare structure path has no target<->template alignment to lower)"
        )
    return tuple(normalized)


def _component_reference_to_token(
    reference: ComponentReference,
    token_index: dict[tuple[str, int], int],
) -> int:
    r"""Resolve one ``(instance_id, component_index)`` reference to its token row."""

    key = (reference[0], reference[1])
    if key not in token_index:
        raise ValueError(f"condition references residue {reference!r} not present in the lowered system")
    return token_index[key]


# --------------------------------------------------------------------------------------
# Frozen §3.7 X1-B whole-assembly ``total_len`` realization.
# --------------------------------------------------------------------------------------

# Placeholder generated component per polymer type (mirrors ODesign's generated-residue
# component ids): the realized backbone identity the diffusion designer overwrites.
_GENERATED_COMPONENT = {"protein": "UNK", "rna": "N", "dna": "DN"}


def _default_assembly(total_len: int | None) -> PlacementSpec:
    r"""The degenerate whole-assembly PlacementSpec when only a ``total_len`` budget is given.

    One declaration group ``"assembly"`` with one generated protein placement ``"chain_A"``
    spanning the whole ``total_len`` budget.
    """

    if total_len is None or total_len <= 0:
        raise ValueError("a BoltzGen assembly without a BoltzGenPlacements requires a positive total_len")
    return {"assembly": {"chain_A": GeneratedSegment("protein", total_len, total_len)}}


def _resolve_generated_lengths(placements: PlacementSpec, total_len: int | None) -> dict[str, int]:
    r"""Draw each generated placement's realized length from the family ``total_len`` budget.

    Frozen §3.7: length re-homes to the family. ``total_len`` is the whole-assembly budget
    that constrains the SUM of the generated placement lengths (upstream BoltzGen samples per
    segment and accepts when the total lands in the ``total_len`` window). Here the budget is a
    single target token count, so lengths are resolved deterministically: every placement starts
    at ``min_length`` and the remaining budget is distributed in declaration order up to each
    ``max_length``. ``total_len is None`` realizes every placement at its ``max_length``.
    """

    segments: list[tuple[str, GeneratedSegment]] = []
    for chain in placements.values():
        for placement_id, part in chain.items():
            if type(part) is not GeneratedSegment:
                continue
            if part.min_length < 0 or part.max_length < part.min_length:
                raise ValueError(
                    f"placement {placement_id!r} has an invalid length range "
                    f"[{part.min_length}, {part.max_length}]"
                )
            segments.append((placement_id, part))

    if total_len is None:
        return {placement_id: part.max_length for placement_id, part in segments}

    total_min = sum(part.min_length for _, part in segments)
    total_max = sum(part.max_length for _, part in segments)
    if not total_min <= total_len <= total_max:
        raise ValueError(
            f"total_len={total_len} is outside the assembly's generated-length range "
            f"[{total_min}, {total_max}]"
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
    r"""Realize one fully-generated whole-assembly PlacementSpec into a system + ``placement_map``.

    Frozen §3.7 X1-B (``input=None``): the PlacementSpec is ONE whole-assembly declaration group whose
    inner placements are the assembly chains; every placement is a
    :class:`~ofoldx.data.placement.GeneratedSegment` (an :class:`~ofoldx.data.placement.ExistingSpan`
    needs an input system and belongs to the implemented extend-existing X1-A path,
    :func:`_materialize_extension`). Each placement realizes to its own family-minted
    chain instance whose length is drawn from the family ``total_len`` budget; the realized chain id
    rides per-occurrence in ``ComponentReference[0]``. A zero-length placement realizes to ``()``
    (frozen coverage rule) and mints no instance.

    Realized-id convention (canonical, overseer-fixed): the realized instance id is minted by the
    shared :class:`~ofoldx.data.placement.InstanceIdAllocator`, which emits ``f"chain_{n}"`` with ``n``
    a 0-BASED counter advanced in (outer-group, inner-placement) declaration order over the NON-EMPTY
    realized placements only (a zero-length placement mints no chain and consumes no index). All four
    generative families share this single helper — there is no family prefix.

    Coverage (frozen §3.7 EXHAUSTION, W6 caveat c): the realized ``MolecularSystem`` is built PURELY
    from these placements (``entities`` / ``instances`` are populated only in the loop below), so every
    realized output component belongs to exactly one placement's occurrence tuple — a full partition
    with no unaddressable tail. A ranged placement (``min_length != max_length``) absorbs the whole-
    assembly ``total_len`` slack in ``_resolve_generated_lengths`` rather than seeding a free tail.
    """

    instance_of = _placement_instance_of(placements)  # validates global placement_id uniqueness
    lengths = _resolve_generated_lengths(placements, total_len)

    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    placement_map: PlacementMap = {}
    allocator = InstanceIdAllocator()
    entity_index = 0
    for group, chain in placements.items():
        realized_group: dict[str, tuple[ComponentReference, ...]] = {}
        for placement_id, part in chain.items():
            if type(part) is not GeneratedSegment:
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
    r"""The degenerate extend-existing PlacementSpec when only a ``total_len`` budget is given.

    Mirrors the released protein-binder path (``data_protein_binder.py``:
    ``Structure.concatenate(str_native, Structure.empty_protein(seq_len))``): the input system is
    carried verbatim and ONE generated protein chain of length ``total_len`` is appended after it.
    """

    if total_len is None or total_len <= 0:
        raise ValueError("total_len extension of an input system requires a positive total_len")
    return {"assembly": {"chain_A": GeneratedSegment("protein", total_len, total_len)}}


def _materialize_extension(
    system: MolecularSystem,
    placements: PlacementSpec | None,
    total_len: int | None,
) -> tuple[MolecularSystem, PlacementMap, dict[str, str], frozenset[ComponentReference]]:
    r"""Extend an EXISTING input system with generated rows (frozen §3.7 X1-A extend-existing).

    THE ONE CORRECT PATH (owner ruling, placement core froze 2026-07-18), faithful to released
    BoltzGen (``task/predict/data_protein_binder.py`` / ``data_from_yaml.py`` →
    ``Structure.concatenate``):

    * **Carry-verbatim** — every input instance is kept WHOLE with its original ``instance_id`` /
      entity / coordinates / bonds, and an :class:`~ofoldx.data.placement.ExistingSpan` occurrence
      REUSES its source ``(instance_id, index)``. Source rows are neither copied nor renumbered, so
      the §3.7.3 ``inpaint_sequence`` (which addresses existing residues by their source
      ``ComponentReference``) stays intact.
    * **Non-colliding generated ids** — each :class:`~ofoldx.data.placement.GeneratedSegment` mints
      a fresh ``chain_{n}`` via the shared :class:`~ofoldx.data.placement.InstanceIdAllocator`,
      SKIPPING any ``chain_{n}`` that collides with a carried source instance id, in
      (outer-group, inner-placement) declaration order. Generated chains are appended AFTER the
      carried source (matching released ``concatenate(str_native, empty_protein(...))``).

    ``placements is None`` is the ``total_len`` extension: one generated protein chain of length
    ``total_len`` is appended (see :func:`_default_extension`). A ``total_len`` given alongside an
    explicit :class:`~ofoldx.models.boltzgen.model.BoltzGenPlacements` constrains the SUM of the
    generated placement lengths exactly as the fully-generated path does
    (:func:`_resolve_generated_lengths`).

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
    instances: dict[str, str] = dict(system.instances)  # source carried verbatim, input order first
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
            if type(part) is ExistingSpan:
                if part.instance_id not in source_ids:
                    raise ValueError(
                        f"ExistingSpan placement {placement_id!r} names instance {part.instance_id!r}, "
                        "which is not an instance of the input system"
                    )
                source_length = len(system.entities[system.instances[part.instance_id]])
                if not (0 <= part.start <= part.stop <= source_length):
                    raise ValueError(
                        f"ExistingSpan placement {placement_id!r} interval [{part.start}, {part.stop}) is "
                        f"outside source instance {part.instance_id!r} of length {source_length}"
                    )
                # Carry-verbatim: reuse the source (instance_id, index); the source instance is already
                # carried whole in ``instances`` so its coordinates / MSA / bonds are untouched.
                realized_group[placement_id] = tuple(
                    (part.instance_id, index) for index in range(part.start, part.stop)
                )
                continue
            if type(part) is not GeneratedSegment:
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
            "extending an input system must realize at least one non-empty generated placement "
            "(nothing to design)"
        )
    extended = MolecularSystem(entities, instances, system.bonds)
    return extended, placement_map, instance_of, frozenset(generated)


def _materialize_covalent_bonds(
    system: MolecularSystem,
    covalent_bonds: tuple[BoltzGenCovalentBond, ...],
    placement_map: PlacementMap,
    instance_of: dict[str, str],
) -> MolecularSystem:
    r"""Resolve requested :class:`SymbolicBond` links and inject them into ``system.bonds``.

    Frozen §3.7.4: each :class:`~ofoldx.models.boltzgen.model.BoltzGenCovalentBond` wraps a
    :class:`~ofoldx.data.bonds.SymbolicBond` whose endpoints are lowered to realized
    :class:`~ofoldx.data.molecular.AtomSite` occurrences (an existing ``AtomSite`` passes through;
    a generated :class:`~ofoldx.data.placement.SymbolicAtomSite` resolves through
    ``placement_map`` / ``instance_of``). The realized :class:`~ofoldx.data.molecular.SystemBond`
    is added to the system so the released head-to-tail detector (``_cyclic_period``) derives
    ``cyclic_period`` exactly as it does for a bond the input system already carried; a bond whose
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
    r"""The polymer residues carrying a generated-placeholder identity (no reference conformer).

    A de-novo backbone the design step produced — and that the inverse-fold step re-lowers — marks
    its generated positions with the family placeholder component per polymer type
    (:data:`_GENERATED_COMPONENT`: protein ``UNK``, rna ``N``, dna ``DN``), exactly as
    :func:`_materialize_assembly` / :func:`_materialize_extension` mint them. A protein ``UNK``
    placeholder ships NO reference conformer (``get_component("UNK").reference_coordinates`` is
    empty), so the released ``backbone_only`` inverse-fold featurizer restricts it to the backbone
    and takes the GLY backbone template (:func:`_generated_reference_coordinates`). This set names
    those residues so the ``BoltzGenIFoldProcessor`` re-lowering reuses the identical substitution
    the ``BoltzGenProcessor`` design pass applies; a real fixed-context residue (its own conformer)
    is excluded, leaving it on the released ``_reference_coordinates`` path unchanged.
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
    r"""Reference coordinates for an assembly containing generated placement residues.

    A generated PROTEIN residue (placeholder ``UNK`` identity, no conformer) takes its
    backbone coordinate from a GLY template — the same substitution ODesign makes for
    generated backbones. Every other residue (fixed input, or a generated RNA/DNA placeholder
    whose ``N`` / ``DN`` component ships audited reference coordinates) uses its own component
    conformer, and a ligand ``Molecule`` component with no baked conformer falls back to the
    released RDKit-embedded conformer — byte-identical to the released ``_reference_coordinates``
    path for every non-generated-protein residue.
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
            isinstance(entity, PolymerSequence)
            and entity.polymer_type == "protein"
            and residue in generated_residues
        )
        if is_generated_protein:
            coordinate = gly_backbone.get(site.atom_id)
            if coordinate is None:
                raise ValueError(f"generated protein backbone has no GLY template coordinate for {site!r}")
            coordinates.append(coordinate)
            continue
        # Non-generated residue: reproduce the released ``_reference_coordinates`` lowering exactly
        # (baked component conformer, else the RDKit ligand fallback for a Molecule entity).
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


class BoltzGenProcessor:
    r"""Lower one BoltzGen structure-design request to ``BoltzGenInput``.

    Family-owned §3.7 whole-assembly topology (``total_len``), §3.7.2 spatial
    conditioning (``conditions``), and the released schema controls. The neutral
    conditioning channels the trunk always consumes (contact conditioning, token
    distance, binding/design/secondary-structure ids) are written for every token so
    the parity-passing graph runs unchanged.
    """

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Self:
        r"""Update output placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template | str | PathLike[str]] = (),
    ) -> BoltzGenInput:
        r"""Validate and lower one design request.

        Args:
            input: Optional target/context system, structure, or path. ``None`` means a
                fully-generated whole-assembly determined by the ``BoltzGenPlacements`` topology
                and the family-owned ``total_len`` length budget.
            total_len: Family-owned whole-assembly token count (frozen §3.7 length budget). When
                ``input`` is ``None`` it drives per-placement length realization; otherwise it
                extends the input system by one generated protein chain of length ``total_len``
                (frozen §3.7 X1-A, released protein-binder path).
            conditions: Family-owned §3.7.2 schema records, optionally including one
                :class:`~ofoldx.models.boltzgen.model.BoltzGenPlacements` whole-assembly topology.
            msa: Optional MSA evidence for the input polymer chains.
            templates: Optional template evidence (structure-design template channel).

        Returns:
            The lowered ``BoltzGenInput`` with its ``layout`` populated.
        """

        condition_values = tuple(conditions)
        placements = tuple(condition for condition in condition_values if type(condition) is BoltzGenPlacements)
        if len(placements) > 1:
            raise ValueError("BoltzGen accepts at most one BoltzGenPlacements condition")
        covalent_bonds = tuple(
            condition for condition in condition_values if type(condition) is BoltzGenCovalentBond
        )
        spatial_conditions = tuple(
            condition
            for condition in condition_values
            if type(condition) not in (BoltzGenPlacements, BoltzGenCovalentBond)
        )
        system, structure = _normalize_structure_input(input)

        placement_map: PlacementMap = {}
        instance_of: dict[str, str] = {}
        generated_residues: frozenset[ComponentReference] = frozenset()
        if system is None:
            # Frozen §3.7 X1-B: a fully-generated whole-assembly. The BoltzGenPlacements (or, absent
            # one, a single ``total_len``-long generated protein chain) is one declaration group
            # whose inner placements are the assembly chains; per-chain lengths are drawn from the
            # family-owned ``total_len`` budget and written into ``placement_map`` (the realized
            # family-owned chain id rides in ComponentReference[0]).
            if not placements and total_len is None:
                raise ValueError(
                    "a fully-generated BoltzGen assembly (input=None) requires a BoltzGenPlacements "
                    "condition or a total_len length budget"
                )
            placement = placements[0].placements if placements else _default_assembly(total_len)
            system, placement_map, instance_of = _materialize_assembly(placement, total_len)
            generated_residues = frozenset(
                reference
                for group in placement_map.values()
                for occurrences in group.values()
                for reference in occurrences
            )
            structure = None
        elif placements or total_len is not None:
            # Frozen §3.7 X1-A: extend the EXISTING input system with generated rows. Source instances
            # are carried verbatim (whole, keeping their ids/coords/bonds) and the generated placements
            # mint fresh non-colliding chain_{n} instances appended after the source — faithful to
            # released BoltzGen (``Structure.concatenate(str_native, empty_protein(...))``). The carried
            # ``structure`` still provides coordinates/resolved for the source atoms; generated atoms
            # fall back to reference conformers exactly like the fully-generated path.
            placement = placements[0].placements if placements else None
            system, placement_map, instance_of, generated_residues = _materialize_extension(
                system, placement, total_len
            )
        # else: a pure existing-input design (no total_len, no BoltzGenPlacements) — the system is
        # used as-is with no generated rows (placement_map stays empty).
        template_values = _normalize_templates(templates)
        if not system.instances:
            raise ValueError("BoltzGen requires at least one molecular instance")

        if covalent_bonds:
            system = _materialize_covalent_bonds(system, covalent_bonds, placement_map, instance_of)

        normalized_msa = _normalize_msa(msa, system)
        return self._prepare(
            system,
            structure=structure,
            msa=normalized_msa,
            conditions=spatial_conditions,
            placement_map=placement_map,
            instance_of=instance_of,
            generated_residues=generated_residues,
            templates=template_values,
        )

    def _prepare(
        self,
        system: MolecularSystem,
        *,
        structure: Structure | None,
        msa: MSA | MSACollection | None,
        conditions: tuple[BoltzGenSchema, ...],
        placement_map: PlacementMap | None = None,
        instance_of: dict[str, str] | None = None,
        generated_residues: frozenset[ComponentReference] = frozenset(),
        templates: tuple[Template, ...] = (),
    ) -> BoltzGenInput:
        r"""Build the released token/atom/MSA tensors plus BoltzGen conditioning channels.

        FLAG (node4, parity-untested): faithful best-effort; see module docstring.

        ``generated_residues`` names the GENERATED (design) output residues — the frozen §3.7
        placement occurrences realized from a :class:`~ofoldx.data.placement.GeneratedSegment`
        (a fully-generated X1-B assembly or the X1-A rows appended onto an input system). A
        carried :class:`~ofoldx.data.placement.ExistingSpan` occurrence is NOT in this set: it is
        an existing/context residue. The set drives (i) the backbone-only + GLY-template
        featurization of generated protein residues and (ii) the automatic ``design_mask``.
        """

        # Frozen §3.7: generated placement residues carry an ``UNK`` identity but no reference
        # conformer, so — exactly like ODesign — they are restricted to the protein backbone and
        # take GLY-template reference coordinates. The set is empty for a pure existing-input
        # design, leaving that (parity-passing) featurization byte-identical.
        tokens, atom_sites = _build_tokens(system, generated_residues=generated_residues)
        token_count = len(tokens)
        atom_features, ref_pos, ref_space, atom_mask, atom_to_token, atom_token_index, site_to_atom = (
            _build_atom_features(system, tokens, atom_sites, structure, generated_residues=generated_residues)
        )
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

        token_to_rep_atom = torch.tensor(
            [site_to_atom[cast(AtomSite, token["distogram_site"])] for token in tokens], dtype=torch.long
        )
        representative_atom_mask = torch.zeros(atom_features.shape[0], dtype=torch.bool)
        representative_atom_mask[token_to_rep_atom] = True
        token_bonds = _token_bond_indicator(system, site_to_token, token_count)
        msa_values = _msa_features(msa, tokens, system)
        residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
        molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
        cyclic_period = torch.tensor(
            [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens], dtype=torch.float32
        )

        # token_to_bb4_atoms (N, CA, C, O) and center coords, used only by the (gated)
        # token-distance module. FLAG: neutral geometry when the input has no coords.
        token_to_bb4 = torch.stack(
            [
                torch.tensor(
                    [site_to_atom[site] for site in cast(tuple[AtomSite, ...], token["bb4_sites"])], dtype=torch.long
                )
                for token in tokens
            ]
        )
        center_coords = ref_pos[token_to_rep_atom]

        # BoltzGen conditioning channels realized from the family records.
        contact_conditioning, contact_threshold = _contact_channels(conditions, token_index_by_residue, token_count)
        design_mask = _design_mask(conditions, token_index_by_residue, token_count, generated_residues)
        binding_type = _binding_ids(conditions, token_index_by_residue, token_count)
        ss_type = _ss_ids(conditions, token_index_by_residue, token_count)
        symmetry_index = _symmetry_index(conditions, token_index_by_residue, tokens)
        # Frozen §3.7.2: a BoltzGenStructureGroup ties its members into one reference-space group
        # whose intra-group pairwise geometry is visible; the released featurizer realizes exactly
        # one channel from it — ``token_distance_mask`` (reference_space_id stays per-residue in the
        # atom featurizer, unchanged by structure groups).
        structure_group = _structure_group_ids(conditions, token_index_by_residue, token_count, design_mask)
        token_distance_mask = _token_distance_mask(structure_group)
        # Released self-template channel (``template_from_tokens``); ``None`` leaves every template
        # field unset so the trunk skips the template module (== released all-zero dummy template).
        template_values = _template_channels(templates, tokens, token_count)

        tensor_values: dict[str, Tensor] = {
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
            "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
            "entity_index": torch.tensor([cast(int, token["entity_index"]) for token in tokens]),
            "symmetry_index": symmetry_index,
            "token_index": torch.arange(token_count),
            "atom_features": atom_features,
            "atom_to_token": atom_to_token,
            "atom_mask": atom_mask,
            "reference_positions": ref_pos,
            "reference_space_id": ref_space,
            "representative_atom_mask": representative_atom_mask,
            "atom_token_index": atom_token_index,
            "token_to_rep_atom": token_to_rep_atom,
            "residue_type": torch.nn.functional.one_hot(residue_types, 33).float(),
            "msa_profile": msa_values.pop("msa_profile"),
            "deletion_mean": msa_values.pop("deletion_mean"),
            "token_bonds": token_bonds,
            "type_bonds": torch.zeros(token_count, token_count, dtype=torch.long),
            "contact_conditioning": contact_conditioning,
            "contact_threshold": contact_threshold,
            "chain_type": molecule_types,
            "center_coords": center_coords,
            "token_to_bb4_atoms": token_to_bb4,
            "coords": ref_pos,
            "token_distance_mask": token_distance_mask,
            "molecule_type": molecule_types,
            "cyclic_period": cyclic_period,
            "method_feature": torch.zeros(token_count, dtype=torch.long),
            "modified": torch.zeros(token_count, dtype=torch.long),
            "binding_type": binding_type,
            "design_mask": design_mask,
            "ss_type": ss_type,
            **msa_values,
            **(template_values or {}),
        }
        placed = {
            name: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(device=self.device)
            for name, value in tensor_values.items()
        }
        layout = BoltzGenLayout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(cast("AtomSite | tuple[str, int]", token["token_site"]) for token in tokens),
            output_residues=tuple(cast("ComponentReference", token["reference"]) for token in tokens),
            output_instance_ids=tuple(cast(str, token["instance_id"]) for token in tokens),
            placement_map=placement_map if placement_map is not None else {},
            instance_of=instance_of if instance_of is not None else {},
        )
        return BoltzGenInput(**{name: value.unsqueeze(0) for name, value in placed.items()}, layout=layout)


class BoltzGenIFoldProcessor:
    r"""Lower one fixed backbone and its inverse-fold constraints to ``BoltzGenIFoldInput``.

    Family-owned constraints only: ``design`` positions (``design_mask``), per-residue
    amino-acid constraints (``aa_constraints`` → ``aa_constraint_mask``), and symmetric
    groups (``symmetry`` → ``symmetric_group``). It does **not** reuse RFdiffusion's frozen
    §3.7.3 ``inpaint_sequence``; BoltzGen inverse-fold constraints are family-owned.
    """

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Self:
        r"""Update output placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
    ) -> BoltzGenIFoldInput:
        r"""Validate and lower one fixed backbone to inverse-fold input."""

        if isinstance(input, str | PathLike):
            input = read_structure(Path(input).expanduser())
        if not isinstance(input, Structure):
            raise TypeError("BoltzGenIFoldProcessor requires a fixed backbone Structure or structure path")
        system = input.system
        if not system.instances:
            raise ValueError("BoltzGen inverse folding requires at least one molecular instance")

        normalized_msa = _normalize_msa(msa, system)
        return self._prepare(
            input,
            msa=normalized_msa,
            design=tuple(design),
            aa_constraints=tuple(aa_constraints),
            symmetry=tuple(tuple(unit) for unit in symmetry),
        )

    def _prepare(
        self,
        structure: Structure,
        *,
        msa: MSA | MSACollection | None,
        design: tuple[ComponentReference, ...],
        aa_constraints: tuple[BoltzGenResidueConstraint, ...],
        symmetry: tuple[tuple[ComponentReference, ...], ...],
    ) -> BoltzGenIFoldInput:
        r"""Build the inverse-fold token/atom tensors plus the caller-supplied masks.

        FLAG (node4, parity-untested): faithful best-effort; see module docstring.
        """

        system = structure.system
        # A de-novo backbone (design→inverse-fold hand-off) carries generated protein positions with
        # the ``UNK`` placeholder identity, which ships NO reference conformer. Detect those residues
        # so the shared featurizer restricts them to the backbone and takes the GLY-template reference
        # coordinates — the identical substitution the design processor applies (released
        # ``backbone_only`` inverse-fold path). Empty for a real fixed backbone, leaving it unchanged.
        generated_residues = _generated_placeholder_residues(system)
        tokens, atom_sites = _build_tokens(system, generated_residues=generated_residues)
        token_count = len(tokens)
        atom_features, ref_pos, ref_space, atom_mask, atom_to_token, atom_token_index, site_to_atom = (
            _build_atom_features(system, tokens, atom_sites, structure, generated_residues=generated_residues)
        )
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
        msa_values = _msa_features(msa, tokens, system)
        residue_types = torch.tensor([cast(int, token["residue_type"]) for token in tokens])
        molecule_types = torch.tensor([cast(int, token["molecule_type"]) for token in tokens])
        cyclic_period = torch.tensor(
            [_cyclic_period(system, cast(str, token["instance_id"])) for token in tokens], dtype=torch.float32
        )
        token_to_rep_atom = torch.tensor(
            [site_to_atom[cast(AtomSite, token["distogram_site"])] for token in tokens], dtype=torch.long
        )
        token_to_bb4 = torch.stack(
            [
                torch.tensor(
                    [site_to_atom[site] for site in cast(tuple[AtomSite, ...], token["bb4_sites"])], dtype=torch.long
                )
                for token in tokens
            ]
        )
        center_coords = ref_pos[token_to_rep_atom]

        design_mask = _design_mask_from_positions(design, token_index_by_residue, token_count)
        aa_constraint_mask = _aa_constraint_mask(aa_constraints, token_index_by_residue, token_count)
        symmetric_group = _symmetry_index(
            (BoltzGenSymmetry(units=symmetry),) if symmetry else (), token_index_by_residue, tokens
        )

        tensor_values: dict[str, Tensor] = {
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "residue_index": torch.tensor([cast(int, token["residue_index"]) for token in tokens]),
            "chain_index": torch.tensor([cast(int, token["chain_index"]) for token in tokens]),
            "atom_features": atom_features,
            "atom_to_token": atom_to_token,
            "atom_mask": atom_mask,
            "reference_positions": ref_pos,
            "reference_space_id": ref_space,
            "residue_type": torch.nn.functional.one_hot(residue_types, 33).float(),
            "msa_profile": msa_values["msa_profile"],
            "deletion_mean": msa_values["deletion_mean"],
            "token_bonds": token_bonds,
            "type_bonds": torch.zeros(token_count, token_count, dtype=torch.long),
            "chain_type": molecule_types,
            "center_coords": center_coords,
            "token_to_bb4_atoms": token_to_bb4,
            "coords": ref_pos,
            "cyclic_period": cyclic_period,
            "method_feature": torch.zeros(token_count, dtype=torch.long),
            "modified": torch.zeros(token_count, dtype=torch.long),
            "design_mask": design_mask,
            "aa_constraint_mask": aa_constraint_mask,
            "symmetric_group": symmetric_group,
        }
        placed = {
            name: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(device=self.device)
            for name, value in tensor_values.items()
        }
        layout = BoltzGenIFoldLayout(
            system=system,
            atom_sites=atom_sites,
            token_sites=tuple(cast("AtomSite | tuple[str, int]", token["token_site"]) for token in tokens),
            output_residues=tuple(cast("ComponentReference", token["reference"]) for token in tokens),
            output_instance_ids=tuple(cast(str, token["instance_id"]) for token in tokens),
        )
        return BoltzGenIFoldInput(**{name: value.unsqueeze(0) for name, value in placed.items()}, layout=layout)


# --------------------------------------------------------------------------------------
# Shared featurization helpers (BoltzGen-local; reuse the boltz1 pure helpers within).
# --------------------------------------------------------------------------------------


def _build_tokens(
    system: MolecularSystem,
    generated_residues: frozenset[ComponentReference] = frozenset(),
) -> tuple[list[dict[str, object]], tuple[AtomSite, ...]]:
    r"""Build the released per-token records and the flat model atom order.

    FLAG: mirrors the Boltz-1 tokenizer (polymer residue = one token; ligand atom = one
    token). BoltzGen shares the Boltz data pipeline, but this tokenization is not
    numerically verified here.

    ``generated_residues`` names the frozen §3.7 whole-assembly generated placement residues;
    a generated PROTEIN residue is restricted to its backbone ``{N, CA, C, O}`` atoms (it has
    a placeholder ``UNK`` identity and no side-chain conformer), mirroring ODesign. Empty for
    an existing-input design, so that tokenization is unchanged.
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
                    entity.polymer_type == "protein"
                    and (instance_id, monomer_index) in generated_residues
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
                    distogram_name = "CA" if component_id == "GLY" else "CB"
                    bb4_names = ("N", "CA", "C", "O")
                else:
                    center_name = "C1'"
                    distogram_name = (
                        "C4"
                        if component_id in {"A", "G", "DA", "DG"}
                        else "C2" if component_id in {"C", "U", "DC", "DT"} else "C1'"
                    )
                    bb4_names = ("C1'", "C3'", "C4'", "C1'")
                center_site = by_name.get(center_name, atom_sites[0])
                distogram_site = by_name.get(distogram_name, center_site)
                tokens.append(
                    {
                        "atom_sites": atom_sites,
                        "token_site": (instance_id, monomer_index),
                        "reference": ComponentReference((instance_id, monomer_index)),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": monomer_index,
                        "residue_index": monomer_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": molecule_type,
                        "residue_type": residue_type,
                        "distogram_site": distogram_site,
                        "bb4_sites": tuple(by_name.get(name, center_site) for name in bb4_names),
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
                        "reference": ComponentReference((instance_id, site.component_index)),
                        "instance_id": instance_id,
                        "entity_id": entity_id,
                        "monomer_index": None,
                        "residue_index": site.component_index,
                        "chain_index": chain_index,
                        "entity_index": entity_index,
                        "symmetry_index": symmetry_index,
                        "molecule_type": 3,
                        "residue_type": 22,
                        "distogram_site": site,
                        "bb4_sites": (site, site, site, site),
                    }
                )

    atom_sites = tuple(site for token in tokens for site in cast(tuple[AtomSite, ...], token["atom_sites"]))
    return tokens, atom_sites


def _build_atom_features(
    system: MolecularSystem,
    tokens: list[dict[str, object]],
    atom_sites: tuple[AtomSite, ...],
    structure: Structure | None,
    generated_residues: frozenset[ComponentReference] = frozenset(),
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[AtomSite, int]]:
    r"""Build the 388-wide BoltzGen/Boltz-2 atom reference feature vector and index maps.

    FLAG: the 388 layout is ``[ref_pos(3), charge(1), element(128), atom_name_chars(4x64)]``
    (no atom-pad-mask channel, unlike Boltz-1's 389). Not parity-verified here.

    ``generated_residues`` (frozen §3.7 X1-B) selects the whole-assembly generated placement
    residues whose ``UNK`` identity carries no reference conformer; their backbone atoms take
    GLY-template coordinates (mirroring ODesign). Empty for an existing-input design, which then
    uses the released ``_reference_coordinates`` path unchanged.
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
    resolved = torch.zeros(atom_count, dtype=torch.bool)
    if structure is not None:
        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        for atom_index, site in enumerate(atom_sites):
            source_index = coordinate_index.get(site)
            if source_index is not None and bool(structure.coordinate_mask[source_index]):
                resolved[atom_index] = True

    ref_pos = torch.zeros(padded_atom_count, 3)
    ref_pos[:atom_count] = torch.tensor(reference_coordinates)
    ref_space = torch.zeros(padded_atom_count, dtype=torch.long)
    ref_space[:atom_count] = torch.tensor(reference_space_ids)
    _augment_reference_positions(ref_pos[:atom_count], ref_space[:atom_count], resolved)

    atomic_number = torch.zeros(padded_atom_count, dtype=torch.long)
    charge = torch.zeros(padded_atom_count)
    atom_name_chars = torch.zeros(padded_atom_count, 4, dtype=torch.long)
    atom_mask = torch.zeros(padded_atom_count, dtype=torch.bool)
    atom_to_token = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_token_index = torch.zeros(padded_atom_count, dtype=torch.long)
    atom_names = _model_atom_names(system, atom_sites)
    periodic_table = GetPeriodicTable()
    for atom_index, (site, atom_name) in enumerate(zip(atom_sites, atom_names)):
        atom = system.atom(site)
        atomic_number[atom_index] = periodic_table.GetAtomicNumber(atom.element)
        charge[atom_index] = atom.formal_charge
        atom_name_chars[atom_index] = torch.tensor(
            [ord(character) - 32 if character != " " else 0 for character in atom_name.ljust(4)]
        )
        atom_mask[atom_index] = True
        token_index = site_to_token[site]
        atom_to_token[atom_index] = token_index
        atom_token_index[atom_index] = cast(tuple[AtomSite, ...], tokens[token_index]["atom_sites"]).index(site)

    atom_features = torch.cat(
        (
            ref_pos,
            charge.unsqueeze(-1),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(atom_name_chars, 64).reshape(padded_atom_count, 256).float(),
        ),
        dim=-1,
    )
    return atom_features, ref_pos, ref_space, atom_mask, atom_to_token, atom_token_index, site_to_atom


def _contact_channels(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
) -> tuple[Tensor, Tensor]:
    r"""Realize contact records into the ``[N, N, 5]`` one-hot + ``[N, N]`` threshold."""

    classes = torch.zeros(token_count, token_count, dtype=torch.long)  # 0 == UNSPECIFIED
    threshold = torch.zeros(token_count, token_count)
    for condition in conditions:
        if type(condition) is not BoltzGenContact:
            continue
        class_id = _CONTACT_KIND_TO_CLASS[condition.kind]
        contact = condition.contact
        rows = [_component_reference_to_token(reference, token_index_by_residue) for reference in contact.side_a]
        columns = [_component_reference_to_token(reference, token_index_by_residue) for reference in contact.side_b]
        for row in rows:
            for column in columns:
                classes[row, column] = class_id
                classes[column, row] = class_id
                threshold[row, column] = contact.cutoff_angstrom
                threshold[column, row] = contact.cutoff_angstrom
    contact_conditioning = torch.nn.functional.one_hot(classes, _CONTACT_CLASSES).float()
    return contact_conditioning, threshold


def _design_mask(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
    generated_residues: frozenset[ComponentReference] = frozenset(),
) -> Tensor:
    r"""Realize the ``[N]`` boolean design mask.

    A token is designable iff it is a GENERATED placement residue (released: the appended /
    minted generated rows are always designed — ``data_protein_binder.py`` marks the appended
    chain, ``empty_protein`` supplies its placeholder identity) OR it is named by an explicit
    :class:`~ofoldx.models.boltzgen.model.BoltzGenDesignMask` record (redesign of an existing
    residue). The two sources are unioned.
    """

    positions: list[ComponentReference] = list(generated_residues)
    for condition in conditions:
        if type(condition) is BoltzGenDesignMask:
            positions.extend(condition.positions)
    return _design_mask_from_positions(tuple(positions), token_index_by_residue, token_count)


def _structure_group_ids(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
    design_mask: Tensor,
) -> Tensor:
    r"""Realize ``BoltzGenStructureGroup`` records into the per-token ``[N]`` structure-group ids.

    Frozen §3.7.2 (released ``design_info.res_structure_groups`` → ``tokens["structure_group"]``):
    the default group is ``0`` (free / ungrouped); each record assigns its ``group_id`` to every
    named member. A designed/generated residue must stay in group ``0`` — its backbone geometry is
    not fixed, so it cannot be a distance-conditioning anchor (released asserts exactly this in the
    training selector and ``DesignInfo.is_valid``).
    """

    ids = torch.zeros(token_count, dtype=torch.long)
    for condition in conditions:
        if type(condition) is not BoltzGenStructureGroup:
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
    r"""Realize the per-token structure groups into the released ``[N, N]`` ``token_distance_mask``.

    Byte-faithful to ``featurizer.process_token_features`` (BoltzGen): the whole matrix is ``1`` when
    every token shares group ``1`` (the released efficient special case), otherwise cell ``(i, j)`` is
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
    token_count: int,
) -> Tensor:
    r"""Realize ``BoltzGenBinding`` records into the ``[N]`` binding-specification ids."""

    ids = torch.full((token_count,), _BINDING_UNSPECIFIED, dtype=torch.long)
    for condition in conditions:
        if type(condition) is not BoltzGenBinding:
            continue
        for reference in condition.binding.residues:
            ids[_component_reference_to_token(reference, token_index_by_residue)] = _BINDING_BINDING
        for reference in condition.not_binding:
            ids[_component_reference_to_token(reference, token_index_by_residue)] = _BINDING_NOT_BINDING
    return ids


def _ss_ids(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
) -> Tensor:
    r"""Realize ``BoltzGenSecondaryStructure`` records into the ``[N]`` ss ids."""

    ids = torch.full((token_count,), _SS_UNSPECIFIED, dtype=torch.long)
    for condition in conditions:
        if type(condition) is not BoltzGenSecondaryStructure:
            continue
        for reference, label in condition.assignments:
            ids[_component_reference_to_token(reference, token_index_by_residue)] = _SS_LABEL_TO_ID[label]
    return ids


def _symmetry_index(
    conditions: tuple[BoltzGenSchema, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    tokens: list[dict[str, object]],
) -> Tensor:
    r"""Realize ``BoltzGenSymmetry`` records into the ``[N]`` symmetry-copy index.

    Defaults to the tokenizer's per-entity symmetry-copy index; an explicit
    ``BoltzGenSymmetry`` overrides matched positions with the unit ordinal.
    """

    ids = torch.tensor([cast(int, token["symmetry_index"]) for token in tokens], dtype=torch.long)
    for condition in conditions:
        if type(condition) is not BoltzGenSymmetry:
            continue
        for unit_index, unit in enumerate(condition.units):
            for reference in unit:
                ids[_component_reference_to_token(reference, token_index_by_residue)] = unit_index
    return ids


def _aa_constraint_mask(
    aa_constraints: tuple[BoltzGenResidueConstraint, ...],
    token_index_by_residue: dict[tuple[str, int], int],
    token_count: int,
) -> Tensor:
    r"""Realize per-residue AA constraints into the ``[N, 20]`` canonical-AA block mask.

    Matches the released ``parse_residue_constraints`` contract: a ``[N, 20]`` mask over the 20
    canonical amino acids (``CANONICAL_TOKENS`` order) where ``0`` is allowed and ``1`` disallowed;
    the default is all-zero (unconstrained). ``BoltzGenIFoldModel._build_constraint_logit_mask``
    consumes it as ``> 0`` -> ``-1e6`` additive logit bias. Each ``allowed`` whitelist blocks every
    canonical AA except the listed codes, accumulated with ``max`` so overlapping constraints
    intersect (only AAs allowed by every covering constraint survive).
    """

    mask = torch.zeros(token_count, NUM_CANONICAL_TOKENS, dtype=torch.float32)
    for constraint in aa_constraints:
        token = _component_reference_to_token(constraint.position, token_index_by_residue)
        block = torch.ones(NUM_CANONICAL_TOKENS, dtype=torch.float32)
        for code in constraint.allowed:
            block[CANONICAL_TOKENS.index(code)] = 0.0
        mask[token] = torch.maximum(mask[token], block)
    return mask


# --------------------------------------------------------------------------------------
# Template channel (released ``template_from_tokens`` self-template featurizer).
#
# Released BoltzGen has ONE real template featurizer, ``task/predict/data_from_generated.
# template_from_tokens``: a SELF-template that writes the target's own per-token backbone
# geometry (``compute_frame`` frame, CB distogram atom, CA center, residue-type one-hot)
# for a subset of tokens named by a ``token_mask`` (the fixed / non-design tokens, released
# ``design_mask_templates=true`` -> ``~design_mask``). There is no external-template-file
# featurizer upstream (``TemplateInfo`` is defined but never lowered; every released loader
# passes ``load_dummy_templates`` except the folding stage's self-template). This lowering
# reproduces ``template_from_tokens`` per-token exactly and, generalizing to the ofoldx
# ``Template`` evidence, sources the geometry from an aligned template Structure (the released
# self-template is the identity-alignment, template==target special case, validated exact vs
# the released tokenizer + featurizer).
# --------------------------------------------------------------------------------------


def _template_frame(
    nitrogen: np.ndarray, alpha: np.ndarray, carbon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Backbone frame for one residue — verbatim released ``tokenizer.compute_frame``.

    Gram-Schmidt over ``(C - CA, N - CA)``: ``e1`` along ``C - CA``, ``e2`` the ``N - CA``
    component orthogonal to ``e1``, ``e3 = e1 x e2``; the rotation columns are ``[e1, e2, e3]``
    and the translation is ``CA``. The ``+ 1e-10`` norm epsilons match released bit-for-bit.
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
    r"""The token center atom (released ``atom_center``): CA for protein, C1' for nucleic."""

    return "CA" if polymer_type == "protein" else "C1'"


def _template_disto_atom(component_id: str, polymer_type: str) -> str:
    r"""The token distogram atom (released ``atom_disto``), mirroring :func:`_build_tokens`."""

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
    r"""Look up one template atom's coordinate + presence, or ``(None, False)`` if absent."""

    entry = coordinates.get((instance_id, component_index, atom_id))
    if entry is None:
        return None, False
    return entry


def _template_channels(
    templates: tuple[Template, ...],
    tokens: list[dict[str, object]],
    token_count: int,
) -> dict[str, Tensor] | None:
    r"""Lower aligned :class:`~ofoldx.data.evidence.Template` evidence into the BoltzGen template tensors.

    Returns ``None`` when no templates are supplied (the model then skips the template module,
    numerically identical to the released design-stage ``load_dummy_templates`` all-zero channel).
    Otherwise returns the nine per-template ``[T, N, ...]`` feature tensors the ``TemplateModule``
    consumes, reproducing released ``template_from_tokens`` per aligned token: residue-type one-hot
    (``num_tokens == 33``), ``compute_frame`` backbone frame (protein, all of N/CA/C resolved),
    CB (``disto``) coords + mask, CA (``center``) coords, ``template_mask`` / ``visibility_ids`` set
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
    restype_ids = torch.zeros(tdim, token_count, dtype=torch.long)
    frame_rot = torch.zeros(tdim, token_count, 3, 3)
    frame_t = torch.zeros(tdim, token_count, 3)
    cb_coords = torch.zeros(tdim, token_count, 3)
    ca_coords = torch.zeros(tdim, token_count, 3)
    cb_mask = torch.zeros(tdim, token_count)
    frame_mask = torch.zeros(tdim, token_count)
    template_mask = torch.zeros(tdim, token_count)
    visibility_ids = torch.zeros(tdim, token_count)

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

                restype = _residue_type(component_id, polymer_type)
                for target_token in targets:
                    restype_ids[template_index, target_token] = restype
                    if center_coord is not None:
                        ca_coords[template_index, target_token] = torch.from_numpy(
                            np.asarray(center_coord, dtype=np.float32)
                        )
                    if disto_coord is not None:
                        cb_coords[template_index, target_token] = torch.from_numpy(
                            np.asarray(disto_coord, dtype=np.float32)
                        )
                    cb_mask[template_index, target_token] = 1.0 if disto_present else 0.0
                    if has_frame:
                        frame_rot[template_index, target_token] = torch.from_numpy(
                            np.asarray(rotation, dtype=np.float32)
                        )
                        frame_t[template_index, target_token] = torch.from_numpy(
                            np.asarray(translation, dtype=np.float32)
                        )
                        frame_mask[template_index, target_token] = 1.0
                    template_mask[template_index, target_token] = 1.0
                    visibility_ids[template_index, target_token] = 1.0

    return {
        "template_restype": torch.nn.functional.one_hot(restype_ids, 33),
        "template_frame_rot": frame_rot,
        "template_frame_t": frame_t,
        "template_cb": cb_coords,
        "template_ca": ca_coords,
        "template_mask_cb": cb_mask,
        "template_mask_frame": frame_mask,
        "template_mask": template_mask,
        "visibility_ids": visibility_ids,
    }


# --------------------------------------------------------------------------------------
# Field-compatible Boltz-1 featurizer helpers, duplicated verbatim into this family so BoltzGen
# is self-contained (no cross-family import into boltz1). Behaviour is field-compatible for the
# token/atom/MSA fields BoltzGen shares with Boltz-1.
# --------------------------------------------------------------------------------------

# Boltz-1 shares the 33-token residue vocabulary with Boltz-2 (``vocab_size=33`` in both configs): protein
# residues occupy tokens 2..21, the family UNKNOWN tokens are 22/27/32, RNA 23..27, DNA 28..32, and token 1
# is the MSA gap. The Processor owns this symbol->token mapping exactly as ``boltz2/processor.py`` does.
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
# Processor owns the symbol->token mapping keyed by the entity's resolved polymer_type, reproducing the
# family ``_residue_type`` path: standard residues map to their model token and everything else to the
# family UNKNOWN token. RNA one-letter symbols equal their component IDs directly.
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
    r"""Map one UPPERCASE one-letter MSA symbol (or ``-``) to a Boltz-1 token."""

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
    r"""Build the symmetric inter-token bond indicator ``[N, N, 1]`` consumed by ``token_bond_projection``.

    The trailing singleton mirrors the released featurizer (``bonds = bonds.unsqueeze(-1)``): the design
    trunk feeds it straight into ``nn.Linear(1, ...)`` and the inverse-fold ``build_knn_graph`` indexes
    ``token_bonds[..., 0]`` (released ``inverse_fold`` asserts ``shape[-1] == 1``).
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
        for bond in entity.bonds:
            add(
                AtomSite(instance_id, bond.component_a, bond.atom_a),
                AtomSite(instance_id, bond.component_b, bond.atom_b),
            )
    for bond in system.bonds:
        add(bond.site_a, bond.site_b)
    return token_bonds.unsqueeze(-1)


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
        # its dense row_pairing/row_is_paired plan across the paired and unpaired pools -- the same partition
        # the pre-migration nested-tuple plan drew, now read off the dense arrays (frozen §3.6 evidence API).
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
            value.entity_id: cast(PolymerSequence, system.entities[value.entity_id]).polymer_type
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
            # pre-migration taxonomic-pair semantics.
            taxonomic_pair = (
                isinstance(msa, MSACollection)
                and present > 1
                and all(is_paired for _, _, is_paired in selections)
            )
            for column_index, source_row, is_paired in selections:
                value = columns[column_index]
                polymer_type = polymer_types[value.entity_id]
                aligned = value.paired_sequences[source_row] if is_paired else value.unpaired_sequences[source_row]
                deletion_source = (
                    value.paired_deletion_counts if is_paired else value.unpaired_deletion_counts
                )[source_row]
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
        "msa": rows,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_paired": paired,
        "msa_mask": torch.ones_like(rows, dtype=torch.bool),
        "msa_profile": torch.nn.functional.one_hot(rows, 33).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


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
    r"""Apply the released conformer augmentation with a caller-injected RNG.

    Per OWNER principle P1 the seed→generator resolution lives at the Pipeline boundary; this
    reference-conformer roto-translation draws from the injected ``generator`` (``None`` means the
    ambient RNG, matching the upstream Boltz featurizer). No hardcoded seed is created here.
    """

    if not len(space_ids):
        return
    for space_id in range(int(space_ids.max().item())):
        included = space_ids == space_id
        mask = resolved[included]
        if not bool(mask.any()):
            continue
        values = positions[included]
        center = (values * mask[:, None]).sum(dim=0, keepdim=True) / mask.sum()
        quaternion = torch.randn(
            4,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )
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
        positions[included] = (values - center) @ rotation + torch.randn(
            1,
            3,
            dtype=values.dtype,
            device=values.device,
            generator=generator,
        )


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
    r"""Materialize canonical molecule data as RDKit chemistry without consulting provenance."""

    from rdkit import Chem

    editable = Chem.RWMol()
    atom_index: dict[tuple[int, str], int] = {}
    for component_index, component in enumerate(molecule.components):
        for atom in component.atoms:
            rdkit_atom = Chem.Atom(atom.element)
            if atom.isotope is not None:
                rdkit_atom.SetIsotope(atom.isotope)
            rdkit_atom.SetFormalCharge(atom.formal_charge)
            rdkit_atom.SetNumRadicalElectrons(atom.radical_electrons)
            rdkit_atom.SetNumExplicitHs(atom.explicit_hydrogens)
            rdkit_atom.SetNoImplicit(not atom.implicit_hydrogens_allowed)
            rdkit_atom.SetIsAromatic(atom.aromatic)
            atom_index[(component_index, atom.atom_id)] = editable.AddAtom(rdkit_atom)

    bond_types = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
    }
    for component_index, component in enumerate(molecule.components):
        for bond in component.bonds or ():
            editable.AddBond(
                atom_index[(component_index, bond.atom_a)],
                atom_index[(component_index, bond.atom_b)],
                bond_types[bond.order],
            )
            editable.GetBondWithIdx(editable.GetNumBonds() - 1).SetIsAromatic(bond.aromatic)
    for bond in molecule.bonds:
        editable.AddBond(
            atom_index[(bond.component_a, bond.atom_a)],
            atom_index[(bond.component_b, bond.atom_b)],
            bond_types[bond.order],
        )
        editable.GetBondWithIdx(editable.GetNumBonds() - 1).SetIsAromatic(bond.aromatic)

    value = editable.GetMol()
    Chem.SanitizeMol(value)
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


def _normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    if msa is None:
        return None
    if isinstance(msa, MSA | MSACollection):
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
    path = Path(msa).expanduser()
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA is unambiguous only for one protein entity; pass MSACollection")
    records = read_fasta_records(path)
    if not records:
        raise ValueError("MSA FASTA contains no rows")
    # Project each row to the query axis (drop query-gap columns, fold ``.`` into ``-``) with all-zero
    # deletion counts; row 0 is the ungapped query.
    query = records[0].sequence.upper()
    keep = [index for index, symbol in enumerate(query) if symbol not in "-."]
    width = len(keep)
    projected: list[str] = []
    for record in records:
        sequence = record.sequence.upper()
        projected.append("".join("-" if sequence[index] in "-." else sequence[index] for index in keep))
    return MSA(
        entity_id=proteins[0],
        unpaired_sequences=tuple(projected),
        unpaired_deletion_counts=np.zeros((len(projected), width), dtype=np.uint32),
        paired_sequences=(projected[0],),
        paired_deletion_counts=np.zeros((1, width), dtype=np.uint32),
    )
