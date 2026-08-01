r"""
Decoded OInvFold polymer and ligand inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...data.components import get_component
from ...data.molecular import (
    AMINO_ACID_ONE_TO_THREE,
    AMINO_ACID_THREE_TO_ONE,
    AtomSite,
    ChemicalComponent,
    ComponentReference,
    replace_components,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline
from ...pipelines.input import resolve_structure_input
from ...results import Score
from .model import (
    OINVFOLD_LIGAND_ELEMENTS,
    OINVFOLD_POLYMER_TOKENS,
    OInvFoldLigandInput,
    OInvFoldLigandModel,
    OInvFoldOutput,
    OInvFoldPolymerInput,
    OInvFoldPolymerModel,
)
from .processor import (
    OInvFoldLigandProcessor,
    OInvFoldPolymerProcessor,
    oinvfold_polymer_atom_ids,
)


@dataclass(eq=False)
class OInvFoldPolymerCandidate:
    r"""
    One decoded full-chain sequence and design-mask realization.

    Attributes:
        structure: Canonical Structure with only design-masked identities replaced.
        sequence: Full-chain decoded OInvFold sequence.
        total_log_probability: Sum of raw log probabilities over designed positions.
        mean_log_probability: Mean raw log probability over designed positions.
    """

    structure: Structure
    sequence: str
    total_log_probability: float
    mean_log_probability: float

    @property
    def scores(self) -> dict[str, Score]:
        r"""Return sequence log probabilities in the shared score vocabulary."""

        return {
            "total_log_probability": Score(self.total_log_probability, unit="nat", preferred="higher"),
            "mean_log_probability": Score(
                self.mean_log_probability,
                unit="nat/residue",
                preferred="higher",
            ),
        }


@dataclass(eq=False)
class OInvFoldPolymerResult:
    r"""
    Raw polymer predictions and generation-ordered decoded candidates.
    """

    target: Structure
    component_references: tuple[ComponentReference, ...]
    design_mask: NDArray[np.bool_]
    logits: NDArray
    log_probabilities: NDArray
    probabilities: NDArray
    candidates: tuple[OInvFoldPolymerCandidate, ...]

    @property
    def best(self) -> OInvFoldPolymerCandidate:
        r"""Return the highest-ranked greedy or beam candidate."""

        return self.candidates[0]


@dataclass(eq=False)
class OInvFoldLigandCandidate:
    r"""
    One atom-element assignment without fabricated molecular connectivity.

    Attributes:
        context: Optional canonical context retained from the input.
        slot_ids: Ordered caller-owned provisional atom identifiers.
        coordinates: Coordinates aligned with ``slot_ids``.
        elements: Predicted element-category labels aligned with ``slot_ids``.
        per_atom_probabilities: Temperature-view probabilities of the selected labels.
        total_log_probability: Sum of raw per-atom log probabilities.
        mean_log_probability: Mean raw per-atom log probability.
        bond_state: Explicitly unresolved bond topology.
    """

    context: Structure | None
    slot_ids: tuple[str, ...]
    coordinates: NDArray
    elements: tuple[str, ...]
    per_atom_probabilities: NDArray
    total_log_probability: float
    mean_log_probability: float
    bond_state: Literal["unresolved"] = "unresolved"

    @property
    def scores(self) -> dict[str, Score]:
        r"""Return element-assignment log probabilities in the shared score vocabulary."""

        return {
            "total_log_probability": Score(self.total_log_probability, unit="nat", preferred="higher"),
            "mean_log_probability": Score(
                self.mean_log_probability,
                unit="nat/atom",
                preferred="higher",
            ),
        }


@dataclass(eq=False)
class OInvFoldLigandResult:
    r"""
    Raw atom predictions and generation-ordered element assignments.
    """

    context: Structure | None
    slot_ids: tuple[str, ...]
    coordinates: NDArray
    logits: NDArray
    log_probabilities: NDArray
    probabilities: NDArray
    candidates: tuple[OInvFoldLigandCandidate, ...]

    @property
    def best(self) -> OInvFoldLigandCandidate:
        r"""Return the highest-ranked greedy or beam candidate."""

        return self.candidates[0]


class OInvFoldPolymerPipeline(PairedPipeline[OInvFoldPolymerModel, OInvFoldPolymerProcessor]):
    r"""
    Full-chain protein, DNA, or RNA OInvFold inference.

    Load a published pair with [`ofoldx.pipeline`][]. Design conditions on observed
    backbone coordinates, so it takes a structure read from a file rather than a topology
    built from a sequence.
    """

    model_architecture = "oinvfold"

    def prepare_design(
        self,
        input: Structure | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldPolymerInput:
        r"""Prepare one polymer structure for OInvFold inference."""

        structure = resolve_structure_input(input)
        if not isinstance(structure, Structure):
            raise TypeError("OInvFold polymer design requires a Structure")
        return self.processor(
            structure,
            instance_id=instance_id,
            design_positions=design_positions,
            filter_by_occupancy=filter_by_occupancy,
        )

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldPolymerResult:
        r"""
        Decode a full polymer chain and replace only design-masked identities.

        ``temperature`` changes only the returned probability view. Greedy and
        beam candidates are selected from the released unscaled predictions.
        Greedy decoding returns one candidate; ``topk`` is the beam width only
        when ``use_beam`` is true.
        """

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        model_input = self.prepare_design(
            input,
            instance_id=instance_id,
            design_positions=design_positions,
            filter_by_occupancy=filter_by_occupancy,
        )
        with torch.no_grad():
            output = self.model(model_input)
        return self.decode_design(
            model_input,
            output,
            topk=topk,
            use_beam=use_beam,
            temperature=temperature,
        )

    def decode_design(
        self,
        input: OInvFoldPolymerInput,
        output: OInvFoldOutput,
        *,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
    ) -> OInvFoldPolymerResult:
        r"""Decode raw polymer logits without executing the Model."""

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        log_probabilities = output.log_probabilities
        allowed = _polymer_allowed_indices(self.processor.config.modality)
        token_sequences = _decode_token_sequences(
            log_probabilities[input.design_mask],
            allowed,
            topk=topk,
            use_beam=use_beam,
        )
        probability_view = torch.softmax(output.logits / temperature, dim=-1)
        candidates = tuple(self._candidate(input, log_probabilities, tokens) for tokens in token_sequences)
        component_references = tuple(
            input.structure.system.component_reference(component_index)
            for component_index in input.component_index.detach().cpu().tolist()
        )
        return OInvFoldPolymerResult(
            target=input.structure,
            component_references=component_references,
            design_mask=input.design_mask.detach().cpu().numpy(),
            logits=_numpy(output.logits),
            log_probabilities=_numpy(log_probabilities),
            probabilities=_numpy(probability_view),
            candidates=candidates,
        )

    def _candidate(
        self,
        input: OInvFoldPolymerInput,
        log_probabilities: torch.Tensor,
        tokens: tuple[int, ...],
    ) -> OInvFoldPolymerCandidate:
        rows = torch.nonzero(input.design_mask, as_tuple=False).flatten()
        indices = torch.tensor(tokens, dtype=torch.long, device=log_probabilities.device)
        selected = log_probabilities[
            rows,
            indices,
        ]
        return OInvFoldPolymerCandidate(
            structure=_realize_polymer_structure(
                input,
                tokens,
                self.processor.config.modality,
            ),
            sequence=_polymer_candidate_sequence(
                input,
                tokens,
                self.processor.config.modality,
            ),
            total_log_probability=float(selected.sum()),
            mean_log_probability=float(selected.mean()),
        )


class OInvFoldLigandPipeline(PairedPipeline[OInvFoldLigandModel, OInvFoldLigandProcessor]):
    r"""
    Coordinate-conditioned ligand atom-element inference.
    """

    model_architecture = "oinvfold_ligand"

    def prepare_design(
        self,
        coordinates: torch.Tensor | NDArray,
        *,
        slot_ids: Sequence[str],
        context: Structure | None = None,
        vocabulary_logit_mask: torch.Tensor | NDArray | None = None,
    ) -> OInvFoldLigandInput:
        r"""Prepare provisional ligand atoms for element assignment."""

        return self.processor(
            coordinates,
            slot_ids=slot_ids,
            context=context,
            vocabulary_logit_mask=vocabulary_logit_mask,
        )

    def design(
        self,
        coordinates: torch.Tensor | NDArray,
        *,
        slot_ids: Sequence[str],
        context: Structure | None = None,
        vocabulary_logit_mask: torch.Tensor | NDArray | None = None,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
    ) -> OInvFoldLigandResult:
        r"""
        Assign element categories to ordered provisional atom coordinates.

        No ``Molecule`` or ``Structure`` is fabricated for the decoded atoms:
        bond topology remains explicitly unresolved.
        """

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        model_input = self.prepare_design(
            coordinates,
            slot_ids=slot_ids,
            context=context,
            vocabulary_logit_mask=vocabulary_logit_mask,
        )
        with torch.no_grad():
            output = self.model(model_input)
        return self.decode_design(
            model_input,
            output,
            topk=topk,
            use_beam=use_beam,
            temperature=temperature,
        )

    def prepare_design_structure(
        self,
        input: Structure | str | PathLike[str],
        *,
        atom_sites: Sequence[AtomSite] | None = None,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldLigandInput:
        r"""Prepare atom sites from one canonical ligand Structure."""

        structure = resolve_structure_input(input)
        if not isinstance(structure, Structure):
            raise TypeError("OInvFold ligand structure design requires a Structure")
        return self.processor.from_structure(
            structure,
            atom_sites=atom_sites,
            filter_by_occupancy=filter_by_occupancy,
        )

    def design_structure(
        self,
        input: Structure | str | PathLike[str],
        *,
        atom_sites: Sequence[AtomSite] | None = None,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldLigandResult:
        r"""
        Assign element categories to atom sites from one known canonical ligand.

        The input sites are lowered directly before model execution. The
        decoded result still leaves candidate bond topology
        unresolved rather than mutating the input molecular graph.
        """

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        model_input = self.prepare_design_structure(
            input,
            atom_sites=atom_sites,
            filter_by_occupancy=filter_by_occupancy,
        )
        with torch.no_grad():
            output = self.model(model_input)
        return self.decode_design(
            model_input,
            output,
            topk=topk,
            use_beam=use_beam,
            temperature=temperature,
        )

    def decode_design(
        self,
        input: OInvFoldLigandInput,
        output: OInvFoldOutput,
        *,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
    ) -> OInvFoldLigandResult:
        r"""Decode raw ligand element logits without executing the Model."""

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        log_probabilities = output.log_probabilities
        token_sequences = _decode_token_sequences(
            log_probabilities,
            tuple(range(35)),
            topk=topk,
            use_beam=use_beam,
        )
        probability_view = torch.softmax(output.logits / temperature, dim=-1)
        candidates = tuple(
            _ligand_candidate(input, log_probabilities, probability_view, tokens) for tokens in token_sequences
        )
        return OInvFoldLigandResult(
            context=input.context,
            slot_ids=input.slot_ids,
            coordinates=_numpy(input.coordinates),
            logits=_numpy(output.logits),
            log_probabilities=_numpy(log_probabilities),
            probabilities=_numpy(probability_view),
            candidates=candidates,
        )


def _decode_token_sequences(
    log_probabilities: torch.Tensor,
    allowed: tuple[int, ...],
    *,
    topk: int,
    use_beam: bool,
) -> tuple[tuple[int, ...], ...]:
    allowed_tensor = torch.tensor(
        allowed,
        dtype=torch.long,
        device=log_probabilities.device,
    )
    if not use_beam:
        relative = log_probabilities[:, allowed_tensor].argmax(dim=-1)
        return (tuple(allowed_tensor[relative].tolist()),)

    if topk <= 0:
        raise ValueError("OInvFold beam topk must be positive")
    beam_width = topk
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    host_log_probabilities = log_probabilities.detach().cpu()
    for row in host_log_probabilities:
        candidates = [(tokens + (token,), score + float(row[token])) for tokens, score in beams for token in allowed]
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        beams = candidates[:beam_width]
    return tuple(tokens for tokens, _ in beams)


def _polymer_allowed_indices(
    modality: str,
) -> tuple[int, ...]:
    if modality == "protein":
        return tuple(range(20))
    if modality == "dna":
        return (20, 21, 22, 23)
    return (20, 21, 22, 24)


def _realize_polymer_structure(
    input: OInvFoldPolymerInput,
    tokens: tuple[int, ...],
    modality: str,
) -> Structure:
    source = input.structure
    source_system = source.system
    component_indices = input.component_index.detach().cpu().tolist()
    design_mask = input.design_mask.detach().cpu().tolist()
    changed_rows: list[int] = []
    changed_components: list[int] = []
    replacements: list[ChemicalComponent] = []
    token_iterator = iter(tokens)
    for row, (component_index, design) in enumerate(zip(component_indices, design_mask, strict=True)):
        if not design:
            continue
        token = next(token_iterator)
        replacement = _component_for_token(token, modality)
        changed_rows.append(row)
        changed_components.append(component_index)
        replacements.append(replacement)
    if not changed_rows:
        return input.structure

    system, old_to_new, replacement_to_new = replace_components(
        source_system,
        torch.tensor(
            changed_components,
            dtype=torch.long,
            device=source_system.component_symbol.device,
        ),
        tuple(replacements),
    )
    atom_count = system.atom_symbol.numel()
    coordinates = torch.zeros(
        atom_count,
        3,
        dtype=source.coordinates.dtype,
        device=source.coordinates.device,
    )
    coordinate_mask = torch.zeros(
        atom_count,
        dtype=torch.bool,
        device=source.coordinate_mask.device,
    )
    occupancy = torch.zeros(
        atom_count,
        dtype=source.occupancy.dtype,
        device=source.occupancy.device,
    )
    occupancy_mask = torch.zeros(
        atom_count,
        dtype=torch.bool,
        device=source.occupancy_mask.device,
    )
    b_factor = torch.zeros(
        atom_count,
        dtype=source.b_factor.dtype,
        device=source.b_factor.device,
    )
    b_factor_mask = torch.zeros(
        atom_count,
        dtype=torch.bool,
        device=source.b_factor_mask.device,
    )
    retained = old_to_new >= 0
    retained_target = old_to_new[retained]
    coordinates[retained_target] = source.coordinates[retained]
    coordinate_mask[retained_target] = source.coordinate_mask[retained]
    occupancy[retained_target] = source.occupancy[retained]
    occupancy_mask[retained_target] = source.occupancy_mask[retained]
    b_factor[retained_target] = source.b_factor[retained]
    b_factor_mask[retained_target] = source.b_factor_mask[retained]

    replacement_offset = 0
    coordinate_atom_index = input.coordinate_atom_index.detach().cpu()
    for row, replacement in zip(
        changed_rows,
        replacements,
        strict=True,
    ):
        replacement_atom_index = {
            atom.atom_id: replacement_offset + index for index, atom in enumerate(replacement.atoms)
        }
        new_atom_ids = oinvfold_polymer_atom_ids(
            modality,
            frozenset(replacement_atom_index),
        )
        old_atoms = coordinate_atom_index[row]
        for old_atom, atom_id in zip(
            old_atoms.tolist(),
            new_atom_ids,
            strict=True,
        ):
            new_atom = int(replacement_to_new[replacement_atom_index[atom_id]])
            if new_atom < 0:
                continue
            coordinates[new_atom] = source.coordinates[old_atom]
            coordinate_mask[new_atom] = source.coordinate_mask[old_atom]
            occupancy[new_atom] = source.occupancy[old_atom]
            occupancy_mask[new_atom] = source.occupancy_mask[old_atom]
            b_factor[new_atom] = source.b_factor[old_atom]
            b_factor_mask[new_atom] = source.b_factor_mask[old_atom]
        replacement_offset += len(replacement.atoms)

    parent_source_index = torch.arange(
        source.system.component_symbol.numel(),
        dtype=torch.int64,
        device=source.component_parent_ptr.device,
    )
    parent_source_index[changed_components] = -1
    component_parent_ptr, parent_component_id = source.select_component_parents(parent_source_index)
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
        occupancy=occupancy,
        occupancy_mask=occupancy_mask,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        source_symbols=source.source_symbols,
        residue_id=source.residue_id,
        insertion_code=source.insertion_code,
        component_parent_ptr=component_parent_ptr,
        parent_component_id=parent_component_id,
        batch_size=[],
    )


def _polymer_candidate_sequence(
    input: OInvFoldPolymerInput,
    tokens: tuple[int, ...],
    modality: str,
) -> str:
    token_iterator = iter(tokens)
    component_indices = input.component_index.detach().cpu().tolist()
    design_mask = input.design_mask.detach().cpu().tolist()
    sequence_tokens = None if input.sequence_tokens is None else input.sequence_tokens.detach().cpu().tolist()
    letters = []
    for row, (component_index, design) in enumerate(zip(component_indices, design_mask, strict=True)):
        if design:
            letters.append(OINVFOLD_POLYMER_TOKENS[next(token_iterator)])
            continue
        if sequence_tokens is not None:
            letters.append(OINVFOLD_POLYMER_TOKENS[sequence_tokens[row]])
            continue
        component_id = input.structure.system.symbols[int(input.structure.system.component_symbol[component_index])]
        if modality == "protein":
            letters.append(AMINO_ACID_THREE_TO_ONE.get(component_id, "X"))
        elif modality == "dna":
            letters.append({"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U"}.get(component_id, "N"))
        else:
            letters.append(component_id if component_id in {"A", "C", "G", "T", "U"} else "N")
    return "".join(letters)


def _component_for_token(
    token: int,
    modality: str,
) -> ChemicalComponent:
    letter = OINVFOLD_POLYMER_TOKENS[token]
    if modality == "protein":
        return get_component(AMINO_ACID_ONE_TO_THREE[letter])
    if modality == "dna":
        return get_component({"A": "DA", "C": "DC", "G": "DG", "T": "DT"}[letter])
    return get_component(letter)


def _ligand_candidate(
    input: OInvFoldLigandInput,
    log_probabilities: torch.Tensor,
    probability_view: torch.Tensor,
    tokens: tuple[int, ...],
) -> OInvFoldLigandCandidate:
    indices = torch.tensor(tokens, dtype=torch.long, device=log_probabilities.device)
    rows = torch.arange(indices.shape[0], device=indices.device)
    selected_log_probabilities = log_probabilities[rows, indices]
    selected_probabilities = probability_view[rows, indices]
    return OInvFoldLigandCandidate(
        context=input.context,
        slot_ids=input.slot_ids,
        coordinates=_numpy(input.coordinates),
        elements=tuple(OINVFOLD_LIGAND_ELEMENTS[index] for index in tokens),
        per_atom_probabilities=_numpy(selected_probabilities),
        total_log_probability=float(selected_log_probabilities.sum()),
        mean_log_probability=float(selected_log_probabilities.mean()),
    )


def _numpy(tensor: torch.Tensor) -> NDArray:
    return tensor.detach().cpu().numpy()
