r"""
Decoded OInvFold polymer and ligand inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...data.components import get_component
from ...data.molecular import (
    AMINO_ACID_ONE_TO_THREE,
    AtomSite,
    ChemicalComponent,
    ComponentReference,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline
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
    OInvFoldLigandSlots,
    OInvFoldPolymerProcessor,
    _polymer_atom_ids,
)


@dataclass(eq=False)
class OInvFoldPolymerCandidate:
    r"""
    One decoded full-chain sequence and design-mask realization.

    Args:
        structure: Canonical Structure with only design-masked identities replaced.
        sequence: Full-chain decoded OInvFold sequence.
        total_log_probability: Sum of raw per-position log probabilities.
        mean_log_probability: Mean raw per-position log probability.
    """

    structure: Structure
    sequence: str
    total_log_probability: float
    mean_log_probability: float


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


@dataclass(eq=False)
class OInvFoldLigandCandidate:
    r"""
    One atom-element assignment without fabricated molecular connectivity.

    Args:
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


class OInvFoldPolymerPipeline(PairedPipeline[OInvFoldPolymerModel, OInvFoldPolymerProcessor]):
    r"""
    Full-chain protein, DNA, or RNA OInvFold inference.
    """

    model_architecture = "oinvfold"

    def design(
        self,
        input: Structure,
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
        model_input = self.processor(
            input,
            instance_id=instance_id,
            design_positions=design_positions,
            filter_by_occupancy=filter_by_occupancy,
        )
        with torch.no_grad():
            output = self.model(model_input)
        allowed = _polymer_allowed_indices(self.processor.config.modality)
        token_sequences = _decode_token_sequences(
            output,
            allowed,
            topk=topk,
            use_beam=use_beam,
        )
        probability_view = torch.softmax(output.logits / temperature, dim=-1)
        candidates = tuple(self._candidate(model_input, output, tokens) for tokens in token_sequences)
        return OInvFoldPolymerResult(
            target=input,
            component_references=model_input.component_references,
            design_mask=model_input.design_mask.detach().cpu().numpy(),
            logits=_numpy(output.logits),
            log_probabilities=_numpy(output.log_probabilities),
            probabilities=_numpy(probability_view),
            candidates=candidates,
        )

    def _candidate(
        self,
        input: OInvFoldPolymerInput,
        output: OInvFoldOutput,
        tokens: tuple[int, ...],
    ) -> OInvFoldPolymerCandidate:
        indices = torch.tensor(tokens, dtype=torch.long, device=output.logits.device)
        selected = output.log_probabilities[
            torch.arange(indices.shape[0], device=indices.device),
            indices,
        ]
        return OInvFoldPolymerCandidate(
            structure=_realize_polymer_structure(
                input,
                tokens,
                self.processor.config.modality,
            ),
            sequence="".join(OINVFOLD_POLYMER_TOKENS[index] for index in tokens),
            total_log_probability=float(selected.sum()),
            mean_log_probability=float(selected.mean()),
        )


class OInvFoldLigandPipeline(PairedPipeline[OInvFoldLigandModel, OInvFoldLigandProcessor]):
    r"""
    Coordinate-conditioned ligand atom-element inference.
    """

    model_architecture = "oinvfold_ligand"

    def design(
        self,
        input: OInvFoldLigandSlots,
        *,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
    ) -> OInvFoldLigandResult:
        r"""
        Assign element categories to ordered provisional atom slots.

        No ``Molecule`` or ``Structure`` is fabricated for the decoded atoms:
        bond topology remains explicitly unresolved.
        """

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        return self._design(
            self.processor(input),
            topk=topk,
            use_beam=use_beam,
            temperature=temperature,
        )

    def design_structure(
        self,
        input: Structure,
        *,
        atom_sites: Sequence[AtomSite] | None = None,
        topk: int = 1,
        use_beam: bool = False,
        temperature: float = 1.0,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldLigandResult:
        r"""
        Assign element categories to atom sites from one known canonical ligand.

        The input sites are converted to family-owned slots before model
        execution. The decoded result still leaves candidate bond topology
        unresolved rather than mutating the input molecular graph.
        """

        if temperature <= 0:
            raise ValueError("OInvFold temperature must be positive")
        slots = self.processor.slots_from_structure(
            input,
            atom_sites=atom_sites,
            filter_by_occupancy=filter_by_occupancy,
        )
        return self._design(
            self.processor(slots),
            topk=topk,
            use_beam=use_beam,
            temperature=temperature,
        )

    def _design(
        self,
        model_input: OInvFoldLigandInput,
        *,
        topk: int,
        use_beam: bool,
        temperature: float,
    ) -> OInvFoldLigandResult:
        with torch.no_grad():
            output = self.model(model_input)
        allowed = tuple(range(35))
        token_sequences = _decode_token_sequences(
            output,
            allowed,
            topk=topk,
            use_beam=use_beam,
        )
        probability_view = torch.softmax(output.logits / temperature, dim=-1)
        candidates = tuple(
            _ligand_candidate(
                model_input,
                output,
                probability_view,
                tokens,
            )
            for tokens in token_sequences
        )
        return OInvFoldLigandResult(
            context=model_input.context,
            slot_ids=model_input.slot_ids,
            coordinates=_numpy(model_input.coordinates),
            logits=_numpy(output.logits),
            log_probabilities=_numpy(output.log_probabilities),
            probabilities=_numpy(probability_view),
            candidates=candidates,
        )


def _decode_token_sequences(
    output: OInvFoldOutput,
    allowed: tuple[int, ...],
    *,
    topk: int,
    use_beam: bool,
) -> tuple[tuple[int, ...], ...]:
    allowed_tensor = torch.tensor(
        allowed,
        dtype=torch.long,
        device=output.logits.device,
    )
    if not use_beam:
        relative = output.logits[:, allowed_tensor].argmax(dim=-1)
        return (tuple(allowed_tensor[relative].tolist()),)

    if topk <= 0:
        raise ValueError("OInvFold beam topk must be positive")
    beam_width = topk
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    log_probabilities = output.log_probabilities.detach().float().cpu()
    for row in log_probabilities:
        candidates = [(tokens + (token,), score + float(row[token])) for tokens, score in beams for token in allowed]
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        beams = candidates[:beam_width]
    return tuple(tokens for tokens, _ in beams)


def _polymer_allowed_indices(
    modality: Literal["protein", "dna", "rna"],
) -> tuple[int, ...]:
    if modality == "protein":
        return tuple(range(20))
    if modality == "dna":
        return (20, 21, 22, 23)
    return (20, 21, 22, 24)


def _realize_polymer_structure(
    input: OInvFoldPolymerInput,
    tokens: tuple[int, ...],
    modality: Literal["protein", "dna", "rna"],
) -> Structure:
    replacements = {
        reference: _component_for_token(token, modality)
        for reference, token, design in zip(
            input.component_references,
            tokens,
            input.design_mask.tolist(),
            strict=True,
        )
        if design
    }
    changed = {
        reference
        for reference, component in replacements.items()
        if input.structure.system.component(reference).component_id != component.component_id
    }
    if not changed:
        return input.structure

    system = input.structure.system.replace_components(replacements)
    source_index = {site: index for index, site in enumerate(input.structure.sites)}
    sites = [site for site in input.structure.sites if (site.instance_id, site.component_index) not in changed]
    source_rows = [
        index
        for index, site in enumerate(input.structure.sites)
        if (site.instance_id, site.component_index) not in changed
    ]
    coordinates = [input.structure.coordinates[index] for index in source_rows]
    coordinate_mask = [input.structure.coordinate_mask[index] for index in source_rows]
    occupancy = (
        None if input.structure.occupancy is None else [input.structure.occupancy[index] for index in source_rows]
    )
    b_factor = None if input.structure.b_factor is None else [input.structure.b_factor[index] for index in source_rows]

    for reference, old_slots in zip(
        input.component_references,
        input.coordinate_slots,
        strict=True,
    ):
        if reference not in changed:
            continue
        component = replacements[reference]
        new_atom_ids = _polymer_atom_ids(modality, component)
        for old_site, atom_id in zip(old_slots, new_atom_ids, strict=True):
            index = source_index[old_site]
            sites.append(AtomSite(reference[0], reference[1], atom_id))
            coordinates.append(input.structure.coordinates[index])
            coordinate_mask.append(input.structure.coordinate_mask[index])
            if occupancy is not None and input.structure.occupancy is not None:
                occupancy.append(input.structure.occupancy[index])
            if b_factor is not None and input.structure.b_factor is not None:
                b_factor.append(input.structure.b_factor[index])

    return Structure(
        system,
        tuple(sites),
        np.asarray(coordinates, dtype=input.structure.coordinates.dtype),
        np.asarray(coordinate_mask, dtype=np.bool_),
        None if occupancy is None else np.asarray(occupancy),
        None if b_factor is None else np.asarray(b_factor),
        input.structure.residue_numbers,
    )


def _component_for_token(
    token: int,
    modality: Literal["protein", "dna", "rna"],
) -> ChemicalComponent:
    letter = OINVFOLD_POLYMER_TOKENS[token]
    if modality == "protein":
        return get_component(AMINO_ACID_ONE_TO_THREE[letter])
    if modality == "dna":
        return get_component({"A": "DA", "C": "DC", "G": "DG", "T": "DT"}[letter])
    return get_component(letter)


def _ligand_candidate(
    input: OInvFoldLigandInput,
    output: OInvFoldOutput,
    probability_view: torch.Tensor,
    tokens: tuple[int, ...],
) -> OInvFoldLigandCandidate:
    indices = torch.tensor(tokens, dtype=torch.long, device=output.logits.device)
    rows = torch.arange(indices.shape[0], device=indices.device)
    selected_log_probabilities = output.log_probabilities[rows, indices]
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
    return tensor.detach().float().cpu().numpy()
