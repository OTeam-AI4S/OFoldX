# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
OInvFold polymer and ligand inverse-folding Pipelines.

A polymer design decodes canonical classes on the prepared rows (greedy, or beam over the modality's classes);
``produce_design`` builds each candidate's final target once. Ligand element assignment never fabricates a
molecule: its candidates keep slot identifiers, coordinates and elements, with bonds unresolved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...errors import RequestRef, fail
from ...pipelines import PairedPipeline
from ...pipelines.base import task_call
from ...results import DesignCandidate, DesignResult, Score, floating_numpy
from ..design import DesignRecord, PositionSelection, produce_design
from ..preparation import PreparedInput
from .model import (
    OINVFOLD_LIGAND_ELEMENTS,
    OInvFoldLigandInput,
    OInvFoldLigandModel,
    OInvFoldOutput,
    OInvFoldPolymerInput,
    OInvFoldPolymerModel,
)
from .processor import OINVFOLD_DESIGN_CLASSES, OInvFoldLigandProcessor, OInvFoldPolymerProcessor


@dataclass(eq=False)
class OInvFoldPolymerResult(DesignResult):
    r"""
    Raw per-row predictions over the 72 canonical classes and the candidates in rank order, each with its design record.

    Attributes:
        components: The component of each row.
        design_mask: Whether each row is designed.
        logits: Raw logits (``-inf`` outside the released output columns).
        log_probabilities: Raw log probabilities.
        probabilities: The temperature view of the probabilities.
        records: Per candidate, what ``produce_design`` did.
    """

    components: tuple[ComponentRef, ...]
    design_mask: NDArray[np.bool_]
    logits: NDArray
    log_probabilities: NDArray
    probabilities: NDArray
    records: tuple[DesignRecord, ...] = ()

    @property
    def best(self) -> DesignCandidate:
        r"""The highest-ranked greedy or beam candidate."""
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
    Full-chain protein, DNA, or RNA OInvFold inverse folding on observed backbone coordinates.
    """

    task = "design"

    model_architecture = "oinvfold"

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[OInvFoldPolymerInput]:
        r"""
        Prepare the polymer instance holding the designed positions.

        Notes:
            Preparation selects one polymer instance and requires observed N/CA/C/O atoms for protein, or seven backbone
            atoms and glycosidic N9 (otherwise N1) for nucleic acids. Rows retain their canonical classes while reading
            the
            released output columns. Both nucleic types' A, C and G use shared columns; inosine reads A, and T/U use the
            modality's pyrimidine column. The class axis has 72 canonical classes; logits outside the released 25
            columns
            are -inf. Training crops retain every designed row unless all rows are designed.

            Frames require noncoincident, noncollinear points. Virtual global nodes use the centroid and declared
            Cartesian
            axes rather than an inferred unique molecular orientation. Global edges carry actual relative rotations and
            translations. The released polymer indexing's constant identity/zero global features are not reproduced, so
            checkpoint predictions differ and need separate quality evaluation.
        """
        return self.processor.prepare_input(structure, design=design, filter_by_occupancy=filter_by_occupancy)

    def forward(self, input: OInvFoldPolymerInput) -> OInvFoldOutput:
        r"""Evaluate the class logits of the prepared rows."""
        with torch.no_grad():
            return self.model(input)

    def design(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        num_beams: int = 1,
        use_beam: bool = False,
        probability_temperature: float = 1.0,
        filter_by_occupancy: bool = True,
        library: ComponentLibrary | None = None,
    ) -> OInvFoldPolymerResult:
        r"""
        Decode a polymer chain and build each candidate's target with ``produce_design``.

        ``probability_temperature`` changes only the returned probability view; greedy and beam candidates are
        selected from the unscaled log probabilities. Greedy decoding returns one candidate; ``num_beams`` is the beam
        width only when ``use_beam`` is true. Definitions come from ``library`` (default the configured one).
        """

        prepared = self.prepare(structure, design=design, filter_by_occupancy=filter_by_occupancy)
        return self.postprocess(
            prepared,
            self.forward(prepared.features),
            num_beams=num_beams,
            use_beam=use_beam,
            probability_temperature=probability_temperature,
            library=library,
        )

    def postprocess(
        self,
        prepared: PreparedInput[OInvFoldPolymerInput],
        output: OInvFoldOutput,
        *,
        num_beams: int = 1,
        use_beam: bool = False,
        probability_temperature: float = 1.0,
        library: ComponentLibrary | None = None,
    ) -> OInvFoldPolymerResult:
        r"""
        Decode raw polymer logits without executing the Model.

        Notes:
            A decoded class representing the source occurrence retains that occurrence: MSE predicted as M stays MSE.
            Other designed positions are replaced by the class's standard component with backbone guide coordinates
            retained.
            The loss refuses scored classes outside the predicted vocabulary.

            Greedy decoding returns one candidate. Beam decoding uses ``num_beams`` as a width and returns only
            supported candidates; excluded classes do not pad the beam. Selection uses unscaled logits and raw log
            probabilities. ``probability_temperature`` changes the returned probability view only. Candidate scores
            include total and mean raw log probabilities; raw results retain logits, log probabilities, and the
            temperature-view probabilities.
        """

        _require_temperature(probability_temperature)
        input = prepared.features
        log_probabilities = output.log_probabilities
        rows = torch.nonzero(input.design_mask, as_tuple=False).flatten()
        sequences = _decode_token_sequences(
            log_probabilities[rows],
            OINVFOLD_DESIGN_CLASSES[self.processor.config.modality],
            num_beams=num_beams,
            use_beam=use_beam,
        )
        system = prepared.context.structure.system
        positions = input.component_index[rows].cpu().numpy()
        scope = frozenset(system.component_refs(positions))
        library = configured_library() if library is None else library
        candidates, records = [], []
        for tokens in sequences:
            selected = log_probabilities[rows, torch.tensor(tokens, device=rows.device)]
            target, record = produce_design(
                prepared,
                np.array(tokens, dtype=np.int64),
                positions=positions,
                table=self.processor.layout,
                library=library,
            )
            scores = {
                "total_log_probability": Score(float(selected.sum()), "nat", "higher", scope),
                "mean_log_probability": Score(float(selected.mean()), "nat/residue", "higher", scope),
            }
            candidates.append(DesignCandidate(target, scores))
            records.append(record)
        return OInvFoldPolymerResult(
            candidates=tuple(candidates),
            components=system.component_refs(input.component_index.cpu().numpy()),
            design_mask=input.design_mask.cpu().numpy(),
            logits=floating_numpy(output.logits),
            log_probabilities=floating_numpy(log_probabilities),
            probabilities=floating_numpy(torch.softmax(output.logits / probability_temperature, dim=-1)),
            records=tuple(records),
        )

    __call__ = task_call(design)


class OInvFoldLigandPipeline(PairedPipeline[OInvFoldLigandModel, OInvFoldLigandProcessor]):
    r"""
    Coordinate-conditioned ligand atom-element inference.
    """

    task = "design"

    model_architecture = "oinvfold_ligand"

    def prepare(
        self,
        input: torch.Tensor | NDArray,
        *,
        slot_ids: Sequence[str],
        context: Structure | None = None,
        slot_logit_bias: torch.Tensor | NDArray | None = None,
    ) -> OInvFoldLigandInput:
        r"""
        Prepare provisional ligand atoms for element assignment.

        Notes:
            Frames require noncoincident, noncollinear defining points. Virtual global nodes use the coordinate centroid
            and declared Cartesian axes, which do not assert a uniquely inferred molecular orientation. Global edges
            retain
            actual relative rotations and translations. Element assignments leave bonds, formal charge and canonical
            component
            identity unresolved.
        """

        return self.processor(
            input,
            slot_ids=slot_ids,
            context=context,
            slot_logit_bias=slot_logit_bias,
        )

    def forward(self, input: OInvFoldLigandInput) -> OInvFoldOutput:
        r"""Evaluate element or residue logits for the prepared structure."""

        with torch.no_grad():
            return self.model(input)

    def design(
        self,
        coordinates: torch.Tensor | NDArray,
        *,
        slot_ids: Sequence[str],
        context: Structure | None = None,
        slot_logit_bias: torch.Tensor | NDArray | None = None,
        num_beams: int = 1,
        use_beam: bool = False,
        probability_temperature: float = 1.0,
    ) -> OInvFoldLigandResult:
        r"""
        Assign element categories to ordered provisional atom coordinates.

        No ``Molecule`` or ``Structure`` is fabricated for the decoded atoms:
        bond topology remains explicitly unresolved.
        """

        model_input = self.prepare(
            coordinates,
            slot_ids=slot_ids,
            context=context,
            slot_logit_bias=slot_logit_bias,
        )
        output = self.forward(model_input)
        return self.postprocess(
            model_input,
            output,
            num_beams=num_beams,
            use_beam=use_beam,
            probability_temperature=probability_temperature,
        )

    def prepare_from_structure(
        self,
        structure: Structure,
        *,
        instance: str | None = None,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[OInvFoldLigandInput]:
        r"""Prepare the observed atoms of one ligand instance of a structure."""
        return self.processor.prepare_input(structure, instance=instance, filter_by_occupancy=filter_by_occupancy)

    def design_from_structure(
        self,
        structure: Structure,
        *,
        instance: str | None = None,
        num_beams: int = 1,
        use_beam: bool = False,
        probability_temperature: float = 1.0,
        filter_by_occupancy: bool = True,
    ) -> OInvFoldLigandResult:
        r"""
        Assign element categories to the observed atoms of one ligand instance.

        The result leaves candidate bond topology unresolved rather than editing the input molecular graph.
        """

        prepared = self.prepare_from_structure(structure, instance=instance, filter_by_occupancy=filter_by_occupancy)
        return self.postprocess(
            prepared.features,
            self.forward(prepared.features),
            num_beams=num_beams,
            use_beam=use_beam,
            probability_temperature=probability_temperature,
        )

    def postprocess(
        self,
        input: OInvFoldLigandInput,
        output: OInvFoldOutput,
        *,
        num_beams: int = 1,
        use_beam: bool = False,
        probability_temperature: float = 1.0,
    ) -> OInvFoldLigandResult:
        r"""
        Decode raw ligand element logits without executing the Model.

        Notes:
            Greedy decoding returns one candidate. Beam decoding uses ``num_beams`` as a width and returns only
            supported candidates; excluded classes do not pad the beam. Selection uses unscaled logits and raw log
            probabilities. ``probability_temperature`` changes the returned probability view only. Candidate scores
            include total and mean raw log probabilities; raw results retain logits, log probabilities, and the
            temperature-view probabilities.
        """

        _require_temperature(probability_temperature)
        log_probabilities = output.log_probabilities
        token_sequences = _decode_token_sequences(
            log_probabilities,
            tuple(range(35)),
            num_beams=num_beams,
            use_beam=use_beam,
        )
        probability_view = torch.softmax(output.logits / probability_temperature, dim=-1)
        candidates = tuple(
            _ligand_candidate(input, log_probabilities, probability_view, tokens) for tokens in token_sequences
        )
        return OInvFoldLigandResult(
            context=input.context,
            slot_ids=input.slot_ids,
            coordinates=floating_numpy(input.coordinates),
            logits=floating_numpy(output.logits),
            log_probabilities=floating_numpy(log_probabilities),
            probabilities=floating_numpy(probability_view),
            candidates=candidates,
        )

    __call__ = task_call(design)


def _require_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature <= 0:
        fail("value_invalid", RequestRef("probability_temperature", str(temperature)))


def _decode_token_sequences(
    log_probabilities: torch.Tensor,
    allowed: tuple[int, ...],
    *,
    num_beams: int,
    use_beam: bool,
) -> tuple[tuple[int, ...], ...]:
    allowed_tensor = torch.tensor(
        allowed,
        dtype=torch.long,
        device=log_probabilities.device,
    )
    allowed_values = log_probabilities[:, allowed_tensor]
    if not bool(torch.isfinite(allowed_values).any(dim=-1).all()):
        fail("value_invalid", RequestRef("output", "a decoded row without a finite allowed class"))
    if not use_beam:
        relative = allowed_values.argmax(dim=-1)
        return (tuple(allowed_tensor[relative].tolist()),)

    if num_beams <= 0:
        fail("value_invalid", RequestRef("num_beams", str(num_beams)))
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    host_log_probabilities = log_probabilities.detach().cpu()
    for row in host_log_probabilities:
        candidates = [
            (tokens + (token,), score + float(row[token]))
            for tokens, score in beams
            for token in allowed
            if bool(torch.isfinite(row[token]))
        ]
        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        beams = candidates[:num_beams]
    return tuple(tokens for tokens, _ in beams)


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
        coordinates=floating_numpy(input.coordinates),
        elements=tuple(OINVFOLD_LIGAND_ELEMENTS[index] for index in tokens),
        per_atom_probabilities=floating_numpy(selected_probabilities),
        total_log_probability=float(selected_log_probabilities.sum()),
        mean_log_probability=float(selected_log_probabilities.mean()),
    )
