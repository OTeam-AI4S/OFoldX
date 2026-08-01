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
LigandMPNN sequence design, scoring and probability Pipelines.

A design samples canonical classes on the prepared rows; ``produce_design`` builds each candidate's final target once,
and the design-and-pack composite packs exactly the replaced positions of each candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...errors import RequestRef, fail
from ...metrics.sequence import known_recovery_mask, sequence_nll, token_recovery
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import (
    AutoregressiveProbabilityResult,
    DesignAndPackResult,
    DesignCandidate,
    DesignResult,
    PackResult,
    Score,
    ScoreResult,
)
from ...results.probabilities import bind_autoregressive_probabilities
from ..conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..design import DesignRecord, PositionSelection, produce_design
from ..mpnn_packer import MpnnPackerPipeline
from ..preparation import PreparedInput
from ..processing import model_log_probabilities
from .model import LigandMpnnInput, LigandMpnnModel, LigandMpnnOutput, LigandMpnnSamplingOutput
from .processor import LigandMpnnProcessor


@dataclass(eq=False)
class LigandMpnnDesignResult(DesignResult):
    r"""LigandMPNN designs and each candidate's design record."""

    records: tuple[DesignRecord, ...] = ()


def _scope(prepared: PreparedInput[LigandMpnnInput], rows: Tensor) -> frozenset[ComponentRef]:
    r"""The components of the selected rows of a prepared input."""
    refs = prepared.context.structure.system.component_refs(prepared.features.component_index[0].cpu().numpy())
    return frozenset(ref for ref, keep in zip(refs, rows.cpu().tolist()) if keep)


class _LigandMpnnProbabilityOperations(PairedPipeline[LigandMpnnModel, LigandMpnnProcessor]):
    r"""Shared fixed-backbone probability preparation and evaluation."""

    model_architecture = "ligandmpnn"

    def prepare_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[LigandMpnnInput]:
        r"""Prepare one LigandMPNN probability-evaluation input."""
        return self.processor.prepare_input(
            structure,
            design=design,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )

    def _probabilities(
        self,
        operation: str,
        structure: Structure,
        *,
        design: PositionSelection,
        conditions: Sequence[TiedResidueIdentities],
        use_atom_context: bool,
        filter_by_occupancy: bool,
        num_decoding_orders: int,
        coordinate_noise: float,
        use_side_chain_context: bool,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> AutoregressiveProbabilityResult:
        prepared = self.prepare_probabilities(
            structure,
            design=design,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=prepared.features.backbone_coordinates.device)
        output = getattr(self.model, operation)(
            prepared.features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_probabilities(prepared, output)

    def autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""
        Sequence-conditioned autoregressive probabilities over independently drawn decoding orders.

        Args:
            structure: A read structure.
            design: The rows treated as designable (default every valid row).
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether the nearest non-protein, non-water heavy atoms are visible to the model.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator.
        """

        return self._probabilities(
            "autoregressive_probabilities",
            structure,
            design=design,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            seed=seed,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Sequence-free autoregressive probabilities over independently drawn decoding orders."""
        return self._probabilities(
            "sequence_free_autoregressive_probabilities",
            structure,
            design=design,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            seed=seed,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Target-last conditional probabilities for every row (each target decoded after all others)."""
        return self._probabilities(
            "conditional_probabilities",
            structure,
            design=design,
            conditions=(),
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            seed=seed,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Target-first unconditional probabilities for every row (each target decoded before all others)."""
        return self._probabilities(
            "unconditional_probabilities",
            structure,
            design=design,
            conditions=(),
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            seed=seed,
            generator=generator,
        )

    def decode_probabilities(
        self, prepared: PreparedInput[LigandMpnnInput], output: LigandMpnnOutput
    ) -> AutoregressiveProbabilityResult:
        r"""Bind raw probability logits and ranks to their canonical residue axis."""
        input = prepared.features
        return bind_autoregressive_probabilities(
            prepared.context.structure,
            input.component_index,
            input.sequence_tokens,
            input.residue_mask,
            input.design_mask,
            output.logits,
            output.decoding_ranks,
        )


class LigandMpnnPipeline(_LigandMpnnProbabilityOperations):
    r"""
    Protein sequence design pipeline for LigandMPNN.

    Every verb conditions on observed backbone and context coordinates, so it takes a read structure.
    """

    task = "design"

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[LigandMpnnInput]:
        r"""Prepare one fixed-backbone LigandMPNN design input."""
        return self.processor.prepare_input(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: LigandMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnSamplingOutput:
        r"""Execute the prepared design operation in its native draw order."""
        generator = resolve_generator(seed, generator, device=input.backbone_coordinates.device)
        return self.model.design(
            input,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[LigandMpnnInput],
        output: LigandMpnnSamplingOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> LigandMpnnDesignResult:
        r"""Build each candidate's final target with ``produce_design`` (definitions from ``library``, default the
        configured one) and score it."""
        input = prepared.features
        designed = (input.residue_mask & input.design_mask)[0]
        interface = designed & input.ligand_interface_mask[0].bool()
        designed_nll = sequence_nll(output.log_probabilities, output.sequence_tokens, designed)
        interface_nll = sequence_nll(output.log_probabilities, output.sequence_tokens, interface)
        native = known_recovery_mask(input.sequence_tokens[0], designed)
        recovery = token_recovery(output.sequence_tokens, input.sequence_tokens, native)
        native_scope = _scope(prepared, native)
        scores = torch.stack(
            (designed_nll, (-designed_nll).exp(), interface_nll, (-interface_nll).exp(), recovery), dim=-1
        )
        scores = scores.detach().cpu().double().numpy()
        designed_scope, interface_scope = _scope(prepared, designed), _scope(prepared, interface)
        rows = designed.cpu().numpy()
        positions = input.component_index[0].cpu().numpy()[rows]
        classes = output.sequence_tokens.detach().cpu().numpy()[:, rows]
        library = configured_library() if library is None else library
        candidates, records = [], []
        for index in range(classes.shape[0]):
            target, record = produce_design(
                prepared, classes[index], positions=positions, table=self.processor.layout, library=library
            )
            nll, confidence, interface_nll_value, interface_confidence, recovery_value = scores[index].tolist()
            values = {
                "designed_sequence_nll": Score(nll, "nat/residue", "lower", designed_scope),
                "designed_sequence_confidence": Score(confidence, None, "higher", designed_scope),
                "sequence_recovery": Score(recovery_value, None, "higher", native_scope),
            }
            if interface_scope:
                values["ligand_interface_nll"] = Score(interface_nll_value, "nat/residue", "lower", interface_scope)
                values["ligand_interface_confidence"] = Score(interface_confidence, None, "higher", interface_scope)
            candidates.append(DesignCandidate(target=target, scores=values))
            records.append(record)
        return LigandMpnnDesignResult(candidates=tuple(candidates), records=tuple(records))

    def design(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> LigandMpnnDesignResult:
        r"""
        Design LigandMPNN sequences with ligand and residue context.

        Args:
            structure: A read structure.
            design: The positions to design (default every valid protein row).
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group (default one).
            global_logit_bias: Bias vector over the 72 canonical classes added at every row.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            use_atom_context: Whether the nearest non-protein, non-water heavy atoms are visible to the model.
            ligand_interface_cutoff: Distance in ångström below which a row is at the ligand interface.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator.
            library: The definitions replacements are built from (default the configured artifact).

        Returns:
            Candidates in sampling order, each the final target of its design (MSE sampled as ``M`` stays MSE), and
            each candidate's design record.
        """

        prepared = self.prepare(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output, library=library)

    __call__ = task_call(design)


class LigandMpnnScorePipeline(_LigandMpnnProbabilityOperations):
    r"""Score fixed-backbone sequences with LigandMPNN."""

    task = "score"

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[LigandMpnnInput]:
        r"""Prepare the target's classes for scoring."""
        return self.prepare_probabilities(
            structure,
            design=design,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: LigandMpnnInput,
        *,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnOutput:
        r"""Execute the prepared score operation in its native draw order."""
        generator = resolve_generator(seed, generator, device=input.backbone_coordinates.device)
        return self.model.score(
            input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def postprocess(self, prepared: PreparedInput[LigandMpnnInput], output: LigandMpnnOutput) -> ScoreResult:
        r"""Designed and global sequence NLL of the input's classes over the evaluated decoding orders."""
        input = prepared.features
        designed = input.residue_mask & input.design_mask
        if not bool(designed.any()):
            fail("value_invalid", RequestRef("design", "no valid scoring position"))
        tokens = input.sequence_tokens.repeat_interleave(output.logits.shape[0], dim=0)
        log_probabilities = model_log_probabilities(output.logits, input.residue_mask)
        designed_values = sequence_nll(log_probabilities, tokens, designed).detach().cpu().double().numpy()
        global_values = sequence_nll(log_probabilities, tokens, input.residue_mask).detach().cpu().double().numpy()
        return ScoreResult(
            target=prepared.context.structure,
            scores={
                "designed_sequence_nll": Score(
                    float(designed_values.mean()),
                    "nat/residue",
                    "lower",
                    _scope(prepared, designed[0]),
                    designed_values,
                ),
                "global_sequence_nll": Score(
                    float(global_values.mean()),
                    "nat/residue",
                    "lower",
                    _scope(prepared, input.residue_mask[0]),
                    global_values,
                ),
            },
        )

    def score(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score the target's classes over independently drawn decoding orders.

        Args:
            structure: A read structure.
            design: Rows included in the designed-sequence score (default every valid row).
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether the nearest non-protein, non-water heavy atoms are visible.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator.
        """

        prepared = self.prepare(
            structure,
            design=design,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared.features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(score)


@dataclass(eq=False)
class LigandMpnnDesignAndPackPipelineInput:
    r"""The prepared design input and the observation policy shared with packing."""

    design: PreparedInput[LigandMpnnInput]
    filter_by_occupancy: bool


class LigandMpnnDesignAndPackPipeline(CompositePipeline):
    r"""
    Design protein sequences with LigandMPNN and pack the replaced positions of every candidate.

    Args:
        designer: LigandMPNN sequence-design pipeline.
        packer: Side-chain packing pipeline.

    Notes:
        Repositories store the designer under ``design/`` and the packer under ``packer/``. Preparation retains the
        source structure, the exact design scope, and one ``filter_by_occupancy`` policy. Setting that policy to
        ``False`` keeps observed zero-occupancy atoms in both design and packing. One caller generator advances
        through sequence design and every candidate's packing.
    """

    task = "design"

    component_layout = (
        ComponentSpec("designer", "design", LigandMpnnPipeline),
        ComponentSpec("packer", "packer", MpnnPackerPipeline),
    )

    def __init__(self, *, designer: LigandMpnnPipeline, packer: MpnnPackerPipeline) -> None:
        super().__init__({"designer": designer, "packer": packer})

    @property
    def designer(self) -> LigandMpnnPipeline:
        return cast(LigandMpnnPipeline, self.components["designer"])

    @property
    def packer(self) -> MpnnPackerPipeline:
        return cast(MpnnPackerPipeline, self.components["packer"])

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnDesignAndPackPipelineInput:
        r"""Prepare the source structure and the exact sequence-design scope."""
        prepared = self.designer.prepare(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        return LigandMpnnDesignAndPackPipelineInput(design=prepared, filter_by_occupancy=filter_by_occupancy)

    def forward(
        self,
        input: LigandMpnnDesignAndPackPipelineInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> tuple[LigandMpnnDesignResult, tuple[PackResult, ...]]:
        r"""Design identities, then pack each candidate's replaced positions (or every packable residue)."""
        prepared = input.design
        generator = resolve_generator(seed, generator, device=prepared.features.residue_mask.device)
        output = self.designer.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        designs = self.designer.postprocess(prepared, output, library=library)
        packs = tuple(
            self.packer.pack(
                candidate.target,
                positions=PositionSelection() if repack_everything else PositionSelection(indices=record.replaced()),
                use_atom_context=use_packing_atom_context,
                filter_by_occupancy=input.filter_by_occupancy,
                num_samples=num_packs_per_candidate,
                num_denoising_steps=num_denoising_steps,
                num_torsion_samples=num_torsion_samples,
                generator=generator,
            )
            for candidate, record in zip(designs.candidates, designs.records, strict=True)
        )
        return designs, packs

    def postprocess(
        self,
        input: LigandMpnnDesignAndPackPipelineInput,
        output: tuple[LigandMpnnDesignResult, tuple[PackResult, ...]],
    ) -> DesignAndPackResult:
        r"""Assemble candidate-aligned sequence designs and side-chain predictions."""
        designs, packs = output
        return DesignAndPackResult(designs=designs, packs=packs)

    def design(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> DesignAndPackResult:
        r"""
        Design LigandMPNN sequences and pack every resulting candidate.

        Only the positions a candidate replaced are packed (a position retained as its source component, such as MSE
        sampled as ``M``, is not), unless ``repack_everything``. The ``PackResult`` tuple follows candidate order.
        ``seed`` and ``generator`` are mutually exclusive.
        """

        prepared = self.prepare(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            use_packing_atom_context=use_packing_atom_context,
            repack_everything=repack_everything,
            num_packs_per_candidate=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            seed=seed,
            generator=generator,
            library=library,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(design)
