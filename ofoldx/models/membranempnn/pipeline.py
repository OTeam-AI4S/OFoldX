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
Membrane-conditioned sequence design, scoring, probabilities and side-chain packing.

A design samples canonical classes on ProteinMPNN's prepared rows; ``produce_design`` builds each candidate's final
target once, and the design-and-pack composite packs exactly the replaced positions of each candidate.
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
from ...data.tokenization import TOKENS
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
from ..proteinmpnn.model import ProteinMpnnInput, ProteinMpnnOutput, ProteinMpnnSamplingOutput
from ..proteinmpnn.pipeline import (
    ProteinMpnnDesignAndPackPipelineInput,
    omit_token_mask,
    residue_scopes,
    score_result,
)
from .model import MembraneMpnnModel
from .processor import MembraneMpnnProcessor


@dataclass(eq=False)
class MembraneMpnnDesignResult(DesignResult):
    r"""MembraneMPNN designs and each candidate's design record."""

    records: tuple[DesignRecord, ...] = ()


class _MembraneMpnnProbabilityOperations(PairedPipeline[MembraneMpnnModel, MembraneMpnnProcessor]):
    r"""Shared fixed-backbone probability preparation and evaluation."""

    model_architecture = "membranempnn"

    def prepare_probabilities(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""Prepare the target's classes, or an explicit one-letter sequence, for probability evaluation."""
        return self.processor.prepare_input(
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
        )

    def _probabilities(
        self,
        operation: str,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int],
        sequence: str | Mapping[str, str] | None,
        design: PositionSelection,
        conditions: Sequence[TiedResidueIdentities],
        filter_by_occupancy: bool,
        num_decoding_orders: int,
        coordinate_noise: float,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> AutoregressiveProbabilityResult:
        prepared = self.prepare_probabilities(
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=prepared.features.residue_mask.device)
        output = getattr(self.model, operation)(
            prepared.features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        return self.decode_probabilities(prepared, output)

    def autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Sequence-conditioned autoregressive probabilities over independently drawn decoding orders."""
        return self._probabilities(
            "autoregressive_probabilities",
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Sequence-free autoregressive probabilities over independently drawn decoding orders."""
        return self._probabilities(
            "sequence_free_autoregressive_probabilities",
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Target-last conditional probabilities for every row (each target decoded after all others)."""
        return self._probabilities(
            "conditional_probabilities",
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=(),
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Target-first unconditional probabilities for every row (each target decoded before all others)."""
        return self._probabilities(
            "unconditional_probabilities",
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=(),
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def decode_probabilities(
        self, prepared: PreparedInput[ProteinMpnnInput], output: ProteinMpnnOutput
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


class MembraneMpnnPipeline(_MembraneMpnnProbabilityOperations):
    r"""Design and evaluate proteins with scalar or per-residue membrane labels."""

    task = "design"

    def prepare(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""Prepare the backbone rows, labels and sequence constraints."""
        return self.processor.prepare_input(
            structure,
            membrane_labels=membrane_labels,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: ProteinMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnSamplingOutput:
        r"""Execute the prepared design operation in its native draw order."""
        generator = resolve_generator(seed, generator, device=input.residue_mask.device)
        omit_mask = omit_token_mask(global_omit_tokens, device=input.residue_mask.device)
        if omit_mask is None:
            omit_mask = torch.zeros(len(TOKENS), dtype=torch.bool, device=input.residue_mask.device)
        return self.model.design(
            input,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=omit_mask,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[ProteinMpnnInput],
        output: ProteinMpnnSamplingOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> MembraneMpnnDesignResult:
        r"""Build each candidate's final target with ``produce_design`` (definitions from ``library``, default the
        configured one) and report its untempered sequence confidence and recovery."""
        input = prepared.features
        designed = input.residue_mask & input.design_mask
        confidence = torch.exp(-sequence_nll(output.log_probabilities, output.sequence_tokens, designed))
        native = known_recovery_mask(input.sequence_tokens, designed)
        recovery = token_recovery(output.sequence_tokens, input.sequence_tokens, native)
        refs = prepared.context.structure.system.component_refs(input.component_index[0].cpu().numpy())
        native_scope = frozenset(ref for ref, keep in zip(refs, native[0].cpu().tolist()) if keep)
        scores = torch.stack((confidence, recovery), dim=-1).detach().cpu().double().numpy()
        scope, _ = residue_scopes(prepared)
        rows = designed[0].cpu().numpy()
        positions = input.component_index[0].cpu().numpy()[rows]
        classes = output.sequence_tokens.detach().cpu().numpy()[:, rows]
        library = configured_library() if library is None else library
        candidates, records = [], []
        for index in range(classes.shape[0]):
            target, record = produce_design(
                prepared, classes[index], positions=positions, table=self.processor.layout, library=library
            )
            candidates.append(
                DesignCandidate(
                    target=target,
                    scores={
                        "designed_sequence_confidence": Score(float(scores[index, 0]), None, "higher", scope),
                        "sequence_recovery": Score(float(scores[index, 1]), None, "higher", native_scope),
                    },
                )
            )
            records.append(record)
        return MembraneMpnnDesignResult(candidates=tuple(candidates), records=tuple(records))

    def design(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> MembraneMpnnDesignResult:
        r"""
        Design membrane-conditioned sequences on a fixed backbone.

        Args:
            structure: A read structure.
            membrane_labels: One label for every row, or labels of named residues (the others zero).
            design: The positions to design (default every valid protein residue).
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group (default one).
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            global_logit_bias: Global bias vector over the 72 canonical classes.
            global_omit_tokens: One-letter classes excluded from sampling.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator.
            library: The definitions replacements are built from (default the configured artifact).
        """

        prepared = self.prepare(
            structure,
            membrane_labels=membrane_labels,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_tokens=global_omit_tokens,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output, library=library)

    __call__ = task_call(design)


class MembraneMpnnScorePipeline(_MembraneMpnnProbabilityOperations):
    r"""
    Score fixed-backbone sequences with MembraneMPNN.

    Notes:
        Scores are designed and global sequence NLL in nats per residue, retaining the exact position scopes and
        per-order evaluations. The input sequence is evaluated by default; ``use_sequence=False`` selects
        sequence-free predictions. Probability operations also expose conditional and unconditional logits,
        canonical component references, and decoding ranks.
    """

    task = "score"

    def prepare(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""Prepare the target's classes, or an explicit one-letter sequence, for scoring."""
        return self.prepare_probabilities(
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: ProteinMpnnInput,
        *,
        use_sequence: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
        r"""Execute the prepared score operation in its native draw order."""
        generator = resolve_generator(seed, generator, device=input.residue_mask.device)
        return self.model.score(
            input,
            use_sequence=use_sequence,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def postprocess(self, prepared: PreparedInput[ProteinMpnnInput], output: ProteinMpnnOutput) -> ScoreResult:
        r"""Designed and global sequence NLL of the input's classes over the evaluated decoding orders."""
        return score_result(prepared, output)

    def score(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        use_sequence: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""Report scoped sequence NLL with sequence context or sequence-free predictions."""
        prepared = self.prepare(
            structure,
            membrane_labels=membrane_labels,
            sequence=sequence,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared.features,
            use_sequence=use_sequence,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(score)


class MembraneMpnnDesignAndPackPipeline(CompositePipeline):
    r"""
    Design membrane-conditioned protein identities and pack the replaced positions of every candidate.

    Notes:
        Repositories store the designer under ``design/`` and the packer under ``packer/``. Preparation retains the
        source structure, the exact design scope, and one ``filter_by_occupancy`` policy. Setting that policy to
        ``False`` keeps observed zero-occupancy atoms in both design and packing. One caller generator advances
        through sequence design and every candidate's packing.
    """

    task = "design"

    component_layout = (
        ComponentSpec("designer", "design", MembraneMpnnPipeline),
        ComponentSpec("packer", "packer", MpnnPackerPipeline),
    )

    def __init__(self, *, designer: MembraneMpnnPipeline, packer: MpnnPackerPipeline) -> None:
        super().__init__({"designer": designer, "packer": packer})

    @property
    def designer(self) -> MembraneMpnnPipeline:
        return cast(MembraneMpnnPipeline, self.components["designer"])

    @property
    def packer(self) -> MpnnPackerPipeline:
        return cast(MpnnPackerPipeline, self.components["packer"])

    def prepare(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> ProteinMpnnDesignAndPackPipelineInput:
        r"""Prepare the source structure and the exact sequence-design scope."""
        prepared = self.designer.prepare(
            structure,
            membrane_labels=membrane_labels,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            filter_by_occupancy=filter_by_occupancy,
        )
        return ProteinMpnnDesignAndPackPipelineInput(design=prepared, filter_by_occupancy=filter_by_occupancy)

    def forward(
        self,
        input: ProteinMpnnDesignAndPackPipelineInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> tuple[MembraneMpnnDesignResult, tuple[PackResult, ...]]:
        r"""Design identities, then pack each candidate's replaced positions (or every packable residue)."""
        prepared = input.design
        generator = resolve_generator(seed, generator, device=prepared.features.residue_mask.device)
        output = self.designer.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_tokens=global_omit_tokens,
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
        input: ProteinMpnnDesignAndPackPipelineInput,
        output: tuple[MembraneMpnnDesignResult, tuple[PackResult, ...]],
    ) -> DesignAndPackResult:
        r"""Assemble candidate-aligned sequence designs and side-chain predictions."""
        designs, packs = output
        return DesignAndPackResult(designs=designs, packs=packs)

    def design(
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
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
        Run membrane sequence design, then pack each candidate's replaced positions.

        Only the positions a candidate replaced are packed (a position retained as its source component, such as MSE
        sampled as ``M``, is not), unless ``repack_everything``. The ``PackResult`` tuple follows candidate order.
        ``seed`` and ``generator`` are mutually exclusive.
        """

        prepared = self.prepare(
            structure,
            membrane_labels=membrane_labels,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_tokens=global_omit_tokens,
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
