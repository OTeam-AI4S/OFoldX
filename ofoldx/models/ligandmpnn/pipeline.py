r"""
LigandMPNN sequence design and probability Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import cast

import torch
from torch import Tensor

from ...data import ENTITY_PROTEIN, get_component
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    ComponentReference,
    MolecularSystem,
    repartition_entities_by_instance_sequence,
    replace_components,
)
from ...data.structure import Structure
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.input import resolve_structure_input
from ...results import (
    AutoregressiveProbabilityResult,
    DesignAndPackResult,
    DesignCandidate,
    DesignResult,
    Score,
    ScoreResult,
)
from ..mpnn_packer import MpnnPackerPipeline
from .model import (
    AMINO_ACID_COMPONENT_IDS,
    LigandMpnnInput,
    LigandMpnnModel,
    LigandMpnnOutput,
    LigandMpnnSamplingOutput,
)
from .processor import LigandMpnnProcessor


class LigandMpnnPipeline(PairedPipeline[LigandMpnnModel, LigandMpnnProcessor]):
    r"""
    Protein sequence design pipeline for LigandMPNN.

    Load a published pair with [`ofoldx.pipeline`][]. Every verb conditions on observed
    backbone and context coordinates, so it takes a structure read from a file rather
    than a topology built from a sequence.
    """

    model_architecture = "ligandmpnn"

    def prepare_design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnInput:
        r"""Prepare one fixed-backbone LigandMPNN design input."""

        structure = cast(Structure, resolve_structure_input(input))
        return self.processor(
            structure,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Design LigandMPNN sequences with ligand and residue context.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with complete backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            logit_bias: Global bias vector over the 20 generatable identities.
            residue_logit_bias: Per-residue bias vectors over the 20 generatable identities.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            ligand_interface_cutoff: Distance cutoff in angstroms used to mark interface residues.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            A ``DesignResult`` containing the sampled sequence candidates.

        Notes:
            Candidates follow sampling order. ``seed`` and ``generator`` are mutually exclusive.
        """

        model_input = self.prepare_design(
            input,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        output = self.model.design(
            model_input,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_design(model_input, output)

    def decode_design(
        self,
        input: LigandMpnnInput,
        output: LigandMpnnSamplingOutput,
    ) -> DesignResult:
        r"""Decode raw LigandMPNN sequence samples into designed molecular systems."""

        losses = -output.log_probabilities.gather(-1, output.sequence_tokens.unsqueeze(-1)).squeeze(-1)
        designed = (input.residue_mask & input.design_positions)[0]
        interface = designed & input.ligand_interface_mask[0].bool()
        designed_row = designed.to(dtype=losses.dtype)
        interface_row = interface.to(dtype=losses.dtype)
        designed_nll = (losses * designed_row).sum(dim=-1) / designed_row.sum(dim=-1)
        interface_nll = (losses * interface_row).sum(dim=-1) / interface_row.sum(dim=-1)
        designed_confidence = torch.exp(-designed_nll)
        interface_confidence = torch.exp(-interface_nll)
        recovery = (
            (output.sequence_tokens == input.sequence_tokens.repeat(output.sequence_tokens.shape[0], 1)).to(
                dtype=losses.dtype
            )
            * designed_row
        ).sum(dim=-1) / designed_row.sum(dim=-1)

        references = _component_references(input.structure.system, input.component_index[0])
        designed_host = designed.detach().cpu().tolist()
        interface_host = interface.detach().cpu().tolist()
        designed_scope = frozenset(
            reference
            for reference, selected in zip(references, designed_host, strict=True)
            if reference is not None and selected
        )
        interface_scope = frozenset(
            reference
            for reference, selected in zip(references, interface_host, strict=True)
            if reference is not None and selected
        )
        designed_confidence_host = designed_confidence.detach().cpu().tolist()
        interface_confidence_host = interface_confidence.detach().cpu().tolist()
        recovery_host = recovery.detach().cpu().tolist()

        candidates = tuple(
            DesignCandidate(
                system=_realize_sequence(
                    input.structure.system,
                    input.component_index[0],
                    output.sequence_tokens[index],
                    designed,
                ),
                scores={
                    "designed_sequence_confidence": Score(
                        float(designed_confidence_host[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                    **(
                        {
                            "ligand_interface_confidence": Score(
                                float(interface_confidence_host[index]),
                                preferred="higher",
                                scope=interface_scope,
                            )
                        }
                        if interface_scope
                        else {}
                    ),
                    "sequence_recovery": Score(
                        float(recovery_host[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                },
            )
            for index in range(output.sequence_tokens.shape[0])
        )
        return DesignResult(candidates=candidates)

    def prepare_score(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnInput:
        r"""Prepare one observed sequence for LigandMPNN scoring."""

        structure = cast(Structure, resolve_structure_input(input))
        return self.processor(
            structure,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
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
        Score the observed sequence over independently drawn decoding orders.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences included in the designed-sequence score.
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise.
            use_side_chain_context: Whether fixed protein side chains contribute context atoms.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            A ``ScoreResult`` containing designed and global sequence NLL scores.
        """

        model_input = self.prepare_score(
            input,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        output = self.model.score(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_score(model_input, output)

    def decode_score(
        self,
        input: LigandMpnnInput,
        output: LigandMpnnOutput,
    ) -> ScoreResult:
        r"""Decode raw evaluation logits into designed and global sequence NLL scores."""

        if not bool((input.residue_mask & input.design_positions).any()):
            raise ValueError("score requires at least one valid scoring position")
        sequence_tokens = input.sequence_tokens.repeat_interleave(output.logits.shape[0], dim=0)
        losses = -output.log_probabilities.gather(-1, sequence_tokens.unsqueeze(-1)).squeeze(-1)
        valid = input.residue_mask.repeat_interleave(output.logits.shape[0], dim=0).to(losses.dtype)
        designed = (
            (input.residue_mask & input.design_positions)
            .repeat_interleave(output.logits.shape[0], dim=0)
            .to(losses.dtype)
        )
        designed_values = (losses * designed).sum(dim=-1) / designed.sum(dim=-1)
        global_values = (losses * valid).sum(dim=-1) / valid.sum(dim=-1)
        host_designed_values = designed_values.detach().cpu()
        host_global_values = global_values.detach().cpu()
        if host_designed_values.dtype == torch.bfloat16:
            host_designed_values = host_designed_values.float()
            host_global_values = host_global_values.float()
        designed_evaluations = host_designed_values.numpy()
        global_evaluations = host_global_values.numpy()
        references = _component_references(input.structure.system, input.component_index[0])
        design_positions = input.design_positions[0].detach().cpu().tolist()
        residue_mask = input.residue_mask[0].detach().cpu().tolist()
        designed_scope = frozenset(
            reference
            for reference, selected, present in zip(references, design_positions, residue_mask, strict=True)
            if reference is not None and selected and present
        )
        global_scope = frozenset(
            reference
            for reference, present in zip(references, residue_mask, strict=True)
            if reference is not None and present
        )
        return ScoreResult(
            target=input.structure,
            scores={
                "designed_sequence_nll": Score(
                    value=float(designed_evaluations.mean()),
                    unit="nat/residue",
                    preferred="lower",
                    scope=designed_scope,
                    evaluations=designed_evaluations,
                ),
                "global_sequence_nll": Score(
                    value=float(global_evaluations.mean()),
                    unit="nat/residue",
                    preferred="lower",
                    scope=global_scope,
                    evaluations=global_evaluations,
                ),
            },
        )

    def prepare_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnInput:
        r"""Prepare one LigandMPNN probability-evaluation input."""

        structure = cast(Structure, resolve_structure_input(input))
        return self.processor(
            structure,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
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
        Compute sequence-conditioned LigandMPNN autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences treated as designable. ``None`` selects every
                valid protein occurrence.
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``AutoregressiveProbabilityResult`` with sequence-conditioned logits and decoding ranks.

        Notes:
            The leading tensor axis enumerates decoding orders, and the residue axis follows
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        model_input = self.prepare_probabilities(
            input,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        output = self.model.autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_probabilities(model_input, output)

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
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
        Compute sequence-free LigandMPNN autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences treated as designable. ``None`` selects every
                valid protein occurrence.
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``AutoregressiveProbabilityResult`` with sequence-free logits and decoding ranks.

        Notes:
            The leading tensor axis enumerates decoding orders, and the residue axis follows
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        model_input = self.prepare_probabilities(
            input,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        output = self.model.sequence_free_autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_probabilities(model_input, output)

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""
        Compute target-last conditional probabilities for every residue row.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences marked as designable in the returned carrier.
                ``None`` selects every valid protein occurrence.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent base decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``AutoregressiveProbabilityResult`` with each target decoded after all other residues.

        Notes:
            ``logits`` use decoding-order, target-residue, and vocabulary axes;
            ``decoding_ranks`` add a final residue-order axis. Residue axes follow
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        model_input = self.prepare_probabilities(
            input,
            positions=positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        output = self.model.conditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_probabilities(model_input, output)

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""
        Compute target-first unconditional probabilities for every residue row.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences marked as designable in the returned carrier.
                ``None`` selects every valid protein occurrence.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent base decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_side_chain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``AutoregressiveProbabilityResult`` with each target decoded before all other residues.

        Notes:
            ``logits`` use decoding-order, target-residue, and vocabulary axes;
            ``decoding_ranks`` add a final residue-order axis. Residue axes follow
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        model_input = self.prepare_probabilities(
            input,
            positions=positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        output = self.model.unconditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        return self.decode_probabilities(model_input, output)

    def decode_probabilities(
        self,
        input: LigandMpnnInput,
        output: LigandMpnnOutput,
    ) -> AutoregressiveProbabilityResult:
        r"""Bind raw probability logits and ranks to their canonical residue axis."""

        return AutoregressiveProbabilityResult(
            target=input.structure,
            component_references=_component_references(input.structure.system, input.component_index[0]),
            sequence_tokens=input.sequence_tokens[0],
            residue_mask=input.residue_mask[0],
            design_positions=input.design_positions[0],
            logits=output.logits,
            decoding_ranks=output.decoding_ranks,
        )


class LigandMpnnDesignAndPackPipeline(CompositePipeline):
    r"""
    Design protein sequences with LigandMPNN and pack their side chains.

    Args:
        design: LigandMPNN sequence-design pipeline.
        pack: Side-chain packing pipeline.
    """

    pair_layout = (
        CompositePair("design", "", LigandMpnnPipeline),
        CompositePair("pack", "packer", MpnnPackerPipeline),
    )

    def __init__(
        self,
        *,
        design: LigandMpnnPipeline,
        pack: MpnnPackerPipeline,
    ) -> None:
        super().__init__({"design": design, "pack": pack})

    @property
    def designer(self) -> LigandMpnnPipeline:
        return cast(LigandMpnnPipeline, self.components["design"])

    @property
    def packer(self) -> MpnnPackerPipeline:
        return cast(MpnnPackerPipeline, self.components["pack"])

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
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
    ) -> DesignAndPackResult:
        r"""
        Design LigandMPNN sequences and pack every resulting candidate.

        Args:
            input: A ``Structure`` or local structure path, parsed once for both stages.
            positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with complete backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            logit_bias: Global bias vector over the 20 generatable identities.
            residue_logit_bias: Per-residue bias vectors over the 20 generatable identities.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible during design.
            ligand_interface_cutoff: Distance cutoff in angstroms used to mark interface residues.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates during design.
            use_side_chain_context: Whether fixed protein side-chain atoms are used during design.
            use_packing_atom_context: Whether the packing stage uses nonprotein atom context.
            repack_everything: Whether to pack every eligible residue instead of only designed
                positions.
            num_packs_per_candidate: Number of packed structures generated per sequence candidate.
            num_denoising_steps: Number of side-chain denoising steps per packed structure.
            num_torsion_samples: Number of torsion proposals evaluated per denoising step.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator advanced through design before packing.

        Returns:
            The sequence-design result and its aligned side-chain packing results.

        Notes:
            The ``PackResult`` tuple follows candidate order, and each result's predictions follow
            packing-sample order. ``seed`` and ``generator`` are mutually exclusive.
        """

        structure = cast(Structure, resolve_structure_input(input))
        model_input = self.designer.prepare_design(
            structure,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        device = model_input.residue_mask.device
        generator = resolve_generator(seed, generator, device=device)
        design_output = self.designer.model.design(
            model_input,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        design_result = self.designer.decode_design(model_input, design_output)
        if repack_everything:
            pack_positions: Sequence[ComponentReference] | None = None
        else:
            component_indices = model_input.component_index[0].detach().cpu().tolist()
            references = tuple(
                model_input.structure.system.component_reference(component) for component in component_indices
            )
            selected = (model_input.residue_mask[0] & model_input.design_positions[0]).tolist()
            pack_positions = [reference for reference, keep in zip(references, selected, strict=True) if keep]
        packs = self.packer.pack_sequences(
            structure,
            sequences=[_candidate_protein_sequences(candidate.system) for candidate in design_result.candidates],
            pack_positions=pack_positions,
            use_atom_context=use_packing_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_samples=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return DesignAndPackResult(designs=design_result, packs=packs)


def _component_references(
    system: MolecularSystem,
    component_index: Tensor,
) -> tuple[ComponentReference | None, ...]:
    return tuple(
        None if component < 0 else system.component_reference(component)
        for component in component_index.detach().cpu().tolist()
    )


def _realize_sequence(
    system: MolecularSystem,
    component_index: Tensor,
    sampled_tokens: Tensor,
    design_positions: Tensor,
) -> MolecularSystem:
    r"""Realize one candidate while preserving entity and bond identity."""

    component_indices = component_index.detach().cpu().tolist()
    sampled_token_values = sampled_tokens.detach().cpu().tolist()
    selected_positions = design_positions.detach().cpu().tolist()
    selected_components = [
        component for component, selected in zip(component_indices, selected_positions, strict=True) if selected
    ]
    replacements = [
        get_component(AMINO_ACID_COMPONENT_IDS[int(token)])
        for token, selected in zip(sampled_token_values, selected_positions, strict=True)
        if selected
    ]
    realized, _, _ = replace_components(
        system,
        torch.tensor(
            selected_components,
            dtype=torch.int64,
            device=system.component_symbol.device,
        ),
        replacements,
    )
    return repartition_entities_by_instance_sequence(system, realized)


def _candidate_protein_sequences(
    system: MolecularSystem,
) -> dict[str, tuple[str, ...]]:
    r"""
    Map each protein instance to its realized component-ID sequence.

    The composition passes only complete per-instance component-ID sequences to the
    Packer; it never forwards logits, ranks, interface masks, or feature tensors.
    Candidate systems partition changed copies within each original entity exactly as
    the designer realizes them, so reading instances here follows that realization
    without a second split.
    """

    instance_entity = system.instance_entity.detach().cpu().tolist()
    entity_type = system.entity_type.detach().cpu().tolist()
    instance_symbol = system.instance_symbol.detach().cpu().tolist()
    component_ptr = system.instance_component_ptr.detach().cpu().tolist()
    component_symbol = system.component_symbol.detach().cpu().tolist()
    sequences: dict[str, tuple[str, ...]] = {}
    for instance, entity in enumerate(instance_entity):
        if entity_type[entity] != ENTITY_PROTEIN:
            continue
        instance_id = system.symbols[instance_symbol[instance]]
        sequences[instance_id] = tuple(
            system.symbols[symbol] for symbol in component_symbol[component_ptr[instance] : component_ptr[instance + 1]]
        )
    return sequences
