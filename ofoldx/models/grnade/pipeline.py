r"""
Decoded gRNAde inverse folding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike

import torch

from ...data.components import get_component
from ...data.molecular import ComponentReference, MolecularSystem, replace_components
from ...data.structure import Structure
from ...io import read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import DesignCandidate, DesignResult, Score
from .model import GRNADE_COMPONENT_IDS, GrnadeInput, GrnadeModel, GrnadeSamplingOutput
from .processor import GrnadeProcessor, GrnadeV032Processor


def _repartition_entities_by_sequence(
    original: MolecularSystem,
    realized: MolecularSystem,
) -> MolecularSystem:
    r"""Split original entity copies when their realized sequences diverge."""

    symbols = list(realized.symbols)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}

    def intern(symbol: str) -> int:
        index = symbol_index.get(symbol)
        if index is None:
            index = len(symbols)
            symbols.append(symbol)
            symbol_index[symbol] = index
        return index

    def component_ids(system: MolecularSystem, instance: int) -> tuple[str, ...]:
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        return tuple(system.symbols[int(symbol)] for symbol in system.component_symbol[start:stop].tolist())

    instances_by_entity: list[list[int]] = [[] for _ in range(original.entity_symbol.numel())]
    for instance, entity in enumerate(original.instance_entity.tolist()):
        instances_by_entity[entity].append(instance)

    used_ids = {original.symbols[int(symbol)] for symbol in original.entity_symbol.tolist()}
    entity_symbol: list[int] = []
    entity_type: list[int] = []
    entity_molecule_type: list[int] = []
    instance_entity = torch.empty_like(realized.instance_entity)
    for entity, instances in enumerate(instances_by_entity):
        entity_id = original.symbols[int(original.entity_symbol[entity])]
        if not instances:
            entity_symbol.append(intern(entity_id))
            entity_type.append(int(original.entity_type[entity]))
            entity_molecule_type.append(int(original.entity_molecule_type[entity]))
            continue
        original_signature = component_ids(original, instances[0])
        groups: dict[tuple[str, ...], list[int]] = {}
        for instance in instances:
            groups.setdefault(component_ids(realized, instance), []).append(instance)
        primary = original_signature if original_signature in groups else next(iter(groups))
        ordered = (primary, *(signature for signature in groups if signature != primary))
        for partition, signature in enumerate(ordered):
            if partition == 0:
                output_id = entity_id
            else:
                suffix = 2
                output_id = f"{entity_id}_{suffix}"
                while output_id in used_ids:
                    suffix += 1
                    output_id = f"{entity_id}_{suffix}"
                used_ids.add(output_id)
            output_entity = len(entity_symbol)
            entity_symbol.append(intern(output_id))
            entity_type.append(int(original.entity_type[entity]))
            entity_molecule_type.append(int(original.entity_molecule_type[entity]))
            instance_entity[groups[signature]] = output_entity

    return realized.replace(  # type: ignore[attr-defined]
        symbols=tuple(symbols),
        system_entity_ptr=realized.system_entity_ptr.new_tensor([0, len(entity_symbol)]),
        entity_symbol=realized.entity_symbol.new_tensor(entity_symbol),
        entity_type=realized.entity_type.new_tensor(entity_type),
        entity_molecule_type=realized.entity_molecule_type.new_tensor(entity_molecule_type),
        instance_entity=instance_entity,
    )


class GrnadePipeline(PairedPipeline[GrnadeModel, GrnadeProcessor]):
    r"""
    RNA inverse folding with the released gRNAde v1 graph.

    Load a published pair with [`ofoldx.pipeline`][]. Design conditions on observed
    backbone coordinates, so it takes a structure read from a file rather than a topology
    built from a sequence.
    """

    model_architecture = "grnade"

    def design(
        self,
        input: Structure | Sequence[Structure] | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        base_pairs: Sequence[tuple[ComponentReference, ComponentReference]] = (),
        use_coordinates: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        logit_bias: torch.Tensor | None = None,
        filter_by_occupancy: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Sample RNA sequences from one structural context.

        Args:
            input: One RNA ``Structure``, aligned conformations, or one local
                structure path.
            instance_id: Target RNA instance when the input is ambiguous.
            design_positions: RNA component occurrences allowed to change.
            base_pairs: Undirected secondary-structure pairs.
            use_coordinates: Whether to include three-dimensional geometry.
            num_candidates: Number of sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            logit_bias: Optional additive logits in gRNAde vocabulary order.
            filter_by_occupancy: Whether zero-occupancy coordinates are unobserved.
            seed: Optional sampling seed.
            generator: Optional caller-owned random generator.

        Returns:
            Sampled systems and their sequence scores.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            instance_id=instance_id,
            design_positions=design_positions,
            base_pairs=base_pairs,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(
            seed,
            generator,
            device=model_input.node_scalar_features.device,
        )
        sampled = self.model.sample(
            model_input,
            num_samples=num_candidates,
            temperature=temperature,
            logit_bias=logit_bias,
            generator=generator,
        )
        return _decode_design(model_input, sampled)


class GrnadeV032Pipeline(PairedPipeline[GrnadeModel, GrnadeV032Processor]):
    r"""RNA inverse folding with the released gRNAde v0.3.2 graph."""

    model_architecture = "grnade_v0_3_2"

    def design(
        self,
        input: Structure | Sequence[Structure] | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        use_coordinates: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        logit_bias: torch.Tensor | None = None,
        filter_by_occupancy: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""Sample with one released v0.3.2 conformation-count variant."""

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.processor(
            structure,
            instance_id=instance_id,
            design_positions=design_positions,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )
        sampled = self.model.sample(
            model_input,
            num_samples=num_candidates,
            temperature=temperature,
            logit_bias=logit_bias,
            generator=generator,
        )
        return _decode_design(model_input, sampled)


def _decode_design(
    input: GrnadeInput,
    output: GrnadeSamplingOutput,
) -> DesignResult:
    design = input.design_mask.unsqueeze(0).expand_as(output.sequence_tokens)
    count = design.sum(dim=-1)
    known_design = design & (input.sequence_tokens < len(GRNADE_COMPONENT_IDS))
    known_count = known_design.sum(dim=-1)

    selected_log_probability = output.log_probabilities.gather(
        -1,
        output.sequence_tokens.unsqueeze(-1),
    ).squeeze(-1)
    total_log_probability = (selected_log_probability * design).sum(dim=-1)
    # Guard the degenerate empty-scope rows (no designable / no known-native residues) so the
    # reported metrics stay finite instead of dividing 0/0 into NaN (and exp(-NaN) perplexity).
    mean_log_probability = total_log_probability / count.clamp_min(1)
    native_recovery = ((output.sequence_tokens == input.sequence_tokens) & known_design).sum(dim=-1) / known_count.clamp_min(1)

    component_index = input.component_index.detach().cpu().tolist()
    component_references = tuple(input.structure.system.component_reference(index) for index in component_index)
    host_design = input.design_mask.detach().cpu().tolist()
    scope = frozenset(
        reference for reference, selected in zip(component_references, host_design, strict=True) if selected
    )
    native_scope = frozenset(
        reference
        for reference, selected, token in zip(
            component_references,
            host_design,
            input.sequence_tokens.detach().cpu().tolist(),
            strict=True,
        )
        if selected and token < len(GRNADE_COMPONENT_IDS)
    )
    selected_component_index = input.component_index[input.design_mask].to(
        device=input.structure.system.component_symbol.device
    )
    host_tokens = output.sequence_tokens.detach().cpu().tolist()
    totals = total_log_probability.detach().cpu().tolist()
    means = mean_log_probability.detach().cpu().tolist()
    recoveries = native_recovery.detach().cpu().tolist()

    candidates = []
    for token_row, total, mean, recovery in zip(
        host_tokens,
        totals,
        means,
        recoveries,
        strict=True,
    ):
        replacements = tuple(
            get_component(GRNADE_COMPONENT_IDS[token])
            for token, selected in zip(token_row, host_design, strict=True)
            if selected
        )
        original_system = input.structure.system
        system, _, _ = replace_components(
            original_system,
            selected_component_index,
            replacements,
        )
        system = _repartition_entities_by_sequence(original_system, system)
        candidates.append(
            DesignCandidate(
                system=system,
                scores={
                    "total_log_probability": Score(
                        total,
                        unit="nat",
                        preferred="higher",
                        scope=scope,
                    ),
                    "mean_log_probability": Score(
                        mean,
                        unit="nat/residue",
                        preferred="higher",
                        scope=scope,
                    ),
                    "perplexity": Score(
                        math.exp(-mean),
                        preferred="lower",
                        scope=scope,
                    ),
                    "native_recovery": Score(
                        recovery,
                        preferred="higher",
                        scope=native_scope,
                    ),
                },
            )
        )
    return DesignResult(candidates=tuple(candidates))
