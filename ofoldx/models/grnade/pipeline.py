r"""
Decoded gRNAde inverse folding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike

import numpy as np
import torch
from numpy.typing import NDArray

from ...data.components import get_component
from ...data.molecular import ComponentReference, replace_components
from ...data.structure import Structure
from ...io import read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import DesignCandidate, DesignResult, Score
from .model import GRNADE_COMPONENT_IDS, GrnadeInput, GrnadeModel
from .processor import GrnadeProcessor


@dataclass(eq=False)
class GrnadeDesignResult(DesignResult):
    r"""
    Decoded RNA designs and their aligned raw autoregressive predictions.

    Args:
        candidates: Designed molecular systems in sampling order.
        target: Canonical source structure.
        component_references: RNA residue axis.
        design_mask: Rows whose identities were designed.
        sequence_tokens: Sampled gRNAde token IDs.
        logits: Realized autoregressive logits.
        log_probabilities: Log-softmax values aligned with ``logits``.
    """

    target: Structure
    component_references: tuple[ComponentReference, ...]
    design_mask: NDArray[np.bool_]
    sequence_tokens: NDArray
    logits: NDArray
    log_probabilities: NDArray


class GrnadePipeline(PairedPipeline[GrnadeModel, GrnadeProcessor]):
    r"""
    RNA inverse-folding Pipeline for gRNAde.
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
    ) -> GrnadeDesignResult:
        r"""
        Sample RNA identities for one fixed structural context.

        Args:
            input: One canonical RNA structure, aligned conformations of the
                same RNA sequence, or one local structure path. The first
                conformation supplies the Result target and decoded system.
            instance_id: Target RNA instance when the input is not unambiguous.
            design_positions: RNA component occurrences allowed to change.
            base_pairs: Undirected secondary-structure pairs. Crossing pairs
                express pseudoknots directly.
            use_coordinates: Whether to condition on the conformations' P,
                C4', and base-nitrogen geometry. ``False`` designs from 1D+2D
                context only.
            num_candidates: Number of complete sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            logit_bias: Optional additive logits with shape
                ``(num_residues, 4)`` in gRNAde vocabulary order.
            filter_by_occupancy: Whether zero-occupancy coordinates are unobserved.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            Decoded systems plus the aligned raw gRNAde predictions.
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
        if not bool(model_input.design_mask.any()):
            raise ValueError("gRNAde design requires at least one design position")
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
        return _decode_design(model_input, sampled.sequence_tokens, sampled.logits, sampled.log_probabilities)


def _decode_design(
    input: GrnadeInput,
    sequence_tokens: torch.Tensor,
    logits: torch.Tensor,
    log_probabilities: torch.Tensor,
) -> GrnadeDesignResult:
    selected_log_probabilities = log_probabilities.gather(
        -1,
        sequence_tokens.unsqueeze(-1),
    ).squeeze(-1)
    design = input.design_mask.unsqueeze(0).expand_as(selected_log_probabilities)
    total_log_probability = (selected_log_probabilities * design).sum(dim=-1)
    mean_log_probability = total_log_probability / design.sum(dim=-1)

    host_tokens = sequence_tokens.detach().cpu()
    host_total = total_log_probability.detach().cpu().tolist()
    host_mean = mean_log_probability.detach().cpu().tolist()
    host_design = input.design_mask.detach().cpu().tolist()
    component_index = input.component_index.detach().cpu().tolist()
    component_references = tuple(input.structure.system.component_reference(index) for index in component_index)
    designed_scope = frozenset(
        reference
        for reference, selected in zip(
            component_references,
            host_design,
            strict=True,
        )
        if selected
    )
    selected_component_index = [index for index, selected in zip(component_index, host_design, strict=True) if selected]
    selected_component_index_tensor = torch.tensor(
        selected_component_index,
        dtype=torch.long,
        device=input.structure.system.component_symbol.device,
    )
    candidates = []
    for index, token_row in enumerate(host_tokens):
        replacements = tuple(
            get_component(GRNADE_COMPONENT_IDS[int(token)])
            for token, selected in zip(
                token_row.tolist(),
                host_design,
                strict=True,
            )
            if selected
        )
        system, _, _ = replace_components(
            input.structure.system,
            selected_component_index_tensor,
            replacements,
        )
        candidates.append(
            DesignCandidate(
                system=system,
                scores={
                    "total_log_probability": Score(
                        float(host_total[index]),
                        unit="nat",
                        preferred="higher",
                        scope=designed_scope,
                    ),
                    "mean_log_probability": Score(
                        float(host_mean[index]),
                        unit="nat/residue",
                        preferred="higher",
                        scope=designed_scope,
                    ),
                },
            )
        )
    return GrnadeDesignResult(
        candidates=tuple(candidates),
        target=input.structure,
        component_references=component_references,
        design_mask=input.design_mask.detach().cpu().numpy(),
        sequence_tokens=host_tokens.numpy(),
        logits=logits.detach().cpu().numpy(),
        log_probabilities=log_probabilities.detach().cpu().numpy(),
    )
