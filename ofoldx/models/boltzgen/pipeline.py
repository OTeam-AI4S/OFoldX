r"""OFoldX's biological operation boundary for the released BoltzGen design pipeline.

``BoltzGenPipeline`` binds the two family Model/Processor pairs and the composed boltz2
fold / affinity stages into the released multi-stage flow
(``design -> inverse_folding -> folding -> affinity -> analysis/filter``, upstream
``cli/boltzgen.py`` ``step_names``) and returns the family-owned ``BoltzGenResult``. It is
not a base class and shares no runtime request object with any other family (mirroring
``Boltz1Pipeline``).

§4 composition (read-only): the folding and affinity stages compose with the boltz2 family
by read-only import (locked: reference, do not edit boltz2). ``fold`` / ``affinity`` and
their Processors are **injected optional dependencies** typed against boltz2 names; BoltzGen
never adds a verb to boltz2.

FLAG (composition, partially resolved): the earlier BLOCKING flag was that boltz2 exposed no
fold verb. boltz2 has since landed ``Boltz2Model.fold`` + ``Boltz2Processor`` (generic fold
pair) in parallel, so the folding stage is now wireable by injecting them. ``do_refold=True``
without an injected fold Model+Processor still raises a clear error.

FLAG (node4, parity-untested): the tensor->Structure decode, the designed-backbone ->
inverse-fold re-lowering, and the predicted-sequence remap are best-effort and not
numerically verified in this environment (no torch/numpy). The sequence-into-system remap
(mutating residue identities of the designed system) is left owner/node4-gated; the decoded
candidate carries the designed/refolded ``Structure`` and the predicted sequence as a
``Metric`` axis rather than mutating the canonical system.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np
import torch
from typing_extensions import Self

from ...data.molecular import ComponentReference
from ...data.structure import Structure
from ...pipelines._generator import resolve_generator
from ...results import Metric, Score
from .model import (
    BoltzGenCandidate,
    BoltzGenIFoldInput,
    BoltzGenIFoldModel,
    BoltzGenInput,
    BoltzGenModel,
    BoltzGenResidueConstraint,
    BoltzGenResult,
    BoltzGenSchema,
)
from .processor import BoltzGenIFoldProcessor, BoltzGenProcessor

if TYPE_CHECKING:
    # Read-only boltz2 composition types (never edited by this family).
    from ..boltz2 import Boltz2AffinityModel, Boltz2AffinityProcessor, Boltz2Model, Boltz2Processor


class BoltzGenPipeline:
    r"""Bind the BoltzGen design/inverse-fold pairs and composed boltz2 stages.

    Args:
        design_model: The structure-diffusion designer.
        design_processor: Its lowering Processor.
        ifold_model: The inverse-folding designer.
        ifold_processor: Its lowering Processor.
        fold: Optional injected boltz2 fold Model (read-only composition).
        fold_processor: Optional injected boltz2 fold Processor.
        affinity: Optional injected boltz2 affinity Model (read-only composition).
        affinity_processor: Optional injected boltz2 affinity Processor.
    """

    def __init__(
        self,
        *,
        design_model: BoltzGenModel,
        design_processor: BoltzGenProcessor,
        ifold_model: BoltzGenIFoldModel,
        ifold_processor: BoltzGenIFoldProcessor,
        fold: Boltz2Model | None = None,
        fold_processor: Boltz2Processor | None = None,
        affinity: Boltz2AffinityModel | None = None,
        affinity_processor: Boltz2AffinityProcessor | None = None,
    ) -> None:
        self.design_model = design_model
        self.design_processor = design_processor
        self.ifold_model = ifold_model
        self.ifold_processor = ifold_processor
        self.fold = fold
        self.fold_processor = fold_processor
        self.affinity = affinity
        self.affinity_processor = affinity_processor

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Self:
        r"""Place every bound Model and Processor on one device and dtype in place."""

        self.design_model.to(device=device, dtype=dtype)
        self.design_processor.to(device=device, dtype=dtype)
        self.ifold_model.to(device=device, dtype=dtype)
        self.ifold_processor.to(device=device, dtype=dtype)
        for component in (self.fold, self.fold_processor, self.affinity, self.affinity_processor):
            if component is not None:
                component.to(device=device, dtype=dtype)
        return self

    def design(
        self,
        input: object | None = None,
        *,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        protocol: str = "protein-anything",
        num_designs: int = 1,
        num_sampling_steps: int | None = None,
        temperature: float | None = None,
        do_refold: bool = True,
        do_affinity: bool = False,
        ranking: str | Sequence[tuple[str, float]] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenResult:
        r"""Run the released multi-stage design flow and return a ``BoltzGenResult``.

        Stage order (upstream ``step_names``): (1) design lowering + structure sampling,
        (2) inverse folding on the sampled backbone, (3) optional boltz2 refold, (4)
        optional boltz2 affinity, (5) analysis decode into ``BoltzGenCandidate`` s.

        Args:
            input: Optional target/context system, structure, or path.
            total_len: Family-owned whole-assembly token count (owner-gated when it mints rows).
            conditions: Family-owned §3.7.2 schema records.
            protocol: Upstream default-override protocol name (``cli/boltzgen.py``); selects which
                Scores exist and which thresholds gate. FLAG: only recorded here; the per-protocol
                override table is left for node4.
            num_designs: Number of structure samples to draw.
            num_sampling_steps: Optional EDM sampling steps.
            temperature: Optional inverse-fold sampling temperature.
            do_refold: Whether to run the boltz2 refold stage (requires injected fold Model+Processor).
            do_affinity: Whether to run the boltz2 affinity stage (requires injected affinity Model+Processor).
            ranking: Optional single Score name or weighted key; not frozen into the Result.
            seed: Optional integer seed (mutually exclusive with ``generator``).
            generator: Optional RNG for reproducible sampling.

        Returns:
            One ``BoltzGenResult`` with one candidate per drawn design.
        """

        if do_refold and (self.fold is None or self.fold_processor is None):
            raise RuntimeError(
                "do_refold=True requires an injected boltz2 fold Model and Processor "
                "(BoltzGenPipeline.fold / .fold_processor); the folding stage composes with boltz2 "
                "read-only and cannot run without them"
            )
        if do_affinity and (self.affinity is None or self.affinity_processor is None):
            raise RuntimeError(
                "do_affinity=True requires an injected boltz2 affinity Model and Processor "
                "(BoltzGenPipeline.affinity / .affinity_processor)"
            )
        generator = resolve_generator(seed, generator, device=self.design_processor.device)

        design_batch = self.design_processor(input, total_len=total_len, conditions=tuple(conditions))
        design_output = self.design_model.design(
            design_batch,
            num_samples=num_designs,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        candidates: list[BoltzGenCandidate] = []
        for sample_index in range(num_designs):
            backbone = _decode_structure(design_batch, design_output.pred_coords[0, sample_index])
            ifold_batch = self.ifold_processor(backbone)
            ifold_output = self.ifold_model.design(ifold_batch, temperature=temperature, generator=generator)
            candidate = self._analyze(
                backbone=backbone,
                ifold_batch=ifold_batch,
                ifold_output=ifold_output,
                do_refold=do_refold,
                do_affinity=do_affinity,
                generator=generator,
            )
            candidates.append(candidate)
        return BoltzGenResult(candidates=tuple(candidates))

    def inverse_fold(
        self,
        input: Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        temperature: float | None = None,
        num_sequences: int = 1,
        generator: torch.Generator | None = None,
    ) -> BoltzGenResult:
        r"""Run inverse folding (stage 2) standalone on a fixed backbone."""

        ifold_batch = self.ifold_processor(input, design=design, aa_constraints=aa_constraints, symmetry=symmetry)
        backbone = _fixed_backbone_structure(input)
        candidates: list[BoltzGenCandidate] = []
        for _ in range(num_sequences):
            ifold_output = self.ifold_model.design(ifold_batch, temperature=temperature, generator=generator)
            candidates.append(
                self._analyze(
                    backbone=backbone,
                    ifold_batch=ifold_batch,
                    ifold_output=ifold_output,
                    do_refold=False,
                    do_affinity=False,
                    generator=generator,
                )
            )
        return BoltzGenResult(candidates=tuple(candidates))

    def _analyze(
        self,
        *,
        backbone: Structure,
        ifold_batch: BoltzGenIFoldInput,
        ifold_output: object,
        do_refold: bool,
        do_affinity: bool,
        generator: torch.Generator | None,
    ) -> BoltzGenCandidate:
        r"""Assemble one candidate: attach the inverse-fold sequence metric and composed scores.

        FLAG (node4): the predicted-sequence remap into the canonical system is owner-gated;
        the candidate keeps the designed/refolded backbone and carries the predicted per-residue
        NLL as a ``Metric``. Refold and affinity Scores are read off the composed boltz2 results.
        """

        layout = ifold_batch.layout
        assert layout is not None
        scores: dict[str, Score] = {}
        metrics: dict[str, Metric] = {}

        logits = getattr(ifold_output, "logits")
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        tokens = getattr(ifold_output, "tokens")
        axis = tuple(layout.output_residues)
        if tokens is not None:
            per_residue_nll = -log_probs.gather(-1, tokens[0].long().unsqueeze(-1)).squeeze(-1)
            metrics["per_residue_nll"] = Metric(
                values=per_residue_nll.detach().cpu().numpy().astype(np.float32),
                axis=axis,
                unit="nats",
            )

        structure = backbone
        if do_refold:
            assert self.fold is not None and self.fold_processor is not None
            fold_batch = self.fold_processor(backbone)
            fold_result = self.fold.fold(fold_batch, num_samples=1, generator=generator)
            prediction = fold_result.predictions[0]
            structure = prediction.structure
            for name in ("plddt", "ptm", "iptm"):
                if name in prediction.scores:
                    scores[name] = prediction.scores[name]

        if do_affinity:
            assert self.affinity is not None and self.affinity_processor is not None
            affinity_batch = self.affinity_processor(structure)
            affinity_result = self.affinity.score(affinity_batch, generator=generator)
            if "affinity_pred_value" in affinity_result.scores:
                scores["binding_affinity"] = affinity_result.scores["affinity_pred_value"]
            if "affinity_probability_binary" in affinity_result.scores:
                scores["binding_probability"] = affinity_result.scores["affinity_probability_binary"]

        return BoltzGenCandidate(structure=structure, scores=scores, metrics=metrics)


def _decode_structure(batch: BoltzGenInput, coords: torch.Tensor) -> Structure:
    r"""Map one sample's model atom coords back onto canonical sites → ``Structure``.

    Mirrors ``Boltz2AffinityModel.score``'s decode: place the model atom order onto the
    system's complete heavy-atom sites. FLAG (node4): parity-untested.
    """

    layout = batch.layout
    if layout is None or layout.system is None:
        raise ValueError("BoltzGenPipeline requires a Processor-populated layout with a system to decode")
    system = layout.system
    complete_sites = [site for site, atom in system.iter_atoms() if atom.element != "H"]
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
    model_coords = coords[: len(layout.atom_sites)].float().cpu().numpy()
    for site, values in zip(layout.atom_sites, model_coords):
        index = complete_index[site]
        coordinates[index] = values
        coordinate_mask[index] = True
    return Structure(
        system=system,
        sites=tuple(complete_sites),
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
    )


def _fixed_backbone_structure(input: object) -> Structure:
    r"""Return the input as a ``Structure`` for candidate carrying (standalone inverse fold)."""

    from pathlib import Path

    from ...io.structure import read_structure

    if isinstance(input, str | PathLike):
        return read_structure(Path(input).expanduser())
    if isinstance(input, Structure):
        return input
    raise TypeError("inverse_fold requires a fixed backbone Structure or structure path")
