r"""ODesign design-and-assign composition (§4.11.6).

The released ODesign protein deliverable is a two-stage composition under the single
``design`` verb: the ODesign core generates backbone coordinates (S seeds × M diffusion
samples), and the OInvFold sequence step assigns sequences to each backbone (K top-k
variants). Upstream the OInvFold sequence module for the protein modality *is* the released
ProteinMPNN / LigandMPNN network (``mpnn_model_type`` dual-model dispatch,
``infer_runner.py:84-91``), so this Pipeline **composes the already-registered
``ofoldx/proteinmpnn`` and ``ofoldx/ligandmpnn`` Pipelines read-only** rather than
re-vendoring the sequence network. The result is the family-owned M×S×K
:class:`ODesignDesignResult`.

Sampling knobs (§4.11.6) — "beam / temperature / top-k" for the protein modality:
  * ``temperature`` (upstream ``invfold_temp``) → the MPNN sampler temperature;
  * ``num_sequences`` = top-k (upstream ``invfold_topk``) → MPNN ``num_candidates``;
  * ``use_beam`` is **inert for protein** and MUST NOT be a functional protein knob — the
    released ``proteinmpnn_inference`` ignores it (``proteinmpnn/inference.py`` ``_ = use_beam``).
    Beam decoding exists only on the non-protein OInvFold checkpoints (``oinvfold_*.ckpt``,
    gRNAde), which are not vendored and whose call type differs (§4.11.9 flag 6, owner-gated).
    This Pipeline accepts ``use_beam`` and ignores it, exactly as the released protein path does.

Owner-gated / node4-verification flags carried here (not verifiable by reading, no torch):
  * FLAG(parse_invfold): the per-chain designed-position partition (upstream ``parse_invfold``
    ``design_mask``) is realized here as "design every designable protein position of the
    backbone" (``positions=None``), which is faithful for the released fully-de-novo binder
    case. Partial motif-scaffolding (a backbone mixing fixed and designed residues) needs the
    exact per-chain ``design_mask`` and is deferred with the §3.7.3 PlacementSpec contract.
  * FLAG(model dispatch): upstream dispatches ProteinMPNN vs LigandMPNN *per designed chain*
    by the released strict 8 Å atom-context cutoff. Here dispatch is per backbone by whether
    the backbone system carries any non-protein (ligand / nucleic) context; LigandMPNN then
    applies its own ``ligand_interface_cutoff=8.0``. The per-chain split is the refinement.
  * FLAG(sequence-into-backbone): each candidate carries the assigned-sequence
    ``MolecularSystem`` (from the MPNN candidate) as ``system`` and the ODesign backbone
    ``Structure`` as ``structure``; rewriting the assigned sequence back into the backbone
    ``Structure``'s atom rows (side-chain atom-count change vs the GLY backbone template) is a
    downstream structural op left for node4.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from typing import Literal

import torch
from typing_extensions import Self

from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import resolve_generator
from ...pipelines._composite import _CompositePair, _CompositePipeline
from ...results import Score
from ..ligandmpnn import LigandMpnnPipeline
from ..proteinmpnn import ProteinMpnnPipeline
from .model import (
    Contact,
    DesiredCovalentLink,
    ODesignDesignCandidate,
    ODesignDesignResult,
    ODesignModel,
    ODesignPlacements,
    ReferenceCoordinates,
)
from .processor import ODesignProcessor


def _has_atom_context(system: MolecularSystem) -> bool:
    r"""Report whether the backbone system carries non-protein (ligand / nucleic) context.

    Any entity that is not a protein ``PolymerSequence`` is atom context for the released
    LigandMPNN sequence model (small molecules, ions, and nucleic chains). A system of only
    protein chains routes to the sequence-only ProteinMPNN model. (FLAG(model dispatch):
    upstream refines this to a per-chain strict 8 Å test.)
    """

    return any(
        not (isinstance(entity, PolymerSequence) and entity.polymer_type == "protein")
        for entity in system.entities.values()
    )


def _odesign_scores(scores: dict[str, Score], *, used_ligand: bool) -> dict[str, Score]:
    r"""Map the released MPNN per-sequence Scores to the §4.11.7 ODesign score table.

    The OInvFold protein module *is* the frozen ProteinMPNN / LigandMPNN network, so the
    per-sequence Scores reuse the §5.3.2 frozen semantics verbatim (raw ``log_softmax``, not
    the tempered sampling distribution). ProteinMPNN emits ``designed_sequence_nll``
    (nat/residue) while LigandMPNN emits ``designed_sequence_confidence`` = ``exp(-nll)``;
    each is the exact inverse of the other, so both branches populate the same four keys with
    identical units / ``preferred`` / scope. ``ligand_interface_confidence`` is present only
    when the LigandMPNN atom-context model was used (its scope is the LigandMPNN interface
    subset ``I``); the empty-scope / NaN rule of §5.3.2 is carried through unchanged from the
    source Score (LigandMPNN already emits ``scope=frozenset()`` + ``NaN`` for a chain with no
    ligand interface).
    """

    recovery = scores["sequence_recovery"]
    if "designed_sequence_nll" in scores:  # ProteinMPNN emits the nll directly.
        nll = scores["designed_sequence_nll"]
        nll_value = nll.value
        designed_scope = nll.scope
        confidence_value = math.exp(-nll_value) if math.isfinite(nll_value) else float("nan")
    else:  # LigandMPNN emits the confidence; recover the nll as its exact inverse.
        confidence = scores["designed_sequence_confidence"]
        designed_scope = confidence.scope
        confidence_value = confidence.value
        nll_value = -math.log(confidence_value) if confidence_value > 0 else float("nan")
    mapped: dict[str, Score] = {
        "designed_sequence_nll": Score(
            nll_value,
            unit="nat/residue",
            preferred="lower",
            scope=designed_scope,
        ),
        "designed_sequence_confidence": Score(
            confidence_value,
            preferred="higher",
            scope=designed_scope,
        ),
        "sequence_recovery": Score(
            recovery.value,
            preferred="higher",
            scope=recovery.scope,
        ),
    }
    if used_ligand:
        interface = scores["ligand_interface_confidence"]
        mapped["ligand_interface_confidence"] = Score(
            interface.value,
            preferred="higher",
            scope=interface.scope,
        )
    return mapped


class ODesignPipeline(_CompositePipeline):
    r"""Compose the ODesign backbone designer with the OInvFold (MPNN) sequence step.

    This is the full released protein deliverable (§4.11.6): ODesign generates backbones over
    S seeds × M diffusion samples, and each backbone is assigned K sequence variants by the
    read-only ProteinMPNN / LigandMPNN Pipelines. It mirrors the ``ProteinMpnnDesignAndPack``
    composition structure — a coordinating object holding child Pipelines with a shared
    :meth:`to` — rather than the single-pair ``_PairedPipeline``, because the sequence network
    is a *separately registered* family loaded read-only, never re-vendored.

    It is a DESIGN §7.4 multi-pair concrete Pipeline (see ``_PAIR_LAYOUT`` below) and is registered
    so the working protein deliverable routes through the repository factory. The remaining
    de-novo ligand/NA generation gap is a runtime capability limit (see ``__init__.py`` / README),
    not a registration gate.
    """

    # DESIGN §7.4/§7.3 pair-layout: the ODesign backbone pair owns the runnable root, and the
    # two read-only OInvFold sequence designers live at fixed sub-directories. Each designer is a
    # single-pair child Pipeline, so its reloaded pair is wrapped back into its Pipeline class.
    _PAIR_LAYOUT = (
        _CompositePair("", "model", "processor"),
        _CompositePair("protein_designer", "protein_designer", None, wrap=ProteinMpnnPipeline),
        _CompositePair("ligand_designer", "ligand_designer", None, wrap=LigandMpnnPipeline),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        protein_designer: ProteinMpnnPipeline,
        ligand_designer: LigandMpnnPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.protein_designer = protein_designer
        self.ligand_designer = ligand_designer

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Place the ODesign pair and both sequence Pipelines on one device / dtype."""

        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)
        self.protein_designer.to(device=device, dtype=dtype)
        self.ligand_designer.to(device=device, dtype=dtype)
        return self

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 0.1,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        receptor_mode: Literal["rigid", "flex"] | None = None,
        conditions: Sequence[Contact | ODesignPlacements | DesiredCovalentLink | ReferenceCoordinates] = (),
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        use_beam: bool = False,
    ) -> ODesignDesignResult:
        r"""Generate the released M×S×K backbone-and-sequence design grid."""

        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if num_seeds <= 0 or num_backbones <= 0 or num_sequences <= 0:
            raise ValueError("num_seeds, num_backbones, and num_sequences must be positive")
        # §4.11.6: use_beam is inert on the released protein path (upstream `_ = use_beam`).
        # It is accepted for call-signature parity with the non-protein OInvFold paths and
        # never influences protein decoding; beam is owner-gated (§4.11.9 flag 6).
        _ = use_beam

        seed_list = self._resolve_seeds(num_seeds, seeds, seed, generator)

        batch = self._build_batch(input, conditions, receptor_mode)

        candidates: list[ODesignDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            backbones = self.model.design(
                batch,
                num_candidates=num_backbones,
                generator=resolve_generator(int(backbone_seed), None, device=batch.reference_positions.device),
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for backbone_index, backbone in enumerate(backbones.candidates):
                structure = backbone.structure
                if structure is None:  # pragma: no cover - design always decodes a Structure
                    raise RuntimeError("ODesign backbone candidate is missing its decoded Structure")
                used_ligand = _has_atom_context(structure.system)
                designer = self.ligand_designer if used_ligand else self.protein_designer
                sequence_result = designer.design(
                    structure,
                    num_candidates=num_sequences,
                    temperature=temperature,
                    seed=int(backbone_seed) * 1_000_003 + backbone_index,
                )
                for sequence_index, variant in enumerate(sequence_result.candidates):
                    candidates.append(
                        ODesignDesignCandidate(
                            system=variant.system,
                            structure=structure,
                            scores=_odesign_scores(variant.scores, used_ligand=used_ligand),
                            seed_index=seed_index,
                            backbone_index=backbone_index,
                            sequence_index=sequence_index,
                        )
                    )
        return ODesignDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )

    def _resolve_seeds(
        self,
        num_seeds: int,
        seeds: Sequence[int] | None,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> tuple[int, ...]:
        r"""Fix the S seed list (explicit ``configs.seeds`` or drawn from one base stream)."""

        if seeds is not None:
            seed_list = tuple(int(value) for value in seeds)
            if not seed_list:
                raise ValueError("seeds must not be empty")
            return seed_list
        if generator is None:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
        drawn = torch.randint(
            1,
            torch.iinfo(torch.int32).max,
            (num_seeds,),
            generator=generator,
        )
        return tuple(int(value) for value in drawn.tolist())

    def _build_batch(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        conditions: Sequence[Contact | ODesignPlacements | DesiredCovalentLink | ReferenceCoordinates],
        receptor_mode: Literal["rigid", "flex"] | None,
    ):
        r"""Lower the canonical input once, honoring an optional per-call receptor_mode override."""

        if receptor_mode is None:
            return self.processor(input, conditions=conditions)
        previous = self.processor.config.receptor_mode
        self.processor.config.receptor_mode = receptor_mode
        try:
            return self.processor(input, conditions=conditions)
        finally:
            self.processor.config.receptor_mode = previous
