r"""PXDesign's biological operation boundary.

Two Pipelines live here because both diverge from the shared coordinate-free
:class:`~ofoldx.pipelines.design.DesignPipeline` (which lowers
``AllowedResidueIdentities`` / ``TiedResidueIdentities`` conditions and returns the frozen
coordinate-free ``DesignResult``):

- :class:`PxDesignPipeline` binds the native PXDesign-d generator Model+Processor and owns
  the ``design`` verb over PXDesign's OWN topology arg (``PlacementSpec``) and per-residue target
  label (``Hotspot``), returning the family-owned :class:`PxDesignResult`. It mirrors
  :class:`~ofoldx.models.proteinmpnn.pipeline.ProteinMpnnPipeline`.

- :class:`PxDesignSuitePipeline` is the released *composition* pipeline. Per DESIGN §6.4 it
  is a **live-only** composition with **no single owning Model and no checkpoint**: it holds
  already-registered child pipelines by reference and composes them. It is intentionally NOT
  a ``_PairedPipeline`` and is never ``register_pipeline``-ed, so it mints no
  ``pipeline.json`` model-repository artifact of its own; it is reconstructed at runtime from
  the registered ``ofoldx.pxdesign`` + ``ofoldx.proteinmpnn`` + ``ofoldx.alphafold2``
  (AF2-IG, pending) + ``ofoldx.protenix`` repositories.

The suite is largely DEPENDENCY-GATED: the AF2 initial-guess (AF2-IG) fold entry point does
not yet exist in the OFoldX AlphaFold2 family, and the released ranking / geometry / RMSD
metrics come from the un-vendored ``pxdbench`` (PXDesignBench) submodule. Those stages raise
rather than fabricate parity. ProteinMPNN / AlphaFold2 / Protenix interfaces are imported
read-only and never edited from here.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

import torch
from typing_extensions import Self

from ...data import PlacementSpec
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines import resolve_generator
from ...pipelines._paired import _PairedPipeline
from .model import Hotspot, PxDesignModel, PxDesignResult
from .processor import PxDesignProcessor


class PxDesignPipeline(_PairedPipeline):
    r"""Bind one PXDesign-d Model--Processor pair and own the ``design`` verb."""

    model: PxDesignModel
    processor: PxDesignProcessor

    def __init__(
        self,
        *,
        model: PxDesignModel,
        processor: PxDesignProcessor,
    ) -> None:
        super().__init__(model=model, processor=processor)

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        conditions: Sequence[PlacementSpec | Hotspot] = (),
        num_candidates: int = 1,
        num_sampling_steps: int = 400,
        sample_chunk_size: int | None = 10,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PxDesignResult:
        r"""Generate ranked target--binder backbone candidates for one released target.

        The Processor lowers ``input`` plus PXDesign's own ``PlacementSpec`` topology and optional
        ``Hotspot`` label to a :class:`~ofoldx.models.pxdesign.model.PxDesignInput`; the
        Model runs the released EDM sampler and decodes each coordinate sample. The returned
        :class:`PxDesignResult` carries raw generator backbones whose ``scores`` are empty
        (the released sampler emits no per-sample confidence); the suite Pipeline populates
        the evaluation Scores and ranking.
        """

        batch = self.processor(input, conditions=conditions)
        generator = resolve_generator(seed, generator, device=batch.reference_positions.device)
        return self.model.design(
            batch,
            num_candidates=num_candidates,
            num_sampling_steps=num_sampling_steps,
            sample_chunk_size=sample_chunk_size,
            generator=generator,
        )


class PxDesignSuitePipeline:
    r"""Live-only composition of the released PXDesign suite (DESIGN §6.4).

    Composes already-registered child pipelines held by reference:

    - ``generator`` — a :class:`PxDesignPipeline` producing PXDesign-d backbones.
    - ``sequencer`` — a ``ProteinMpnnPipeline`` designing the binder sequence
      (``num_candidates`` = num sequences, ``temperature``).
    - ``af2_evaluator`` — the AlphaFold2 initial-guess (AF2-IG) evaluator. **Currently
      dependency-gated:** the released stage is AF2 *initial-guess multimer* (seed the
      backbone into the recycling ``prev_pos`` channel and read the interface metrics
      ``i_pTM`` / ``unscaled_i_pAE``), which the deterministic-monomer OFoldX AF2 family
      does not yet expose. Passing an evaluator without an initial-guess fold entry point
      makes :meth:`run` raise ``NotImplementedError`` at the evaluation stage.
    - ``protenix_evaluator`` — the optional Protenix confidence evaluator (extended preset).

    Ranking / filtering / success-bucketing (helpers.py:32-289) and geometry / RMSD metrics
    come from the un-vendored ``pxdbench`` submodule, so the scoring stage is likewise
    dependency-gated. This class owns no checkpoint, adds no registration, and is never
    saved to a model repository.
    """

    def __init__(
        self,
        *,
        generator: PxDesignPipeline,
        sequencer: object,
        af2_evaluator: object,
        protenix_evaluator: object | None = None,
    ) -> None:
        self.generator = generator
        self.sequencer = sequencer
        self.af2_evaluator = af2_evaluator
        self.protenix_evaluator = protenix_evaluator

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Place every child pipeline on one device and dtype in place."""

        self.generator.to(device=device, dtype=dtype)
        for child in (self.sequencer, self.af2_evaluator, self.protenix_evaluator):
            if child is not None and hasattr(child, "to"):
                child.to(device=device, dtype=dtype)  # type: ignore[attr-defined]
        return self

    def run(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        conditions: Sequence[PlacementSpec | Hotspot] = (),
        preset: Literal["custom", "preview", "extended"] = "custom",
        num_samples: int = 5,
        num_sampling_steps: int = 400,
        eta_type: str = "const",
        eta_min: float = 2.5,
        eta_max: float = 2.5,
        num_seqs: int = 8,
        mpnn_temperature: float = 0.1,
        eval_complex: bool = True,
        eval_binder_monomer: bool = True,
        eval_protenix: bool = False,
        eval_protenix_mini: bool = False,
        use_msa: bool = True,
        n_max_runs: int = 1,
        return_topk: int = 5,
        min_total_return: int = 10,
        max_success_return: int = 20,
        extended_w_af2: float = 0.5,
        extended_w_ptx: float = 0.5,
        early_stop: bool = True,
        min_early_stop_rounds: int = 0,
        min_early_stop_successes: int = 1,
        target_template_rmsd_thres: float = 2.0,
        seeds: Sequence[int] = (),
        dtype: torch.dtype | None = None,
    ) -> PxDesignResult:
        r"""Run the released backbone->sequence->evaluate->rank suite.

        This inventories every released stage control. The preset bundles mirror
        ``runner/presets.py``: ``preview`` = AF2-only at ``num_samples=100``; ``extended`` =
        AF2 + Protenix at ``num_samples=500``; both keep ``num_sampling_steps=400``. Ranking
        mode is inferred from the produced Protenix columns rather than a flag.

        The evaluation and scoring stages are dependency-gated (AF2-IG entry point and the
        ``pxdbench`` submodule); until they land this raises rather than fabricate parity.
        """

        preview = preset == "preview"
        extended = preset == "extended"
        effective_num_samples = 100 if preview else 500 if extended else num_samples
        run_protenix = extended or eval_protenix or eval_protenix_mini

        # Stage 1 — PXDesign-d backbones (native, real).
        generator_seed = seeds[0] if seeds else None
        backbones = self.generator.design(
            input,
            conditions=conditions,
            num_candidates=effective_num_samples,
            num_sampling_steps=num_sampling_steps,
            seed=generator_seed,
        )

        # Stage 2 — ProteinMPNN sequencing (registered family, DONE). The exact released
        # ``num_seqs`` / ``mpnn_temperature`` binder-mode defaults and interface-metric
        # extraction live in the un-vendored ``pxdbench`` submodule, so the concrete
        # per-backbone sequencing loop is gated on that dependency landing.
        del num_seqs, mpnn_temperature  # released controls, wired once pxdbench lands.

        # Stage 3 — AF2-IG evaluation (DEPENDENCY-GATED).
        if not self._has_initial_guess_fold(self.af2_evaluator):
            raise NotImplementedError(
                "PXDesign suite evaluation requires the AlphaFold2 initial-guess (AF2-IG) "
                "fold entry point (seed backbone into the recycling prev_pos channel + "
                "interface metrics i_pTM / unscaled_i_pAE). The current OFoldX AlphaFold2 "
                "family is deterministic monomer with no initial-guess path; AF2-IG must be "
                "added by the AlphaFold2 family before the suite can evaluate. Not fabricated "
                "here."
            )

        # Stage 4 — optional Protenix evaluation, and Stage 5 — pxdbench ranking / geometry /
        # RMSD / success bucketing. Both are dependency-gated on the un-vendored PXDesignBench
        # (``pxdbench``) submodule; the exact success buckets, weighted rank fusion
        # (extended_w_af2 / extended_w_ptx), and geometry/Rg metrics come from it.
        del (
            run_protenix,
            eval_complex,
            eval_binder_monomer,
            use_msa,
            n_max_runs,
            return_topk,
            min_total_return,
            max_success_return,
            extended_w_af2,
            extended_w_ptx,
            early_stop,
            min_early_stop_rounds,
            min_early_stop_successes,
            target_template_rmsd_thres,
            eta_type,
            eta_min,
            eta_max,
            dtype,
            backbones,
        )
        raise NotImplementedError(
            "PXDesign suite ranking / filtering / success-bucketing and geometry / RMSD "
            "metrics come from the un-vendored PXDesignBench (``pxdbench``) submodule "
            "(github.com/bytedance/PXDesignBench). It is not vendored, so the evaluation Score "
            "population and released two-tier rank fusion are not reproduced here. Not "
            "fabricated."
        )

    @staticmethod
    def _has_initial_guess_fold(evaluator: object) -> bool:
        r"""Report whether the AF2 evaluator exposes an initial-guess fold entry point.

        The released AF2-IG stage seeds the designed backbone into AF2's recycling
        ``prev_pos`` channel. Until the AlphaFold2 family exposes such an entry point (e.g. a
        ``fold(..., initial_guess=...)`` parameter or a dedicated ``fold_initial_guess``),
        the suite cannot run its evaluation stage and this returns ``False``.
        """

        if evaluator is None:
            return False
        if hasattr(evaluator, "fold_initial_guess"):
            return True
        fold = getattr(evaluator, "fold", None)
        if fold is None:
            return False
        try:
            import inspect

            parameters = inspect.signature(fold).parameters
        except (TypeError, ValueError):
            return False
        return "initial_guess" in parameters or "prev_pos" in parameters
