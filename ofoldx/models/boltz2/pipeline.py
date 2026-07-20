r"""The concrete two-stage Boltz-2 affinity call boundary (DESIGN 5.4.3 / 6.2)."""

from __future__ import annotations

from os import PathLike

import torch

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem
from ...data.structure import Structure
from ...pipelines._composite import _CompositePair, _CompositePipeline
from ...pipelines._generator import resolve_generator
from ...results import AffinityResult
from .model import Boltz2AffinityModel, Boltz2Model
from .processor import Boltz2AffinityProcessor, Boltz2Processor


class Boltz2AffinityPipeline(_CompositePipeline):
    r"""Bind the released Boltz-2 folding and affinity pairs into one two-stage affinity op.

    The Pipeline holds two distinct concrete Models over two distinct checkpoints (a composite Model is never
    built): a folding/confidence pair (``structure_model``/``structure_processor``) and the pair-only affinity
    pair (``model``/``processor``). Its one public op, ``score``, runs the frozen two-stage selection of
    DESIGN 5.4.3 / 6.2:

    1. Fold the whole system and score each first-stage prediction's whole-call ``confidence_score``.
    2. Sort those predictions by descending ``confidence_score`` into ``structure_predictions``.
    3. Pass the rank-0 canonical ``Structure`` (never a file/tensor) through the affinity Processor.
    4. Run the affinity Model's independent second-pass sampling with its max-iPTM no-pTM-fallback selection.
    5. Return a flat ``AffinityResult`` (second-pass ``target`` + first-stage ``structure_predictions`` +
       the six official Scores), never nesting the second-pass ScoreResult.

    Per DESIGN 6.2 the concrete score signature names each scientifically distinct sampling count the upstream
    inventory proves. The two passes are two independent upstream ``predict_args`` dicts, so they do not share
    one sampling count: the structure/confidence pass uses ``num_samples`` / ``num_sampling_steps`` (upstream
    ``diffusion_samples`` / ``sampling_steps``, defaults 1 / 200) and the affinity pass uses
    ``affinity_num_samples`` / ``affinity_num_sampling_steps`` (upstream ``diffusion_samples_affinity`` /
    ``sampling_steps_affinity``, defaults 5 / 200). Upstream reference:
    ``reference/boltz/src/boltz/main.py:1298-1306`` (structure ``predict_args``) and
    ``reference/boltz/src/boltz/main.py:1371-1379`` (affinity ``predict_affinity_args``); the per-pass CLI
    defaults are at ``main.py:863-867`` / ``main.py:857-861`` (structure) and ``main.py:1002-1006`` /
    ``main.py:996-1000`` (affinity).

    The two passes' per-pass recycling counts are already faithful without a pipeline argument: each Model
    defaults recycling to its own config, so the structure pass recycles ``Boltz2Config.recycling_steps`` (3,
    upstream ``main.py:1299``) and the affinity pass recycles ``Boltz2AffinityConfig.recycling_steps`` (5,
    upstream's hardcoded ``main.py:1372``). MSA is routed to both Processors because upstream reads both passes
    from the one shared ``msa_dir`` (``main.py:1274`` structure, ``main.py:1361`` affinity); the affinity crop /
    ``method="other"`` divergence (``main.py:1367-1368``) is owned inside ``Boltz2AffinityProcessor``, not the
    Pipeline. No ``to(...)`` placement policy is frozen (DESIGN 6.3), so none is exposed yet.
    """

    model: Boltz2AffinityModel
    processor: Boltz2AffinityProcessor
    structure_model: Boltz2Model
    structure_processor: Boltz2Processor

    # DESIGN §7.4/§7.3 pair-layout: the affinity pass owns the runnable root pair and the
    # structure/confidence pass lives at the fixed ``structure/`` sub-directory (§6.2/§7.5).
    _PAIR_LAYOUT = (
        _CompositePair("", "model", "processor"),
        _CompositePair("structure", "structure_model", "structure_processor"),
    )

    def __init__(
        self,
        *,
        model: Boltz2AffinityModel,
        processor: Boltz2AffinityProcessor,
        structure_model: Boltz2Model,
        structure_processor: Boltz2Processor,
    ) -> None:
        self.model = model
        self.processor = processor
        self.structure_model = structure_model
        self.structure_processor = structure_processor

    def score(
        self,
        input: MolecularSystem | Structure | str | PathLike[str],
        *,
        ligand: str | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        num_sampling_steps: int | None = None,
        affinity_num_samples: int = 5,
        affinity_num_sampling_steps: int | None = None,
        molecular_weight_correction: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AffinityResult:
        r"""Run the frozen two-stage Boltz-2 affinity selection and return a flat ``AffinityResult``.

        Args:
            input: The molecular system, canonical ``Structure``, or path lowered by the structure Processor.
            ligand: The affinity ligand instance selector consumed by the affinity Processor.
            msa: MSA routed to both Processors (upstream reads both passes from the one shared ``msa_dir``).
            num_samples: Structure/confidence-pass diffusion samples (upstream ``diffusion_samples``, default 1;
                ``reference/boltz/src/boltz/main.py:863-867,1301``).
            num_sampling_steps: Structure/confidence-pass EDM steps (upstream ``sampling_steps``, default 200 via
                the sampler config; ``main.py:857-861,1300``).
            affinity_num_samples: Affinity-pass diffusion samples (upstream ``diffusion_samples_affinity``,
                default 5; ``main.py:1002-1006,1374``).
            affinity_num_sampling_steps: Affinity-pass EDM steps (upstream ``sampling_steps_affinity``, default
                200 via the sampler config; ``main.py:996-1000,1373``).
            molecular_weight_correction: Whether the affinity ensemble applies the released MW calibration.
            seed: Optional integer seed; mutually exclusive with ``generator``. It seeds one RNG once and both
                passes then draw from that single continued stream, mirroring upstream's single
                ``seed_everything(seed)`` before the two sequential passes (``main.py:1101-1102``).
            generator: Optional RNG shared across both passes; mutually exclusive with ``seed``.
        """

        # Upstream (main.py:1298-1306) structure/confidence pass: its own diffusion_samples / sampling_steps and
        # its own config recycling (Boltz2Config.recycling_steps == 3). MSA is routed here and to the affinity
        # Processor because upstream reads both passes from the one shared msa_dir.
        fold_batch = self.structure_processor(input, msa=msa)

        # Resolve seed|generator once and thread the single generator through both passes so the affinity pass
        # continues the same stream rather than re-seeding to an identical state. This mirrors upstream's single
        # seed_everything(seed) (main.py:1101-1102) before the two sequential predict passes; passing the same
        # integer seed to each pass would instead reset the affinity pass to the structure pass's starting noise.
        generator = resolve_generator(seed, generator, device=fold_batch.atom_mask.device)

        fold_result = self.structure_model.fold(
            fold_batch,
            num_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        # Rank first-stage predictions by descending confidence_score; Python's stable sort keeps generation
        # order among ties (earliest-tie stable), so structure_predictions is confidence-descending, not
        # generation-ordered.
        structure_predictions = tuple(
            sorted(
                fold_result.predictions,
                key=lambda prediction: prediction.scores["confidence_score"].value,
                reverse=True,
            )
        )
        best_structure = structure_predictions[0].structure

        # Upstream (main.py:1371-1379) affinity pass: its own diffusion_samples_affinity / sampling_steps_affinity
        # and its own config recycling (Boltz2AffinityConfig.recycling_steps == 5, upstream's hardcoded
        # main.py:1372). The affinity crop / method="other" divergence (main.py:1367-1368) is owned inside
        # Boltz2AffinityProcessor.
        affinity_batch = self.processor(best_structure, ligand=ligand, msa=msa)
        score_result = self.model.score(
            affinity_batch,
            num_samples=affinity_num_samples,
            num_sampling_steps=affinity_num_sampling_steps,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        return AffinityResult(
            target=score_result.target,
            structure_predictions=structure_predictions,
            scores=score_result.scores,
            metrics=score_result.metrics,
        )
