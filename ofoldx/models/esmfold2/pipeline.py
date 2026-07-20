r"""OFoldX's biological operation boundary for released ESMFold2 folding and binder design.

Owns the one Processor call and the family's released biological verbs. ``fold`` is the
convenience wrapper over ``ESMFold2Processor`` + ``ESMFold2Model.fold``, returning the
shared ``FoldResult`` unchanged. ``design`` freezes the released binder-design signature
(gradient descent on per-position soft-sequence logits through the experimental recurrent
graph, then a discrete refold) and returns the family-owned, coordinate-producing
``ESMFold2DesignResult``; its body is blocked on the experimental checkpoints and raises
``NotImplementedError`` this wave. Upstream provenance:
``biohub_esm/cookbook/tutorials/binder_design.py`` (``compute_distogram_iptm_proxy`` L423,
binder prompt ``'#'`` mutable positions L110, CDR annotation L386/L400).

Per DESIGN.md 7.2 and Principle 13 this concrete Pipeline is not registered
(no ``register_pipeline`` / ``pipeline.json``) until the experimental binder weights land
and ``design`` is fully implemented; this wave freezes its signature at contract level.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch
from typing_extensions import Self

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...results import FoldResult
from .model import ESMFold2DesignResult, ESMFold2Model
from .processor import ESMFold2Processor


class ESMFold2Pipeline:
    r"""Bind one ESMFold2 Model-Processor pair for the released folding and design verbs.

    Marker id ``"ofoldx.esmfold2"``. The Pipeline is not a base class and shares no runtime
    request object with any other family. ``fold`` returns the shared ``FoldResult``;
    ``design`` returns the family-owned ``ESMFold2DesignResult``.
    """

    model: ESMFold2Model
    processor: ESMFold2Processor

    def __init__(
        self,
        *,
        model: ESMFold2Model,
        processor: ESMFold2Processor,
    ) -> None:
        self.model = model
        self.processor = processor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Place the Model and Processor on one device and dtype in place.

        No whole-Model ``dtype`` is forced: per DESIGN.md 4.1.1 the audited release stores
        its head in FP32 and the ESMC child in BF16, and an explicit ``dtype`` here would
        intentionally collapse that mixture. Ordinary device movement preserves it.
        """

        self.model.to(device=device, dtype=dtype)
        self.processor.to(device=device, dtype=dtype)
        return self

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_loops: int | None = None,
        num_sampling_steps: int | None = None,
        language_model_mask_probability: float | None = None,
        language_model_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        denoising_early_exit_rmsd: float | None = None,
    ) -> FoldResult:
        r"""Lower one folding input and run released ESMFold2 folding into a ``FoldResult``.

        Forwards every effective released inference control to ``ESMFold2Model.fold``; the
        Fast variant rejects ``msa`` at the paired Processor boundary. ``Structure`` inputs
        are rejected there too (topology-only fold).
        """

        batch = self.processor(input, msa=msa)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            seed=seed,
            generator=generator,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            language_model_mask_probability=language_model_mask_probability,
            language_model_dropout=language_model_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_subsample_at_inference=msa_subsample_at_inference,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            denoising_early_exit_rmsd=denoising_early_exit_rmsd,
        )

    def design(
        self,
        target: str | PathLike[str] | Structure | MolecularSystem,
        *,
        binder_length: int,
        binder_prompt: str | None = None,
        num_candidates: int = 1,
        num_optimization_steps: int = 150,
        remove_cysteines: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ESMFold2DesignResult:
        r"""Design binders against one target and refold each into an ``ESMFold2DesignResult``.

        Frozen released signature: gradient descent on per-position soft-sequence logits
        through the experimental recurrent graph (``binder_prompt`` fixes amino acids at
        positions, ``'#'`` marks a mutable position, ``binder_design.py`` L110), followed by
        a discrete refold of the optimized sequence with the release model. Each candidate
        owns the complete folded ``target|binder`` complex; binder length is fixed by
        ``binder_length`` per call but varies across separate calls, which is the
        coordinate-producing, variable-length property carried by ``ESMFold2DesignResult``.

        BLOCKED: the experimental recurrent-graph checkpoints are not local, so this body is
        inventoried at contract level only this wave.
        """

        raise NotImplementedError(
            "ESMFold2 binder design is blocked on the experimental recurrent-graph "
            "checkpoints (biohub_esm type='experimental'), which are not on node4/HF; "
            "the released design signature is frozen at contract level (DESIGN.md 5.3.1) "
            "and ESMFold2Pipeline is not registered until those weights and the published "
            "MultiMolecule ESMC asset land"
        )
