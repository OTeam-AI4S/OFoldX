r"""
PxDesign multi-model design stages.

The pxdesign-specific transform that turns generated backbones into all-atom candidates: assign sequences with
native ProteinMPNN ([`assign_sequences`][ofoldx.models.pxdesign.stages.assign_sequences]) and filter with a native
structure predictor ([`filter_candidates`][ofoldx.models.pxdesign.stages.filter_candidates]). These are the family
glue an [`AllAtomDesignPipeline`][ofoldx.pipelines.design.AllAtomDesignPipeline] injects as its stage callables --
the generic pipeline stays family-agnostic; the impedance-matching between PxDesign's token layout and a sibling
Designer / Folder lives here in the family (single file), never in the shared pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from ..contract import (
    AllAtomDesignCandidate,
    DesignCandidate,
    SequenceDesignCandidate,
    StructurePredictionOutput,
)
from .model import PxDesignInput


def assign_sequences(
    candidates: DesignCandidate | StructurePredictionOutput | Tensor,
    *,
    batch: PxDesignInput,
    model: Any | None = None,
    mpnn_model: Any | None = None,
    pretrained_path: str | None = None,
    temperature: float | Tensor | None = None,
    num_samples: int | None = None,
    amino_acid_bias: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DesignCandidate:
    r"""
    Assign sequences to PxDesign backbone candidates with native ProteinMPNN.

    Args:
        candidates: Generated PxDesign candidates or coords.
        batch: Original PxDesign conditioning batch carrying token metadata.
        model: Loaded ProteinMPNN-compatible model or callable.
        mpnn_model: Alias for ``model`` for pipeline-style configs.
        pretrained_path: Optional OFoldX ProteinMPNN pretrained directory.
        temperature: Optional ProteinMPNN sampling temperature.
        num_samples: Sequence samples per generated backbone.
        amino_acid_bias: Optional full-token or selected-token amino-acid bias.
        generator: Optional torch RNG used by ProteinMPNN sampling.

    Returns:
        Candidate output with ``sequence_tokens``, optional ``sequence_logits``,
        and MPNN scores attached.

    Raises:
        ValueError: If ``candidates`` carry no backbone coords, or neither a model nor a pretrained path is given.

    Examples:
        >>> from ofoldx.models.pxdesign import assign_sequences  # doctest: +SKIP
        >>> assigned = assign_sequences(candidates, batch=batch, mpnn_model=mpnn)  # doctest: +SKIP
        >>> assigned.sequence_tokens.shape  # doctest: +SKIP
        torch.Size([1, 128])
    """

    output = _as_design_output(candidates, batch=batch)
    if output.structure_prediction is None:
        raise ValueError("PxDesign sequence assignment requires candidate coords")
    sequence_model = _resolve_proteinmpnn_model(
        model=model,
        mpnn_model=mpnn_model,
        pretrained_path=pretrained_path,
    )
    return _assign_sequences_with_model(
        output,
        batch=batch,
        sequence_model=sequence_model,
        temperature=temperature,
        num_samples=num_samples,
        amino_acid_bias=amino_acid_bias,
        generator=generator,
    )


def filter_candidates(
    candidates: DesignCandidate | StructurePredictionOutput | Tensor,
    *,
    batch: PxDesignInput | None = None,
    predictor: Any | None = None,
    model: Any | None = None,
    pretrained_path: str | None = None,
    prediction: Any | None = None,
    predictions: Any | None = None,
    predictor_input: Any | None = None,
    input_batch: Any | None = None,
    input_builder: Callable[..., Any] | None = None,
    score_fn: Callable[..., Mapping[str, Tensor]] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> DesignCandidate:
    r"""
    Score PxDesign candidates with a native structure-predictor hook.

    The hook can be a loaded Protenix, AlphaFold3, Boltz, or compatible
    callable. If the hook needs its own model-local input, pass ``predictor_input``
    or ``input_builder``. Direct ``prediction`` outputs are accepted for pipeline
    composition and tests. When ``thresholds`` is given, a boolean ``filter_pass``
    score is added by comparing the derived scores against each rule.

    Args:
        candidates: Generated PxDesign candidates or coords to score.
        batch: Original PxDesign conditioning batch, used for token-masked score reductions.
        predictor: Loaded structure predictor or callable hook.
        model: Alias for ``predictor`` for pipeline-style configs.
        pretrained_path: Optional OFoldX structure-predictor pretrained directory.
        prediction: Precomputed predictor output, bypassing predictor resolution.
        predictions: Alias for ``prediction``.
        predictor_input: Optional model-local input passed straight to the predictor.
        input_batch: Alias for ``predictor_input``.
        input_builder: Optional callable building the predictor input from the candidate output.
        score_fn: Optional callable returning a custom score mapping to use instead of the built-in metrics.
        thresholds: Optional per-score pass/fail rules; adds a ``filter_pass`` score when given.

    Returns:
        Candidate output with ``filter_*`` scores (and optional ``filter_pass``) merged into ``scores`` and a
        ``filtering`` entry recorded in ``metadata``.

    Raises:
        ValueError: If the predictor yields no usable tensor scores, or a threshold references a missing score.
        TypeError: If the resolved predictor is neither callable nor exposes ``generate``.

    Examples:
        >>> from ofoldx.models.pxdesign import filter_candidates  # doctest: +SKIP
        >>> scored = filter_candidates(
        ...     candidates, batch=batch, predictor=af3, thresholds={"filter_mean_plddt": 0.7}
        ... )  # doctest: +SKIP
        >>> scored.scores["filter_pass"]  # doctest: +SKIP
        tensor([1., 0.])
    """

    output = _as_design_output(candidates, batch=batch)
    predictor_output = predictions if predictions is not None else prediction
    if predictor_output is None:
        predictor_model = _resolve_structure_predictor(
            predictor=predictor,
            model=model,
            pretrained_path=pretrained_path,
        )
        predictor_output = _run_structure_predictor(
            predictor_model,
            output,
            batch=batch,
            predictor_input=predictor_input if predictor_input is not None else input_batch,
            input_builder=input_builder,
        )

    filter_scores = _filter_scores(
        output,
        predictor_output,
        batch=batch,
        score_fn=score_fn,
    )
    if thresholds is not None:
        filter_scores["filter_pass"] = _threshold_pass(filter_scores, thresholds).to(dtype=torch.float32)

    scores = _merge_score_dicts(output.scores, filter_scores)
    metadata = dict(output.metadata or {})
    metadata["filtering"] = {
        "backend": "native_structure_predictor_hook",
        "scores": tuple(filter_scores),
        "thresholds": None if thresholds is None else dict(thresholds),
    }
    return AllAtomDesignCandidate(
        sequence_logits=output.sequence_logits,
        sequence_tokens=output.sequence_tokens,
        structure_prediction=output.structure_prediction,
        scores=scores,
        metadata=metadata,
    )


def _as_design_output(
    candidates: DesignCandidate | StructurePredictionOutput | Tensor,
    *,
    batch: PxDesignInput | None,
) -> AllAtomDesignCandidate:
    r"""
    Normalize raw candidates into an [`AllAtomDesignCandidate`][ofoldx.models.contract.AllAtomDesignCandidate].

    The composite mixes sequence and structure across stages, so it works on the joint
    [`AllAtomDesignCandidate`][ofoldx.models.contract.AllAtomDesignCandidate] (all fields present, unset ones
    ``None``): a [`DesignCandidate`][ofoldx.models.contract.DesignCandidate] of any mode is lifted into one, a bare
    [`StructurePredictionOutput`][ofoldx.models.contract.StructurePredictionOutput] becomes its structure, and a raw
    coord tensor is wrapped in a structure prediction with a placeholder distogram sized from ``batch``.

    Args:
        candidates: A design candidate, a structure-prediction output, or a raw coord tensor.
        batch: Optional conditioning batch used to size the placeholder distogram for tensor inputs.

    Returns:
        The candidates as an [`AllAtomDesignCandidate`][ofoldx.models.contract.AllAtomDesignCandidate].

    Raises:
        TypeError: If ``candidates`` is none of the supported types.
    """

    if isinstance(candidates, AllAtomDesignCandidate):
        return candidates
    if isinstance(candidates, DesignCandidate):
        return AllAtomDesignCandidate(
            sequence_logits=candidates.sequence_logits,
            sequence_tokens=candidates.sequence_tokens,
            structure_prediction=candidates.structure_prediction,
            scores=candidates.scores,
            metadata=candidates.metadata,
        )
    if isinstance(candidates, StructurePredictionOutput):
        return AllAtomDesignCandidate(structure_prediction=candidates)
    if torch.is_tensor(candidates):
        if batch is None:
            distogram_logits = candidates.new_zeros(())
        else:
            num_tokens = batch.token_mask.shape[-1]
            distogram_logits = candidates.new_zeros(*batch.token_mask.shape, num_tokens, 1)
        return AllAtomDesignCandidate(
            structure_prediction=StructurePredictionOutput(
                pred_coords=candidates,
                distogram_logits=distogram_logits,
            )
        )
    raise TypeError(f"unsupported PxDesign candidate type: {type(candidates)!r}")


def _resolve_proteinmpnn_model(
    *,
    model: Any | None,
    mpnn_model: Any | None,
    pretrained_path: str | None,
) -> Any:
    r"""
    Resolve the ProteinMPNN model/callable used for sequence assignment.

    Accepts either alias (``model`` / ``mpnn_model``) or a pretrained path; loading from a path is deferred
    to avoid importing ProteinMPNN unless needed.

    Args:
        model: A loaded ProteinMPNN-compatible model or callable.
        mpnn_model: Alias for ``model``.
        pretrained_path: OFoldX ProteinMPNN pretrained directory to load from.

    Returns:
        The resolved sequence model.

    Raises:
        ValueError: If both aliases disagree, or none of the three sources is provided.
    """

    if model is not None and mpnn_model is not None and model is not mpnn_model:
        raise ValueError("pass only one of sequence_assignment['model'] or sequence_assignment['mpnn_model']")
    resolved = model if model is not None else mpnn_model
    if resolved is not None:
        return resolved
    if pretrained_path is not None:
        from ..proteinmpnn import ProteinMPNNDesigner

        return ProteinMPNNDesigner.from_pretrained(pretrained_path)
    raise ValueError(
        "PxDesign sequence_assignment requires a loaded ProteinMPNN model/callable via "
        "sequence_assignment={'model': model} or an OFoldX checkpoint path via "
        "sequence_assignment={'pretrained_path': path}."
    )


def _resolve_structure_predictor(
    *,
    predictor: Any | None,
    model: Any | None,
    pretrained_path: str | None,
) -> Any:
    r"""
    Resolve the structure predictor/callable used for candidate filtering.

    Accepts either alias (``predictor`` / ``model``) or a pretrained path; loading from a path is deferred to
    avoid importing the auto model unless needed.

    Args:
        predictor: A loaded structure predictor or callable.
        model: Alias for ``predictor``.
        pretrained_path: OFoldX structure-predictor pretrained directory to load from.

    Returns:
        The resolved predictor.

    Raises:
        ValueError: If both aliases disagree, or none of the three sources is provided.
    """

    if predictor is not None and model is not None and predictor is not model:
        raise ValueError("pass only one of filtering['predictor'] or filtering['model']")
    resolved = predictor if predictor is not None else model
    if resolved is not None:
        return resolved
    if pretrained_path is not None:
        from ..auto import AutoModel

        return AutoModel.from_pretrained(pretrained_path)
    raise ValueError(
        "PxDesign filtering requires a structure predictor/callable via filtering={'predictor': model}, "
        "a direct prediction via filtering={'prediction': output}, or an OFoldX checkpoint path via "
        "filtering={'pretrained_path': path}."
    )


def _assign_sequences_with_model(
    output: DesignCandidate,
    *,
    batch: PxDesignInput,
    sequence_model: Any,
    temperature: float | Tensor | None,
    num_samples: int | None,
    amino_acid_bias: Tensor | None,
    generator: torch.Generator | None,
) -> DesignCandidate:
    r"""
    Run ProteinMPNN over every flattened backbone candidate and scatter the results back.

    For each candidate it builds a ProteinMPNN input over the protein tokens with a complete backbone, runs the
    model, and scatters the assigned tokens/logits/scores back onto the full token grid, then concatenates all
    candidates and repeats the structure prediction for multi-sequence samples.

    Args:
        output: Candidate output whose ``structure_prediction`` provides the backbone coords.
        batch: Conditioning batch supplying token metadata and backbone atom indices.
        sequence_model: Resolved ProteinMPNN model/callable.
        temperature: Optional ProteinMPNN sampling temperature.
        num_samples: Sequence samples per backbone candidate.
        amino_acid_bias: Optional per-token amino-acid bias.
        generator: Optional torch RNG for ProteinMPNN sampling.

    Returns:
        Candidate output with assigned ``sequence_tokens``, optional ``sequence_logits``, MPNN scores, and a
        ``sequence_assignment`` metadata entry.

    Raises:
        ValueError: If the candidate output carries no structure-prediction coords.
    """

    structure_prediction = output.structure_prediction
    if structure_prediction is None:
        raise ValueError("PxDesign sequence assignment requires structure_prediction coords")
    coords_flat, candidate_batch_index, original_candidate_shape = _flatten_candidate_coords(
        structure_prediction.pred_coords,
        batch=batch,
    )

    token_rows: list[Tensor] = []
    logit_rows: list[Tensor] = []
    log_prob_rows: list[Tensor] = []
    sampling_prob_rows: list[Tensor] = []
    likelihood_rows: list[Tensor] = []
    sequence_counts: list[int] = []
    for candidate_index, coords in enumerate(coords_flat):
        batch_index = int(candidate_batch_index[candidate_index])
        mpnn_input, selected_tokens = _proteinmpnn_input_from_pxdesign(
            batch,
            coords,
            batch_index=batch_index,
            amino_acid_bias=amino_acid_bias,
        )
        mpnn_output = _run_proteinmpnn_model(
            sequence_model,
            mpnn_input,
            temperature=temperature,
            num_samples=num_samples,
            generator=generator,
        )
        assigned_tokens, assigned_logits, assigned_scores = _scatter_mpnn_output(
            mpnn_output,
            batch=batch,
            batch_index=batch_index,
            selected_tokens=selected_tokens,
        )
        token_rows.append(assigned_tokens)
        if assigned_logits is not None:
            logit_rows.append(assigned_logits)
        if "mpnn_log_probs" in assigned_scores:
            log_prob_rows.append(assigned_scores["mpnn_log_probs"])
        if "mpnn_sampling_probs" in assigned_scores:
            sampling_prob_rows.append(assigned_scores["mpnn_sampling_probs"])
        if "mpnn_sequence_log_likelihood" in assigned_scores:
            likelihood_rows.append(assigned_scores["mpnn_sequence_log_likelihood"])
        sequence_counts.append(int(assigned_tokens.shape[0]))

    sequence_tokens = torch.cat(token_rows, dim=0) if token_rows else batch.token_mask.new_empty(0, 0)
    sequence_logits = torch.cat(logit_rows, dim=0) if logit_rows else None
    scores: dict[str, Tensor] = {}
    if log_prob_rows:
        scores["mpnn_log_probs"] = torch.cat(log_prob_rows, dim=0)
    if sampling_prob_rows:
        scores["mpnn_sampling_probs"] = torch.cat(sampling_prob_rows, dim=0)
    if likelihood_rows:
        scores["mpnn_sequence_log_likelihood"] = torch.cat(likelihood_rows, dim=0)

    structure_with_sequences = _repeat_structure_prediction_for_sequences(
        structure_prediction,
        sequence_counts=sequence_counts,
        coords_flat=coords_flat,
    )
    metadata = dict(output.metadata or {})
    metadata["sequence_assignment"] = {
        "backend": "native_proteinmpnn",
        "candidate_shape": tuple(int(value) for value in original_candidate_shape),
        "num_sequence_samples": int(num_samples or 1),
    }
    return AllAtomDesignCandidate(
        sequence_logits=sequence_logits,
        sequence_tokens=sequence_tokens,
        structure_prediction=structure_with_sequences,
        scores=_merge_score_dicts(output.scores, scores),
        metadata=metadata,
    )


def _flatten_candidate_coords(pred_coords: Tensor, *, batch: PxDesignInput) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    r"""
    Flatten leading candidate/sample dims and map each candidate back to its input-batch row.

    Args:
        pred_coords: Candidate coords `[..., N_atom, 3]` whose leading dims index candidates/samples.
        batch: Conditioning batch used to validate the atom count and infer the batch mapping.

    Returns:
        A tuple of the flattened coords `[num_candidates, N_atom, 3]`, the per-candidate batch index
        `[num_candidates]`, and the original leading shape.

    Raises:
        ValueError: If the atom axis mismatches, the coords lack a leading axis, or the leading dims cannot be
            mapped to the input batch size.
    """

    num_atoms = batch.atom_mask.shape[-1]
    if pred_coords.shape[-2:] != (num_atoms, 3):
        raise ValueError(
            "PxDesign candidate coords must end with " f"(N_atom, 3) = {(num_atoms, 3)}, got {tuple(pred_coords.shape)}"
        )
    if pred_coords.dim() < 3:
        raise ValueError("PxDesign candidate coords must have at least three dimensions")
    leading_shape = pred_coords.shape[:-2]
    batch_size = batch.token_mask.shape[0]
    if not leading_shape:
        raise ValueError("PxDesign candidate coords need a batch or sample axis")
    if leading_shape[0] == batch_size:
        candidate_batch_index = torch.arange(batch_size, device=pred_coords.device)
        for repeat in leading_shape[1:]:
            candidate_batch_index = candidate_batch_index.repeat_interleave(int(repeat))
    elif batch_size == 1:
        candidate_batch_index = torch.zeros(math.prod(leading_shape), device=pred_coords.device, dtype=torch.long)
    else:
        raise ValueError(
            "cannot map PxDesign candidate coords to input batch: "
            f"coordinate leading shape {tuple(leading_shape)}, batch size {batch_size}"
        )
    return pred_coords.reshape(-1, num_atoms, 3), candidate_batch_index.to(dtype=torch.long), tuple(leading_shape)


def _proteinmpnn_input_from_pxdesign(
    batch: PxDesignInput,
    coords: Tensor,
    *,
    batch_index: int,
    amino_acid_bias: Tensor | None,
) -> Any:
    r"""
    Build a [`ProteinMPNNInput`][] for one candidate over its designable protein tokens.

    Selects the protein tokens that are valid, protein-like, and have a complete N/CA/C/O backbone, gathers
    their backbone coords, and packages the ProteinMPNN input plus the selected-token index for scatter-back.

    Args:
        batch: Conditioning batch supplying token metadata and backbone atom indices.
        coords: Flattened coords `[N_atom, 3]` for this candidate.
        batch_index: Row of ``batch`` this candidate maps to.
        amino_acid_bias: Optional per-token amino-acid bias.

    Returns:
        A tuple of the built [`ProteinMPNNInput`][] and the selected-token index tensor.

    Raises:
        ValueError: If backbone atom indices are missing, or no protein token has a complete backbone.
    """

    if batch.backbone_atom_indices is None:
        raise ValueError(
            "PxDesign sequence assignment needs batch.backbone_atom_indices. Use PxDesignProcessor "
            "or provide N/CA/C/O atom indices shaped (B, N_token, 4)."
        )
    backbone_indices = batch.backbone_atom_indices[batch_index].to(device=coords.device, dtype=torch.long)
    token_mask = batch.token_mask[batch_index].to(device=coords.device, dtype=torch.bool)
    design_mask = (
        torch.ones_like(token_mask)
        if batch.design_token_mask is None
        else batch.design_token_mask[batch_index].to(device=coords.device, dtype=torch.bool)
    )
    protein_token_mask = _protein_like_token_mask(batch, batch_index=batch_index).to(device=coords.device)
    has_backbone = (backbone_indices >= 0).all(dim=-1)
    selected = token_mask & has_backbone & protein_token_mask
    if not bool(selected.any()):
        raise ValueError("PxDesign sequence assignment found no protein tokens with complete N/CA/C/O backbone atoms")

    selected_tokens = torch.nonzero(selected, as_tuple=True)[0]
    selected_backbone = backbone_indices[selected_tokens].reshape(-1)
    backbone_coords = coords.index_select(0, selected_backbone).reshape(1, selected_tokens.numel(), 4, 3)
    sequence_tokens = _restype_to_mpnn_tokens(batch.restype[batch_index]).to(device=coords.device)[selected_tokens]
    selected_bias = _selected_amino_acid_bias(
        amino_acid_bias,
        batch_index=batch_index,
        selected_tokens=selected_tokens,
        device=coords.device,
        dtype=coords.dtype,
    )

    from ..proteinmpnn import ProteinMPNNInput

    return (
        ProteinMPNNInput(
            backbone_coords=backbone_coords,
            sequence_tokens=sequence_tokens.unsqueeze(0),
            token_mask=torch.ones(1, selected_tokens.numel(), device=coords.device, dtype=coords.dtype),
            residue_index=batch.residue_index[batch_index, selected_tokens].to(device=coords.device).unsqueeze(0),
            chain_index=batch.chain_index[batch_index, selected_tokens].to(device=coords.device).unsqueeze(0),
            design_mask=design_mask[selected_tokens].to(dtype=coords.dtype).unsqueeze(0),
            amino_acid_bias=selected_bias,
        ),
        selected_tokens,
    )


def _run_proteinmpnn_model(
    sequence_model: Any,
    mpnn_input: Any,
    *,
    temperature: float | Tensor | None,
    num_samples: int | None,
    generator: torch.Generator | None,
) -> DesignCandidate:
    r"""
    Invoke a ProteinMPNN model/callable and normalize its output to a [`DesignCandidate`][].

    Prefers a ``generate`` method; otherwise calls the model directly (retrying without kwargs on a
    ``TypeError``) and adapts a returned mapping's ``logits``/``S`` keys into the output fields.

    Args:
        sequence_model: The ProteinMPNN model or callable.
        mpnn_input: The built ProteinMPNN input.
        temperature: Optional sampling temperature, dropped when ``None``.
        num_samples: Optional sample count, dropped when ``None``.
        generator: Optional torch RNG, dropped when ``None``.

    Returns:
        The ProteinMPNN result as a [`DesignCandidate`][].

    Raises:
        TypeError: If the model neither exposes ``generate`` nor is callable.
    """

    kwargs = {"temperature": temperature, "num_samples": num_samples, "generator": generator}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    designer = getattr(sequence_model, "design", None)
    if callable(designer):
        return designer(mpnn_input, **kwargs)
    if callable(sequence_model):
        try:
            result = sequence_model(mpnn_input, **kwargs)
        except TypeError:
            result = sequence_model(mpnn_input)
        if isinstance(result, DesignCandidate):
            return result
        if isinstance(result, Mapping):
            sequence_logits = result.get("sequence_logits")
            if sequence_logits is None:
                sequence_logits = result.get("logits")
            sequence_tokens = result.get("sequence_tokens")
            if sequence_tokens is None:
                sequence_tokens = result.get("S")
            return SequenceDesignCandidate(
                sequence_logits=sequence_logits,
                sequence_tokens=sequence_tokens,
                scores={key: value for key, value in result.items() if torch.is_tensor(value)},
            )
    raise TypeError("ProteinMPNN assignment model must expose generate(input, ...) or be callable")


def _scatter_mpnn_output(
    mpnn_output: DesignCandidate,
    *,
    batch: PxDesignInput,
    batch_index: int,
    selected_tokens: Tensor,
) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
    r"""
    Scatter a candidate's ProteinMPNN output back onto the full token grid.

    Starts from the batch's base restype tokens, overwrites the selected (designed) tokens with the MPNN
    assignment, and scatters any per-token logits and log-prob/sampling-prob scores; sequence-level
    log-likelihoods are reduced to a per-sample mean.

    Args:
        mpnn_output: The per-candidate ProteinMPNN output.
        batch: Conditioning batch supplying base restypes and the token count.
        batch_index: Row of ``batch`` this candidate maps to.
        selected_tokens: Index of tokens ProteinMPNN redesigned.

    Returns:
        A tuple of full-grid sequence tokens, optional full-grid logits, and a dict of scattered MPNN scores.

    Raises:
        ValueError: If the output has no ``sequence_tokens`` or its length mismatches ``selected_tokens``.
    """

    if mpnn_output.sequence_tokens is None:
        raise ValueError("ProteinMPNN sequence assignment returned no sequence_tokens")
    mpnn_tokens = mpnn_output.sequence_tokens
    if mpnn_tokens.dim() == 1:
        mpnn_tokens = mpnn_tokens.unsqueeze(0)
    if mpnn_tokens.shape[-1] != selected_tokens.numel():
        raise ValueError(
            "ProteinMPNN sequence length does not match selected PxDesign protein tokens: "
            f"{mpnn_tokens.shape[-1]} vs {selected_tokens.numel()}"
        )
    base_tokens = _restype_to_mpnn_tokens(batch.restype[batch_index]).to(device=mpnn_tokens.device)
    full_tokens = base_tokens.unsqueeze(0).expand(mpnn_tokens.shape[0], -1).clone()
    full_tokens[:, selected_tokens.to(device=mpnn_tokens.device)] = mpnn_tokens.to(dtype=torch.long)

    full_logits = None
    if mpnn_output.sequence_logits is not None:
        logits = mpnn_output.sequence_logits
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        full_logits = logits.new_zeros(mpnn_tokens.shape[0], batch.token_mask.shape[-1], logits.shape[-1])
        full_logits[:, selected_tokens.to(device=logits.device), :] = logits

    scores: dict[str, Tensor] = {}
    if mpnn_output.scores is not None:
        log_probs = mpnn_output.scores.get("log_probs")
        if torch.is_tensor(log_probs):
            scores["mpnn_log_probs"] = _scatter_token_score(
                log_probs,
                num_tokens=batch.token_mask.shape[-1],
                selected_tokens=selected_tokens,
            )
        sampling_probs = mpnn_output.scores.get("sampling_probs")
        if torch.is_tensor(sampling_probs):
            scores["mpnn_sampling_probs"] = _scatter_token_score(
                sampling_probs,
                num_tokens=batch.token_mask.shape[-1],
                selected_tokens=selected_tokens,
            )
        sequence_log_likelihood = mpnn_output.scores.get("sequence_log_likelihood")
        if isinstance(sequence_log_likelihood, Tensor):
            scores["mpnn_sequence_log_likelihood"] = sequence_log_likelihood.reshape(mpnn_tokens.shape[0], -1).mean(-1)
    return full_tokens, full_logits, scores


def _scatter_token_score(score: Tensor, *, num_tokens: int, selected_tokens: Tensor) -> Tensor:
    r"""
    Scatter a per-selected-token score into a zero-filled full-token tensor.

    Args:
        score: Score over selected tokens, `[num_selected, ...]` or `[batch, num_selected, ...]`.
        num_tokens: Full token count of the output grid.
        selected_tokens: Index of the selected tokens within the full grid.

    Returns:
        The score placed at ``selected_tokens`` in a `[batch, num_tokens, ...]` zero tensor.

    Examples:
        >>> import torch
        >>> _scatter_token_score(
        ...     torch.tensor([[1.0, 2.0]]), num_tokens=4, selected_tokens=torch.tensor([1, 3])
        ... ).tolist()
        [[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [1.0, 2.0]]]
    """

    if score.dim() == 2:
        score = score.unsqueeze(0)
    full_shape = (score.shape[0], num_tokens, *score.shape[2:])
    full = score.new_zeros(full_shape)
    full[:, selected_tokens.to(device=score.device), ...] = score
    return full


def _repeat_structure_prediction_for_sequences(
    structure_prediction: StructurePredictionOutput,
    *,
    sequence_counts: list[int],
    coords_flat: Tensor,
) -> StructurePredictionOutput:
    r"""
    Repeat each candidate's structure prediction to match its number of assigned sequences.

    When every candidate produced exactly one sequence the prediction is returned unchanged; otherwise the
    flattened coords are repeat-interleaved by ``sequence_counts`` so coords and sequences stay aligned.

    Args:
        structure_prediction: The per-candidate structure prediction to repeat.
        sequence_counts: Number of assigned sequences per candidate, in candidate order.
        coords_flat: Flattened candidate coords `[num_candidates, N_atom, 3]`.

    Returns:
        A structure prediction whose coords are repeated to one row per assigned sequence.
    """

    if all(count == 1 for count in sequence_counts):
        return structure_prediction
    repeat_counts = torch.tensor(sequence_counts, device=coords_flat.device, dtype=torch.long)
    repeated_coords = coords_flat.repeat_interleave(repeat_counts, dim=0)
    return StructurePredictionOutput(
        pred_coords=repeated_coords,
        distogram_logits=structure_prediction.distogram_logits,
        plddt_logits=structure_prediction.plddt_logits,
        pae_logits=structure_prediction.pae_logits,
        pde_logits=structure_prediction.pde_logits,
        resolved_logits=structure_prediction.resolved_logits,
        bfactor_logits=structure_prediction.bfactor_logits,
        token_bond_type_logits=structure_prediction.token_bond_type_logits,
        denoised_coords=structure_prediction.denoised_coords,
        augmented_target=structure_prediction.augmented_target,
        noise_level=structure_prediction.noise_level,
    )


def _run_structure_predictor(
    predictor: Any,
    output: DesignCandidate,
    *,
    batch: PxDesignInput | None,
    predictor_input: Any | None,
    input_builder: Callable[..., Any] | None,
) -> Any:
    r"""
    Run a structure predictor over candidate coords, building its input as needed.

    Uses ``input_builder`` (retrying without ``batch`` on a ``TypeError``) or an explicit ``predictor_input``
    when given, calling ``generate`` if available; otherwise calls the predictor directly on the output.

    Args:
        predictor: The structure predictor or callable.
        output: Candidate output to score.
        batch: Optional conditioning batch passed to the input builder / predictor.
        predictor_input: Optional pre-built model-local input.
        input_builder: Optional callable building the predictor input from the output.

    Returns:
        The raw predictor output.

    Raises:
        TypeError: If ``predictor`` is neither callable nor exposes ``generate``.
    """

    if input_builder is not None:
        try:
            predictor_input = input_builder(output, batch=batch)
        except TypeError:
            predictor_input = input_builder(output)
    if predictor_input is not None:
        folder = getattr(predictor, "fold", None)
        if callable(folder):
            return folder(predictor_input)
        return predictor(predictor_input)
    if callable(predictor):
        try:
            return predictor(output, batch=batch)
        except TypeError:
            return predictor(output)
    raise TypeError("structure predictor must be callable or expose generate(input)")


def _filter_key(key: str) -> str:
    r"""
    Namespace a score key under the ``filter_`` prefix, idempotently.

    Args:
        key: A raw or already-prefixed score name.

    Returns:
        The key with a single ``filter_`` prefix.

    Examples:
        >>> _filter_key("plddt")
        'filter_plddt'
        >>> _filter_key("filter_plddt")
        'filter_plddt'
    """

    return key if key.startswith("filter_") else f"filter_{key}"


def _filter_scores(
    output: DesignCandidate,
    predictor_output: Any,
    *,
    batch: PxDesignInput | None,
    score_fn: Callable[..., Mapping[str, Tensor]] | None,
) -> dict[str, Tensor]:
    r"""
    Derive ``filter_*`` scores from a predictor output (or a custom scoring callable).

    With ``score_fn`` the caller's scores are namespaced and returned. Otherwise built-in metrics are derived:
    passthrough tensor scores, coordinate RMSD against the candidate coords when compatible, and expected
    pLDDT / PAE / PDE from the predictor's logits.

    Args:
        output: The candidate output being filtered.
        predictor_output: The raw structure-predictor output.
        batch: Optional conditioning batch for token-masked reductions.
        score_fn: Optional custom scoring callable overriding the built-in metrics.

    Returns:
        A dict of ``filter_*`` score tensors.

    Raises:
        ValueError: If no usable tensor score can be derived.
    """

    if score_fn is not None:
        try:
            custom_scores = score_fn(output, predictor_output, batch=batch)
        except TypeError:
            custom_scores = score_fn(output, predictor_output)
        return {_filter_key(key): value for key, value in custom_scores.items()}

    prediction = _prediction_structure_output(predictor_output)
    filter_scores: dict[str, Tensor] = {}
    scores: Mapping[str, Any] | None = None
    if isinstance(predictor_output, DesignCandidate):
        scores = predictor_output.scores
    elif isinstance(predictor_output, Mapping):
        scores = predictor_output
    if scores is not None:
        for key, value in scores.items():
            if torch.is_tensor(value):
                filter_scores[_filter_key(key)] = value
    if prediction is not None:
        if output.structure_prediction is not None and _coords_compatible(
            output.structure_prediction.pred_coords,
            prediction.pred_coords,
        ):
            filter_scores["filter_coord_rmsd"] = _coordinate_rmsd(
                output.structure_prediction.pred_coords,
                prediction.pred_coords,
            )
        if prediction.plddt_logits is not None:
            plddt = _expected_unit_interval_metric(prediction.plddt_logits)
            filter_scores["filter_plddt"] = plddt
            filter_scores["filter_mean_plddt"] = _masked_mean_token_score(plddt, batch=batch)
        if prediction.pae_logits is not None:
            filter_scores["filter_mean_pae"] = _expected_unit_interval_metric(prediction.pae_logits).mean(dim=(-1, -2))
        if prediction.pde_logits is not None:
            filter_scores["filter_mean_pde"] = _expected_unit_interval_metric(prediction.pde_logits).mean(dim=(-1, -2))
    if not filter_scores:
        raise ValueError(
            "PxDesign filtering predictor returned no usable tensor scores. Return a StructurePredictionOutput, "
            "a DesignCandidate with scores, a tensor mapping, or pass filtering={'score_fn': callable}."
        )
    return filter_scores


def _prediction_structure_output(predictor_output: Any) -> StructurePredictionOutput | None:
    r"""
    Extract a [`StructurePredictionOutput`][] from a predictor output, or ``None``.

    Unwraps a [`DesignCandidate`][], passes a structure output through, and builds one from a mapping
    carrying ``pred_coords`` (supplying a placeholder distogram when absent).

    Args:
        predictor_output: A structure output, generator output, or mapping.

    Returns:
        The structure prediction, or ``None`` when none can be extracted.
    """

    if isinstance(predictor_output, StructurePredictionOutput):
        return predictor_output
    if isinstance(predictor_output, DesignCandidate):
        return predictor_output.structure_prediction
    if isinstance(predictor_output, Mapping):
        pred_coords = predictor_output.get("pred_coords")
        distogram_logits = predictor_output.get("distogram_logits")
        if isinstance(pred_coords, Tensor):
            if not isinstance(distogram_logits, Tensor):
                distogram_logits = pred_coords.new_zeros(())
            return StructurePredictionOutput(
                pred_coords=pred_coords,
                distogram_logits=distogram_logits,
                plddt_logits=predictor_output.get("plddt_logits"),
                pae_logits=predictor_output.get("pae_logits"),
                pde_logits=predictor_output.get("pde_logits"),
                resolved_logits=predictor_output.get("resolved_logits"),
            )
    return None


def _coords_compatible(left: Tensor, right: Tensor) -> bool:
    r"""
    Return whether two coord tensors share their trailing `(N_atom, 3)` shape.

    Args:
        left: A coord tensor.
        right: Another coord tensor.

    Returns:
        ``True`` when the last two dims match.

    Examples:
        >>> import torch
        >>> _coords_compatible(torch.zeros(2, 3), torch.zeros(5, 2, 3))
        True
        >>> _coords_compatible(torch.zeros(2, 3), torch.zeros(4, 3))
        False
    """

    return left.shape[-2:] == right.shape[-2:]


def _coordinate_rmsd(reference: Tensor, predicted: Tensor) -> Tensor:
    r"""
    Compute per-structure coordinate RMSD, broadcasting a single reference or prediction.

    Args:
        reference: Reference coords `[..., N_atom, 3]`.
        predicted: Predicted coords `[..., N_atom, 3]`.

    Returns:
        Per-structure RMSD `[num_structures]`.

    Raises:
        ValueError: If the flattened batch shapes are incompatible.

    Examples:
        >>> import torch
        >>> round(float(_coordinate_rmsd(torch.zeros(1, 2, 3), torch.ones(1, 2, 3))[0]), 4)
        1.7321
    """

    ref = reference.reshape(-1, *reference.shape[-2:])
    pred = predicted.reshape(-1, *predicted.shape[-2:])
    if ref.shape[0] == 1 and pred.shape[0] > 1:
        ref = ref.expand(pred.shape[0], -1, -1)
    if pred.shape[0] == 1 and ref.shape[0] > 1:
        pred = pred.expand(ref.shape[0], -1, -1)
    if ref.shape != pred.shape:
        raise ValueError(
            f"cannot compute coordinate RMSD for shapes {tuple(reference.shape)} and {tuple(predicted.shape)}"
        )
    return torch.sqrt((ref - pred).square().sum(dim=-1).mean(dim=-1).clamp_min(0.0))


def _expected_unit_interval_metric(logits: Tensor) -> Tensor:
    r"""
    Compute the expected value of a categorical over a `[0, 1]` bin grid.

    Softmaxes ``logits`` over the last (bin) axis and dots with evenly spaced bin centers on `[0, 1]`, the
    standard reduction for pLDDT/PAE/PDE-style binned confidence logits.

    Args:
        logits: Binned logits `[..., num_bins]`.

    Returns:
        The expected metric `[...]`.

    Examples:
        >>> import torch
        >>> _expected_unit_interval_metric(torch.zeros(1, 3)).round(decimals=4).tolist()
        [0.5]
    """

    bins = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (logits.softmax(dim=-1) * bins).sum(dim=-1)


def _masked_mean_token_score(score: Tensor, *, batch: PxDesignInput | None) -> Tensor:
    r"""
    Reduce a per-token score to a per-structure mean, masking padding when a batch is given.

    Falls back to an unmasked mean when no batch is available or the token axis does not match the batch's
    token mask.

    Args:
        score: Per-token score, typically `[num_structures, num_tokens]`.
        batch: Optional conditioning batch supplying the token mask.

    Returns:
        Per-structure mean score.
    """

    if batch is None or score.shape[-1] != batch.token_mask.shape[-1]:
        return score.reshape(score.shape[0], -1).mean(dim=-1) if score.dim() > 1 else score.mean().reshape(1)
    mask = batch.token_mask.to(device=score.device, dtype=score.dtype)
    if score.shape[0] != mask.shape[0] and mask.shape[0] == 1:
        mask = mask.expand(score.shape[0], -1)
    return (score * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _threshold_pass(scores: Mapping[str, Tensor], thresholds: Mapping[str, Any]) -> Tensor:
    r"""
    Combine per-score threshold rules into a single per-candidate pass mask.

    Each rule is compared against its score and ANDed together; multi-token scores must pass at every token.

    Args:
        scores: Score name to score-tensor mapping.
        thresholds: Score name to threshold rule (see `_threshold_rule`).

    Returns:
        A boolean pass mask over candidates.

    Raises:
        ValueError: If a threshold references a missing score, or the threshold mapping is empty.

    Examples:
        >>> import torch
        >>> _threshold_pass({"filter_plddt": torch.tensor([0.9, 0.4])}, {"filter_plddt": 0.5}).tolist()
        [True, False]
    """

    pass_mask: Tensor | None = None
    for key, rule in thresholds.items():
        if key not in scores:
            raise ValueError(f"filter threshold references missing score {key!r}")
        op, threshold = _threshold_rule(rule)
        value = scores[key]
        current = _compare_threshold(value, op=op, threshold=float(threshold))
        if current.dim() > 1:
            current = current.reshape(current.shape[0], -1).all(dim=-1)
        pass_mask = current if pass_mask is None else pass_mask & current
    if pass_mask is None:
        raise ValueError("filter thresholds must not be empty")
    return pass_mask


def _threshold_rule(rule: Any) -> tuple[str, float]:
    r"""
    Normalize a threshold rule into an ``(op, value)`` pair.

    Accepts a bare number (defaulting the op to ``">="``), an ``(op, value)`` pair, or a mapping with ``op``
    and ``value`` keys.

    Args:
        rule: The raw threshold rule.

    Returns:
        The comparison operator and threshold value.

    Examples:
        >>> _threshold_rule(0.8)
        ('>=', 0.8)
        >>> _threshold_rule((">", 0.5))
        ('>', 0.5)
        >>> _threshold_rule({"op": "<=", "value": 2.0})
        ('<=', 2.0)
    """

    if isinstance(rule, Mapping):
        return str(rule.get("op", ">=")), float(rule["value"])
    if isinstance(rule, tuple | list) and len(rule) == 2:
        return str(rule[0]), float(rule[1])
    return ">=", float(rule)


def _compare_threshold(value: Tensor, *, op: str, threshold: float) -> Tensor:
    r"""
    Compare a score tensor against a scalar threshold using a named operator.

    Args:
        value: The score tensor to compare.
        op: One of ``">"``, ``">="``, ``"<"``, ``"<="``.
        threshold: The scalar threshold.

    Returns:
        A boolean tensor of the elementwise comparison.

    Raises:
        ValueError: If ``op`` is not a supported operator.

    Examples:
        >>> import torch
        >>> _compare_threshold(torch.tensor([0.2, 0.9]), op=">=", threshold=0.5).tolist()
        [False, True]
    """

    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"unsupported filter threshold op: {op!r}")


def _merge_score_dicts(
    left: Mapping[str, Tensor] | None,
    right: Mapping[str, Tensor] | None,
) -> dict[str, Tensor] | None:
    r"""
    Merge two optional score dicts, with ``right`` overriding ``left`` on key collisions.

    Args:
        left: Base score mapping, or ``None``.
        right: Overriding score mapping, or ``None``.

    Returns:
        The merged dict, or ``None`` when the result would be empty.

    Examples:
        >>> import torch
        >>> sorted(_merge_score_dicts({"a": torch.tensor(1.0)}, {"b": torch.tensor(2.0)}))
        ['a', 'b']
        >>> _merge_score_dicts(None, None) is None
        True
    """

    merged: dict[str, Tensor] = {}
    if left is not None:
        merged.update(left)
    if right is not None:
        merged.update(right)
    return merged or None


_PROTEINMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_PXDESIGN_RESTYPE_TO_ONE_LETTER = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)
_PXDESIGN_TO_PROTEINMPNN = torch.tensor(
    [
        _PROTEINMPNN_ALPHABET.index(letter)
        for letter in (
            *_PXDESIGN_RESTYPE_TO_ONE_LETTER,
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
        )
    ],
    dtype=torch.long,
)


def _restype_to_mpnn_tokens(restype: Tensor) -> Tensor:
    r"""
    Map PxDesign restype ids (or one-hots) to ProteinMPNN alphabet token ids.

    Argmaxes one-hot restype inputs, clamps ids into range, and gathers through the
    ``_PXDESIGN_TO_PROTEINMPNN`` lookup table.

    Args:
        restype: Restype ids `[...]` or one-hot restypes `[..., num_restypes]`.

    Returns:
        ProteinMPNN token ids of the same leading shape.

    Examples:
        >>> import torch
        >>> _restype_to_mpnn_tokens(torch.tensor([0, 1, 7])).tolist()
        [0, 14, 5]
    """

    if restype.dim() >= 2 and restype.shape[-1] >= _PXDESIGN_TO_PROTEINMPNN.numel():
        restype = restype.argmax(dim=-1)
    restype_ids = restype.to(dtype=torch.long).clamp(min=0, max=_PXDESIGN_TO_PROTEINMPNN.numel() - 1)
    table = _PXDESIGN_TO_PROTEINMPNN.to(device=restype_ids.device)
    return table[restype_ids]


def _protein_like_token_mask(batch: PxDesignInput, *, batch_index: int) -> Tensor:
    r"""
    Mark tokens ProteinMPNN can design: natural amino acids plus PxDesign design placeholders.

    Natural protein restypes (ids ``0..19``) and design-token restypes (ids ``32..35``, plus any
    ``design_token_mask`` entries) are considered protein-like.

    Args:
        batch: Conditioning batch supplying restypes and the optional design-token mask.
        batch_index: Row of ``batch`` to inspect.

    Returns:
        A boolean per-token mask of protein-like tokens.
    """

    restype = batch.restype[batch_index]
    if restype.dim() >= 2 and restype.shape[-1] >= _PXDESIGN_TO_PROTEINMPNN.numel():
        restype = restype.argmax(dim=-1)
    restype_ids = restype.to(dtype=torch.long)
    natural_protein = (restype_ids >= 0) & (restype_ids < 20)
    design_protein = (restype_ids >= 32) & (restype_ids < 36)
    if batch.design_token_mask is not None:
        design_protein = design_protein | batch.design_token_mask[batch_index].to(device=restype_ids.device)
    return natural_protein | design_protein


def _selected_amino_acid_bias(
    amino_acid_bias: Tensor | None,
    *,
    batch_index: int,
    selected_tokens: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    r"""
    Slice a full amino-acid bias down to the selected tokens for one candidate.

    Indexes out the batch row when the bias carries a batch axis, then gathers the selected tokens (unless the
    bias is already provided per selected token) and adds a leading batch axis for ProteinMPNN.

    Args:
        amino_acid_bias: Bias `[B, N_token, 21]`, `[N_token, 21]`, or `[L_selected, 21]`; ``None`` to skip.
        batch_index: Row of the batch axis to select.
        selected_tokens: Index of the selected tokens.
        device: Device to move the bias onto.
        dtype: Dtype to cast the bias to.

    Returns:
        The selected bias `[1, L_selected, 21]`, or ``None`` when no bias is given.

    Raises:
        ValueError: If the bias does not have a valid ``(..., 21)`` shape.
    """

    if amino_acid_bias is None:
        return None
    bias = amino_acid_bias.to(device=device, dtype=dtype)
    if bias.dim() == 3:
        bias = bias[batch_index]
    if bias.dim() != 2 or bias.shape[-1] != 21:
        raise ValueError("amino_acid_bias must have shape (B, N_token, 21), (N_token, 21), or (L_selected, 21)")
    if bias.shape[0] == selected_tokens.numel():
        selected_bias = bias
    else:
        selected_bias = bias.index_select(0, selected_tokens.to(device=device))
    return selected_bias.unsqueeze(0)


__all__ = ["assign_sequences", "filter_candidates"]
