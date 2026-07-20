r"""OFoldX's biological operation boundary for released ESMFold2 folding and binder design.

Owns the one Processor call and the family's released biological verbs. ``fold`` is the
convenience wrapper over ``ESMFold2Processor`` + ``ESMFold2Model.fold``, returning the
shared ``FoldResult`` unchanged. ``design`` runs the released binder-design procedure
(gradient descent on per-position soft-sequence logits through the experimental recurrent
graph, then a discrete refold) and returns the family-owned, coordinate-producing
``ESMFold2DesignResult``. Upstream provenance:
``biohub_esm/cookbook/tutorials/binder_design.py`` (Algorithm 11 gradient-guided sequence
optimisation L828, Algorithm 12 contact losses L260, Algorithm 13 globularity L323,
Algorithm 15 distogram ipTM proxy L423, binder prompt ``'#'`` mutable positions L110).

One Pipeline binds one Model/Processor pair. ``fold`` requires the released
``ESMFold2Model``; ``design`` requires the experimental inversion model
``ESMFold2ExperimentalModel`` — the graph whose distogram is differentiable w.r.t. the
soft sequence. Per DESIGN.md 7.2 the concrete ``register_pipeline`` marker is still owned
by the family package, but both verbs are now implemented.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import Self

from ...data.evidence import MSA, MSACollection
from ...data.molecular import MolecularSystem, PolymerSequence, restype_3to1
from ...data.structure import Structure
from ...references import classify_reference
from ...results import FoldResult, Metric, PairMetric, Score
from .model import (
    ESMFold2DesignCandidate,
    ESMFold2DesignResult,
    ESMFold2ExperimentalModel,
    ESMFold2Input,
    ESMFold2Model,
)
from .processor import ESMFold2ExperimentalProcessor, ESMFold2Processor

# 20-dim design amino-acid space, ordered ALA..VAL (residue-type ids 2..21 minus 2), the
# offset the experimental trunk one-hot uses; ``F.pad(design, (2, 11))`` lifts it to the
# 33-class ``res_type`` space (2 leading pad/gap slots, 11 trailing nucleic slots).
_AA3_ORDER = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
_AA1_TO_IDX = {restype_3to1[three]: index for index, three in enumerate(_AA3_ORDER)}
_IDX_TO_AA1 = {index: restype_3to1[three] for index, three in enumerate(_AA3_ORDER)}
_AA_DIMS = 20
_CYS_IDX = _AA1_TO_IDX["C"]
_MUTABLE_TOKEN = "#"

# Algorithm 11 optimisation constants (binder_design.py L91-L96).
_LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
_LEARNING_RATE = 0.1
_TEMPERATURE_MIN = 1e-2


def _get_mid_points(device: torch.device) -> torch.Tensor:
    r"""128 distance-bin midpoints over the 1-57 angstrom experimental distogram range."""

    boundaries = torch.linspace(2, 52.0, 127, device=device)
    lower = torch.tensor([1.0], device=device)
    upper = torch.tensor([52.0 + 5.0], device=device)
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def _binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def _masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1], device=y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def _masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


def _contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    r"""Algorithm 12 contact losses — entropy contact with a sequence-separation constraint."""

    con_loss = _binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = torch.abs(p_dist) >= min_sep
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = _masked_min_k(con_loss, mask=binder_mask, k=num_contacts)
    return _masked_average(per_residue, mask=chain_mask)


def _structure_losses(distogram_logits: torch.Tensor, binder_length: int) -> dict[str, torch.Tensor]:
    r"""Weighted intra-contact + inter-contact + globularity objective (binder at suffix)."""

    bin_distance = _get_mid_points(distogram_logits.device)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0

    intra = _contact_loss(
        distogram_logits, bin_distance, num_contacts=2, min_sep=9, cutoff=14.0,
        chain_mask=is_binder, binder_mask=is_binder,
    )
    inter = _contact_loss(
        distogram_logits, bin_distance, num_contacts=1, min_sep=0, cutoff=22.0,
        chain_mask=1 - is_binder, binder_mask=is_binder,
    )

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    rg_bins = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(rg_bins), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    glob = F.elu(rg_term - rg_th)

    total = (
        _LOSS_WEIGHTS["intra_contact"] * intra
        + _LOSS_WEIGHTS["inter_contact"] * inter
        + _LOSS_WEIGHTS["glob"] * glob
    )
    return {"intra_contact": intra, "inter_contact": inter, "glob": glob, "total": total}


def _binding_confidence_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _distogram_iptm_proxy(distogram_logits: torch.Tensor, target_length: int, binder_length: int) -> float:
    r"""Algorithm 15 — entropy-of-binding proxy for target|binder ipTM (binder at suffix)."""

    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    bin_distance = _get_mid_points(distogram_logits.device)
    binder_to_target = _binding_confidence_entropy(
        distogram_logits[target_length:, :target_length, :], bin_distance, cutoff=22.0
    )
    sorted_entropies, _ = torch.sort(binder_to_target.reshape(-1))
    k = min(binder_length, sorted_entropies.numel())
    mean_entropy = float(sorted_entropies[:k].mean())
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _normalized_gradient(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    r"""Per-sample length-normalised gradient over the mutable positions (binder_design.py L815)."""

    masked_grad = grad * gradient_mask
    index_has_grad = torch.square(masked_grad).sum(-1) > 0
    eff_len = index_has_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, dim=(-1, -2))
    normalized = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(eff_len[:, None, None])
    return normalized * gradient_mask


def _target_one_letter(target: str | PathLike[str] | Structure | MolecularSystem) -> str:
    r"""Resolve one single-chain protein target to an unambiguous one-letter sequence."""

    system: MolecularSystem | None = None
    if isinstance(target, Structure):
        system = target.system
    elif isinstance(target, MolecularSystem):
        system = target
    if system is not None:
        entities = [entity for entity in system.entities.values() if isinstance(entity, PolymerSequence)]
        proteins = [entity for entity in entities if entity.polymer_type == "protein"]
        if len(proteins) != 1:
            raise NotImplementedError("ESMFold2 design targets one single-chain protein target")
        return "".join(restype_3to1[component.component_id] for component in proteins[0].components)

    classified = classify_reference(target)
    if isinstance(classified, Path):
        from multimolecule.io import read_fasta_records

        records = read_fasta_records(classified)
        return records[0].sequence.strip().upper()
    return str(classified).strip().upper()


class ESMFold2Pipeline:
    r"""Bind one ESMFold2 Model-Processor pair for the released folding and design verbs.

    Marker id ``"ofoldx.esmfold2"``. The Pipeline is not a base class and shares no runtime
    request object with any other family. ``fold`` (released ``ESMFold2Model``) returns the
    shared ``FoldResult``; ``design`` (experimental ``ESMFold2ExperimentalModel``) returns
    the family-owned ``ESMFold2DesignResult``. A given Pipeline is bound with the model the
    verb it will run requires.
    """

    model: ESMFold2Model | ESMFold2ExperimentalModel
    processor: ESMFold2Processor | ESMFold2ExperimentalProcessor

    def __init__(
        self,
        *,
        model: ESMFold2Model | ESMFold2ExperimentalModel,
        processor: ESMFold2Processor | ESMFold2ExperimentalProcessor,
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

        if not isinstance(self.model, ESMFold2Model):
            raise TypeError("fold requires the released ESMFold2Model; this Pipeline is bound for design")

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

        Runs Algorithm 11: gradient descent on per-position soft-sequence logits through the
        experimental recurrent graph under a cosine temperature schedule. ``binder_prompt``
        fixes amino acids at positions and marks mutable positions with ``'#'``
        (``binder_design.py`` L110); ``None`` makes every position mutable. Cysteines are
        excluded from the mutable search when ``remove_cysteines`` is set. Each candidate's
        best-scoring sequence (Algorithm 15 distogram-ipTM proxy) is discretely refolded with
        the experimental confidence head; the candidate owns the complete folded
        ``target|binder`` complex, whose binder length is fixed by ``binder_length`` per call
        but varies across calls — the coordinate-producing, variable-length property carried
        by ``ESMFold2DesignResult``.

        The ESMC pseudo-perplexity sequence regulariser (Algorithm 14) is not applied here;
        this runs the structure objective, which is the released gradient-inversion primitive.
        """

        if not isinstance(self.model, ESMFold2ExperimentalModel):
            raise TypeError(
                "design requires the experimental ESMFold2ExperimentalModel inversion graph; "
                "this Pipeline is bound with the released ESMFold2Model"
            )
        if binder_length <= 0:
            raise ValueError("binder_length must be positive")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")

        prompt = _MUTABLE_TOKEN * binder_length if binder_prompt is None else binder_prompt
        if len(prompt) != binder_length:
            raise ValueError("binder_prompt length must equal binder_length")

        model = self.model
        model.eval().requires_grad_(False)
        device = model.z_init_1.weight.device
        processor = self.processor

        target_sequence = _target_one_letter(target)
        target_length = len(target_sequence)
        target_batch = processor(PolymerSequence.from_one_letter(target_sequence))
        target_soft = F.one_hot(target_batch.res_type.long(), num_classes=33).float().to(device)

        base_generator = generator
        if base_generator is None and seed is not None:
            base_generator = torch.Generator(device="cpu").manual_seed(int(seed))

        candidates: list[ESMFold2DesignCandidate] = []
        for _ in range(num_candidates):
            logits, gradient_mask = self._initial_logits(prompt, remove_cysteines, base_generator, device)
            best_sequence = ""
            best_proxy = -1.0

            optimizer = torch.optim.SGD([logits], lr=_LEARNING_RATE)
            for step in range(num_optimization_steps):
                t = (step + 1) / num_optimization_steps
                temperature = _TEMPERATURE_MIN + (1 - _TEMPERATURE_MIN) * 0.5 * (1 + math.cos(math.pi * t))

                optimizer.zero_grad()
                design = F.softmax(logits / temperature, dim=-1)
                binder_sequence = "".join(_IDX_TO_AA1[int(i)] for i in design.argmax(dim=-1)[0])
                batch = processor([
                    PolymerSequence.from_one_letter(target_sequence),
                    PolymerSequence.from_one_letter(binder_sequence),
                ])
                res_type_soft = torch.cat([target_soft, F.pad(design, (2, 11))], dim=1)

                output = model.forward(
                    batch, res_type_soft=res_type_soft, num_loops=1, num_sampling_steps=1,
                    calculate_confidence=False, return_structure=False,
                )
                distogram = output["distogram_logits"].float()
                losses = _structure_losses(distogram, binder_length)
                grad = torch.autograd.grad(losses["total"].mean(), logits)[0]
                logits.grad = _normalized_gradient(grad, gradient_mask)
                for group in optimizer.param_groups:
                    group["lr"] = _LEARNING_RATE * temperature
                optimizer.step()

                proxy = _distogram_iptm_proxy(distogram.detach(), target_length, binder_length)
                if proxy > best_proxy:
                    best_proxy = proxy
                    best_sequence = binder_sequence

            candidates.append(self._refold(processor, model, target_sequence, best_sequence, target_length, seed))

        return ESMFold2DesignResult(candidates=tuple(candidates))

    def _initial_logits(
        self,
        prompt: str,
        remove_cysteines: bool,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Seed soft-sequence logits and the mutable-position gradient mask from the prompt."""

        length = len(prompt)
        noise = 0.01 * torch.randn(1, length, _AA_DIMS, generator=generator)
        logits = torch.zeros(1, length, _AA_DIMS)
        mask = torch.ones(1, length, _AA_DIMS)
        for i, symbol in enumerate(prompt):
            if symbol == _MUTABLE_TOKEN:
                logits[:, i, :] = noise[:, i, :]
            else:
                if symbol not in _AA1_TO_IDX:
                    raise ValueError(f"binder_prompt symbol {symbol!r} is not a standard amino acid or '#'")
                logits[:, i, _AA1_TO_IDX[symbol]] = 10.0
                mask[:, i, :] = 0.0
        if remove_cysteines:
            logits[:, :, _CYS_IDX] = -1e6
            mask[:, :, _CYS_IDX] = 0.0
        logits = logits.to(device).requires_grad_(True)
        return logits, mask.to(device)

    def _refold(
        self,
        processor: ESMFold2ExperimentalProcessor,
        model: ESMFold2ExperimentalModel,
        target_sequence: str,
        binder_sequence: str,
        target_length: int,
        seed: int | None,
    ) -> ESMFold2DesignCandidate:
        r"""Discretely refold one designed target|binder complex into a scored candidate."""

        batch = processor([
            PolymerSequence.from_one_letter(target_sequence),
            PolymerSequence.from_one_letter(binder_sequence),
        ])
        with torch.no_grad():
            output = model.forward(
                batch, num_loops=None, num_diffusion_samples=1, calculate_confidence=True, seed=seed,
            )
        structure = self._structure(batch, output)
        binder_length = len(binder_sequence)
        proxy = _distogram_iptm_proxy(output["distogram_logits"].float(), target_length, binder_length)
        scores = {
            "iptm": Score(float(output["iptm"][0].item()), preferred="higher"),
            "ptm": Score(float(output["ptm"][0].item()), preferred="higher"),
            "complex_plddt": Score(float(output["complex_plddt"][0].item()), preferred="higher"),
            "distogram_iptm_proxy": Score(proxy, preferred="higher"),
        }
        layout = batch.layout
        atom_count = len(layout.atom_sites)
        metrics = {
            "plddt": Metric(values=output["plddt"][0].float().cpu().numpy(), axis=layout.token_sites),
            "plddt_per_atom": Metric(
                values=output["plddt_per_atom"][0, :atom_count].float().cpu().numpy(),
                axis=layout.atom_sites,
            ),
        }
        pair_metrics = {
            "pae": PairMetric(
                values=output["pae"][0].float().cpu().numpy(),
                row_axis=layout.token_sites,
                column_axis=layout.token_sites,
                unit="angstrom",
            ),
        }
        return ESMFold2DesignCandidate(
            structure=structure, scores=scores, metrics=metrics, pair_metrics=pair_metrics
        )

    @staticmethod
    def _structure(batch: ESMFold2Input, output: dict[str, torch.Tensor]) -> Structure:
        r"""Scatter model atoms onto the complete non-hydrogen sites of the target|binder system."""

        layout = batch.layout
        atom_count = len(layout.atom_sites)
        raw_coordinates = output["sample_atom_coords"][0, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)
        coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        coordinates[model_to_complete] = raw_coordinates
        coordinate_mask[model_to_complete] = True
        return Structure(
            system=layout.system,
            sites=complete_sites,
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        )
