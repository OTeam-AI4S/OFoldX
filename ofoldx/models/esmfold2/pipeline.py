r"""
ESMFold2 structure-prediction and binder-design pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from os import PathLike
from pathlib import Path
from typing import cast

import chanfig
import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import Self

from ...artifacts import save_model
from ...data.evidence import MSA, MSACollection
from ...data.molecular import AMINO_ACID_THREE_TO_ONE, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...hub import resolve_repository, upload_repository
from ...io import read_fasta_records, read_msa
from ...pipelines import PIPELINES, PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldResult, Metric, PairMetric, Score
from ...results.values import best
from .model import (
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalSample,
    EsmFold2Input,
    EsmFold2Model,
    compute_esmc_masked_lm_logits,
)
from .processor import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    molecular_system_from_named_sequences,
    molecular_system_from_sequences,
)

# 20-dim design amino-acid space, ordered ALA..VAL (residue-type ids 2..21 minus 2), the
# offset used by the experimental trunk's residue-type features. ``F.pad(design, (2, 11))``
# lifts it to the 33-class ``residue_type`` space (2 leading pad/gap slots, 11 trailing
# nucleic slots).
_AA3_ORDER = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
_AMINO_ACID_TO_INDEX = {AMINO_ACID_THREE_TO_ONE[three]: index for index, three in enumerate(_AA3_ORDER)}
_INDEX_TO_AMINO_ACID = {index: AMINO_ACID_THREE_TO_ONE[three] for index, three in enumerate(_AA3_ORDER)}
_AA_DIMS = 20
_CYSTEINE_INDEX = _AMINO_ACID_TO_INDEX["C"]
_MUTABLE_TOKEN = "#"

# Binder-design optimization constants.
_LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
_LEARNING_RATE = 0.1
_TEMPERATURE_MIN = 1e-2

# ESMC pseudo-perplexity regularizer mask-sampling constants. The gradient weight blending
# this masked-LM prior into the structure objective is stored in
# ``EsmFold2ExperimentalConfig.language_model_weight`` and is not selected per call.
_ESMC_MASK_FRACTION = 0.15
_LM_MASK_PASSES = 4
_LM_LOSS_CHUNK = 128


class EsmFold2Pipeline(PairedPipeline[EsmFold2Model, EsmFold2Processor]):
    r"""
    Structure prediction pipeline for ESMFold 2.
    """

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
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
        r"""
        Predict structures from a sequence or molecular input.

        Args:
            input: Sequence text, FASTA path, one or more protein sequences, molecular
                system, or structure topology. Coordinates in a ``Structure`` are ignored.
            msa: Optional MSA evidence or alignment path. Configurations without MSA
                conditioning reject this argument.
            num_samples: Number of coordinate samples and returned predictions.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.
            recycling_steps: Number of recurrent trunk passes. ``None`` uses the model
                configuration.
            num_sampling_steps: Number of diffusion sampling steps. ``None`` uses the
                model configuration.
            language_model_mask_probability: Probability of masking each nonspecial
                language-model token. ``None`` uses the model configuration.
            language_model_dropout: Dropout probability for language-model pair features.
                ``None`` uses the model configuration.
            msa_max_depth: Maximum number of MSA rows sampled per recurrent pass. ``None``
                keeps all rows.
            msa_column_mask_rate: Probability of masking a column in nonquery MSA rows.
            msa_subsample_at_inference: Whether to resample MSA rows on each recurrent pass.
            max_inference_sigma: Largest diffusion sigma included in sampling. ``None``
                keeps the full schedule.
            noise_scale: Sampler noise multiplier. ``None`` uses the model configuration.
            step_scale: Sampler step-size multiplier. ``None`` uses the model configuration.
            denoising_early_exit_rmsd: Aligned RMSD threshold, in angstroms, for ending
                denoising early. ``None`` disables early exit.

        Returns:
            A [`FoldResult`][ofoldx.results.FoldResult] containing predictions in sampling
            order.
        """

        if msa is not None and not self.processor.config.msa_conditioning:
            raise ValueError("this ESMFold2 checkpoint is the single-sequence Fast variant")
        system = _fold_system(input)
        msa = _fold_msa(msa, system)
        generator = resolve_generator(seed, generator, device=self.processor.device)
        batch = self.processor(system, msa=msa)
        return self.model.fold(
            batch,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
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


class EsmFold2DesignPipeline(PairedPipeline[EsmFold2ExperimentalModel, EsmFold2ExperimentalProcessor]):
    r"""
    Binder sequence design pipeline for experimental ESMFold 2.
    """

    @classmethod
    def from_repository(
        cls,
        repository: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Load the design Pipeline and its ESMC masked-LM child.

        Args:
            repository: Local directory or Hub repository id.
            revision: Optional Hub revision.
            token: Optional Hub authentication token.
            device: Device for the loaded Models and Processor.
            dtype: Floating-point dtype for loaded Models and Processor.

        Returns:
            An ``EsmFold2DesignPipeline`` with the masked-LM child attached.
        """

        from multimolecule import AutoTokenizer, ProteinTokenizer
        from transformers import AutoModelForMaskedLM

        root = resolve_repository(repository, revision=revision, token=token).path
        pipeline = super().from_repository(root, token=token, device=device, dtype=dtype)
        esmc_repository = root / "esmc"
        pipeline.processor.tokenizer = cast(
            ProteinTokenizer,
            AutoTokenizer.from_pretrained(esmc_repository, local_files_only=True),
        )
        esmc = AutoModelForMaskedLM.from_pretrained(esmc_repository, local_files_only=True, dtype=dtype)
        if device is not None:
            esmc.to(device=device)
        pipeline.model.load_esmc(esmc)
        return pipeline

    def save(
        self,
        directory: str | PathLike[str],
        *,
        max_shard_size: int | str = "5GB",
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
    ) -> None:
        r"""
        Save the experimental design pair, ESMC child, and tokenizer.

        Args:
            directory: Local repository directory.
            max_shard_size: Maximum size of each model-weight shard.
            repo_id: Optional Hub repository receiving the saved directory.
            token: Optional Hub authentication token.
            private: Whether a newly created Hub repository is private.
            commit_message: Optional Hub commit message.
        """

        from huggingface_hub import save_torch_state_dict

        pipeline_id = PIPELINES.id_for(type(self))
        if pipeline_id is None:
            raise ValueError(f"{type(self).__name__} is not a registered concrete Pipeline; it cannot be saved")
        esmc = self.model.esmc
        lm_head = self.model.esmc_lm_head
        if esmc is None or lm_head is None:
            raise RuntimeError("saving ESMFold2 design requires an attached EsmCForMaskedLM")

        destination = Path(directory).expanduser()
        self.model.esmc = None
        self.model.esmc_lm_head = None
        try:
            save_model(
                destination,
                model=self.model,
                processor=self.processor,
                max_shard_size=max_shard_size,
            )
        finally:
            self.model.esmc = esmc
            self.model.esmc_lm_head = lm_head

        esmc_destination = destination / "esmc"
        esmc_destination.mkdir(parents=True, exist_ok=True)
        esmc.config.save_pretrained(esmc_destination)
        esmc_state = {f"model.{name}": value for name, value in esmc.state_dict().items()}
        esmc_state.update({f"lm_head.{name}": value for name, value in lm_head.state_dict().items()})
        save_torch_state_dict(
            esmc_state,
            esmc_destination,
            filename_pattern="model{suffix}.safetensors",
            max_shard_size=max_shard_size,
            safe_serialization=True,
        )
        self.processor.tokenizer.save_pretrained(esmc_destination)
        chanfig.NestedDict({"pipeline": pipeline_id}).dump(destination / "pipeline.json")

        if repo_id is not None:
            upload_repository(
                destination,
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )

    def design(
        self,
        input: str | PathLike[str] | Structure | MolecularSystem,
        *,
        binder_length: int,
        binder_prompt: str | None = None,
        num_candidates: int = 1,
        num_optimization_steps: int = 150,
        remove_cysteines: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> EsmFold2DesignResult:
        r"""
        Design and refold binder candidates against one single-chain protein target.

        Args:
            input: One-letter sequence, FASTA path, ``Structure``, or ``MolecularSystem``
                resolving to one protein chain.
            binder_length: Number of residues in each designed binder.
            binder_prompt: Optional prompt of length ``binder_length``. One-letter amino acids
                fix positions and ``'#'`` marks mutable positions; ``None`` makes every position
                mutable. ``antibody_cdr_prompt`` can construct a CDR-only prompt.
            num_candidates: Number of independently optimized and refolded candidates.
            num_optimization_steps: Gradient optimization steps per candidate.
            remove_cysteines: Whether to exclude cysteine from mutable positions.
            seed: Optional integer seed, mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            An ``EsmFold2DesignResult`` containing the generation-ordered folded candidates.

        Notes:
            Optimization updates soft residue logits at mutable positions, retains each
            candidate's best distogram-ipTM proxy, and refolds the resulting discrete sequence.
            The optional masked-LM gradient uses the Model's ``language_model_weight``; a nonzero
            weight requires an attached masked-LM head.
        """

        if binder_length <= 0:
            raise ValueError("binder_length must be positive")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if num_optimization_steps <= 0:
            raise ValueError("num_optimization_steps must be positive")
        prompt = _MUTABLE_TOKEN * binder_length if binder_prompt is None else binder_prompt
        if len(prompt) != binder_length:
            raise ValueError("binder_prompt length must equal binder_length")

        model = self.model
        # The sequence-prior weight is fixed by the Model configuration rather than inferred from
        # the input or prompt.
        language_model_weight = model.config.language_model_weight
        if language_model_weight != 0 and model.esmc_lm_head is None:
            raise RuntimeError("binder design with a nonzero language_model_weight requires an EsmCForMaskedLM")
        processor = self.processor
        base_generator = resolve_generator(seed, generator, device=model.left_pair_projection.weight.device)
        with _frozen_eval(model):
            return self._run_design(
                model=model,
                processor=processor,
                input=input,
                prompt=prompt,
                binder_length=binder_length,
                num_candidates=num_candidates,
                num_optimization_steps=num_optimization_steps,
                remove_cysteines=remove_cysteines,
                language_model_weight=language_model_weight,
                generator=base_generator,
            )

    def _run_design(
        self,
        *,
        model: EsmFold2ExperimentalModel,
        processor: EsmFold2ExperimentalProcessor,
        input: str | PathLike[str] | Structure | MolecularSystem,
        prompt: str,
        binder_length: int,
        num_candidates: int,
        num_optimization_steps: int,
        remove_cysteines: bool,
        language_model_weight: float,
        generator: torch.Generator | None,
    ) -> EsmFold2DesignResult:
        device = model.left_pair_projection.weight.device

        target_sequence = _target_one_letter(input)
        target_length = len(target_sequence)
        target_batch = processor(PolymerSequence.from_one_letter(target_sequence))
        target_residue_type_features = (
            F.one_hot(target_batch.residue_type_ids.long(), num_classes=33).float().to(device)
        )

        candidates: list[EsmFold2DesignCandidate] = []
        for _ in range(num_candidates):
            logits, gradient_mask = self._initial_logits(prompt, remove_cysteines, generator, device)
            best_sequence = ""
            best_proxy = -math.inf

            optimizer = torch.optim.SGD([logits], lr=_LEARNING_RATE)
            for step in range(num_optimization_steps):
                t = (step + 1) / num_optimization_steps
                temperature = _TEMPERATURE_MIN + (1 - _TEMPERATURE_MIN) * 0.5 * (1 + math.cos(math.pi * t))

                optimizer.zero_grad()
                design = F.softmax(logits / temperature, dim=-1)
                binder_sequence = "".join(_INDEX_TO_AMINO_ACID[int(i)] for i in design.argmax(dim=-1)[0])
                batch = processor(
                    [
                        PolymerSequence.from_one_letter(target_sequence),
                        PolymerSequence.from_one_letter(binder_sequence),
                    ]
                )
                residue_type_probabilities = torch.cat([target_residue_type_features, F.pad(design, (2, 11))], dim=1)

                distogram = model.forward(
                    batch,
                    residue_type_probabilities=residue_type_probabilities,
                    recycling_steps=2,
                    generator=generator,
                ).float()
                losses = _structure_losses(distogram, binder_length)
                structure_grad = torch.autograd.grad(losses["total"].mean(), logits)[0]
                combined_grad = _normalized_gradient(structure_grad, gradient_mask)

                # Add the ESMC pseudo-perplexity regularizer whenever a masked-LM head is attached.
                # Each gradient is length-normalized over mutable positions before summation.
                if model.esmc_lm_head is not None:
                    design_lm = F.softmax(logits / temperature, dim=-1)
                    score_mask = gradient_mask.sum(dim=-1) > 0
                    plm_loss = _esmc_pseudo_perplexity_nll(
                        model,
                        design_lm,
                        score_mask,
                        generator=generator,
                    )
                    plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]
                    plm_term = _normalized_gradient(plm_grad, gradient_mask)
                    combined_grad = combined_grad + language_model_weight * plm_term

                logits.grad = combined_grad
                for group in optimizer.param_groups:
                    group["lr"] = _LEARNING_RATE * temperature
                optimizer.step()

                proxy = _distogram_iptm_proxy(distogram.detach(), target_length, binder_length)
                if step == 0 or proxy > best_proxy:
                    best_proxy = proxy
                    best_sequence = binder_sequence

            candidates.append(
                self._refold(
                    processor,
                    model,
                    target_sequence,
                    best_sequence,
                    target_length,
                    generator,
                )
            )

        return EsmFold2DesignResult(candidates=tuple(candidates))

    def _initial_logits(
        self,
        prompt: str,
        remove_cysteines: bool,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Seed soft-sequence logits and the mutable-position gradient mask from the prompt.
        """

        length = len(prompt)
        noise = 0.01 * torch.randn(
            1,
            length,
            _AA_DIMS,
            generator=generator,
            device=device,
        )
        logits = torch.zeros(1, length, _AA_DIMS, device=device)
        mask = torch.ones(1, length, _AA_DIMS, device=device)
        for i, symbol in enumerate(prompt):
            if symbol == _MUTABLE_TOKEN:
                logits[:, i, :] = noise[:, i, :]
                if remove_cysteines:
                    logits[:, i, _CYSTEINE_INDEX] = -1e6
                    mask[:, i, _CYSTEINE_INDEX] = 0.0
            else:
                if symbol not in _AMINO_ACID_TO_INDEX:
                    raise ValueError(f"binder_prompt symbol {symbol!r} is not a standard amino acid or '#'")
                logits[:, i, _AMINO_ACID_TO_INDEX[symbol]] = 10.0
                mask[:, i, :] = 0.0
        return logits.requires_grad_(True), mask

    def _refold(
        self,
        processor: EsmFold2ExperimentalProcessor,
        model: EsmFold2ExperimentalModel,
        target_sequence: str,
        binder_sequence: str,
        target_length: int,
        generator: torch.Generator | None,
    ) -> EsmFold2DesignCandidate:
        r"""
        Discretely refold one designed target|binder complex into a scored candidate.
        """

        batch = processor(
            [
                PolymerSequence.from_one_letter(target_sequence),
                PolymerSequence.from_one_letter(binder_sequence),
            ]
        )
        with torch.no_grad():
            prediction = model.sample_with_confidence(
                batch,
                recycling_steps=None,
                num_diffusion_samples=1,
                generator=generator,
            )
        sample = prediction.sample
        confidence = prediction.confidence
        structure = self._structure(batch, sample)
        binder_length = len(binder_sequence)
        proxy = _distogram_iptm_proxy(sample.distogram_logits.float(), target_length, binder_length)
        scores = {
            "iptm": Score(float(confidence.iptm[0].item()), preferred="higher"),
            "ptm": Score(float(confidence.ptm[0].item()), preferred="higher"),
            "mean_plddt": Score(float(confidence.mean_plddt[0].item()) * 100.0, preferred="higher"),
            "distogram_iptm_proxy": Score(proxy, preferred="higher"),
        }
        layout = batch.layout
        atom_count = len(layout.atom_sites)
        metrics = {
            "plddt": Metric(
                values=confidence.plddt[0].float().cpu().numpy() * 100.0,
                axis=layout.token_sites,
            ),
            "plddt_per_atom": Metric(
                values=confidence.plddt_per_atom[0, :atom_count].float().cpu().numpy() * 100.0,
                axis=layout.atom_sites,
            ),
        }
        pair_metrics = {
            "pae": PairMetric(
                values=confidence.pae[0].float().cpu().numpy(),
                row_axis=layout.token_sites,
                column_axis=layout.token_sites,
                unit="angstrom",
            ),
        }
        return EsmFold2DesignCandidate(structure=structure, scores=scores, metrics=metrics, pair_metrics=pair_metrics)

    @staticmethod
    def _structure(input: EsmFold2Input, sample: EsmFold2ExperimentalSample) -> Structure:
        r"""
        Scatter model atoms onto the complete non-hydrogen sites of the target|binder system.
        """

        layout = input.layout
        atom_count = len(layout.atom_sites)
        raw_coordinates = sample.coordinates[0, :atom_count].to(device="cpu", dtype=torch.float32).numpy()
        complete_sites = [site for site, atom in layout.system.iter_atoms() if atom.element != "H"]
        complete_index = {site: index for index, site in enumerate(complete_sites)}
        model_to_complete = np.asarray([complete_index[site] for site in layout.atom_sites], dtype=np.intp)
        coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
        coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
        coordinates[model_to_complete] = raw_coordinates
        coordinate_mask[model_to_complete] = True
        return Structure(
            system=layout.system,
            sites=tuple(complete_sites),
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
        )


@dataclass(eq=False)
class EsmFold2DesignCandidate:
    r"""
    One folded target-binder complex and its aligned scientific values.

    Args:
        structure: Folded target-binder complex.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
        pair_metrics: Two-axis metrics keyed by stable scientific names.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)
    pair_metrics: dict[str, PairMetric] = field(default_factory=dict)


@dataclass(eq=False)
class EsmFold2DesignResult:
    r"""
    A generation-ordered collection of folded binder designs.

    Args:
        candidates: Folded target-binder complexes in generation order.
    """

    candidates: tuple[EsmFold2DesignCandidate, ...]

    def best(self, score: str) -> EsmFold2DesignCandidate:
        r"""
        Return the earliest best candidate under a named score.

        Args:
            score: Comparable score name present on every candidate.

        Returns:
            The best candidate, with generation order breaking ties.
        """

        return best(self.candidates, score)


@contextmanager
def _frozen_eval(model: torch.nn.Module) -> Iterator[None]:
    r"""
    Run one design call in frozen eval mode, then restore the exact module state.
    """

    training = tuple((module, module.training) for module in model.modules())
    requires_grad = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    model.eval().requires_grad_(False)
    try:
        yield
    finally:
        for parameter, enabled in requires_grad:
            parameter.requires_grad_(enabled)
        for module, enabled in training:
            module.training = enabled


def _fold_system(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem:
    r"""
    Resolve the fold Pipeline's inline-sequence and FASTA convenience.
    """

    if isinstance(input, Structure):
        return input.system
    if isinstance(input, MolecularSystem):
        return input
    if not isinstance(input, str | PathLike):
        return molecular_system_from_sequences(input)

    classified = classify_reference(input)
    if isinstance(classified, Path):
        named = [
            (
                record.id or f"chain_{index + 1}",
                PolymerSequence.from_one_letter(record.sequence.strip().upper()),
            )
            for index, record in enumerate(read_fasta_records(classified))
        ]
    else:
        named = [("protein", PolymerSequence.from_one_letter(classified.strip().upper()))]
    return molecular_system_from_named_sequences(named)


def _fold_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Read path evidence once and bind a single-entity alignment to its protein.
    """

    if msa is None or isinstance(msa, MSA | MSACollection):
        return msa
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an alignment path is unambiguous only for one protein entity; pass MSACollection")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])


def _target_one_letter(input: str | PathLike[str] | Structure | MolecularSystem) -> str:
    r"""
    Resolve one single-chain protein target to an unambiguous one-letter sequence.
    """

    system: MolecularSystem | None = None
    if isinstance(input, Structure):
        system = input.system
    elif isinstance(input, MolecularSystem):
        system = input
    if system is not None:
        entities = [entity for entity in system.entities.values() if isinstance(entity, PolymerSequence)]
        proteins = [entity for entity in entities if entity.polymer_type == "protein"]
        if len(proteins) != 1:
            raise NotImplementedError("ESMFold2 design targets one single-chain protein target")
        return "".join(AMINO_ACID_THREE_TO_ONE[component.component_id] for component in proteins[0].components)

    classified = classify_reference(cast(str | PathLike[str], input))
    if isinstance(classified, Path):
        records = tuple(read_fasta_records(classified))
        if len(records) != 1:
            raise ValueError("ESMFold2 design requires exactly one target FASTA record")
        return records[0].sequence.strip().upper()
    return str(classified).strip().upper()


@cache
def _language_model_amino_acid_layout() -> tuple[int, int, tuple[int, ...]]:
    r"""
    Locate the 20 standard amino acids in the MultiMolecule ESMC vocabulary.

    Returns ``(amino_acid_start, amino_acid_stop, permutation)`` where the returned range is the
    contiguous vocabulary block holding the 20 standard amino acids and
    ``permutation[folding_trunk_index]`` is the column of the folding-trunk amino acid
    ``folding_trunk_index`` (``_AA3_ORDER`` order) inside that block. The permutation is derived
    from the tokenizer rather than assuming fixed vocabulary ids.
    """

    from multimolecule import ProteinTokenizer

    vocabulary = ProteinTokenizer().get_vocab()
    amino_acid_token_indices = {amino_acid: vocabulary[amino_acid] for amino_acid in _INDEX_TO_AMINO_ACID.values()}
    amino_acid_start = min(amino_acid_token_indices.values())
    amino_acid_stop = max(amino_acid_token_indices.values()) + 1
    if amino_acid_stop - amino_acid_start != _AA_DIMS:
        raise ValueError("ESMC standard amino-acid vocabulary ids are not contiguous")
    language_model_order = [
        amino_acid
        for amino_acid, _ in sorted(
            amino_acid_token_indices.items(),
            key=lambda item: item[1],
        )
    ]
    permutation = tuple(language_model_order.index(_INDEX_TO_AMINO_ACID[index]) for index in range(_AA_DIMS))
    return amino_acid_start, amino_acid_stop, permutation


def _folding_trunk_to_language_model_amino_acid_matrix(device: torch.device) -> torch.Tensor:
    r"""
    ``[20, 20]`` permutation from folding-trunk amino-acid order to ESMC LM order.
    """

    _, _, permutation = _language_model_amino_acid_layout()
    matrix = torch.zeros(_AA_DIMS, _AA_DIMS, device=device)
    for folding_trunk_index, language_model_index in enumerate(permutation):
        matrix[folding_trunk_index, language_model_index] = 1.0
    return matrix


def _hard_categorical_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probabilities, dim=-1), num_classes=probabilities.size(-1)).to(probabilities.dtype)


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    r"""
    Discrete value on the forward pass, continuous gradient on the backward pass.
    """

    return continuous + (discrete - continuous).detach()


def _esmc_pseudo_perplexity_nll(
    model: EsmFold2ExperimentalModel,
    design: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    generator: torch.Generator | None,
    n_passes: int = _LM_MASK_PASSES,
    mask_fraction: float = _ESMC_MASK_FRACTION,
    chunk_size: int = _LM_LOSS_CHUNK,
) -> torch.Tensor:
    r"""
    Approximate ESMC pseudo-perplexity NLL of a soft designed sequence.

    ``design`` is ``[B, L, 20]`` soft amino-acid probabilities in folding-trunk order (the
    binder's per-position softmax). ``score_mask`` is ``[B, L]`` marking the mutable
    positions eligible for masking. Builds a differentiable categorical ESMC input (with a
    straight-through discretisation), samples ``n_passes`` random ``mask_fraction`` masks over
    the mutable positions, and returns the per-sample mean masked-token NLL of the designed
    distribution under the frozen ESMC masked-LM head. Gradient flows back to ``design``
    through the frozen trunk, giving a differentiable sequence-naturalness prior.
    """

    device = design.device
    esmc = model.esmc
    lm_head = model.esmc_lm_head
    if esmc is None or lm_head is None:
        raise RuntimeError("ESMC pseudo-perplexity regulariser requires an attached ESMC masked-LM head")
    esmc_config = model.config.esmc
    vocabulary_size = esmc_config.vocab_size
    embed_dtype = esmc.get_input_embeddings().weight.dtype
    amino_acid_start, amino_acid_stop, _ = _language_model_amino_acid_layout()
    folding_trunk_to_language_model = _folding_trunk_to_language_model_amino_acid_matrix(device)

    batch_size, binder_length = design.shape[0], design.shape[1]
    target_language_model_probabilities = design @ folding_trunk_to_language_model
    input_language_model_probabilities = _straight_through(
        _hard_categorical_probabilities(target_language_model_probabilities),
        target_language_model_probabilities,
    )

    input_token_probabilities = torch.zeros(
        batch_size,
        binder_length + 2,
        vocabulary_size,
        dtype=embed_dtype,
        device=device,
    )
    input_token_probabilities[:, 0, esmc_config.bos_token_id] = 1
    input_token_probabilities[:, -1, esmc_config.eos_token_id] = 1
    input_token_probabilities[:, 1:-1, amino_acid_start:amino_acid_stop] = input_language_model_probabilities.to(
        embed_dtype
    )

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(batch_size, -1)
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(vocabulary_size, dtype=embed_dtype, device=device)
    mask_token[esmc_config.mask_token_id] = 1

    all_masked_rows: list[torch.Tensor] = []
    all_pass_masks: list[torch.Tensor] = []
    for sample in range(batch_size):
        position_indices = score_mask[sample].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError("ESMC pseudo-perplexity score mask selected zero positions")
        num_masked = max(1, math.ceil(mask_fraction * num_positions))
        random_scores = torch.rand(
            (n_passes, num_positions),
            device=device,
            generator=generator,
        )
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros((n_passes, binder_length), dtype=torch.bool, device=device)
        pass_masks[torch.arange(n_passes, device=device)[:, None], position_indices[masked_offsets]] = True
        masked = input_token_probabilities[sample : sample + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked[mask_rows, mask_cols + 1] = mask_token
        all_masked_rows.append(masked)
        all_pass_masks.append(pass_masks)

    masked_sequences = torch.cat(all_masked_rows, dim=0)
    language_model_mask = torch.ones(
        masked_sequences.shape[0],
        masked_sequences.shape[1],
        device=device,
    )
    logit_chunks: list[torch.Tensor] = []
    for start in range(0, masked_sequences.shape[0], chunk_size):
        chunk = masked_sequences[start : start + chunk_size]
        logit_chunks.append(
            compute_esmc_masked_lm_logits(
                esmc,
                lm_head,
                chunk,
                language_model_mask[start : start + chunk_size],
            )
        )
    logits = torch.cat(logit_chunks, dim=0)

    losses: list[torch.Tensor] = []
    for sample, pass_masks in enumerate(all_pass_masks):
        start = sample * n_passes
        stop = start + n_passes
        target_weights = target_language_model_probabilities[sample]
        log_probs = logits[start:stop].log_softmax(dim=-1)[
            :,
            1:-1,
            amino_acid_start:amino_acid_stop,
        ]
        nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
        losses.append(nlls[pass_masks].mean())
    return torch.stack(losses, dim=0)


def _get_mid_points(device: torch.device) -> torch.Tensor:
    r"""
    128 distance-bin midpoints over the 1-57 angstrom experimental distogram range.
    """

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
    r"""
    Compute entropy contact loss with a sequence-separation constraint.
    """

    con_loss = _binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = torch.abs(p_dist) >= min_sep
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = _masked_min_k(con_loss, mask=binder_mask, k=num_contacts)
    return _masked_average(per_residue, mask=chain_mask)


def _structure_losses(distogram_logits: torch.Tensor, binder_length: int) -> dict[str, torch.Tensor]:
    r"""
    Weighted intra-contact + inter-contact + globularity objective (binder at suffix).
    """

    bin_distance = _get_mid_points(distogram_logits.device)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0

    intra = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )
    inter = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
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
        _LOSS_WEIGHTS["intra_contact"] * intra + _LOSS_WEIGHTS["inter_contact"] * inter + _LOSS_WEIGHTS["glob"] * glob
    )
    return {"intra_contact": intra, "inter_contact": inter, "glob": glob, "total": total}


def _binding_confidence_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _distogram_iptm_proxy(distogram_logits: torch.Tensor, target_length: int, binder_length: int) -> float:
    r"""
    Return an entropy-of-binding proxy for target-binder ipTM.
    """

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


def cdr_distogram_iptm_proxy(distogram_logits: torch.Tensor, target_length: int, binder_sequence: str) -> float:
    r"""
    Score target contacts made by antibody CDR residues.

    Args:
        distogram_logits: Pairwise distance-bin logits with shape ``(N, N, D)`` or
            ``(B, N, N, D)``. For a four-dimensional tensor, only the first batch item
            is scored.
        target_length: Number of target tokens preceding the binder tokens.
        binder_sequence: Antibody Fv sequence aligned to the binder-token rows.

    Returns:
        An entropy-based contact confidence in the closed interval ``[0, 1]``.
    """

    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    bin_distance = _get_mid_points(distogram_logits.device)
    cdr_indices = antibody_cdr_indices(binder_sequence)
    cdr_rows = [target_length + i for i in cdr_indices]
    cdr_to_target = _binding_confidence_entropy(
        distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
    )
    sorted_entropies, _ = torch.sort(cdr_to_target.reshape(-1))
    k = min(len(cdr_indices), sorted_entropies.numel())
    mean_entropy = float(sorted_entropies[:k].mean())
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _normalized_gradient(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    r"""
    Return a per-sample length-normalized gradient over mutable positions.
    """

    masked_grad = grad * gradient_mask
    index_has_grad = torch.square(masked_grad).sum(-1) > 0
    eff_len = index_has_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, dim=(-1, -2))
    normalized = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(eff_len[:, None, None])
    return normalized * gradient_mask
