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

Predicted-sequence remap (implemented, validated): released BoltzGen WRITES the designed
sequence back into the output structure — ``DesignWriter.write_on_batch_end`` (inverse_fold)
sets ``sample["ccd"][design_mask] = argmax(res_type)`` (via ``const.tokens`` id->name) before
``Structure.from_feat`` + ``add_side_chains``. :func:`_remap_designed_sequence` reproduces this:
the inverse-fold argmax residue identities are written into the output ``Structure`` at the
design positions (backbone coords carried, designed side chains added at the origin marked
present, per released ``add_side_chains``), so the candidate ``Structure`` carries the DESIGNED
sequence and the refold folds it. Validated exact vs the released ``const.tokens`` / ``convert_ccd``
id->name remap. The per-residue NLL remains exposed as a ``Metric`` axis.

FLAG (node4, parity-untested): the tensor->Structure decode and the designed-backbone ->
inverse-fold re-lowering are best-effort and not numerically verified end to end (the ifold
re-lowering of a de-novo ``UNK`` backbone still needs generated-residue reference coordinates,
a separate node4 concern from the remap).
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from typing_extensions import Self

from ...data.components import get_component
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem, PolymerSequence
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

        # Released design→inverse-fold hand-off: the generated (design) positions the design step
        # marked drive the inverse-fold ``design_mask`` (released reads it from the design step's
        # metadata) and, through it, the predicted-sequence remap in ``_analyze``.
        design_references = _design_references(design_batch)

        candidates: list[BoltzGenCandidate] = []
        for sample_index in range(num_designs):
            backbone = _decode_structure(design_batch, design_output.pred_coords[0, sample_index])
            ifold_batch = self.ifold_processor(backbone, design=design_references)
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
        r"""Assemble one candidate: remap the designed sequence, attach the NLL metric and scores.

        Faithful to released BoltzGen (``DesignWriter.write_on_batch_end`` with ``inverse_fold``):
        the inverse-fold ``argmax`` residue identities are written back into the output Structure at
        the design positions (see :func:`_remap_designed_sequence`), so the candidate carries the
        DESIGNED sequence in its ``Structure`` (not UNK/GLY placeholders) and the refold stage folds
        that designed sequence. The per-residue NLL is additionally exposed as a ``Metric`` axis.
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

        # Released remap: write the inverse-fold argmax residue identities into the output Structure
        # at the design positions, so the candidate carries the DESIGNED sequence (not placeholders)
        # and the refold folds it.
        structure = _remap_designed_sequence(backbone, ifold_batch, ifold_output)
        if do_refold:
            assert self.fold is not None and self.fold_processor is not None
            fold_batch = self.fold_processor(structure)
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


# Model token id -> canonical component id, matching the released ``const.tokens`` ordering
# (0 ``<pad>``, 1 ``-``, 2..21 ALA..VAL, 22 UNK, 23..27 A/G/C/U/N, 28..32 DA/DG/DC/DT/DN) and the
# BoltzGenProcessor residue-type vocabulary. This is the inverse of the released remap's
# ``const.tokens[argmax(res_type)]`` id->name lookup.
_DESIGN_TOKEN_TO_COMPONENT: dict[int, str] = dict(
    enumerate(
        (
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "UNK", "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN",
        ),
        start=2,
    )
)  # fmt: skip


def _design_references(design_batch: BoltzGenInput) -> tuple[ComponentReference, ...]:
    r"""The concrete design (generated) residue references the design step marked.

    Read off ``design_batch.design_mask`` over ``layout.output_residues``; only concrete
    :class:`~ofoldx.data.molecular.ComponentReference` rows are returned (a symbolic generated row
    from a not-yet-materialized assembly is skipped — the decode path always materializes first).
    """

    layout = design_batch.layout
    mask = design_batch.design_mask
    if layout is None or mask is None:
        return ()
    flags = mask[0].bool().tolist()
    return tuple(
        reference
        for reference, flag in zip(layout.output_residues, flags)
        if flag and isinstance(reference, tuple)
    )


def _remap_designed_sequence(
    backbone: Structure,
    ifold_batch: BoltzGenIFoldInput,
    ifold_output: object,
) -> Structure:
    r"""Write the inverse-fold argmax residue identities into the design positions of ``backbone``.

    Faithful port of the released ``DesignWriter`` inverse-fold remap
    (``sample["ccd"][design_mask] = argmax(res_type)`` then ``Structure.from_feat`` +
    ``Structure.add_side_chains``): at every design token, the argmax token id is mapped to its
    canonical component (``const.tokens`` id->name) and written into the output system's residue
    identity; backbone ``{N, CA, C, O}`` coordinates are carried from the sampled backbone and the
    designed residue's remaining heavy atoms are added as origin (zero-coordinate) sites marked
    present — the exact dummy side-chain convention of released ``add_side_chains``. Non-design
    residues (fixed context) are carried verbatim. Returns ``backbone`` unchanged when there is no
    design mask (nothing to remap).
    """

    layout = ifold_batch.layout
    tokens = getattr(ifold_output, "tokens", None)
    design_mask = ifold_batch.design_mask
    if layout is None or tokens is None or design_mask is None:
        return backbone

    token_ids = tokens[0].long().tolist()
    flags = design_mask[0].bool().tolist()
    designed: dict[ComponentReference, str] = {}
    for token_index, reference in enumerate(layout.output_residues):
        if not flags[token_index]:
            continue
        name = _DESIGN_TOKEN_TO_COMPONENT.get(token_ids[token_index])
        if name is not None:
            designed[reference] = name
    if not designed:
        return backbone
    return _write_designed_structure(backbone, designed)


def _write_designed_structure(backbone: Structure, designed: dict[ComponentReference, str]) -> Structure:
    r"""Rebuild ``backbone`` with the designed component identities and dummy designed side chains."""

    system = backbone.system
    design_positions = set(designed)

    # Rebuild entities per instance so each designed residue's identity is independent (the shared
    # entity a de-novo chain rides on is minted per chain, so this only ever splits genuinely shared
    # entities the redesign touches).
    new_entities: dict[str, PolymerSequence | object] = {}
    new_instances: dict[str, str] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            components = list(entity.components)
            changed = False
            for index in range(len(components)):
                name = designed.get((instance_id, index))
                if name is not None:
                    components[index] = get_component(name)
                    changed = True
            if changed:
                new_entity_id = f"{entity_id}__designed_{instance_id}"
                new_entities[new_entity_id] = PolymerSequence(entity.polymer_type, tuple(components))
                new_instances[instance_id] = new_entity_id
                continue
        new_entities.setdefault(entity_id, entity)
        new_instances[instance_id] = entity_id

    def _resolves(candidate_system: MolecularSystem, site: AtomSite) -> bool:
        try:
            candidate_system.atom(site)
        except (KeyError, IndexError):
            return False
        return True

    bondless = MolecularSystem(cast(dict, new_entities), new_instances)
    kept_bonds = frozenset(
        bond for bond in system.bonds if _resolves(bondless, bond.site_a) and _resolves(bondless, bond.site_b)
    )
    new_system = MolecularSystem(cast(dict, new_entities), new_instances, kept_bonds)

    old_coordinates = {
        site: (backbone.coordinates[index], bool(backbone.coordinate_mask[index]))
        for index, site in enumerate(backbone.sites)
    }
    backbone_atoms = {"N", "CA", "C", "O"}
    sites: list[AtomSite] = []
    coordinates: list[np.ndarray] = []
    coordinate_mask: list[bool] = []
    for site, atom in new_system.iter_atoms():
        if atom.element == "H":
            continue
        reference = (site.instance_id, site.component_index)
        if reference in design_positions and site.atom_id not in backbone_atoms:
            # Released ``add_side_chains``: designed side-chain atoms are added at the origin and
            # marked present (the designed sequence is realized; side-chain coords are not predicted).
            coordinate, present = np.zeros(3, dtype=np.float32), True
        else:
            coordinate, present = old_coordinates.get(site, (np.zeros(3, dtype=np.float32), False))
        sites.append(site)
        coordinates.append(np.asarray(coordinate, dtype=np.float32))
        coordinate_mask.append(present)

    return Structure(
        system=new_system,
        sites=tuple(sites),
        coordinates=np.asarray(coordinates, dtype=np.float32).reshape(len(sites), 3),
        coordinate_mask=np.asarray(coordinate_mask, dtype=np.bool_),
    )
