# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

"""Behavior of ofoldx.models.boltzgen.pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.boltzgen.model import BoltzGenDesignMask, BoltzGenInverseFoldingModel, BoltzGenModel
from ofoldx.models.boltzgen.pipeline import (
    BoltzGenBackbonePipeline,
    BoltzGenCandidateResult,
    BoltzGenDesignCandidate,
    BoltzGenInverseFoldingPipeline,
    BoltzGenPipeline,
)
from ofoldx.models.boltzgen.processor import (
    BoltzGenInverseFoldingProcessor,
    BoltzGenProcessor,
    decode_boltzgen_atom14,
)
from ofoldx.models.conditions import GeneratedSegment, SymbolicBond
from ofoldx.models.design import PositionSelection
from ofoldx.models.preparation import PreparationContext, PreparedInput, RegenerationScope

PLACEMENT = {"binder": {"binder": GeneratedSegment("protein", 2, 2)}}


@pytest.fixture
def composite_flow(peptide, monkeypatch):
    structure = peptide("ALA", "GLY")

    class Generator:
        processor = SimpleNamespace(device=torch.device("cpu"))
        identity = "ALA"

        def prepare(self, *args, **kwargs):
            return SimpleNamespace(msa=None)

        def forward(self, input, *, num_candidates, **kwargs):
            return tuple(
                BoltzGenDesignCandidate(
                    structure=structure,
                    codesign_structure=structure,
                    atom14_residue_types=(self.identity,),
                    atom14_recognized=(self.identity != "UNK",),
                    source_checkpoint=None,
                    designed_positions=(ComponentRef("A", 0),),
                    connected_designed_positions=(ComponentRef("A", 0),),
                )
                for _ in range(num_candidates)
            )

        def postprocess(self, input, output):
            return BoltzGenCandidateResult(output)

    diverse, adherence = Generator(), Generator()
    pipeline = BoltzGenPipeline(
        diverse_generator=diverse,
        adherence_generator=adherence,
        inverse_folder=object(),
        soluble_designer=object(),
        folder=object(),
        affinity_scorer=object(),
        esmfold2=object(),
    )
    calls = []

    def redesign(candidate, *, num_sequences, inverse_fold_model, **kwargs):
        calls.append(inverse_fold_model)
        return BoltzGenCandidateResult(tuple(candidate.copied() for _ in range(num_sequences)))

    monkeypatch.setattr(pipeline, "inverse_fold", redesign)
    monkeypatch.setattr(pipeline, "_fold_candidate", lambda candidate, **kwargs: candidate)
    monkeypatch.setattr(pipeline, "score_esmfold2", lambda candidate, **kwargs: candidate)
    return pipeline, diverse, calls, structure


def test_composite_auto_inverse_folding_expands_candidates(composite_flow):
    pipeline, _, calls, structure = composite_flow
    prepared = pipeline.prepare(structure, num_candidates=2, num_sequences_per_backbone=3, seed=7)
    result = pipeline.forward(prepared, num_sampling_steps=1)
    assert len(result) == 6 and calls == ["solublempnn", "solublempnn"]
    assert all(candidate.codesign_structure is structure for candidate in result)


def test_composite_can_keep_codesign_without_inverse_folding(composite_flow):
    pipeline, _, calls, structure = composite_flow
    prepared = pipeline.prepare(structure, num_candidates=2, inverse_fold_model=None, seed=7)
    result = pipeline.forward(prepared, num_sampling_steps=1)
    assert len(result) == 2 and calls == []
    assert all(candidate.structure is structure and candidate.codesign_structure is structure for candidate in result)


def test_composite_without_inverse_folding_refuses_unknown_or_sequence_controls(composite_flow):
    pipeline, diverse, _, structure = composite_flow
    with pytest.raises(InvalidRequestError, match="value_invalid"):
        pipeline.prepare(structure, inverse_fold_model=None, num_sequences_per_backbone=2)
    prepared = pipeline.prepare(structure, inverse_fold_model=None)
    with pytest.raises(InvalidRequestError, match="value_invalid"):
        pipeline.forward(prepared, global_omit_tokens=("C",))
    diverse.identity = "UNK"
    with pytest.raises(InvalidRequestError, match="value_invalid"):
        pipeline.forward(prepared, num_sampling_steps=1)


def test_composite_inverse_folding_model_selection(composite_flow):
    pipeline, _, _, structure = composite_flow
    assert pipeline.prepare(structure, protocol="protein-small_molecule").inverse_model == "boltzif"
    assert pipeline.prepare(structure, protocol="protein-anything").inverse_model == "solublempnn"
    assert pipeline.prepare(structure, protocol="protein-small_molecule", inverse_fold_model=None).inverse_model is None
    with pytest.raises(InvalidRequestError, match="value_invalid"):
        pipeline.prepare(structure, protocol="protein-small_molecule", inverse_fold_model="solublempnn")
    with pytest.raises(InvalidRequestError, match="value_invalid"):
        pipeline.prepare(structure, inverse_fold_model="invalid")


def test_atom14_identity_markers_and_boundary():
    anchors = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    raw = torch.full((5, 14, 3), 20.0)
    raw[:, :4] = anchors
    raw[0, 5:14] = anchors[3]  # ALA: nine markers at O
    raw[1, 4:14] = anchors[3]  # GLY: ten markers at O
    raw[2, 9:14] = anchors[0]  # LYS: five markers at N
    raw[3, 5:14] = anchors[0]  # recognized UNK: nine markers at N
    raw[4, 4] = anchors[1]  # an invalid signature involving CA
    assert decode_boltzgen_atom14(raw) == (
        ("ALA", "GLY", "LYS", "UNK", "UNK"),
        (True, True, True, True, False),
    )
    boundary = raw[3:4].clone()
    boundary[0, 5] = torch.tensor([0.5, 0.0, 0.0])
    assert decode_boltzgen_atom14(boundary) == (("UNK",), (True,))
    boundary[0, 5, 0] = 0.5001
    assert decode_boltzgen_atom14(boundary) == (("SER",), (True,))
    boundary[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    boundary[0, 5, 0] = 0.5  # equidistant to N and CA: the first anchor wins
    assert decode_boltzgen_atom14(boundary) == (("UNK",), (True,))


def test_codesign_decodes_each_sample_and_preserves_target(peptide, design_library, design_config, monkeypatch):
    with torch.device("cpu"):
        model = BoltzGenModel(design_config).eval()
    pipeline = BoltzGenBackbonePipeline(model=model, processor=BoltzGenProcessor())
    actual_sample = model.sample
    cases = (("ALA", "SER"), ("invalid", "LYS"))
    anchors = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])

    def sample_with_markers(input, **kwargs):
        output = actual_sample(input, **kwargs)
        tokens = torch.where(input.design_mask[0])[0]
        for sample_index, identities in enumerate(cases):
            for position, (token, identity) in enumerate(zip(tokens, identities)):
                rows = torch.where(input.atom_to_token[0] == token)[0]
                raw = torch.full((14, 3), 20.0 + 10 * position)
                raw[:4] = anchors + 10 * position
                if identity == "ALA":
                    raw[5:] = raw[3]
                elif identity == "SER":
                    raw[6:] = raw[0]
                elif identity == "LYS":
                    raw[9:] = raw[0]
                else:
                    raw[4] = raw[1]
                output.coordinates[0, sample_index, rows] = raw
        return output

    monkeypatch.setattr(model, "sample", sample_with_markers)
    source = peptide("ALA", "GLY")
    result = pipeline(
        source,
        placement=PLACEMENT,
        library=design_library,
        num_candidates=2,
        diffusion_batch_size=2,
        num_sampling_steps=1,
        num_recycles=1,
        seed=3,
    )
    assert len(result.candidates) == 2
    first, second = result.candidates
    assert first.atom14_residue_types == ("ALA", "SER") and first.atom14_recognized == (True, True)
    assert second.atom14_residue_types == ("UNK", "LYS") and second.atom14_recognized == (False, True)
    for candidate in result.candidates:
        assert candidate.codesign_structure is candidate.structure
        target = candidate.structure
        for component in range(2):
            for name in ("N", "CA", "C", "O"):
                site = AtomSite("A", component, name)
                source_index = source.system.atom_index([site])[0]
                target_index = target.system.atom_index([site])[0]
                np.testing.assert_array_equal(target.coordinates[target_index], source.coordinates[source_index])
    for name, row in (("CB", 4), ("OG", 5)):
        site = AtomSite("chain_0", 1, name)
        index = first.structure.system.atom_index([site])[0]
        assert first.structure.coordinate_evidence[index] == CoordinateEvidence.GENERATED
        np.testing.assert_allclose(first.structure.coordinates[index], first.atom14_coordinates[1, row].numpy())
    unknown_cb = second.structure.system.atom_index([AtomSite("chain_0", 0, "CB")])[0]
    assert unknown_cb < 0 or not second.structure.coordinate_mask[unknown_cb]


def test_invalid_codesign_does_not_retain_existing_identity(peptide, design_library, design_config, monkeypatch):
    model = BoltzGenModel(design_config).eval()
    pipeline = BoltzGenBackbonePipeline(model=model, processor=BoltzGenProcessor())
    sample = model.sample

    def sample_invalid(input, **kwargs):
        output = sample(input, **kwargs)
        token = torch.where(input.design_mask[0])[0][0]
        rows = torch.where(input.atom_to_token[0] == token)[0]
        raw = torch.full((14, 3), 20.0)
        raw[:4] = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
        raw[4] = raw[1]  # a CA-near marker has no upstream identity signature
        output.coordinates[0, 0, rows] = raw
        return output

    monkeypatch.setattr(model, "sample", sample_invalid)
    candidate = pipeline(
        peptide("ALA", "GLY"),
        conditions=(BoltzGenDesignMask((ComponentRef("A", 0),)),),
        library=design_library,
        num_sampling_steps=1,
        num_recycles=1,
        seed=3,
    ).candidates[0]
    names = candidate.structure.system.text("component", candidate.structure.system.component_symbol)
    assert names.tolist() == ["UNK", "GLY"]
    assert candidate.atom14_residue_types == ("UNK",) and candidate.atom14_recognized == (False,)


@pytest.mark.parametrize(
    ("identity", "fixed_atom", "refuse"),
    (("ALA", "CB", False), ("SER", "CA", False), ("SER", "CB", True)),
)
def test_codesign_preserves_protected_atoms(
    peptide, design_library, design_config, monkeypatch, identity, fixed_atom, refuse
):
    model = BoltzGenModel(design_config).eval()
    processor = BoltzGenProcessor()
    pipeline = BoltzGenBackbonePipeline(model=model, processor=processor)
    sample, prepare = model.sample, processor.prepare_input

    def prepare_with_protected_atom(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        system = prepared.context.structure.system
        atom = system.atom_index([AtomSite("A", 0, fixed_atom)])[0]
        component = system.component_index([ComponentRef("A", 0)])[0]
        keep = np.zeros(len(system.atom_symbol), dtype=bool)
        keep[atom] = True
        context = prepared.context
        scope = RegenerationScope.components(system, np.array([component]), keep_atoms=keep)
        return PreparedInput(
            prepared.features, PreparationContext(context.structure, context.slots, scope, context.record, context.spec)
        )

    def sample_identity(input, **kwargs):
        output = sample(input, **kwargs)
        token = torch.where(input.design_mask[0])[0][0]
        rows = torch.where(input.atom_to_token[0] == token)[0]
        raw = torch.full((14, 3), 20.0)
        raw[:4] = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
        raw[5 if identity == "ALA" else 6 :] = raw[3 if identity == "ALA" else 0]
        output.coordinates[0, 0, rows] = raw
        return output

    monkeypatch.setattr(processor, "prepare_input", prepare_with_protected_atom)
    monkeypatch.setattr(model, "sample", sample_identity)
    source = peptide("ALA", "GLY")
    kwargs = {
        "conditions": (BoltzGenDesignMask((ComponentRef("A", 0),)),),
        "library": design_library,
        "num_sampling_steps": 1,
        "num_recycles": 1,
        "seed": 3,
    }
    if refuse:
        with pytest.raises(InvalidRequestError, match="scope_violation"):
            pipeline(source, **kwargs)
        return
    candidate = pipeline(source, **kwargs).candidates[0]
    original = source.system.atom_index([AtomSite("A", 0, fixed_atom)])[0]
    preserved = candidate.structure.system.atom_index([AtomSite("A", 0, fixed_atom)])[0]
    np.testing.assert_array_equal(candidate.structure.coordinates[preserved], source.coordinates[original])
    assert candidate.structure.coordinate_evidence[preserved] == source.coordinate_evidence[original]
    names = candidate.structure.system.text("component", candidate.structure.system.component_symbol)
    assert names[0] == identity
    for name in ("N", "CA", "C", "O"):
        source_atom = source.system.atom_index([AtomSite("A", 1, name)])[0]
        target_atom = candidate.structure.system.atom_index([AtomSite("A", 1, name)])[0]
        np.testing.assert_array_equal(candidate.structure.coordinates[target_atom], source.coordinates[source_atom])


def test_codesign_rejects_removing_a_declared_crosslink(peptide, design_library, design_config, monkeypatch):
    model = BoltzGenModel(design_config).eval()
    pipeline = BoltzGenBackbonePipeline(model=model, processor=BoltzGenProcessor())
    sample = model.sample

    def sample_alanine(input, **kwargs):
        output = sample(input, **kwargs)
        token = torch.where(input.design_mask[0])[0][0]
        rows = torch.where(input.atom_to_token[0] == token)[0]
        raw = torch.full((14, 3), 20.0)
        raw[:4] = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
        raw[5:] = raw[3]
        output.coordinates[0, 0, rows] = raw
        return output

    monkeypatch.setattr(model, "sample", sample_alanine)
    link = SymbolicBond(
        AtomSite("A", 0, "SG"),
        AtomSite("B", 0, "SG"),
        leaving_a=frozenset({"HG"}),
        leaving_b=frozenset({"HG"}),
    )
    conditions = (BoltzGenDesignMask((ComponentRef("A", 0),)), link)
    with pytest.raises(UnsupportedError, match="design_breaks_connection"):
        pipeline(
            peptide("CYS", copies=2),
            conditions=conditions,
            library=design_library,
            num_sampling_steps=1,
            num_recycles=1,
            seed=3,
        )


class TestBackboneDesign:
    @pytest.fixture
    def pipeline(self, design_config):
        torch.manual_seed(37)
        model = BoltzGenModel(design_config).double().eval()
        return BoltzGenBackbonePipeline(model=model, processor=BoltzGenProcessor(dtype=torch.float64))

    @pytest.fixture
    def candidate(self, pipeline, peptide, design_library, monkeypatch):
        # The small randomly initialized Model has no trained identity markers.
        # Give its generated rows the valid GLY signature while retaining sampled backbone coordinates.
        sample = pipeline.model.sample

        def sample_glycine(input, **kwargs):
            output = sample(input, **kwargs)
            for token in torch.where(input.design_mask[0])[0]:
                rows = torch.where(input.atom_to_token[0] == token)[0]
                output.coordinates[:, :, rows[4:]] = output.coordinates[:, :, rows[3:4]]
            return output

        monkeypatch.setattr(pipeline.model, "sample", sample_glycine)
        result = pipeline.design(
            peptide("ALA", "GLY"),
            placement=PLACEMENT,
            library=design_library,
            num_candidates=1,
            num_sampling_steps=1,
            num_recycles=1,
            seed=3,
        )
        return result.candidates[0]

    def test_designed_positions(self, candidate):
        assert candidate.designed_positions == (ComponentRef("chain_0", 0), ComponentRef("chain_0", 1))
        assert candidate.connected_designed_positions == candidate.designed_positions

    def test_target_coordinates_kept(self, pipeline, peptide, design_library, candidate):
        planned = pipeline.prepare(peptide("ALA", "GLY"), placement=PLACEMENT, library=design_library)
        target = np.arange(planned.structure.system.component_atom_ptr[2])
        np.testing.assert_array_equal(candidate.structure.coordinates[target], planned.structure.coordinates[target])
        assert (candidate.structure.coordinate_evidence[target] == CoordinateEvidence.OBSERVED).all()

    def test_invalid_candidate_count(self, pipeline, peptide, design_library):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            pipeline.design(peptide("ALA"), placement=PLACEMENT, library=design_library, num_candidates=0)

    def test_generated_coordinates_written(self, candidate):
        system = candidate.structure.system
        generated = np.arange(system.component_atom_ptr[2], len(system.atom_symbol))
        written = candidate.structure.coordinate_mask[generated]
        assert written.any() and not written.all()  # the modelled backbone, not the unaddressed side chain
        evidence = candidate.structure.coordinate_evidence[generated][written]
        assert (evidence == CoordinateEvidence.GENERATED).all()


class TestInverseFolding:
    @pytest.fixture
    def pipeline(self, inverse_folding_config):
        torch.manual_seed(11)
        model = BoltzGenInverseFoldingModel(inverse_folding_config).double().eval()
        processor = BoltzGenInverseFoldingProcessor(dtype=torch.float64)
        return BoltzGenInverseFoldingPipeline(model=model, processor=processor)

    def test_sampled_sequence(self, pipeline, peptide, library):
        result = pipeline.design(
            peptide("ALA", "GLY", "LYS"),
            global_omit_tokens=tuple("CDEFGHIKLMNPQRSTVWY"),
            temperature=None,
            library=library,
            seed=5,
        )
        candidate = result.candidates[0]
        system = candidate.structure.system
        assert system.text("component", system.component_symbol).tolist() == ["ALA", "ALA", "ALA"]
        assert candidate.designed_positions == tuple(ComponentRef("A", index) for index in range(3))

    def test_selected_positions(self, pipeline, peptide, library):
        result = pipeline.design(
            peptide("ALA", "GLY", "LYS"),
            design=PositionSelection(components=(ComponentRef("A", 1),)),
            global_omit_tokens=tuple("CDEFGHIKLMNPQRSTVWY"),
            temperature=None,
            library=library,
            seed=5,
        )
        system = result.candidates[0].structure.system
        assert system.text("component", system.component_symbol).tolist() == ["ALA", "ALA", "LYS"]


@pytest.fixture
def ranked_screening_candidates(peptide, monkeypatch):
    from ofoldx.results import FoldPrediction, Score, ScoreResult

    structure = peptide("ALA", "GLY")
    scores = {
        "design_to_target_iptm": Score(0.5),
        "design_ptm": Score(0.5),
        "min_design_to_target_pae": Score(2.0),
    }
    candidates = tuple(
        BoltzGenDesignCandidate(
            structure=structure,
            source_checkpoint=source,
            designed_positions=(ComponentRef("A", 0),),
            connected_designed_positions=(ComponentRef("A", 0),),
            whole_fold_predictions=(FoldPrediction(structure, scores=dict(scores)),),
            whole_fold_selected_index=0,
            affinity_result=ScoreResult(structure, scores={"affinity_probability_binary1": Score(0.5)}),
        )
        for source in ("diverse", "adherence")
    )
    pipeline = BoltzGenPipeline(
        diverse_generator=object(),
        adherence_generator=object(),
        inverse_folder=object(),
        soluble_designer=object(),
        folder=object(),
        affinity_scorer=object(),
        esmfold2=object(),
    )

    def analysis(candidate, *, from_inverse_folding, **kwargs):
        first = candidate.source_checkpoint == "diverse"
        original = None if from_inverse_folding else (10 if first else 1)
        refolded = 1 if first else 10

        def atomic(value):
            return SimpleNamespace(
                contacts=SimpleNamespace(contacts=SimpleNamespace(num_hydrogen_bonds=value, num_charged_pairs=value)),
                sasa=SimpleNamespace(sasa=SimpleNamespace(delta=value, target_unbound=20.0, target_bound=10.0)),
                contacts_unavailable=None,
                sasa_unavailable=None,
                strict_contacts=None,
                strict_sasa=None,
                strict_contacts_unavailable="missing side chain",
                strict_sasa_unavailable="missing OXT",
            )

        return SimpleNamespace(
            designed_sequence="A" if first else "G",
            num_design=1,
            bb_rmsd=1.0,
            bb_design_rmsd=1.0,
            bb_target_aligned_design_rmsd=1.0,
            design_folding_bb_rmsd=1.0,
            amino_acid_fractions={},
            secondary_structure_fractions={},
            liabilities={},
            design_hydrophobicity=0.0,
            design_chain_hydrophobicity=0.0,
            original_atomic=None if original is None else atomic(original),
            refolded_atomic=atomic(refolded),
            binding_site_under_3angstrom=None,
            binding_site_under_4angstrom=None,
            binding_site_under_5angstrom=None,
            binding_site_under_6angstrom=None,
            binding_site_under_7angstrom=None,
            binding_site_under_8angstrom=None,
            binding_site_under_9angstrom=None,
            hydrophobic_patch=None,
            hydrophobic_patch_unavailable=None,
            strict_largest_hydrophobic_patch=None,
            strict_largest_hydrophobic_patch_unavailable=None,
        )

    monkeypatch.setattr(pipeline, "_analyze_candidate", analysis)
    return pipeline, candidates


@pytest.mark.parametrize(("inverse_model", "first"), [(None, 0), ("boltzif", 1)])
def test_public_selection_uses_original_without_if_and_refolded_with_if(
    ranked_screening_candidates, inverse_model, first
):
    pipeline, candidates = ranked_screening_candidates
    prepared = SimpleNamespace(
        protocol="protein-small_molecule",
        inverse_model=inverse_model,
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=None),
    )
    phase = "original" if inverse_model is None else "refolded"
    weights = {f"num_{phase}_hydrogen_bonds": 1.0, f"num_{phase}_charged_pairs": 1.0, f"target_delta_sasa_{phase}": 1.0}
    result = pipeline.postprocess(prepared, candidates, filter_biased=False, metric_weights=weights)
    assert result.ranked_indices[0] == first
    assert (result.candidates[0].analysis.original_atomic is None) == (inverse_model is not None)


def test_public_selection_refuses_unavailable_strict_metric(ranked_screening_candidates):
    pipeline, candidates = ranked_screening_candidates
    prepared = SimpleNamespace(
        protocol="protein-small_molecule",
        inverse_model="boltzif",
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=None),
    )
    with pytest.raises(InvalidRequestError, match="strict_target_delta_sasa_refolded") as caught:
        pipeline.postprocess(
            prepared, candidates, filter_biased=False, metric_weights={"strict_target_delta_sasa_refolded": 1.0}
        )
    assert caught.value.detail["reason"] == "metric_unavailable"
    assert "missing OXT" in str(caught.value.detail["details"])


def test_redesign_without_target_does_not_fabricate_sasa(peptide, design_library):
    from ofoldx.results import FoldPrediction, Score

    structure = peptide("ALA", "GLY")
    positions = (ComponentRef("A", 0), ComponentRef("A", 1))
    folded = FoldPrediction(
        structure,
        scores={
            "esmfold2_score": Score(0.8),
            "ptm": Score(0.8),
            "iptm": Score(0.0),
            "design_residue_iptm": Score(0.0),
            "design_to_target_iptm": Score(0.0),
        },
    )
    candidate = BoltzGenDesignCandidate(
        structure=structure,
        source_checkpoint="diverse",
        designed_positions=positions,
        connected_designed_positions=positions,
        whole_fold_predictions=(folded,),
        whole_fold_selected_index=0,
        esmfold2_predictions=(folded,),
        esmfold2_selected_index=0,
        esmfold2_score_metric="ptm",
    )
    pipeline = BoltzGenPipeline(
        diverse_generator=object(),
        adherence_generator=object(),
        inverse_folder=object(),
        soluble_designer=object(),
        folder=object(),
        affinity_scorer=object(),
        esmfold2=object(),
    )
    prepared = SimpleNamespace(
        protocol="protein-redesign",
        inverse_model=None,
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=design_library),
    )
    result = pipeline.postprocess(prepared, (candidate,), filter_biased=False, filter_cysteine=False)
    analysis = result.candidates[0].analysis
    assert analysis.original_atomic.sasa is None and analysis.refolded_atomic.sasa is None
    assert (
        analysis.original_atomic.sasa_unavailable == analysis.refolded_atomic.sasa_unavailable == "no target components"
    )


def test_public_selection_refuses_unavailable_filter(ranked_screening_candidates):
    from ofoldx.analysis import CandidateFilter

    pipeline, candidates = ranked_screening_candidates
    prepared = SimpleNamespace(
        protocol="protein-small_molecule",
        inverse_model="boltzif",
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=None),
    )
    with pytest.raises(InvalidRequestError, match="strict_num_refolded_hydrogen_bonds") as caught:
        pipeline.postprocess(
            prepared,
            candidates,
            filter_biased=False,
            additional_filters=(CandidateFilter("strict_num_refolded_hydrogen_bonds", 1.0, False),),
        )
    assert caught.value.detail["reason"] == "metric_unavailable"
    assert "missing side chain" in str(caught.value.detail["details"])


def test_redesign_uses_boltz2_scores_and_tie_breaker(ranked_screening_candidates):
    from ofoldx.results import Score

    pipeline, candidates = ranked_screening_candidates
    prepared = SimpleNamespace(
        protocol="protein-redesign",
        inverse_model=None,
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=None),
    )
    for candidate, design_iptm, tie in zip(candidates, (0.9, 0.1), (0.1, 0.9), strict=True):
        candidate.whole_refold.scores.update(
            {
                "design_residue_iptm": Score(design_iptm),
                "iptm": Score(0.5),
                "ptm": Score(0.5),
                "design_to_target_iptm": Score(tie),
            }
        )
    result = pipeline.postprocess(prepared, candidates, filter_biased=False, filter_cysteine=False)
    assert result.ranked_indices[0] == 0
    candidates[1].whole_refold.scores["design_residue_iptm"] = Score(0.9)
    result = pipeline.postprocess(prepared, candidates, filter_biased=False, filter_cysteine=False)
    assert result.ranked_indices[0] == 1
    result = pipeline.postprocess(
        prepared,
        candidates,
        filter_biased=False,
        filter_cysteine=False,
        metric_weights={"design_residue_iptm": None, "design_to_target_iptm": 1.0},
    )
    assert result.ranked_indices[0] == 1


def test_redesign_requires_ranking_scores(ranked_screening_candidates):
    from ofoldx.results import Score

    pipeline, candidates = ranked_screening_candidates
    prepared = SimpleNamespace(
        protocol="protein-redesign",
        inverse_model=None,
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=None),
    )
    for candidate in candidates:
        candidate.whole_refold.scores.update({"design_residue_iptm": Score(0.5), "ptm": Score(0.5)})
    with pytest.raises(InvalidRequestError, match="iptm") as caught:
        pipeline.postprocess(prepared, candidates, filter_biased=False, filter_cysteine=False)
    assert caught.value.detail["reason"] == "metric_unavailable"


def test_single_chain_redesign_empty_interface_follows_upstream_ranking(ranked_screening_candidates):
    from ofoldx.results import Score

    pipeline, candidates = ranked_screening_candidates
    prepared = SimpleNamespace(
        protocol="protein-redesign",
        inverse_model=None,
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=None),
    )
    for candidate, ptm in zip(candidates, (0.1, 0.9), strict=True):
        candidate.whole_refold.scores.update(
            {
                "design_residue_iptm": Score(0.0),
                "iptm": Score(0.0),
                "ptm": Score(ptm),
                "design_to_target_iptm": Score(0.0),
            }
        )
    result = pipeline.postprocess(prepared, candidates, filter_biased=False, filter_cysteine=False)
    assert result.ranked_indices == (0, 1)


def test_redesign_same_chain_has_no_interchain_contacts(peptide, design_library):
    from ofoldx.results import FoldPrediction, Score

    structure = peptide("ALA", "GLY")
    design = (ComponentRef("A", 0),)
    connected = (ComponentRef("A", 0), ComponentRef("A", 1))
    folded = FoldPrediction(
        structure,
        scores={
            "esmfold2_score": Score(0.8),
            "ptm": Score(0.8),
            "iptm": Score(0.0),
            "design_residue_iptm": Score(0.0),
            "design_to_target_iptm": Score(0.0),
        },
    )
    candidate = BoltzGenDesignCandidate(
        structure=structure,
        source_checkpoint="diverse",
        designed_positions=design,
        connected_designed_positions=connected,
        whole_fold_predictions=(folded,),
        whole_fold_selected_index=0,
        esmfold2_predictions=(folded,),
        esmfold2_selected_index=0,
        esmfold2_score_metric="ptm",
    )
    pipeline = BoltzGenPipeline(
        diverse_generator=object(),
        adherence_generator=object(),
        inverse_folder=object(),
        soluble_designer=object(),
        folder=object(),
        affinity_scorer=object(),
        esmfold2=object(),
    )
    prepared = SimpleNamespace(
        protocol="protein-redesign",
        inverse_model=None,
        budget=1,
        top_budget=1,
        diverse_input=SimpleNamespace(library=design_library),
    )
    result = pipeline.postprocess(prepared, (candidate,), filter_biased=False, filter_cysteine=False)
    atomic = result.candidates[0].analysis.refolded_atomic
    assert atomic.contacts is None and atomic.strict_contacts is None
    assert atomic.contacts_unavailable and atomic.strict_contacts_unavailable
    assert atomic.sasa is not None
