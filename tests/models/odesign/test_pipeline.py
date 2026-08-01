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

"""Behavior of ofoldx.models.odesign.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data.molecular import AtomSite, ComponentRef, MonomerType
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.design import PositionSelection
from ofoldx.models.grnade import GrnadeConfig, GrnadeModel, GrnadePipeline, GrnadeProcessor
from ofoldx.models.grnade.model import GrnadeSamplingOutput
from ofoldx.models.odesign import (
    ODesignBackboneComponent,
    ODesignConfig,
    ODesignDnaPipeline,
    ODesignLigandCoordinateComponent,
    ODesignLigandProcessor,
    ODesignModel,
    ODesignProcessor,
    ODesignProcessorConfig,
    ODesignRnaPipeline,
    designed_chains,
)
from ofoldx.models.odesign.model import ODesignSamplingOutput
from ofoldx.models.odesign.pipeline import has_ligand_atom_context
from ofoldx.models.oinvfold import (
    OInvFoldPolymerConfig,
    OInvFoldPolymerModel,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerProcessor,
    OInvFoldPolymerProcessorConfig,
)

FIRST = PositionSelection(components=(ComponentRef("A", 0),))


def small_model() -> ODesignModel:
    with torch.device("meta"):
        return ODesignModel(
            ODesignConfig(
                trunk={"num_pairformer_blocks": 1, "num_msa_stages": 1},
                denoiser={"num_token_encoder_blocks": 1, "num_atom_encoder_blocks": 1, "num_atom_decoder_blocks": 1},
            )
        )


def sampling_output(prepared, samples: np.ndarray) -> ODesignSamplingOutput:
    tokens = len(prepared.features.token_mask[0])
    return ODesignSamplingOutput(
        coordinates=torch.from_numpy(samples)[None, None],
        distogram_logits=torch.zeros(1, 1, tokens, tokens, 4),
        token_bond_logits=torch.zeros(1, 1, tokens, tokens, 2),
        batch_size=[1],
    )


class TestDesignedChains:
    def test_grouped_by_instance(self, peptide):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        (symbol, positions), *rest = designed_chains(structure, PositionSelection(), MonomerType.PROTEIN)
        assert (symbol, positions.tolist()) == ("A", [0, 1])
        assert not rest

    def test_other_monomer_refused(self, peptide):
        with pytest.raises(UnsupportedError, match="model_capacity"):
            designed_chains(peptide("ALA", "GLY"), PositionSelection(), MonomerType.RNA)


class TestLigandContext:
    @pytest.mark.parametrize("ligands,expected", [((), False), (("ACT",), True)])
    def test_observed_heavy_atoms(self, peptide, ligands, expected):
        assert has_ligand_atom_context(peptide("ALA", "GLY", ligands=ligands)) is expected


class TestBackboneDecode:
    def test_scope_only(self, peptide):
        structure = peptide("ALA", "GLY")
        fixed = AtomSite("A", 1, "CA")
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
        prepared = processor.prepare_input(structure, design=FIRST, fixed_atoms=(fixed,))
        component = ODesignBackboneComponent(model=small_model(), processor=processor)
        samples = np.zeros((1, len(prepared.context.slots.atom), 3))
        (sample,) = component.postprocess(prepared, sampling_output(prepared, samples))
        assert not sample.unbound_linkers
        decoded = sample.structure
        system = structure.system
        designed = np.arange(int(system.component_atom_ptr[0]), int(system.component_atom_ptr[1]))
        context = structure.system.atom_index([fixed])[0]
        placed = decoded.coordinate_mask[designed]
        assert (decoded.coordinate_evidence[designed][placed] == CoordinateEvidence.GENERATED).all()
        assert not placed.all()  # the generated residue's unaddressed atoms are withdrawn
        np.testing.assert_array_equal(decoded.coordinates[context], structure.coordinates[context])


@pytest.mark.parametrize(("instances", "designed"), [(("A",), ("A",)), (("A", "B"), ("B",)), (("A", "B"), ("A", "B"))])
def test_rna_design_selects_each_requested_chain(library, placed, monkeypatch, instances, designed):
    system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("A", "U"))}, dict.fromkeys(instances, "R")), library
    )
    structure = placed(system)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)

    def sample_backbone(input, **_options):
        samples = input.observed_positions[0].detach().cpu().numpy()[None]
        return sampling_output(type("Prepared", (), {"features": input})(), samples)

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    model = GrnadeModel(
        GrnadeConfig(
            node_hidden_scalar_size=8,
            node_hidden_vector_size=2,
            edge_hidden_scalar_size=8,
            edge_hidden_vector_size=2,
            num_blocks=1,
            dropout=0.0,
        )
    )

    def sample_sequence(input, *, num_samples, **_options):
        rows = input.sequence_tokens.shape[0]
        sequence = torch.full((num_samples, rows), TOKEN_INDEX["RNA_A"], dtype=torch.long)
        logits = torch.zeros(num_samples, rows, len(TOKENS))
        return GrnadeSamplingOutput(sequence_tokens=sequence, logits=logits, batch_size=[])

    monkeypatch.setattr(model, "sample", sample_sequence)
    pipeline = ODesignRnaPipeline(
        backbone=backbone, inverse_folder=GrnadePipeline(model=model, processor=GrnadeProcessor())
    )
    result = pipeline(structure, design=PositionSelection(instances=designed), seed=7, library=library)
    assert len(result.candidates) == 1
    assert len(result.candidates[0].records) == len(designed)


class TestLigandCoordinates:
    def test_generated_slots_decoded(self, peptide):
        processor = ODesignLigandProcessor()
        prepared = processor.prepare_input(peptide("ALA", "GLY"), generated_atoms=3)
        component = ODesignLigandCoordinateComponent(model=small_model(), processor=processor)
        samples = np.arange(2 * len(prepared.context.slots.atom) * 3, dtype=np.float64).reshape(2, -1, 3)
        decoded, logits = component.postprocess(prepared, sampling_output(prepared, samples))
        assert len(decoded) == 2
        assert decoded[0].slot_ids == ("0", "1", "2")
        np.testing.assert_array_equal(decoded[1].coordinates, samples[1, -3:])
        assert logits.shape == (3, 3, 2)
        assert decoded[0].context is prepared.context.structure


class TestRouteGuards:
    def test_modality_must_match(self):
        with torch.device("meta"):
            inverse_folder = OInvFoldPolymerPipeline(
                model=OInvFoldPolymerModel(OInvFoldPolymerConfig(num_encoder_blocks=1)),
                processor=OInvFoldPolymerProcessor(OInvFoldPolymerProcessorConfig(modality="protein")),
            )
        backbone = ODesignBackboneComponent(
            model=small_model(), processor=ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
        )
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            ODesignDnaPipeline(backbone=backbone, inverse_folder=inverse_folder)


@pytest.mark.parametrize("kind", ["rna", "dna"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("instances,designed", [(("A",), ("A",)), (("A", "B"), ("A", "B"))])
def test_generated_nucleic_linkers_follow_sample_and_real_identity(
    library, monkeypatch, kind, instances, designed, unknown, dtype
):
    from ofoldx.data.structure import Structure
    from ofoldx.models.oinvfold.model import OInvFoldOutput

    polymer = "polyribonucleotide" if kind == "rna" else "polydeoxyribonucleotide"
    sequence = ("A", "C") if kind == "rna" else ("DA", "DC")
    if unknown:
        sequence = ("N", "N") if kind == "rna" else ("DN", "DN")
    system, _, _ = build_system(
        Recipe({"R": PolymerSequence(polymer, sequence)}, dict.fromkeys(instances, "R")), library
    )
    source = Structure.unobserved(system)
    source_names = system.text("atom", system.atom_symbol).copy()
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"), dtype=dtype)
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)
    prepared = processor.prepare_input(source, design=PositionSelection(instances=designed))
    slots = prepared.context.slots
    positions = []
    if unknown:
        for row in np.flatnonzero(slots.atom < 0):
            positions.append((int(slots.layout.token_component[slots.token[row]]), int(row)))
    else:
        for index, instance in enumerate(instances):
            for offset, identity in enumerate(sequence):
                atom = system.atom_index(
                    [AtomSite(instance, offset, "N9" if identity in ("A", "G", "DA", "DG") else "N1")]
                )[0]
                row = np.flatnonzero(slots.atom == atom)
                assert len(row) == 1
                positions.append((index * len(sequence) + offset, int(row[0])))
    assert len(positions) == 2 * len(instances)
    samples = []

    def sample_backbone(input, *, num_samples, **_options):
        ref = input.reference_positions[0].detach().float().cpu().numpy()
        token = input.atom_to_token[0].detach().cpu().numpy()
        values = np.stack(
            [ref * (1 + sample * 0.1) + token[:, None] * np.array([5.0, 0.0, 0.0]) for sample in range(num_samples)]
        )
        output = sampling_output(type("Prepared", (), {"features": input})(), values)
        output.coordinates = output.coordinates.to(dtype=dtype)
        samples.extend(output.coordinates[0, 0].to(torch.float64).cpu().numpy())
        return output

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    if kind == "rna":
        with torch.device("meta"):
            model = GrnadeModel(
                GrnadeConfig(
                    node_hidden_scalar_size=8,
                    node_hidden_vector_size=2,
                    edge_hidden_scalar_size=8,
                    edge_hidden_vector_size=2,
                    num_blocks=1,
                    dropout=0.0,
                )
            )

        def sample_sequence(input, *, num_samples, **_options):
            assert input.edge_index.shape[-1] > 0
            rows = len(input.sequence_tokens)
            tokens = torch.tensor([TOKEN_INDEX["RNA_C"] if i % 2 == 0 else TOKEN_INDEX["RNA_A"] for i in range(rows)])
            return GrnadeSamplingOutput(
                sequence_tokens=tokens.repeat(num_samples, 1),
                logits=torch.zeros(num_samples, rows, len(TOKENS)),
                batch_size=[],
            )

        monkeypatch.setattr(model, "sample", sample_sequence)
        pipeline = ODesignRnaPipeline(
            backbone=backbone, inverse_folder=GrnadePipeline(model=model, processor=GrnadeProcessor())
        )
    else:
        with torch.device("meta"):
            model = OInvFoldPolymerModel(OInvFoldPolymerConfig(num_encoder_blocks=1, modality="dna"))
        child = OInvFoldPolymerPipeline(
            model=model, processor=OInvFoldPolymerProcessor(OInvFoldPolymerProcessorConfig(modality="dna"))
        )

        def forward(input):
            logits = torch.full((len(input.component_index), len(TOKENS)), -20.0)
            logits[:, TOKEN_INDEX["DNA_A"]] = 10.0
            logits[:, TOKEN_INDEX["DNA_C"]] = 9.0
            return OInvFoldOutput(logits=logits, batch_size=[])

        monkeypatch.setattr(child, "forward", forward)
        pipeline = ODesignDnaPipeline(backbone=backbone, inverse_folder=child)
    result = pipeline(
        source,
        design=PositionSelection(instances=designed),
        num_backbones=2,
        num_sequences=2,
        num_sampling_steps=2,
        num_recycles=1,
        seed=7,
        library=library,
    )
    assert len(result.candidates) == 4
    for candidate_index, candidate in enumerate(result.candidates):
        target = candidate.structure
        identities = target.system.text("component", target.system.component_symbol)
        for component, row in positions:
            ref = target.system.component_refs(np.array([component]))[0]
            name = "N9" if identities[component] in ("A", "G", "DA", "DG") else "N1"
            atom = int(target.system.atom_index([AtomSite(ref.instance, ref.component, name)])[0])
            assert atom >= 0
            assert target.coordinate_evidence[atom] == CoordinateEvidence.GENERATED
            np.testing.assert_array_equal(target.coordinates[atom], samples[candidate_index // 2][row])
    np.testing.assert_array_equal(system.text("atom", system.atom_symbol), source_names)


def test_fixed_nucleic_linker_is_not_replaced_by_generated_coordinates(library, placed, monkeypatch):
    system, _, _ = build_system(Recipe({"R": PolymerSequence("polyribonucleotide", ("U", "C"))}, {"A": "R"}), library)
    source = placed(system)
    fixed = AtomSite("A", 0, "N1")
    atom = int(system.atom_index([fixed])[0])
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)

    def sample_backbone(input, **_options):
        values = input.observed_positions[0].detach().cpu().numpy()[None] + 100.0
        return sampling_output(type("Prepared", (), {"features": input})(), values)

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    with torch.device("meta"):
        model = GrnadeModel(
            GrnadeConfig(
                node_hidden_scalar_size=8,
                node_hidden_vector_size=2,
                edge_hidden_scalar_size=8,
                edge_hidden_vector_size=2,
                num_blocks=1,
                dropout=0.0,
            )
        )

    def sample_sequence(input, *, num_samples, **_options):
        rows = len(input.sequence_tokens)
        return GrnadeSamplingOutput(
            sequence_tokens=torch.full((num_samples, rows), TOKEN_INDEX["RNA_U"], dtype=torch.long),
            logits=torch.zeros(num_samples, rows, len(TOKENS)),
            batch_size=[],
        )

    monkeypatch.setattr(model, "sample", sample_sequence)
    pipeline = ODesignRnaPipeline(
        backbone=backbone, inverse_folder=GrnadePipeline(model=model, processor=GrnadeProcessor())
    )
    target = pipeline(source, design=FIRST, fixed_atoms=(fixed,), seed=7, library=library).candidates[0].structure
    target_atom = int(target.system.atom_index([fixed])[0])
    np.testing.assert_array_equal(target.coordinates[target_atom], source.coordinates[atom])
    assert target.coordinate_evidence[target_atom] == source.coordinate_evidence[atom]


@pytest.mark.parametrize(("polymer", "unknown"), [("polyribonucleotide", "N"), ("polydeoxyribonucleotide", "DN")])
def test_backbone_sample_exposes_unbound_linkers_without_changing_chemistry(library, polymer, unknown):
    from ofoldx.data.structure import Structure

    system, _, _ = build_system(Recipe({"R": PolymerSequence(polymer, (unknown, unknown))}, {"A": "R"}), library)
    source = Structure.unobserved(system)
    original_atoms = system.text("atom", system.atom_symbol).copy()
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    prepared = processor.prepare_input(source)
    component = ODesignBackboneComponent(model=small_model(), processor=processor)
    slots = prepared.context.slots
    samples = np.arange(2 * len(slots.atom) * 3, dtype=np.float64).reshape(2, len(slots.atom), 3)
    outputs = component.postprocess(prepared, sampling_output(prepared, samples))
    assert len(outputs) == 2
    rows = np.flatnonzero(slots.atom < 0)
    assert len(rows) == 2
    refs = system.component_refs(slots.layout.token_component[slots.token[rows]])
    for sample_index, output in enumerate(outputs):
        assert output.structure.system is system
        np.testing.assert_array_equal(
            output.structure.system.text("atom", output.structure.system.atom_symbol), original_atoms
        )
        assert tuple(output.unbound_linkers) == refs
        for row, ref in zip(rows, refs, strict=True):
            linker = output.unbound_linkers[ref]
            np.testing.assert_array_equal(linker.coordinates, samples[sample_index, row])
            assert linker.evidence == CoordinateEvidence.GENERATED
    samples[0, rows[0], 0] = np.nan
    with pytest.raises(InvalidRequestError, match="value_invalid"):
        component.postprocess(prepared, sampling_output(prepared, samples))


def test_unknown_linker_rejects_unmappable_sequence_identity(library, monkeypatch):
    from ofoldx.data.structure import Structure

    system, _, _ = build_system(Recipe({"R": PolymerSequence("polyribonucleotide", ("N", "N"))}, {"A": "R"}), library)
    source = Structure.unobserved(system)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)

    def sample_backbone(input, **_options):
        values = input.reference_positions[0].detach().cpu().numpy().copy()
        values += input.atom_to_token[0].detach().cpu().numpy()[:, None] * np.array([5.0, 0.0, 0.0])
        return sampling_output(type("Prepared", (), {"features": input})(), values[None])

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    with torch.device("meta"):
        model = GrnadeModel(
            GrnadeConfig(
                node_hidden_scalar_size=8,
                node_hidden_vector_size=2,
                edge_hidden_scalar_size=8,
                edge_hidden_vector_size=2,
                num_blocks=1,
                dropout=0.0,
            )
        )

    def sample_invalid(input, *, num_samples, **_options):
        rows = len(input.sequence_tokens)
        return GrnadeSamplingOutput(
            sequence_tokens=torch.full((num_samples, rows), TOKEN_INDEX["RNA_X"], dtype=torch.long),
            logits=torch.zeros(num_samples, rows, len(TOKENS)),
            batch_size=[],
        )

    monkeypatch.setattr(model, "sample", sample_invalid)
    pipeline = ODesignRnaPipeline(
        backbone=backbone, inverse_folder=GrnadePipeline(model=model, processor=GrnadeProcessor())
    )
    with pytest.raises(InvalidRequestError, match="design_token_unmappable"):
        pipeline(source, seed=7, library=library)
    assert system.text("component", system.component_symbol).tolist() == ["N", "N"]


@pytest.mark.parametrize("identity", ["U", "C", "A"])
@pytest.mark.parametrize("kind", ["rna", "dna"])
@pytest.mark.parametrize("fixed_atom", ["N1", "C1'"])
@pytest.mark.parametrize(
    ("instances", "selection", "fixed_instances"),
    [(("A",), None, ("A",)), (("A", "B"), None, ("A", "B")), (("A", "B"), ("B",), ("A",))],
)
def test_fixed_nucleic_atom_is_not_lost_after_sequence_design(
    library, placed, monkeypatch, identity, kind, fixed_atom, instances, selection, fixed_instances
):
    from ofoldx.data.structure import Structure
    from ofoldx.provenance import Basis, Resolution, state

    polymer = "polyribonucleotide" if kind == "rna" else "polydeoxyribonucleotide"
    sequence = ("U", "C") if kind == "rna" else ("DT", "DC")
    system, _, _ = build_system(
        Recipe({"R": PolymerSequence(polymer, sequence)}, dict.fromkeys(instances, "R")), library
    )
    source = placed(system)
    fixed = tuple(AtomSite(instance, 0, fixed_atom) for instance in fixed_instances)
    design = PositionSelection() if selection is None else PositionSelection(instances=selection)
    original_atoms = system.atom_index(fixed)
    arrays = source.build()
    arrays.columns["coordinate_quantum"][original_atoms] = 0.001
    for name, value in (("occupancy", 0.73), ("occupancy_uncertainty", 0.02), ("b_factor", 29.0)):
        arrays.columns[name][original_atoms] = value
        arrays.columns[name + "_state"][original_atoms] = state(Resolution.KNOWN, Basis.DECLARED)
    source = Structure.from_arrays(system, arrays)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)

    def sample_backbone(input, **_options):
        values = input.observed_positions[0].detach().cpu().numpy()[None] + 100.0
        return sampling_output(type("Prepared", (), {"features": input})(), values)

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    if kind == "rna":
        with torch.device("meta"):
            model = GrnadeModel(
                GrnadeConfig(
                    node_hidden_scalar_size=8,
                    node_hidden_vector_size=2,
                    edge_hidden_scalar_size=8,
                    edge_hidden_vector_size=2,
                    num_blocks=1,
                    dropout=0.0,
                )
            )

        def sample_sequence(input, *, num_samples, **_options):
            rows = len(input.sequence_tokens)
            return GrnadeSamplingOutput(
                sequence_tokens=torch.full((num_samples, rows), TOKEN_INDEX["RNA_" + identity], dtype=torch.long),
                logits=torch.zeros(num_samples, rows, len(TOKENS)),
                batch_size=[],
            )

        monkeypatch.setattr(model, "sample", sample_sequence)
        pipeline = ODesignRnaPipeline(
            backbone=backbone, inverse_folder=GrnadePipeline(model=model, processor=GrnadeProcessor())
        )
    else:
        from ofoldx.models.oinvfold.model import OInvFoldOutput

        with torch.device("meta"):
            model = OInvFoldPolymerModel(OInvFoldPolymerConfig(num_encoder_blocks=1, modality="dna"))
        child = OInvFoldPolymerPipeline(
            model=model, processor=OInvFoldPolymerProcessor(OInvFoldPolymerProcessorConfig(modality="dna"))
        )

        def score_sequence(input):
            logits = torch.full((len(input.component_index), len(TOKENS)), -torch.inf)
            logits[:, TOKEN_INDEX["DNA_" + ("T" if identity == "U" else identity)]] = 0.0
            return OInvFoldOutput(logits=logits, batch_size=[])

        monkeypatch.setattr(child, "forward", score_sequence)
        pipeline = ODesignDnaPipeline(backbone=backbone, inverse_folder=child)
    if identity != "U" and fixed_atom == "N1" and selection is None:
        with pytest.raises(UnsupportedError, match="model_capacity"):
            pipeline(source, design=design, fixed_atoms=fixed, seed=7, library=library)
        return
    target = pipeline(source, design=design, fixed_atoms=fixed, seed=7, library=library).candidates[0].structure
    new_atoms = target.system.atom_index(fixed)
    for field in (
        "coordinates",
        "coordinate_evidence",
        "coordinate_quantum",
        "occupancy",
        "occupancy_state",
        "occupancy_uncertainty",
        "occupancy_uncertainty_state",
        "b_factor",
        "b_factor_state",
    ):
        np.testing.assert_array_equal(getattr(target, field)[new_atoms], getattr(source, field)[original_atoms])


def test_protein_fixed_sidechain_is_rejected_before_backbone_generation(peptide):
    source = peptide("ALA", "GLY")
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
    with pytest.raises(UnsupportedError, match="model_capacity"):
        processor.prepare_input(source, design=FIRST, fixed_atoms=(AtomSite("A", 0, "CB"),))


def test_fixed_backbone_does_not_allow_breaking_disulfide(library, placed, monkeypatch):
    from ofoldx.chemistry.construction import RecipeBond
    from ofoldx.data.molecular import BondKind
    from ofoldx.models.design import produce_design
    from ofoldx.models.ligandmpnn import LigandMpnnConfig, LigandMpnnModel, LigandMpnnPipeline, LigandMpnnProcessor
    from ofoldx.models.odesign import ODesignProteinPipeline
    from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnModel, ProteinMpnnPipeline, ProteinMpnnProcessor

    bond = RecipeBond(AtomSite("A", 0, "SG"), AtomSite("B", 0, "SG"), 1, BondKind.COVALENT, (("HG",), ("HG",)))
    recipe = Recipe({"P": PolymerSequence("polypeptide(L)", ("CYS", "CYS"))}, {"A": "P", "B": "P"}, (bond,))
    system, _, _ = build_system(recipe, library)
    source = placed(system)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)

    def sample_backbone(input, **_options):
        values = input.observed_positions[0].detach().cpu().numpy()[None]
        return sampling_output(type("Prepared", (), {"features": input})(), values)

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    with torch.device("meta"):
        protein = ProteinMpnnPipeline(model=ProteinMpnnModel(ProteinMpnnConfig()), processor=ProteinMpnnProcessor())
        ligand = LigandMpnnPipeline(model=LigandMpnnModel(LigandMpnnConfig()), processor=LigandMpnnProcessor())

    def alanine(target, *, design, **_options):
        prepared = protein.processor.prepare_input(target, design=design)
        return produce_design(
            prepared,
            np.array([TOKEN_INDEX["A"]]),
            positions=np.array([0]),
            table=protein.processor.layout,
            library=library,
        )

    monkeypatch.setattr(protein, "design", alanine)
    pipeline = ODesignProteinPipeline(backbone=backbone, protein_inverse_folder=protein, ligand_inverse_folder=ligand)
    with pytest.raises(UnsupportedError, match="design_breaks_connection"):
        pipeline(source, design=FIRST, fixed_atoms=(AtomSite("A", 0, "CA"),), seed=7, library=library)


@pytest.mark.parametrize("same_chain", [False, True])
def test_grnade_rejects_unsupported_context_only_in_designed_chain(
    library, inosine_components, placed, monkeypatch, same_chain
):
    if same_chain:
        recipe = Recipe({"R": PolymerSequence("polyribonucleotide", ("A", "I"))}, {"A": "R"})
        design = PositionSelection(components=(ComponentRef("A", 0),))
    else:
        recipe = Recipe(
            {
                "R": PolymerSequence("polyribonucleotide", ("A", "U")),
                "I": PolymerSequence("polyribonucleotide", ("I",)),
            },
            {"A": "I", "B": "R"},
        )
        design = PositionSelection(instances=("B",))
    system, _, _ = build_system(recipe, library, inosine_components)
    source = placed(system)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    backbone = ODesignBackboneComponent(model=small_model(), processor=processor)

    def sample_backbone(input, **_options):
        values = input.observed_positions[0].detach().cpu().numpy()[None]
        return sampling_output(type("Prepared", (), {"features": input})(), values)

    monkeypatch.setattr(backbone.model, "sample", sample_backbone)
    with torch.device("meta"):
        model = GrnadeModel(
            GrnadeConfig(
                node_hidden_scalar_size=8,
                node_hidden_vector_size=2,
                edge_hidden_scalar_size=8,
                edge_hidden_vector_size=2,
                num_blocks=1,
                dropout=0.0,
            )
        )
    calls = []

    def sample_sequence(input, *, num_samples, **_options):
        calls.append(1)
        rows = len(input.sequence_tokens)
        return GrnadeSamplingOutput(
            sequence_tokens=torch.full((num_samples, rows), TOKEN_INDEX["RNA_A"], dtype=torch.long),
            logits=torch.zeros(num_samples, rows, len(TOKENS)),
            batch_size=[],
        )

    monkeypatch.setattr(model, "sample", sample_sequence)
    pipeline = ODesignRnaPipeline(
        backbone=backbone, inverse_folder=GrnadePipeline(model=model, processor=GrnadeProcessor())
    )
    if same_chain:
        with pytest.raises(UnsupportedError, match="model_capacity"):
            pipeline(source, design=design, seed=7, library=library)
        assert not calls
    else:
        result = pipeline(source, design=design, seed=7, library=library)
        assert len(result.candidates) == 1 and calls == [1]
        assert (
            result.candidates[0].structure.system.text(
                "component", result.candidates[0].structure.system.component_symbol
            )[0]
            == "I"
        )
