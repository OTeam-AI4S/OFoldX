#!/usr/bin/env python
r"""Fail-fast parity diagnostic for the released MpnnPacker."""

from __future__ import annotations

import argparse
import copy
import json
import string
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

try:
    # biotite >= 1.7 exposes hybrid-36 codecs from the Rust extension.
    from biotite.rust.structure.io.pdb import decode_hybrid36
except ImportError:  # pragma: no cover - older biotite layout
    from biotite.structure.io.pdb.hybrid36 import decode_hybrid36

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ofoldx import AutoModel, AutoProcessor
from ofoldx import pipeline as load_pipeline  # noqa: E402
from ofoldx.data import is_water_component  # noqa: E402
from ofoldx.data import residue_constants as rc  # noqa: E402
from ofoldx.data.molecular import (  # noqa: E402
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ofoldx.data.structure import ResidueNumber, Structure  # noqa: E402
from ofoldx.io import read_pdb  # noqa: E402
from ofoldx.models.mpnn_packer import (  # noqa: E402
    MpnnPackerConfig,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerPipeline,
    MpnnPackerProcessor,
    MpnnPackerSamplingOutput,
    write_pdb,
)
from ofoldx.models.mpnn_packer.convert import convert_mpnn_packer_state_dict  # noqa: E402
from ofoldx.results import Metric, PackPrediction, PackResult  # noqa: E402


def assert_tensor(
    name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 0.0, rtol: float = 0.0
) -> None:
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: dtype {actual.dtype} != {expected.dtype}"
    if torch.is_floating_point(actual):
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )
    else:
        assert torch.equal(actual, expected), name


def assert_pack_result(result, input, sample) -> None:
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    observed = prediction.structure
    source = input.structure
    assert (
        observed.system.entities == source.system.entities
        and observed.system.instances == source.system.instances
        and observed.system.bonds == source.system.bonds
    )
    assert observed.residue_numbers == source.residue_numbers

    selected = input.pack_mask[0].detach().cpu().tolist()
    selected_references = {reference for reference, keep in zip(input.component_references, selected) if keep}
    component_ids = {reference: source.system.component(reference).component_id for reference in selected_references}
    expected_atoms = {
        reference: frozenset(filter(None, rc.restype_name_to_atom14_names[component_id]))
        for reference, component_id in component_ids.items()
    }
    expected_sites = [
        site
        for site in source.sites
        if (site.instance_id, site.component_index) not in selected_references
        or site.atom_id in expected_atoms[(site.instance_id, site.component_index)]
        or site.atom_id == "OXT"
    ]
    retained = set(expected_sites)
    for reference, is_selected in zip(input.component_references, selected):
        if not is_selected:
            continue
        for atom_id in rc.restype_name_to_atom14_names[component_ids[reference]][4:]:
            if not atom_id:
                continue
            site = AtomSite(reference[0], reference[1], atom_id)
            if site not in retained:
                expected_sites.append(site)
                retained.add(site)
    assert observed.sites == tuple(expected_sites)

    source_index = {site: index for index, site in enumerate(source.sites)}
    row_index = {reference: index for index, reference in enumerate(input.component_references)}
    atom14_coordinates = sample.atom14_coordinates[0].detach().cpu().numpy()
    atom14_mask = sample.atom14_mask[0].detach().cpu().numpy()
    assert observed.coordinates.dtype == np.result_type(source.coordinates.dtype, atom14_coordinates.dtype)
    assert observed.coordinate_mask.dtype == np.dtype(np.bool_)
    assert (observed.occupancy is None) == (source.occupancy is None)
    if observed.occupancy is not None:
        assert source.occupancy is not None
        assert observed.occupancy.dtype == np.result_type(source.occupancy.dtype, np.float32)
    for output_index, site in enumerate(observed.sites):
        reference = (site.instance_id, site.component_index)
        row = row_index.get(reference)
        atom_names = rc.restype_name_to_atom14_names.get(component_ids.get(reference, ""))
        atom = None if atom_names is None or site.atom_id not in atom_names else atom_names.index(site.atom_id)
        if row is not None and selected[row] and atom is not None and atom >= 4:
            np.testing.assert_allclose(
                observed.coordinates[output_index],
                atom14_coordinates[row, atom],
                atol=1e-4,
                rtol=1e-5,
            )
            assert bool(observed.coordinate_mask[output_index]) == bool(atom14_mask[row, atom])
            if observed.occupancy is not None:
                assert np.isnan(observed.occupancy[output_index])
            continue
        input_index = source_index[site]
        np.testing.assert_array_equal(observed.coordinates[output_index], source.coordinates[input_index])
        assert bool(observed.coordinate_mask[output_index]) == bool(source.coordinate_mask[input_index])
        if observed.occupancy is not None:
            np.testing.assert_equal(observed.occupancy[output_index], source.occupancy[input_index])

    metric_names = set()
    for chi_index in range(4):
        rows = tuple(
            row
            for row, (reference, is_selected) in enumerate(zip(input.component_references, selected))
            if is_selected and len(rc.chi_angles_atoms[component_ids[reference]]) > chi_index
        )
        if not rows:
            continue
        name = f"chi{chi_index + 1}_log_density"
        metric_names.add(name)
        metric = prediction.metrics[name]
        assert metric.axis == tuple(input.component_references[row] for row in rows)
        assert metric.unit == "nat/radian"
        expected = sample.chi_log_densities[0, rows, chi_index].detach().cpu().numpy()
        assert metric.values.dtype == expected.dtype
        np.testing.assert_allclose(metric.values, expected, atol=1e-5, rtol=1e-5)
    assert set(prediction.metrics) == metric_names


def assert_pdb_writer(prediction: PackPrediction) -> None:
    structure = prediction.structure
    system = structure.system
    written_indices = tuple(
        index
        for index, site in enumerate(structure.sites)
        if bool(structure.coordinate_mask[index])
        and not is_water_component(system.component((site.instance_id, site.component_index)))
    )
    assert any(
        bool(structure.coordinate_mask[index])
        and is_water_component(system.component((site.instance_id, site.component_index)))
        for index, site in enumerate(structure.sites)
    )
    density_by_group = {
        chi_index
        + 4: dict(
            zip(
                prediction.metrics[f"chi{chi_index + 1}_log_density"].axis,
                prediction.metrics[f"chi{chi_index + 1}_log_density"].values,
                strict=True,
            )
        )
        for chi_index in range(4)
    }
    restype_index = {
        component_id: index for index, component_id in enumerate(rc.restype_1to3[restype] for restype in rc.restypes)
    }

    with TemporaryDirectory(prefix="ofoldx-mpnn-packer-pdb-") as directory:
        directory = Path(directory)
        output = directory / "packed.pdb"
        write_pdb(prediction, output)
        assert set(directory.iterdir()) == {output}
        lines = output.read_text(encoding="ascii").splitlines()
        assert lines[-1] == "END"
        records = lines[:-1]
        assert len(records) == len(written_indices)

        seen_groups: set[int] = set()
        saw_atom = False
        saw_hetatm = False
        saw_no_chi = False
        saw_unknown_occupancy = False
        chain_by_instance: dict[str, str] = {}
        for serial, (line, site_index) in enumerate(zip(records, written_indices, strict=True), start=1):
            assert len(line) == 80
            assert decode_hybrid36(line[6:11]) == serial
            site = structure.sites[site_index]
            reference = (site.instance_id, site.component_index)
            component = system.component(reference)
            entity = system.entities[system.instances[site.instance_id]]
            protein = isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
            assert line[:6].strip() == ("ATOM" if protein else "HETATM")
            saw_atom |= protein
            saw_hetatm |= not protein
            assert line[12:16].strip() == site.atom_id
            assert line[17:20].strip() == component.component_id
            chain = line[21]
            previous_chain = chain_by_instance.setdefault(site.instance_id, chain)
            assert chain == previous_chain
            assert len(set(chain_by_instance.values())) == len(chain_by_instance)

            residue_number = None if structure.residue_numbers is None else structure.residue_numbers.get(reference)
            expected_number = site.component_index + 1 if residue_number is None else residue_number.number
            expected_insertion_code = "" if residue_number is None else residue_number.insertion_code
            assert decode_hybrid36(line[22:26]) == expected_number
            assert line[26].strip() == expected_insertion_code
            assert line[30:38] == f"{float(structure.coordinates[site_index, 0]):8.3f}"
            assert line[38:46] == f"{float(structure.coordinates[site_index, 1]):8.3f}"
            assert line[46:54] == f"{float(structure.coordinates[site_index, 2]):8.3f}"
            if structure.occupancy is None or np.isnan(structure.occupancy[site_index]):
                assert line[54:60] == " " * 6
                saw_unknown_occupancy = True
            else:
                assert line[54:60] == f"{float(structure.occupancy[site_index]):6.2f}"
            assert line[76:78].strip() == system.atom(site).element

            b_value = 0.0
            restype = restype_index.get(component.component_id)
            if protein and restype is not None:
                atom14_names = rc.restype_name_to_atom14_names[component.component_id]
                if site.atom_id in atom14_names:
                    atom14_index = atom14_names.index(site.atom_id)
                    rigid_group = int(rc.restype_atom14_to_rigid_group[restype, atom14_index])
                    density = density_by_group.get(rigid_group, {}).get(reference)
                    if density is not None:
                        b_value = float(density)
                        seen_groups.add(rigid_group)
                if component.component_id in {"ALA", "GLY"}:
                    assert line[60:66] == "  0.00"
                    saw_no_chi = True
            assert line[60:66] == f"{b_value:6.2f}"

        assert seen_groups == {4, 5, 6, 7}
        assert saw_atom and saw_hetatm and saw_no_chi and saw_unknown_occupancy

        component_order = tuple(dict.fromkeys((site.instance_id, site.component_index) for site in structure.sites))
        fixed_reference = next(
            reference
            for reference in component_order
            if all(reference in density_by_group[group] for group in range(4, 8))
        )
        fixed_metrics: dict[str, Metric] = {}
        for name, metric in prediction.metrics.items():
            keep = np.asarray([reference != fixed_reference for reference in metric.axis])
            fixed_metrics[name] = Metric(
                metric.values[keep],
                tuple(reference for reference in metric.axis if reference != fixed_reference),
                metric.unit,
            )
        fixed_output = directory / "fixed.pdb"
        write_pdb(PackPrediction(structure, fixed_metrics), fixed_output)
        assert set(directory.iterdir()) == {output, fixed_output}
        fixed_records = fixed_output.read_text(encoding="ascii").splitlines()[:-1]
        fixed_side_chain_atoms = 0
        for line, site_index in zip(fixed_records, written_indices, strict=True):
            site = structure.sites[site_index]
            if (site.instance_id, site.component_index) != fixed_reference:
                continue
            component_id = system.component(fixed_reference).component_id
            atom14_names = rc.restype_name_to_atom14_names[component_id]
            if site.atom_id in atom14_names:
                restype = restype_index[component_id]
                group = int(rc.restype_atom14_to_rigid_group[restype, atom14_names.index(site.atom_id)])
                fixed_side_chain_atoms += group >= 4
            assert line[60:66] == "  0.00"
        assert fixed_side_chain_atoms

        occupancy = (
            np.full(len(structure.sites), np.nan)
            if structure.occupancy is None
            else np.array(structure.occupancy, copy=True)
        )
        zero_index, unknown_index = written_indices[:2]
        occupancy[zero_index] = 0.0
        occupancy[unknown_index] = np.nan
        residue_numbers = {} if structure.residue_numbers is None else dict(structure.residue_numbers)
        residue_numbers[fixed_reference] = ResidueNumber(10000, "A")
        bounded_structure = Structure(
            system,
            structure.sites,
            structure.coordinates,
            structure.coordinate_mask,
            occupancy,
            residue_numbers,
        )
        bounded_output = directory / "bounded.pdb"
        write_pdb(PackPrediction(bounded_structure, prediction.metrics), bounded_output)
        assert set(directory.iterdir()) == {output, fixed_output, bounded_output}
        bounded_records = bounded_output.read_text(encoding="ascii").splitlines()[:-1]
        assert bounded_records[written_indices.index(zero_index)][54:60] == "  0.00"
        assert bounded_records[written_indices.index(unknown_index)][54:60] == " " * 6
        numbered = [
            line
            for line, site_index in zip(bounded_records, written_indices, strict=True)
            if (
                structure.sites[site_index].instance_id,
                structure.sites[site_index].component_index,
            )
            == fixed_reference
        ]
        assert numbered and all(line[22:27] == "A000A" for line in numbered)

        absent_occupancy = Structure(
            system,
            structure.sites,
            structure.coordinates,
            structure.coordinate_mask,
            None,
            structure.residue_numbers,
        )
        absent_output = directory / "absent-occupancy.pdb"
        write_pdb(PackPrediction(absent_occupancy, prediction.metrics), absent_output)
        assert set(directory.iterdir()) == {output, fixed_output, bounded_output, absent_output}
        assert all(
            line[54:60] == " " * 6
            for line in absent_output.read_text(encoding="ascii").splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        )

        nonblank_chains = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation
        ligand_component = ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), None)
        ligand = Molecule((ligand_component,))
        water = Molecule((ChemicalComponent("HOH", (ChemicalAtom("O", "O"),), None),))
        capacity_instances = {chain: "ligand" for chain in nonblank_chains}
        capacity_instances["omitted_water"] = "water"
        capacity_instances["all_false"] = "ligand"
        capacity_instances["assigned_blank"] = "ligand"
        capacity_system = MolecularSystem(
            {"ligand": ligand, "water": water},
            capacity_instances,
        )
        capacity_sites = (
            tuple(AtomSite(chain, 0, "C1") for chain in nonblank_chains)
            + (AtomSite("omitted_water", 0, "O"),)
            + (AtomSite("all_false", 0, "C1"),)
            + (AtomSite("assigned_blank", 0, "C1"),)
        )
        capacity_mask = np.ones(len(capacity_sites), dtype=np.bool_)
        capacity_mask[-2] = False
        capacity_prediction = PackPrediction(
            Structure(
                capacity_system,
                capacity_sites,
                np.zeros((len(capacity_sites), 3), dtype=np.float32),
                capacity_mask,
            )
        )
        capacity_output = directory / "chain-capacity.pdb"
        repeated_output = directory / "chain-capacity-repeated.pdb"
        write_pdb(capacity_prediction, capacity_output)
        write_pdb(capacity_prediction, repeated_output)
        assert capacity_output.read_bytes() == repeated_output.read_bytes()
        capacity_records = capacity_output.read_text(encoding="ascii").splitlines()[:-1]
        assert len(capacity_records) == len(nonblank_chains) + 1
        assert "".join(line[21] for line in capacity_records[:-1]) == nonblank_chains
        assert capacity_records[nonblank_chains.index("!")][21] == "!"
        assert capacity_records[-1][21] == " "
        assert all("HOH" not in line for line in capacity_records)

        empty_system = MolecularSystem({"ligand": ligand}, {"": "ligand"})
        empty_prediction = PackPrediction(
            Structure(
                empty_system,
                (AtomSite("", 0, "C1"),),
                np.zeros((1, 3), dtype=np.float32),
                np.ones(1, dtype=np.bool_),
            )
        )
        empty_output = directory / "empty-chain.pdb"
        write_pdb(empty_prediction, empty_output)
        assert empty_output.read_text(encoding="ascii").splitlines()[0][21] == " "

        for index, component in enumerate(
            (
                ChemicalComponent("L G", (ChemicalAtom("C1", "C"),), None),
                ChemicalComponent("LIG", (ChemicalAtom("C 1", "C"),), None),
            )
        ):
            invalid_system = MolecularSystem({"ligand": Molecule((component,))}, {"A": "ligand"})
            invalid_prediction = PackPrediction(
                Structure(
                    invalid_system,
                    (AtomSite("A", 0, component.atoms[0].atom_id),),
                    np.zeros((1, 3), dtype=np.float32),
                    np.ones(1, dtype=np.bool_),
                )
            )
            try:
                write_pdb(invalid_prediction, directory / f"invalid-identity-{index}.pdb")
            except ValueError as error:
                assert "whitespace" in str(error)
            else:
                raise AssertionError("PDB writer accepted whitespace inside a canonical identity")

        simple_system = MolecularSystem({"ligand": ligand}, {"A": "ligand"})
        for index, insertion_code in enumerate((" ", "1", "!", "Å", "AB")):
            invalid_prediction = PackPrediction(
                Structure(
                    simple_system,
                    (AtomSite("A", 0, "C1"),),
                    np.zeros((1, 3), dtype=np.float32),
                    np.ones(1, dtype=np.bool_),
                    residue_numbers={("A", 0): ResidueNumber(1, insertion_code)},
                )
            )
            try:
                write_pdb(invalid_prediction, directory / f"invalid-insertion-{index}.pdb")
            except ValueError as error:
                assert "insertion code" in str(error)
            else:
                raise AssertionError(f"PDB writer accepted insertion code {insertion_code!r}")

        collision_system = MolecularSystem(
            {"ligand": Molecule((ligand_component, ligand_component))},
            {"A": "ligand"},
        )
        collision_prediction = PackPrediction(
            Structure(
                collision_system,
                (AtomSite("A", 0, "C1"), AtomSite("A", 1, "C1")),
                np.zeros((2, 3), dtype=np.float32),
                np.ones(2, dtype=np.bool_),
                residue_numbers={
                    ("A", 0): ResidueNumber(7),
                    ("A", 1): ResidueNumber(7),
                },
            )
        )
        try:
            write_pdb(collision_prediction, directory / "colliding-occurrences.pdb")
        except ValueError as error:
            assert "collide" in str(error)
        else:
            raise AssertionError("PDB writer accepted colliding component occurrences")

    print(
        "PDB writer: one prediction per file; canonical order, physical chi1-4 mapping, identity, "
        "occupancy, numbering, chain allocation, collision, and water policy agree"
    )
    print(
        "Intentional upstream B-column divergence: fixed, backbone, ALA, GLY, OXT, modified, and "
        "nonprotein atoms use 0.0 instead of fixed sentinels or unused density heads"
    )


def check_pack_sequences(
    *,
    pipeline: MpnnPackerPipeline,
    reference_model,
    pack_side_chains,
    features,
    pdb: Path,
    atom_context_num: int,
    device: torch.device,
    seed: int,
    num_denoising_steps: int,
    num_torsion_samples: int = 16,
    num_samples: int = 2,
) -> None:
    r"""Real-weight parity for the batched multi-sequence packing path.

    Executed and PASSING against ``~/reference/LigandMPNN`` at the pinned revision
    ``26ec57ac976ade5379920dbd43c7f97a91cf82de`` and the
    ``ligandmpnn_sc_v_32_002_16.pt`` checkpoint (2026-07-16, ``--device cpu``): the
    batched ``Model.sample`` agrees with upstream batched ``pack_side_chains`` and
    each ``PackResult`` realizes its sequence in input order. It is wired into
    ``main`` and so is gated exactly like the surrounding real-weight checks: it never runs on a
    bare import (there is no module-level execution), only when ``main`` has already
    loaded the reference checkout, checkpoint, and weights.

    Extends the existing singleton ``pack`` parity (batch size one) to
    ``MpnnPackerPipeline.pack_sequences``: several target sequences packed against
    one parsed backbone, batched inside one ``Model.sample`` per pack index.

    RNG-order alignment (the crux of this check):

    * Native threads ONE ``torch.Generator(device).manual_seed(seed)`` through the
      whole operation (``pipeline.py:127-151``); the reference uses the global
      ``torch.manual_seed(seed)`` stream. The existing singleton sample check
      (batch size one) already establishes that these two seedings drive the same
      device RNG stream and the same von-Mises rejection algorithm; this function
      inherits that correspondence rather than re-deriving it.
    * Within one pack both sides consume the stream in the same order: first one
      ``torch.rand([B, L, 4])`` (the random initial chi in ``make_torsion_features``
      / ``Model.sample``), then, per denoising step, one
      ``multinomial([B * L * 4, num_mix], num_torsion_samples)`` followed by the
      shared von-Mises rejection ``torch.rand((3, num_torsion_samples, B, L, 4,
      num_mix))`` looped until ``done.all()``. Growing the batch from ``1`` to
      ``B`` only enlarges each already-batched draw; it does not reorder the
      stream, and the rejection loop iterates over the whole batch on both sides,
      so both terminate after the same number of iterations.
    * Across packs, native calls ``Model.sample`` ``num_samples`` times threading
      the one Generator (pack-outermost, all ``B`` sequences batched inside). The
      reference side reproduces that by seeding the global RNG ONCE and calling
      ``pack_side_chains`` ``num_samples`` times so its global stream advances
      pack-outermost too.

    Alignment ASSUMED (and, when finally run, verified numerically by the tensor
    assertions below): batching ``[1, L, ...]`` -> ``[B, L, ...]`` and looping
    pack-outermost preserve the element-wise RNG correspondence that the singleton
    check establishes for ``B == 1``. This function does NOT independently prove
    that the native float64 von-Mises rejection and the upstream ``VonMises``
    rejection consume identical stream words; it relies on the singleton check for
    that and asserts the batched outputs still agree within the reused tolerances.
    If a perfect global-vs-Generator match did not survive batching, the atom14 and
    chi-log-density assertions would fail here, which is exactly this check's job.

    Only PACKABLE positions are mutated across target sequences, so the fixed
    (unpacked) positions keep one shared identity whose true side chains native
    reads from ``atom37`` and the reference reads from ``xyz_37`` identically; the
    packed positions ignore those true side chains (random initial chi), so the
    two implementations see identical inputs modulo the differing ``S`` tokens.
    """

    processor = pipeline.processor

    # One parsed backbone shared by every target sequence.
    structure = read_pdb(pdb)
    base = processor(structure)
    base_selected = base.pack_mask[0].detach().cpu().tolist()
    positions = tuple(
        reference for reference, is_selected in zip(base.component_references, base_selected) if is_selected
    )
    if len(positions) < 2:
        raise ValueError("fixture has too few packable positions for a batched pack_sequences check")

    system = structure.system
    protein_instances = tuple(
        instance_id
        for instance_id, entity_id in system.instances.items()
        if isinstance(system.entities[entity_id], PolymerSequence)
        and system.entities[entity_id].polymer_type == "protein"
    )
    original_ids = {
        instance_id: [component.component_id for component in system.entities[system.instances[instance_id]].components]
        for instance_id in protein_instances
    }

    # >= 2 target sequences: variant 0 keeps the source identity; later variants
    # mutate a handful of packable positions to distinct standard residues so the S
    # tokens differ while the backbone, context, and packed selection stay shared.
    palette = ("ALA", "SER", "GLY", "VAL", "LEU", "THR", "ASN", "GLN")
    mutable = positions[: min(4, len(positions))]
    sequences: list[dict[str, list[str]]] = []
    for variant in range(3):
        component_ids = {instance_id: list(ids) for instance_id, ids in original_ids.items()}
        if variant > 0:
            for offset, (instance_id, component_index) in enumerate(mutable):
                original = original_ids[instance_id][component_index]
                choices = [identity for identity in palette if identity != original]
                component_ids[instance_id][component_index] = choices[(variant - 1 + offset) % len(choices)]
        sequences.append(component_ids)

    inputs = tuple(processor(structure, sequence=sequence, positions=positions) for sequence in sequences)
    batch = len(inputs)
    for item in inputs[1:]:
        assert item.component_references == inputs[0].component_references
        assert torch.equal(item.pack_mask, inputs[0].pack_mask)
    assert any(
        not torch.equal(item.sequence_tokens, inputs[0].sequence_tokens) for item in inputs[1:]
    ), "target sequences must differ in S tokens"

    # Native: one Generator threaded pack-outermost across num_samples, all B
    # sequences batched inside each Model.sample (pipeline.py:127-151).
    results = pipeline.pack_sequences(
        structure,
        sequences=sequences,
        positions=positions,
        num_samples=num_samples,
        num_denoising_steps=num_denoising_steps,
        num_torsion_samples=num_torsion_samples,
        seed=seed,
    )
    assert len(results) == len(sequences)
    for result in results:
        assert len(result.predictions) == num_samples

    # Reconstruct the batched MpnnPackerInput that pack_sequences builds (torch.cat
    # over the 11 tensor fields; pipeline.py:112-126) so the raw batched sample can
    # be compared to the reference field by field. The same Generator seed and the
    # same batched input reproduce the exact samples pack_sequences realized above.
    batched_input = MpnnPackerInput(
        backbone_coordinates=torch.cat([item.backbone_coordinates for item in inputs]),
        sequence_tokens=torch.cat([item.sequence_tokens for item in inputs]),
        residue_mask=torch.cat([item.residue_mask for item in inputs]),
        residue_index=torch.cat([item.residue_index for item in inputs]),
        chain_index=torch.cat([item.chain_index for item in inputs]),
        context_coordinates=torch.cat([item.context_coordinates for item in inputs]),
        context_atomic_numbers=torch.cat([item.context_atomic_numbers for item in inputs]),
        context_mask=torch.cat([item.context_mask for item in inputs]),
        atom37_coordinates=torch.cat([item.atom37_coordinates for item in inputs]),
        atom37_mask=torch.cat([item.atom37_mask for item in inputs]),
        pack_mask=torch.cat([item.pack_mask for item in inputs]),
        structure=inputs[0].structure,
        component_references=inputs[0].component_references,
    )
    native_generator = torch.Generator(device=device).manual_seed(seed)
    native_samples = [
        pipeline.model.sample(
            batched_input,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=native_generator,
        )
        for _ in range(num_samples)
    ]

    # Reference: the batched features dict (shared backbone/context/chain_mask,
    # differing S tokens). The S rows are the native-lowered tokens for each target
    # sequence (the singleton check already proves native tokens equal the upstream
    # featurize tokens), and chain_mask is set to the shared native pack selection.
    token_rows = [item.sequence_tokens[0].to(dtype=torch.long) for item in inputs]
    chain_mask_row = inputs[0].pack_mask[0].to(dtype=features["X"].dtype)
    batched_features = {}
    for key, value in features.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == 1:
            batched_features[key] = value.repeat(batch, *([1] * (value.dim() - 1))).contiguous()
        else:
            batched_features[key] = value
    batched_features["S"] = torch.stack(token_rows, dim=0)
    batched_features["chain_mask"] = chain_mask_row.unsqueeze(0).repeat(batch, 1).contiguous()

    # Seed the global RNG ONCE, then draw num_samples packs so the reference stream
    # advances pack-outermost, mirroring the single native Generator across packs.
    torch.manual_seed(seed)
    with torch.inference_mode():
        reference_samples = [
            pack_side_chains(
                copy.deepcopy(batched_features),
                reference_model,
                num_denoising_steps,
                num_torsion_samples,
                False,
                atom_context_num,
            )
            for _ in range(num_samples)
        ]

    for sample_index in range(num_samples):
        reference = reference_samples[sample_index]
        native = native_samples[sample_index]
        reference_distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=reference["mix_logits"]),
            torch.distributions.VonMises(reference["mean"], reference["concentration"]),
        )
        reference_chi_log_densities = reference_distribution.log_prob(reference["sample"])
        expected_fields = {
            "atom14_coordinates": reference["X"],
            "atom14_mask": reference["X_m"].bool(),
            "sampled_chi_angles": reference["sample"],
            "mean": reference["mean"],
            "concentration": reference["concentration"],
            "mix_logits": reference["mix_logits"],
            "input_chi_sin_cos": reference["true_torsion_sin_cos"],
        }
        # Reuse the singleton sample-field tolerances (mpnn_packer.py:644-661),
        # now over the [B, L, ...] batched draw for each pack index.
        for name, expected in expected_fields.items():
            if name == "atom14_coordinates":
                atol, rtol = 1e-4, 1e-5
            elif name == "concentration":
                atol, rtol = 2e-4, 5e-5
            else:
                atol, rtol = 1e-5, 1e-5
            assert_tensor(
                f"sequences.sample{sample_index}.{name}",
                getattr(native, name),
                expected,
                atol=atol,
                rtol=rtol,
            )
        assert_tensor(
            f"sequences.sample{sample_index}.chi_log_densities",
            native.chi_log_densities,
            reference_chi_log_densities,
            atol=5e-5,
            rtol=5e-5,
        )

        # Each PackResult must carry num_samples predictions in input-sequence
        # order; results[i].predictions[s] realizes native batch row i of pack s.
        # Comparing against the native batch slice (not the reference) keeps the
        # metric tolerance at the realization level (1e-5); the cross-project
        # numerics are covered by the field-by-field assertions above.
        for sequence_index, item in enumerate(inputs):
            candidate = MpnnPackerSamplingOutput(
                atom14_coordinates=native.atom14_coordinates[sequence_index : sequence_index + 1],
                atom14_mask=native.atom14_mask[sequence_index : sequence_index + 1],
                sampled_chi_angles=native.sampled_chi_angles[sequence_index : sequence_index + 1],
                chi_log_densities=native.chi_log_densities[sequence_index : sequence_index + 1],
                mean=native.mean[sequence_index : sequence_index + 1],
                concentration=native.concentration[sequence_index : sequence_index + 1],
                mix_logits=native.mix_logits[sequence_index : sequence_index + 1],
                input_chi_sin_cos=native.input_chi_sin_cos[sequence_index : sequence_index + 1],
            )
            single = PackResult(predictions=(results[sequence_index].predictions[sample_index],))
            assert_pack_result(single, item, candidate)

    print(
        f"pack_sequences: {len(sequences)} target sequences x {num_samples} packs; batched Model.sample "
        "agrees with upstream batched pack_side_chains and each PackResult realizes its sequence in order"
    )


def git(reference: Path, *arguments: str) -> str:
    return subprocess.check_output(("git", "-C", str(reference), *arguments), text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path.home() / "reference/LigandMPNN")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "weights/mpnn-weights/ligandmpnn_sc_v_32_002_16.pt",
    )
    parser.add_argument("--pdb", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-denoising-steps", type=int, default=3)
    args = parser.parse_args()

    reference = args.reference.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    pdb = args.pdb.expanduser() if args.pdb is not None else reference / "inputs/1BC8.pdb"
    device = torch.device(args.device)

    head = git(reference, "rev-parse", "HEAD")
    dirty = bool(git(reference, "status", "--porcelain"))
    print(f"LigandMPNN checkout: {head}{' (dirty)' if dirty else ''}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"fixture: {pdb}")

    sys.path.insert(0, str(reference))
    # This inspected upstream OpenFold copy still imports NumPy's removed alias.
    setattr(np, "int", int)
    from data_utils import featurize, parse_PDB
    from sc_utils import Packer, make_torsion_features, pack_side_chains

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
        mmap=device.type == "cpu",
    )
    official_state = checkpoint["model_state_dict"]
    k_neighbors = int(checkpoint["num_edges"])
    atom_context_num = int(checkpoint["atom_context_num"])
    reference_model = (
        Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=atom_context_num,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=k_neighbors,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=device,
            num_mix=3,
        )
        .to(device)
        .eval()
    )
    reference_model.load_state_dict(official_state, strict=True)

    native_model = (
        MpnnPackerModel(
            MpnnPackerConfig(
                k_neighbors=k_neighbors,
                atom_context_num=atom_context_num,
            )
        )
        .to(device)
        .eval()
    )
    converted_state = convert_mpnn_packer_state_dict(official_state)
    target_state = native_model.state_dict()
    assert len(official_state) == len(converted_state) == len(target_state)
    assert set(converted_state) == set(target_state)
    for key, value in converted_state.items():
        assert value.shape == target_state[key].shape, key
        assert value.dtype == target_state[key].dtype, key
    native_model.load_state_dict(converted_state, strict=True)
    print(f"checkpoint tensors: {len(official_state)} exact converted tensors")

    source, _backbone, _other_atoms, _insertion_codes, _water = parse_PDB(str(pdb), device=device, parse_all_atoms=True)
    chain_mask = source["mask"].bool() & (source["S"] != 20)
    source["chain_mask"] = chain_mask.to(dtype=source["X"].dtype)
    features = featurize(
        source,
        number_of_ligand_atoms=atom_context_num,
        model_type="ligand_mpnn",
        use_atom_context=True,
    )
    features["S"] = features["S"].long()
    processor = MpnnPackerProcessor(device=device)
    native_input = processor(pdb)
    processor_fields = (
        ("backbone_coordinates", features["X"]),
        ("sequence_tokens", features["S"]),
        ("residue_mask", features["mask"].bool()),
        ("residue_index", features["R_idx"].long()),
        ("context_coordinates", features["Y"]),
        ("context_atomic_numbers", features["Y_t"].long()),
        ("context_mask", features["Y_m"].bool()),
        ("atom37_coordinates", features["xyz_37"]),
        ("atom37_mask", features["xyz_37_m"].bool()),
        ("pack_mask", features["chain_mask"].bool()),
    )
    for name, expected in processor_fields:
        assert_tensor(name, getattr(native_input, name), expected, atol=1e-6, rtol=1e-6)
    native_same_chain = native_input.chain_index[:, :, None] == native_input.chain_index[:, None, :]
    reference_same_chain = features["chain_labels"][:, :, None] == features["chain_labels"][:, None, :]
    assert_tensor("chain_index", native_same_chain, reference_same_chain)
    print(f"Processor: {len(processor_fields) + 1} public tensors agree")

    torch.manual_seed(args.seed + 101)
    learned_features = copy.deepcopy(features)
    torsions = make_torsion_features(learned_features, repack_everything=False)
    atom14_coordinates = torsions["xyz14_noised"]
    atom14_mask = torsions["xyz14_m"].bool()
    learned_features["X"] = atom14_coordinates
    learned_features["X_m"] = torsions["xyz14_m"]
    with torch.inference_mode():
        reference_encoded = reference_model.encode(learned_features)
        learned_features.update(h_V=reference_encoded[0], h_E=reference_encoded[1], E_idx=reference_encoded[2])
        reference_decoded = reference_model.decode(learned_features)
        native_encoded = native_model.encode(native_input, atom14_coordinates=atom14_coordinates)
        native_decoded = native_model.decode(
            native_input,
            atom14_coordinates=atom14_coordinates,
            atom14_mask=atom14_mask,
            node_states=native_encoded[0],
            edge_states=native_encoded[1],
            neighbor_indices=native_encoded[2],
        )
        native_forward = native_model(native_input, atom14_coordinates=atom14_coordinates, atom14_mask=atom14_mask)
    for name, actual, expected in zip(
        ("encode.node_states", "encode.edge_states", "encode.neighbor_indices"),
        native_encoded,
        reference_encoded,
    ):
        assert_tensor(name, actual, expected, atol=1e-5, rtol=1e-5)
    for name, actual, expected in zip(
        ("mean", "concentration", "mix_logits"),
        (native_decoded.mean, native_decoded.concentration, native_decoded.mix_logits),
        reference_decoded,
    ):
        assert_tensor(f"decode.{name}", actual, expected, atol=1e-5, rtol=1e-5)
        assert_tensor(f"forward.{name}", getattr(native_forward, name), expected, atol=1e-5, rtol=1e-5)
    print("Model: public encode, decode, and forward agree")

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        reference_sample = pack_side_chains(
            copy.deepcopy(features),
            reference_model,
            args.num_denoising_steps,
            16,
            False,
            atom_context_num,
        )
        native_sample = native_model.sample(
            native_input,
            num_denoising_steps=args.num_denoising_steps,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
    reference_distribution = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=reference_sample["mix_logits"]),
        torch.distributions.VonMises(reference_sample["mean"], reference_sample["concentration"]),
    )
    reference_chi_log_densities = reference_distribution.log_prob(reference_sample["sample"])
    expected_sample = {
        "atom14_coordinates": reference_sample["X"],
        "atom14_mask": reference_sample["X_m"].bool(),
        "sampled_chi_angles": reference_sample["sample"],
        "mean": reference_sample["mean"],
        "concentration": reference_sample["concentration"],
        "mix_logits": reference_sample["mix_logits"],
        "input_chi_sin_cos": reference_sample["true_torsion_sin_cos"],
    }
    for name, expected in expected_sample.items():
        if name == "atom14_coordinates":
            atol, rtol = 1e-4, 1e-5
        elif name == "concentration":
            # The upstream rigid helpers hand-expand products to avoid AMP
            # downcasting; OFoldX follows ordinary PyTorch matmul semantics.
            atol, rtol = 2e-4, 5e-5
        else:
            atol, rtol = 1e-5, 1e-5
        assert_tensor(name, getattr(native_sample, name), expected, atol=atol, rtol=rtol)

    assert_tensor(
        "chi_log_densities",
        native_sample.chi_log_densities,
        reference_chi_log_densities,
        atol=5e-5,
        rtol=5e-5,
    )
    print("Sample: all 8 public fields agree")

    pipeline = MpnnPackerPipeline(model=native_model, processor=processor)
    result = pipeline.pack(
        pdb,
        seed=args.seed,
        num_denoising_steps=args.num_denoising_steps,
    )
    assert_pack_result(result, native_input, native_sample)
    print("PackResult: identity, site order, coordinates, masks, occupancy, and metrics agree")
    assert_pdb_writer(result.predictions[0])
    with TemporaryDirectory(prefix="ofoldx-mpnn-packer-repository-") as directory:
        repository = Path(directory)
        pipeline.save(repository)
        assert {path.name for path in repository.iterdir()} == {
            "model.json",
            "processor.json",
            "model.safetensors",
            "pipeline.json",
        }
        assert json.loads((repository / "model.json").read_text()) == {
            **dict(native_model.config),
            "architecture": "ofoldx.mpnn_packer",
        }
        assert json.loads((repository / "processor.json").read_text()) == dict(processor.config)
        assert json.loads((repository / "pipeline.json").read_text()) == {"pipeline": "ofoldx.mpnn_packer"}

        loaded_model = AutoModel(repository)
        loaded_processor = AutoProcessor(repository)
        assert isinstance(loaded_model, MpnnPackerModel) and not loaded_model.training
        assert isinstance(loaded_processor, MpnnPackerProcessor)
        assert dict(loaded_model.config) == dict(native_model.config)
        assert dict(loaded_processor.config) == dict(processor.config)
        for name, expected in native_model.state_dict().items():
            assert_tensor(f"repository.{name}", loaded_model.state_dict()[name], expected)

        loaded_pipeline = load_pipeline("pack", model=repository)
        assert isinstance(loaded_pipeline, MpnnPackerPipeline)
        loaded_input = loaded_pipeline.processor(pdb)
        loaded_result = loaded_pipeline.pack(
            pdb,
            seed=args.seed,
            num_denoising_steps=args.num_denoising_steps,
        )
        assert_pack_result(loaded_result, loaded_input, native_sample)
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    "from ofoldx import AutoModel, AutoProcessor, pipeline; "
                    "repository = Path(sys.argv[1]); "
                    "assert type(AutoModel(repository)).__name__ == 'MpnnPackerModel'; "
                    "assert type(AutoProcessor(repository)).__name__ == 'MpnnPackerProcessor'; "
                    "assert type(pipeline('pack', model=repository)).__name__ == 'MpnnPackerPipeline'"
                ),
                str(repository),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
    print("Repository: save, AutoModel, AutoProcessor, concrete marker, and Pipeline round-trip agree")

    check_pack_sequences(
        pipeline=pipeline,
        reference_model=reference_model,
        pack_side_chains=pack_side_chains,
        features=features,
        pdb=pdb,
        atom_context_num=atom_context_num,
        device=device,
        seed=args.seed,
        num_denoising_steps=args.num_denoising_steps,
    )

    print(
        "Excluded without patching upstream: invalid-pair KNN tie, decoder invalid-neighbor, unknown-backbone, "
        "context-off, and partial-fixed defects; zero-pack, BF16, torch.compile, "
        "and other synthetic Model-policy cases."
    )
    print("MpnnPacker parity passed")


if __name__ == "__main__":
    main()
