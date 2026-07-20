#!/usr/bin/env python
r"""Qualify OFoldX RFdiffusion against the released implementation.

Reference and native modes run in their own environments and exchange one local
tensor capture. Native mode replays the captured initial state and raw
self-conditioning with reverse noise disabled, then separately checks that the
public ``Model.design(generator=...)`` path is reproducible. The two projects use
different RNG implementations, so equal integer seeds are deliberately not treated
as equal stochastic trajectories. No report is written: comparison is printed to
stdout and failure is signaled by a nonzero exit code. ``matrix`` runs every
declared checkpoint/mode case and therefore requires every checkpoint in
``--checkpoint-dir``. ``--compile-only`` runs a self-contained tiny-model smoke
through default Inductor with ``fullgraph=True`` and checks every output leaf.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import pickle
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CASES = {
    "base": {"checkpoint": "Base_ckpt.pt", "final_step": 49},
    "complex": {"checkpoint": "Complex_base_ckpt.pt", "final_step": 49},
    "fold": {"checkpoint": "Complex_Fold_base_ckpt.pt", "final_step": 49},
    "inpaint": {"checkpoint": "InpaintSeq_ckpt.pt", "final_step": 49},
    "inpaint_fold": {"checkpoint": "InpaintSeq_Fold_ckpt.pt", "final_step": 49},
    "active_site": {"checkpoint": "ActiveSite_ckpt.pt", "final_step": 49},
    "symmetry": {"checkpoint": "Base_epoch8_ckpt.pt", "final_step": 49},
    "partial": {"checkpoint": "Base_ckpt.pt", "final_step": 9, "partial_steps": 10},
}
SCHEDULE_NAME = "T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl"
SCHEDULE_SOURCE_SHA256 = "cd79d3b77b3dc80d727ae0295f932b88cdfc0e5805e0ced29ce3c719518a1e23"
SCHEDULE_ASSET_SHA256 = "89f97b26604e875e05907d465856d9b5ae5f9375482cc463dfa8d007d6e05db8"


def _schedule_parity(reference_root: Path) -> dict:
    from ofoldx.models.rfdiffusion.model import rfdiffusion_schedule

    source_bytes = (reference_root / "schedules" / SCHEDULE_NAME).read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SCHEDULE_SOURCE_SHA256:
        raise ValueError("audited RFdiffusion schedule source asset SHA-256 does not match")
    source = pickle.loads(source_bytes)
    packaged = rfdiffusion_schedule()
    asset = importlib.resources.files("ofoldx.models.rfdiffusion").joinpath("rfdiffusion_schedule.npz")
    asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
    comparisons = {}
    for name in source:
        expected = source[name]
        actual = packaged[name]
        comparisons[name] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "shape_matches": actual.shape == expected.shape,
            "dtype_matches": actual.dtype == expected.dtype,
            "values_match": np.array_equal(actual, expected),
        }
    passed = (
        tuple(packaged) == tuple(source)
        and asset_sha256 == SCHEDULE_ASSET_SHA256
        and all(
            value["shape_matches"] and value["dtype_matches"] and value["values_match"]
            for value in comparisons.values()
        )
    )
    return {
        "source_sha256": source_sha256,
        "asset_sha256": asset_sha256,
        "comparisons": comparisons,
        "passed": passed,
    }


def _write_fixture(path: Path, chains: dict[str, int], *, zinc: bool = False) -> None:
    lines = []
    serial = 1
    atoms = {
        "N": (-1.20, 0.70, -0.10, "N"),
        "CA": (0.00, 0.00, 0.00, "C"),
        "C": (1.30, 0.50, 0.20, "C"),
        "O": (1.80, 1.55, 0.10, "O"),
        "CB": (-0.20, -1.25, 0.85, "C"),
    }
    for chain_index, (chain, length) in enumerate(chains.items()):
        for residue in range(1, length + 1):
            base = np.array((3.8 * (residue - 1), 12.0 * chain_index, 0.3 * chain_index))
            for atom_name, (*offset, element) in atoms.items():
                x, y, z = base + np.asarray(offset)
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} ALA {chain}{residue:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}  "
                )
                serial += 1
        lines.append("TER")
    if zinc:
        lines.append(f"HETATM{serial:5d} ZN    ZN Z   1       3.800   4.000   1.000  1.00 20.00          ZN2+")
    lines.extend(("END", ""))
    path.write_text("\n".join(lines))


def _fixture(case: str, directory: Path) -> Path | None:
    if case in {"base", "symmetry"}:
        return None
    path = directory / f"{case}.pdb"
    if case == "complex":
        _write_fixture(path, {"A": 3})
    elif case == "fold":
        _write_fixture(path, {"A": 6})
    elif case == "inpaint":
        _write_fixture(path, {"A": 3})
    elif case == "inpaint_fold":
        _write_fixture(path, {"A": 6})
    elif case == "active_site":
        _write_fixture(path, {"A": 3}, zinc=True)
    elif case == "partial":
        _write_fixture(path, {"A": 4})
    else:
        raise KeyError(case)
    return path


def _configure_reference(config: object, case: str, checkpoint: Path, fixture: Path | None) -> None:
    values = CASES[case]
    config.inference.ckpt_override_path = str(checkpoint)
    config.inference.input_pdb = None if fixture is None else str(fixture)
    config.inference.final_step = values["final_step"]
    config.inference.align_motif = True
    config.inference.symmetry = None
    config.inference.recenter = True
    config.inference.radius = 10.0
    config.inference.deterministic = False
    config.contigmap.contigs = None
    config.contigmap.inpaint_seq = None
    config.contigmap.inpaint_str = None
    config.contigmap.inpaint_str_helix = None
    config.contigmap.inpaint_str_strand = None
    config.contigmap.inpaint_str_loop = None
    config.diffuser.partial_T = values.get("partial_steps")
    config.denoiser.noise_scale_ca = 0.0
    config.denoiser.final_noise_scale_ca = 0.0
    config.denoiser.ca_noise_schedule_type = "constant"
    config.denoiser.noise_scale_frame = 0.0
    config.denoiser.final_noise_scale_frame = 0.0
    config.denoiser.frame_noise_schedule_type = "constant"
    config.ppi.hotspot_res = None
    config.potentials.guiding_potentials = None
    config.potentials.guide_scale = 10.0
    config.potentials.guide_decay = "constant"
    config.potentials.substrate = None
    config.scaffoldguided.scaffoldguided = False
    config.scaffoldguided.scaffold_dir = None

    if case == "base":
        config.contigmap.contigs = ["4-4"]
    elif case == "complex":
        config.contigmap.contigs = ["A1-3/0 4-4"]
        config.ppi.hotspot_res = ["A2"]
    elif case == "fold":
        config.contigmap.contigs = ["A1-6"]
        config.contigmap.inpaint_str = ["A1-6"]
        config.contigmap.inpaint_str_helix = ["A1-6"]
        config.scaffoldguided.scaffoldguided = True
    elif case == "inpaint":
        config.contigmap.contigs = ["2-2/A1-3/2-2"]
        config.contigmap.inpaint_seq = ["A2-2"]
    elif case == "inpaint_fold":
        config.contigmap.contigs = ["A1-6"]
        config.contigmap.inpaint_seq = ["A1-6"]
        config.contigmap.inpaint_str = ["A1-6"]
        config.contigmap.inpaint_str_helix = ["A1-6"]
        config.scaffoldguided.scaffoldguided = True
    elif case == "active_site":
        config.contigmap.contigs = ["2-2/A2-2/2-2"]
        config.potentials.guiding_potentials = ["type:substrate_contacts,s:1,r_0:8,rep_r_0:5,rep_s:2,rep_r_min:1"]
        config.potentials.guide_scale = 1.0
        config.potentials.substrate = "ZN"
    elif case == "symmetry":
        config.contigmap.contigs = ["4-4"]
        config.inference.symmetry = "C2"
    elif case == "partial":
        config.contigmap.contigs = ["A1-4"]
        config.contigmap.inpaint_seq = ["A1-4"]
        config.contigmap.inpaint_str = ["A1-4"]
    else:
        raise KeyError(case)


def _reference(case: str, checkpoint: Path, reference_root: Path, seed: int, fixture: Path | None) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(reference_root / "env" / "SE3Transformer"))
    sys.path.insert(0, str(reference_root))
    from omegaconf import OmegaConf
    from rfdiffusion.inference import utils as reference_utils

    config = OmegaConf.load(reference_root / "config" / "inference" / "base.yaml")
    config.inference.schedule_directory_path = str(reference_root / "schedules")
    _configure_reference(config, case, checkpoint, fixture)
    sampler = reference_utils.sampler_selector(config)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x_t, sequence = sampler.sample_init()
    initial_x_t = x_t.detach().cpu().clone()
    px0_values = []
    xt_values = []
    plddt_values = []
    self_conditioning_values = []
    timesteps = []
    final_step = int(CASES[case]["final_step"])
    for timestep in range(int(sampler.t_step_input), final_step - 1, -1):
        px0, x_t, sequence, plddt = sampler.sample_step(
            t=timestep,
            x_t=x_t,
            seq_init=sequence,
            final_step=final_step,
        )
        timesteps.append(timestep)
        px0_values.append(px0.detach().cpu())
        xt_values.append(x_t.detach().cpu())
        plddt_values.append(plddt[0].detach().cpu())
        self_conditioning_values.append(sampler.prev_pred.detach().cpu())
    return {
        "case": case,
        "checkpoint": checkpoint.name,
        "seed": seed,
        "initial_x_t": initial_x_t,
        "timesteps": torch.tensor(timesteps, dtype=torch.int64),
        "px0": torch.stack(px0_values),
        "x_t_minus_1": torch.stack(xt_values),
        "plddt": torch.stack(plddt_values),
        "self_conditioning": torch.stack(self_conditioning_values),
        "sequence": sequence.argmax(dim=-1).detach().cpu(),
        "fixed_structure_mask": sampler.mask_str.squeeze().detach().cpu(),
        "fixed_sequence_mask": sampler.mask_seq.squeeze().detach().cpu(),
    }


def _reference_sites(structure: object, instance_id: str, start: int, stop: int) -> tuple:
    return tuple(
        site
        for site, present in zip(structure.sites, structure.coordinate_mask, strict=True)
        if present
        and site.instance_id == instance_id
        and site.component_index is not None
        and start <= site.component_index < stop
    )


def _native_input(case: str, fixture: Path | None, device: str) -> tuple[object, dict]:
    from ofoldx.io.structure import read_structure
    from ofoldx.models.rfdiffusion import RFdiffusionProcessor
    from ofoldx.models.rfdiffusion.model import (
        ExistingSpan,
        GeneratedSegment,
        Hotspot,
        PointGroupSymmetry,
        ReferenceCoordinates,
        RFdiffusionPlacements,
        SecondaryStructure,
        SubstrateInteraction,
        SymbolicResidueSite,
    )

    processor = RFdiffusionProcessor(device=device)
    if case == "active_site":
        from ofoldx.data import ChemicalAtom, ChemicalComponent, PolymerSequence

        zinc = ChemicalComponent(
            "ZN",
            (ChemicalAtom("ZN", "Zn", formal_charge=2, implicit_hydrogens_allowed=False),),
            bonds=frozenset(),
        )
        structure = read_structure(fixture, components={"ZN": zinc})
    else:
        structure = None if fixture is None else read_structure(fixture)
    processor_kwargs = {}
    conditions = []

    if case == "base":
        conditions.append(RFdiffusionPlacements({"chain_1": {"design": GeneratedSegment("protein", 4, 4)}}, (4,)))
    elif case == "complex":
        conditions.extend(
            (
                RFdiffusionPlacements(
                    {
                        "chain_1": {"binder": GeneratedSegment("protein", 4, 4)},
                        "chain_2": {"A:0": ExistingSpan("A", 0, 3)},
                    },
                    (4, 3),
                ),
                ReferenceCoordinates(_reference_sites(structure, "A", 0, 3), 0.0),
                # Frozen §3.7.4 single-sided distance-free Hotspot (owner verdict D3): the existing
                # target residue A:1 (upstream ``ppi.hotspot_res=["A2"]``) realized as hotspot_mask.
                Hotspot((("A", 1),)),
            )
        )
    elif case == "fold":
        residues = tuple(("A", index) for index in range(6))
        conditions.extend(
            (
                RFdiffusionPlacements({"chain_1": {"A:0": ExistingSpan("A", 0, 6)}}, (6,)),
                SecondaryStructure(residues, "helix"),
            )
        )
    elif case == "inpaint":
        conditions.extend(
            (
                RFdiffusionPlacements(
                    {
                        "chain_1": {
                            "left": GeneratedSegment("protein", 2, 2),
                            "A:0": ExistingSpan("A", 0, 3),
                            "right": GeneratedSegment("protein", 2, 2),
                        }
                    },
                    (7,),
                ),
                ReferenceCoordinates(_reference_sites(structure, "A", 0, 3), 0.0),
            )
        )
        processor_kwargs["inpaint_sequence"] = (("A", 1),)
    elif case == "inpaint_fold":
        residues = tuple(("A", index) for index in range(6))
        conditions.extend(
            (
                RFdiffusionPlacements({"chain_1": {"A:0": ExistingSpan("A", 0, 6)}}, (6,)),
                SecondaryStructure(residues, "helix"),
            )
        )
        processor_kwargs["inpaint_sequence"] = residues
    elif case == "active_site":
        substrate = tuple(
            site
            for site, present in zip(structure.sites, structure.coordinate_mask, strict=True)
            if present
            and type(structure.system.entities[structure.system.instances[site.instance_id]]) is not PolymerSequence
        )
        conditions.extend(
            (
                RFdiffusionPlacements(
                    {
                        "chain_1": {
                            "left": GeneratedSegment("protein", 2, 2),
                            "A:1": ExistingSpan("A", 1, 2),
                            "right": GeneratedSegment("protein", 2, 2),
                        }
                    },
                    (5,),
                ),
                ReferenceCoordinates(_reference_sites(structure, "A", 1, 2), 0.0),
                SubstrateInteraction(
                    (SymbolicResidueSite("left", 0), SymbolicResidueSite("right", 1)),
                    substrate,
                    center_angstrom=2.0,  # released d_0
                    scale_angstrom=8.0,  # released r_0
                    attraction_strength=1.0,  # released s (matches _configure_reference's "s:1")
                    repulsion_slope=2.0,  # released rep_s
                    repulsion_cutoff_angstrom=5.0,  # released rep_r_0
                ),
            )
        )
    elif case == "symmetry":
        conditions.extend(
            (
                RFdiffusionPlacements(
                    {
                        "chain_1": {
                            "unit_1": GeneratedSegment("protein", 2, 2),
                            "unit_2": GeneratedSegment("protein", 2, 2),
                        }
                    },
                    (4,),
                ),
                PointGroupSymmetry((("unit_1",), ("unit_2",)), "cyclic", 2, 1e-4),
            )
        )
    elif case == "partial":
        conditions.append(RFdiffusionPlacements({"chain_1": {"A:0": ExistingSpan("A", 0, 4)}}, (4,)))
        processor_kwargs["inpaint_sequence"] = tuple(("A", index) for index in range(4))
    else:
        raise KeyError(case)
    return processor(structure, conditions=tuple(conditions), **processor_kwargs), {
        "processor": processor,
        "structure": structure,
    }


def _rigid_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"coordinate shapes differ: {actual.shape} != {expected.shape}")
    rmsd_values = []
    maximum_values = []
    for actual_step, expected_step in zip(actual, expected, strict=True):
        moving = actual_step.reshape(-1, 3)
        target = expected_step.reshape(-1, 3)
        finite = np.isfinite(moving).all(axis=-1) & np.isfinite(target).all(axis=-1)
        moving = moving[finite]
        target = target[finite]
        moving_center = moving.mean(axis=0)
        target_center = target.mean(axis=0)
        left, _, right = np.linalg.svd((moving - moving_center).T @ (target - target_center))
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(left @ right))
        rotation = left @ correction @ right
        aligned = (moving - moving_center) @ rotation + target_center
        distances = np.linalg.norm(aligned - target, axis=-1)
        rmsd_values.append(float(np.sqrt(np.mean(distances**2))))
        maximum_values.append(float(distances.max()))
    return max(rmsd_values), max(maximum_values)


def _native(
    case: str,
    checkpoint: Path,
    capture_path: Path,
    seed: int,
    fixture: Path | None,
    reference_root: Path,
    device: str,
    coordinate_tolerance: float,
    plddt_tolerance: float,
) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    from ofoldx.data import Structure
    from ofoldx.models.rfdiffusion.convert import convert_rfdiffusion_checkpoint
    from ofoldx.models.rfdiffusion.model import (
        PointGroupSymmetry,
        RFdiffusionGuidance,
        RFdiffusionSampler,
        RFdiffusionSymmetry,
        one_hot_sequence,
    )

    expected = torch.load(capture_path, map_location="cpu", weights_only=True)
    if expected["case"] != case or expected["checkpoint"] != checkpoint.name or expected["seed"] != seed:
        raise ValueError("reference capture case, checkpoint, or seed does not match this invocation")

    schedule = _schedule_parity(reference_root)
    batch, _context = _native_input(case, fixture, device)
    model = convert_rfdiffusion_checkpoint(checkpoint).eval().to(device)
    values = CASES[case]

    sampler = RFdiffusionSampler(
        model.config,
        network=model.network,
        allatom=model.allatom,
        final_step=int(values["final_step"]),
        partial_steps=values.get("partial_steps"),
        noise_scale_ca=0.0,
        final_noise_scale_ca=0.0,
        ca_noise_schedule="constant",
        noise_scale_frame=0.0,
        final_noise_scale_frame=0.0,
        frame_noise_schedule="constant",
        align_motif=True,
        symmetry_recenter=True,
        symmetry_radius_angstrom=10.0,
        guidance_scale=1.0 if case == "active_site" else 10.0,
        guidance_decay="constant",
    )
    diffuser = sampler._make_diffuser(50)
    guidance = RFdiffusionGuidance(batch.layout)
    denoiser = sampler._make_denoiser(
        len(batch.layout.output_residues),
        diffuser=diffuser,
        guidance=guidance if guidance else None,
    )
    mask_structure = batch.fixed_structure_mask.to(device)
    diffusion_mask = mask_structure.squeeze(0)
    sequence = one_hot_sequence(expected["sequence"].to(device))
    symmetry_conditions = tuple(
        condition for condition in batch.layout.conditions if type(condition) is PointGroupSymmetry
    )
    symmetry = None if not symmetry_conditions else RFdiffusionSymmetry(symmetry_conditions[0], batch.layout)
    residue_index = batch.residue_index.to(device)
    if symmetry is not None:
        residue_index = symmetry.chain_broken_residue_index(residue_index)
    secondary_structure = batch.secondary_structure[0].to(device)
    block_adjacency = batch.block_adjacency[0].to(device)
    hotspot_mask = batch.hotspot_mask[0].to(device)
    cyclic_mask = batch.cyclic_mask[0].to(device)

    replay_px0 = []
    replay_xt = []
    replay_plddt = []
    replay_self_conditioning = []
    final_step = int(values["final_step"])
    for index, timestep_tensor in enumerate(expected["timesteps"]):
        timestep = int(timestep_tensor)
        noisy_coords = (expected["initial_x_t"] if index == 0 else expected["x_t_minus_1"][index - 1]).to(device)
        previous_prediction = None if index == 0 else expected["self_conditioning"][index - 1].squeeze(0).to(device)
        (
            msa_masked,
            msa_full,
            sequence_input,
            noisy_coords_input,
            template_1d,
            template_2d,
            template_coords,
            alpha_t,
        ) = sampler._preprocess(
            sequence,
            noisy_coords,
            timestep,
            mask_structure=mask_structure,
            diffuser=diffuser,
            previous_prediction=previous_prediction,
            secondary_structure=secondary_structure,
            block_adjacency=block_adjacency,
            hotspot_mask=hotspot_mask,
        )
        with torch.no_grad():
            output = model.network(
                msa_masked,
                msa_full,
                sequence_input,
                noisy_coords_input,
                residue_index,
                template_1d=template_1d,
                template_2d=template_2d,
                template_coords=template_coords,
                alpha_t=alpha_t,
                msa_prev=None,
                pair_prev=None,
                state_prev=None,
                motif_mask=diffusion_mask,
                cyclic_reses=cyclic_mask,
            )
            predicted_backbone, alpha, plddt = output[2], output[4], output[6]
            sequence_tokens = sequence_input.argmax(dim=-1)
            _frames, predicted_full = sampler.allatom(sequence_tokens, predicted_backbone, alpha)
            predicted_coords = predicted_full.squeeze(0)[:, :14]
        if symmetry is not None:
            symmetric_coords, _symmetric_sequence = symmetry.apply(
                predicted_coords,
                sequence,
                recenter=True,
                radius_angstrom=10.0,
            )
            current_self_conditioning = symmetric_coords[:, :3]
            with torch.no_grad():
                _frames, predicted_full = sampler.allatom(
                    sequence_tokens,
                    current_self_conditioning[None],
                    alpha,
                )
                predicted_coords = predicted_full.squeeze(0)[:, :14]
        else:
            current_self_conditioning = predicted_backbone.squeeze(0)
        if timestep > final_step:
            next_coords, predicted_coords = denoiser.get_next_pose(
                noisy_coords=noisy_coords,
                pred_coords=predicted_coords,
                t=timestep,
                diffusion_mask=diffusion_mask,
                align_motif=True,
                include_motif_sidechains=model.config.preprocess.motif_sidechain_input,
            )
        else:
            next_coords = predicted_coords
        if symmetry is not None:
            next_coords, sequence = symmetry.apply(
                next_coords,
                sequence,
                recenter=True,
                radius_angstrom=10.0,
            )
        replay_px0.append(predicted_coords.detach().cpu())
        replay_xt.append(next_coords.detach().cpu())
        replay_plddt.append(plddt.squeeze(0).detach().cpu())
        replay_self_conditioning.append(current_self_conditioning.detach().cpu()[None])

    replay_px0_array = torch.stack(replay_px0).numpy()
    replay_xt_array = torch.stack(replay_xt).numpy()
    replay_plddt_array = torch.stack(replay_plddt).numpy()
    replay_self_conditioning_array = torch.stack(replay_self_conditioning).numpy()
    plddt_error = float(np.max(np.abs(replay_plddt_array - expected["plddt"].numpy())))
    px0_rmsd, px0_maximum = _rigid_errors(
        replay_px0_array[..., :4, :],
        expected["px0"].numpy()[..., :4, :],
    )
    xt_rmsd, xt_maximum = _rigid_errors(
        replay_xt_array[..., :4, :],
        expected["x_t_minus_1"].numpy()[..., :4, :],
    )
    self_conditioning_rmsd, self_conditioning_maximum = _rigid_errors(
        replay_self_conditioning_array,
        expected["self_conditioning"].numpy(),
    )
    step_errors = []
    for index, timestep in enumerate(expected["timesteps"].tolist()):
        step_px0_rmsd, _step_px0_maximum = _rigid_errors(
            replay_px0_array[index : index + 1, ..., :4, :],
            expected["px0"].numpy()[index : index + 1, ..., :4, :],
        )
        step_xt_rmsd, _step_xt_maximum = _rigid_errors(
            replay_xt_array[index : index + 1, ..., :4, :],
            expected["x_t_minus_1"].numpy()[index : index + 1, ..., :4, :],
        )
        step_self_conditioning_rmsd, _step_self_conditioning_maximum = _rigid_errors(
            replay_self_conditioning_array[index : index + 1],
            expected["self_conditioning"].numpy()[index : index + 1],
        )
        step_errors.append(
            {
                "timestep": timestep,
                "plddt_max_absolute_error": float(
                    np.max(np.abs(replay_plddt_array[index] - expected["plddt"].numpy()[index]))
                ),
                "px0_aligned_rmsd": step_px0_rmsd,
                "x_t_minus_1_aligned_rmsd": step_xt_rmsd,
                "self_conditioning_aligned_rmsd": step_self_conditioning_rmsd,
            }
        )

    public_results = []
    for _ in range(2):
        generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
        public_results.append(
            model.design(
                batch,
                num_sampling_steps=50,
                final_step=final_step,
                partial_steps=values.get("partial_steps"),
                align_motif=True,
                symmetry_recenter=True,
                symmetry_radius_angstrom=10.0,
                guidance_scale=1.0 if case == "active_site" else 10.0,
                guidance_decay="constant",
                generator=generator,
            )
        )
    first_candidate, second_candidate = (result.candidates[0] for result in public_results)
    if type(first_candidate.structure) is not Structure or type(second_candidate.structure) is not Structure:
        raise TypeError("RFdiffusion design must return a Structure")
    public_reproducible = np.array_equal(
        first_candidate.structure.coordinates,
        second_candidate.structure.coordinates,
        equal_nan=True,
    )
    first_metrics = first_candidate.metrics
    second_metrics = second_candidate.metrics
    public_diagnostics_reproducible = first_metrics.keys() == second_metrics.keys() and all(
        np.array_equal(first_metrics[name].values, second_metrics[name].values, equal_nan=True)
        for name in first_metrics
    )
    fixed_masks_match = np.array_equal(
        batch.fixed_structure_mask[0].detach().cpu().numpy(),
        expected["fixed_structure_mask"].numpy(),
    ) and np.array_equal(
        batch.fixed_sequence_mask[0].detach().cpu().numpy(),
        expected["fixed_sequence_mask"].numpy(),
    )
    designed_instances = tuple(dict.fromkeys(batch.layout.output_instance_ids))
    unknown_count = 0
    for instance_id in designed_instances:
        entity = first_candidate.structure.system.entities[first_candidate.structure.system.instances[instance_id]]
        unknown_count += sum(component.component_id == "UNK" for component in entity.components)
    expected_unknown_count = int((expected["sequence"] == 21).sum())

    passed = (
        fixed_masks_match
        and unknown_count == expected_unknown_count
        and schedule["passed"]
        and public_reproducible
        and public_diagnostics_reproducible
        and plddt_error <= plddt_tolerance
        and px0_rmsd <= coordinate_tolerance
        and xt_rmsd <= coordinate_tolerance
        and self_conditioning_rmsd <= coordinate_tolerance
    )
    return {
        "case": case,
        "checkpoint": checkpoint.name,
        "seed": seed,
        "schedule": schedule,
        "fixed_masks_match": fixed_masks_match,
        "unknown_residue_counts": [unknown_count, expected_unknown_count],
        "algorithm_replay": {
            "timesteps": expected["timesteps"].tolist(),
            "plddt_max_absolute_error": plddt_error,
            "px0_aligned_rmsd": px0_rmsd,
            "px0_aligned_maximum": px0_maximum,
            "x_t_minus_1_aligned_rmsd": xt_rmsd,
            "x_t_minus_1_aligned_maximum": xt_maximum,
            "self_conditioning_aligned_rmsd": self_conditioning_rmsd,
            "self_conditioning_aligned_maximum": self_conditioning_maximum,
            "steps": step_errors,
        },
        "public_generator_smoke": {
            "structure_reproducible": public_reproducible,
            "diagnostics_reproducible": public_diagnostics_reproducible,
        },
        "coordinate_tolerance": coordinate_tolerance,
        "plddt_tolerance": plddt_tolerance,
        "passed": passed,
    }


def _checkpoint(case: str, checkpoint_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return checkpoint_dir / str(CASES[case]["checkpoint"])


def _compile_only() -> None:
    from ofoldx.models.rfdiffusion import RFdiffusionConfig, RFdiffusionModel

    batch, _ = _native_input("base", None, "cpu")
    residue = torch.arange(batch.noisy_coords.shape[1], dtype=batch.noisy_coords.dtype)
    calpha = torch.stack((3.8 * residue, 0.2 * residue.remainder(2), 0.1 * residue), dim=-1)
    batch.noisy_coords[:, :, 0] = calpha + torch.tensor((-1.2, 0.7, -0.1))
    batch.noisy_coords[:, :, 1] = calpha
    batch.noisy_coords[:, :, 2] = calpha + torch.tensor((1.3, 0.5, 0.2))

    torch.manual_seed(0)
    se3 = {
        "num_layers": 1,
        "num_channels": 8,
        "num_degrees": 2,
        "num_attention_heads": 2,
        "div": 2,
        "l0_in_features": 8,
        "l0_out_features": 8,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 8,
    }
    config = RFdiffusionConfig(
        {
            "network": {
                "num_extra_blocks": 0,
                "num_main_blocks": 0,
                "num_refinement_blocks": 1,
                "msa_size": 16,
                "full_msa_size": 8,
                "pair_size": 16,
                "template_size": 8,
                "num_msa_attention_heads": 2,
                "num_pair_attention_heads": 2,
                "num_template_attention_heads": 2,
                "hidden_size": 8,
                "template_hidden_size": 8,
                "dropout": 0.0,
                "se3_param_full": se3,
                "se3_param_topk": se3,
            },
            "preprocess": {"structure_self_conditioning": True},
        }
    )
    model = RFdiffusionModel(config).eval()
    with torch.inference_mode():
        eager = model(batch)
        compiled = torch.compile(model, fullgraph=True)(batch)

    checks = {}
    for name in (
        "single",
        "pair",
        "backbone_coords",
        "state",
        "torsion_angles",
        "sequence_logits",
        "plddt",
        "plddt_logits",
        "resolved_logits",
        "backbone_trajectory",
        "torsion_trajectory",
    ):
        actual = getattr(compiled, name)
        expected = getattr(eager, name)
        torch.testing.assert_close(actual, expected)
        checks[name] = list(actual.shape)
    for index, (actual, expected) in enumerate(zip(compiled.distance_logits, eager.distance_logits, strict=True)):
        torch.testing.assert_close(actual, expected)
        checks[f"distance_logits.{index}"] = list(actual.shape)

    print(json.dumps({"mode": "compile-only", "outputs": checks, "passed": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("reference", "native"))
    parser.add_argument("case", nargs="?", choices=(*CASES, "matrix"))
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile a tiny model with default Inductor/fullgraph and compare every output leaf",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.home() / "weights" / "rfdiffusion-weights",
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path.home() / "reference" / "RFdiffusion",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coordinate-tolerance", type=float, default=0.05)
    parser.add_argument("--plddt-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.compile_only:
        if args.mode is not None or args.case is not None:
            parser.error("--compile-only does not accept mode or case")
        _compile_only()
        return
    if args.mode is None or args.case is None:
        parser.error("mode and case are required unless --compile-only is used")
    if args.capture is None:
        parser.error("--capture is required for reference/native parity")
    if args.case == "matrix" and args.checkpoint is not None:
        parser.error("--checkpoint is valid only for one named case")

    cases = tuple(CASES) if args.case == "matrix" else (args.case,)
    if args.case == "matrix":
        args.capture.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in cases:
        checkpoint = _checkpoint(case, args.checkpoint_dir, args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing {case} checkpoint: {checkpoint}")
        capture_path = args.capture / f"{case}.pt" if args.case == "matrix" else args.capture
        with tempfile.TemporaryDirectory(prefix=f"ofoldx-rfdiffusion-{case}-") as directory:
            fixture = _fixture(case, Path(directory))
            if args.mode == "reference":
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _reference(case, checkpoint, args.reference_root, args.seed, fixture),
                    capture_path,
                )
                reports.append({"case": case, "capture": str(capture_path)})
            else:
                reports.append(
                    _native(
                        case,
                        checkpoint,
                        capture_path,
                        args.seed,
                        fixture,
                        args.reference_root,
                        args.device,
                        args.coordinate_tolerance,
                        args.plddt_tolerance,
                    )
                )
    if len(reports) == 1:
        output = reports[0]
    elif args.mode == "reference":
        output = {"cases": reports}
    else:
        output = {"cases": reports, "passed": all(report["passed"] for report in reports)}
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.mode == "native" and not all(report["passed"] for report in reports):
        raise SystemExit("RFdiffusion public design parity failed")


if __name__ == "__main__":
    main()
