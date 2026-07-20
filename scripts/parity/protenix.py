#!/usr/bin/env python
r"""Maintained Protenix v0.5 parity and public-surface acceptance harness.

The harness reads the official checkout and released checkpoints from caller-owned
local paths.  It never resolves Hub assets, downloads weights, writes reports, or
weakens its numerical tolerances from the command line.
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.paths import reference_root, weight_path  # noqa: E402

DEFAULT_CHECKPOINT = weight_path("protenix_base_default_v0.5.0.pt")
CONSTRAINT_CHECKPOINT = weight_path("protenix_base_constraint_v0.5.0.pt")
REFERENCE_ROOT = reference_root("Protenix")
SEED = 7

# Chosen before executing the maintained matrix.  Both comparisons are FP32.
COMPILE_ATOL = 1e-4
COMPILE_RTOL = 1e-4
FORWARD_ATOL = 1e-4
FORWARD_RTOL = 1e-4


class Blocked(RuntimeError):
    """A required production surface is not implemented yet."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("semantic", "load", "compile", "forward", "processor", "released", "repository", "all"),
    )
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--default-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--constraint-checkpoint", type=Path, default=CONSTRAINT_CHECKPOINT)
    parser.add_argument("--variant", choices=("all", "default", "constraint"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    stages = (
        ("semantic", "load", "compile", "forward", "processor", "released", "repository")
        if args.mode == "all"
        else (args.mode,)
    )
    for stage in stages:
        print(f"RUN protenix {stage}", flush=True)
        if stage == "semantic":
            semantic_only()
        elif stage == "load":
            strict_load(args.default_checkpoint, args.constraint_checkpoint)
        elif stage == "compile":
            compile_forward(args.device)
        elif stage == "forward":
            forward_parity(
                reference=args.reference_root,
                default_checkpoint=args.default_checkpoint,
                constraint_checkpoint=args.constraint_checkpoint,
                variant=args.variant,
                device_name=args.device,
                seed=args.seed,
            )
        elif stage == "processor":
            processor_packing(reference=args.reference_root)
        elif stage == "released":
            released_surface(
                default_checkpoint=args.default_checkpoint,
                constraint_checkpoint=args.constraint_checkpoint,
                device_name=args.device,
                seed=args.seed,
            )
        else:
            repository_surface(
                checkpoint=args.default_checkpoint,
                device_name=args.device,
                seed=args.seed,
            )
        print(f"PASS protenix {stage}", flush=True)
    return 0


def semantic_only() -> None:
    """Exercise the canonical mixed-system facts without a model or checkpoint."""

    import numpy as np

    from ofoldx.data import (
        MSA,
        AtomSite,
        ChemicalAtom,
        ChemicalBond,
        ChemicalComponent,
        MolecularSystem,
        Molecule,
        PolymerSequence,
        SystemBond,
    )
    from ofoldx.models.protenix import (
        ProtenixAtomContact,
        ProtenixPocket,
        ProtenixTokenContact,
    )

    protein = PolymerSequence.from_one_letter("AC", polymer_type="protein")
    rna = PolymerSequence.from_one_letter("AG", polymer_type="rna")
    dna = PolymerSequence.from_one_letter("CT", polymer_type="dna")
    ligand = Molecule(
        components=(
            ChemicalComponent(
                component_id="LIG",
                atoms=(
                    ChemicalAtom("C1", "C", formal_charge=0, implicit_hydrogens=0),
                    ChemicalAtom("O1", "O", formal_charge=0, implicit_hydrogens=0),
                ),
                bonds=frozenset({ChemicalBond("C1", "O1", order=2.0)}),
                reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
            ),
        ),
    )
    protein_site = AtomSite("P", 1, "SG")
    ligand_site = AtomSite("L", 0, "C1")
    ligand_oxygen = AtomSite("L", 0, "O1")
    covalent_bond = SystemBond(protein_site, ligand_site, origin="user")
    system = MolecularSystem(
        entities={"protein": protein, "rna": rna, "dna": dna, "ligand": ligand},
        instances={"P": "protein", "R": "rna", "D": "dna", "L": "ligand"},
        bonds=frozenset({covalent_bond}),
    )
    msa = MSA(
        entity_id="protein",
        unpaired_sequences=("AC", "AG"),
        unpaired_deletion_counts=np.array([[0, 0], [0, 2]], dtype=np.uint32),
        paired_sequences=("AC",),
        paired_deletion_counts=np.zeros((1, 2), dtype=np.uint32),
    )
    conditions = (
        ProtenixTokenContact(
            endpoint_a=("P", 0),
            endpoint_b=ligand_oxygen,
            max_distance=8.0,
        ),
        ProtenixAtomContact(
            atom_a=protein_site,
            atom_b=ligand_site,
            max_distance=2.2,
        ),
        ProtenixPocket(
            binder="L",
            contact_residues=(("P", 0),),
            max_distance=6.0,
        ),
    )
    # Every condition endpoint must resolve exactly against the system (the old
    # validate_conditions entry point folded into per-endpoint resolution here).
    system.component(("P", 0))
    system.atom(ligand_oxygen)
    system.atom(protein_site)
    system.atom(ligand_site)

    entity_types = {entity.polymer_type for entity in system.entities.values() if type(entity) is PolymerSequence}
    if entity_types != {"protein", "rna", "dna"}:
        raise AssertionError(f"mixed polymer types differ: {entity_types!r}")
    if system.bonds != frozenset({covalent_bond}):
        raise AssertionError("the canonical protein-ligand covalent bond was not retained")
    if system.atom(protein_site).element != "S" or system.atom(ligand_site).element != "C":
        raise AssertionError("covalent endpoints lost their exact atom identities")
    if msa.unpaired_sequences[0] != "AC" or len(msa.unpaired_sequences[0]) != len(protein.components):
        raise AssertionError("the canonical MSA query no longer maps exactly to its protein entity")
    if tuple(type(condition).__name__ for condition in conditions) != (
        "ProtenixTokenContact",
        "ProtenixAtomContact",
        "ProtenixPocket",
    ):
        raise AssertionError("canonical conditions changed order or identity")
    print(
        "semantic canonical mixed protein/RNA/DNA/ligand + covalent bond + MSA + conditions",
        flush=True,
    )


def strict_load(default_checkpoint: Path, constraint_checkpoint: Path) -> None:
    """Strict-convert both released v0.5 graphs, one at a time."""

    import torch

    from ofoldx.models.protenix.convert import convert_protenix_checkpoint

    for name, checkpoint in (
        ("default", default_checkpoint),
        ("constraint", constraint_checkpoint),
    ):
        checkpoint = _local_file(checkpoint, f"{name} checkpoint")
        model = convert_protenix_checkpoint(checkpoint)
        state = model.state_dict()
        if not state or any(tensor.is_meta for tensor in state.values()):
            raise AssertionError(f"{name} checkpoint did not materialize a complete native state")
        print(
            f"strict-load {name}: {len(state)} tensors, "
            f"{sum(tensor.numel() for tensor in state.values()):,} values",
            flush=True,
        )
        del state, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compile_forward(device_name: str) -> None:
    """Compile a tiny complete explicit forward and compare every output leaf."""

    import torch

    from ofoldx.models.protenix import ProtenixConfig, ProtenixModel, ProtenixProcessor

    device = _device(device_name)
    config = ProtenixConfig()
    config.single_size = 16
    config.pair_size = 8
    config.recycling_steps = 1
    config.trunk.num_pairformer_blocks = 0
    config.trunk.pairformer_dropout = 0.0
    config.confidence.num_blocks = 0
    config.confidence.num_pair_attention_heads = 2
    config.confidence.triangle_multiplication_hidden_size = 8
    config.confidence.triangle_attention_hidden_size = 4
    config.confidence.num_triangle_attention_heads = 2
    config.denoiser.num_token_encoder_blocks = 0
    config.denoiser.num_token_encoder_attention_heads = 2
    config.denoiser.atom_hidden_size = 8
    config.denoiser.atom_pair_hidden_size = 4
    config.denoiser.token_hidden_size = 16
    # The shared AtomSelfAttentionStack projects its pair bias per block, so a zero-block
    # atom encoder/decoder is unsupported (empty torch.cat), unlike the Pairformer, token,
    # and confidence stacks which return their input unchanged at zero blocks. One block is
    # the minimal count that both exercises the atom-attention path under compile and stays tiny.
    config.denoiser.num_atom_encoder_blocks = 1
    config.denoiser.num_atom_encoder_attention_heads = 2
    config.denoiser.num_atom_decoder_blocks = 1
    config.denoiser.num_atom_decoder_attention_heads = 2
    config.denoiser.num_query_atoms = 4
    config.denoiser.num_key_atoms = 8

    torch.manual_seed(SEED)
    model = ProtenixModel(config).set_backend("torch").to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batch = ProtenixProcessor(device=device)("AC")
    noisy_coordinates = torch.randn(
        1,
        1,
        batch.atom_features.shape[-2],
        3,
        generator=generator,
    ).to(device)
    noise_level = torch.ones(1, 1, device=device)

    with torch.inference_mode():
        eager = model(batch, noisy_coordinates=noisy_coordinates, noise_level=noise_level, recycling_steps=1)
        compiled_model = torch.compile(model, fullgraph=True)
        compiled = compiled_model(
            batch,
            noisy_coordinates=noisy_coordinates,
            noise_level=noise_level,
            recycling_steps=1,
        )
    _compare_output_objects(
        eager,
        compiled,
        atol=COMPILE_ATOL,
        rtol=COMPILE_RTOL,
        label="compile",
    )


def forward_parity(
    *,
    reference: Path,
    default_checkpoint: Path,
    constraint_checkpoint: Path,
    variant: str,
    device_name: str,
    seed: int,
) -> None:
    """Compare the current caller-supplied-noise forward with Protenix v0.5."""

    import torch

    device = _device(device_name)
    reference = reference.expanduser().resolve()
    if not (reference / "protenix/model/protenix.py").is_file():
        raise FileNotFoundError(f"Protenix checkout is absent or incomplete: {reference}")
    revision, dirty = _upstream_git_state(reference)
    print(
        f"SOURCE protenix checkout={reference} revision={revision} dirty={dirty}",
        flush=True,
    )
    variants = ("default", "constraint") if variant == "all" else (variant,)
    checkpoints = {
        "default": _local_file(default_checkpoint, "default checkpoint"),
        "constraint": _local_file(constraint_checkpoint, "constraint checkpoint"),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    from ofoldx.models.protenix import (
        ProtenixConstraintProcessor,
        ProtenixConstraintProcessorConfig,
        ProtenixProcessor,
        ProtenixProcessorConfig,
    )

    system, msa, conditions = _canonical_case()
    for current in variants:
        # The constraint graph consumes token contacts only through the distinct constraint
        # Processor signature (which declares the ``token_contacts`` keyword); the default
        # Processor omits that keyword and carries all-zero constraint channels.
        constrained = current == "constraint"
        if constrained:
            processor = ProtenixConstraintProcessor(ProtenixConstraintProcessorConfig())
            batch = processor(system, msa=msa, token_contacts=conditions)
        else:
            processor = ProtenixProcessor(ProtenixProcessorConfig())
            batch = processor(system, msa=msa)
        features = _reference_features(batch, include_constraints=constrained)
        noisy = torch.randn(
            1,
            batch.atom_features.shape[-2],
            3,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        noise = torch.tensor([1.25], dtype=torch.float32)
        expected = _run_reference_explicit(
            reference=reference,
            checkpoint=checkpoints[current],
            model_name=f"protenix_base_{current}_v0.5.0",
            features=features,
            noisy_coordinates=noisy,
            noise_level=noise,
            device=device,
        )
        actual = _run_native_explicit(
            checkpoint=checkpoints[current],
            batch=batch,
            noisy_coordinates=noisy,
            noise_level=noise,
            device=device,
        )
        _compare_mappings(
            expected,
            actual,
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
            label=f"forward {current}",
        )


def released_surface(
    *,
    default_checkpoint: Path,
    constraint_checkpoint: Path,
    device_name: str,
    seed: int,
) -> None:
    """Accept the v0.5.3 sampler, FoldResult, ranking, and released constraints."""

    import torch

    from ofoldx.models import protenix as family

    processor_class = getattr(family, "ProtenixProcessor", None)
    processor_config_class = getattr(family, "ProtenixProcessorConfig", None)
    if processor_class is None or processor_config_class is None:
        raise Blocked("ProtenixProcessor and ProtenixProcessorConfig are not implemented")
    from ofoldx.models.protenix.convert import convert_protenix_checkpoint

    device = _device(device_name)
    system, msa, conditions = _canonical_case()

    default_model = convert_protenix_checkpoint(_local_file(default_checkpoint, "default checkpoint"))
    default_model.to(device).eval().set_backend("torch")
    default_processor = processor_class(processor_config_class(), device=device)
    default_batch = default_processor(system, msa=msa)
    default_result = default_model.fold(
        default_batch,
        num_samples=2,
        generator=torch.Generator(device=default_batch.atom_features.device).manual_seed(seed),
    )
    _check_fold_result(default_result, num_samples=2)
    del default_batch, default_result, default_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    constraint_processor_class = getattr(family, "ProtenixConstraintProcessor", None)
    constraint_processor_config_class = getattr(family, "ProtenixConstraintProcessorConfig", None)
    if constraint_processor_class is None or constraint_processor_config_class is None:
        raise Blocked("ProtenixConstraintProcessor and ProtenixConstraintProcessorConfig are not implemented")

    constraint_model = convert_protenix_checkpoint(_local_file(constraint_checkpoint, "constraint checkpoint"))
    constraint_model.to(device).eval().set_backend("torch")
    # The v0.5.3 constraint checkpoint consumes token contacts only through the distinct
    # constraint Processor signature, which declares the ``token_contacts`` keyword the
    # default Processor deliberately omits.
    constraint_processor = constraint_processor_class(constraint_processor_config_class(), device=device)
    contact = (conditions[0],)
    constraint_batch = constraint_processor(system, msa=msa, token_contacts=contact)
    constraint_result = constraint_model.fold(
        constraint_batch,
        num_samples=1,
        generator=torch.Generator(device=constraint_batch.atom_features.device).manual_seed(seed),
    )
    _check_fold_result(constraint_result, num_samples=1)


def repository_surface(*, checkpoint: Path, device_name: str, seed: int) -> None:
    """Accept repository writing, Auto factories, FoldPipeline, and top-level fold."""

    from ofoldx.models import protenix as family

    processor_class = getattr(family, "ProtenixProcessor", None)
    processor_config_class = getattr(family, "ProtenixProcessorConfig", None)
    if processor_class is None or processor_config_class is None:
        raise Blocked("Protenix Processor surface is not implemented")
    try:
        from ofoldx.models.registry import _registration

        registration = _registration("ofoldx.protenix")
    except ValueError as error:
        raise Blocked("ofoldx.protenix is not registered for fold") from error
    if "fold" not in registration.standard_tasks:
        raise Blocked("ofoldx.protenix registration does not declare fold")

    from ofoldx import AutoModel, AutoProcessor, fold
    from ofoldx.artifacts import save_model
    from ofoldx.models.protenix.convert import convert_protenix_checkpoint
    from ofoldx.pipelines import FoldPipeline

    device = _device(device_name)
    system, msa, _ = _canonical_case()
    model = convert_protenix_checkpoint(_local_file(checkpoint, "default checkpoint"))
    processor = processor_class(processor_config_class())
    with TemporaryDirectory(prefix="ofoldx-protenix-parity-") as directory:
        repository = Path(directory) / "model"
        save_model(repository, model=model, processor=processor)
        auto_model = AutoModel(repository, device=device)
        auto_processor = AutoProcessor(repository, device=device)
        if type(auto_model) is not registration.model or type(auto_processor) is not registration.processor:
            raise AssertionError("Auto factories did not restore the registered Protenix pair")

        pipeline_result = FoldPipeline(model=auto_model, processor=auto_processor).fold(
            system,
            msa=msa,
            num_samples=1,
            seed=seed,
        )
        _check_fold_result(pipeline_result, num_samples=1)
        del pipeline_result, auto_model, auto_processor
        gc.collect()

        top_level_result = fold(
            system,
            model=repository,
            device=device,
            msa=msa,
            num_samples=1,
            seed=seed,
        )
        _check_fold_result(top_level_result, num_samples=1)


def processor_packing(*, reference: Path) -> None:
    r"""Independently verify the native ProtenixProcessor packs each feature channel.

    The forward stage slices the native processor's *own* packed channels
    (``restype`` [384:416], ``profile`` [416:448], ``deletion_mean`` [448],
    ``ref_element`` [1:129], ``ref_charge``, ``ref_atom_name_chars`` [129:385]) and
    feeds them to BOTH graphs, so a native channel-packing bug is consumed identically
    on both sides and masked.  This stage ties every packed channel to independent
    ground truth: the upstream Protenix featurizer's own encoders for restype/element/
    atom-name, and the canonical MSA/system directly for profile/deletion_mean/charge.
    It loads no checkpoint and needs no GPU; the Processor packs on CPU.
    """

    import numpy as np
    import torch

    from ofoldx.data.molecular import AtomSite, PolymerSequence
    from ofoldx.models.protenix import ProtenixProcessor, ProtenixProcessorConfig

    system, msa, _ = _canonical_case()
    batch = ProtenixProcessor(ProtenixProcessorConfig())(system, msa=msa)

    token_features = batch.token_features[0].detach().cpu()
    atom_features = batch.atom_features[0].detach().cpu()
    charges = batch.reference_charges[0, :, 0].detach().cpu()
    reference_mask = batch.reference_mask[0].detach().cpu().float()
    atom_sites = batch.atom_sites
    token_sites = batch.token_sites

    # Frozen channel widths and boundaries (a packing bug most easily shows up here).
    token_shape = tuple(token_features.shape)
    if token_shape != (len(token_sites), 449):
        raise AssertionError(f"token_features layout is {token_shape}, expected {(len(token_sites), 449)}")
    if tuple(atom_features.shape) != (len(atom_sites), 385):
        raise AssertionError(f"atom_features layout is {tuple(atom_features.shape)}, expected {(len(atom_sites), 385)}")
    padding = float(token_features[:, :384].abs().max())
    if padding != 0.0:
        raise AssertionError(f"token_features[:384] must be zero padding, saw max abs {padding:g}")

    _activate_reference(reference)
    from protenix.data.constants import PROT_STD_RESIDUES_ONE_TO_THREE, STD_RESIDUES_WITH_GAP
    from protenix.data.core.featurizer import Featurizer

    # restype [384:416]: upstream's own one-hot over STD_RESIDUES_WITH_GAP, keyed to token identity. An
    # atom-level token (ligand/non-standard) is UNK; a residue token uses its component id when standard.
    def _restype_name(site: Any) -> str:
        if isinstance(site, AtomSite):
            return "UNK"
        component_id = system.component(site).component_id
        return component_id if component_id in STD_RESIDUES_WITH_GAP else "UNK"

    restype_names = [_restype_name(site) for site in token_sites]
    _assert_channel_equal(
        token_features[:, 384:416],
        Featurizer.restype_onehot_encoded(restype_names),
        label="token_features[384:416] restype",
    )

    # ref_element [1:129]: upstream's own element one-hot, keyed to each atom's true element.
    elements = [system.atom(site).element.upper() for site in atom_sites]
    _assert_channel_equal(
        atom_features[:, 1:129],
        Featurizer.elem_onehot_encoded(elements),
        label="atom_features[1:129] ref_element",
    )

    # ref_atom_name_chars [129:385]: upstream's own 4x64 char one-hot, keyed to each atom name.
    atom_names = [site.atom_id for site in atom_sites]
    _assert_channel_equal(
        atom_features[:, 129:385],
        Featurizer.ref_atom_name_chars_encoded(atom_names).reshape(len(atom_sites), 256),
        label="atom_features[129:385] ref_atom_name_chars",
    )

    # atom_features channel 0 carries the reference conformer mask.
    _assert_channel_equal(atom_features[:, 0], reference_mask, label="atom_features[0] reference_mask")

    # ref_charge: each atom's true formal charge (independent of the packed channels).
    expected_charge = torch.tensor([float(system.atom(site).formal_charge) for site in atom_sites])
    _assert_channel_equal(charges, expected_charge, label="reference_charges ref_charge")

    # profile [416:448] + deletion_mean [448]: computed straight from the canonical MSA, so a
    # channel that merely duplicated restype (single-column depth-1 one-hot) would be caught -- the
    # canonical protein column has CYS/GLY at 0.5 each, which no restype one-hot can equal.
    expected_profile = torch.zeros(len(token_sites), 32)
    expected_deletion = torch.zeros(len(token_sites))
    deletion_counts = np.asarray(msa.unpaired_deletion_counts)
    one_to_three = PROT_STD_RESIDUES_ONE_TO_THREE
    for index, site in enumerate(token_sites):
        # An atom-level token (ligand/non-standard) carries no MSA column, so its profile is the plain
        # restype one-hot and its deletion mean is zero -- exactly the native non-protein branch.
        if isinstance(site, AtomSite):
            expected_profile[index] = Featurizer.restype_onehot_encoded([restype_names[index]])[0]
            continue
        instance_id, column = site
        entity = system.entities[system.instances[instance_id]]
        protein_alignment = (
            getattr(msa, "entity_id", None) == system.instances[instance_id]
            and isinstance(entity, PolymerSequence)
            and entity.polymer_type == "protein"
        )
        if not protein_alignment:
            expected_profile[index] = Featurizer.restype_onehot_encoded([restype_names[index]])[0]
            continue
        column_names = [one_to_three.get(sequence[column], "UNK") for sequence in msa.unpaired_sequences]
        expected_profile[index] = Featurizer.restype_onehot_encoded(column_names).mean(dim=0)
        expected_deletion[index] = float(deletion_counts[:, column].mean())
    _assert_channel_equal(token_features[:, 416:448], expected_profile, label="token_features[416:448] profile")
    _assert_channel_equal(token_features[:, 448], expected_deletion, label="token_features[448] deletion_mean")

    # A fully hardcoded anchor independent of BOTH implementations: oxygen -> atomic number 8 ->
    # element index 7, and the first canonical protein token is ALA -> restype index 0.
    oxygen_index = atom_sites.index(AtomSite("L", 0, "O1"))
    oxygen_channel = int(atom_features[oxygen_index, 1:129].argmax())
    if oxygen_channel != 7:
        raise AssertionError(f"ligand O1 ref_element one-hot fires at {oxygen_channel}, expected 7 (oxygen)")
    first_restype = int(token_features[0, 384:416].argmax())
    if first_restype != 0:
        raise AssertionError(f"first protein token restype one-hot fires at {first_restype}, expected 0 (ALA)")

    print(
        "processor packing verified: restype/profile/deletion_mean + ref_element/ref_charge/"
        f"ref_atom_name_chars over {len(token_sites)} tokens / {len(atom_sites)} atoms",
        flush=True,
    )


def _assert_channel_equal(actual: Any, expected: Any, *, label: str, atol: float = 1e-6) -> None:
    import torch

    left = actual.detach().float().cpu()
    right = expected.detach().float().cpu() if hasattr(expected, "detach") else torch.as_tensor(expected).float()
    if tuple(left.shape) != tuple(right.shape):
        raise AssertionError(f"{label} shape differs: {tuple(left.shape)} != {tuple(right.shape)}")
    delta = (left - right).abs()
    max_absolute = float(delta.max()) if delta.numel() else 0.0
    passed = bool(torch.allclose(left, right, atol=atol, rtol=0.0))
    print(f"{label}: shape={tuple(left.shape)} max_abs={max_absolute:.8g} passed={passed}", flush=True)
    if not passed:
        raise AssertionError(f"{label} disagrees with independent ground truth: max abs {max_absolute:g} > {atol:g}")


def _canonical_case() -> tuple[Any, Any, tuple[Any, ...]]:
    import numpy as np

    from ofoldx.data import (
        MSA,
        AtomSite,
        ChemicalAtom,
        ChemicalBond,
        ChemicalComponent,
        MolecularSystem,
        Molecule,
        PolymerSequence,
        SystemBond,
    )
    from ofoldx.models.protenix import ProtenixTokenContact

    protein = PolymerSequence.from_one_letter("AC", polymer_type="protein")
    rna = PolymerSequence.from_one_letter("AG", polymer_type="rna")
    dna = PolymerSequence.from_one_letter("CT", polymer_type="dna")
    ligand = Molecule(
        components=(
            ChemicalComponent(
                component_id="LIG",
                atoms=(
                    ChemicalAtom("C1", "C", formal_charge=0),
                    ChemicalAtom("O1", "O", formal_charge=0),
                ),
                bonds=frozenset({ChemicalBond("C1", "O1", order=2.0)}),
                reference_coordinates=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0)),
            ),
        ),
    )
    system = MolecularSystem(
        entities={"protein": protein, "rna": rna, "dna": dna, "ligand": ligand},
        instances={"P": "protein", "R": "rna", "D": "dna", "L": "ligand"},
        bonds=frozenset({SystemBond(AtomSite("P", 1, "SG"), AtomSite("L", 0, "C1"))}),
    )
    msa = MSA(
        entity_id="protein",
        unpaired_sequences=("AC", "AG"),
        unpaired_deletion_counts=np.array([[0, 0], [0, 1]], dtype=np.uint32),
        paired_sequences=("AC",),
        paired_deletion_counts=np.zeros((1, 2), dtype=np.uint32),
    )
    conditions = (
        ProtenixTokenContact(
            endpoint_a=("P", 0),
            endpoint_b=AtomSite("L", 0, "O1"),
            max_distance=8.0,
        ),
    )
    return system, msa, conditions


def _reference_features(batch: Any, *, include_constraints: bool = False) -> dict[str, Any]:
    import torch

    token_features = batch.token_features[0].detach().cpu()
    atom_features = batch.atom_features[0].detach().cpu()
    atom_to_token = batch.atom_to_token[0].detach().cpu()
    token_to_rep_atom = batch.token_to_rep_atom[0].detach().cpu()
    representative = torch.zeros(atom_features.shape[0], dtype=torch.bool)
    representative[token_to_rep_atom] = True
    # The v0.5 contract now carries the lowered MSA as three direct leaves (integer
    # restype rows, plus per-cell has_deletion and deletion_value) rather than a single
    # packed msa_features tensor; feed exactly those to the upstream graph.
    msa = batch.msa[0].detach().cpu()
    has_deletion = batch.has_deletion[0].detach().cpu()
    deletion_value = batch.deletion_value[0].detach().cpu()
    features: dict[str, Any] = {
        "residue_index": batch.residue_index[0].detach().cpu(),
        "token_index": batch.token_index[0].detach().cpu(),
        "asym_id": batch.chain_index[0].detach().cpu(),
        "entity_id": batch.entity_index[0].detach().cpu(),
        "sym_id": batch.symmetry_index[0].detach().cpu(),
        "restype": token_features[:, 384:416],
        "profile": token_features[:, 416:448],
        "deletion_mean": token_features[:, 448],
        "has_frame": torch.ones(token_features.shape[0], dtype=torch.bool),
        "atom_to_token_idx": atom_to_token,
        "atom_to_tokatom_idx": batch.atom_token_index[0].detach().cpu(),
        "ref_pos": batch.reference_positions[0].detach().cpu(),
        "ref_mask": batch.reference_mask[0].detach().cpu().float(),
        "ref_element": atom_features[:, 1:129],
        "ref_charge": batch.reference_charges[0, :, 0].detach().cpu(),
        "ref_atom_name_chars": atom_features[:, 129:385].reshape(-1, 4, 64),
        "ref_space_uid": batch.reference_space_id[0].detach().cpu(),
        "pae_rep_atom_mask": representative.float(),
        "distogram_rep_atom_mask": representative.float(),
        "msa": msa,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "prot_pair_num_alignments": torch.tensor(0, dtype=torch.int32),
        "prot_unpair_num_alignments": torch.tensor(0, dtype=torch.int32),
        "rna_pair_num_alignments": torch.tensor(0, dtype=torch.int32),
        "rna_unpair_num_alignments": torch.tensor(0, dtype=torch.int32),
        "token_bonds": batch.token_bonds[0].detach().cpu(),
        "is_distillation": torch.zeros(1),
    }
    if include_constraints:
        # The four constraint pair-bias channels are now top-level leaves of the input
        # contract (all-zero for the default graph); the upstream graph still consumes
        # them nested under a single ``constraint_feature`` mapping.
        features["constraint_feature"] = {
            name: getattr(batch, name)[0].detach().cpu()
            for name in ("pocket", "contact", "contact_atom", "substructure")
        }
    return features


def _run_reference_explicit(
    *,
    reference: Path,
    checkpoint: Path,
    model_name: str,
    features: dict[str, Any],
    noisy_coordinates: Any,
    noise_level: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    _activate_reference(reference)
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    from protenix.data.utils import make_dummy_feature
    from protenix.model.protenix import Protenix, update_input_feature_dict

    base = copy.deepcopy(configs_base)
    base["data"] = copy.deepcopy(data_configs)
    _deep_update(base, copy.deepcopy(model_configs[model_name]))
    base["model_name"] = model_name
    config = parse_configs(base, fill_required_with_null=True)
    config.enable_tf32 = False
    config.enable_diffusion_shared_vars_cache = False
    config.enable_efficient_fusion = False
    config.triangle_attention = "torch"
    config.triangle_multiplicative = "torch"
    config.model.N_cycle = 1
    config.model.template_embedder.n_blocks = 0
    config.esm.enable = False
    config.infer_setting.chunk_size = None

    reference_features = make_dummy_feature(copy.deepcopy(features), dummy_feats=["template"])
    reference_features = _move_mapping(reference_features, device)
    # Construct the reference graph on CPU (real tensors, not ``meta``): upstream
    # ``DiffusionTransformer.__init__`` calls ``drop_path_value.item()`` at build time, which
    # torch forbids on a meta tensor, and ``FourierEmbedding.__init__`` seeds a CPU
    # ``torch.Generator`` and draws ``torch.randn(..., generator=generator)`` with it -- which a
    # ``torch.device("cuda")`` build context would break (cpu generator vs cuda default device).
    # Both are fine on CPU; the strict assign-load then rebinds every parameter to the checkpoint
    # tensor and ``model.to(device)`` moves the remaining non-persistent buffers.
    model = Protenix(config)
    state = _checkpoint_state(checkpoint, map_location=device)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"reference strict load differs: {incompatible}")
    del state
    model.eval()
    model.to(device)
    _pin_reference_msa(model)

    reference_features = model.relative_position_encoding.generate_relp(reference_features)
    reference_features = update_input_feature_dict(reference_features)
    noisy_coordinates = noisy_coordinates.to(device)
    noise_level = noise_level.to(device)
    with torch.inference_mode():
        token_embedding, single, pair = model.get_pairformer_output(
            reference_features,
            N_cycle=1,
            inplace_safe=False,
            chunk_size=None,
        )
        coordinates = model.diffusion_module(
            noisy_coordinates,
            noise_level,
            reference_features,
            token_embedding,
            single,
            pair,
            pair_z=None,
            p_lm=None,
            c_l=None,
            inplace_safe=False,
            chunk_size=None,
            use_conditioning=True,
            enable_efficient_fusion=False,
        )
        plddt, pae, pde, resolved = model.confidence_head(
            input_feature_dict=reference_features,
            s_inputs=token_embedding,
            s_trunk=single,
            z_trunk=pair,
            pair_mask=None,
            x_pred_coords=coordinates,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            inplace_safe=False,
            chunk_size=None,
        )
        output = {
            "coordinates": coordinates,
            "distogram_logits": model.distogram_head(pair),
            "plddt_logits": plddt,
            "pae_logits": pae,
            "pde_logits": pde,
            "resolved_logits": resolved,
        }
    result = {name: value.detach().float().cpu() for name, value in output.items()}
    del output, model, reference_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _pin_reference_msa(model: Any) -> None:
    r"""Force the released MSAModule to consume every MSA row deterministically.

    The upstream ``MSAModule.forward`` thins the MSA at inference through
    ``sample_msa_feature_dict_random_without_replacement`` -> ``sample_indices``, which draws a
    random subset size with ``torch.randint`` and a random permutation with ``torch.randperm`` on
    the global RNG.  The native explicit forward under comparison consumes every MSA row in its
    original order, so an unpinned reference silently denoises against a random MSA subset and the
    trunk pair diverges by tens of absolute units.  Selecting the ``topk`` strategy with a
    lower bound and cutoff above any real depth keeps all rows in order, matching the native input
    exactly.  This removes a stochastic confound only; it does not touch weights or tolerances.
    """

    module = getattr(model, "msa_module", None)
    configs = getattr(module, "msa_configs", None)
    if not isinstance(configs, dict):
        return
    configs["strategy"] = "topk"
    configs["test_cutoff"] = 1 << 30
    configs["test_lowerb"] = 1 << 30
    configs["train_cutoff"] = 1 << 30
    configs["train_lowerb"] = 1 << 30


def _run_native_explicit(
    *,
    checkpoint: Path,
    batch: Any,
    noisy_coordinates: Any,
    noise_level: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    from ofoldx.models.protenix.convert import convert_protenix_checkpoint

    model = convert_protenix_checkpoint(checkpoint).to(device).eval().set_backend("torch")
    batch = _move_dataclass(batch, device)
    with torch.inference_mode():
        output = model(
            batch,
            noisy_coordinates=noisy_coordinates.unsqueeze(0).to(device),
            noise_level=noise_level.unsqueeze(0).to(device),
            recycling_steps=1,
        )
    result = {field.name: getattr(output, field.name).squeeze(0).detach().float().cpu() for field in fields(output)}
    del output, model, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _check_fold_result(result: Any, *, num_samples: int) -> None:
    from ofoldx.results import FoldResult, Metric, PairMetric, Score

    if type(result) is not FoldResult:
        raise AssertionError(f"public fold returned {type(result).__name__}, not FoldResult")
    if len(result.predictions) != num_samples:
        raise AssertionError(f"public fold returned {len(result.predictions)} predictions, expected {num_samples}")
    for prediction in result.predictions:
        # The frozen §5.2 FoldPrediction exposes decoded values only through its typed mappings
        # (DESIGN.md §5.2): metrics["plddt"], pair_metrics["pae"/"pde"], scores["ptm"]. There is no
        # ``plddt``/``pae`` attribute and no ``condition_outcomes`` field on the frozen carrier.
        if type(prediction.metrics.get("plddt")) is not Metric:
            raise AssertionError("Protenix FoldPrediction lacks a public pLDDT Metric")
        pair_metrics = prediction.pair_metrics
        if type(pair_metrics.get("pae")) is not PairMetric or type(pair_metrics.get("pde")) is not PairMetric:
            raise AssertionError("Protenix FoldPrediction lacks a public PAE or PDE PairMetric")
        if type(prediction.scores.get("ptm")) is not Score:
            raise AssertionError("Protenix FoldPrediction lacks a public pTM Score")
        required_scores = {"plddt", "gpde", "ptm", "iptm", "has_clash", "ranking_score"}
        if not required_scores <= prediction.scores.keys():
            raise AssertionError(f"Protenix confidence/ranking scores are incomplete: {prediction.scores.keys()}")
    if result.best("ranking_score") not in result.predictions:
        raise AssertionError("FoldResult.best did not rank Protenix predictions")


def _compare_output_objects(expected: Any, actual: Any, *, atol: float, rtol: float, label: str) -> None:
    expected_fields = tuple(field.name for field in fields(expected))
    actual_fields = tuple(field.name for field in fields(actual))
    if actual_fields != expected_fields:
        raise AssertionError(f"{label} output leaves differ: {expected_fields!r} != {actual_fields!r}")
    _compare_mappings(
        {name: getattr(expected, name) for name in expected_fields},
        {name: getattr(actual, name) for name in actual_fields},
        atol=atol,
        rtol=rtol,
        label=label,
    )


def _compare_mappings(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    import torch

    if actual.keys() != expected.keys():
        raise AssertionError(f"{label} leaves differ: {expected.keys()} != {actual.keys()}")
    failed: list[str] = []
    for name in expected:
        left = expected[name].detach().float().cpu()
        right = actual[name].detach().float().cpu()
        if left.shape != right.shape:
            raise AssertionError(f"{label}.{name} shape differs: {tuple(left.shape)} != {tuple(right.shape)}")
        delta = (left - right).abs()
        max_absolute = float(delta.max()) if delta.numel() else 0.0
        relative = delta / left.abs().clamp_min(1e-9)
        max_relative = float(relative.max()) if relative.numel() else 0.0
        passed = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
        print(
            f"{label}.{name}: shape={tuple(left.shape)} max_abs={max_absolute:.8g} "
            f"max_rel={max_relative:.8g} passed={passed}",
            flush=True,
        )
        # Report every leaf before failing so a single divergent leaf never masks
        # whether the remaining leaves agree; the fixed tolerance is unchanged.
        if not passed:
            failed.append(name)
    if failed:
        raise AssertionError(f"{label} leaves {failed} exceed fixed atol={atol:g}, rtol={rtol:g}")


def _activate_reference(reference: Path) -> None:
    import os

    # The released ``LayerNorm`` factory selects the CUDA-only ``FusedLayerNorm`` whenever
    # ``LAYERNORM_TYPE`` defaults to ``fast_layernorm``; that kernel raises "input must be a CUDA
    # tensor" on a CPU parity run.  Any other value selects the reference's own torch
    # ``OpenFoldLayerNorm`` (plain ``F.layer_norm``), which runs on CPU and matches the native
    # torch LayerNorm the compared graph uses.  A caller that wants the fused path keeps its own
    # setting.  This is read at reference import time, so it must precede the ``protenix`` import.
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    reference_text = str(reference)
    if reference_text not in sys.path:
        sys.path.insert(0, reference_text)
    loaded = sys.modules.get("protenix")
    if loaded is not None:
        source = Path(inspect.getfile(loaded)).resolve()
        if reference not in source.parents:
            raise RuntimeError(f"a different Protenix package is already imported: {source}")


def _upstream_git_state(reference: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ("git", "-C", str(reference), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "-C", str(reference), "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Blocked(f"cannot identify the Protenix checkout Git state at {reference}") from error
    return revision, dirty


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _checkpoint_state(checkpoint: Path, *, map_location: Any) -> dict[str, Any]:
    import torch

    loaded = torch.load(checkpoint, map_location=map_location, weights_only=True)
    state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def _move_mapping(values: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: (
            value.to(device)
            if isinstance(value, torch.Tensor)
            else _move_mapping(value, device) if isinstance(value, dict) else value
        )
        for key, value in values.items()
    }


def _move_dataclass(value: Any, device: Any) -> Any:
    import torch

    moved = {}
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, torch.Tensor):
            item = item.to(device)
        elif hasattr(item, "__dataclass_fields__") and any(
            isinstance(getattr(item, nested.name), torch.Tensor) for nested in fields(item)
        ):
            item = _move_dataclass(item, device)
        moved[field.name] = item
    return type(value)(**moved)


def _local_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local {label} does not exist: {path}")
    return path


def _device(name: str) -> Any:
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


if __name__ == "__main__":
    # Keep every diagnostic on stdout so Slurm captures one self-contained stream.
    sys.stderr = sys.stdout
    try:
        raise SystemExit(main())
    except Blocked as error:
        print(f"BLOCKED protenix: {error}", flush=True)
        raise SystemExit(2) from None
    except Exception as error:
        import traceback

        traceback.print_exc()
        print(f"FAIL protenix: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1) from None
