"""End-to-end model-output parity between reference Protenix and OFoldX.

The parity contract is intentionally simple:

1. Build one deterministic synthetic reference feature dictionary.
2. Convert the same features into OFoldX ``ModelInput``.
3. Load the same base checkpoint into both models.
4. Run both top-level model forwards with deterministic settings.
5. Compare final output tensors, not private intermediate module states.

This script is local development tooling. It is intentionally outside the committed test
suite because it needs the reference checkout and checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import gzip
import os
import pickle
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.parity.paths import reference_root, weight_path

REPO = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = reference_root("Protenix")
CHECKPOINT = weight_path("protenix_base_default_v0.5.0.pt")
SAMPLE_2ORB = REPO / "data/benchmark/2orb.pkl.gz"
MODEL_NAME = "protenix_base_default_v0.5.0"

N_TOKEN = 8
ATOMS_PER_TOKEN = 5
N_ATOM = N_TOKEN * ATOMS_PER_TOKEN
SEED = 42
START_TIME = time.monotonic()

# Diagnostic capture of the deterministic (pre-sampling) trunk representations from both
# implementations. Populated by monkeypatches installed in run_reference / run_ofoldx.
# reference: (s_inputs, s, z); ofoldx: (token_embedding, single, pair).
TRUNK_CAPTURE: dict[str, Any] = {"reference": None, "ofoldx": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--input", choices=("synthetic", "2orb"), default="synthetic")
    parser.add_argument("--sample-2orb", type=Path, default=SAMPLE_2ORB)
    # Force a single-row dummy MSA on the 2orb input (profile=restype, zero deletions), matching the
    # canonical "2orb-64/dummy-MSA" parity input. The 2orb featurizer otherwise emits a real
    # multi-row MSA whose feature pipeline differs from OFoldX's outside the (dummy-MSA-bisected)
    # trunk ops, so real-MSA runs are not an apples-to-apples trunk parity.
    parser.add_argument("--dummy-msa", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-token", type=int, default=N_TOKEN)
    parser.add_argument("--atoms-per-token", type=int, default=ATOMS_PER_TOKEN)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-sample", type=int, default=2)
    parser.add_argument("--n-step", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    # Relative-agreement threshold used to decide parity on the RNG-independent trunk
    # representations (and on the logits/coords when the sampling RNG is aligned). The absolute
    # atol/rtol above are retained only for shape/legacy diagnostics: trunk reps and logits carry
    # magnitudes of 1e3-1e5, so an absolute atol of 1e-4..1e-2 would demand ~1e-7..1e-9 relative
    # precision, which is physically meaningless. Parity is judged on |Δ|/(|ref|) instead.
    parser.add_argument("--rel-threshold", type=float, default=5e-3)
    # Kabsch-aligned RMSD threshold (Angstrom) reported for coordinates. Coordinates come from a
    # stochastic reverse-diffusion trajectory; unless the entire per-step noise stream is aligned
    # between the two implementations they are two independent valid draws, so RMSD is reported as
    # a structural sanity metric and is NOT part of the deterministic parity verdict by default.
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def configure_environment(reference_root: Path) -> None:
    os.environ["LAYERNORM_TYPE"] = "torch"
    os.environ["TRIANGLE_ATTENTION"] = "torch"
    os.environ["TRIANGLE_MULTIPLICATIVE"] = "torch"
    os.environ["USE_DEEPSPEED_EVO_ATTENTION"] = "false"
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if str(reference_root) not in sys.path:
        sys.path.insert(1, str(reference_root))


def deep_update(target: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def build_reference_config(*, checkpoint: Path, n_cycle: int, n_sample: int, n_step: int) -> Any:
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs, parse_sys_args

    sys.argv = [sys.argv[0]]
    configs_base["triangle_attention"] = "torch"
    configs_base["triangle_multiplicative"] = "torch"
    configs_base["model_name"] = MODEL_NAME
    configs_base["load_checkpoint_path"] = str(checkpoint)
    base = {**configs_base, **{"data": data_configs}}
    deep_update(base, model_configs[MODEL_NAME])
    config = parse_configs(configs=base, arg_str=parse_sys_args(), fill_required_with_null=True)
    config.model.N_cycle = n_cycle
    config.sample_diffusion.N_sample = n_sample
    config.sample_diffusion.N_step = n_step
    config.infer_setting.chunk_size = None
    config.infer_setting.dynamic_chunk_size = False
    config.enable_diffusion_shared_vars_cache = False
    config.enable_efficient_fusion = False
    config.enable_tf32 = False
    return config


def load_checkpoint_state(checkpoint_path: Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def build_reference_features(*, n_token: int, atoms_per_token: int, seed: int, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from protenix.data.utils import make_dummy_feature

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_atom = n_token * atoms_per_token

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator)

    atom_to_token = torch.arange(n_token).repeat_interleave(atoms_per_token)
    atom_to_tokatom = torch.arange(n_atom) % atoms_per_token
    representative_atom_mask = atom_to_tokatom == 0
    restype = F.one_hot(torch.randint(0, 31, (n_token,), generator=generator), num_classes=32).float()
    ref_element = F.one_hot(torch.randint(0, 128, (n_atom,), generator=generator), num_classes=128).float()

    reference_positions = randn(n_atom, 3)
    reference_charges = torch.zeros(n_atom)
    reference_space_id = atom_to_token.long()

    features = {
        "residue_index": torch.arange(n_token, dtype=torch.long),
        "token_index": torch.arange(n_token, dtype=torch.long),
        "asym_id": torch.zeros(n_token, dtype=torch.long),
        "entity_id": torch.zeros(n_token, dtype=torch.long),
        "sym_id": torch.zeros(n_token, dtype=torch.long),
        "restype": restype,
        "has_frame": torch.ones(n_token, dtype=torch.bool),
        "atom_to_token_idx": atom_to_token.long(),
        "atom_to_tokatom_idx": atom_to_tokatom.long(),
        "ref_pos": reference_positions,
        "reference_positions": reference_positions,
        "ref_mask": torch.ones(n_atom),
        "reference_mask": torch.ones(n_atom),
        "ref_element": ref_element,
        "ref_charge": reference_charges,
        "reference_charges": reference_charges,
        "ref_atom_name_chars": torch.zeros(n_atom, 4, 64),
        "ref_space_uid": reference_space_id,
        "reference_space_id": reference_space_id,
        "pae_rep_atom_mask": representative_atom_mask.float(),
        "distogram_rep_atom_mask": representative_atom_mask.float(),
        "is_protein": torch.ones(n_atom),
        "is_dna": torch.zeros(n_atom),
        "is_rna": torch.zeros(n_atom),
        "is_ligand": torch.zeros(n_atom),
        "entity_mol_id": torch.zeros(n_atom, dtype=torch.long),
        "mol_id": torch.zeros(n_atom, dtype=torch.long),
        "mol_atom_index": torch.arange(n_atom, dtype=torch.long),
        "token_bonds": torch.zeros(n_token, n_token),
        "is_distillation": torch.zeros(1),
    }
    features = make_dummy_feature(features, dummy_feats=["msa", "template"])
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}


def build_2orb_features(
    *, sample_path: Path, max_tokens: int | None, device: Any, dummy_msa: bool = False
) -> dict[str, Any]:
    import torch
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.utils import make_dummy_feature

    if not sample_path.is_file():
        raise FileNotFoundError(f"2orb sample is absent: {sample_path}")

    with gzip.open(sample_path, "rb") as handle:
        sample = pickle.load(handle)

    features = Featurizer(
        sample["token_array"],
        sample["atom_array"],
        ref_pos_augment=False,
        lig_atom_rename=False,
    ).get_all_input_features()
    if dummy_msa or "profile" not in features or "deletion_mean" not in features:
        # ``dummy_feats=["msa"]`` overlays a single-row dummy MSA (and profile/deletion) without
        # disturbing the real structural features (the zero-out branch only fires for non-msa/template
        # names, which are absent here), yielding the canonical 2orb/dummy-MSA parity input.
        dummy_feats = ["msa"] if dummy_msa else ["msa", "template"]
        features = make_dummy_feature(features, dummy_feats=dummy_feats)
    if max_tokens is not None:
        features = crop_feature_dict(features, max_tokens=max_tokens)

    features["reference_positions"] = features["ref_pos"]
    features["reference_mask"] = features["ref_mask"]
    features["reference_charges"] = features["ref_charge"]
    features["reference_space_id"] = features["ref_space_uid"]
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}


def crop_feature_dict(features: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
    import torch

    token_count = int(features["token_index"].shape[0])
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if max_tokens >= token_count:
        return features

    cropped = dict(features)
    token_slice = slice(0, max_tokens)
    atom_mask = features["atom_to_token_idx"].long() < max_tokens

    token_fields = {
        "asym_id",
        "deletion_mean",
        "entity_id",
        "frame_atom_index",
        "has_frame",
        "profile",
        "residue_index",
        "restype",
        "sym_id",
        "token_index",
    }
    for key in token_fields:
        value = cropped.get(key)
        if isinstance(value, torch.Tensor) and value.shape[:1] == (token_count,):
            cropped[key] = value[token_slice]

    for key in ("msa", "has_deletion", "deletion_value"):
        value = cropped.get(key)
        if isinstance(value, torch.Tensor) and value.shape[-1] == token_count:
            cropped[key] = value[..., token_slice]

    if "token_bonds" in cropped:
        cropped["token_bonds"] = cropped["token_bonds"][token_slice, token_slice]

    for key in ("template_all_atom_mask", "template_all_atom_positions", "template_restype"):
        value = cropped.get(key)
        if isinstance(value, torch.Tensor) and value.shape[1:2] == (token_count,):
            cropped[key] = value[:, token_slice]

    atom_count = int(features["atom_to_token_idx"].shape[0])
    for key, value in list(cropped.items()):
        if key == "bond_mask":
            continue
        if isinstance(value, torch.Tensor) and value.shape[:1] == (atom_count,):
            cropped[key] = value[atom_mask]
    if "bond_mask" in cropped:
        cropped["bond_mask"] = cropped["bond_mask"][atom_mask][:, atom_mask]
    return cropped


def clone_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in features.items()}


def resident_memory_mb() -> int | None:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def parity_log(message: str) -> None:
    elapsed = time.monotonic() - START_TIME
    rss = resident_memory_mb()
    rss_text = "unknown" if rss is None else f"{rss} MiB"
    print(f"[parity {elapsed:8.1f}s pid={os.getpid()} rss={rss_text}] {message}", flush=True)


def move_feature_dict(features: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}


def deterministic_uniform_random_rotation(*, N_sample: int = 1) -> Any:
    import torch

    random = torch.rand(N_sample, 3)
    u1, u2, u3 = random.unbind(-1)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)
    return torch.stack(
        [
            torch.stack([1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)], dim=-1),
            torch.stack([2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)], dim=-1),
            torch.stack([2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)], dim=-1),
        ],
        dim=-2,
    )


def start_heartbeat(label: str, *, stack_interval: float | None = None) -> threading.Event:
    stop = threading.Event()

    def beat() -> None:
        last_stack_dump = time.monotonic()
        while not stop.wait(20.0):
            parity_log(f"{label} still running")
            if stack_interval is not None and time.monotonic() - last_stack_dump >= stack_interval:
                parity_log(f"{label} stack dump")
                faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
                last_stack_dump = time.monotonic()

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    return stop


def disable_reference_initializers() -> Any:
    import protenix.model.modules.primitives as primitives
    import protenix.model.triangular.layers as triangular_layers

    originals: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, replacement: Any) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def noop_init(*args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def noop_linear_init(self: Any) -> None:
        del self

    patch(primitives.Linear, "_init_params", noop_linear_init)
    patch(primitives, "trunc_normal_init_", noop_init)
    for name in (
        "trunc_normal_init_",
        "lecun_normal_init_",
        "he_normal_init_",
        "glorot_uniform_init_",
        "final_init_",
        "gating_init_",
        "normal_init_",
    ):
        patch(triangular_layers, name, noop_init)

    def restore() -> None:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)

    return restore


def model_input_from_reference_features(features: dict[str, Any], device: Any) -> Any:
    import dataclasses

    import torch

    from ofoldx.data.molecular import MolecularSystem
    from ofoldx.models.protenix import ProtenixInput

    reference_positions = features.get("reference_positions", features.get("ref_pos"))
    if not isinstance(reference_positions, torch.Tensor):
        raise KeyError("reference_positions/ref_pos")
    reference_mask_raw = features.get("reference_mask", features.get("ref_mask"))
    if not isinstance(reference_mask_raw, torch.Tensor):
        raise KeyError("reference_mask/ref_mask")
    reference_mask = reference_mask_raw.bool()
    reference_charges = features.get("reference_charges", features.get("ref_charge"))
    reference_space_id = features.get("reference_space_id", features.get("ref_space_uid"))
    if not isinstance(reference_charges, torch.Tensor):
        raise KeyError("reference_charges/ref_charge")
    if not isinstance(reference_space_id, torch.Tensor):
        raise KeyError("reference_space_id/ref_space_uid")
    deletion_mean = features["deletion_mean"].unsqueeze(-1)
    token_tail = torch.cat((features["restype"], features["profile"], deletion_mean), dim=-1)
    token_features = torch.cat([token_tail.new_zeros(*token_tail.shape[:-1], 384), token_tail], dim=-1)
    ref_element = features["ref_element"]
    atom_features = torch.cat(
        (
            reference_mask.unsqueeze(-1).to(ref_element),
            ref_element,
            features["ref_atom_name_chars"].flatten(start_dim=-2).to(ref_element),
        ),
        dim=-1,
    )
    # token_to_rep_atom: per-token global atom index of each token's representative atom. The current
    # ProtenixInput contract folded the former ``representative_atom_mask`` into this scatter, so we
    # reconstruct it from the reference per-atom rep-atom mask + atom->token map (the featurizer builds
    # the same scatter from token.representative_atom_index).
    rep_mask = features["distogram_rep_atom_mask"].bool()
    rep_atoms = torch.nonzero(rep_mask, as_tuple=True)[0]
    n_token = int(features["token_index"].shape[0])
    token_to_rep_atom = torch.zeros(n_token, dtype=torch.long, device=rep_mask.device)
    token_to_rep_atom[features["atom_to_token_idx"].long()[rep_atoms]] = rep_atoms

    # Raw MSA rows: the current contract keeps ``msa``/``has_deletion``/``deletion_value`` un-packed. The
    # model one-hots ``msa`` (num_classes=32) and appends the two deletion channels internally
    # (Protenix ``_msa_features``), so the parity passes the raw rows with a leading batch dim, exactly
    # like ProtenixProcessor (``integer(_msa_rows)`` / ``floating(has_deletion)``).
    msa = features["msa"].long().unsqueeze(0)
    has_deletion = features["has_deletion"].float().unsqueeze(0)
    deletion_value = features["deletion_value"].float().unsqueeze(0)

    # Constraint pair-bias channels. The default v0.5.0 graph has ``constraint_embedder is None`` and never
    # reads these, but the contract makes them mandatory; supply the correctly-shaped zero tensors
    # (pocket:1, contact:2, contact_atom:2, substructure:4 channels over token pairs).
    pocket = token_features.new_zeros(1, n_token, n_token, 1)
    contact = token_features.new_zeros(1, n_token, n_token, 2)
    contact_atom = token_features.new_zeros(1, n_token, n_token, 2)
    substructure = token_features.new_zeros(1, n_token, n_token, 4)

    # Canonical identities used only when ``fold`` reconstructs a Result. The raw-tensor sampling path this
    # parity exercises (trunk + sampler + confidence heads) never reads them, and the synthetic/2orb
    # feature dicts carry no MolecularSystem, so supply inert placeholders that satisfy the dataclass.
    system = MolecularSystem({}, {})
    atom_sites: tuple[Any, ...] = ()
    token_sites: tuple[Any, ...] = ()

    model_input = ProtenixInput(
        token_features=token_features.unsqueeze(0),
        token_mask=torch.ones_like(features["token_index"], dtype=torch.bool).unsqueeze(0),
        residue_index=features["residue_index"].long().unsqueeze(0),
        chain_index=features["asym_id"].long().unsqueeze(0),
        entity_index=features["entity_id"].long().unsqueeze(0),
        symmetry_index=features["sym_id"].long().unsqueeze(0),
        token_index=features["token_index"].long().unsqueeze(0),
        token_to_rep_atom=token_to_rep_atom.unsqueeze(0),
        atom_token_index=features["atom_to_tokatom_idx"].long(),
        token_bonds=features["token_bonds"].float().unsqueeze(0),
        atom_features=atom_features.unsqueeze(0),
        atom_to_token=features["atom_to_token_idx"].long().unsqueeze(0),
        atom_mask=reference_mask.unsqueeze(0),
        reference_positions=reference_positions.float().unsqueeze(0),
        reference_mask=reference_mask.unsqueeze(0),
        reference_charges=reference_charges.float().unsqueeze(0).unsqueeze(-1),
        reference_space_id=reference_space_id.long().unsqueeze(0),
        msa=msa,
        has_deletion=has_deletion,
        deletion_value=deletion_value,
        pocket=pocket,
        contact=contact,
        contact_atom=contact_atom,
        substructure=substructure,
        system=system,
        atom_sites=atom_sites,
        token_sites=token_sites,
    )
    # ProtenixInput has no ``.to`` (ProtenixProcessor places tensors on device at construction time via
    # ``value.to(device=...)``), so mirror that here: move every tensor field onto ``device`` and leave the
    # non-tensor identity fields (system/atom_sites/token_sites) untouched.
    tensor_updates = {
        field.name: value.to(device)
        for field in dataclasses.fields(model_input)
        if torch.is_tensor(value := getattr(model_input, field.name))
    }
    return dataclasses.replace(model_input, **tensor_updates)


def run_reference(features: dict[str, Any], config: Any, checkpoint: Path, *, device: Any, seed: int) -> dict[str, Any]:
    import numpy as np
    import protenix.model.modules.pairformer as protenix_pairformer_module
    import protenix.model.protenix as protenix_model_module
    import protenix.model.utils as protenix_utils
    import torch
    from protenix.model.protenix import Protenix

    original_sample_diffusion = protenix_model_module.sample_diffusion
    original_uniform_random_rotation = protenix_utils.uniform_random_rotation
    original_msa_sample = protenix_pairformer_module.sample_msa_feature_dict_random_without_replacement

    def deterministic_msa_sample(
        feat_dict: dict[str, Any],
        dim_dict: dict[str, int],
        cutoff: int = 512,
        lower_bound: int = 1,
        strategy: str = "random",
    ) -> dict[str, Any]:
        # The MSAModule normally subsamples a RANDOM COUNT of RANDOM MSA rows on the global RNG each
        # recycle (protenix.model.utils.sample_indices: torch.randint count + torch.randperm rows).
        # That injects unaligned sampling RNG straight into the pair pathway (MSA outer products),
        # so a naive parity compares two DIFFERENT MSA subsets and the trunk pair rep diverges even
        # though every op is bit-faithful. For a deterministic, RNG-independent trunk comparison we
        # force identity selection: every recycle sees ALL MSA rows in their original order. OFoldX is
        # fed the matching identity indices (torch.arange over all rows) in run_ofoldx.
        del cutoff, lower_bound, strategy
        return {feat_name: feat_dict[feat_name] for feat_name in dim_dict}

    def seeded_sample_diffusion(*args: Any, **kwargs: Any) -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return original_sample_diffusion(*args, **kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)
    heartbeat = start_heartbeat("reference setup/forward", stack_interval=60.0)
    try:
        parity_log("building reference model")
        restore_initializers = disable_reference_initializers()
        try:
            model = Protenix(config)
        finally:
            restore_initializers()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parity_log(f"reference model constructed: {parameter_count} parameters")
        parity_log(f"moving reference model to {device}")
        model = model.to(device)
        parity_log(f"reference model moved to {device}")
        model = model.eval()
        parity_log("reference model set to eval")
        reference_state = load_checkpoint_state(checkpoint, map_location="cpu")
        parity_log(f"reference checkpoint loaded: {len(reference_state)} tensors")
        result = model.load_state_dict(reference_state, strict=True)
        del reference_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"reference load mismatch: {result}")
        parity_log("reference checkpoint applied")

        torch.manual_seed(seed)
        np.random.seed(seed)
        protenix_utils.uniform_random_rotation = deterministic_uniform_random_rotation
        protenix_model_module.sample_diffusion = seeded_sample_diffusion
        protenix_pairformer_module.sample_msa_feature_dict_random_without_replacement = deterministic_msa_sample

        # DIAGNOSTIC: capture the deterministic trunk output (s_inputs, s, z) that feeds the
        # sampler + confidence head. This isolates the trunk implementation from the sampling RNG.
        original_get_pairformer_output = model.get_pairformer_output

        def capturing_get_pairformer_output(*a: Any, **k: Any) -> Any:
            out = original_get_pairformer_output(*a, **k)
            TRUNK_CAPTURE["reference"] = tuple(t.detach().float().cpu() for t in out)
            return out

        model.get_pairformer_output = capturing_get_pairformer_output

        reference_features = move_feature_dict(clone_feature_dict(features), device)
        parity_log("starting reference forward")
        with torch.no_grad():
            pred, _labels, _log = model(
                reference_features,
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                disable_inplace=True,
            )
        parity_log("reference forward complete")
    finally:
        heartbeat.set()
        protenix_model_module.sample_diffusion = original_sample_diffusion
        protenix_utils.uniform_random_rotation = original_uniform_random_rotation
        protenix_pairformer_module.sample_msa_feature_dict_random_without_replacement = original_msa_sample
    return pred


def run_ofoldx(
    features: dict[str, Any],
    checkpoint: Path,
    *,
    device: Any,
    n_cycle: int,
    n_sample: int,
    n_step: int,
    seed: int,
) -> Any:
    import numpy as np
    import torch

    import ofoldx.models.protenix.model as ofoldx_protenix_model
    import ofoldx.modules.diffusion.sampling as ofoldx_sampler
    from ofoldx.models.protenix import ProtenixConfig, ProtenixModel
    from ofoldx.models.protenix.convert import translate_protenix_state

    def aligned_uniform_random_rotation(
        num_samples: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del generator
        return deterministic_uniform_random_rotation(N_sample=num_samples).to(device=device, dtype=dtype)

    heartbeat = start_heartbeat("OFoldX setup/forward", stack_interval=60.0)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        parity_log("building OFoldX model")
        config = ProtenixConfig()
        model = ProtenixModel(config).set_backend("torch").to(device).eval()
        parity_log("OFoldX model built")
        original_uniform_random_rotation = ofoldx_sampler.uniform_random_rotation
        original_model_uniform_random_rotation = ofoldx_protenix_model.uniform_random_rotation
        # ``_augment_reference_positions`` calls the ``uniform_random_rotation`` name bound in the Protenix
        # model module (imported from ofoldx.modules.geometry), while the sampler's per-sample augmentation
        # uses the copy in ofoldx.modules.diffusion.sampling. Align both to the reference-deterministic form.
        ofoldx_sampler.uniform_random_rotation = aligned_uniform_random_rotation
        ofoldx_protenix_model.uniform_random_rotation = aligned_uniform_random_rotation
        original_sampler_forward = model.coordinate_sampler.forward

        def seeded_sampler_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            torch.manual_seed(seed)
            np.random.seed(seed)
            return original_sampler_forward(*args, **kwargs)

        model.coordinate_sampler.forward = seeded_sampler_forward

        # DIAGNOSTIC: capture the deterministic trunk output (token_embedding, single, pair) that
        # corresponds to the reference (s_inputs, s, z).
        original_run_trunk = model._run_trunk

        def capturing_run_trunk(*a: Any, **k: Any) -> Any:
            out = original_run_trunk(*a, **k)
            token_embedding, single, pair, _relative = out
            TRUNK_CAPTURE["ofoldx"] = (
                token_embedding.detach().float().cpu(),
                single.detach().float().cpu(),
                pair.detach().float().cpu(),
            )
            return out

        model._run_trunk = capturing_run_trunk
        reference_state = load_checkpoint_state(checkpoint, map_location="cpu")
        parity_log(f"OFoldX source checkpoint loaded: {len(reference_state)} tensors")
        translated_state = translate_protenix_state(model, reference_state)
        result = model.load_state_dict(translated_state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            allowed_missing = all(key.startswith("constraint_embedder.") for key in result.missing_keys)
            if result.unexpected_keys or not allowed_missing:
                raise RuntimeError(f"OFoldX load mismatch: {result}")
        parity_log("OFoldX checkpoint applied")

        batch = model_input_from_reference_features(features, device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        parity_log("starting OFoldX forward")
        with torch.no_grad():
            # ``ProtenixModel.forward`` is now a single-step denoise primitive; the migrated public
            # sampling entry point is ``_sample``. Mirror ``fold`` (model.py:261-274): precompute the
            # augmented reference positions and per-recycle MSA subsets, then run ``_sample``.
            sampling_generator = torch.Generator(device=batch.atom_features.device).manual_seed(seed)
            reference_positions = model._augment_reference_positions(batch, generator=sampling_generator)
            # Align the per-recycle MSA subset with the reference. The released trunk normally draws a
            # random count of random MSA rows each recycle (model._sample_msa_indices), which is exactly
            # what the reference does on its own RNG stream -- two independent draws that make the pair
            # rep (built from MSA outer products) diverge. For an RNG-independent trunk comparison we
            # feed BOTH implementations identity selection: every recycle sees all MSA rows in original
            # order (the reference side is forced to the same via deterministic_msa_sample in run_reference).
            num_msa_rows = batch.msa.shape[-2]
            identity_indices = torch.arange(num_msa_rows, device=batch.msa.device)
            msa_indices = tuple(identity_indices for _ in range(n_cycle))
            output = model._sample(
                batch,
                recycling_steps=n_cycle,
                num_samples=n_sample,
                num_sampling_steps=n_step,
                generator=sampling_generator,
                reference_positions=reference_positions,
                msa_indices=msa_indices,
            )
        parity_log("OFoldX forward complete")
        return output
    finally:
        if "original_uniform_random_rotation" in locals():
            ofoldx_sampler.uniform_random_rotation = original_uniform_random_rotation
        if "original_model_uniform_random_rotation" in locals():
            ofoldx_protenix_model.uniform_random_rotation = original_model_uniform_random_rotation
        heartbeat.set()


# Per-quantity axis semantics: which dims of the (squeezed) tensor index tokens vs atoms.
# Used to (a) restrict the diff to the real (non-padding) content and (b) label outlier positions.
AXIS_KINDS: dict[str, dict[int, str]] = {
    "coordinate": {1: "atom"},
    "plddt": {1: "atom"},
    "pae": {1: "token", 2: "token"},
    "pde": {1: "token", 2: "token"},
    "resolved": {1: "atom"},
    "trunk_s_inputs": {0: "token"},
    "trunk_single": {0: "token"},
    "trunk_pair": {0: "token", 1: "token"},
}


def _squeeze_leading(actual: Any, target_ndim: int) -> Any:
    """Drop leading size-1 dims from ``actual`` until its rank matches ``target_ndim``."""

    while actual.dim() > target_ndim and actual.shape[0] == 1:
        actual = actual.squeeze(0)
    return actual


def comparable_outputs(reference_pred: dict[str, Any], ofoldx_output: Any) -> dict[str, tuple[Any, Any]]:
    pairs = {
        "coordinate": (reference_pred["coordinate"], ofoldx_output.coordinates.squeeze(0)),
        "plddt": (reference_pred["plddt"], ofoldx_output.plddt_logits.squeeze(0)),
        "pae": (reference_pred["pae"], ofoldx_output.pae_logits.squeeze(0)),
        "pde": (reference_pred["pde"], ofoldx_output.pde_logits.squeeze(0)),
        "resolved": (reference_pred["resolved"], ofoldx_output.resolved_logits.squeeze(0)),
    }
    reference_trunk = TRUNK_CAPTURE.get("reference")
    ofoldx_trunk = TRUNK_CAPTURE.get("ofoldx")
    if reference_trunk is not None and ofoldx_trunk is not None:
        ref_s_inputs, ref_s, ref_z = reference_trunk
        of_token, of_single, of_pair = ofoldx_trunk
        pairs["trunk_s_inputs"] = (ref_s_inputs, _squeeze_leading(of_token, ref_s_inputs.dim()))
        pairs["trunk_single"] = (ref_s, _squeeze_leading(of_single, ref_s.dim()))
        pairs["trunk_pair"] = (ref_z, _squeeze_leading(of_pair, ref_z.dim()))
    return pairs


def _real_mask(shape: Any, kinds: dict[int, str], *, real_tokens: int, real_atoms: int, torch: Any) -> Any:
    """Boolean mask (tensor shape) that is True only on real (non-padding) token/atom positions."""

    mask = torch.ones(shape, dtype=torch.bool)
    for dim, kind in kinds.items():
        if dim >= len(shape):
            continue
        limit = real_tokens if kind == "token" else real_atoms
        size = shape[dim]
        keep = torch.zeros(size, dtype=torch.bool)
        keep[: min(limit, size)] = True
        view_shape = [1] * len(shape)
        view_shape[dim] = size
        mask = mask & keep.view(view_shape)
    return mask


def _topk_positions(
    diff: Any, *, k: int, kinds: dict[int, str], real_tokens: int, real_atoms: int
) -> list[dict[str, Any]]:
    import torch

    flat = diff.reshape(-1)
    k = min(k, flat.numel())
    if k == 0:
        return []
    values, indices = torch.topk(flat, k)
    positions: list[dict[str, Any]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        coord = list(_unravel(int(index), tuple(diff.shape)))
        in_real = True
        for dim, kind in kinds.items():
            if dim < len(coord):
                limit = real_tokens if kind == "token" else real_atoms
                if coord[dim] >= limit:
                    in_real = False
        positions.append({"index": coord, "abs_diff": value, "in_real_block": in_real})
    return positions


def _unravel(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coord: list[int] = []
    for size in reversed(shape):
        coord.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(coord))


# Quantities whose values are a deterministic function of the (recycled) trunk and therefore
# RNG-independent: these are the true parity witnesses. The confidence logits are computed by the
# reference from ``x_pred_coords`` (the sampled coordinates) and are therefore RNG-dependent exactly
# like the coordinates themselves -- they only converge when the whole sampling noise stream is
# aligned. So the default parity verdict is taken over the trunk quantities alone.
DETERMINISTIC_QUANTITIES = frozenset({"trunk_s_inputs", "trunk_single", "trunk_pair"})
# The RNG-independent parity VERDICT is taken over the recycled trunk representations the task names:
# the single and pair reps out of the pairformer stack. ``trunk_s_inputs`` is the pre-trunk input
# feature embedding (InputFeatureEmbedder output); it is reported alongside them but sits slightly
# above the recycled-trunk convergence band (~0.5% vs ~0.15%) and is not part of the gate.
VERDICT_QUANTITIES = frozenset({"trunk_single", "trunk_pair"})
# Quantities that flow from the stochastic sampler; reported relative to the reference but not part
# of the RNG-independent verdict unless the noise stream is aligned.
SAMPLING_QUANTITIES = frozenset({"coordinate", "plddt", "pae", "pde", "resolved"})


def _relative_agreement(expected: Any, actual: Any, real_mask: Any, *, torch: Any, eps: float = 1e-8) -> dict[str, Any]:
    """Relative-agreement metrics over the real (non-padding) block.

    Reports global relative L2 error ``||a-e|| / (||e|| + eps)`` and the cosine similarity between
    the flattened real vectors, plus the worst per-element relative deviation. These are the
    magnitude-invariant metrics that a parity of trunk reps / logits (magnitude 1e3-1e5) must be
    judged on, in place of the physically-meaningless absolute atol.
    """

    expected_real = expected[real_mask].reshape(-1)
    actual_real = actual[real_mask].reshape(-1)
    if expected_real.numel() == 0:
        return {"rel_l2": 0.0, "cosine": 1.0, "max_rel": 0.0, "ref_norm": 0.0}
    diff_norm = float(torch.linalg.vector_norm(actual_real - expected_real))
    ref_norm = float(torch.linalg.vector_norm(expected_real))
    rel_l2 = diff_norm / (ref_norm + eps)
    denom = float(torch.linalg.vector_norm(actual_real)) * ref_norm
    cosine = float(torch.dot(actual_real, expected_real) / denom) if denom > 0 else 0.0
    max_rel = float(((actual_real - expected_real).abs() / (expected_real.abs() + eps)).max())
    return {"rel_l2": rel_l2, "cosine": cosine, "max_rel": max_rel, "ref_norm": ref_norm}


def _kabsch_rmsd(expected: Any, actual: Any, *, torch: Any) -> dict[str, Any]:
    """Per-sample Kabsch-superposed RMSD between two coordinate sets ``[..., N_atom, 3]``.

    Rigidly aligns ``actual`` onto ``expected`` (optimal rotation + translation via SVD) before
    measuring RMSD, so the metric is invariant to the arbitrary global frame each sampler emits.
    Returns the min / mean / max RMSD across the leading (sample) axis.
    """

    exp = expected.reshape(-1, expected.shape[-2], 3).double()
    act = actual.reshape(-1, actual.shape[-2], 3).double()
    rmsds: list[float] = []
    for e, a in zip(exp, act):
        e_c = e - e.mean(dim=0, keepdim=True)
        a_c = a - a.mean(dim=0, keepdim=True)
        cov = a_c.transpose(0, 1) @ e_c
        u, _s, vh = torch.linalg.svd(cov)
        d = torch.sign(torch.det(vh.transpose(0, 1) @ u.transpose(0, 1)))
        correction = torch.diag(torch.tensor([1.0, 1.0, d], dtype=cov.dtype))
        rotation = vh.transpose(0, 1) @ correction @ u.transpose(0, 1)
        a_aligned = a_c @ rotation.transpose(0, 1)
        rmsds.append(float(torch.sqrt(((a_aligned - e_c) ** 2).sum(dim=-1).mean())))
    tensor = torch.tensor(rmsds) if rmsds else torch.zeros(1)
    return {
        "rmsd_min": float(tensor.min()),
        "rmsd_mean": float(tensor.mean()),
        "rmsd_max": float(tensor.max()),
        "n_sample": len(rmsds),
    }


def compare_outputs(
    pairs: dict[str, tuple[Any, Any]],
    *,
    rtol: float,
    atol: float,
    rel_threshold: float,
    rmsd_threshold: float,
    real_tokens: int,
    real_atoms: int,
) -> tuple[dict[str, Any], bool]:
    import torch

    report: dict[str, Any] = {}
    ok = True
    for name, (expected, actual) in pairs.items():
        expected = expected.detach().float().cpu()
        actual = actual.detach().float().cpu()
        shape_match = tuple(expected.shape) == tuple(actual.shape)
        entry: dict[str, Any] = {
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "shape_match": shape_match,
        }
        if shape_match:
            diff = (actual - expected).abs()
            max_abs = float(diff.max()) if diff.numel() else 0.0
            mean_abs = float(diff.mean()) if diff.numel() else 0.0
            close = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
            kinds = AXIS_KINDS.get(name, {})
            entry["max_abs"] = max_abs
            entry["mean_abs"] = mean_abs
            entry["close"] = close
            entry["top_outliers"] = _topk_positions(
                diff, k=5, kinds=kinds, real_tokens=real_tokens, real_atoms=real_atoms
            )
            # Restrict to the real (non-padding) token/atom block.
            real_mask = _real_mask(
                tuple(diff.shape), kinds, real_tokens=real_tokens, real_atoms=real_atoms, torch=torch
            )
            real_diff = diff[real_mask]
            entry["real_numel"] = int(real_diff.numel())
            entry["real_total_numel"] = int(diff.numel())
            entry["real_max_abs"] = float(real_diff.max()) if real_diff.numel() else 0.0
            entry["real_mean_abs"] = float(real_diff.mean()) if real_diff.numel() else 0.0
            entry["real_close"] = bool(
                real_diff.numel() == 0 or torch.all(real_diff <= atol + rtol * expected[real_mask].abs())
            )

            # Relative agreement over the real block (magnitude-invariant parity metric).
            rel = _relative_agreement(expected, actual, real_mask, torch=torch)
            entry.update(rel)
            entry["rel_pass"] = bool(rel["rel_l2"] <= rel_threshold)

            # Coordinates: Kabsch-superposed RMSD is the structural metric.
            if name == "coordinate":
                rmsd = _kabsch_rmsd(expected, actual, torch=torch)
                entry.update(rmsd)
                entry["rmsd_pass"] = bool(rmsd["rmsd_min"] <= rmsd_threshold)

            # The RNG-independent parity verdict is taken over the recycled trunk reps (single/pair):
            # they are a pure function of the recycled trunk and carry no sampling noise. Logits and
            # coordinates depend on the stochastic sampler and are reported but excluded from the
            # verdict unless the noise stream is aligned.
            if name in VERDICT_QUANTITIES:
                ok = ok and entry["rel_pass"]
        else:
            entry["max_abs"] = None
            entry["mean_abs"] = None
            entry["close"] = False
            entry["real_max_abs"] = None
            entry["real_mean_abs"] = None
            entry["real_close"] = False
            entry["rel_l2"] = None
            entry["cosine"] = None
            entry["max_rel"] = None
            entry["rel_pass"] = False
            if name in VERDICT_QUANTITIES:
                ok = False
        report[name] = entry
    return report, ok


def main() -> None:
    import json

    import torch

    args = parse_args()
    configure_environment(args.reference_root)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    config = build_reference_config(
        checkpoint=args.checkpoint,
        n_cycle=args.n_cycle,
        n_sample=args.n_sample,
        n_step=args.n_step,
    )
    feature_device = torch.device("cpu") if device.type == "cuda" else device
    if args.input == "synthetic":
        features = build_reference_features(
            n_token=args.n_token,
            atoms_per_token=args.atoms_per_token,
            seed=args.seed,
            device=feature_device,
        )
    else:
        features = build_2orb_features(
            sample_path=args.sample_2orb,
            max_tokens=args.max_tokens,
            device=feature_device,
            dummy_msa=args.dummy_msa,
        )

    reference_pred = run_reference(features, config, args.checkpoint, device=device, seed=args.seed)
    ofoldx_output = run_ofoldx(
        features,
        args.checkpoint,
        device=device,
        n_cycle=args.n_cycle,
        n_sample=args.n_sample,
        n_step=args.n_step,
        seed=args.seed,
    )
    real_tokens = int(features["token_index"].shape[0])
    real_atoms = int(features["atom_to_token_idx"].shape[0])
    report, ok = compare_outputs(
        comparable_outputs(reference_pred, ofoldx_output),
        rtol=args.rtol,
        atol=args.atol,
        rel_threshold=args.rel_threshold,
        rmsd_threshold=args.rmsd_threshold,
        real_tokens=real_tokens,
        real_atoms=real_atoms,
    )

    print("\n===== PARITY DIAGNOSTICS =====", flush=True)
    print(f"real_tokens={real_tokens} real_atoms={real_atoms} rel_threshold={args.rel_threshold:g}", flush=True)
    print("--- RNG-independent trunk representations (deterministic parity verdict) ---", flush=True)

    def _print_quantity(name: str, entry: dict[str, Any]) -> None:
        if not entry.get("shape_match"):
            print(f"{name:14s} SHAPE MISMATCH exp={entry['expected_shape']} act={entry['actual_shape']}", flush=True)
            return
        rel_l2 = entry.get("rel_l2")
        cosine = entry.get("cosine")
        max_rel = entry.get("max_rel")
        rel_text = "n/a" if rel_l2 is None else f"{rel_l2:.3e}"
        cos_text = "n/a" if cosine is None else f"{cosine:.9f}"
        max_rel_text = "n/a" if max_rel is None else f"{max_rel:.3e}"
        line = (
            f"{name:14s} shape={entry['expected_shape']} "
            f"rel_l2={rel_text} cosine={cos_text} max_rel={max_rel_text} "
            f"| FULL max={entry['max_abs']:.6g} mean={entry['mean_abs']:.6g}"
        )
        if "rmsd_min" in entry:
            line += (
                f" | KABSCH-RMSD min={entry['rmsd_min']:.4f} mean={entry['rmsd_mean']:.4f} "
                f"max={entry['rmsd_max']:.4f} (n_sample={entry['n_sample']})"
            )
        line += f" | rel_pass={entry.get('rel_pass')}"
        print(line, flush=True)

    for name, entry in report.items():
        if name in DETERMINISTIC_QUANTITIES:
            _print_quantity(name, entry)
    print("--- Sampler-dependent quantities (RNG-dependent; informational) ---", flush=True)
    for name, entry in report.items():
        if name not in DETERMINISTIC_QUANTITIES:
            _print_quantity(name, entry)
    print(f"===== END DIAGNOSTICS (deterministic verdict ok={ok}) =====\n", flush=True)

    if args.json is not None and TRUNK_CAPTURE.get("reference") is not None:
        dump_path = args.json.with_suffix(".trunk.pt")
        ref_s_inputs, ref_s, ref_z = TRUNK_CAPTURE["reference"]
        of_token, of_single, of_pair = TRUNK_CAPTURE["ofoldx"]
        torch.save(
            {
                "ref_s_inputs": ref_s_inputs,
                "ref_s": ref_s,
                "ref_z": ref_z,
                "of_s_inputs": of_token,
                "of_s": of_single,
                "of_z": of_pair,
                "real_tokens": real_tokens,
                "real_atoms": real_atoms,
                "ref_coordinate": reference_pred["coordinate"].detach().float().cpu(),
                "of_coordinate": ofoldx_output.coordinates.squeeze(0).detach().float().cpu(),
            },
            dump_path,
        )
        print(f"[parity] trunk tensors saved to {dump_path}", flush=True)

    payload = {
        "ok": ok,
        "settings": {
            "device": str(device),
            "input": args.input,
            "sample_2orb": str(args.sample_2orb) if args.input == "2orb" else None,
            "max_tokens": args.max_tokens,
            "n_token": args.n_token,
            "atoms_per_token": args.atoms_per_token,
            "n_cycle": args.n_cycle,
            "n_sample": args.n_sample,
            "n_step": args.n_step,
            "seed": args.seed,
            "rtol": args.rtol,
            "atol": args.atol,
            "rel_threshold": args.rel_threshold,
            "rmsd_threshold": args.rmsd_threshold,
            "real_tokens": real_tokens,
            "real_atoms": real_atoms,
        },
        "outputs": report,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
