"""
Profile the current Protenix model core workload.

The workload mirrors the M1 parity path:

1. strict-load the Protenix v0.5 base checkpoint into OFoldX,
2. run the 4-cycle trunk on the deterministic parity dump input,
3. run distogram,
4. run the coordinate-update denoiser using the parity dump trunk tensors.

It emits PyTorch profiler Chrome traces. Open them with ``chrome://tracing`` or
Perfetto UI to inspect flame charts.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ofoldx.models.protenix import ProtenixConfig, ProtenixInput, ProtenixModel  # noqa: E402
from ofoldx.models.protenix.convert import translate_protenix_state  # noqa: E402

BENCHMARK_DATA = REPO / "data" / "benchmark"
DEFAULT_PROTENIX_BASE_CKPT_PATH = BENCHMARK_DATA / "protenix_base_default_v0.5.0.pt"
DEFAULT_PROTENIX_REFERENCE_PATH = BENCHMARK_DATA / "Protenix"
DEFAULT_PROTENIX_2ORB_SAMPLE_PATH = BENCHMARK_DATA / "2orb.pkl.gz"
PROTENIX_BASE_REF_DUMP_PATH = Path("scripts/parity/ref_dump/ref_base_v0.5.0.pt")

PROTENIX_BASE_CKPT_PATH = DEFAULT_PROTENIX_BASE_CKPT_PATH
PROTENIX_REFERENCE_PATH = DEFAULT_PROTENIX_REFERENCE_PATH
PROTENIX_2ORB_SAMPLE_PATH = DEFAULT_PROTENIX_2ORB_SAMPLE_PATH


def import_reference_protenix_module(
    module_name: str,
    *,
    reference_root: Path = DEFAULT_PROTENIX_REFERENCE_PATH,
) -> ModuleType:
    """
    Import ``module_name`` from the reference Protenix checkout.
    """

    reference_root = reference_root.resolve()
    reference_path = str(reference_root)
    sys.path[:] = [path for path in sys.path if not _is_reference_path(path, reference_root)]
    sys.path.insert(0, reference_path)
    _clear_protenix_modules()
    module = importlib.import_module(module_name)
    _assert_reference_module(module, reference_root)
    return module


def _is_reference_path(path: str, reference_root: Path) -> bool:
    try:
        return Path(path).resolve() == reference_root
    except OSError:
        return path == str(reference_root)


def _clear_protenix_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "protenix" or module_name.startswith("protenix."):
            del sys.modules[module_name]


def _assert_reference_module(module: ModuleType, reference_root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ImportError(f"{module.__name__} has no __file__; expected reference Protenix module")
    try:
        Path(module_file).resolve().relative_to(reference_root)
    except ValueError as error:
        raise ImportError(
            f"{module.__name__} imported from {module_file}, expected it under {reference_root}"
        ) from error


def load_protenix_base_checkpoint_state(checkpoint_path: Path = DEFAULT_PROTENIX_BASE_CKPT_PATH) -> dict[str, Tensor]:
    """
    Load the v0.5 base checkpoint and strip the DDP prefix.
    """

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return {key.removeprefix("module."): value for key, value in state.items() if torch.is_tensor(value)}


def pack_reference_msa_features(input_feature_dict: dict[str, Any]) -> Tensor | None:
    if not {"msa", "has_deletion", "deletion_value"} <= input_feature_dict.keys():
        return None

    msa = torch.nn.functional.one_hot(input_feature_dict["msa"].long(), num_classes=32).float()
    has_deletion = input_feature_dict["has_deletion"].float().unsqueeze(-1)
    deletion_value = input_feature_dict["deletion_value"].float().unsqueeze(-1)
    return torch.cat([msa, has_deletion, deletion_value], dim=-1)


def model_input_from_reference_dump(input_feature_dict: dict[str, Any]) -> ProtenixInput:
    reference_mask = input_feature_dict["reference_mask"].bool()
    atom_to_token = input_feature_dict["atom_to_token_idx"].long()
    representative_atom_mask = input_feature_dict["distogram_rep_atom_mask"].bool()
    deletion_mean = input_feature_dict["deletion_mean"].unsqueeze(-1)
    token_features = torch.cat(
        (input_feature_dict["restype"], input_feature_dict["profile"], deletion_mean),
        dim=-1,
    )
    ref_element = input_feature_dict["ref_element"]
    atom_features = torch.cat(
        (
            reference_mask.unsqueeze(-1).to(ref_element),
            ref_element,
            input_feature_dict["ref_atom_name_chars"].flatten(start_dim=-2).to(ref_element),
        ),
        dim=-1,
    )
    msa_features = pack_reference_msa_features(input_feature_dict)

    return ProtenixInput(
        token_features=token_features.unsqueeze(0),
        token_mask=torch.ones_like(input_feature_dict["token_index"], dtype=torch.bool).unsqueeze(0),
        residue_index=input_feature_dict["residue_index"].long().unsqueeze(0),
        chain_index=input_feature_dict["asym_id"].long().unsqueeze(0),
        entity_index=input_feature_dict["entity_id"].long().unsqueeze(0),
        symmetry_index=input_feature_dict["sym_id"].long().unsqueeze(0),
        token_index=input_feature_dict["token_index"].long().unsqueeze(0),
        representative_atom_mask=representative_atom_mask,
        atom_token_index=input_feature_dict["atom_to_tokatom_idx"].long(),
        token_to_rep_atom=_token_to_rep_atom(
            atom_to_token, representative_atom_mask, token_features.shape[-2]
        ).unsqueeze(0),
        token_bonds=input_feature_dict["token_bonds"].float().unsqueeze(0),
        atom_features=atom_features.unsqueeze(0),
        atom_to_token=atom_to_token.unsqueeze(0),
        atom_mask=reference_mask.unsqueeze(0),
        reference_positions=input_feature_dict["reference_positions"].float().unsqueeze(0),
        reference_mask=reference_mask.unsqueeze(0),
        reference_charges=input_feature_dict["reference_charges"].float().unsqueeze(0).unsqueeze(-1),
        reference_space_id=input_feature_dict["reference_space_id"].long().unsqueeze(0),
        msa_features=None if msa_features is None else msa_features.unsqueeze(0),
        msa_mask=None,
        template_features=None,
    )


def diffusion_input_from_reference_dump(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    diffusion_input = {
        key: value for key, value in input_feature_dict.items() if torch.is_tensor(value) or key == "pad_info"
    }
    diffusion_input["reference_charges"] = input_feature_dict["reference_charges"].float().unsqueeze(-1)
    return diffusion_input


def _token_to_rep_atom(atom_to_token: Tensor, representative_atom_mask: Tensor, num_tokens: int) -> Tensor:
    token_to_rep = torch.zeros(num_tokens, dtype=torch.long, device=atom_to_token.device)
    rep_atom_indices = torch.nonzero(representative_atom_mask, as_tuple=False).flatten()
    if rep_atom_indices.numel():
        token_to_rep[atom_to_token[rep_atom_indices].long()] = rep_atom_indices.long()
    return token_to_rep


def denoiser_kwargs_from_batch(
    batch: ProtenixInput,
    diffusion_input: dict[str, Any],
    relative_features: Tensor,
) -> dict[str, Any]:
    """
    Adapt parity/profiling feature dumps to the explicit OFoldX denoiser API.
    """

    return {
        "atom_features": batch.atom_features[0],
        "atom_to_token": batch.atom_to_token[0],
        "reference_positions": batch.reference_positions[0],
        "reference_charges": batch.reference_charges[0],
        "atom_mask": batch.atom_mask[0].to(dtype=batch.atom_features.dtype),
        "relative_features": relative_features[0],
        "d_lm": diffusion_input.get("d_lm"),
        "v_lm": diffusion_input.get("v_lm"),
        "pad_info": diffusion_input.get("pad_info"),
    }


class ProtenixProfileWorkload(nn.Module):
    """
    Current model core profile workload.
    """

    requires_grad = False

    def __init__(
        self,
        model: ProtenixModel,
        batch: ProtenixInput,
        diffusion_input: dict[str, Any],
        noisy_coords: Tensor,
        t_hat: Tensor,
        recycling_steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.diffusion_input = diffusion_input
        self.noisy_coords = noisy_coords
        self.t_hat = t_hat
        self.recycling_steps = recycling_steps

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s_inputs, single, pair, relative_features = self.model._run_trunk(
            self.batch,
            recycling_steps=self.recycling_steps,
        )
        distogram_logits = self.model.distogram_head(pair)
        r_update = self.model.diffusion_module.predict_coordinate_update(
            noisy_coords=self.noisy_coords,
            noise_level=self.t_hat,
            **denoiser_kwargs_from_batch(self.batch, self.diffusion_input, relative_features),
            s_inputs=s_inputs[0],
            s_trunk=single[0],
            z_trunk=pair[0],
            pair_z=None,
            p_lm=None,
            c_l=None,
        )
        return single, pair, distogram_logits, r_update


class ProtenixTrunkWorkload(nn.Module):
    """
    Profile input embedding + recycling trunk.
    """

    requires_grad = False

    def __init__(self, model: ProtenixModel, batch: ProtenixInput, *, recycling_steps: int) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.recycling_steps = recycling_steps

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model._run_trunk(self.batch, recycling_steps=self.recycling_steps)


class ProtenixDiffusionWorkload(nn.Module):
    """
    Profile only the coordinate-update path with real trunk tensors.
    """

    requires_grad = False

    def __init__(
        self,
        model: ProtenixModel,
        batch: ProtenixInput,
        diffusion_input: dict[str, Any],
        relative_features: Tensor,
        s_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        noisy_coords: Tensor,
        t_hat: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.diffusion_input = diffusion_input
        self.relative_features = relative_features
        self.s_inputs = s_inputs
        self.single = single
        self.pair = pair
        self.noisy_coords = noisy_coords
        self.t_hat = t_hat

    def forward(self) -> tuple[Tensor]:
        r_update = self.model.diffusion_module.predict_coordinate_update(
            noisy_coords=self.noisy_coords,
            noise_level=self.t_hat,
            **denoiser_kwargs_from_batch(self.batch, self.diffusion_input, self.relative_features),
            s_inputs=self.s_inputs,
            s_trunk=self.single,
            z_trunk=self.pair,
            pair_z=None,
            p_lm=None,
            c_l=None,
        )
        return (r_update,)


class ProtenixConfidenceWorkload(nn.Module):
    """
    Profile only the confidence head with real trunk/predicted-coordinate tensors.
    """

    requires_grad = False

    def __init__(
        self,
        model: ProtenixModel,
        batch: ProtenixInput,
        s_inputs: Tensor,
        single: Tensor,
        pair: Tensor,
        pred_coords: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.s_inputs = s_inputs
        self.single = single
        self.pair = pair
        self.pred_coords = pred_coords

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        confidence = self.model.confidence_head(
            self.single,
            pair=self.pair,
            token_embedding=self.s_inputs,
            predicted_coords=self.pred_coords,
            representative_atom_mask=self.model._require_representative_atom_mask(self.batch),
            atom_to_token=self.batch.atom_to_token,
            atom_token_index=self.model._require_atom_token_index(self.batch),
            mask=self.batch.token_mask[:, :, None] & self.batch.token_mask[:, None, :],
        )
        return (
            confidence.plddt_logits,
            confidence.pae_logits,
            confidence.pde_logits,
            confidence.resolved_logits,
        )


class ProtenixForwardWorkload(nn.Module):
    """
    Full current ProtenixModel forward workload.
    """

    requires_grad = False

    def __init__(
        self,
        model: ProtenixModel,
        batch: ProtenixInput,
        *,
        num_samples: int,
        recycling_steps: int,
        sampling_steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.num_samples = num_samples
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        output = self.model(
            self.batch,
            recycling_steps=self.recycling_steps,
            num_samples=self.num_samples,
            sampling_steps=self.sampling_steps,
        )
        pde_logits = output.aux["pde_logits"]
        resolved_logits = output.aux["resolved_logits"]
        return output.pred_coords, output.distogram_logits, output.plddt_logits, pde_logits, resolved_logits


class ProtenixTrainWorkload(nn.Module):
    """
    Full current ProtenixModel forward + backward workload.
    """

    requires_grad = True

    def __init__(
        self,
        model: ProtenixModel,
        batch: ProtenixInput,
        *,
        num_samples: int,
        recycling_steps: int,
        sampling_steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.num_samples = num_samples
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps

    def forward(self) -> tuple[Tensor]:
        output = self.model(
            self.batch,
            recycling_steps=self.recycling_steps,
            num_samples=self.num_samples,
            sampling_steps=self.sampling_steps,
        )
        loss = output.pred_coords.float().square().mean()
        loss = loss + output.distogram_logits.float().square().mean()
        loss = loss + output.plddt_logits.float().square().mean()
        if output.pae_logits is not None:
            loss = loss + output.pae_logits.float().square().mean()
        for tensor in output.aux.values():
            loss = loss + tensor.float().square().mean()
        # Return the loss (with grad_fn); the caller (run_once) drives backward
        # OUTSIDE forward so the same path works under DDP (which requires
        # backward on the returned output, after forward, to sync gradients).
        return (loss,)


class BackwardWorkload(nn.Module):
    """
    Wrap a tuple-output workload with synthetic loss; caller runs backward.
    """

    requires_grad = True

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self) -> tuple[Tensor]:
        outputs = self.module()
        loss = sum(t.float().square().mean() for t in outputs if torch.is_tensor(t))
        return (loss,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/profile/protenix")
    parser.add_argument("--input", choices=("parity", "2orb"), default="parity")
    parser.add_argument(
        "--workload",
        choices=("core", "trunk", "diffusion", "confidence", "forward", "train"),
        default="core",
    )
    parser.add_argument("--mode", choices=("eager", "compiled", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trace-iters", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--recycling-steps", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="Crop real pkl inputs to the first N tokens.")
    parser.add_argument("--backward", action="store_true", help="Run synthetic loss backward for segment workloads.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-cpu", action="store_true")
    parser.add_argument("--profile-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--chrome-trace", action="store_true")
    parser.add_argument("--operator-profiler", choices=("dispatch", "cuda"), default="dispatch")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PROTENIX_BASE_CKPT_PATH)
    parser.add_argument("--protenix-root", type=Path, default=DEFAULT_PROTENIX_REFERENCE_PATH)
    parser.add_argument("--sample-2orb", type=Path, default=DEFAULT_PROTENIX_2ORB_SAMPLE_PATH)
    return parser.parse_args()


def configure_data_paths(*, checkpoint: Path, protenix_root: Path, sample_2orb: Path) -> None:
    global PROTENIX_BASE_CKPT_PATH, PROTENIX_REFERENCE_PATH, PROTENIX_2ORB_SAMPLE_PATH

    PROTENIX_BASE_CKPT_PATH = checkpoint
    PROTENIX_REFERENCE_PATH = protenix_root
    PROTENIX_2ORB_SAMPLE_PATH = sample_2orb


def git_metadata() -> dict[str, Any]:
    """
    Return lightweight repository state for reproducibility.
    """

    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def build_parity_batch(device: torch.device) -> tuple[ProtenixInput, dict[str, Any], Tensor, Tensor]:
    dump = torch.load(REPO / PROTENIX_BASE_REF_DUMP_PATH, map_location="cpu", weights_only=False)
    batch = model_input_from_reference_dump(dump["input"]).to(device)
    diffusion_input = diffusion_input_from_reference_dump(dump["input"])
    r_noisy = dump["diffusion"]["r_noisy"].to(device)
    t_hat = dump["diffusion"]["t_hat"].to(device)
    return batch, diffusion_input, r_noisy, t_hat


def build_2orb_batch(
    device: torch.device,
    *,
    num_samples: int,
    max_tokens: int | None,
) -> tuple[ProtenixInput, dict[str, Any], Tensor, Tensor]:
    if not PROTENIX_2ORB_SAMPLE_PATH.is_file():
        raise FileNotFoundError(f"2orb sample is absent: {PROTENIX_2ORB_SAMPLE_PATH}")

    featurizer_module = import_reference_protenix_module(
        "protenix.data.core.featurizer", reference_root=PROTENIX_REFERENCE_PATH
    )
    utils_module = import_reference_protenix_module("protenix.data.utils", reference_root=PROTENIX_REFERENCE_PATH)
    Featurizer = featurizer_module.Featurizer
    make_dummy_feature = utils_module.make_dummy_feature

    with gzip.open(PROTENIX_2ORB_SAMPLE_PATH, "rb") as handle:
        sample = pickle.load(handle)

    feature_dict = Featurizer(
        sample["token_array"],
        sample["atom_array"],
        reference_positions_augment=False,
        lig_atom_rename=False,
    ).get_all_input_features()
    if "profile" not in feature_dict or "deletion_mean" not in feature_dict:
        feature_dict = make_dummy_feature(feature_dict, ["msa", "template"])
    if max_tokens is not None:
        feature_dict = crop_feature_dict(feature_dict, max_tokens=max_tokens)

    restype = feature_dict["restype"].float()
    profile_tensor = feature_dict["profile"].float()
    deletion_mean = feature_dict["deletion_mean"].float()
    if deletion_mean.ndim == restype.ndim - 1:
        deletion_mean = deletion_mean.unsqueeze(-1)

    reference_mask = feature_dict["reference_mask"].bool()
    ref_element = feature_dict["ref_element"].float()
    ref_atom_name_chars = feature_dict["ref_atom_name_chars"].float()
    reference_charges = feature_dict["reference_charges"].float()
    if reference_charges.ndim == reference_mask.ndim:
        reference_charges = reference_charges.unsqueeze(-1)

    msa_features = None
    if all(key in feature_dict for key in ("msa", "has_deletion", "deletion_value")):
        msa = torch.nn.functional.one_hot(feature_dict["msa"].long(), num_classes=32).float()
        msa_features = torch.cat(
            [
                msa,
                feature_dict["has_deletion"].float().unsqueeze(-1),
                feature_dict["deletion_value"].float().unsqueeze(-1),
            ],
            dim=-1,
        ).unsqueeze(0)

    atom_to_token = feature_dict["atom_to_token_idx"].long()
    representative_atom_mask = feature_dict["distogram_rep_atom_mask"].bool()
    batch = ProtenixInput(
        token_features=torch.cat((restype, profile_tensor, deletion_mean), dim=-1).unsqueeze(0),
        token_mask=torch.ones_like(feature_dict["token_index"], dtype=torch.bool).unsqueeze(0),
        residue_index=feature_dict["residue_index"].long().unsqueeze(0),
        chain_index=feature_dict["asym_id"].long().unsqueeze(0),
        entity_index=feature_dict["entity_id"].long().unsqueeze(0),
        symmetry_index=feature_dict["sym_id"].long().unsqueeze(0),
        token_index=feature_dict["token_index"].long().unsqueeze(0),
        representative_atom_mask=representative_atom_mask,
        atom_token_index=feature_dict["atom_to_tokatom_idx"].long(),
        token_to_rep_atom=_token_to_rep_atom(atom_to_token, representative_atom_mask, restype.shape[-2]).unsqueeze(0),
        token_bonds=feature_dict["token_bonds"].float().unsqueeze(0),
        atom_features=torch.cat(
            (
                reference_mask.unsqueeze(-1).to(ref_element),
                ref_element,
                ref_atom_name_chars.flatten(start_dim=-2).to(ref_element),
            ),
            dim=-1,
        ).unsqueeze(0),
        atom_to_token=atom_to_token.unsqueeze(0),
        atom_mask=reference_mask.unsqueeze(0),
        reference_positions=feature_dict["reference_positions"].float().unsqueeze(0),
        reference_mask=reference_mask.unsqueeze(0),
        reference_charges=reference_charges.unsqueeze(0),
        reference_space_id=feature_dict["reference_space_id"].long().unsqueeze(0),
        msa_features=msa_features,
        msa_mask=None,
    ).to(device)

    diffusion_input: dict[str, Any] = {
        "relp": None,
        "atom_to_token_idx": batch.atom_to_token[0],
        "reference_positions": batch.reference_positions[0],
        "reference_charges": batch.reference_charges[0],
        "reference_mask": batch.reference_mask[0].to(dtype=batch.atom_features.dtype),
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
    }
    generator = torch.Generator(device=device).manual_seed(20260602)
    r_noisy = torch.randn(
        num_samples,
        batch.atom_features.shape[-2],
        3,
        generator=generator,
        device=device,
        dtype=batch.atom_features.dtype,
    )
    t_hat = torch.full((num_samples,), 16.0, device=device, dtype=batch.atom_features.dtype)
    return batch, diffusion_input, r_noisy, t_hat


def crop_feature_dict(feature_dict: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
    """
    Crop featurized inputs to the first ``max_tokens`` tokens.
    """

    token_count = int(feature_dict["token_index"].shape[0])
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if max_tokens >= token_count:
        return feature_dict

    cropped = dict(feature_dict)
    token_slice = slice(0, max_tokens)
    atom_mask = feature_dict["atom_to_token_idx"].long() < max_tokens

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
        if key in cropped and torch.is_tensor(cropped[key]) and cropped[key].shape[:1] == (token_count,):
            cropped[key] = cropped[key][token_slice]

    for key in ("msa", "has_deletion", "deletion_value"):
        if key in cropped and torch.is_tensor(cropped[key]) and cropped[key].shape[-1] == token_count:
            cropped[key] = cropped[key][..., token_slice]

    if "token_bonds" in cropped:
        cropped["token_bonds"] = cropped["token_bonds"][token_slice, token_slice]

    for key in ("template_all_atom_mask", "template_all_atom_positions", "template_restype"):
        if key in cropped and torch.is_tensor(cropped[key]) and cropped[key].shape[1:2] == (token_count,):
            cropped[key] = cropped[key][:, token_slice]

    atom_count = int(feature_dict["atom_to_token_idx"].shape[0])
    for key, value in list(cropped.items()):
        if key == "bond_mask":
            continue
        if torch.is_tensor(value) and value.shape[:1] == (atom_count,):
            cropped[key] = value[atom_mask]
    if "bond_mask" in cropped:
        cropped["bond_mask"] = cropped["bond_mask"][atom_mask][:, atom_mask]
    return cropped


def build_workload(
    device: torch.device,
    *,
    input_name: str,
    num_samples: int,
    recycling_steps: int,
    sampling_steps: int,
    max_tokens: int | None = None,
    workload_name: str = "core",
) -> nn.Module:
    model = ProtenixModel(ProtenixConfig()).to(device)
    reference_state = load_protenix_base_checkpoint_state(PROTENIX_BASE_CKPT_PATH)
    state = translate_protenix_state(model, reference_state)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("constraint_embedder.") for key in result.missing_keys):
        raise RuntimeError(f"Protenix checkpoint load mismatch: {result}")
    model.eval()
    if input_name == "parity":
        batch, diffusion_input, r_noisy, t_hat = build_parity_batch(device)
    elif input_name == "2orb":
        batch, diffusion_input, r_noisy, t_hat = build_2orb_batch(
            device,
            num_samples=num_samples,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"unsupported input: {input_name}")

    if workload_name == "trunk":
        return ProtenixTrunkWorkload(model.eval(), batch, recycling_steps=recycling_steps).eval()
    if workload_name == "forward":
        return ProtenixForwardWorkload(
            model.eval(),
            batch,
            num_samples=num_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        ).eval()
    if workload_name == "train":
        return ProtenixTrainWorkload(
            model.train(),
            batch,
            num_samples=num_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        ).train()

    with torch.no_grad():
        d_lm, v_lm, pad_info = model._local_atom_geometry_cache(batch)
    diffusion_input["d_lm"] = d_lm[0]
    diffusion_input["v_lm"] = v_lm[0]
    diffusion_input["pad_info"] = {
        key: value[0] if torch.is_tensor(value) and value.shape[:1] == (1,) else value
        for key, value in pad_info.items()
    }
    diffusion_input = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in diffusion_input.items()
    }
    if workload_name == "core":
        return ProtenixProfileWorkload(
            model=model.eval(),
            batch=batch,
            diffusion_input=diffusion_input,
            noisy_coords=r_noisy,
            t_hat=t_hat,
            recycling_steps=recycling_steps,
        ).eval()
    if workload_name in {"diffusion", "confidence"}:
        with torch.no_grad():
            s_inputs, single, pair, relative_features = model._run_trunk(batch, recycling_steps=recycling_steps)
        if workload_name == "diffusion":
            diffusion_input = dict(diffusion_input)
            diffusion_input["relp"] = relative_features[0]
            return ProtenixDiffusionWorkload(
                model=model.eval(),
                batch=batch,
                diffusion_input=diffusion_input,
                relative_features=relative_features,
                s_inputs=s_inputs[0],
                single=single[0],
                pair=pair[0],
                noisy_coords=r_noisy,
                t_hat=t_hat,
            ).eval()
        with torch.no_grad():
            pred_coords = model._predict_coords(
                batch,
                s_inputs=s_inputs,
                single=single,
                pair=pair,
                relative_features=relative_features,
                num_samples=num_samples,
                sampling_steps=sampling_steps,
            )
        return ProtenixConfidenceWorkload(
            model=model.eval(),
            batch=batch,
            s_inputs=s_inputs,
            single=single,
            pair=pair,
            pred_coords=pred_coords,
        ).eval()
    raise ValueError(f"unsupported workload: {workload_name}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def workload_metadata(workload: nn.Module) -> dict[str, Any]:
    """
    Collect stable workload metadata for benchmark reports.
    """

    module = workload
    while hasattr(module, "module") and isinstance(module.module, nn.Module):
        module = module.module

    metadata: dict[str, Any] = {
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "requires_grad": bool(getattr(workload, "requires_grad", False)),
    }

    batch = getattr(module, "batch", None)
    if batch is not None:
        metadata.update(
            {
                "batch_size": int(batch.token_features.shape[0]),
                "tokens": int(batch.token_features.shape[-2]),
                "atoms": int(batch.atom_features.shape[-2]),
                "token_feature_size": int(batch.token_features.shape[-1]),
                "atom_feature_size": int(batch.atom_features.shape[-1]),
            }
        )
        if batch.msa_features is not None:
            metadata["msa_shape"] = list(batch.msa_features.shape)

    model = getattr(module, "model", None)
    if model is not None:
        metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        metadata["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

    if hasattr(module, "num_samples"):
        metadata["num_samples"] = int(module.num_samples)

    return metadata


def run_once(workload: nn.Module) -> None:
    # DDP does not forward the wrapped module's custom ``requires_grad`` flag, so
    # unwrap it to decide grad mode; otherwise a DDP-wrapped train workload would
    # run under no_grad and its loss would have no grad_fn.
    base = getattr(workload, "module", workload) if isinstance(workload, DistributedDataParallel) else workload
    if getattr(base, "requires_grad", False):
        for parameter in workload.parameters():
            parameter.grad = None
        with torch.enable_grad():
            outputs = workload()
        # Backward runs OUTSIDE forward so DDP's reducer (set up after forward)
        # can hook the gradient all-reduce; the workload returns the live loss.
        loss = outputs[0] if isinstance(outputs, tuple) else outputs
        loss.backward()
    else:
        with torch.no_grad():
            outputs = workload()
    # Keep outputs live until kernels are scheduled.
    if not isinstance(outputs, tuple):
        raise RuntimeError("unexpected workload output")


def benchmark(workload: nn.Module, *, warmup: int, iters: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        run_once(workload)
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iters):
        run_once(workload)
        synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    current_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    current_reserved = torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
    return {
        "iters": float(iters),
        "total_seconds": elapsed,
        "seconds_per_iter": elapsed / iters,
        "peak_memory_gib": peak_memory / 1024**3,
        "peak_memory_reserved_gib": peak_reserved / 1024**3,
        "final_memory_allocated_gib": current_allocated / 1024**3,
        "final_memory_reserved_gib": current_reserved / 1024**3,
    }


class DispatchOperatorProfiler:
    """
    Aggregate aten operator timings without using CUDA profiler APIs.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "sample_shapes": ""})

    def record(self, name: str, seconds: float, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        row = self.rows[name]
        row["calls"] += 1
        row["seconds"] += seconds
        if not row["sample_shapes"]:
            row["sample_shapes"] = summarize_shapes((args, kwargs))

    def table(self, *, limit: int = 80) -> str:
        ordered = sorted(self.rows.items(), key=lambda item: item[1]["seconds"], reverse=True)
        lines = [
            f"{'Operator':70} {'Calls':>10} {'Total ms':>12} {'Avg us':>12}  Shapes",
            "-" * 120,
        ]
        for name, row in ordered[:limit]:
            total_ms = row["seconds"] * 1000
            avg_us = row["seconds"] * 1_000_000 / max(1, row["calls"])
            lines.append(f"{name[:70]:70} {row['calls']:10d} {total_ms:12.3f} {avg_us:12.3f}  {row['sample_shapes']}")
        return "\n".join(lines)

    def jsonable(self) -> list[dict[str, Any]]:
        ordered = sorted(self.rows.items(), key=lambda item: item[1]["seconds"], reverse=True)
        return [
            {
                "operator": name,
                "calls": row["calls"],
                "total_seconds": row["seconds"],
                "avg_seconds": row["seconds"] / max(1, row["calls"]),
                "sample_shapes": row["sample_shapes"],
            }
            for name, row in ordered
        ]


def summarize_shapes(value: Any, *, limit: int = 12) -> str:
    shapes: list[str] = []

    def visit(item: Any) -> None:
        if len(shapes) >= limit:
            return
        if torch.is_tensor(item):
            shapes.append(f"{tuple(item.shape)}:{str(item.dtype).removeprefix('torch.')}")
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    suffix = " ..." if len(shapes) >= limit else ""
    return ", ".join(shapes) + suffix


def export_dispatch_profile(
    workload: nn.Module,
    *,
    label: str,
    out_dir: Path,
    trace_iters: int,
    device: torch.device,
) -> tuple[Path, str]:
    from torch.utils._python_dispatch import TorchDispatchMode

    operator_profiler = DispatchOperatorProfiler(device)

    class OperatorTimingMode(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
        ) -> Any:
            kwargs = {} if kwargs is None else kwargs
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = func(*args, **kwargs)
                end.record()
                end.synchronize()
                seconds = start.elapsed_time(end) / 1000
            else:
                started = time.perf_counter()
                output = func(*args, **kwargs)
                seconds = time.perf_counter() - started
            operator_profiler.record(str(func), seconds, args, kwargs)
            return output

    with OperatorTimingMode():
        for _ in range(trace_iters):
            run_once(workload)
    synchronize(device)

    table = operator_profiler.table()
    table_path = out_dir / f"{label}.dispatch_operators.txt"
    json_path = out_dir / f"{label}.dispatch_operators.json"
    table_path.write_text(table + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(operator_profiler.jsonable(), indent=2) + "\n", encoding="utf-8")
    return json_path, table


def export_trace(
    workload: nn.Module,
    *,
    label: str,
    out_dir: Path,
    trace_iters: int,
    device: torch.device,
    with_stack: bool,
    profile_cpu: bool,
    record_shapes: bool,
    profile_memory: bool,
    with_flops: bool,
    chrome_trace: bool,
) -> tuple[Path | None, str]:
    del profile_cpu
    trace_path = out_dir / f"{label}.trace.json"
    with torch.autograd.profiler.profile(
        use_cuda=device.type == "cuda",
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as profiler:
        for _ in range(trace_iters):
            run_once(workload)
    synchronize(device)
    if chrome_trace:
        profiler.export_chrome_trace(str(trace_path))
    else:
        trace_path = None
    operator_table = profiler.key_averages().table(
        sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
        row_limit=80,
    )
    table_path = out_dir / f"{label}.operators.txt"
    table_path.write_text(operator_table + "\n", encoding="utf-8")
    return trace_path, operator_table


def profile_mode(
    *,
    label: str,
    workload: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "workload_metadata": workload_metadata(workload)}
    if label == "compiled":
        compile_started = time.perf_counter()
        workload = torch.compile(
            workload,
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )
        run_once(workload)
        synchronize(device)
        result["compile_first_call_seconds"] = time.perf_counter() - compile_started

    result.update(benchmark(workload, warmup=args.warmup, iters=args.iters, device=device))
    if args.trace_iters > 0:
        if args.operator_profiler == "dispatch":
            operator_path, operator_table = export_dispatch_profile(
                workload,
                label=label,
                out_dir=args.out_dir,
                trace_iters=args.trace_iters,
                device=device,
            )
            result["operator_table"] = str(args.out_dir / f"{label}.dispatch_operators.txt")
            result["operator_json"] = str(operator_path)
        else:
            trace_path, operator_table = export_trace(
                workload,
                label=label,
                out_dir=args.out_dir,
                trace_iters=args.trace_iters,
                device=device,
                with_stack=args.with_stack,
                profile_cpu=args.profile_cpu,
                record_shapes=args.profile_shapes,
                profile_memory=args.profile_memory,
                with_flops=args.profile_flops,
                chrome_trace=args.chrome_trace,
            )
            if trace_path is not None:
                result["trace"] = str(trace_path)
            result["operator_table"] = str(args.out_dir / f"{label}.operators.txt")
        result["operator_table_preview"] = "\n".join(operator_table.splitlines()[:25])
    return result


def main() -> None:
    args = parse_args()
    if args.recycling_steps is None:
        args.recycling_steps = 4 if args.workload == "train" else 10
    if args.sampling_steps is None:
        args.sampling_steps = 20 if args.workload == "train" else 200
    configure_data_paths(
        checkpoint=args.checkpoint,
        protenix_root=args.protenix_root,
        sample_2orb=args.sample_2orb,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    modes = ["eager", "compiled"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        workload = build_workload(
            device,
            input_name=args.input,
            num_samples=args.num_samples,
            recycling_steps=args.recycling_steps,
            sampling_steps=args.sampling_steps,
            max_tokens=args.max_tokens,
            workload_name=args.workload,
        )
        if args.backward:
            if args.workload == "train":
                raise ValueError("--workload train already includes backward; do not also pass --backward")
            workload = BackwardWorkload(workload).train()
        result = profile_mode(label=mode, workload=workload, args=args, device=device)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "device": str(device),
        "git": git_metadata(),
        "input": args.input,
        "workload": args.workload,
        "includes_backward": args.workload == "train" or args.backward,
        "max_tokens": args.max_tokens,
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "torch": torch.__version__,
        "results": results,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
