r"""Compare the complete released ODesign graph with OFoldX."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# flash-attn v1->v2 compat: ODesign's ``openfold_local`` imports the pre-2.0 name
# ``flash_attn_unpadded_kvpacked_func``, renamed to ``flash_attn_varlen_kvpacked_func``
# in flash-attn >=2.0 (installed: 2.8.3). It is imported-but-not-called here because the
# parity env forces torch attention (``TRIANGLE_ATTENTION=torch``), so aliasing the name
# in memory before ODesign is imported is a pure import-resolution fix. Done in this
# writable harness rather than the read-only env site-packages.
try:  # noqa: SIM105
    import flash_attn.flash_attn_interface as _fai

    if not hasattr(_fai, "flash_attn_unpadded_kvpacked_func"):
        _fai.flash_attn_unpadded_kvpacked_func = _fai.flash_attn_varlen_kvpacked_func
except Exception:  # flash_attn absent -> ODesign uses its own attention fallback
    pass

# chanfig compat: chanfig<=0.0.116 (latest) omits ``typing.Literal`` handling in
# ``conform_annotation`` -> for ``ODesignProcessorConfig.receptor_mode: Literal["rigid",
# "flex"]`` it falls through to ``isinstance(data, typing.Literal)``, which raises
# ``TypeError: typing.Literal cannot be used with isinstance()``. This is a third-party
# library gap (no fixed release exists), patched here in the writable harness like the
# flash_attn shim above rather than editing the read-only base-env site-packages or the
# model config. Membership against the Literal args is the correct validation.
try:  # noqa: SIM105
    from typing import Literal, get_args, get_origin

    import chanfig.flat_dict as _chanfig_flat_dict
    import chanfig.utils.annotation as _chanfig_annotation

    _chanfig_conform = _chanfig_annotation.conform_annotation

    def _conform_annotation_literal(data: Any, annotation: Any) -> bool:
        if get_origin(annotation) is Literal:
            return data in get_args(annotation)
        return _chanfig_conform(data, annotation)

    _chanfig_annotation.conform_annotation = _conform_annotation_literal
    _chanfig_flat_dict.conform_annotation = _conform_annotation_literal
except Exception:  # chanfig absent or already handles Literal
    pass

from scripts.parity.paths import reference_root, repo_root, weight_path

REPO = repo_root()
REFERENCE_ROOT = reference_root("ODesign")
CHECKPOINT = weight_path("odesign_base_prot_flex.pt")


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()
    configure_environment(args.reference_root)
    device = torch.device(args.device)

    log_stage(started_at, "build_config")
    reference_config = build_reference_config(
        args.reference_root,
        n_cycle=args.n_cycle,
    )
    log_stage(started_at, "load_checkpoint")
    reference_state = load_reference_state(args.checkpoint)
    log_stage(started_at, "build_reference_model")
    reference_model = build_reference_model(reference_config, reference_state, device=device)
    log_stage(started_at, "build_ofoldx_model")
    ofoldx_model = build_ofoldx_model(reference_config, reference_state, device=device)
    log_stage(started_at, "build_inputs")
    reference_input, reference_diffusion_input, ofoldx_input = build_inputs(
        n_token=args.n_token,
        n_msa=args.n_msa,
        seed=args.seed,
        device=device,
    )

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    noisy_coordinates = torch.randn(
        1,
        args.num_samples,
        ofoldx_input.atom_features.size(-2),
        3,
        device=device,
        generator=generator,
    )
    noise_level = 16 * torch.exp(-1.2 + 1.5 * torch.randn(1, args.num_samples, device=device, generator=generator))
    log_stage(started_at, "forward_reference")
    with torch.no_grad():
        reference_representations = reference_model.get_pairformer_output(
            reference_input,
            N_cycle=args.n_cycle,
            inplace_safe=False,
            chunk_size=None,
        )
        reference_pairwise = reference_model.pairwise_head(reference_representations)
        reference_scaled_coordinates = noisy_coordinates.squeeze(0) / torch.sqrt(
            16**2 + noise_level.squeeze(0).square().unsqueeze(-1).unsqueeze(-1)
        )
        reference_coordinate_update = reference_model.diffusion_module(
            x_noisy=reference_scaled_coordinates,
            t_hat_noise_level=noise_level.squeeze(0),
            input_data=reference_diffusion_input,
            input_embedding=reference_representations,
            inplace_safe=False,
            chunk_size=None,
        )
        sigma = noise_level.squeeze(0).unsqueeze(-1).unsqueeze(-1)
        reference_coordinates = (
            16**2 / (16**2 + sigma.square()) * noisy_coordinates.squeeze(0)
            + 16 * sigma / torch.sqrt(16**2 + sigma.square()) * reference_coordinate_update
        )

        log_stage(started_at, "forward_ofoldx")
        token_embedding, single, pair, relative_features = ofoldx_model._run_trunk(
            ofoldx_input,
            recycling_steps=args.n_cycle,
        )
        ofoldx_distogram = ofoldx_model.distogram_head(pair)
        ofoldx_bond_logits = ofoldx_model.token_bond_type_head(pair)
        context = ofoldx_model._denoise_context(
            ofoldx_input,
            token_embedding,
            single,
            pair,
            relative_features,
        )
        scaled_coordinates = ofoldx_model.coordinate_denoiser.preconditioner.scale_noisy_coords(
            noisy_coordinates,
            noise_level,
        )
        ofoldx_coordinate_update = ofoldx_model.coordinate_denoiser.predict_coordinate_update(
            context.to_denoising_input(scaled_coordinates, noise_level)
        )
        ofoldx_coordinates = ofoldx_model.coordinate_denoiser.preconditioner.combine_update(
            noisy_coordinates,
            noise_level,
            ofoldx_coordinate_update,
        )

    log_stage(started_at, "compare")
    results = compare_outputs(
        reference_representations=reference_representations,
        reference_pairwise=reference_pairwise,
        token_embedding=token_embedding,
        single=single,
        pair=pair,
        ofoldx_distogram=ofoldx_distogram,
        ofoldx_bond_logits=ofoldx_bond_logits,
        reference_coordinate_update=reference_coordinate_update,
        ofoldx_coordinate_update=ofoldx_coordinate_update,
        reference_coordinates=reference_coordinates,
        ofoldx_coordinates=ofoldx_coordinates,
        atol=args.atol,
        rtol=args.rtol,
    )
    ok = all(result["ok"] for result in results.values())
    payload = {
        "ok": ok,
        "checkpoint": str(args.checkpoint),
        "reference_root": str(args.reference_root),
        "n_cycle": args.n_cycle,
        "n_token": args.n_token,
        "n_msa": args.n_msa,
        "num_samples": args.num_samples,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not ok:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-cycle", type=int, default=1)
    parser.add_argument("--n-token", type=int, default=5)
    parser.add_argument("--n-msa", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def log_stage(started_at: float, name: str) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[stage {elapsed:8.1f}s] {name}", flush=True)


def configure_environment(reference_root: Path) -> None:
    os.environ["LAYERNORM_TYPE"] = "torch"
    os.environ["TRIANGLE_ATTENTION"] = "torch"
    os.environ["TRIANGLE_MULTIPLICATIVE"] = "torch"
    os.environ["USE_DEEPSPEED_EVO_ATTENTION"] = "false"
    for path in (REPO, reference_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def build_reference_config(
    reference_root: Path,
    *,
    n_cycle: int,
) -> Any:
    from omegaconf import OmegaConf

    model = OmegaConf.load(reference_root / "configs/model/_base.yaml")
    config = OmegaConf.create(
        {
            "exp": {
                "model": OmegaConf.load(reference_root / "configs/model/_base.yaml"),
                "bond_reconstruction": True,
            },
            "model": model,
            "data": {
                "msa": {
                    "enable": True,
                    "strategy": "random",
                    "sample_cutoff": {"train": 16384, "test": 16384},
                    "min_size": {"train": 1, "test": 1},
                }
            },
            "data_condition": ["constraint_distogram"],
            "diffusion_batch_size": 1,
        }
    )
    OmegaConf.resolve(config)
    config.model.N_cycle = n_cycle
    config.model.blocks_per_ckpt = None
    config.model.use_memory_efficient_kernel = False
    config.model.use_deepspeed_evo_attention = False
    config.model.use_lma = False
    config.model.template_embedder.blocks_per_ckpt = None
    config.model.msa_module.blocks_per_ckpt = None
    config.model.pairformer.blocks_per_ckpt = None
    config.model.diffusion_module.blocks_per_ckpt = None
    return config


def load_reference_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def reference_model_state(reference_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = dict(reference_state)
    for key, value in reference_state.items():
        if key == "distogram_head" or key.startswith("distogram_head."):
            state["pairwise_head." + key] = value
        if key == "bond_type_predictor" or key.startswith("bond_type_predictor."):
            state["pairwise_head.bond_type_head" + key[len("bond_type_predictor") :]] = value
    return state


def build_reference_model(config: Any, state: dict[str, torch.Tensor], *, device: torch.device) -> Any:
    from src.model.odesign import ODesign

    model = ODesign(config)
    missing, unexpected = model.load_state_dict(reference_model_state(state), strict=False)
    print(f"reference_load missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("reference_missing_sample=" + ", ".join(missing[:10]))
    return model.to(device).eval()


def build_ofoldx_model(config: Any, state: dict[str, torch.Tensor], *, device: torch.device) -> Any:
    from ofoldx.models.odesign import ODesignConfig, ODesignModel
    from ofoldx.models.odesign.convert import odesign_reference_key, translate_reference_state

    model_config = ODesignConfig()
    model_config.recycling_steps = int(config.model.N_cycle)
    model_config.trunk.num_pairformer_blocks = int(config.model.pairformer.n_blocks)
    with torch.device("meta"):
        model = ODesignModel(model_config)
    translated = translate_reference_state(model, state, odesign_reference_key)
    model.load_state_dict(translated, strict=True, assign=True)
    # Force the pure-PyTorch operator path so the OFoldX graph is compared against the
    # torch reference on the same numerics. ODesignModel defaults to set_backend("auto"),
    # which resolves to the fused OOps Triton kernels on CUDA; those are a different
    # implementation (not bit-parity) and additionally JIT-hang under this interpreter.
    # Every sibling parity harness (protenix, boltz2) forces torch the same way, matching
    # the reference env (TRIANGLE_MULTIPLICATIVE=torch, etc.) set in configure_environment.
    return model.to(device).eval().set_backend("torch")


def build_inputs(
    *,
    n_token: int,
    n_msa: int,
    seed: int,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    from src.api.model_interface import DiffusionInput, PairFormerInput

    from ofoldx.data.molecular import MolecularSystem, PolymerSequence
    from ofoldx.models.odesign import ODesignInput, ODesignProcessor

    # Data-model migration: the ODesign processor consumes a canonical MolecularSystem /
    # Structure / local path, not a raw one-letter sequence string (a bare string is now
    # read as a file path). Build the poly-glycine system explicitly, mirroring the wired
    # nampnn / esmfold2 parity harnesses.
    system = MolecularSystem(
        {"A": PolymerSequence.from_one_letter("G" * n_token, polymer_type="protein")},
        {"A": "A"},
    )
    base = ODesignProcessor(device=device)(system)
    generator = torch.Generator(device=device).manual_seed(seed)
    n_atom = base.atom_features.shape[-2]

    residue_index = torch.arange(n_token, device=device)
    token_index = torch.arange(n_token, device=device)
    chain_index = torch.zeros(n_token, dtype=torch.long, device=device)
    entity_index = torch.zeros(n_token, dtype=torch.long, device=device)
    symmetry_index = torch.zeros(n_token, dtype=torch.long, device=device)
    atom_to_token = torch.arange(n_token, device=device).repeat_interleave(n_atom // n_token)
    reference_space_id = atom_to_token.clone()

    restype = F.one_hot(
        torch.randint(0, 35, (n_token,), generator=generator, device=device),
        num_classes=35,
    ).float()
    profile = F.one_hot(
        torch.randint(0, 32, (n_token,), generator=generator, device=device),
        num_classes=32,
    ).float()
    deletion_mean = torch.rand(n_token, generator=generator, device=device)
    hotspot = torch.rand(n_token, generator=generator, device=device)
    msa = torch.randint(0, 32, (n_msa, n_token), generator=generator, device=device)
    has_deletion = torch.randint(0, 2, (n_msa, n_token), generator=generator, device=device).float()
    deletion_value = torch.rand(n_msa, n_token, generator=generator, device=device)

    reference_positions = torch.randn(n_atom, 3, generator=generator, device=device)
    reference_charges = torch.randn(n_atom, generator=generator, device=device)
    reference_mask = torch.ones(n_atom, dtype=torch.bool, device=device)
    atom_mask = torch.ones(n_atom, dtype=torch.bool, device=device)
    ref_element = F.one_hot(
        torch.randint(0, 129, (n_atom,), generator=generator, device=device),
        num_classes=129,
    ).float()
    atom_name_chars = F.one_hot(
        torch.randint(0, 64, (n_atom, 4), generator=generator, device=device),
        num_classes=64,
    ).float()
    atom_features = torch.cat(
        (reference_mask.unsqueeze(-1), ref_element, atom_name_chars.reshape(n_atom, 4 * 64)),
        dim=-1,
    )

    token_bonds = torch.randint(0, 2, (n_token, n_token), generator=generator, device=device).float()
    token_bonds = torch.triu(token_bonds, diagonal=1)
    token_bonds = token_bonds + token_bonds.transpose(-1, -2)
    token_features = torch.cat(
        (
            torch.zeros(n_token, 384, device=device),
            restype,
            profile,
            deletion_mean.unsqueeze(-1),
            hotspot.unsqueeze(-1),
        ),
        dim=-1,
    )
    msa_features = torch.cat(
        (
            F.one_hot(msa, num_classes=32).float(),
            has_deletion.unsqueeze(-1),
            deletion_value.unsqueeze(-1),
        ),
        dim=-1,
    )
    constraint = torch.rand(n_token, n_token, 1, generator=generator, device=device) * 60

    ofoldx_input = ODesignInput(
        token_mask=torch.ones(1, n_token, dtype=torch.bool, device=device),
        residue_index=residue_index.unsqueeze(0),
        chain_index=chain_index.unsqueeze(0),
        entity_index=entity_index.unsqueeze(0),
        symmetry_index=symmetry_index.unsqueeze(0),
        token_index=token_index.unsqueeze(0),
        cyclic_period=torch.zeros(1, n_token, dtype=torch.long, device=device),
        token_features=token_features.unsqueeze(0),
        token_bonds=token_bonds.unsqueeze(0),
        atom_features=atom_features.unsqueeze(0),
        atom_to_token=atom_to_token.unsqueeze(0),
        atom_mask=atom_mask.unsqueeze(0),
        reference_positions=reference_positions.unsqueeze(0),
        reference_space_id=reference_space_id.unsqueeze(0),
        reference_mask=reference_mask.unsqueeze(0),
        reference_charges=reference_charges.unsqueeze(0).unsqueeze(-1),
        observed_positions=torch.zeros(1, n_atom, 3, device=device),
        observed_mask=torch.zeros(1, n_atom, dtype=torch.bool, device=device),
        fixed_coordinate_mask=torch.zeros(1, n_atom, dtype=torch.bool, device=device),
        constraint_distogram=constraint.unsqueeze(0),
        msa_features=msa_features.unsqueeze(0),
        msa_mask=torch.ones(1, n_msa, n_token, dtype=torch.bool, device=device),
        layout=base.layout,
    )
    reference_input = PairFormerInput(
        residue_index=residue_index,
        restype=restype,
        token_bonds=token_bonds,
        profile=profile,
        deletion_mean=deletion_mean,
        is_hotspot_residue=hotspot,
        msa=msa,
        has_deletion=has_deletion,
        deletion_value=deletion_value,
        msa_token_mask=torch.zeros(n_token, dtype=torch.bool, device=device),
        ref_pos=reference_positions,
        ref_space_uid=reference_space_id,
        ref_element=ref_element,
        ref_charge=reference_charges,
        ref_mask=reference_mask,
        ref_atom_name_chars=atom_name_chars,
        atom_to_token_idx=atom_to_token,
        asym_id=chain_index,
        entity_id=entity_index,
        sym_id=symmetry_index,
        token_index=token_index,
        constraint_feature=constraint,
        token_bond_gen_mask=torch.ones(n_token, n_token, device=device),
        is_cyclic_token=torch.zeros(n_token, dtype=torch.long, device=device),
    )
    reference_diffusion_input = DiffusionInput(
        ref_pos=reference_positions,
        ref_space_uid=reference_space_id,
        ref_element=ref_element,
        ref_mask=reference_mask,
        ref_atom_name_chars=atom_name_chars,
        ref_charge=reference_charges,
        atom_to_token_idx=atom_to_token,
        residue_index=residue_index,
        asym_id=chain_index,
        entity_id=entity_index,
        sym_id=symmetry_index,
        token_index=token_index,
        is_condition_atom=torch.zeros(n_atom, dtype=torch.bool, device=device),
        is_cyclic_token=torch.zeros(n_token, dtype=torch.long, device=device),
    )
    return reference_input, reference_diffusion_input, ofoldx_input


def compare_outputs(
    *,
    reference_representations: Any,
    reference_pairwise: Any,
    token_embedding: torch.Tensor,
    single: torch.Tensor,
    pair: torch.Tensor,
    ofoldx_distogram: torch.Tensor,
    ofoldx_bond_logits: torch.Tensor,
    reference_coordinate_update: torch.Tensor,
    ofoldx_coordinate_update: torch.Tensor,
    reference_coordinates: torch.Tensor,
    ofoldx_coordinates: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, Any]]:
    comparisons = {
        "s_inputs": (token_embedding.squeeze(0), reference_representations.s_inputs),
        "single": (single.squeeze(0), reference_representations.s),
        "pair": (pair.squeeze(0), reference_representations.z),
        "distogram": (ofoldx_distogram.squeeze(0), reference_pairwise.distogram),
        "token_bond_type_logits": (
            ofoldx_bond_logits.squeeze(0),
            reference_pairwise.token_bond_type_logits,
        ),
        "coordinate_update": (
            ofoldx_coordinate_update.squeeze(0),
            reference_coordinate_update,
        ),
        "coordinates": (
            ofoldx_coordinates.squeeze(0),
            reference_coordinates,
        ),
    }
    return {
        name: compare_tensor(actual, expected, atol=atol, rtol=rtol) for name, (actual, expected) in comparisons.items()
    }


def compare_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    diff = (actual - expected).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    max_ref = float(expected.abs().max()) if expected.numel() else 0.0
    return {
        "ok": torch.allclose(actual, expected, atol=atol, rtol=rtol),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_abs / (max_ref + 1e-9),
        "shape": list(actual.shape),
    }


if __name__ == "__main__":
    main()
