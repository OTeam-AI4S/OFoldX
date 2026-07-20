"""Boltz-2 diffusion forward numerical-parity harness.

Runs the reference DiffusionConditioning + DiffusionModule (score network) against the OFoldX
DiffusionConditioning + DiffusionScoreModel on a shared synthetic batch (random trunk reps + noisy
positions), comparing the predicted per-atom position update.

    python -m scripts.parity.boltz2.diffusion_parity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from scripts.parity.paths import reference_root, weight_path

sys.path.insert(0, str(reference_root("boltz") / "src"))

from scripts.parity.boltz2.forward_parity import B, M, N, build_feats, subset  # noqa: E402

CHECKPOINT = weight_path("boltz2_conf.ckpt")
TOKEN_S, TOKEN_Z = 384, 128


def atom_features(feats: dict) -> torch.Tensor:
    return torch.cat(
        [
            feats["ref_pos"],
            feats["ref_charge"].unsqueeze(-1),
            feats["ref_element"],
            feats["ref_atom_name_chars"].reshape(B, M, 4 * 64),
        ],
        dim=-1,
    )


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    rel = diff.max() / (b.abs().max() + 1e-9)
    ok = diff.max() < 1e-3
    print(
        f"[{'ok' if ok else 'FAIL'}] {name}: max_abs={diff.max():.3e} "
        f"mean_abs={diff.mean():.3e} max_rel={rel:.3e} shape={tuple(a.shape)}"
    )
    return ok


def to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def converted_subset(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) + 1 :]: value for key, value in state.items() if key.startswith(prefix + ".")}


def main() -> None:
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning as RefConditioning
    from boltz.model.modules.diffusionv2 import DiffusionModule as RefScoreModel

    from ofoldx.models.boltz2 import convert
    from ofoldx.models.boltz2.model import DiffusionConditioning, DiffusionScoreModel

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    converted = convert.convert_trunk(state)
    feats = to_device(build_feats(), device)

    gen = torch.Generator(device=device).manual_seed(7)
    s_trunk = torch.randn(B, N, TOKEN_S, generator=gen, device=device)
    s_inputs = torch.randn(B, N, TOKEN_S, generator=gen, device=device)
    z_trunk = torch.randn(B, N, N, TOKEN_Z, generator=gen, device=device)
    rel_pos = torch.randn(B, N, N, TOKEN_Z, generator=gen, device=device)
    r_noisy = torch.randn(B, M, 3, generator=gen, device=device)
    times = torch.randn(B, generator=gen, device=device)

    # Reference.
    ref_cond = RefConditioning(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atom_s=128,
        atom_z=16,
        atom_feature_dim=388,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=24,
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        conditioning_transition_layers=2,
    )
    ref_cond.load_state_dict(subset(state, "diffusion_conditioning"), strict=True)
    ref_cond.to(device).eval()
    ref_sm = RefScoreModel(
        token_s=TOKEN_S,
        atom_s=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=24,
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        conditioning_transition_layers=2,
    )
    ref_sm.load_state_dict(subset(state, "structure_module.score_model"), strict=True)
    ref_sm.to(device).eval()

    with torch.no_grad():
        q, c, to_keys, aeb, adb, ttb = ref_cond(s_trunk, z_trunk, rel_pos, feats)
        ref_dc = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": aeb,
            "atom_dec_bias": adb,
            "token_trans_bias": ttb,
        }
        ref_update = ref_sm(s_inputs, s_trunk, r_noisy, times, feats, ref_dc, multiplicity=1)

    # OFoldX.
    dc = DiffusionConditioning(atom_feature_size=388)
    dc.load_state_dict(converted_subset(converted, "diffusion_conditioning"), strict=True)
    dc.to(device).eval()
    sm = DiffusionScoreModel(operator_backend="eager")
    sm.load_state_dict(converted_subset(converted, "structure_module.score_model"), strict=True)
    sm.to(device).eval()

    af = atom_features(feats)
    a2t = feats["atom_to_token"].argmax(-1)  # OFoldX index contract (reference keeps the one-hot)
    with torch.no_grad():
        cond = dc(
            s_trunk,
            z_trunk,
            rel_pos,
            atom_features=af,
            reference_positions=feats["ref_pos"],
            atom_mask=feats["atom_pad_mask"],
            reference_space_id=feats["ref_space_uid"],
            atom_to_token=a2t,
        )
        mine_update = sm(
            s_inputs,
            s_trunk,
            r_noisy,
            times,
            cond,
            atom_mask=feats["atom_pad_mask"],
            atom_to_token=a2t,
            token_mask=feats["token_pad_mask"],
        )

    ok = compare("diffusion position update", mine_update, ref_update)

    # Preconditioned single denoising step (EDM c_skip/c_out/c_in/c_noise around the score model).
    from boltz.model.modules.diffusionv2 import AtomDiffusion as RefAtomDiffusion

    from ofoldx.models.boltz2.model import AtomDiffusion

    hparams = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"]
    score_model_args = {**hparams["score_model_args"], "token_s": TOKEN_S, "atom_s": 128}
    ref_ad = RefAtomDiffusion(score_model_args=score_model_args, sigma_data=16)
    ref_ad.load_state_dict(subset(state, "structure_module"), strict=False)
    ref_ad.to(device).eval()
    my_ad = AtomDiffusion(sm, sigma_data=16).to(device)

    noisy = torch.randn(B, M, 3, generator=gen, device=device)
    with torch.no_grad():
        ref_den = ref_ad.preconditioned_network_forward(
            noisy,
            5.0,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "diffusion_conditioning": ref_dc,
                "multiplicity": 1,
            },
        )
        my_den = my_ad.preconditioned_network_forward(
            noisy,
            5.0,
            multiplicity=1,
            single_inputs=s_inputs,
            single_trunk=s_trunk,
            conditioning=cond,
            atom_mask=feats["atom_pad_mask"],
            atom_to_token=a2t,
            token_mask=feats["token_pad_mask"],
        )
    ok = compare("preconditioned denoised coords", my_den, ref_den) and ok
    raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
