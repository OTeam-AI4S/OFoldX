"""Numeric parity for the BoltzGen STRUCTURE-diffusion denoiser + reverse-EDM sampler.

The released ``folding_model_parity`` only checks a strict state-dict load, and the ifold parity
covers the sequence decoder; the structure diffusion score network (``diffusion_conditioning`` +
``structure_module.score_model``) and the reverse-EDM sampler (``AtomDiffusion.sample``) had no
numeric parity. This script builds that coverage against the official BoltzGen checkout
(``OFOLDX_REFERENCE_BOLTZGEN``), loading the *same* checkpoint tensors into both stacks.

Two comparisons, both confound-controlled:

1. Denoiser-forward (fixed noise): identical trunk/atom conditioning + identical noised coords and
   sigma are fed to ``preconditioned_network_forward`` on each side; the denoised coords are compared.
   No RNG at all (deterministic given inputs).

2. Reverse-EDM sampler: the full ``AtomDiffusion.sample`` reverse loop is driven on both sides. The
   only stochastic primitives (initial noise, per-step coordinate augmentation, per-step churn noise)
   are monkeypatched to serve *identical* scripted tensors to both implementations, so the two sample
   loops are numerically identical by construction *iff* the deterministic step math (schedule, gammas,
   t_hat inflation, EDM precondition, rigid alignment, derivative, Euler update) matches. This removes
   the RNG confound that would otherwise dominate (the two stacks draw rotations with different
   algorithms), isolating the sampler arithmetic.

Everything runs in float32 on one GPU with torch backends to keep attention reduction order comparable.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch

from scripts.parity.boltzgen import reference
from scripts.parity.paths import boltzgen_checkpoint

CKPT = boltzgen_checkpoint("boltzgen1_diverse.ckpt")

# Model dims (boltzgen.yaml `model:` block for boltzgen1_diverse).
TOKEN_S = 384
TOKEN_Z = 128
ATOM_S = 128
ATOM_Z = 16
ATOM_FEATURE_DIM = 388
REF_ELEMENT_DIM = 128  # 388 = ref_pos(3) + ref_charge(1) + ref_element(128) + ref_name_chars(4*64)
WINDOW_Q = 32
WINDOW_K = 128

# score_model_args / diffusion_process_args (design.yaml overrides for reverse sampling).
SCORE_MODEL_ARGS = {
    "token_s": TOKEN_S,
    "atom_s": ATOM_S,
    "atoms_per_window_queries": WINDOW_Q,
    "atoms_per_window_keys": WINDOW_K,
    "predict_res_type": False,
    "sigma_data": 16,
    "dim_fourier": 256,
    "atom_encoder_depth": 3,
    "atom_encoder_heads": 4,
    "token_layers": 1,
    "token_transformer_depth": 24,
    "token_transformer_heads": 16,
    "diffusion_pairformer_args": {"num_blocks": 0, "num_heads": 2, "dropout": 0, "use_s_to_z": False},
    "atom_decoder_depth": 3,
    "atom_decoder_heads": 4,
    "conditioning_transition_layers": 2,
    "transformer_post_ln": False,
    "activation_checkpointing": False,
}
DIFFUSION_PROCESS_ARGS = {
    "sigma_min": 0.0004,
    "sigma_max": 160.0,
    "sigma_data": 16.0,
    "rho": 7,
    "P_mean": -1.2,
    "P_std": 1.5,
    "gamma_0": 0.8,
    "gamma_min": 1.0,
    "noise_scale": 1.0,
    "step_scale": 1.0,
    "mse_rotational_alignment": True,
    "coordinate_augmentation": True,
    "alignment_reverse_diff": True,
    "synchronize_sigmas": False,
    "sampling_schedule": "dilated",
    "time_dilation": 2.667,
    "time_dilation_start": 0.6,
    "time_dilation_end": 0.8,
}
STEP_SCALE = 1.8
NOISE_SCALE = 0.95


def _rand_rotation(gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """A single (1, 3, 3) proper-ish rotation via QR of a seeded random matrix."""
    a = torch.randn(3, 3, generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # fix QR sign ambiguity
    return q.unsqueeze(0)


def build_inputs(num_tokens: int, num_atoms: int, device: torch.device, seed: int = 0):
    """Build matched synthetic trunk outputs + atom feats consumed identically by both stacks."""
    assert num_atoms % WINDOW_Q == 0, "num_atoms must be a multiple of the atom window (32)"
    gen = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    # Trunk outputs (arbitrary but shared, correct dims).
    s_trunk = rn(1, num_tokens, TOKEN_S)
    s_inputs = rn(1, num_tokens, TOKEN_S)
    z_trunk = rn(1, num_tokens, num_tokens, TOKEN_Z)
    rpe = rn(1, num_tokens, num_tokens, TOKEN_Z)

    # Atom reference features.
    ref_pos = rn(1, num_atoms, 3) * 5.0
    ref_charge = rn(1, num_atoms)
    ref_element = rn(1, num_atoms, REF_ELEMENT_DIM)
    ref_atom_name_chars = rn(1, num_atoms, 4, 64)

    # Contiguous atom->token assignment covering every token (monotone, each token >=1 atom).
    base = torch.arange(num_atoms, device=device) * num_tokens // num_atoms  # in [0, num_tokens)
    atom_to_token_idx = base.long().unsqueeze(0)  # (1, A)
    atom_to_token_onehot = torch.zeros(1, num_atoms, num_tokens, device=device)
    atom_to_token_onehot[0, torch.arange(num_atoms, device=device), atom_to_token_idx[0]] = 1.0

    atom_mask = torch.ones(1, num_atoms, device=device)
    token_mask = torch.ones(1, num_tokens, device=device)
    ref_space_uid = atom_to_token_idx.clone()  # atoms in same token share a reference-space id

    # ofoldx packs the 388-d atom feature exactly as reference cats it in AtomEncoder.
    atom_features = torch.cat(
        [ref_pos, ref_charge.unsqueeze(-1), ref_element, ref_atom_name_chars.reshape(1, num_atoms, 4 * 64)],
        dim=-1,
    )
    assert atom_features.shape[-1] == ATOM_FEATURE_DIM

    feats_ref = {
        "ref_pos": ref_pos,
        "atom_pad_mask": atom_mask,
        "ref_space_uid": ref_space_uid,
        "ref_charge": ref_charge,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "atom_to_token": atom_to_token_onehot,
        "token_pad_mask": token_mask,
    }
    return {
        "s_trunk": s_trunk,
        "s_inputs": s_inputs,
        "z_trunk": z_trunk,
        "rpe": rpe,
        "atom_features": atom_features,
        "ref_pos": ref_pos,
        "atom_mask": atom_mask,
        "token_mask": token_mask,
        "ref_space_uid": ref_space_uid,
        "atom_to_token_idx": atom_to_token_idx,
        "feats_ref": feats_ref,
    }


def token_transformer_depth(state) -> int:
    r"""Count the score-model token-transformer blocks present in the checkpoint.

    Handles both the old nested ``token_transformer_layers.0.layers.<i>`` naming (diverse/adherence)
    and the flat ``token_transformer.layers.<i>`` naming (structuretrained-small).
    """

    import re

    indices = set()
    for key in state:
        match = re.search(r"structure_module\.score_model\.token_transformer(?:_layers\.0)?\.layers\.(\d+)\.", key)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        raise ValueError("no score_model token_transformer layers found in checkpoint")
    return max(indices) + 1


def select_ofoldx_config(state):
    r"""Pick the BoltzGen structure-design config that matches the checkpoint's realization.

    The ``boltzgen1_structuretrained_small`` artifact is a reduced (Miniformer, shallower, no
    token-distance / design conditioning) build of the same contract; every other released
    structure-design checkpoint (diverse / adherence) uses the default full config. The presence of a
    ``token_distance_module`` is the clean discriminator (small omits it entirely).
    """

    from ofoldx.models.boltzgen.config import BoltzGenConfig

    is_small = not any(key.startswith("token_distance_module.") for key in state)
    return BoltzGenConfig.structuretrained_small() if is_small else BoltzGenConfig()


def build_ofoldx(device: torch.device, checkpoint: Path = CKPT, config=None):
    from ofoldx.models.boltzgen.convert import convert_structure_design
    from ofoldx.models.boltzgen.model import BoltzGenModel

    ck = reference.load_checkpoint(checkpoint)
    if config is None:
        config = select_ofoldx_config(ck["state_dict"])
    model = BoltzGenModel(config).set_backend("torch")
    converted = convert_structure_design(ck["state_dict"])
    model.load_state_dict(converted, strict=True)
    return model.to(device).float().eval()


def build_reference(state, device: torch.device, depth: int = SCORE_MODEL_ARGS["token_transformer_depth"]):
    reference.configure_reference()
    from boltzgen.model.modules.diffusion import AtomDiffusion
    from boltzgen.model.modules.diffusion_conditioning import DiffusionConditioning

    score_model_args = dict(SCORE_MODEL_ARGS)
    score_model_args["token_transformer_depth"] = depth

    dc = DiffusionConditioning(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atom_s=ATOM_S,
        atom_z=ATOM_Z,
        atoms_per_window_queries=WINDOW_Q,
        atoms_per_window_keys=WINDOW_K,
        atom_encoder_depth=score_model_args["atom_encoder_depth"],
        atom_encoder_heads=score_model_args["atom_encoder_heads"],
        token_transformer_depth=depth,
        token_transformer_heads=score_model_args["token_transformer_heads"],
        atom_decoder_depth=score_model_args["atom_decoder_depth"],
        atom_decoder_heads=score_model_args["atom_decoder_heads"],
        atom_feature_dim=ATOM_FEATURE_DIM,
        conditioning_transition_layers=score_model_args["conditioning_transition_layers"],
    )
    ad = AtomDiffusion(score_model_args=score_model_args, **DIFFUSION_PROCESS_ARGS)

    dc_state = {
        k[len("diffusion_conditioning.") :]: v for k, v in state.items() if k.startswith("diffusion_conditioning.")
    }
    ad_state = {}
    for k, v in state.items():
        if not k.startswith("structure_module."):
            continue
        inner = k[len("structure_module.") :]
        # Checkpoint predates the token_transformer refactor (token_layers=1): the single-entry
        # `token_transformer_layers.0` ModuleList is now the flat `token_transformer` DiffusionTransformer.
        inner = inner.replace("score_model.token_transformer_layers.0.layers.", "score_model.token_transformer.layers.")
        ad_state[inner] = v
    dc.load_state_dict(dc_state, strict=True)
    ad.load_state_dict(ad_state, strict=True)
    return dc.to(device).float().eval(), ad.to(device).float().eval()


def ref_conditioning(dc, inp):
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = dc(
        inp["s_trunk"], inp["z_trunk"], inp["rpe"], inp["feats_ref"]
    )
    return {
        "q": q,
        "c": c,
        "to_keys": to_keys,
        "atom_enc_bias": atom_enc_bias,
        "atom_dec_bias": atom_dec_bias,
        "token_trans_bias": token_trans_bias,
    }


def ofoldx_conditioning(model, inp):
    return model.diffusion_conditioning(
        inp["s_trunk"],
        inp["z_trunk"],
        inp["rpe"],
        atom_features=inp["atom_features"],
        reference_positions=inp["ref_pos"],
        atom_mask=inp["atom_mask"],
        reference_space_id=inp["ref_space_uid"],
        atom_to_token=inp["atom_to_token_idx"],
    )


# ---------------------------------------------------------------------------
# Deterministic RNG scripting for the sampler comparison.
# ---------------------------------------------------------------------------
class ScriptedRandom:
    """Serves identical scripted tensors to both sample() loops, indexed by call order/shape.

    coord-shaped (1, A, 3) torch.randn calls -> [x_T_unit, eps_unit[0], eps_unit[1], ...]
    (1, 1, 3) torch.randn calls (ofoldx per-step translation) -> tr[i]
    augmentation calls (ofoldx uniform_random_rotation / ref compute_random_augmentation) -> R[i] (+ tr[i])
    Any other shape falls through to the real torch.randn.
    """

    def __init__(self, num_atoms, steps, device, seed=1234):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.coord_shape = (1, num_atoms, 3)
        self.tr_shape = (1, 1, 3)
        self.x_t_unit = torch.randn(self.coord_shape, generator=gen, device=device, dtype=torch.float32)
        self.eps = [
            torch.randn(self.coord_shape, generator=gen, device=device, dtype=torch.float32) for _ in range(steps)
        ]
        self.tr = [torch.randn(self.tr_shape, generator=gen, device=device, dtype=torch.float32) for _ in range(steps)]
        self.R = [_rand_rotation(gen, device) for _ in range(steps)]
        self._real_randn = torch.randn
        self.device = device
        self.reset()

    def reset(self):
        self.coord_i = 0
        self.tr_i = 0
        self.rot_i = 0

    @staticmethod
    def _shape(args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
            return tuple(int(x) for x in args[0])
        return tuple(int(x) for x in args)

    def randn(self, *args, **kwargs):
        shape = self._shape(args)
        device = kwargs.get("device", self.device)
        if shape == self.coord_shape:
            out = self.x_t_unit if self.coord_i == 0 else self.eps[self.coord_i - 1]
            self.coord_i += 1
            return out.to(device)
        if shape == self.tr_shape:
            out = self.tr[self.tr_i]
            self.tr_i += 1
            return out.to(device)
        return self._real_randn(*args, **kwargs)

    def next_rotation(self):
        out = self.R[self.rot_i]
        self.rot_i += 1
        return out

    def next_augmentation(self):
        R = self.R[self.rot_i]
        tr = self.tr[self.rot_i]
        self.rot_i += 1
        return R, tr


@contextlib.contextmanager
def patched_rng(scripted, ofoldx_model_mod, ref_diffusion_mod):
    """Patch torch.randn + both augmentation entrypoints to serve scripted tensors."""
    real_randn = torch.randn
    real_uni = ofoldx_model_mod.uniform_random_rotation
    real_aug = ref_diffusion_mod.compute_random_augmentation

    def fake_uni(n, *, device=None, dtype=None, generator=None):
        return scripted.next_rotation().to(device if device is not None else scripted.device)

    def fake_aug(multiplicity, s_trans=1.0, device=None, dtype=torch.float32):
        R, tr = scripted.next_augmentation()
        dev = device if device is not None else scripted.device
        return R.to(dev), (tr * s_trans).to(dev)

    torch.randn = scripted.randn
    ofoldx_model_mod.uniform_random_rotation = fake_uni
    ref_diffusion_mod.compute_random_augmentation = fake_aug
    try:
        yield
    finally:
        torch.randn = real_randn
        ofoldx_model_mod.uniform_random_rotation = real_uni
        ref_diffusion_mod.compute_random_augmentation = real_aug


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    inp = build_inputs(args.num_tokens, args.num_atoms, device, seed=args.seed)

    ck = reference.load_checkpoint(args.checkpoint)
    state = ck["state_dict"]
    depth = token_transformer_depth(state)
    config = select_ofoldx_config(state)
    print(
        f"[config] token_transformer_depth={depth} ofoldx_config={type(config).__name__}"
        f"{'(structuretrained_small)' if config.design.use_token_distance is False else ''}"
    )
    ref_dc, ref_ad = build_reference(state, device, depth)
    model = build_ofoldx(device, args.checkpoint, config)

    ref_cond = ref_conditioning(ref_dc, inp)
    ado_cond = ofoldx_conditioning(model, inp)

    # ---- 1. Denoiser-forward parity (fixed noise) --------------------------------
    ngen = torch.Generator(device=device).manual_seed(99)
    noised = torch.randn(1, args.num_atoms, 3, generator=ngen, device=device, dtype=torch.float32) * 12.0
    sigma = torch.full((1,), 9.3, device=device, dtype=torch.float32)

    ado_denoised = model.structure_module.preconditioned_network_forward(
        noised,
        sigma,
        multiplicity=1,
        atom_mask=inp["atom_mask"],
        single_inputs=inp["s_inputs"],
        single_trunk=inp["s_trunk"],
        conditioning=ado_cond,
        atom_to_token=inp["atom_to_token_idx"],
        token_mask=inp["token_mask"],
    )
    ref_denoised, _ = ref_ad.preconditioned_network_forward(
        noised,
        sigma,
        training=False,
        network_condition_kwargs={
            "multiplicity": 1,
            "s_inputs": inp["s_inputs"],
            "s_trunk": inp["s_trunk"],
            "feats": inp["feats_ref"],
            "diffusion_conditioning": ref_cond,
        },
    )
    fwd_abs = (ado_denoised - ref_denoised).abs().max().item()
    fwd_rel = fwd_abs / max(ref_denoised.abs().max().item(), 1e-9)
    print(
        f"[denoiser-forward] max_abs={fwd_abs:.3e} rel={fwd_rel:.3e} "
        f"ref_scale={ref_denoised.abs().max().item():.3e}"
    )

    # ---- 2. Reverse-EDM sampler parity -------------------------------------------
    import boltzgen.model.modules.diffusion as ref_diffusion_mod

    import ofoldx.models.boltzgen.model as ofoldx_model_mod

    scripted = ScriptedRandom(args.num_atoms, args.steps, device, seed=4321)

    with patched_rng(scripted, ofoldx_model_mod, ref_diffusion_mod):
        scripted.reset()
        ado_coords = model.structure_module.sample(
            inp["atom_mask"],
            num_sampling_steps=args.steps,
            multiplicity=1,
            generator=None,
            step_scale=STEP_SCALE,
            noise_scale=NOISE_SCALE,
            single_inputs=inp["s_inputs"],
            single_trunk=inp["s_trunk"],
            conditioning=ado_cond,
            atom_to_token=inp["atom_to_token_idx"],
            token_mask=inp["token_mask"],
        )
        ado_counts = (scripted.coord_i, scripted.tr_i, scripted.rot_i)

        scripted.reset()
        ref_out = ref_ad.sample(
            inp["atom_mask"],
            num_sampling_steps=args.steps,
            multiplicity=1,
            step_scale=STEP_SCALE,
            noise_scale=NOISE_SCALE,
            s_inputs=inp["s_inputs"],
            s_trunk=inp["s_trunk"],
            feats=inp["feats_ref"],
            diffusion_conditioning=ref_cond,
        )
        ref_counts = (scripted.coord_i, scripted.tr_i, scripted.rot_i)

    ref_coords = ref_out["sample_atom_coords"]
    smp_abs = (ado_coords - ref_coords).abs().max().item()
    smp_rel = smp_abs / max(ref_coords.abs().max().item(), 1e-9)
    print(
        f"[reverse-sampler] steps={args.steps} max_abs={smp_abs:.3e} rel={smp_rel:.3e} "
        f"ref_scale={ref_coords.abs().max().item():.3e}"
    )
    print(f"  rng-draw counts (coord, tr, rot): ofoldx={ado_counts} ref={ref_counts}")
    # ofoldx draws an extra per-step translation stream; ref draws tr inside augmentation.
    assert ado_counts[0] == args.steps + 1, "ofoldx coord-randn count unexpected (hidden RNG?)"
    assert ref_counts[0] == args.steps + 1, "ref coord-randn count unexpected (hidden RNG?)"
    assert ado_counts[2] == args.steps and ref_counts[2] == args.steps, "augmentation count mismatch"

    print(f"[summary] denoiser max_abs={fwd_abs:.3e} | sampler max_abs={smp_abs:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--num-atoms", type=int, default=96)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
