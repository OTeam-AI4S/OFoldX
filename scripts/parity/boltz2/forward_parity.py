"""Boltz-2 forward numerical-parity harness.

Builds a small synthetic feature batch, runs the reference Boltz-2 trunk (modules loaded from the public
checkpoint) and the OFoldX :class:`Boltz2Model` (loaded via the converter from the same checkpoint),
and compares activations. This catches op-order / masking / transpose bugs that structural load-parity
cannot.

    python -m scripts.parity.boltz2.forward_parity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from scripts.parity.paths import reference_root, weight_path

sys.path.insert(0, str(reference_root("boltz") / "src"))

CHECKPOINT = weight_path("boltz2_conf.ckpt")
RECYCLING_STEPS = 1
B, N, M, S, T = 1, 5, 64, 4, 2  # batch, tokens, atoms (2 windows), msa seqs, templates
NUM_TOKENS, NUM_ELEMENTS = 33, 128


def subset(state: dict, prefix: str) -> dict:
    return {k[len(prefix) + 1 :]: v for k, v in state.items() if k.startswith(prefix + ".")}


def build_feats() -> dict:
    g = torch.Generator().manual_seed(1)
    rand = lambda *s: torch.randn(*s, generator=g)
    randint = lambda hi, *s: torch.randint(0, hi, s, generator=g)
    onehot = lambda hi, *s: torch.nn.functional.one_hot(randint(hi, *s), hi).float()
    return {
        "res_type": onehot(NUM_TOKENS, B, N),
        "profile": rand(B, N, NUM_TOKENS).softmax(-1),
        "deletion_mean": rand(B, N).abs(),
        "ref_pos": rand(B, M, 3),
        "ref_charge": rand(B, M),
        "ref_element": onehot(NUM_ELEMENTS, B, M),
        "ref_atom_name_chars": onehot(64, B, M, 4),
        "atom_pad_mask": torch.ones(B, M),
        "ref_space_uid": randint(N, B, M).float(),
        "atom_to_token": torch.nn.functional.one_hot(randint(N, B, M), N).float(),
        "method_feature": randint(12, B, N),
        "modified": randint(2, B, N),
        "cyclic_period": torch.zeros(B, N),
        "mol_type": randint(4, B, N),
        "asym_id": randint(2, B, N),
        "entity_id": randint(2, B, N),
        "sym_id": randint(2, B, N),
        "residue_index": torch.arange(N).unsqueeze(0),
        "token_index": torch.arange(N).unsqueeze(0),
        "token_bonds": (rand(B, N, N) > 0).float(),
        "type_bonds": randint(7, B, N, N),
        "contact_conditioning": torch.nn.functional.one_hot(torch.zeros(B, N, N, dtype=torch.long), 5).float(),
        "contact_threshold": rand(B, N, N).abs() * 10,
        "token_pad_mask": torch.ones(B, N),
        "msa": randint(NUM_TOKENS, B, S, N),
        "has_deletion": (rand(B, S, N) > 0).float(),
        "deletion_value": rand(B, S, N).abs(),
        "msa_paired": (rand(B, S, N) > 0).float(),
        "msa_mask": torch.ones(B, S, N),
        "template_restype": onehot(NUM_TOKENS, B, T, N),
        "template_frame_rot": torch.eye(3).expand(B, T, N, 3, 3).contiguous(),
        "template_frame_t": rand(B, T, N, 3),
        "template_mask_frame": torch.ones(B, T, N),
        "template_cb": rand(B, T, N, 3),
        "template_ca": rand(B, T, N, 3),
        "template_mask_cb": torch.ones(B, T, N),
        "visibility_ids": randint(2, B, T, N).float(),
        "template_mask": torch.ones(B, T, N),
    }


def to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def allo_input(feats: dict):
    from ofoldx.models.boltz2 import Boltz2Input

    atom_features = torch.cat(
        [
            feats["ref_pos"],
            feats["ref_charge"].unsqueeze(-1),
            feats["ref_element"],
            feats["ref_atom_name_chars"].reshape(B, M, 4 * 64),
        ],
        dim=-1,
    )
    device = feats["token_pad_mask"].device
    return Boltz2Input(
        atom_features=atom_features,
        reference_positions=feats["ref_pos"],
        atom_mask=feats["atom_pad_mask"],
        reference_space_id=feats["ref_space_uid"],
        atom_to_token=feats["atom_to_token"].argmax(-1),  # OFoldX index contract (reference keeps the one-hot)
        representative_atom_mask=torch.zeros(B, M, dtype=torch.bool, device=device),
        atom_token_index=torch.zeros(B, M, dtype=torch.long, device=device),
        token_to_rep_atom=torch.zeros(B, N, dtype=torch.long, device=device),
        residue_type=feats["res_type"],
        msa_profile=feats["profile"],
        deletion_mean=feats["deletion_mean"],
        method_feature=feats["method_feature"],
        modified=feats["modified"],
        cyclic_period=feats["cyclic_period"],
        chain_type=feats["mol_type"],
        chain_index=feats["asym_id"],
        entity_index=feats["entity_id"],
        symmetry_index=feats["sym_id"],
        residue_index=feats["residue_index"],
        token_index=feats["token_index"],
        token_bonds=feats["token_bonds"],
        type_bonds=feats["type_bonds"],
        contact_conditioning=feats["contact_conditioning"],
        contact_threshold=feats["contact_threshold"],
        token_mask=feats["token_pad_mask"],
        msa=feats["msa"],
        has_deletion=feats["has_deletion"],
        deletion_value=feats["deletion_value"],
        msa_paired=feats["msa_paired"],
        msa_mask=feats["msa_mask"],
        template_restype=feats["template_restype"],
        template_frame_rot=feats["template_frame_rot"],
        template_frame_t=feats["template_frame_t"],
        template_mask_frame=feats["template_mask_frame"],
        template_cb=feats["template_cb"],
        template_ca=feats["template_ca"],
        template_mask_cb=feats["template_mask_cb"],
        visibility_ids=feats["visibility_ids"],
        template_mask=feats["template_mask"],
    )


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    rel = diff.max() / (b.abs().max() + 1e-9)
    ok = diff.max() < 1e-3
    print(f"[{'ok' if ok else 'FAIL'}] {name}: max_abs={diff.max():.3e} max_rel={rel:.3e} shape={tuple(a.shape)}")
    return ok


def reference_trunk(state: dict, hparams: dict, feats: dict) -> dict:
    from boltz.model.layers.pairformer import PairformerModule
    from boltz.model.modules.encodersv2 import RelativePositionEncoder
    from boltz.model.modules.trunkv2 import (
        ContactConditioning,
        DistogramModule,
        InputEmbedder,
        MSAModule,
        TemplateV2Module,
    )
    from torch import nn

    def build(module, prefix):
        module.load_state_dict(subset(state, prefix), strict=True)
        return module.to(device=feats["token_pad_mask"].device).eval()

    input_embedder = build(
        InputEmbedder(
            atom_s=128,
            atom_z=16,
            token_s=384,
            token_z=128,
            atoms_per_window_queries=32,
            atoms_per_window_keys=128,
            atom_feature_dim=388,
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            add_mol_type_feat=True,
            add_method_conditioning=True,
            add_modified_flag=True,
            add_cyclic_flag=True,
        ),
        "input_embedder",
    )
    s_init = build(nn.Linear(384, 384, bias=False), "s_init")
    z_init_1 = build(nn.Linear(384, 128, bias=False), "z_init_1")
    z_init_2 = build(nn.Linear(384, 128, bias=False), "z_init_2")
    rel_pos = build(RelativePositionEncoder(128, fix_sym_check=True, cyclic_pos_enc=True), "rel_pos")
    token_bonds = build(nn.Linear(1, 128, bias=False), "token_bonds")
    token_bonds_type = build(nn.Embedding(7, 128), "token_bonds_type")
    contact = build(ContactConditioning(128, cutoff_min=4.0, cutoff_max=20.0), "contact_conditioning")
    s_norm = build(nn.LayerNorm(384), "s_norm")
    z_norm = build(nn.LayerNorm(128), "z_norm")
    s_recycle = build(nn.Linear(384, 384, bias=False), "s_recycle")
    z_recycle = build(nn.Linear(128, 128, bias=False), "z_recycle")
    template = build(TemplateV2Module(128, template_dim=64, template_blocks=2), "template_module")
    msa = build(
        MSAModule(msa_s=64, token_z=128, token_s=384, msa_blocks=4, msa_dropout=0.15, z_dropout=0.25), "msa_module"
    )
    pairformer = build(
        PairformerModule(384, 128, num_blocks=64, num_heads=16, dropout=0.25, v2=True), "pairformer_module"
    )
    distogram = build(DistogramModule(128, 64), "distogram_module")

    with torch.no_grad():
        s_inputs = input_embedder(feats)
        s_i = s_init(s_inputs)
        z_i = z_init_1(s_inputs)[:, :, None] + z_init_2(s_inputs)[:, None, :]
        z_i = z_i + rel_pos(feats)
        z_i = z_i + token_bonds(feats["token_bonds"].float().unsqueeze(-1))
        z_i = z_i + token_bonds_type(feats["type_bonds"].long())
        z_i = z_i + contact(feats)
        s = torch.zeros_like(s_i)
        z = torch.zeros_like(z_i)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        for _ in range(RECYCLING_STEPS + 1):
            s = s_i + s_recycle(s_norm(s))
            z = z_i + z_recycle(z_norm(z))
            z = z + template(z, feats, pair_mask)
            z = z + msa(z, s_inputs, feats)
            s, z = pairformer(s, z, mask=mask, pair_mask=pair_mask)
        pdistogram = distogram(z)
    return {"s_inputs": s_inputs, "s_init": s_i, "z_init": z_i, "single": s, "pair": z, "distogram": pdistogram}


def main() -> None:
    from ofoldx.models.boltz2 import Boltz2Config, convert
    from ofoldx.models.boltz2.model import Boltz2Model

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Boltz-2 forward parity requires CUDA.")

    device = torch.device("cuda")
    print(f"[device] {device}", flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    hparams = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"]
    feats = to_device(build_feats(), device)

    ref = reference_trunk(state, hparams, feats)

    config = Boltz2Config()
    model = Boltz2Model(config).set_backend("torch").to(device)
    model.load_state_dict(convert.convert_trunk(state), strict=True)
    model.eval()
    with torch.no_grad():
        out = model._trunk(allo_input(feats), recycling_steps=RECYCLING_STEPS)

    results = [
        compare("s_inputs", out.single_inputs, ref["s_inputs"]),
        compare("single (post-trunk)", out.single, ref["single"]),
        compare("pair (post-trunk)", out.pair, ref["pair"]),
        compare("distogram", out.distogram_logits, ref["distogram"].squeeze(-2)),
    ]
    raise SystemExit(0 if all(results) else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
