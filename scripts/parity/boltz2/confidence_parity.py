"""Boltz-2 confidence forward numerical-parity harness (pLDDT / PAE / PDE / resolved logits)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from scripts.parity.paths import reference_root, weight_path

sys.path.insert(0, str(reference_root("boltz") / "src"))

from scripts.parity.boltz2.forward_parity import B, M, N, build_feats, subset, to_device  # noqa: E402

CHECKPOINT = weight_path("boltz2_conf.ckpt")


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


def main() -> None:
    from boltz.model.modules.confidencev2 import ConfidenceModule as RefConfidence

    from ofoldx.models.boltz2 import convert
    from ofoldx.models.boltz2.model import ConfidenceHead

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Boltz-2 confidence parity requires CUDA.")

    device = torch.device("cuda")
    print(f"[device] {device}", flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    feats = to_device(build_feats(), device)
    gen = torch.Generator(device=device).manual_seed(11)
    rep_atom_index = torch.randint(0, M, (B, N), generator=gen, device=device)
    feats["token_to_rep_atom"] = rep_atom_index  # OFoldX index contract
    x_pred = torch.randn(B, M, 3, generator=gen, device=device)
    pred_distogram = torch.randn(B, N, N, 64, generator=gen, device=device)

    ref = RefConfidence(
        token_s=384,
        token_z=128,
        pairformer_args={"num_blocks": 8, "num_heads": 16, "dropout": 0.25},
        num_dist_bins=64,
        token_level_confidence=True,
        max_dist=22,
        add_s_to_z_prod=True,
        add_s_input_to_s=True,
        add_z_input_to_z=True,
        maximum_bond_distance=0,
        bond_type_feature=True,
        confidence_args={"num_plddt_bins": 50, "num_pde_bins": 64, "num_pae_bins": 64, "use_separate_heads": True},
        fix_sym_check=True,
        cyclic_pos_enc=True,
        conditioning_cutoff_min=4.0,
        conditioning_cutoff_max=20.0,
    )
    ref.load_state_dict(subset(state, "confidence_module"), strict=True)
    ref.to(device).eval()

    mine = ConfidenceHead(operator_backend="eager")
    converted = {
        new_key: new_value
        for key, value in subset(state, "confidence_module").items()
        for new_key, new_value in convert.convert_confidence_head(key, value)
    }
    mine.load_state_dict(converted, strict=True)
    mine.to(device).eval()

    s_inputs = torch.randn(B, N, 384, generator=gen, device=device)
    s = torch.randn(B, N, 384, generator=gen, device=device)
    z = torch.randn(B, N, N, 128, generator=gen, device=device)

    # The reference module consumes token_bonds with a trailing singleton; my model unsqueezes (B,N,N).
    # The reference still uses the dense one-hot selection matrix; OFoldX uses the index (built above).
    ref_feats = {
        **feats,
        "token_bonds": feats["token_bonds"].unsqueeze(-1),
        "token_to_rep_atom": torch.nn.functional.one_hot(rep_atom_index, M).float(),
    }
    with torch.no_grad():
        ref_out = ref(s_inputs, s, z, x_pred, ref_feats, pred_distogram, multiplicity=1)
        mine_out = mine(
            s_inputs,
            s,
            z,
            x_pred,
            residue_index=feats["residue_index"],
            chain_index=feats["asym_id"],
            entity_index=feats["entity_id"],
            symmetry_index=feats["sym_id"],
            token_index=feats["token_index"],
            token_bonds=feats["token_bonds"],
            type_bonds=feats["type_bonds"],
            contact_conditioning=feats["contact_conditioning"],
            contact_threshold=feats["contact_threshold"],
            token_to_rep_atom=rep_atom_index,
            token_mask=feats["token_pad_mask"],
            cyclic_period=feats["cyclic_period"],
            multiplicity=1,
        )

    results = [
        compare("pae_logits", mine_out["pae_logits"], ref_out["pae_logits"]),
        compare("pde_logits", mine_out["pde_logits"], ref_out["pde_logits"]),
        compare("plddt_logits", mine_out["plddt_logits"], ref_out["plddt_logits"]),
        compare("resolved_logits", mine_out["resolved_logits"], ref_out["resolved_logits"]),
    ]
    raise SystemExit(0 if all(results) else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
