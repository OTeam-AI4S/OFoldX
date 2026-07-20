"""Forward parity: TokenDistanceModule + v1 TemplateModule vs reference (loaded with diverse weights)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ofoldx.models.boltzgen.convert import convert_template, convert_token_distance
from ofoldx.models.boltzgen.model import BoltzGenInput, TemplateModule, TokenDistanceModule
from scripts.parity.boltzgen import reference
from scripts.parity.paths import boltzgen_checkpoint

CKPT = boltzgen_checkpoint("boltzgen1_diverse.ckpt")


def _folding_batch(feats: dict) -> BoltzGenInput:
    """Build the folding-trunk input dataclass (BoltzGenInput) these modules consume.

    TokenDistanceModule/TemplateModule live on the folding trunk and read
    ``center_coords``, ``token_to_bb4_atoms`` (index form), ``coords``,
    ``token_distance_mask`` and the ``template_*`` fields. The remaining required
    fields are unread here and filled with zeros of plausible shape.
    """

    center = feats["center_coords"]
    coords = feats["coords"]
    batch_size, num_tokens = center.shape[:2]
    num_atoms = coords.shape[1]
    return BoltzGenInput(
        token_mask=torch.ones(batch_size, num_tokens),
        residue_index=torch.arange(num_tokens).unsqueeze(0).expand(batch_size, -1),
        chain_index=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        entity_index=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        symmetry_index=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        token_index=torch.arange(num_tokens).unsqueeze(0).expand(batch_size, -1),
        atom_features=torch.zeros(batch_size, num_atoms, 388),
        atom_to_token=torch.zeros(batch_size, num_atoms, dtype=torch.long),
        atom_mask=torch.ones(batch_size, num_atoms),
        reference_positions=torch.zeros(batch_size, num_atoms, 3),
        reference_space_id=torch.zeros(batch_size, num_atoms),
        representative_atom_mask=torch.zeros(batch_size, num_atoms),
        atom_token_index=torch.zeros(batch_size, num_atoms, dtype=torch.long),
        token_to_rep_atom=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        residue_type=torch.zeros(batch_size, num_tokens, 33),
        msa_profile=torch.zeros(batch_size, num_tokens, 33),
        deletion_mean=torch.zeros(batch_size, num_tokens),
        token_bonds=torch.zeros(batch_size, num_tokens, num_tokens, 1),
        type_bonds=torch.zeros(batch_size, num_tokens, num_tokens, dtype=torch.long),
        contact_conditioning=torch.zeros(batch_size, num_tokens, num_tokens, 1),
        contact_threshold=torch.zeros(batch_size, num_tokens, num_tokens),
        chain_type=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        center_coords=center,
        token_to_bb4_atoms=feats["token_to_bb4_atoms"],
        coords=coords,
        token_distance_mask=feats["token_distance_mask"],
        template_restype=feats.get("template_restype"),
        template_frame_rot=feats.get("template_frame_rot"),
        template_frame_t=feats.get("template_frame_t"),
        template_mask_frame=feats.get("template_mask_frame"),
        template_cb=feats.get("template_cb"),
        template_ca=feats.get("template_ca"),
        template_mask_cb=feats.get("template_mask_cb"),
        template_mask=feats.get("template_mask"),
        visibility_ids=feats.get("visibility_ids"),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    ck = reference.load_checkpoint(args.checkpoint)
    sd = ck["state_dict"]
    reference.configure_reference()
    from boltzgen.model.modules.trunk import TemplateModule as RefTmpl
    from boltzgen.model.modules.trunk import TokenDistanceModule as RefTD

    B, N, M, Z = 1, 6, 24, 128
    pair_mask = torch.ones(B, N, N)
    z = torch.randn(B, N, N, Z)
    rel = torch.randn(B, N, N, Z)

    # --- TokenDistanceModule ---
    bb4_index = torch.randint(0, M, (B, N * 4))
    td_feats = {
        "center_coords": torch.randn(B, N, 3) * 10,
        "token_to_bb4_atoms": bb4_index,  # OFoldX index contract
        "coords": torch.randn(B, M, 3) * 10,
        "token_distance_mask": torch.ones(B, N, N),
    }
    ref_feats = {**td_feats, "token_to_bb4_atoms": torch.eye(M)[bb4_index]}  # reference: dense one-hot
    ref_td = RefTD(
        token_z=Z,
        token_distance_dim=64,
        token_distance_blocks=2,
        dropout=0.0,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_bins=38,
        distance_gaussian_dim=32,
        use_token_distance_feats=True,
    ).eval()
    ref_td.load_state_dict(
        {k[len("token_distance_module.") :]: v for k, v in sd.items() if k.startswith("token_distance_module.")},
        strict=True,
    )
    ado_td = TokenDistanceModule(pair_size=Z, token_distance_size=64, num_blocks=2, operator_backend="eager").eval()
    ado_td.load_state_dict(
        convert_token_distance(
            {k[len("token_distance_module.") :]: v for k, v in sd.items() if k.startswith("token_distance_module.")}
        ),
        strict=True,
    )
    batch = _folding_batch(td_feats)
    with torch.no_grad():
        d = (ref_td(z, ref_feats, pair_mask, rel) - ado_td(z, batch, rel, pair_mask)).abs().max().item()
    print(f"  token_distance forward: max_abs={d:.2e}")
    assert d < 1e-5, "token_distance forward mismatch"

    # --- v1 TemplateModule ---
    T = 2
    tmpl_feats = {
        "template_restype": torch.nn.functional.one_hot(torch.zeros(B, T, N, dtype=torch.long), 33).float(),
        "template_frame_rot": torch.randn(B, T, N, 3, 3),
        "template_frame_t": torch.randn(B, T, N, 3),
        "template_mask_frame": torch.ones(B, T, N),
        "template_cb": torch.randn(B, T, N, 3) * 10,
        "template_ca": torch.randn(B, T, N, 3) * 10,
        "template_mask_cb": torch.ones(B, T, N),
        "visibility_ids": torch.zeros(B, T, N),
        "template_mask": torch.ones(B, T, N, dtype=torch.bool),
    }
    ref_tm = RefTmpl(
        token_z=Z,
        template_dim=64,
        template_blocks=2,
        dropout=0.0,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_bins=38,
    ).eval()
    ref_tm.load_state_dict(
        {k[len("template_module.") :]: v for k, v in sd.items() if k.startswith("template_module.")}, strict=True
    )
    ado_tm = TemplateModule(pair_size=Z, template_size=64, num_blocks=2, operator_backend="eager").eval()
    ado_tm.load_state_dict(
        convert_template({k[len("template_module.") :]: v for k, v in sd.items() if k.startswith("template_module.")}),
        strict=True,
    )
    template_batch = _folding_batch({**td_feats, **tmpl_feats})
    with torch.no_grad():
        d2 = (ref_tm(z, tmpl_feats, pair_mask) - ado_tm(z, template_batch, pair_mask)).abs().max().item()
    print(f"  v1 template forward: max_abs={d2:.2e}")
    assert d2 < 1e-5, "v1 template forward mismatch"
    print("[ok] TokenDistanceModule + v1 TemplateModule forward parity vs reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
