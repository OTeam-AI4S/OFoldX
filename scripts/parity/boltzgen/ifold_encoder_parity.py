"""Forward + load parity for InverseFoldingEncoder vs the reference boltzgen InverseFoldingEncoder.

Builds a small synthetic graph feats dict, loads `inverse_folding_encoder` weights from boltzgen1_ifold.ckpt
into both the reference module (directly) and the OFoldX module (via a clean-name converter), runs both in
eval, and asserts the packed node/edge states and the graph (edge_index, valid_mask) match.
"""

from __future__ import annotations

import torch

from ofoldx.models.boltzgen import BoltzGenIFoldInput
from scripts.parity.boltzgen import reference
from scripts.parity.boltzgen.ifold_gnn_parity import ENCODER_RENAME, _ensure_checkpoint, _sub_state

PREFIX = "inverse_folding_encoder"

# clean OFoldX name -> reference name (top-level + the GNN-layer renames inside layers.<i>).
ENCODER_TOP = {
    "node_input_projection": "linear_token_to_node",
    "edge_input_projection": "linear_token_to_pair",
    "distance_smearing": "distance_gaussian_smearing",
    "layers": "encoder_layers",
}


def _ref_to_ofoldx(ref_state: dict) -> dict:
    out = {}
    for key, value in ref_state.items():
        parts = key.split(".")
        head = parts[0]
        clean_head = next((alias for alias, ref in ENCODER_TOP.items() if ref == head), head)
        if clean_head == "layers":
            # layers.<i>.<gnn-submodule>... ; rename the gnn submodule (3rd part).
            sub = parts[2]
            clean_sub = next((alias for alias, ref in ENCODER_RENAME.items() if ref == sub), sub)
            out[".".join([clean_head, parts[1], clean_sub, *parts[3:]])] = value
        else:
            out[".".join([clean_head, *parts[1:]])] = value
    return out


def _feats(num_tokens: int = 8, num_atoms: int = 32, token_s: int = 384, element_dim: int = 128):
    g = torch.Generator().manual_seed(0)
    n, m = num_tokens, num_atoms
    return {
        "token_resolved_mask": torch.ones(1, n),
        "token_pad_mask": torch.ones(1, n),
        "center_coords": torch.randn(1, n, 3, generator=g) * 10,
        "token_bonds": (torch.randn(1, n, n, 1, generator=g) > 1.0).float(),
        "type_bonds": torch.randint(0, 7, (1, n, n), generator=g),
        "token_to_bb4_atoms": torch.randint(0, m, (1, n * 4), generator=g),  # OFoldX index contract
        "coords": torch.randn(1, m, 3, generator=g) * 10,
        "feature_asym_id": torch.zeros(1, n, dtype=torch.long),
        "feature_residue_index": torch.arange(n).unsqueeze(0),
        "s_inputs": torch.randn(1, n, token_s, generator=g),
        # only needed by the reference's (discarded) node-attr path under enable_input_embedder=True:
        "ref_charge": torch.randn(1, m, generator=g),
        "ref_element": torch.randn(1, m, element_dim, generator=g),
        "atom_to_token": torch.randint(0, n, (1, m), generator=g),  # OFoldX index contract
        "atom_pad_mask": torch.ones(1, m),
        "is_standard": torch.ones(1, n),
        "mol_type": torch.zeros(1, n, dtype=torch.long),
        "modified": torch.zeros(1, n),
    }


def _batch(feats: dict) -> BoltzGenIFoldInput:
    center = feats.get("center_coords")
    coords = feats.get("coords")
    if center is None or coords is None:
        raise ValueError("center_coords and coords are required")
    batch_size, num_tokens = center.shape[:2]
    num_atoms = coords.shape[1]
    device = center.device
    token_mask = feats.get("token_pad_mask", torch.ones(batch_size, num_tokens, device=device))
    residue_index = feats.get("feature_residue_index", torch.arange(num_tokens, device=device).expand(batch_size, -1))
    chain_index = feats.get("feature_asym_id", torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device))
    atom_to_token = feats.get("atom_to_token", torch.zeros(batch_size, num_atoms, dtype=torch.long, device=device))
    token_to_bb4 = feats.get(
        "token_to_bb4_atoms", torch.zeros(batch_size, num_tokens * 4, dtype=torch.long, device=device)
    )
    residue_type = feats.get(
        "res_type_clone",
        feats.get("residue_type", torch.zeros(batch_size, num_tokens, 33, device=device)),
    )
    return BoltzGenIFoldInput(
        token_mask=token_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        atom_features=feats.get("atom_features", torch.zeros(batch_size, num_atoms, 388, device=device)),
        atom_to_token=atom_to_token,
        atom_mask=feats.get("atom_pad_mask", torch.ones(batch_size, num_atoms, device=device)),
        reference_positions=feats.get("reference_positions", coords),
        reference_space_id=feats.get("reference_space_id", torch.zeros(batch_size, num_atoms, device=device)),
        residue_type=residue_type,
        msa_profile=feats.get("msa_profile", torch.zeros(batch_size, num_tokens, 33, device=device)),
        deletion_mean=feats.get("deletion_mean", torch.zeros(batch_size, num_tokens, device=device)),
        token_bonds=feats.get("token_bonds", torch.zeros(batch_size, num_tokens, num_tokens, 1, device=device)),
        type_bonds=feats.get(
            "type_bonds", torch.zeros(batch_size, num_tokens, num_tokens, dtype=torch.long, device=device)
        ),
        chain_type=feats.get(
            "chain_type",
            feats.get("mol_type", torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device)),
        ),
        method_feature=feats.get(
            "method_feature", torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device)
        ),
        modified=feats.get("modified", torch.zeros(batch_size, num_tokens, dtype=torch.long, device=device)),
        cyclic_period=feats.get("cyclic_period", torch.zeros(batch_size, num_tokens, device=device)),
        center_coords=center,
        token_to_bb4_atoms=token_to_bb4,
        coords=coords,
        design_mask=feats.get("design_mask"),
    )


def main() -> None:
    torch.manual_seed(0)
    ckpt = reference.load_checkpoint(_ensure_checkpoint())
    state = _sub_state(ckpt["state_dict"], PREFIX)
    reference.configure_reference()
    from boltzgen.model.modules.inverse_fold import InverseFoldingEncoder as RefEncoder

    from ofoldx.models.boltzgen.model import InverseFoldingEncoder

    ref = RefEncoder(
        atom_s=128,
        atom_z=16,
        token_s=384,
        token_z=128,
        node_dim=128,
        pair_dim=128,
        hidden_dim=128,
        dropout=0.0,
        softmax_dropout=0.0,
        num_encoder_layers=6,
        num_heads=4,
        topk=30,
        enable_input_embedder=True,
    ).eval()
    ref.load_state_dict(state, strict=True)

    ado = InverseFoldingEncoder(
        single_size=384,
        node_size=128,
        edge_size=128,
        hidden_size=128,
        num_layers=6,
        num_attention_heads=4,
        k_neighbors=30,
    ).eval()
    missing, unexpected = ado.load_state_dict(_ref_to_ofoldx(state), strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

    feats = _feats()
    m = feats["coords"].shape[1]
    n = feats["s_inputs"].shape[1]
    # reference consumes the dense one-hot selection matrices; OFoldX uses the indices (built in _feats).
    ref_feats = {
        **feats,
        "token_to_bb4_atoms": torch.eye(m)[feats["token_to_bb4_atoms"]],
        "atom_to_token": torch.eye(n)[feats["atom_to_token"]],
    }
    batch = _batch(feats)
    with torch.no_grad():
        ref_edge, ref_valid, ref_s, ref_z = ref(ref_feats)
        ado_edge, ado_valid, ado_s, ado_z = ado(batch, single_inputs=feats["s_inputs"])

    assert torch.equal(ref_edge, ado_edge), "edge_index mismatch"
    assert torch.equal(ref_valid, ado_valid), "valid_mask mismatch"
    ds = (ref_s - ado_s).abs().max().item()
    dz = (ref_z - ado_z).abs().max().item()
    print(f"  encoder: edges={ref_edge.shape[1]} node={ds:.2e} edge={dz:.2e}")
    assert ds < 1e-5 and dz < 1e-5, "encoder output mismatch"
    print("[ok] InverseFoldingEncoder forward+load parity vs reference")


if __name__ == "__main__":
    main()
