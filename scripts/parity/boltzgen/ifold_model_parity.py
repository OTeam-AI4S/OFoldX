"""Whole-checkpoint load + end-to-end forward parity for the BoltzGen sequence-design model.

Asserts (a) the full boltzgen1_ifold.ckpt strict-loads into BoltzGenIFoldModel via
convert_inverse_folding (0 missing/unexpected, 2 tied duplicates dropped), and (b) the composed forward
(input embedder -> encoder -> decoder) matches the reference modules chained the same way, on a small graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ofoldx.models.boltzgen import BoltzGenIFoldConfig, BoltzGenIFoldModel
from ofoldx.models.boltzgen.convert import convert_inverse_folding
from scripts.parity.boltzgen import reference
from scripts.parity.boltzgen.ifold_encoder_parity import _batch, _feats
from scripts.parity.boltzgen.ifold_gnn_parity import CKPT_DEFAULT, _ensure_checkpoint

INPUT_EMBEDDER_KEYS = (
    "atom_features",
    "reference_positions",
    "atom_mask",
    "reference_space_id",
    "atom_to_token",
    "residue_type",
    "msa_profile",
    "deletion_mean",
    "method_feature",
    "modified",
    "cyclic_period",
    "chain_type",
)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    ckpt = reference.load_checkpoint(_ensure_checkpoint(args.checkpoint))
    state = ckpt["state_dict"]

    config = BoltzGenIFoldConfig()
    config.ifold.num_encoder_layers = 6
    config.ifold.num_decoder_layers = 3
    model = BoltzGenIFoldModel(config).set_backend("torch").eval()
    converted = convert_inverse_folding(state)
    missing, unexpected = model.load_state_dict(converted, strict=True)
    dropped = len(state) - len(converted)
    print(
        f"  whole-ckpt: {len(state)} params, {dropped} tied-dropped, "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    assert not missing and not unexpected, f"missing={missing}\nunexpected={unexpected}"

    # End-to-end forward vs the reference modules chained identically.
    reference.configure_reference()
    from boltzgen.model.modules.inverse_fold import InverseFoldingDecoder as RefDecoder
    from boltzgen.model.modules.inverse_fold import InverseFoldingEncoder as RefEncoder

    def _sub(prefix: str) -> dict:
        return {k[len(prefix) + 1 :]: v for k, v in state.items() if k.startswith(prefix + ".")}

    ref_enc = RefEncoder(
        atom_s=128, atom_z=16, token_s=384, token_z=128, num_encoder_layers=6, topk=30, enable_input_embedder=True
    ).eval()
    ref_enc.load_state_dict(_sub("inverse_folding_encoder"), strict=True)
    ref_dec = RefDecoder(atom_s=128, atom_z=16, token_s=384, token_z=128, num_decoder_layers=3, num_heads=4).eval()
    ref_dec.load_state_dict(_sub("structure_module"), strict=True)

    feats = _feats()
    n, m = feats["s_inputs"].shape[1], feats["coords"].shape[1]  # tokens, atoms (m=32 is a window multiple)
    g = torch.Generator().manual_seed(1)
    feats.update(
        {
            "res_type_clone": torch.zeros(1, n, 33),
            "atom_features": torch.randn(1, m, 388, generator=g),
            "reference_positions": torch.randn(1, m, 3, generator=g),
            "atom_mask": torch.ones(1, m),
            "reference_space_id": torch.randint(0, n, (1, m), generator=g).float(),
            "residue_type": torch.nn.functional.one_hot(torch.randint(0, 33, (1, n), generator=g), 33).float(),
            "msa_profile": torch.randn(1, n, 33, generator=g).softmax(-1),
            "deletion_mean": torch.zeros(1, n),
            "method_feature": torch.zeros(1, n, dtype=torch.long),
            "modified": torch.zeros(1, n, dtype=torch.long),
            "cyclic_period": torch.zeros(1, n),
            "chain_type": torch.zeros(1, n, dtype=torch.long),
            "molecule_type": torch.zeros(1, n, dtype=torch.long),
        }
    )

    with torch.no_grad():
        # OFoldX model computes s_inputs internally; feed the reference the SAME s_inputs for a fair compare.
        s_inputs = model.input_embedder(**{k: feats.get(k) for k in INPUT_EMBEDDER_KEYS})
        # OFoldX uses the atom_to_token / token_to_bb4_atoms indices; the reference consumes dense one-hots.
        ref_feats = {
            **feats,
            "s_inputs": s_inputs,
            "atom_to_token": torch.eye(n)[feats["atom_to_token"]],
            "token_to_bb4_atoms": torch.eye(m)[feats["token_to_bb4_atoms"]],
        }
        edge_index, valid_mask, node, edge = ref_enc(ref_feats)
        ref_logits = ref_dec(node, edge, edge_index, valid_mask, ref_feats)["logits"]
        batch = _batch(feats)
        edge_index, valid_mask, node, edge = model.encoder(batch, single_inputs=s_inputs)
        ado_logits = model.decoder(node, edge, edge_index, valid_mask, batch)

    diff = (ref_logits - ado_logits).abs().max().item()
    print(f"  forward: logits {tuple(ado_logits.shape)} max_abs={diff:.2e}")
    assert diff < 1e-5, "model forward mismatch"
    print("[ok] BoltzGenIFoldModel whole-ckpt load + forward parity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
