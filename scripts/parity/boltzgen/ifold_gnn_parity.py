"""Forward + load parity for the InverseFolding GNN layer vs the reference boltzgen MLPAttnGNN.

Loads the real `encoder_layers.0` weights from boltzgen1_ifold.ckpt into (a) the reference MLPAttnGNN directly
and (b) the OFoldX GraphAttentionLayer via a clean-name converter, runs both on the same small graph in
eval, and asserts the node/edge outputs match. This validates both the converter name mapping and that the
OFoldX forward reproduces the reference. The decoder layer (MLPAttnGNNDecoder vs GraphAttentionLayerNoEdge)
is checked the same way.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from scripts.parity.boltzgen import reference
from scripts.parity.paths import boltzgen_checkpoint

CKPT_DEFAULT = boltzgen_checkpoint("boltzgen1_ifold.ckpt")


def _ensure_checkpoint(path: str | Path | None = None) -> str:
    checkpoint = Path(path) if path is not None else CKPT_DEFAULT
    if checkpoint.is_file():
        return str(checkpoint)
    raise FileNotFoundError(
        f"BoltzGen IFold checkpoint is absent: {checkpoint}. "
        "Place checkpoints under OFOLDX_WEIGHTS_ROOT, which defaults to ~/weights."
    )


# clean OFoldX name -> reference name (the only difference between the two modules).
ENCODER_RENAME = {
    "attention_weight_mlp": "attn_weight_mlp",
    "attention_value_mlp": "attn_value_mlp",
    "attention_output": "attn_output_linear",
    "node_transition": "attn_FFN",
    "edge_transition": "edge_FFN",
}


def _sub_state(state: dict, prefix: str) -> dict:
    return {k[len(prefix) + 1 :]: v for k, v in state.items() if k.startswith(prefix + ".")}


def _to_ofoldx(ref_state: dict, rename: dict) -> dict:
    out = {}
    for key, value in ref_state.items():
        head = key.split(".")[0]
        clean = next((alias for alias, ref in rename.items() if ref == head), None)
        out[key if clean is None else f"{clean}{key[len(head):]}"] = value
    return out


def _small_graph(num_nodes: int = 12, num_edges: int = 40, node_dim: int = 128, edge_dim: int = 128):
    g = torch.Generator().manual_seed(0)
    node = torch.randn(num_nodes, node_dim, generator=g)
    edge = torch.randn(num_edges, edge_dim, generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst])
    return node, edge, edge_index


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    ckpt = reference.load_checkpoint(_ensure_checkpoint(args.checkpoint))
    state = ckpt["state_dict"]
    reference.configure_reference()
    from boltzgen.model.modules.inverse_fold import MLPAttnGNN, MLPAttnGNNDecoder

    from ofoldx.models.boltzgen.model import GraphAttentionLayer, GraphAttentionLayerNoEdge

    enc_cfg = dict(
        node_dim=128,
        pair_dim=128,
        hidden_dim=128,
        dropout=0.0,
        softmax_dropout=0.0,
        transformation_scale_factor=1.0,
        num_heads=4,
    )
    # The reference decoder is sized with pair_dim=128 (attn_weight_mlp -> node*2+pair = 384) but is FED a
    # 256-dim edge state z; cat[s_dst(128), z(256)] = 384 matches. The OFoldX layer expresses this directly
    # as edge_dim=256 (weight from node+edge = 384, value from edge = 256).
    dec_cfg = dict(
        node_dim=128,
        pair_dim=128,
        hidden_dim=128,
        dropout=0.0,
        softmax_dropout=0.0,
        transformation_scale_factor=1.0,
        num_heads=4,
    )
    cases: list[tuple[Any, ...]] = [
        (
            "encoder MLPAttnGNN",
            MLPAttnGNN(**enc_cfg),
            GraphAttentionLayer(),
            "inverse_folding_encoder.encoder_layers.0",
            True,
            128,
        ),
        (
            "decoder MLPAttnGNNDecoder",
            MLPAttnGNNDecoder(**dec_cfg),
            GraphAttentionLayerNoEdge(),
            "structure_module.decoder_layers.0",
            False,
            256,
        ),
    ]

    for name, ref_model, ado_model, prefix, has_edge, edge_dim in cases:
        node, edge, edge_index = _small_graph(edge_dim=edge_dim)
        ref_state = _sub_state(state, prefix)
        ref_model.load_state_dict(ref_state, strict=True)
        missing, unexpected = ado_model.load_state_dict(_to_ofoldx(ref_state, ENCODER_RENAME), strict=False)
        assert not missing and not unexpected, f"{name}: missing={missing} unexpected={unexpected}"
        ref_model.eval()
        ado_model.eval()
        with torch.no_grad():
            ref_out = ref_model(node.clone(), edge.clone(), edge_index)
            ado_out = ado_model(node.clone(), edge.clone(), edge_index)
        if has_edge:
            ds = (ref_out[0] - ado_out[0]).abs().max().item()
            dz = (ref_out[1] - ado_out[1]).abs().max().item()
            print(f"  {name}: node={ds:.2e} edge={dz:.2e}")
            assert ds < 1e-5 and dz < 1e-5, f"{name} mismatch"
        else:
            ds = (ref_out - ado_out).abs().max().item()
            print(f"  {name}: node={ds:.2e}")
            assert ds < 1e-5, f"{name} mismatch"
    print("[ok] InverseFolding GNN layer forward+load parity vs reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
