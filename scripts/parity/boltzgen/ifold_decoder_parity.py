"""Forward + load parity for InverseFoldingDecoder vs the reference, chained after the encoder.

Runs the loaded encoder to produce the node/edge graph, then loads `structure_module` (decoder) weights into
both the reference and OFoldX decoders and compares the dense residue-type logits. res_type_clone=0 makes
the teacher-forcing random edge visibility a no-op (residue_rep = seq_to_s.bias), so the comparison is
deterministic despite the internal torch.rand draw.
"""

from __future__ import annotations

import torch

from scripts.parity.boltzgen import reference
from scripts.parity.boltzgen.ifold_encoder_parity import _batch, _feats, _ref_to_ofoldx
from scripts.parity.boltzgen.ifold_gnn_parity import ENCODER_RENAME, _ensure_checkpoint, _sub_state

# clean OFoldX name -> reference name for the decoder (structure_module).
DECODER_TOP = {
    "layers": "decoder_layers",
    "residue_type_projection": "seq_to_s",
    "residue_type_head": "predictor",
}


def _decoder_ref_to_ofoldx(ref_state: dict) -> dict:
    out = {}
    for key, value in ref_state.items():
        parts = key.split(".")
        head = parts[0]
        clean_head = next((alias for alias, ref in DECODER_TOP.items() if ref == head), head)
        if clean_head == "layers":
            sub = parts[2]
            clean_sub = next((alias for alias, ref in ENCODER_RENAME.items() if ref == sub), sub)
            out[".".join([clean_head, parts[1], clean_sub, *parts[3:]])] = value
        else:
            out[".".join([clean_head, *parts[1:]])] = value
    return out


def main() -> None:
    torch.manual_seed(0)
    ckpt = reference.load_checkpoint(_ensure_checkpoint())
    enc_state = _sub_state(ckpt["state_dict"], "inverse_folding_encoder")
    dec_state = _sub_state(ckpt["state_dict"], "structure_module")
    reference.configure_reference()
    from boltzgen.model.modules.inverse_fold import InverseFoldingDecoder as RefDecoder
    from boltzgen.model.modules.inverse_fold import InverseFoldingEncoder as RefEncoder

    from ofoldx.models.boltzgen.model import InverseFoldingDecoder, InverseFoldingEncoder

    ref_enc = RefEncoder(
        atom_s=128, atom_z=16, token_s=384, token_z=128, num_encoder_layers=6, topk=30, enable_input_embedder=True
    ).eval()
    ref_enc.load_state_dict(enc_state, strict=True)
    ado_enc = InverseFoldingEncoder(num_layers=6).eval()
    ado_enc.load_state_dict(_ref_to_ofoldx(enc_state), strict=False)

    ref_dec = RefDecoder(atom_s=128, atom_z=16, token_s=384, token_z=128, num_decoder_layers=3, num_heads=4).eval()
    ref_dec.load_state_dict(dec_state, strict=True)
    ado_dec = InverseFoldingDecoder(num_layers=3).eval()
    missing, unexpected = ado_dec.load_state_dict(_decoder_ref_to_ofoldx(dec_state), strict=False)
    assert not missing and not unexpected, f"decoder missing={missing} unexpected={unexpected}"

    feats = _feats()
    feats["res_type_clone"] = torch.zeros(1, feats["s_inputs"].shape[1], 33)
    batch = _batch(feats)

    with torch.no_grad():
        edge_index, valid_mask, node, edge = ado_enc(batch, single_inputs=feats["s_inputs"])
        torch.manual_seed(0)
        ref_out = ref_dec(node.clone(), edge.clone(), edge_index, valid_mask, feats)["logits"]
        torch.manual_seed(0)
        ado_out = ado_dec(node.clone(), edge.clone(), edge_index, valid_mask, batch)

    diff = (ref_out - ado_out).abs().max().item()
    print(f"  decoder: logits shape={tuple(ado_out.shape)} max_abs={diff:.2e}")
    assert diff < 1e-5, "decoder logits mismatch"
    print("[ok] InverseFoldingDecoder forward+load parity vs reference")


if __name__ == "__main__":
    main()
