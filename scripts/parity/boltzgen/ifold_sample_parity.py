"""Parity for InverseFoldingDecoder.sample (autoregressive) vs reference, with controlled randomness.

Both use argmax decoding (sampling_temperature=None) and the same seeded randperm order, so the decode is
deterministic and the sampled sequence + logits must match exactly. design_mask is all-True (every position
designed); this parity case leaves the supported homomer-symmetry and amino-acid constraints disabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.parity.boltzgen import reference
from scripts.parity.boltzgen.ifold_encoder_parity import _batch, _feats, _ref_to_ofoldx
from scripts.parity.boltzgen.ifold_gnn_parity import CKPT_DEFAULT, _ensure_checkpoint, _sub_state


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    ckpt = reference.load_checkpoint(_ensure_checkpoint(args.checkpoint))
    state = ckpt["state_dict"]
    reference.configure_reference()
    from boltzgen.model.modules.inverse_fold import InverseFoldingDecoder as RefDecoder
    from boltzgen.model.modules.inverse_fold import InverseFoldingEncoder as RefEncoder

    from ofoldx.models.boltzgen.convert import convert_decoder
    from ofoldx.models.boltzgen.model import InverseFoldingDecoder, InverseFoldingEncoder

    enc_state = _sub_state(state, "inverse_folding_encoder")
    ref_enc = RefEncoder(
        atom_s=128,
        atom_z=16,
        token_s=384,
        token_z=128,
        num_encoder_layers=6,
        topk=30,
        enable_input_embedder=True,
    ).eval()
    ref_enc.load_state_dict(enc_state, strict=True)
    ado_enc = InverseFoldingEncoder(num_layers=6).eval()
    ado_enc.load_state_dict(_ref_to_ofoldx(enc_state), strict=False)

    dec_state = _sub_state(state, "structure_module")
    ref_dec = RefDecoder(
        atom_s=128,
        atom_z=16,
        token_s=384,
        token_z=128,
        num_decoder_layers=3,
        num_heads=4,
        sampling_temperature=None,
        tie_symmetric_sequences=False,
    ).eval()
    ref_dec.load_state_dict(dec_state, strict=True)
    ado_dec = InverseFoldingDecoder(num_layers=3).eval()
    ado_dec.load_state_dict({convert_decoder(k): v for k, v in dec_state.items()}, strict=True)

    feats = _feats()
    n = feats["s_inputs"].shape[1]
    feats["res_type_clone"] = torch.zeros(1, n, 33)
    feats["design_mask"] = torch.ones(1, n)

    batch = _batch(feats)
    with torch.no_grad():
        edge_index, valid_mask, node, edge = ado_enc(batch, single_inputs=feats["s_inputs"])
        torch.manual_seed(7)
        ref_out = ref_dec.sample(node.clone(), edge.clone(), edge_index, valid_mask, feats)
        torch.manual_seed(7)
        ado_logits, ado_residue_type = ado_dec.sample(node.clone(), edge.clone(), edge_index, valid_mask, batch)

    seq_match = torch.equal(ref_out["res_type"].argmax(-1), ado_residue_type.argmax(-1))
    logit_diff = (ref_out["logits"] - ado_logits).abs().max().item()
    print(f"  sample: sequence_match={seq_match} logits_max_abs={logit_diff:.2e}")
    assert seq_match and logit_diff < 1e-5, "sample mismatch"
    print("[ok] InverseFoldingDecoder.sample parity vs reference (seeded argmax)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
