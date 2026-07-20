r"""
Load parity for the ifold input_embedder vs OFoldX AtomConditionedInputEmbedder.

The BoltzGen ifold `input_embedder` is the already forward-verified
`AtomConditionedInputEmbedder` with a single-layer atom encoder. This asserts a
strict load from the real checkpoint through the family converter.
"""

from __future__ import annotations

from ofoldx.models.boltzgen.convert import convert_input_embedder
from scripts.parity.boltzgen import reference
from scripts.parity.boltzgen.ifold_gnn_parity import _ensure_checkpoint, _sub_state


def main() -> None:
    from ofoldx.models.boltzgen.model import AtomConditionedInputEmbedder

    ckpt = reference.load_checkpoint(_ensure_checkpoint())
    state = _sub_state(ckpt["state_dict"], "input_embedder")

    converted = {}
    for key, value in state.items():
        new_key = convert_input_embedder(key)
        if new_key is not None:
            converted[new_key] = value
    dropped = len(state) - len(converted)

    model = AtomConditionedInputEmbedder(
        single_size=384,
        atom_hidden_size=128,
        atom_pair_hidden_size=16,
        num_atom_encoder_blocks=1,
        num_atom_encoder_attention_heads=4,
        operator_backend="eager",
    )
    missing, unexpected = model.load_state_dict(converted, strict=False)
    print(
        f"  input_embedder: {len(state)} ckpt params, {dropped} tied-dropped, "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    assert not missing and not unexpected, f"missing={missing}\nunexpected={unexpected}"
    print("[ok] ifold input_embedder strict-load parity (reuses AtomConditionedInputEmbedder)")


if __name__ == "__main__":
    main()
