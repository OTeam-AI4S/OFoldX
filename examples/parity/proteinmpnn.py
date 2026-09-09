"""Compare OFoldX ProteinMPNN against cloned dauparas/ProteinMPNN.

Prefers a released source checkpoint (convert then teacher-forced log-probs).
Falls back to a randomly initialized miniature model when ``--random`` is set.

This is a local conversion check, not a CI test.

Run::

    python examples/parity/proteinmpnn.py --backend eager
    python examples/parity/proteinmpnn.py --random --backend eager
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ofoldx.data.structure import Structure
from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnInput, ProteinMpnnModel
from ofoldx.models.proteinmpnn.convert import convert_proteinmpnn_model, convert_proteinmpnn_state_dict
from ofoldx.modules.backend._oops import OOPS_AVAILABLE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CKPT_DIR, UPSTREAM_DIR, clone_source, download, load_module, report, unique_backbone

_SOURCE_URL = "https://github.com/dauparas/ProteinMPNN.git"
_CKPT_URL = "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt"


def _input(
    *,
    backbone: torch.Tensor,
    tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
) -> ProteinMpnnInput:
    batch, length = tokens.shape
    return ProteinMpnnInput(
        backbone_coordinates=backbone,
        sequence_tokens=tokens,
        residue_mask=residue_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        design_positions=torch.ones(batch, length, dtype=torch.bool),
        structure=Structure.from_sequence("A" * length),
        component_index=torch.arange(length).expand(batch, length),
        tied_group_ids=torch.arange(length).expand(batch, length),
    )


def _compare(backend: str, upstream, converted, *, length: int, ca_only: bool, seed: int) -> None:
    converted.eval()
    converted.set_backend(backend)
    upstream.eval()
    torch.manual_seed(seed)
    batch = 1
    atom_shape = (3,) if ca_only else (4, 3)
    if ca_only:
        backbone = torch.randn(batch, length, 3) + torch.arange(length).view(1, length, 1)
    else:
        backbone = unique_backbone(batch, length)
    tokens = torch.randint(0, 20, (batch, length))
    residue_mask = torch.ones(batch, length)
    residue_index = torch.arange(length).expand(batch, length)
    chain_index = torch.zeros(batch, length, dtype=torch.long)
    decoding_ranks = torch.arange(length).expand(batch, length)
    decoding_order = torch.argsort(decoding_ranks, dim=-1)
    with torch.no_grad():
        expected = upstream(
            backbone,
            tokens,
            residue_mask,
            residue_mask,
            residue_index,
            chain_index,
            torch.ones(batch, length),
            use_input_decoding_order=True,
            decoding_order=decoding_order,
        )
        actual = converted(
            _input(
                backbone=backbone,
                tokens=tokens,
                residue_mask=residue_mask.bool(),
                residue_index=residue_index,
                chain_index=chain_index,
            ),
            decoding_ranks=decoding_ranks,
        )
        log_probs = F.log_softmax(actual.logits, dim=-1)
    variant = "ca" if ca_only else "backbone"
    report(f"proteinmpnn/{variant}/{backend}", log_probs, expected)


def _from_checkpoint(source: Path, checkpoint: Path):
    ProteinMPNN = load_module(source / "protein_mpnn_utils.py", "protein_mpnn_utils").ProteinMPNN
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    hidden = int(state["W_s.weight"].shape[1])
    neighbors = int(ckpt["num_edges"])
    encoder = len({int(key.split(".", 2)[1]) for key in state if key.startswith("encoder_layers.")})
    decoder = len({int(key.split(".", 2)[1]) for key in state if key.startswith("decoder_layers.")})
    ca_only = state["features.edge_embedding.weight"].shape[1] == 167
    upstream = ProteinMPNN(
        num_letters=21,
        node_features=hidden,
        edge_features=hidden,
        hidden_dim=hidden,
        num_encoder_layers=encoder,
        num_decoder_layers=decoder,
        vocab=21,
        k_neighbors=neighbors,
        augment_eps=0.0,
        dropout=0.1,
        ca_only=ca_only,
    )
    upstream.load_state_dict(state, strict=True)
    converted = convert_proteinmpnn_model(checkpoint)
    return upstream, converted, max(neighbors, 12), ca_only


def _random_pair(source: Path, *, hidden: int, layers: int, neighbors: int, ca_only: bool, seed: int):
    ProteinMPNN = load_module(source / "protein_mpnn_utils.py", "protein_mpnn_utils").ProteinMPNN
    torch.manual_seed(seed)
    upstream = ProteinMPNN(
        num_letters=21,
        node_features=hidden,
        edge_features=hidden,
        hidden_dim=hidden,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
        vocab=21,
        k_neighbors=neighbors,
        augment_eps=0.0,
        dropout=0.1,
        ca_only=ca_only,
    )
    converted = ProteinMpnnModel(
        ProteinMpnnConfig(
            hidden_size=hidden,
            num_encoder_blocks=layers,
            num_decoder_blocks=layers,
            num_neighbors=neighbors,
            dropout=0.1,
            ca_only=ca_only,
        )
    )
    converted.load_state_dict(convert_proteinmpnn_state_dict(upstream.state_dict()), strict=True, assign=True)
    return upstream, converted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=UPSTREAM_DIR / "ProteinMPNN")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "proteinmpnn_v_48_020.pt")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--backend", nargs="+", default=("eager",))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args(argv)
    source = clone_source(_SOURCE_URL, args.source, "protein_mpnn_utils.py")

    for backend in args.backend:
        if backend == "oops" and not OOPS_AVAILABLE:
            print("proteinmpnn/oops: skipped (import oops failed)", flush=True)
            continue
        if args.random:
            length = args.length or 12
            for ca_only in (False, True):
                upstream, converted = _random_pair(
                    source,
                    hidden=32,
                    layers=2,
                    neighbors=args.neighbors,
                    ca_only=ca_only,
                    seed=args.seed,
                )
                _compare(backend, upstream, converted, length=length, ca_only=ca_only, seed=args.seed)
            continue
        checkpoint = download(_CKPT_URL, args.checkpoint)
        print(f"proteinmpnn: source checkpoint {checkpoint}", flush=True)
        upstream, converted, default_length, ca_only = _from_checkpoint(source, checkpoint)
        _compare(
            backend,
            upstream,
            converted,
            length=args.length or default_length,
            ca_only=ca_only,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
