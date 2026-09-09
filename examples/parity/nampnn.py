"""Compare OFoldX NA-MPNN against cloned baker-laboratory/NA-MPNN.

Converts a randomly initialized upstream state dict (released checkpoints are not
in the public clone) and compares teacher-forced log-probabilities on eager.

Run::

    python examples/parity/nampnn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ofoldx.data.structure import Structure
from ofoldx.models.nampnn.model import NA_MPNN_ATOM_NAMES, NaMpnnInput, NaMpnnModel
from ofoldx.models.nampnn import NaMpnnConfig
from ofoldx.models.nampnn.convert import convert_nampnn_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import UPSTREAM_DIR, clone_source, load_module, report

_SOURCE_URL = "https://github.com/baker-laboratory/NA-MPNN.git"

_ATOM_TYPES = list(NA_MPNN_ATOM_NAMES)
_POLYTYPES = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
_RESTYPES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "DA",
    "DC",
    "DG",
    "DT",
    "DX",
    "A",
    "C",
    "G",
    "U",
    "RX",
    "MAS",
    "PAD",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=UPSTREAM_DIR / "NA-MPNN")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args(argv)

    source = clone_source(_SOURCE_URL, args.source, "na_model_utils.py")
    ProteinMPNN = load_module(source / "na_model_utils.py", "na_mpnn_utils").ProteinMPNN
    atom_dict = dict(zip(_ATOM_TYPES, range(len(_ATOM_TYPES))))
    polytype_to_int = dict(zip(_POLYTYPES, range(len(_POLYTYPES))))
    restype_to_int = dict(zip(_RESTYPES, range(len(_RESTYPES))))
    hidden = args.hidden
    neighbors = args.neighbors
    layers = args.layers
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = ckpt["model_state_dict"]
        hidden = int(state["W_s.weight"].shape[1])
        neighbors = int(ckpt.get("num_edges", 32))
        layers = len({int(key.split(".", 2)[1]) for key in state if key.startswith("encoder_layers.")})
        print(f"nampnn: source checkpoint {args.checkpoint}", flush=True)
    else:
        print("nampnn: no public source checkpoint; converting random weights", flush=True)
        state = None

    torch.manual_seed(args.seed)
    upstream = ProteinMPNN(
        node_features=hidden,
        edge_features=hidden,
        hidden_dim=hidden,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
        atom_dict=atom_dict,
        restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int,
        vocab=33,
        num_letters=33,
        k_neighbors=neighbors,
        protein_augment_eps=0.0,
        dna_augment_eps=0.0,
        rna_augment_eps=0.0,
        dropout=0.1,
        decode_protein_first=0,
        include_pred_na_N=1,
    )
    if state is not None:
        upstream.load_state_dict(state, strict=True)
    upstream.eval()
    converted = NaMpnnModel(
        NaMpnnConfig(
            hidden_size=hidden,
            feature_size=hidden,
            num_encoder_blocks=layers,
            num_decoder_blocks=layers,
            num_neighbors=neighbors,
            dropout=0.1,
            include_predicted_na_n=True,
        )
    )
    converted.load_state_dict(convert_nampnn_state_dict(upstream.state_dict()), strict=True, assign=True)
    converted.eval()
    converted.set_backend("eager")

    batch = 1
    length = max(args.length, neighbors)
    num_atoms = len(_ATOM_TYPES)
    coords = torch.zeros(batch, length, num_atoms, 3)
    coords[:, :, :4] = torch.randn(batch, length, 4, 3) + torch.arange(length).view(1, length, 1, 1)
    atom_mask = torch.zeros(batch, length, num_atoms)
    atom_mask[:, :, :4] = 1
    tokens = torch.randint(0, 20, (batch, length))
    residue_mask = torch.ones(batch, length)
    residue_index = torch.arange(length).expand(batch, length)
    chain_index = torch.zeros(batch, length, dtype=torch.long)
    polymer = torch.zeros(batch, length, dtype=torch.long)
    decoding_ranks = torch.arange(length).expand(batch, length)
    feature_dict = {
        "X": coords,
        "X_m": atom_mask,
        "S": tokens,
        "mask": residue_mask,
        "R_idx": residue_index,
        "chain_labels": chain_index,
        "protein_mask": residue_mask,
        "dna_mask": torch.zeros_like(residue_mask),
        "rna_mask": torch.zeros_like(residue_mask),
        "R_polymer_type": polymer,
    }
    ofoldx_input = NaMpnnInput(
        atom_coordinates=coords,
        atom_mask=atom_mask.bool(),
        sequence_tokens=tokens,
        residue_mask=residue_mask.bool(),
        residue_index=residue_index,
        chain_index=chain_index,
        polymer_type_index=polymer,
        design_positions=torch.ones(batch, length, dtype=torch.bool),
        token_group_ids=torch.arange(length).expand(batch, length),
        structure=Structure.from_sequence("A" * length),
        component_index=torch.arange(length).expand(batch, length),
        tied_weights=torch.ones(batch, length),
        residue_logit_bias=torch.zeros(batch, length, 33),
        allowed_token_mask=torch.ones(batch, length, 33, dtype=torch.bool),
    )

    real_randn = torch.randn

    def _fixed_randn(shape, device=None, dtype=None, **kwargs):
        values = torch.arange(shape[-1], device=device, dtype=dtype or torch.float32).expand(shape)
        return values

    torch.randn = _fixed_randn
    try:
        with torch.no_grad():
            expected, _ = upstream(feature_dict)
            actual = converted(ofoldx_input, decoding_ranks=decoding_ranks)
            log_probs = F.log_softmax(actual.logits, dim=-1)
    finally:
        torch.randn = real_randn
    report("nampnn/eager", log_probs, expected)


if __name__ == "__main__":
    main()
