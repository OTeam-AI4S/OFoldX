"""Compare OFoldX LigandMPNN against cloned dauparas/LigandMPNN.

Loads a released ``ligandmpnn_v_32_*`` checkpoint, converts it, and compares
teacher-forced log-probabilities from upstream ``score(..., use_sequence=True)``.

Run::

    python examples/parity/ligandmpnn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ofoldx.data.structure import Structure
from ofoldx.models.ligandmpnn import LigandMpnnConfig, LigandMpnnInput, LigandMpnnModel
from ofoldx.models.ligandmpnn.convert import (
    convert_ligandmpnn_state_dict,
    ligandmpnn_config_from_checkpoint,
    load_ligandmpnn_checkpoint,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CKPT_DIR, UPSTREAM_DIR, clone_source, download, load_module, report, unique_backbone

_SOURCE_URL = "https://github.com/dauparas/LigandMPNN.git"
_CKPT_URL = "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt"


def _input(*, backbone, tokens, residue_mask, residue_index, chain_index, context_xyz, context_z, context_mask):
    batch, length = tokens.shape
    return LigandMpnnInput(
        backbone_coordinates=backbone,
        sequence_tokens=tokens,
        residue_mask=residue_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        context_coordinates=context_xyz,
        context_atomic_numbers=context_z,
        context_mask=context_mask,
        residue_context_coordinates=torch.zeros(batch, length, 32, 3),
        residue_context_mask=torch.zeros(batch, length, 32),
        design_positions=torch.ones(batch, length, dtype=torch.bool),
        ligand_interface_mask=torch.ones(batch, length, dtype=torch.bool),
        structure=Structure.from_sequence("A" * length),
        component_index=torch.arange(length).expand(batch, length),
        tied_group_ids=torch.arange(length).expand(batch, length),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=UPSTREAM_DIR / "LigandMPNN")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "ligandmpnn_v_32_010_25.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=0)
    args = parser.parse_args(argv)

    source = clone_source(_SOURCE_URL, args.source, "model_utils.py")
    checkpoint = download(_CKPT_URL, args.checkpoint)
    print(f"ligandmpnn: source checkpoint {checkpoint}", flush=True)
    ProteinMPNN = load_module(source / "model_utils.py", "ligandmpnn_model_utils").ProteinMPNN
    ckpt = load_ligandmpnn_checkpoint(checkpoint)
    config = ligandmpnn_config_from_checkpoint(ckpt)
    context_atoms = int(ckpt["atom_context_num"])
    upstream = ProteinMPNN(
        num_letters=21,
        node_features=config.hidden_size,
        edge_features=config.hidden_size,
        hidden_dim=config.hidden_size,
        num_encoder_layers=config.num_encoder_blocks,
        num_decoder_layers=config.num_decoder_blocks,
        vocab=21,
        k_neighbors=config.num_neighbors,
        augment_eps=0.0,
        dropout=config.dropout,
        atom_context_num=context_atoms,
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=False,
    )
    upstream.load_state_dict(ckpt["model_state_dict"], strict=True)
    upstream.eval()
    converted = LigandMpnnModel(config)
    converted.load_state_dict(convert_ligandmpnn_state_dict(ckpt["model_state_dict"]), strict=True, assign=True)
    converted.eval()
    converted.set_backend("eager")

    batch = 1
    length = args.length or max(config.num_neighbors, 12)
    torch.manual_seed(args.seed)
    backbone = unique_backbone(batch, length)
    tokens = torch.randint(0, 20, (batch, length))
    residue_mask = torch.ones(batch, length)
    residue_index = torch.arange(length).expand(batch, length)
    chain_index = torch.zeros(batch, length, dtype=torch.long)
    context_xyz = torch.randn(batch, length, context_atoms, 3) + torch.arange(length).view(1, length, 1, 1)
    context_z = torch.full((batch, length, context_atoms), 6, dtype=torch.long)
    context_mask = torch.ones(batch, length, context_atoms)
    decoding_ranks = torch.arange(length).expand(batch, length)
    randn = decoding_ranks.float()
    feature_dict = {
        "X": backbone,
        "S": tokens,
        "mask": residue_mask,
        "R_idx": residue_index,
        "chain_labels": chain_index,
        "Y": context_xyz,
        "Y_t": context_z,
        "Y_m": context_mask,
        "batch_size": batch,
        "chain_mask": residue_mask,
        "randn": randn,
        "symmetry_residues": [[]],
    }
    with torch.no_grad():
        expected = upstream.score(feature_dict, use_sequence=True)["log_probs"]
        actual = converted(
            _input(
                backbone=backbone,
                tokens=tokens,
                residue_mask=residue_mask.bool(),
                residue_index=residue_index,
                chain_index=chain_index,
                context_xyz=context_xyz,
                context_z=context_z,
                context_mask=context_mask.bool(),
            ),
            decoding_ranks=decoding_ranks,
        )
        log_probs = F.log_softmax(actual.logits, dim=-1)
    report("ligandmpnn/eager", log_probs, expected)


if __name__ == "__main__":
    main()
