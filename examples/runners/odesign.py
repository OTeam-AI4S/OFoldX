from __future__ import annotations

from pathlib import Path

from ofoldx.runners import Runner

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "tripeptide.pdb"


def main() -> None:
    runner = Runner(
        {
            "steps": 1,
            "model": {
                "architecture": "odesign",
                "single_size": 16,
                "pair_size": 8,
                "constraint_embeddings": {
                    "hidden_size": 8,
                    "num_blocks": 1,
                    "num_distogram_bins": 4,
                    "dropout": 0.0,
                },
                "trunk": {
                    "num_msa_stages": 1,
                    "num_pairformer_blocks": 1,
                    "pairformer_dropout": 0.0,
                    "msa_dropout": 0.0,
                    "pair_dropout": 0.0,
                },
                "denoiser": {
                    "num_token_encoder_blocks": 1,
                    "num_token_encoder_attention_heads": 1,
                    "atom_hidden_size": 8,
                    "atom_pair_hidden_size": 4,
                    "token_hidden_size": 16,
                    "num_atom_encoder_blocks": 1,
                    "num_atom_encoder_attention_heads": 1,
                    "num_atom_decoder_blocks": 1,
                    "num_atom_decoder_attention_heads": 1,
                    "num_query_atoms": 4,
                    "num_key_atoms": 4,
                },
                "pairwise": {
                    "num_distogram_bins": 4,
                    "bond_hidden_size": 8,
                },
            },
            "forward": {"num_samples": 2, "sample_chunk_size": 1},
            "data": {"train": str(STRUCTURE)},
            "optim": {"type": "adamw", "lr": 1e-4},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "logging": {"enabled": False},
            "workspace": {"root": "runs/odesign"},
            "ckpt": {"enabled": False},
        }
    )
    try:
        runner.train()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
