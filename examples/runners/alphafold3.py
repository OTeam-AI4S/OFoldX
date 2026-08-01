from __future__ import annotations

from pathlib import Path

from ofoldx.runners import Runner

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "tripeptide.pdb"


def main() -> None:
    runner = Runner(
        {
            "steps": 1,
            "model": {
                "architecture": "alphafold3",
                "single_size": 16,
                "pair_size": 16,
                "template": {
                    "hidden_size": 8,
                    "num_blocks": 0,
                    "triangle_attention_hidden_size": 4,
                    "num_triangle_attention_heads": 1,
                },
                "trunk": {"num_pairformer_blocks": 1},
                "recycling_steps": 1,
                "msa_max_depth": 1,
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
                "distogram": {"num_bins": 4},
                "confidence": {
                    "num_plddt_bins": 4,
                    "num_pae_bins": 4,
                    "num_pde_bins": 4,
                    "max_atoms_per_token": 8,
                    "num_blocks": 1,
                    "num_pair_attention_heads": 1,
                    "triangle_multiplication_hidden_size": 4,
                    "triangle_attention_hidden_size": 4,
                    "num_triangle_attention_heads": 1,
                },
            },
            "forward": {"num_samples": 2, "sample_chunk_size": 1},
            "data": {"train": str(STRUCTURE)},
            "optim": {"type": "adamw", "lr": 1e-4},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "logging": {"enabled": False},
            "workspace": {"root": "runs/alphafold3"},
            "ckpt": {"enabled": False},
        }
    )
    try:
        runner.train()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
