# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

from pathlib import Path

from ofoldx.io import read_structure, save_structure
from ofoldx.runners import Runner

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "tripeptide.pdb"


def main() -> None:
    # training datasets read native artifacts only, so the bundled PDB file is read and saved once beforehand
    artifact = Path("runs/alphafold3/tripeptide.safetensors")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_structure(read_structure(STRUCTURE)[0], artifact)
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
                    "triangle_attention_head_size": 4,
                    "num_triangle_attention_heads": 1,
                },
                "trunk": {"num_pairformer_blocks": 1},
                "num_recycles": 1,
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
                    "triangle_attention_head_size": 4,
                    "num_triangle_attention_heads": 1,
                },
            },
            "forward": {"num_samples": 2, "sample_chunk_size": 1},
            "data": {"train": str(artifact)},
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
