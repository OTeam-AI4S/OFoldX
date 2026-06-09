r"""
Shared fixtures for OFoldX tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Keep the committed suite self-contained (DESIGN philosophy: no external data, no large checkpoints):
# point the default CCD provider at a non-existent file so it stays empty. Tests that need reference
# components inject a fake provider instead. This also avoids the RDKit-Mol interpreter-teardown crash
# on dev machines where the real ~0.13 GB pickle exists.
os.environ["OFOLDX_CCD_RDKIT_MOL_FILE"] = "/nonexistent-ccd-for-tests.pkl"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ofoldx.models import AlphaFold3Input  # noqa: E402


def _atom_token_fields(atom_to_token: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    representative_atom_mask = torch.zeros(atom_to_token.shape[0], dtype=torch.bool)
    atom_token_index = torch.zeros(atom_to_token.shape[0], dtype=torch.long)
    token_to_rep_atom = torch.zeros(num_tokens, dtype=torch.long)
    counts = torch.zeros(num_tokens, dtype=torch.long)
    seen = torch.zeros(num_tokens, dtype=torch.bool)
    for atom_index, token_index_tensor in enumerate(atom_to_token):
        token_index = int(token_index_tensor)
        atom_token_index[atom_index] = counts[token_index]
        counts[token_index] += 1
        if not seen[token_index]:
            representative_atom_mask[atom_index] = True
            token_to_rep_atom[token_index] = atom_index
            seen[token_index] = True
    return representative_atom_mask, atom_token_index, token_to_rep_atom


def _dummy_model_input(
    batch_size: int = 1,
    num_tokens: int = 4,
    num_atoms: int = 8,
    token_feature_size: int = 32,
    atom_feature_size: int = 17,
) -> AlphaFold3Input:
    residue_index = torch.arange(num_tokens, dtype=torch.long).expand(batch_size, num_tokens).clone()
    token_index = residue_index.clone()
    atom_to_token_1d = torch.arange(num_atoms, dtype=torch.long).remainder(num_tokens)
    atom_to_token = atom_to_token_1d.expand(batch_size, num_atoms).clone()
    representative_atom_mask, atom_token_index, token_to_rep_atom_1d = _atom_token_fields(atom_to_token_1d, num_tokens)
    token_arange = torch.arange(num_tokens, dtype=torch.long)
    frame_atom_index_1d = torch.stack(
        [token_arange % num_atoms, (token_arange + 1) % num_atoms, (token_arange + 2) % num_atoms],
        dim=-1,
    )

    return AlphaFold3Input(
        token_features=torch.zeros(batch_size, num_tokens, token_feature_size),
        token_mask=torch.ones(batch_size, num_tokens, dtype=torch.bool),
        residue_index=residue_index,
        chain_index=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        atom_features=torch.zeros(batch_size, num_atoms, atom_feature_size),
        atom_to_token=atom_to_token,
        atom_mask=torch.ones(batch_size, num_atoms, dtype=torch.bool),
        reference_positions=torch.zeros(batch_size, num_atoms, 3),
        reference_mask=torch.ones(batch_size, num_atoms, dtype=torch.bool),
        reference_charges=torch.zeros(batch_size, num_atoms, 1),
        entity_index=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        symmetry_index=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        token_index=token_index,
        reference_space_id=torch.zeros(batch_size, num_atoms, dtype=torch.long),
        representative_atom_mask=representative_atom_mask,
        atom_token_index=atom_token_index,
        token_to_rep_atom=token_to_rep_atom_1d.expand(batch_size, num_tokens).clone(),
        molecule_type=torch.zeros(batch_size, num_tokens, dtype=torch.long),
        frame_atom_index=frame_atom_index_1d.expand(batch_size, num_tokens, 3).clone(),
        frame_mask=torch.ones(batch_size, num_tokens, dtype=torch.bool),
        token_bonds=torch.zeros(batch_size, num_tokens, num_tokens),
        # trailing dims are the AF3 constraint feature widths: contact_feature_size=2, num_distogram_bins=39
        contact_constraints=torch.zeros(batch_size, num_tokens, num_tokens, 2),
        constraint_distogram=torch.zeros(batch_size, num_tokens, num_tokens, 39),
        target_positions=torch.zeros(batch_size, num_atoms, 3),
        target_mask=torch.ones(batch_size, num_atoms, dtype=torch.bool),
    )


@pytest.fixture
def dummy_model_input():
    r"""Factory for cheap synthetic ``AlphaFold3Input`` objects."""

    return _dummy_model_input
