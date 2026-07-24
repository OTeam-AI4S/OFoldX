r"""
Shared fixtures for OFoldX tests.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ofoldx.models.alphafold3 import AlphaFold3Input  # noqa: E402

# The seed every test starts from. Arbitrary but fixed: the point is that a test's numeric behaviour is
# independent of how many random draws the preceding tests consumed, not that this value is special.
_GLOBAL_RNG_SEED = 0


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    r"""
    Give every test the same, order-independent global RNG state.

    Several tests build ``nn.Module``s (random-initialised ``Linear`` weights, etc.) or draw samples from
    the *global* generators without seeding first, and torch's default generator is seeded from OS entropy
    at process start. Left alone, the state entering such a test depends on how many draws the earlier
    tests happened to consume, so the same suite passes or fails run to run (and in different collection
    subsets). This autouse fixture snapshots the global RNG state, reseeds every global generator to a
    fixed value before the test, and restores the snapshot afterwards -- so no test can leak RNG state
    into the next one, and unseeded tests become deterministic. Tests that call ``torch.manual_seed`` (or
    use a local ``torch.Generator``) still fully control their own randomness.

    The suite runs on CPU, so only the CPU generators are handled; GPU RNG plumbing can be added if a
    CUDA test box is introduced.
    """

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    random.seed(_GLOBAL_RNG_SEED)
    np.random.seed(_GLOBAL_RNG_SEED)
    torch.manual_seed(_GLOBAL_RNG_SEED)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def _atom_token_fields(atom_to_token: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    atom_token_index = torch.zeros(atom_to_token.shape[0], dtype=torch.long)
    token_to_rep_atom = torch.zeros(num_tokens, dtype=torch.long)
    counts = torch.zeros(num_tokens, dtype=torch.long)
    seen = torch.zeros(num_tokens, dtype=torch.bool)
    for atom_index, token_index_tensor in enumerate(atom_to_token):
        token_index = int(token_index_tensor)
        atom_token_index[atom_index] = counts[token_index]
        counts[token_index] += 1
        if not seen[token_index]:
            token_to_rep_atom[token_index] = atom_index
            seen[token_index] = True
    return atom_token_index, token_to_rep_atom


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
    atom_token_index, token_to_rep_atom_1d = _atom_token_fields(atom_to_token_1d, num_tokens)

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
        atom_token_index=atom_token_index,
        token_to_rep_atom=token_to_rep_atom_1d.expand(batch_size, num_tokens).clone(),
        token_bonds=torch.zeros(batch_size, num_tokens, num_tokens),
    )


@pytest.fixture
def dummy_model_input():
    r"""
    Factory for cheap synthetic ``AlphaFold3Input`` objects.
    """

    return _dummy_model_input
