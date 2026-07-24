from __future__ import annotations

import pytest

from ofoldx.modules.denoiser import CoordinateDenoiser, DenoisingConditioner


@pytest.mark.parametrize("backend", ("eager", "oops"))
def test_coordinate_denoiser_propagates_backend(backend: str) -> None:
    denoiser = CoordinateDenoiser(
        num_encoder_blocks=1,
        num_encoder_attention_heads=2,
        atom_hidden_size=8,
        atom_pair_hidden_size=4,
        token_hidden_size=8,
        single_size=6,
        pair_size=4,
        token_feature_size=5,
        atom_feature_size=7,
        num_atom_encoder_blocks=1,
        num_atom_encoder_attention_heads=2,
        num_atom_decoder_blocks=1,
        num_atom_decoder_attention_heads=2,
        num_query_atoms=2,
        num_key_atoms=4,
        backend=backend,
    )

    owners = [module.backend for module in denoiser.modules() if hasattr(module, "backend")]
    assert owners
    assert set(owners) == {backend}


@pytest.mark.parametrize("backend", ("eager", "oops"))
def test_denoising_conditioner_propagates_backend(backend: str) -> None:
    conditioner = DenoisingConditioner(
        pair_input_size=8,
        pair_size=4,
        single_conditioning_size=11,
        single_size=6,
        backend=backend,
    )

    owners = [module.backend for module in conditioner.modules() if hasattr(module, "backend")]
    assert owners
    assert set(owners) == {backend}
