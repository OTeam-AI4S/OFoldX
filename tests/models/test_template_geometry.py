from __future__ import annotations

import pytest
import torch

from ofoldx.models.boltz2.model import Boltz2TemplateEmbedder
from ofoldx.models.boltzgen.model import BoltzGenMiniformerTemplateEmbedder, BoltzGenTemplateEmbedder
from ofoldx.models.boltzgen.processor import BoltzGenProcessor


@pytest.mark.parametrize(
    "embedder_class", [Boltz2TemplateEmbedder, BoltzGenTemplateEmbedder, BoltzGenMiniformerTemplateEmbedder]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_template_geometry_forward(embedder_class, dtype):
    """A read-only hook observes the geometry entering the learned projection during real forwards."""
    if embedder_class is Boltz2TemplateEmbedder:
        embedder = embedder_class(
            pair_size=4,
            hidden_size=4,
            num_blocks=0,
            vocab_size=1,
            num_bins=4,
            min_distance=1,
            max_distance=7,
            backend="eager",
        )
        projection = embedder.template_projection
    else:
        embedder = embedder_class(
            pair_size=4,
            template_size=4,
            num_blocks=0,
            num_bins=4,
            min_dist=1,
            max_dist=7,
            backend="eager",
        )
        projection = embedder.template_feature_projection
    embedder.to(dtype=dtype).eval()
    positions = torch.tensor([[[[0, 0, 0], [3, 4, 0], [3, 0, 0], [0, 0, 0]]]], dtype=dtype)
    rotations = torch.eye(3, dtype=dtype).expand(1, 1, 4, 3, 3).clone()
    frame_mask = torch.tensor([[[1, 0, 1, 1]]], dtype=dtype)
    beta_carbon_mask = torch.tensor([[[1, 1, 0, 1]]], dtype=dtype)
    visibility = torch.tensor([[[0, 0, 0, 1]]], dtype=dtype)
    template_mask = torch.ones(1, 1, 4, dtype=dtype)
    if embedder_class is Boltz2TemplateEmbedder:
        args = ()
        kwargs = {
            "residue_type_features": torch.ones(1, 1, 4, 1, dtype=dtype),
            "frame_rotations": rotations,
            "frame_translations": positions,
            "frame_mask": frame_mask,
            "beta_carbon_positions": positions,
            "alpha_carbon_positions": positions,
            "beta_carbon_mask": beta_carbon_mask,
            "visibility_index": visibility,
            "template_mask": template_mask,
        }
    else:
        input = BoltzGenProcessor(dtype=dtype)(None, total_len=4)
        input.template_residue_type_features = input.residue_type_features.unsqueeze(1)
        input.template_frame_rotations = rotations
        input.template_frame_translations = positions
        input.template_frame_mask = frame_mask
        input.template_beta_carbon_positions = positions
        input.template_alpha_carbon_positions = positions
        input.template_beta_carbon_mask = beta_carbon_mask
        input.template_visibility_index = visibility
        input.template_mask = template_mask
        args = (input,)
        kwargs = {}
    pair = torch.zeros(1, 4, 4, 4, dtype=dtype)
    pair_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    captured = []

    def capture_geometry(module, args):
        captured.append(args[0].detach().clone())

    hook = projection.register_forward_pre_hook(capture_geometry)
    perturbation = 2e-10
    try:
        with torch.no_grad():
            output = embedder(pair, *args, **kwargs, pair_mask=pair_mask)
            rotations[0, 0, 0, 0, 1] = -perturbation
            rotations[0, 0, 0, 1, 0] = perturbation
            if embedder_class is not Boltz2TemplateEmbedder:
                input.template_frame_rotations = rotations
            perturbed_output = embedder(pair, *args, **kwargs, pair_mask=pair_mask)
    finally:
        hook.remove()

    geometry, perturbed_geometry = captured
    assert geometry.shape == (1, 1, 4, 4, 9)
    directions = geometry[0, 0, :, :, 5:8]
    # Pair [i, j] expresses the displacement to CA_i in frame j.
    expected = torch.tensor(
        [
            [[0, 0, 0], [-0.6, -0.8, 0], [-1, 0, 0], [0, 0, 0]],
            [[0.6, 0.8, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(directions, expected, rtol=0, atol=0)
    assert geometry[0, 0, :3, :3, :4].argmax(dim=-1).tolist() == [[0, 2, 1], [2, 0, 1], [1, 1, 0]]
    assert geometry[0, 0, :3, :3, 4].tolist() == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert geometry[0, 0, :3, :3, 8].tolist() == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    assert not geometry[0, 0, 3, :3].any()
    assert not geometry[0, 0, :3, 3].any()

    perturbed_directions = perturbed_geometry[0, 0, :, :, 5:8]
    torch.testing.assert_close(
        perturbed_directions[2, 0],
        torch.tensor([1, -perturbation, 0], dtype=dtype),
        rtol=4 * torch.finfo(dtype).eps,
        atol=0,
    )
    assert (perturbed_directions - directions).abs().max() <= 2 * perturbation
    assert not perturbed_directions.diagonal(dim1=0, dim2=1).any()
    for value in (geometry, perturbed_geometry, output, perturbed_output):
        assert value.dtype == dtype
        assert torch.isfinite(value).all()
    assert output.shape == perturbed_output.shape == pair.shape
