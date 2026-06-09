from __future__ import annotations

import pytest
import torch

from ofoldx.modules.embedding import TemplateEmbedder, TemplateFeatures


def make_features(num_templates: int = 1, num_tokens: int = 3) -> TemplateFeatures:
    return TemplateFeatures(
        distogram=torch.zeros(1, num_templates, num_tokens, num_tokens, 5),
        pseudo_beta_mask=torch.ones(1, num_templates, num_tokens, num_tokens),
        residue_type=torch.zeros(1, num_templates, num_tokens, dtype=torch.long),
        unit_vector=torch.zeros(1, num_templates, num_tokens, num_tokens, 3),
        backbone_frame_mask=torch.ones(1, num_templates, num_tokens, num_tokens),
        template_mask=torch.ones(1, num_templates, dtype=torch.bool),
    )


def test_features_to():
    moved = make_features().to(dtype=torch.float16)

    # Only floating fields follow the requested dtype; integer/bool fields keep their type.
    assert moved.distogram.dtype == torch.float16
    assert moved.pseudo_beta_mask.dtype == torch.float16
    assert moved.unit_vector.dtype == torch.float16
    assert moved.backbone_frame_mask.dtype == torch.float16
    assert moved.residue_type.dtype == torch.long
    assert moved.template_mask is not None
    assert moved.template_mask.dtype == torch.bool


@pytest.mark.parametrize(("num_blocks", "num_templates"), [(0, 1), (0, 0), (1, 0)])
def test_zero_update_when_disabled(num_blocks, num_templates):
    # Either no triangle blocks or no templates short-circuits to an exact-zero update.
    embedder = TemplateEmbedder(pair_size=8, hidden_size=4, num_blocks=num_blocks, num_distogram_bins=5)
    pair = torch.ones(1, 3, 3, 8)

    output = embedder(pair, make_features(num_templates=num_templates))

    assert torch.equal(output, torch.zeros_like(pair))


def test_single_template_update_matches_materialized_one_hot():
    # The broadcast/sliced-column path must equal a dense projection over the full feature tensor
    # in order [distogram, pseudo_beta_mask, residue_i(K), residue_j(K), unit_vector(3), frame_mask(1)];
    # a wrong broadcast axis, weight-column slice, or feature order would diverge.
    torch.manual_seed(0)
    vocab_size, num_distogram_bins, num_tokens = 2, 3, 3
    embedder = TemplateEmbedder(
        pair_size=8,
        hidden_size=4,
        num_blocks=0,
        vocab_size=vocab_size,
        num_distogram_bins=num_distogram_bins,
    )
    embedder.eval()

    distogram = torch.randn(1, 1, num_tokens, num_tokens, num_distogram_bins)
    pseudo_beta_mask = torch.rand(1, 1, num_tokens, num_tokens)
    unit_vector = torch.randn(1, 1, num_tokens, num_tokens, 3)
    backbone_frame_mask = torch.rand(1, 1, num_tokens, num_tokens)
    residue_type = torch.tensor([[[0, 1, 1]]], dtype=torch.long)  # distinct per-token classes
    features = TemplateFeatures(
        distogram=distogram,
        pseudo_beta_mask=pseudo_beta_mask,
        residue_type=residue_type,
        unit_vector=unit_vector,
        backbone_frame_mask=backbone_frame_mask,
        template_mask=None,
    )

    pair = torch.randn(1, num_tokens, num_tokens, 8)
    normalized_pair = embedder.norm_pair(pair)

    output = embedder._single_template_update(normalized_pair, templates=features, template_index=0, pair_mask=None)

    residue_one_hot = torch.nn.functional.one_hot(residue_type[:, 0], num_classes=vocab_size).to(distogram.dtype)
    residue_i = residue_one_hot.unsqueeze(-2).expand(-1, num_tokens, num_tokens, -1)
    residue_j = residue_one_hot.unsqueeze(-3).expand(-1, num_tokens, num_tokens, -1)
    full_features = torch.cat(
        [
            distogram[:, 0],
            pseudo_beta_mask[:, 0].unsqueeze(-1),
            residue_i,
            residue_j,
            unit_vector[:, 0],
            backbone_frame_mask[:, 0].unsqueeze(-1),
        ],
        dim=-1,
    )
    reference = embedder.pair_projection(normalized_pair) + embedder.template_projection(full_features)
    reference = embedder.hidden_norm(reference)

    assert output.std() > 0.1  # guard against a degenerate (constant) output sneaking the equality through
    # tol: equality is exact in real arithmetic; ~3e-7 slack is single-precision rounding of the two paths.
    torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)
