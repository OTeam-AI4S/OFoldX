from __future__ import annotations

import pytest
import torch

from ofoldx.modules.embedding import TemplateEmbedder, TemplateFeatures


def _make_features(num_templates: int = 1, num_tokens: int = 3) -> TemplateFeatures:
    return TemplateFeatures(
        distogram=torch.zeros(1, num_templates, num_tokens, num_tokens, 5),
        pseudo_beta_mask=torch.ones(1, num_templates, num_tokens, num_tokens),
        residue_type_ids=torch.zeros(1, num_templates, num_tokens, dtype=torch.long),
        unit_vector=torch.zeros(1, num_templates, num_tokens, num_tokens, 3),
        backbone_frame_mask=torch.ones(1, num_templates, num_tokens, num_tokens),
        template_mask=torch.ones(1, num_templates, dtype=torch.bool),
    )


class TestTemplateEmbedder:
    @pytest.mark.parametrize(("num_blocks", "num_templates"), [(0, 1), (1, 0)])
    def test_zero_update_when_disabled(self, num_blocks: int, num_templates: int) -> None:
        embedder = TemplateEmbedder(pair_size=8, hidden_size=4, num_blocks=num_blocks, num_distogram_bins=5)
        pair = torch.ones(1, 3, 3, 8)

        output = embedder(pair, _make_features(num_templates=num_templates))

        assert torch.equal(output, torch.zeros_like(pair))

    def test_template_feature_order(self) -> None:
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
        residue_type_ids = torch.tensor([[[0, 1, 1]]], dtype=torch.long)
        features = TemplateFeatures(
            distogram=distogram,
            pseudo_beta_mask=pseudo_beta_mask,
            residue_type_ids=residue_type_ids,
            unit_vector=unit_vector,
            backbone_frame_mask=backbone_frame_mask,
            template_mask=None,
        )

        pair = torch.randn(1, num_tokens, num_tokens, 8)
        normalized_pair = embedder.pair_norm(pair)
        output = embedder._single_template_update(
            normalized_pair,
            templates=features,
            template_index=0,
            pair_mask=None,
        )

        residue_one_hot = torch.nn.functional.one_hot(
            residue_type_ids[:, 0],
            num_classes=vocab_size,
        ).to(distogram.dtype)
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
        reference = embedder.pair_projection(normalized_pair) + torch.nn.functional.linear(
            full_features,
            embedder.template_projection.weight,
            embedder.template_projection.bias,
        )

        torch.testing.assert_close(output, embedder.hidden_norm(reference))
