import torch

from ofoldx.modules.head import tm_bin_weights, tm_length_scale, tm_weight_from_pae


def test_tm_equations() -> None:
    clamped = 1.24 * (19.0 - 15.0) ** (1.0 / 3.0) - 1.8
    assert tm_length_scale(10.0) == clamped

    counts = torch.tensor([10.0, 64.0])
    scales = 1.24 * (counts.clamp(min=19.0) - 15.0) ** (1.0 / 3.0) - 1.8
    torch.testing.assert_close(tm_length_scale(counts), scales)

    centers = torch.tensor([0.0, 1.0, 4.0])
    expected = 1.0 / (1.0 + (centers / scales[:, None]) ** 2)
    torch.testing.assert_close(tm_bin_weights(centers, counts), expected)

    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -1.0, 0.5]])
    expected_from_pae = (logits.softmax(dim=-1) * expected).sum(dim=-1)
    torch.testing.assert_close(
        tm_weight_from_pae(logits, counts, bin_centers=centers),
        expected_from_pae,
    )
