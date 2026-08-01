import torch
import torch.nn.functional as F

from ofoldx.modules.head import DistogramHead, PairSymmetrizedDistogramHead


class TestDistogramHead:
    def test_equation(self) -> None:
        head = DistogramHead(pair_size=2, num_bins=2)
        weight = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        bias = torch.tensor([0.5, -2.0])
        pair = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        head.projection.weight = torch.nn.Parameter(weight)
        head.projection.bias = torch.nn.Parameter(bias)

        expected = F.linear(pair, weight) + F.linear(pair.transpose(-2, -3), weight) + 2 * bias

        torch.testing.assert_close(head(pair), expected)


class TestPairSymmetrizedDistogramHead:
    def test_equation(self) -> None:
        head = PairSymmetrizedDistogramHead(pair_size=2, num_bins=2)
        weight = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        bias = torch.tensor([0.5, -2.0])
        pair = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        head.projection.weight = torch.nn.Parameter(weight)
        head.projection.bias = torch.nn.Parameter(bias)

        symmetrized_pair = pair + pair.transpose(-2, -3)
        expected = F.linear(symmetrized_pair, weight, bias)

        torch.testing.assert_close(head(pair), expected)
