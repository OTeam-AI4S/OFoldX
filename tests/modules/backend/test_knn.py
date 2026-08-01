import torch

from ofoldx.modules.backend import masked_knn


def test_masked_knn_dense_contract() -> None:
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False, True]])

    distances, indices, valid = masked_knn(coordinates, mask, 3)

    assert indices.tolist() == [[[0, 2, 0], [0, 0, 0], [2, 0, 0]]]
    assert valid.tolist() == [[[True, True, False], [False, False, False], [True, True, False]]]
    torch.testing.assert_close(
        distances,
        torch.tensor([[[0.001, 5.0, 6.0], [1.0, 1.0, 1.0], [0.001, 5.0, 6.0]]]),
        atol=1e-6,
        rtol=1e-6,
    )
