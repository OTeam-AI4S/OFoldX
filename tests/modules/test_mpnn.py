from __future__ import annotations

import torch
from danling import NestedTensor

from ofoldx.modules.mpnn import (
    MpnnEncoderBlock,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
    label_smoothed_nll_loss,
    nearest_neighbors,
    nll_loss,
)


def test_gather_nodes() -> None:
    nodes = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    indices = torch.tensor([[[0, 2], [1, 0], [2, 1]]])

    actual = gather_nodes(nodes, indices)

    expected = torch.tensor([[[[1.0, 10.0], [3.0, 30.0]], [[2.0, 20.0], [1.0, 10.0]], [[3.0, 30.0], [2.0, 20.0]]]])
    torch.testing.assert_close(actual, expected)


def test_gather_edges() -> None:
    edges = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3, 1)
    indices = torch.tensor([[[0, 2], [1, 0], [2, 1]]])

    actual = gather_edges(edges, indices)

    torch.testing.assert_close(actual.squeeze(-1), torch.tensor([[[0.0, 2.0], [4.0, 3.0], [8.0, 7.0]]]))


def test_cat_neighbors_nodes() -> None:
    nodes = torch.tensor([[[1.0], [2.0]]])
    neighbors = torch.tensor([[[[10.0]], [[20.0]]]])
    indices = torch.tensor([[[1], [0]]])

    actual = cat_neighbors_nodes(nodes, neighbors, indices)

    torch.testing.assert_close(actual, torch.tensor([[[[10.0, 2.0]], [[20.0, 1.0]]]]))


def test_nearest_neighbors_padding() -> None:
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])

    distances, indices, valid = nearest_neighbors(coordinates, torch.ones(1, 2), num_neighbors=4)

    assert distances.shape == indices.shape == valid.shape == (1, 2, 4)
    assert torch.all((0 <= indices) & (indices < 2))
    torch.testing.assert_close(valid.sum(dim=-1), torch.full((1, 2), 2.0))
    for row_indices, row_valid in zip(indices[0], valid[0], strict=True):
        assert set(row_indices[row_valid.bool()].tolist()) == {0, 1}


def test_nearest_neighbors_ragged() -> None:
    coordinate_rows = (
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    )
    coordinates = NestedTensor(coordinate_rows)
    mask = NestedTensor([torch.ones(2), torch.ones(3)])

    distances, indices, valid = nearest_neighbors(coordinates, mask, num_neighbors=2)

    assert [tuple(row.shape) for row in distances] == [(2, 2), (3, 2)]
    for row_coordinates, row_distances, row_indices, row_valid in zip(
        coordinate_rows,
        distances,
        indices,
        valid,
        strict=True,
    ):
        expected = torch.sqrt(((row_coordinates[:, None] - row_coordinates[row_indices]) ** 2).sum(dim=-1) + 1e-6)
        torch.testing.assert_close(row_distances, expected)
        assert bool(row_valid.all())


class TestMpnnEncoderBlock:
    def test_backward(self) -> None:
        block = MpnnEncoderBlock(hidden_size=4, edge_input_size=8, dropout=0.0)
        node_states = torch.randn(2, 3, 4, requires_grad=True)
        edge_states = torch.randn(2, 3, 2, 4, requires_grad=True)
        neighbor_indices = torch.tensor([[[0, 1], [1, 2], [2, 0]]]).expand(2, -1, -1)

        updated_nodes, updated_edges = block(
            node_states,
            edge_states,
            neighbor_indices,
            node_mask=torch.ones(2, 3),
            attend_mask=torch.ones(2, 3, 2),
        )
        (updated_nodes.square().mean() + updated_edges.square().mean()).backward()

        assert updated_nodes.shape == node_states.shape
        assert updated_edges.shape == edge_states.shape
        assert node_states.grad is not None and torch.isfinite(node_states.grad).all()
        assert edge_states.grad is not None and torch.isfinite(edge_states.grad).all()


def test_nll_loss() -> None:
    tokens = torch.tensor([[0, 2]])
    log_probabilities = torch.tensor([[[-0.1, -1.0, -2.0], [-3.0, -2.0, -0.4]]])

    torch.testing.assert_close(nll_loss(tokens, log_probabilities), torch.tensor([[0.1, 0.4]]))


def test_label_smoothed_nll_loss() -> None:
    tokens = torch.tensor([[0, 2]])
    log_probabilities = torch.tensor([[[-0.1, -1.0, -2.0], [-3.0, -2.0, -0.4]]])

    actual = label_smoothed_nll_loss(tokens, log_probabilities, weight=0.2)

    torch.testing.assert_close(actual, torch.tensor([[0.25555556, 0.63333333]]))
