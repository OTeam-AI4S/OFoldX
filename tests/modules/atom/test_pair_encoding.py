from __future__ import annotations

import torch
import torch.nn.functional as F

from ofoldx.modules.atom import AtomPairEncoding


class TestAtomPairEncoding:
    def test_prefix_axes(self) -> None:
        encoding = AtomPairEncoding(atom_hidden_size=1, atom_pair_hidden_size=1)
        with torch.no_grad():
            encoding.distance_vector_projection.weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
            encoding.inverse_distance_projection.weight.fill_(3.0)
            encoding.distance_mask_projection.weight.fill_(4.0)
            encoding.left_atom_projection.weight.fill_(5.0)
            encoding.right_atom_projection.weight.fill_(-2.0)
            encoding.feed_forward_input_projection.weight.fill_(0.5)
            encoding.feed_forward_hidden_projection.weight.fill_(2.0)
            encoding.feed_forward_output_projection.weight.fill_(0.25)

        distance_vectors = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]]])
        query_single = torch.tensor([[[[0.2]], [[0.4]]]])
        key_single = torch.tensor([[[[0.3]]]])
        distance_term_mask = torch.tensor([[[[1.0]], [[0.5]]]])
        inverse_term_mask = torch.tensor([[[[0.5]], [[1.0]]]])
        valid_mask = torch.tensor([[[[1.0]], [[0.5]]]])
        pair_bias = torch.tensor([[[[[0.1]], [[0.2]]], [[[0.3]], [[0.4]]]]])

        output = encoding(
            distance_vectors=distance_vectors,
            query_single=query_single,
            key_single=key_single,
            distance_term_mask=distance_term_mask,
            inverse_term_mask=inverse_term_mask,
            valid_mask=valid_mask,
            pair_bias=pair_bias,
            prefix_ndim=1,
        )

        inverse_distance = 1 / (1 + (distance_vectors * distance_vectors).sum(dim=-1, keepdim=True))
        pair = (distance_vectors * torch.tensor([1.0, 2.0, -1.0])).sum(dim=-1, keepdim=True)
        pair = pair * distance_term_mask + 3 * inverse_distance * inverse_term_mask
        pair = pair + 4 * valid_mask + 5 * query_single - 2 * key_single
        pair = pair.unsqueeze(1) + pair_bias
        update = 0.5 * F.relu(pair)
        update = 2 * F.relu(update)
        expected = pair + 0.25 * F.relu(update)

        torch.testing.assert_close(output, expected)
