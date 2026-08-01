import pytest
import torch

from ofoldx.modules.head import AtomConfidenceProjector, expected_plddt


def test_expected_plddt_equation() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    probability = logits.softmax(dim=-1)
    expected = (
        probability[..., 0] * (0.5 * 100.0 / 3.0)
        + probability[..., 1] * (1.5 * 100.0 / 3.0)
        + probability[..., 2] * (2.5 * 100.0 / 3.0)
    )

    torch.testing.assert_close(expected_plddt(logits), expected)


class TestAtomConfidenceProjector:
    def test_atom_limit(self) -> None:
        projector = AtomConfidenceProjector(single_size=4, max_atoms_per_token=2, backend="eager")

        with pytest.raises(
            ValueError,
            match="a token carries 3 atoms but this confidence head was built for max_atoms_per_token=2",
        ):
            projector(
                torch.randn(1, 1, 4),
                torch.zeros(1, 1, dtype=torch.long),
                torch.tensor([[2]]),
            )

    def test_fullgraph(self) -> None:
        projector = AtomConfidenceProjector(single_size=4, max_atoms_per_token=3, backend="eager")
        with torch.no_grad():
            projector.plddt_weight.normal_()
            projector.resolved_weight.normal_()
        compiled = torch.compile(projector, backend="eager", fullgraph=True, dynamic=True)
        single = torch.randn(1, 3, 4)

        for atom_to_token, atom_token_index in (
            (torch.tensor([[0, 0, 1, 2]]), torch.tensor([[0, 1, 0, 0]])),
            (torch.tensor([[1, 1]]), torch.tensor([[0, 1]])),
        ):
            expected = projector(single, atom_to_token, atom_token_index)
            actual = compiled(single, atom_to_token, atom_token_index)
            assert tuple(value.shape for value in actual) == tuple(value.shape for value in expected)
            assert tuple(value.dtype for value in actual) == tuple(value.dtype for value in expected)
            torch.testing.assert_close(actual, expected)
