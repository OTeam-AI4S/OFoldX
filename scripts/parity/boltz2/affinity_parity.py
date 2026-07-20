"""Real-checkpoint parity for both Boltz-2 affinity heads and their combined score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from scripts.parity.paths import boltzgen_checkpoint, reference_root

sys.path.insert(0, str(reference_root("boltz") / "src"))


def checkpoint_subset(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    r"""Strip one exact module prefix from a checkpoint state."""

    prefix = prefix + "."
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    r"""Print one numerical comparison and return whether it passes."""

    difference = (actual.float() - expected.float()).abs()
    passed = difference.max().item() < 1e-4
    print(f"[{'ok' if passed else 'FAIL'}] {name}: " f"max_abs={difference.max():.3e} mean_abs={difference.mean():.3e}")
    return passed


def main() -> None:
    from boltz.model.modules.affinity import AffinityModule

    from ofoldx.models.boltz2.convert import convert_affinity_head
    from ofoldx.models.boltz2.model import (
        Boltz2AffinityHead,
        Boltz2AffinityModel,
        Boltz2Input,
        _Boltz2Representations,
    )

    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["state_dict"]
    hyper_parameters = checkpoint["hyper_parameters"]

    generator = torch.Generator(device=device).manual_seed(17)
    batch_size, token_count, atom_count = 1, 4, 8
    token_mask = torch.ones(batch_size, token_count, device=device)
    molecule_type = torch.tensor([[0, 3, 3, 3]], device=device)
    ligand_mask = torch.tensor([[False, True, True, True]], device=device)
    cross_mask = (
        ligand_mask.unsqueeze(-1) & (molecule_type == 0).unsqueeze(-2)
        | (molecule_type == 0).unsqueeze(-1) & ligand_mask.unsqueeze(-2)
        | ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
    )
    single_inputs = torch.randn(batch_size, token_count, 384, device=device, generator=generator)
    unmasked_pair = torch.randn(batch_size, token_count, token_count, 128, device=device, generator=generator)
    pair = unmasked_pair * cross_mask.unsqueeze(-1)
    coordinates = torch.randn(batch_size, atom_count, 3, device=device, generator=generator)
    representative = torch.tensor([[1, 5, 6, 7]], device=device)
    reference_features = {
        "token_pad_mask": token_mask,
        "mol_type": molecule_type,
        "affinity_token_mask": ligand_mask,
        "token_to_rep_atom": torch.nn.functional.one_hot(representative, atom_count).float(),
    }

    results: list[bool] = []
    expected_heads: list[dict[str, Tensor]] = []
    implementation_heads: list[Boltz2AffinityHead] = []
    for module_name, config_name in (
        ("affinity_module1", "affinity_model_args1"),
        ("affinity_module2", "affinity_model_args2"),
    ):
        reference = AffinityModule(384, 128, **hyper_parameters[config_name])
        reference.load_state_dict(checkpoint_subset(state, module_name), strict=True)
        reference.to(device).eval()

        config = hyper_parameters[config_name]
        implementation = Boltz2AffinityHead(
            num_blocks=config["pairformer_args"]["num_blocks"],
            dropout=config["pairformer_args"]["dropout"],
            operator_backend="eager",
        )
        converted = {
            new_key: new_value
            for key, value in checkpoint_subset(state, module_name).items()
            for new_key, new_value in convert_affinity_head(key, value)
        }
        implementation.load_state_dict(converted, strict=True)
        implementation.to(device).eval()

        with torch.inference_mode():
            expected = reference(single_inputs, pair, coordinates, reference_features)
            actual = implementation(
                single_inputs=single_inputs,
                pair=pair,
                pred_coords=coordinates,
                token_to_rep_atom=representative,
                token_mask=token_mask,
                ligand_mask=ligand_mask,
                receptor_mask=molecule_type == 0,
            )
        expected_heads.append(expected)
        implementation_heads.append(implementation)
        results.append(
            compare(
                f"{module_name}.affinity_pred_value",
                actual["affinity_pred_value"],
                expected["affinity_pred_value"],
            )
        )
        results.append(
            compare(
                f"{module_name}.affinity_logits_binary",
                actual["affinity_logits_binary"],
                expected["affinity_logits_binary"],
            )
        )
        del reference

    class FixedInputEmbedder(nn.Module):
        def __init__(self, output: Tensor) -> None:
            super().__init__()
            self.register_buffer("output", output)

        def forward(self, **_: Tensor | None) -> Tensor:
            return self.output

    residue_type = torch.nn.functional.one_hot(
        torch.tensor([[2, 22, 22, 22]], device=device),
        33,
    ).float()
    affinity_mw = torch.tensor([44.097], device=device)
    batch = Boltz2Input(
        token_mask=token_mask,
        residue_index=torch.zeros(batch_size, token_count, dtype=torch.long, device=device),
        chain_index=torch.tensor([[0, 1, 1, 1]], device=device),
        entity_index=torch.tensor([[0, 1, 1, 1]], device=device),
        symmetry_index=torch.zeros(batch_size, token_count, dtype=torch.long, device=device),
        token_index=torch.arange(token_count, device=device).unsqueeze(0),
        atom_features=torch.zeros(batch_size, atom_count, 388, device=device),
        atom_to_token=torch.zeros(batch_size, atom_count, dtype=torch.long, device=device),
        atom_mask=torch.ones(batch_size, atom_count, dtype=torch.bool, device=device),
        reference_positions=torch.zeros(batch_size, atom_count, 3, device=device),
        reference_space_id=torch.zeros(batch_size, atom_count, dtype=torch.long, device=device),
        representative_atom_mask=torch.zeros(batch_size, atom_count, dtype=torch.bool, device=device),
        atom_token_index=torch.zeros(batch_size, atom_count, dtype=torch.long, device=device),
        token_to_rep_atom=representative,
        residue_type=residue_type,
        msa_profile=residue_type,
        deletion_mean=torch.zeros(batch_size, token_count, device=device),
        token_bonds=torch.zeros(batch_size, token_count, token_count, device=device),
        type_bonds=torch.zeros(batch_size, token_count, token_count, dtype=torch.long, device=device),
        contact_conditioning=torch.nn.functional.one_hot(
            torch.zeros(batch_size, token_count, token_count, dtype=torch.long, device=device),
            5,
        ),
        contact_threshold=torch.zeros(batch_size, token_count, token_count, device=device),
        chain_type=molecule_type,
        molecule_type=molecule_type,
        affinity_token_mask=ligand_mask,
        affinity_mw=affinity_mw,
        affinity_msa_profile=residue_type,
        affinity_deletion_mean=torch.zeros(batch_size, token_count, device=device),
    )
    representations = _Boltz2Representations(
        single_inputs=single_inputs,
        single=torch.zeros_like(single_inputs),
        pair=unmasked_pair,
        distogram_logits=torch.zeros(batch_size, token_count, token_count, 64, device=device),
        relative_position_encoding=torch.zeros_like(unmasked_pair),
    )
    model = Boltz2AffinityModel.__new__(Boltz2AffinityModel)
    nn.Module.__init__(model)
    model.input_embedder = FixedInputEmbedder(single_inputs)
    model.affinity_module1, model.affinity_module2 = implementation_heads

    expected_value = (expected_heads[0]["affinity_pred_value"] + expected_heads[1]["affinity_pred_value"]) / 2
    expected_probability = (
        torch.sigmoid(expected_heads[0]["affinity_logits_binary"])
        + torch.sigmoid(expected_heads[1]["affinity_logits_binary"])
    ) / 2
    with torch.inference_mode():
        full = model._score_affinity(representations, batch, coordinates.unsqueeze(1))
        corrected = model._score_affinity(
            representations,
            batch,
            coordinates.unsqueeze(1),
            molecular_weight_correction=True,
        )
    results.extend(
        (
            compare("combined.affinity_pred_value", full["affinity_pred_value"], expected_value.unsqueeze(1)),
            compare(
                "combined.affinity_probability_binary",
                full["affinity_probability_binary"],
                expected_probability.unsqueeze(1),
            ),
            compare(
                "combined.corrected_affinity_pred_value",
                corrected["affinity_pred_value"],
                (1.03525938 * expected_value - 0.59992683 * affinity_mw.pow(0.3).unsqueeze(-1) + 2.83288489).unsqueeze(
                    1
                ),
            ),
        )
    )

    raise SystemExit(0 if all(results) else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=boltzgen_checkpoint("boltz2_aff.ckpt"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
