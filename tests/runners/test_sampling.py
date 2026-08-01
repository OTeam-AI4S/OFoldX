# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

import json

from ofoldx.runners.sampling import SampleKey, SamplingEntry, WeightedEpochSampler, read_sampling_index


class TestWeightedEpochSampler:
    weights = (1.0, 4.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0)

    def test_determinism(self) -> None:
        first = WeightedEpochSampler(self.weights, 32, seed=7)
        second = WeightedEpochSampler(self.weights, 32, seed=7)

        assert list(first) == list(second)
        first.set_epoch(2)
        assert list(first) != list(second)

    def test_without_replacement(self) -> None:
        sampled = list(WeightedEpochSampler(self.weights, 6, replacement=False, seed=3))

        assert len({key.entry_index for key in sampled}) == 6

    def test_ordinals(self) -> None:
        sampler = WeightedEpochSampler((1.0,), 3)
        assert list(sampler) == [
            SampleKey(0, 0),
            SampleKey(0, 1),
            SampleKey(0, 2),
        ]
        sampler.set_epoch(1)
        assert [key.draw_index for key in sampler] == [3, 4, 5]

    def test_resume(self) -> None:
        sampler = WeightedEpochSampler(self.weights, 8, seed=17)
        sampler.set_epoch(3)
        iterator = iter(sampler)
        _ = [next(iterator) for _ in range(3)]
        state = sampler.state_dict()
        remaining = list(iterator)

        restored = WeightedEpochSampler(self.weights, 8, seed=17)
        restored.load_state_dict(state)

        assert list(restored) == remaining

    def test_distributed(self) -> None:
        ranks = [WeightedEpochSampler(self.weights, 5, seed=11, num_replicas=3, rank=rank) for rank in range(3)]
        global_sampler = WeightedEpochSampler(self.weights, 5, seed=11)

        rank_samples = [list(sampler) for sampler in ranks]
        reconstructed = [rank_samples[index % 3][index // 3] for index in range(6)]
        assert [len(sampler) for sampler in ranks] == [2, 2, 2]
        assert reconstructed[:5] == list(global_sampler)
        assert reconstructed[5].entry_index == reconstructed[0].entry_index
        assert reconstructed[5].draw_index == 5

    def test_distributed_nonreplacement(self) -> None:
        ranks = [
            WeightedEpochSampler(
                self.weights[:5],
                5,
                replacement=False,
                seed=13,
                num_replicas=3,
                rank=rank,
            )
            for rank in range(3)
        ]
        rank_samples = [list(sampler) for sampler in ranks]
        reconstructed = [rank_samples[index % 3][index // 3] for index in range(6)]

        assert len({key.entry_index for key in reconstructed[:5]}) == 5
        assert reconstructed[5].entry_index == reconstructed[0].entry_index


def test_read_index(tmp_path) -> None:
    path = tmp_path / "sampling.jsonl"
    rows = (
        {"structure": "second.cif", "sample_type": "interface", "instance_ids": ["A", "B"], "weight": 2.5},
        {"structure": "first.cif", "sample_type": "structure", "weight": 0.5},
    )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert read_sampling_index(path) == (
        SamplingEntry(str(tmp_path / "second.cif"), "interface", ("A", "B"), 2.5),
        SamplingEntry(str(tmp_path / "first.cif"), "structure", (), 0.5),
    )
