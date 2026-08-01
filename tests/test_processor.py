from __future__ import annotations

import torch

from ofoldx.processor import Processor


class TestProcessor:
    def test_placement(self) -> None:
        processor = Processor()
        assert processor.device == torch.device("cpu")
        assert processor.dtype == torch.float32

        processor = Processor(device="meta", dtype=torch.float64)
        assert processor.to() is processor
        assert processor.device == torch.device("meta")
        assert processor.dtype == torch.float64

        processor.to(device="cpu", dtype=torch.float32)
        assert processor.device == torch.device("cpu")
        assert processor.dtype == torch.float32
