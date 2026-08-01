"""The TensorClass surface used by OFoldX, with field-derived constructors.

TensorDict's constructor stub accepts untyped kwargs. Keep field checking here
without copying each molecular container's schema or replacing runtime methods.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self, dataclass_transform

from torch import Tensor

@dataclass_transform(field_specifiers=(field,))
@dataclass
class TensorClass:
    batch_size: Sequence[int] = field(default=(), kw_only=True)

    def replace(self, **changes: object) -> Self: ...
    def state_dict(self) -> dict[str, Tensor]: ...
