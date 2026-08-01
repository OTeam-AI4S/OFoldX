"""Array snapshots whose numeric storage cannot be made writable by consumers."""

import numpy as np


def immutable_array(values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("immutable_array requires a dtype without Python objects")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
