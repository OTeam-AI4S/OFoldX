"""CIF numeric syntax shared by source columns and declaration consumers.

CIF 1.1 places standard uncertainty after the exponent (1.23e2(4));
PDBx float items also permit it before the exponent (1.23(4)e2).
Both describe 123 with standard uncertainty 4. Neither permits quoted
numbers, empty uncertainties, NaN, infinity or Python's digit separators.
"""

import math
import re

import gemmi

_SU = re.compile(
    r"(?P<m>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))"
    r"(?:\((?P<before>[0-9]+)\))?(?P<e>[eE][+-]?[0-9]+)?"
    r"(?:\((?P<after>[0-9]+)\))?"
)
_INTEGER = re.compile(r"[+-]?[0-9]+")


def number_with_uncertainty(token: str) -> tuple[float, float | None]:
    match = _SU.fullmatch(token)
    if match is None or (match["before"] is not None and match["after"] is not None):
        raise ValueError(f"invalid CIF number {token!r}")
    mantissa, exponent = match["m"], match["e"] or ""
    value = gemmi.cif.as_number(mantissa + exponent)
    if not math.isfinite(value):
        raise ValueError(f"CIF number is outside finite float64 range: {token!r}")
    digits = match["before"] or match["after"]
    if digits is None:
        return value, None
    scale = (int(exponent[1:]) if exponent else 0) - (len(mantissa.split(".")[1]) if "." in mantissa else 0)
    uncertainty = float(digits + "e" + str(scale))
    if not math.isfinite(uncertainty) or (int(digits) != 0 and uncertainty == 0):
        raise ValueError(f"CIF standard uncertainty is outside float64 range: {token!r}")
    return value, uncertainty


def integer(token: str) -> int:
    if _INTEGER.fullmatch(token) is None:
        raise ValueError(f"invalid CIF integer {token!r}")
    return int(token)
