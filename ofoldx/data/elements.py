r"""
Canonical element symbols and the source-level unknown atom type.
"""

from __future__ import annotations

_ELEMENTS = tuple(
    (  # noqa: SIM905  # periodic-table literal reads better as a whitespace-split string
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
        "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
        "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
        "At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
        "Cn Nh Fl Mc Lv Ts Og"
    ).split()
)

#: Element symbols mapped to atomic numbers. ``X`` preserves a source atom whose
#: element is explicitly unknown; it is not inferred to be any real element.
ATOMIC_NUMBER_BY_ELEMENT = {
    "X": 0,
    **{element: index for index, element in enumerate(_ELEMENTS, start=1)},
}

#: Symbols never inferred to be the metal center of a coordination bond: the unknown
#: atom type, non-metals, noble gases, and metalloids. Every other element is a metal.
NON_METAL_ELEMENTS = frozenset(
    "X H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split()  # noqa: SIM905
)
