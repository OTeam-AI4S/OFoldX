"""Stable source metadata codes, independent of tensor storage and models.

Attributes:
    RESOLUTION_KIND_UNKNOWN (int): Resolution value with no specified source category.
    RESOLUTION_KIND_REFINEMENT (int): Resolution from the CIF item ``_refine.ls_d_res_high``.
    RESOLUTION_KIND_REFLECTIONS (int): Resolution from the CIF item ``_reflns.d_resolution_high``.
    RESOLUTION_KIND_EM_RECONSTRUCTION (int): Resolution from the CIF item ``_em_3d_reconstruction.resolution``.
    RESOLUTION_KIND_PDB_REMARK_2 (int): Resolution from a native PDB REMARK 2 record.
"""

RESOLUTION_KIND_UNKNOWN = 0
RESOLUTION_KIND_REFINEMENT = 1
RESOLUTION_KIND_REFLECTIONS = 2
RESOLUTION_KIND_EM_RECONSTRUCTION = 3
RESOLUTION_KIND_PDB_REMARK_2 = 4
