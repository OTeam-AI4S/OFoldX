# Model card notes

Design/limitation notes to fold into the eventual model-card refactor. Each entry
records a place where OFoldX deliberately diverges from a released reference, the
measured residual, and what full parity would cost.

## AlphaFold3 — `fraction_disordered` / `ranking_score` accessibility backend

**Where:** `ofoldx/models/alphafold3/model.py` — `_dssp_accessible_area`,
`_windowed_solvent_accessible_area`, `fraction_disordered`, `ranking_score`.

**Divergence:** Released AF3 derives per-residue solvent accessibility (AlphaFold-RSA,
the input to `fraction_disordered` and hence the disorder term of `ranking_score`)
from `mkdssp` — DSSP's numerical **Lee & Richards** accessibility on each protein
chain in isolation. OFoldX bundles no DSSP binary, so it substitutes **biotite's
Shrake-Rupley SASA** (`biotite.structure.sasa`, 1.4 Å water probe, ProtOr heavy-atom
radii from Tsai 1999, the accessibility model the Sander & Rost max-ASA normalization
was calibrated against), summed per residue over each chain's observed heavy atoms.
The 25-residue reflect-padded window average and the 0.581 rASA disorder cutoff are
identical to upstream.

**Residual vs real `mkdssp`** (validated against `alphafold3.cpp.mkdssp.get_dssp`,
`calculate_surface_accessibility=True`, run in a real-AF3 env; ground truth is
bit-identical across two independent regenerations) on 7 PDBs
(1UBQ, 6LYZ, 1CBS, 1L2Y, 1F0R, 2N9R, 2RVB):

| backend | per-res rASA max_abs | per-res rASA mean_abs | fraction_disordered exact | fraction_disordered max_abs | ranking_score max_abs |
|---|---|---|---|---|---|
| **biotite Shrake-Rupley (kept)** | **0.0252** | **0.0049** | **5/7** | **0.0526** | **0.0263** |
| freesasa Lee-Richards (evaluated) | 0.0260 | 0.0053 | 4/7 | 0.0526 | 0.0263 |

The residual comes almost entirely from the accessibility *model* (SASA radii /
atom set / DSSP's exact accessibility definition), not the numerical *algorithm*:
per-residue windowed rASA from Shrake-Rupley and Lee-Richards agree to ~0.002, well
below the ~0.02–0.026 gap either has to `mkdssp`. Because `fraction_disordered`
smooths over a 25-residue window and then thresholds, per-residue SASA differences
at this level mostly wash out; the two mismatches (2N9R, 1L2Y/2RVB) are borderline
residues sitting within ~0.02 rASA of the 0.581 cutoff.

**freesasa was evaluated and rejected.** `freesasa` (2.2.1) installs cleanly into
the AF3 env and computes Lee & Richards SASA — the same algorithm family as DSSP —
so it was the leading candidate to close the gap. It did not: with ProtOr radii it
matched biotite to the 3rd decimal per residue and slightly regressed the aggregate
metrics (rASA max_abs 0.0252 → 0.0260; `fraction_disordered` exact 5/7 → 4/7, as
1L2Y flips 0.20 → 0.15). Its native radii sets fared no better (per-chain rASA
max_abs: naccess 0.0298, protor 0.0260, oons 0.0287, all ≥ biotite's 0.0252).
Per the owner ruling ("keep freesasa only if it measurably improves the match"),
the biotite Shrake-Rupley backend is retained and freesasa was uninstalled.

**Full parity cost:** Bit-exact agreement with released AF3 would require running
the actual DSSP Lee-Richards accessibility, i.e. bundling / shelling out to a
`mkdssp` binary. The owner declined to bundle a DSSP binary. No pure-Python package
reproduces DSSP's accessibility to better than the ~0.02–0.026 rASA residual above,
so the biotite substitute is the accepted trade-off; its impact on the downstream
`ranking_score` is bounded at ≤0.027 on this set.
