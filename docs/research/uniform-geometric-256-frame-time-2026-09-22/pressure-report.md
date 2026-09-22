# CM11a pressure: the finest per-cycle operators on a work list

Method `uniform-volume` (Uniform Geometric), scene `cm12-figure-7-256` (256³,
16.7M cells), Dawn/Metal on an M1 Max. Switch: `cycletiles`.

Raw data beside this file: `pressure-control-256.json`
(`FLUID_UNIFORM_AB_OFF=cycletiles`) and `pressure-cycletiles-256.json`, both
60 frames, `--max-gpu-bytes=3000000000`, captured back to back on the same
tree. Both arms carry the volume-side edits live in the tree at the time, so
compare the two columns with each other, not with earlier baselines.

## What changed

Level 0 of the CM11a hierarchy is 258³ texels: 8× the cells of L1 and 4096× the
coarsest. Every Full-Cycle and V-cycle ran ~10 dense 17M-thread passes over it
while 2-4% of the lattice is liquid. Those passes now run from a **cycle work
list** of 4³ tiles instead:

- `mgBuildSmoothTiles` already compacted liquid tiles for the smoother's later
  sweeps. It now also classifies each tile as constrained, and writes one
  classification word per tile after the list.
- A new `mgBuildCycleTiles` (one thread per tile, 27 neighbour words) dilates
  that to **liquid tiles ⊕ 1 tile, plus every constrained tile**, and
  `mgPublishSmoothTiles` publishes a second indirect record per level.
- Eight operators gained `…Tiles` entry points that take a tile from the list
  and address its 64 cells with 64 lanes: `mgResidual`, `mgProlongateAdd`,
  `mgProlongateAssign`, `mgCopyPressure`, `mgShiftMinimum`, `mgAddPressure`,
  `mgSaveAccepted`, `mgRestoreRejected`. Their bodies are shared string
  constants with the dense kernels, so the dense text is byte-identical to
  before — only *where* they run changed, never the arithmetic.
- The first sweep of every visit also runs from the cycle list (it was the one
  dense sweep left; sweeps 1..5 already used the liquid list).

Why one tile of dilation is the right reach: a coarse liquid cell always has a
liquid child, so its eight restriction taps and eight `p_min` taps lie within
one cell of liquid, and trilinear prolongation into a fine cell reads coarse
cells within one cell of its parent. A whole 4³ tile of margin covers all of
them.

Why constrained tiles are in the list even far from liquid: `mgDownsampleSubtract`
takes `max(p_min - p)` over the eight children, so a constrained child turns
*its* pressure into a coarse bound, and the first sweep of each visit is what
projects such a row back up to `p_min` after prolongation pushed it under.
Dropping them was measured: the `long-dam` layout fixture stopped converging
and blew past the 240 s lane timeout. The rule stays.

Three dense passes per solve seed the far field the list skips, so that the
passes which are still dense read exactly what they would have read:

- `mgClearMinimum` into the other `p_min` parity (`-FLT_MAX`): outside the list
  `mgShiftMinimum` writes nothing, and `p_min - p` there *is* `-FLT_MAX`.
- `mgClearPressure` into both finest residual destinations (`residual A` and
  `rhs B`). `mgRestrictResidual` averages all eight children unmasked, and a
  Full-Cycle restricts its correction rhs straight to the coarsest solve
  without recomputing a residual on the way, so a coarse row over the far
  field must see the zero `mgResidual` writes at air rows. Without this, a tile
  that left the list since the last solve hands that chain a stale residual —
  which is what blew the post-impact trajectory up during development.

These three run **after `mgBakeCoefficients`, not beside `mgBuildFinestRhs`**:
the shared scratch arena (`uniform-scratch-arena.ts`) lays the finest `V` over
`pressure B`, `rhs B`, `residual A` and `p-min B`, and `V` is only dead once the
coefficients are baked. Seeding them earlier silently corrupts the topology.

The finish section (8 recovery batches × 8 sweeps and their commits) keeps the
control's dense row kernels. It is gated off whenever the solve was accepted,
which is nearly always, so a list there only added indirect launches that
return immediately — it measured as +1.7 ms. Its dense restore also needs the
accepted-pressure field defined everywhere, so the prologue copy into it stays
dense too.

## Before / after

60-frame profile, GPU hardware timestamps, stage means in ms.

### Impact and spread (frames 25-60)

| stage | control | cycletiles | delta |
| --- | ---: | ---: | ---: |
| CM11a Full-Cycles | 36.72 | 22.73 | **-14.00** |
| CM11a topology + RHS pyramid | 15.62 | 16.69 | +1.08 |
| CM11a parity copy + fine residual | 6.23 | 6.16 | -0.07 |
| **all CM11a pressure stages** | **58.57** | **45.58** | **-12.98** |
| total advance GPU (mean) | 199.8 | 187.6 | -12.3 |
| total advance GPU (median) | 204.3 | 189.5 | -14.8 |

### All measured frames (5-60)

| stage | control | cycletiles | delta |
| --- | ---: | ---: | ---: |
| CM11a Full-Cycles | 33.25 | 20.47 | **-12.77** |
| CM11a topology + RHS pyramid | 15.44 | 16.53 | +1.08 |
| CM11a parity copy + fine residual | 6.21 | 6.04 | -0.17 |
| **all CM11a pressure stages** | **54.90** | **43.04** | **-11.85** |
| total advance GPU (mean) | 175.9 | 164.7 | -11.1 |

### Free fall (frames 5-24)

| stage | control | cycletiles | delta |
| --- | ---: | ---: | ---: |
| CM11a Full-Cycles | 26.99 | 16.42 | -10.57 |
| **all CM11a pressure stages** | 48.29 | 38.47 | -9.82 |
| total advance GPU (mean) | 132.7 | 123.5 | -9.2 |

The +1.08 ms on setup is the three seed passes plus `mgBuildCycleTiles` and the
second publish: 5 extra dispatches per solve
(`uniformPressurePassesEncoded` 766 → 771 at frame 10). Volume-side stages move
by up to ±0.8 ms between the two runs; those stages are untouched by this
change and that is run-to-run noise on a shared machine.

## List sizes

`uniformPressureSmoothingTiles` now carries `level` and `list` per record.

| frame | liquid L0 | cycle L0 | capacity |
| --- | ---: | ---: | ---: |
| 10 (free fall) | 4,940 | 31,860 (11.6%) | 274,625 |
| 60 (spread sheet) | 11,427 | 42,712 (15.6%) | 274,625 |

The cycle list is dominated by the constrained container shell (24,578 boundary
tiles at this resolution), not by the dilation. That is the remaining headroom:
a scheme that carried far constrained rows without listing them would cut the
per-cycle work another ~3×, but the direct attempt above broke convergence.

## Exactness

**The simulation is bit-identical.** Over 60 frames at 256³, `maxSpeed_m_s` and
`volumeCellSum` agree between the two arms to every printed digit in all 60
rows, and every pressure convergence field agrees
(`uniformPressureAcceptedResidual`, `uniformPressureInitialResidual`,
`uniformCM11aFineResidualInfinity`, `uniformPressureCyclesExecuted`,
`uniformPressureRejectedCycles`, `uniformPressureRecoverySweeps`,
`uniformCM11aFullCyclesExecuted`, …). The only telemetry that differs is
`uniformPressurePassesEncoded`/`Configured`, by the 5 added dispatches.
Zero validation errors in both arms. Peak live GPU bytes 2,972,294,052 →
2,974,825,836 (+2.5 MB, the two tile-list buffers).

**What does change numerically:** the *published pressure texture* over the far
field. Outside the cycle list the tiled arm leaves rows at the exact zero
`mgBuildFinestRhs` stored, where a dense arm accumulates the trilinear
prolongation of coarse **air** rows. Those rows are dead in both arms — the
smoother, `mgApply` and `mgResidual` mask non-liquid neighbours, the `p_min`
downsample saturates at `-FLT_MAX` whatever `p` is, and the projection reads
air pressure as `0` — which is exactly what the bit-for-bit velocity, volume and
vertex-phi comparison on a coupled trajectory establishes.

Measured in `tests/uniform-pressure-work-dawn.test.ts` (64³, dense vs tiled,
40 frames):

| frame | cells | dense-only cells | max dense-only \|p\| | field max \|p\| |
| --- | ---: | ---: | ---: | ---: |
| 1 | 287,496 | 0 | 0 | 0 |
| 2 | 287,496 | 1,784 | 2.06e-5 | 2.23e-3 |
| 5 | 287,496 | 2,072 | 2.37e-5 | 1.05e-2 |
| 24 | 287,496 | 4,464 | 2.20e-3 | 8.74 |
| 30 | 287,496 | 800 | 1.60e+2 | 3.29e+4 |
| 40 | 287,496 | 0 | 0 | 1.25e+4 |

Every differing texel holds exactly `0` in the tiled arm, and the field maximum
is never on one of them.

## Lanes

- `tests/uniform-pressure-work-dawn.test.ts` — green. `volumeTexture`,
  `velocityTexture` and `vertexPhiTexture` are compared bit for bit as before;
  the pressure comparison now states the far-field exception precisely (a
  difference is allowed only where the tiled arm is still that zero, and the
  field maximum may never sit on one) and logs the table above.
- `tests/uniform-pressure-layout-dawn.test.ts` — green. The schedule comparison
  ignores the list's own setup and the dense far-field seeds
  (`PlannedDispatch.cycleSetup`), compares the normalised operator sequence
  (`mgXxxTiles` → `mgXxx`) and still compares launches for every operator that
  is not listed. Its field comparison carries the same pressure exception.
- `tests/uniform-pressure-safety-dawn.test.ts` — 13/13 green.
