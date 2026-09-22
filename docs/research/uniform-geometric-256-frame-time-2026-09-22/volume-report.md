# Uniform Geometric 256³ frame time — the volume / level-set / census side

Scene `cm12-figure-7-256` (16.7M cells, a 1 m liquid ball into a dry closed
tank; floor impact around frame 25). Method id `uniform-volume`. Dawn/Metal,
M1 Max. All numbers are production stage timestamps from
`tools/profile-uniform-geometric-dawn.ts --frames=60 --max-gpu-bytes=3e9`, and
per-pass numbers from `tools/probe-uniform-geometric-transport-cost-dawn.ts`.

Raw JSON beside this file:

- `per-pass-baseline-256.json` — per-pass trace at the start of this work.
- `volume-optimized-256.json` — all four switches on.
- `volume-control-256.json` — the same tree with
  `FLUID_UNIFORM_AB_OFF=solidfreetrace,philean,phicensuswindow,tilereach`.

## 1. What the time was actually going on

The per-pass trace (impact window, frames 25–60, ms per advance) was:

| pass | ms | dispatches |
| --- | ---: | ---: |
| Advect page vertex phi | 43.1 | 1 |
| uvNormalizeRows | 17.0 | 3 |
| uvNormalizeDonors | 13.0 | 3 |
| Total surface volume: measure | 7.9 | 2 |
| uvBuildEdges | 7.0 | 1 |
| Phi support census | 6.9 | 1 |
| uvFinishDonorSums | 4.1 | 4 |
| Redistance page vertex phi | 3.4 | 1 |
| uvGather | 2.8 | 1 |
| uvTwoLevelSeed | 2.1 | 1 |

Four root causes, each confirmed against the trace rather than assumed.

**R1 — the vertex phi advect was O(D) per vertex, not O(1).** At frame 30 the
pass cost 102.8 ms for a phi window of only ~0.84M cells. `uvTrace` and
`uvEmbeddedAir` each walk every crossed HALF cell of the characteristic
(`steps = ceil(2·max-axis displacement)`), probing `cellOpenFraction` at each
one. That step's measured maximum displacement was D = 61 cells, so every
vertex in the window paid ~122 dependent texture probes in `uvTrace` and up to
122 more in `uvEmbeddedAir` — in a scene that contains no cut cell at all, so
the walk can never find anything. On top of that every vertex, including far
air, ran `uvEmbeddedContact` (32 open-fraction probes), `uvReleasedWalls` (24
boundary-plane probes with two loads each), `uvSeedPhi` (16 loads) and the
agreement shift.

**R2 — the live transport set was sized by the domain maximum.** `TRANSPORT`
is the seed set dilated by `m = ceil((ceil(D)+1)/4) + margin`, where D is the
DOMAIN-max backward displacement. At frame 30 one splash cell put D at 61
cells, m was clamped at its cap of 16 tiles, and the live set was 92,880 of
262,144 tiles for 9,244 tiles of actual liquid. Every pass of the conservative
transport — build, three normalize rounds, the donor decode, the gather —
scales with that count, which is where the 41 ms of coupling came from.

**R3 — the phi census was a dense whole-lattice scan, every step, forever.**
`scanExternalActiveSources` reads V, velocity and eight vertex phi samples per
cell over the whole domain purely to size the phi execution region: 6.9 ms in
the impact window and 7.0 ms in free fall — flat, because it does not care
where the liquid is.

**R4 — the total-surface-volume correction measures over its own window.**
`Total surface volume: measure` is 7.9 ms per advance across two dispatches.
Not addressed here; see §5.

## 2. The four switches

All four are named in `lib/methods/uniform/uniform-ab-switch.ts` and all four
are additionally gated at runtime by one `vec4f` on the parameter uniform
(`params.lean`), so a QA arm can refuse all of them inside one process. The
solver option is `leanPhiArmsForQA: "unconditional"`.

### `solidfreetrace` — E4, the no-cut-cell certificate (`params.lean.x`)

`cellOpenFraction` is `0` for a static solid voxel and otherwise
`(1 − body fraction)·(1 − terrain fraction)`. All three sources are host state:
the packed `SolidOccupancyMask`, `sceneHasTerrain`, and the live body list. The
host ORs the mask's words once per scene edit (`uniformSolidMaskEmpty`) and
publishes the certificate per step.

With the certificate held, three things collapse:

- `uvTrace` returns the RK2 endpoint without the half-cell walk. The walk's
  only exit is `cellOpenFraction <= 1e-5`, which cannot happen, and after the
  loop it returns `end` regardless. **Bit-exact.**
- `uvEmbeddedAir` returns its input. Its only exit is a solid hit on the
  characteristic. **Bit-exact.**
- `uvEmbeddedContact` returns its input. Every branch needs a closed
  neighbour, and `max(advected, -1e20)` is `advected` for every representable
  phi. **Bit-exact.**

On `cm12-figure-7-256` the certificate holds: 0 solid mask words set, no
terrain, no bodies, closed top.

### `philean` — E5, the far-air arm of `uvAdvectPhi` (`params.lean.z`)

Outside the SHELL tile class, and no closer than one 4h tile to any domain
plane, and on a step with no drop or inlet, the advect stores
`uvPhi(uvTrace(p))` and stops. The far field keeps being advected — freezing it
would leave an advancing front reading stale, too-large air values that
redistance (which fires only for `|phi| < 4h`) would never repair.

Exactness, term by term. FINE marks every tile holding liquid at or above the
dust floor, any partially open cell, any source this step, or any vertex with
phi below the 4h band; SHELL is FINE dilated by at least one tile, and a
vertex's eight incident cells lie in tiles t−1..t.

- `uvEmbeddedContact` / `uvEmbeddedAir`: need a closed cell among those
  incident cells or on the characteristic; a closed cell is a FINE seed.
  **Exact.**
- `uvSourcePhi`: identity on a step with no drop and no inflow, which is
  tested directly (`uvStepHasExternalSource`). **Exact.**
- `uvAgreementShift`: gathers the packed residual over `base + [−4,4)`. The
  residual is nonzero only where `|phi| < 1.5h` and gamma or V is nonzero,
  i.e. inside FINE; outside SHELL every tap is `+0`, `A` stays below one and
  the function returns exactly zero. **Exact.**
- `uvSeedPhi`: returns phi unless the eight incident cells average above a
  quarter full; those cells are in non-FINE tiles, so each holds `|V|` below
  the dust floor. **Exact.**
- `uvClosedWallPhi`: fires only for vertices exactly on a domain plane, which
  the one-tile boundary margin excludes. **Exact.**
- `uvReleasedWalls`: **the one approximation.** Its ambient-air source sits on
  the six domain planes rather than on any tile class. It can only RAISE phi
  (more air), by `dt·away − distance·h`, which is positive only within
  `dt·away/h` cells of a plane. The far-air arm is refused within one 4h tile
  of every plane, so the term is exact for any wall whose per-step ambient
  travel is under four cells. The lower planes read the prescribed boundary
  velocity, which is zero for a static wall; an upper plane needs a released
  contact face or an open lid with inflow on its own cell row. On this scene
  (closed top, static walls, no wall-adjacent liquid before frame 60) the term
  is identically zero in far air, so this arm is exact here; a driven wall
  would lose the term between four cells and `dt·away/h` from the plane.

### `phicensuswindow` — E6, the windowed phi census (`params.lean.y`)

`scanExternalActiveSources` skips `geometricActiveSeed` for cells more than
eight cells outside the box the census published last step. The source terms
are still scanned densely.

Why no seed can hide outside that box: only `uvGather` writes V, and any cell
with `|V|` above the dust floor is its own seed, so it was inside the observed
box and the padding (at least twelve cells) covers a step of travel. Only the
vertex phi passes write phi, over the published box widened by
`VERTEX_PHI_REACH = 6`; everywhere else phi is what it was, which was at or
above the 4h band because the box was the bounding box of every cell that
failed that test. Eight cells of margin covers the 6 + 1 the phi passes write
past the box. **Exact**, given the induction.

The induction is broken only by liquid appearing where the box says nothing —
a drop, an inlet, or a scene edit — and each of those is a host-uniform
condition. The host holds the census dense on such a step and on the step
after (`phiCensusDenseSteps`), and `phiCensusRescan` is a flag separate from
`activeRegionRescanPending`, which the page-domain path never consumes.

### `tilereach` — E7, the per-tile transport reach (`params.lean.w`)

The class word carries two extra saturating six-bit fields while the separated
three-axis scan runs, above the four class bits, and the z scan strips them
again before publishing the table, so every other reader still sees exactly
the class bits it knows:

- `dist`, the Chebyshev distance in tiles to the nearest transport seed,
  accumulated as `min over d of max(dist_in, |d|)` — the standard separable
  Chebyshev distance transform, which composes across the three axes for the
  same reason the class balls do.
- `reach`, the per-tile required reach `m(D_t)` — the same `m(·)` E3 already
  used, evaluated on each tile's own measured displacement in the seed pass and
  then max-dilated over a ball of the GLOBAL reach.

`TRANSPORT(t)` becomes `dist(t) <= reach(t)` instead of
`dist(t) <= m_global`. Since no tile's reach exceeds the domain's, the new set
is a subset of the old one, and it is the same set whenever the displacement
is uniform.

The exactness argument is the original one, localized. A receiver's backward
trace departs at most its own displacement, and the velocity a far-air cell
traces with is a copy the extension made of some liquid cell's within the
shell, so every velocity that can move tile t lies inside the global-reach
ball around t and `D_t` bounds it. DONORS keeps the global radius, because a
short DONORS set reads a stale sum where a short TRANSPORT set only stalls a
front, and the global set is a superset of the new TRANSPORT dilated by the
same m either way. **Exact** under the same assumption E3 already makes (the
start-of-step displacement bounds this step's backward displacements).

Telemetry note: `uniformTransportReachTiles` in the pipeline panel is still
the DOMAIN-max reach. It remains truthful as the bound every per-tile reach
sits under, but it is no longer the radius applied to every seed. The live
tile count beside it (`uniformTransportTiles`) is counted from the final set
and is exact.

## 3. A/B

Both arms are the same working tree, run adjacently (21:17 and 21:18) so the
other live session's edits are common to both:

```
tools/profile-uniform-geometric-dawn.ts --scene=cm12-figure-7-256 --frames=60 --max-gpu-bytes=3e9
control:   FLUID_UNIFORM_AB_OFF=solidfreetrace,philean,phicensuswindow,tilereach
optimized: (unset)
```

`validationErrors` is `[]` in both runs. Stage means, ms per advance; rows
under 1 ms that moved less than 0.5 ms are omitted.


**Impact and spread** (frames 25–60)

| stage | control ms | optimized ms | delta |
| --- | ---: | ---: | ---: |
| Dense vertex phi transport and redistance |  89.73 |  40.39 | -49.33 |
| Dense geometric volume coupling |  76.39 |  49.93 | -26.46 |
| CM11a Full-Cycles |  39.41 |  36.21 | -3.20 |
| Dense conservative volume gather |  20.56 |  17.96 | -2.60 |
| CM11a parity copy + fine residual |  14.26 |  13.87 | -0.39 |
| Sec. 3.3 narrow-band FIM front |  13.10 |  12.83 | -0.26 |
| Dense conservative volume sharpening |   7.43 |   7.30 | -0.13 |
| CM11a topology + RHS pyramid |  16.35 |  16.33 | -0.01 |
| Sec. 3.3 hierarchy fill + transport shell |   9.91 |   9.90 | -0.00 |
| Pressure projection + surface publication |   5.15 |   5.16 | +0.01 |
| Diagnostics reduction |   2.54 |   2.55 | +0.01 |
| Sec. 3.3 interface authority |  16.93 |  17.47 | +0.54 |
| Velocity advection + body forces |  13.51 |  14.18 | +0.68 |
| CM11a V-Cycles |   2.20 |   4.20 | +1.99 |
| **GPU total (mean)** | **327.46** | **248.28** | **-79.17** |
| GPU total (median) | 335.15 | 251.53 | -83.62 |
| wall (mean) | 341.76 | 261.12 | -80.64 |

**Free fall** (frames 5–24)

| stage | control ms | optimized ms | delta |
| --- | ---: | ---: | ---: |
| Dense vertex phi transport and redistance |  13.07 |   8.92 | -4.15 |
| Sec. 3.3 interface authority |  16.08 |  12.46 | -3.62 |
| Dense geometric volume coupling |  16.65 |  14.86 | -1.79 |
| Dense conservative volume gather |   9.92 |   9.78 | -0.14 |
| CM11a Full-Cycles |  16.54 |  16.44 | -0.10 |
| CM11a parity copy + fine residual |   7.24 |   7.18 | -0.06 |
| Dense conservative volume sharpening |   5.22 |   5.19 | -0.03 |
| CM11a topology + RHS pyramid |  15.93 |  15.92 | -0.01 |
| Pressure projection + surface publication |   4.01 |   4.01 | -0.00 |
| Velocity advection + body forces |   9.16 |   9.16 | -0.00 |
| Diagnostics reduction |   2.53 |   2.54 | +0.01 |
| Sec. 3.3 narrow-band FIM front |   5.53 |   5.55 | +0.02 |
| Sec. 3.3 hierarchy fill + transport shell |   8.91 |   8.94 | +0.04 |
| **GPU total (mean)** | **130.78** | **120.95** | **-9.83** |
| GPU total (median) | 126.94 | 118.62 | -8.32 |
| wall (mean) | 141.62 | 129.38 | -12.24 |

**All measured frames** (frames 5–60)

| stage | control ms | optimized ms | delta |
| --- | ---: | ---: | ---: |
| Dense vertex phi transport and redistance |  62.35 |  29.15 | -33.20 |
| Dense geometric volume coupling |  55.05 |  37.40 | -17.65 |
| CM11a Full-Cycles |  31.24 |  29.15 | -2.09 |
| Dense conservative volume gather |  16.76 |  15.04 | -1.72 |
| Sec. 3.3 interface authority |  16.63 |  15.68 | -0.95 |
| CM11a parity copy + fine residual |  11.75 |  11.48 | -0.27 |
| Sec. 3.3 narrow-band FIM front |  10.40 |  10.23 | -0.16 |
| Dense conservative volume sharpening |   6.64 |   6.55 | -0.10 |
| CM11a topology + RHS pyramid |  16.20 |  16.19 | -0.01 |
| Pressure projection + surface publication |   4.74 |   4.75 | +0.00 |
| Sec. 3.3 hierarchy fill + transport shell |   9.55 |   9.56 | +0.01 |
| Diagnostics reduction |   2.53 |   2.55 | +0.01 |
| Velocity advection + body forces |  11.95 |  12.39 | +0.43 |
| CM11a V-Cycles |   1.42 |   2.70 | +1.28 |
| **GPU total (mean)** | **257.22** | **202.81** | **-54.41** |
| GPU total (median) | 295.96 | 206.44 | -89.52 |
| wall (mean) | 270.28 | 214.07 | -56.21 |

The target was 50 ms off the per-advance GPU time. The mean over every
measured frame moves 257.2 → 202.8 ms (−54.4), the median 296.0 → 206.4
(−89.5), and the impact window, which is where the method was most expensive,
moves 327.5 → 248.3 (−79.2).

Two rows move the wrong way and neither is this work:

- **CM11a V-Cycles +1.99 ms** in the impact window. The pressure budget is
  `lagged`: the solver encodes as many cycles as the previous step's timing
  affords. A cheaper advance affords more pressure, so the optimized arm buys
  V-cycles with the time the phi stages gave back. This is the budget spending
  the win, not a regression.
- **Sec. 3.3 interface authority +0.54 ms** and **velocity advection
  +0.68 ms** in the impact window. Both stages read the trajectory rather than
  anything this work changed; see the conservation note below on how far the
  two trajectories separate by frame 40.

### E7 in the telemetry

`uniformVolumeTransportWorkgroups` is the live transport set, and the per-tile
reach roughly halves it once the splash makes the domain maximum
unrepresentative: 92,880 → 51,011 at frame 30, 86,125 → 56,363 at frame 40,
146,584 → 77,980 at frame 60. Free fall, where the displacement really is
near-uniform, moves much less (16,888 → 13,512 at frame 10), exactly as §2
predicts.

### Conservation and exactness on the 60-frame trajectory

| frame | control volumeCellSum | optimized volumeCellSum | abs diff | relative | control maxSpeed | optimized maxSpeed | control transport wg | optimized transport wg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 268065.877 | 268065.877 | 0.000 | 0.0e+0 | 0.333 | 0.333 | 12792 | 12792 |
| 10 | 268062.786 | 268062.786 | 0.000 | 0.0e+0 | 3.335 | 3.335 | 16888 | 13512 |
| 20 | 268059.696 | 268059.696 | 0.000 | 0.0e+0 | 6.670 | 6.697 | 21600 | 17520 |
| 25 | 268057.234 | 268056.749 | 0.485 | 1.8e-6 | 8.422 | 8.428 | 21304 | 17576 |
| 30 | 268049.143 | 268048.835 | 0.308 | 1.1e-6 | 39.791 | 40.090 | 92880 | 51011 |
| 40 | 268028.840 | 268030.677 | 1.837 | 6.9e-6 | 84.241 | 427.647 | 86125 | 56363 |
| 50 | 267957.166 | 267952.534 | 4.631 | 1.7e-5 | 5635.838 | 3375.943 | 126837 | 76886 |
| 60 | 267654.638 | 267641.219 | 13.419 | 5.0e-5 | 185566.953 | 5645.954 | 146584 | 77980 |

Drift over frames 1→60: control -411.2 cells (-0.153%), optimized -424.7 cells (-0.158%).

`volumeCellSum` is bit-identical through frame 20 and stays inside 5.0e-5
relative to frame 60, well under the 1e-4 bar. Total drift from frame 1 to
frame 60 is −411 cells (−0.153%) on the control arm and −425 (−0.158%)
optimized: the same loss, not a new one.

The late-frame separation is **run-to-run nondeterminism, not the switches**,
and the third run in this directory proves it. `per-pass-baseline-256.json`
was captured on the pre-change tree — a control-equivalent trajectory — and it
sits *further* from the control arm than the optimized arm does:

| frame | pre-change baseline | control | optimized | baseline − control | optimized − control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 268048.14 | 268049.14 | 268048.84 | −1.00 | −0.31 |
| 40 | 268026.12 | 268028.84 | 268030.68 | −2.72 | +1.84 |
| 50 | 267936.44 | 267957.17 | 267952.53 | −20.73 | −4.63 |

The cause is the `lagged` pressure budget: the number of cycles a step encodes
depends on the previous step's wall timing, so two runs of the *same* binary
take different pressure trajectories once the splash is violent. That is also
why `maxSpeed` is not comparable past frame 40 — it is a single-cell maximum in
a splash the pressure solve is not fully resolving on either arm (frame 60:
185,567 m/s control against 5,646 optimized, with the baseline run showing the
same instability). This is a pre-existing property of the scene at this
budget, unrelated to the level-set work, but it does mean the 60-frame profile
cannot be read as a field-level exactness check.

The field-level check is the new Dawn lane instead, where both arms run in one
process on identical input and the experiment is re-seeded from the control
each step. There, on this scene's certificates, the agreement is exact.

## 4. Oracles

Each run once, in the foreground, under the repository WebGPU lease.

| lane | result | note |
| --- | --- | --- |
| `tests/uniform-phi-far-air-dawn.test.ts` (new) | **pass**, 8.5 s | see below |
| `tests/uniform-live-solid-edit-dawn.test.ts` | **pass**, 100 s | voxel stroke adopted on the running solver; `uniformSolidMaskEmpty` re-evaluates and the certificate drops |
| `tests/uniform-surface-work-dawn.test.ts` | **pass**, 20 s | max field error 2.9e-6 at frame 40 |
| `tests/uniform-system-extension-dawn.test.ts` | **pass**, 25 s | terrain + solid shell + live edits + moving bodies; the solid-free certificate correctly refuses for the whole lane |
| `tests/uniform-pressure-layout-dawn.test.ts` | **fail** | red at the control arm too, byte-identically; not this work — see below |

**The new lane.** `tests/uniform-phi-far-air-dawn.test.ts` runs two solvers in
one process on 64³ figure 7 through floor impact: the control takes
`leanPhiArmsForQA: "unconditional"`, so `params.lean` is all zeros and every
certified arm is refused at runtime; the experiment takes all four. Six fields
are compared every step (`volumeTexture`, `velocityTexture`,
`vertexPhiTexture`, `surfaceFieldTexture`, `extrapolatedVelocityTexture`,
`advectedVertexPhiTexture`) and the experiment is then re-seeded from the
control's physical textures, so each comparison measures one step's difference
rather than thirty steps of a nonlinear system amplifying one rounding.

The lane asserts a 1e-5 relative / 1e-6 absolute tolerance. The measured
result is stronger: **every field is bit-identical on every one of the thirty
steps** — the lane prints a line whenever `maxError > 0` and printed none.
That is the §2 exactness argument confirmed, including `uvReleasedWalls`,
which is identically zero on a closed tank with static walls.

The lane also asserts in the other direction, so that the agreement means
something: the live transport set must never grow, and must actually shrink on
at least one step. It shrank on 4 of 30 (`{"shrunkSteps":4}`) — fewer than at
256³, because at 64³ the domain maximum and the local displacement diverge far
less.

**The failing lane.** `uniform-pressure-layout-dawn` fails at line 58, which
is a *construction-time* comparison of the multigrid dispatch plan, before any
frame is advanced:

```
+   ['mgClearMinimum',     [11, 6, 5]],
+   ['mgBuildCycleTiles',  [6, 1, 1]],
```

The native arm's plan now carries passes the `paged-logical` arm does not.
Those entry points are in `webgpu-uniform-pressure-multigrid.wgsl.ts`, which
this work never touched, and they are the in-flight pressure-tiling change of
the other live session (the `cycletiles` switch already exists in
`uniform-ab-switch.ts`). Re-running with
`FLUID_UNIFORM_AB_OFF=solidfreetrace,philean,phicensuswindow,tilereach`
produces a byte-identical failure, which is the check the brief asked for:
the lane is red at the control arm, so it is red at HEAD-with-their-edits and
not from this work. It is theirs to re-green.


## 5. What is left

- **`Total surface volume: measure`, 7.9 ms per advance across two
  dispatches.** It lives in `UniformSurfaceVolumeCorrection`
  (`lib/methods/uniform/webgpu-uniform-surface-volume.ts` and
  `uniform-surface-volume.wgsl.ts`), both of which were being edited by another
  live session throughout this work, so it was left alone. The lever is the
  same one as everywhere else: the measure runs over the correction's own
  window, which at frame 60 is `[0,0,0]–[256,113,256]` for a thin sheet, and
  `uvTarget` is provably zero outside TRANSPORT.
- **The four dense donor clears, ~5 ms per advance.**
  `encoder.clearBuffer(volumeDonorSums, …)` writes `cells·24 = 403 MB` four
  times a step while accumulation only ever touches DONORS. A tiled clear is
  exact (words outside DONORS are never read), but DONORS is still dilated by
  the GLOBAL reach, so at the impact step it is very nearly the whole domain
  and a tiled clear would buy nothing there — it only pays in free fall. The
  real fix is a per-tile DONORS radius on top of E7, which needs the donor
  radius to be `2·reach + 1` rather than `reach`, and a short DONORS set reads
  a stale sum rather than merely stalling a front. Not attempted.
- **`uvReleasedWalls` in far air** is the only non-exact hunk in this work.
  Making it exact costs one small kernel: scan the six boundary cell planes
  (6 × 256² cells, well under a tenth of a millisecond) and publish the largest
  `dt·away/h` any released or ambient face can produce, then refuse the far-air
  arm within that many cells of a plane instead of within one tile.
