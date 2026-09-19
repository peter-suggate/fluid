# Uniform Geometric: what a tall air column costs

Investigation only. Nothing under `lib/` was changed. Every number here comes
from the fixture and probes in this folder; the raw captures are the JSON files
beside this report.

Question: *we want a large scene with a huge amount of air to cost what a small
scene with the same liquid volume costs.* This report says what the extra air
actually costs today, which pass it is spent in, and who reads the far-air
output of every stage that is still sized by the bounding box.

## 1. Setup

One dial: container height. The reservoir is pinned in absolute metres
(`TALL_AIR_RESERVOIR_M = 0.4 x 0.2 x 0.8 m`, `tall-air-scene.mts`), so every arm
holds the identical 32 x 16 x 64-cell liquid column on the identical floor
footprint, at the identical cell size, with the identical dt and parameters.
Only the air above it grows.

| arm | container height | lattice | cells | liquid cells at t=0 |
|---|---|---|---|---|
| A (1x) | 0.80 m | 64 x 64 x 64 | 262 144 | 35 904 |
| 2x | 1.60 m | 64 x 128 x 64 | 524 288 | 35 904 |
| B (4x) | 3.20 m | 64 x 256 x 64 | 1 048 576 | 35 904 |
| 8x | 6.40 m | 64 x 512 x 64 | 2 097 152 | 35 904 |

Method values are the shipped Uniform Geometric defaults
(`lib/methods/uniform/uniform-volume-method.ts`): `sharpeningWorkMap on`,
`twoLevelVelocity on` (`twoLevelFineReach 2`), `twoLevelExtension tiles`
(`twoLevelShellReach 1`), `twoLevelAdvection tiles`, `transportWorkMap tiles`
(`transportReach 1`), `redistance on`, `extensionFrontSweeps 2`,
`volumePressureRows off`, `liquidCapacityBalancing off`, `timeStep paper`
(1/30 s), `pressureCycleBudget lagged` with headroom 1 and residual tolerance
10 s^-1, `pressureFullCycles 3` / `pressureVCycles 4` / `pressureSweeps 6`,
`velocityTransport semi-lagrangian`.

**Same physics.** `volumeCellSum` is 32767.66 (A) against 32767.63 (B) at frame
70 -- 1e-6 relative -- and stays in step at every checkpoint across all four
heights. Median `maxSpeed` is 1.79 (A) / 1.74 (B) m/s, peak 3.53 m/s, whose
ballistic rise is 0.636 m against the 0.80 m lid of the *shortest* arm, so no
arm's liquid ever touches its ceiling. Live tile counts are equal in absolute
terms. Zero validation errors in all three captures.

## 2. Headline A/B

Arm A = 1x, arm B = 4x, lockstep in one process, 70 frames, 4 warmup.
GPU figures are medians of hardware timestamp seams; the Dawn tick is 65.5 us,
so every seam median is a multiple of 0.0655 ms.

| | A (64^3) | B (64x256x64) | delta | ratio |
|---|---|---|---|---|
| whole advance, GPU | **11.469 ms** | **16.974 ms** | +5.505 | **1.48x** |
| CPU encode | 11.000 ms | 11.237 ms | +0.237 | **1.02x** |
| wall | 22.846 ms | 28.418 ms | +5.572 | 1.24x |
| cells | 262 144 | 1 048 576 | | 4.00x |
| pressure levels | 6 | 6 | 0 | 1.00x |
| pressure plan passes | 2008 | 2008 | 0 | 1.00x |
| pressure passes encoded / step | 948 | 948 | 0 | 1.00x |
| pressure cycles encoded / executed | 2 / 1 | 2 / 1 | 0 | 1.00x |
| extension hierarchy levels | 6 | 8 | +2 | |
| extension passes per invocation | 20 | 24 | +4 | |
| FINE tiles (median) | 1595 / 4096 (38.9%) | 1601 / 16384 (9.8%) | +6 | 1.00x |
| SHELL tiles | 1888 | 1888 | 0 | 1.00x |
| TRANSPORT tiles | 1888 | 1896 | +8 | 1.00x |
| sharpening tiles | 426 | 422 | -4 | 1.00x |
| allocated GPU bytes | 114.7 MiB | 452.0 MiB | | 3.94x |

Four heights, GPU median per advance (two captures; the two 1x controls agree
to 1.1%, so the captures are comparable):

| arm | GPU ms | CPU encode ms | pressure levels | plan passes | encoded passes | ext levels | bytes |
|---|---|---|---|---|---|---|---|
| 1x | 11.47 / 11.60 | 11.00 / 11.26 | 6 | 2008 | 948 | 6 | 114.7 MiB |
| 2x | 13.37 | 11.48 | 6 | 2008 | 948 | 7 | 227.1 MiB |
| 4x | 16.97 | 11.24 | 6 | 2008 | 948 | 8 | 452.0 MiB |
| 8x | 30.34 | **24.90** | **9** | **4228** | **2199** | 9 | 901.9 MiB |

**The 8x cliff is structural, not gradual.** 64 x 512 x 64 fails
`lockstepLevels` (`webgpu-uniform-pressure-multigrid.ts:153-167`): its coarsest
haloed level would be 4 x 18 x 4 = 288 cells against
`UNIFORM_CM11A_COARSEST_LANES = 256` (`:128`). It falls back to
`semiCoarsenedLevels` (`:174-186`), which builds **9** levels, a **4228**-pass plan
and a 2199-pass encoded prefix -- so CPU encode, flat from 1x to 4x, suddenly
doubles. Up to 4x the hierarchy is height-independent; at 8x it is not.

## 3. `activeRegionEnabled` is false, and what that means

Verified as asked.

> **Line numbers in this report are against commit `743c2200`**, not the working
> tree. A second agent is editing `lib/methods/uniform/*` concurrently; at the
> time of writing it had grown that directory's diff from 335 to 422 insertions
> and had already rewritten the very line below. Cite with
> `git show 743c2200:<path>`.

- `webgpu-uniform-reference.ts:548`:
  `this.activeRegionEnabled = !this.geometricVolume && options.activeRegion === true && ...`.
  `uniform-volume-method.ts` forces `activeRegion: false` for Uniform Geometric
  *and* `geometricVolume` is true, so the flag is false twice over. It is
  structurally false for this method, not merely off by default.
  **As of this writing the concurrent agent has already changed this line** to
  `options.activeRegion === true && (!this.geometricVolume || this.volumeDustThreshold > 0) && process.env.FLUID_UNIFORM_ACTIVE_REGION !== "0"`,
  i.e. it is removing exactly the guard measured here. Everything below is the
  baseline with that path off.
- Consequence in the solver: `webgpu-uniform-reference.ts:1310-1316` --
  `dispatch()` takes `dispatchWorkgroupsIndirect` only when
  `activeRegionEnabled`; otherwise every `run()` is
  `dispatchWorkgroups(ceil(nx/4), ceil(ny/4), ceil(nz/4))`, the whole bounding
  box.
- Consequence in pressure: `webgpu-uniform-pressure-multigrid.ts:448-452` --
  `if (this.activeDispatch && entryPoint !== "mgSolveCoarsest" && ...)
  dispatchWorkgroupsIndirect(...) else dispatchWorkgroups(...dispatch.workgroups)`.
  `activeDispatch` is never set, so **all 948 encoded pressure passes are
  full-level direct dispatches** whose extent is `ceil(levelDims/4)`
  (`:601`; level dims carry a one-cell halo, `:339`).
- Consequence in the extension: `webgpu-uniform-velocity-extrapolation.ts:383-390`
  (seed/resolve), `:417-425` (restrict), `:427-438` (prolong), `:440-450` (pack)
  all fall to the direct full-level branch. Only the FIM front update
  (`:406-411`) is indirect and therefore liquid-proportional.

### Per-level pressure workgroups

The plan's shape needs no device: `buildPlan()` decides extent from level
dimensions and the fixed schedule alone. `pressure-plan-census.mts` mirrors that
walk on the CPU, and `--validate` checks it against the plan censuses the Dawn
probe read out of the live solver -- **the mirror matches every captured level,
pass and stage count exactly**, so the tables below cost no GPU time.

Full plan (2008 passes), workgroups summed per level:

| level | haloed dims A | haloed dims B | passes | A workgroups | B workgroups | delta | share of delta |
|---|---|---|---|---|---|---|---|
| L0 | 66^3 | 66x258x66 | 223 | 1 061 215 | 4 057 567 | +2 996 352 | **80.7%** |
| L1 | 34^3 | 34x130x34 | 293 | 213 597 | 783 189 | +569 592 | 15.3% |
| L2 | 18^3 | 18x66x18 | 380 | 47 500 | 161 500 | +114 000 | 3.1% |
| L3 | 10^3 | 10x34x10 | 467 | 12 609 | 37 827 | +25 218 | 0.7% |
| L4 | 6^3 | 6x18x6 | 554 | 4 432 | 11 080 | +6 648 | 0.2% |
| L5 | 4^3 | 4x10x4 | 91 | 91 | 229 | +138 | 0.0% |
| total | | | 2008 | 1 339 444 | 5 051 392 | +3 711 948 | |

Note the inversion: the coarsest levels carry the **most passes** and the least
work. L4 runs 554 passes over 24 workgroups in arm A.

### Per-level pass ms

Encoded prefix only (setup 19 + two Full-Cycles 928 + finish 1 = 948 passes),
which is what the trace seams measure. Pass counts are identical in A and B, so
the launch floor cancels exactly in the delta and each level's share of the
seam delta is its share of the added workgroups.

| seam | stage | passes | A wg | B wg | measured A | measured B | delta |
|---|---|---|---|---|---|---|---|
| CM11a topology + RHS pyramid | setup | 19 | 22 322 | 84 746 | 0.655 | 1.704 | **+1.049** |
| CM11a Full-Cycles | full-cycle | 928 | 439 206 | 1 648 542 | 5.177 | 7.274 | **+2.097** |
| CM11a parity copy + fine residual | finish | 1 | 4 913 | 18 785 | 0.066 | 0.066 | 0.000 |

Per-level ms, apportioned by added workgroups (derived, not separately
measured -- see section 7):

| level | setup delta | Full-Cycles delta | total |
|---|---|---|---|
| L0 | +0.932 | +1.588 | **+2.520 ms (80%)** |
| L1 | +0.098 | +0.391 | +0.489 ms (16%) |
| L2 | +0.015 | +0.091 | +0.106 ms |
| L3 | +0.003 | +0.022 | +0.025 ms |
| L4 | +0.001 | +0.006 | +0.007 ms |
| L5 | 0.000 | 0.000 | 0.000 ms |

So of the whole +5.505 ms A->B delta: **+3.145 ms (57%) is pressure**, and
**+2.520 ms (46% of everything) is level 0 alone**. Added levels contribute
nothing between 1x and 4x, because no level is added.

### The pressure seam is two straight lines

Fitting `cost = passes * floor + workgroups * k` to the Full-Cycles seam on the
1x/4x pair gives **4.758 us per pass** and **1.734 ns per workgroup** (27 ps per
cell visit). Extrapolated, unfitted:

| arm | passes | workgroups | predicted | measured | error |
|---|---|---|---|---|---|
| 2x | 928 | 842 330 | 5.876 ms | 5.833 ms | 0.7% |
| 8x | 2170 | 3 264 732 | 15.986 ms | 16.253 ms | 1.6% |

The 8x point is a *different hierarchy* (9 semi-coarsened levels) from a
different capture and still lands within 1.6%. Consequence: the launch floor is
**85% of arm A's pressure seam** (4.42 of 5.18 ms), 61% of arm B's, 64% of 8x's.
The same fit on the setup seam gives **14.7 us per pass** and **16.8 ns per
workgroup** -- setup's per-cell cost is ~10x a smoother sweep's, which is why 19
passes produce a third of the pressure delta.

## 4. Priced scaling attribution

Every per-step pass, clear and copy, its dispatch extent, what it scales with,
and its share of the +5.505 ms. "Tile early-exit" means the dispatch still
covers the whole bounding box and only the thread body returns early.

| # | seam / operation | passes A/B | dispatch extent | scales with | A ms | B ms | delta | share |
|---|---|---|---|---|---|---|---|---|
| 1 | Sec. 3.3 interface authority (rho-prime, face authority, seed, resolve) | 5 / 5 | **full grid** `ceil(n/4)^3` (`velocity-extrapolation:389-395`) | bounding-box cells | 0.262 | 0.721 | **+0.459** | 8.3% |
| 2 | Sec. 3.3 narrow-band FIM front (2 updates + 3 prepares) | 7 / 7 | **indirect** (`:406-411`) + 3 x `dispatchWorkgroups(1)` | live front only | 0.655 | 0.655 | 0.000 | 0.0% |
| 3 | Sec. 3.3 hierarchy fill + transport shell + 4h publish | 14 / 18 | **full level per level**, 6->8 levels (`:416-438`); pack over `(n+2)/4` (`:438-448`) | bounding-box cells x level count | 0.328 | 0.524 | +0.197 | 3.6% |
| 4 | Dense vertex phi transport and redistance (`uvAdvectPhi`, `uvRedistancePhi`) | 2 / 2 | **full vertex grid** `ceil((n+1)/4)^3` = 4913 -> 18 785 wg (`reference:1517-1519`) | bounding-box **vertices** | 0.459 | 1.180 | **+0.721** | 13.1% |
| 5 | Dense geometric volume coupling (`uvBuildEdges`, 4x full-range `clearBuffer`, 3 rounds of rows/donors/normalize) | 12 / 12 | full grid with **tile early-exit** (`uniform-volume.wgsl.ts:177-186`); clears cover `nx*ny*nz*4` bytes (`reference:1526-1532`) | live tiles (bodies); clears scale with cells | 1.835 | 1.901 | +0.066 | 1.2% |
| 6 | Dense liquid capacity balancing seam (`uvGather` + gammaB->gammaA copy) | 1 / 1 | full grid, tile early-exit (`wgsl:285-298`); copy is **full size** (`reference:1556`) | live tiles; copy scales with cells | 0.197 | 0.262 | +0.066 | 1.2% |
| 7 | Dense conservative volume sharpening (classify + 8 rounds x 4 entries) | 33 / 33 | tile classify over `ceil(tiles/4)`; sharpen entries tile-gated | sharpening tiles (422-426, flat) | 1.114 | 1.180 | +0.066 | 1.2% |
| 8 | Velocity advection + body forces (incl. full `volumeB->volumeA` copy) | 1 / 1 | full grid (`reference:1796,1818`) | bounding-box cells | 0.393 | 0.590 | +0.197 | 3.6% |
| 9 | CM11a topology + RHS pyramid | 19 / 19 | **full level**, every level | bounding-box cells (L0 = 89%) | 0.655 | 1.704 | **+1.048** | 19.0% |
| 10 | CM11a Full-Cycles | 928 / 928 | **full level**, every pass | bounding-box cells (L0 = 76%) + 4.76 us/pass floor | 5.177 | 7.274 | **+2.097** | 38.1% |
| 11 | CM11a parity copy + fine residual | 1 / 1 | full level L0 | bounding-box cells | 0.066 | 0.066 | 0.000 | 0.0% |
| 12 | Pressure projection | 1 / 1 | full grid | bounding-box cells | 0.262 | 0.328 | +0.066 | 1.2% |
| 13 | Dense phi surface publication (`uvPublish`) | 1 / 1 | full grid | bounding-box cells | 0.066 | 0.131 | +0.066 | 1.2% |
| 14 | Diagnostics reduction (`reduceDiagnostics`) | 1 / 1 | full grid (`reference.wgsl.ts:1614`) | bounding-box cells | 0.066 | 0.197 | +0.131 | 2.4% |
| | **sum of seam deltas** | | | | | | **+5.177** | **94.0%** |
| | **whole advance, measured** | | | | 11.469 | 16.974 | **+5.505** | 100% |
| | **unexplained** | | | | | | **+0.328** | **6.0%** |

Also encoded per step and not separately timed: `clearBuffer(this.reductions)`
and `clearBuffer(this.rigidExchange)` (`reference:1657-1658`), both fixed-size
headers that do not scale with the domain; the two-level tile classify chain
over `ceil(tiles/4)` workgroups (`reference:1681-1690`), which scales with
**total** tiles, not live ones; and the async `mapAsync` stats readback
(`reference:1450`), which is off the advance's critical path.

Rolled up by what the work is:

| group | delta ms | share |
|---|---|---|
| CM11a pressure (#9-11) | +3.145 | 57.1% |
| dense full-grid / full-vertex non-pressure (#1, #4, #8, #12, #13, #14) | +1.640 | 29.8% |
| extension hierarchy levels (#3) | +0.197 | 3.6% |
| tile-gated stages (#5, #6, #7) | +0.198 | 3.6% |
| liquid-proportional (#2) | 0.000 | 0.0% |
| unexplained | +0.328 | 6.0% |

**Caution on the small rows.** The Dawn tick is 0.0655 ms. Rows 5, 6, 7, 12 and
13 moved by exactly one tick and row 14 by two; those deltas are at or below
quantization and should be read as "did not measurably grow", not as measured
values. Only rows 1, 4, 9 and 10 have deltas of four ticks or more.

### What the tile gates are already buying

A fourth capture ran each height twice, once with the shipped gates on and once
with `twoLevelVelocity off`, `twoLevelExtension dense`, `twoLevelAdvection
dense`, `transportWorkMap dense`, `sharpeningWorkMap off`:

| arm | tiled GPU | dense GPU | saved | dense scaling 1x->4x |
|---|---|---|---|---|
| 1x | 11.862 ms | 18.154 ms | 6.29 ms (35%) | |
| 4x | 16.777 ms | 50.922 ms | **34.14 ms (67%)** | 2.80x (tiled: 1.41x) |

So the existing work maps already absorb most of the tall-air cost; what is left
is exactly the stages they do not cover. One inversion worth recording: the
**phi seam is cheaper dense** at 4x (0.852 ms dense vs 1.180 ms tiled). The
two-level sampler's per-sample tile-class `atomicLoad`
(`uniform-volume.wgsl.ts:177-180`) costs phi about 38%, and phi gains nothing
from the gate because `uvAdvectPhi` is not tile-gated at all. CPU encode is
unchanged by the gates in every arm (11.3-11.6 ms).

## 5. Dependency inventory

Reader/writer facts for every stage that is still bounding-box sized. No
recommendations.

**`uvAdvectPhi` (`uniform-volume.wgsl.ts:136`) -- full vertex grid, no tile gate.**
Writes `uvPhiOut` at every vertex of `(n+1)^3`. Per far-air vertex it evaluates:
- `uvTrace` + one `uvPhi` trilinear sample (the semi-Lagrangian backtrace);
- `uvClosedWallPhi` (`:117`), which returns immediately unless the vertex lies
  exactly on a closed wall plane;
- `uvReleasedWalls` (`:91`), which runs at **every** vertex regardless of
  distance to any wall. It loops 3 axes x 2 sides; a side is evaluated when
  `released` (only axis 1 can be, since `acceleration` is non-zero only there)
  or when `ambient` (axis 1 upper with `params.boundary.w > 0.5`, i.e. an open
  top). For each evaluated side it probes 4 corners, each a `clampCell` +
  `boundaryVelocity`/`velocity` load, and the term it may write,
  `dt*away - inward*(p[axis]-plane)*h[axis]`, is **explicitly proportional to
  the vertex's distance from that wall plane** -- so the lid term grows with
  container height and is computed at every vertex in the column.
- `uvAgreementShift` and `uvSeedPhi` are gated off by default
  (`phiAgreement off`, `phiSeedFromVolume off`).

**`uvRedistancePhi` (`:141`) -- full vertex grid, no tile gate.** Loads
`uvPhi(p)`; the 8-iteration closest-point search runs only inside
`band = 4*max(h)`, so far-air vertices do one load. The `textureStore` is
unconditional because `uvPhiOut` is a ping-pong target.

**Who reads far-air phi.**
- `uvTwoLevelSeed` (`uniform-volume.wgsl.ts:470`) reads `uvPhiIn` at all 5^3
  vertices of every tile in the domain and seeds the tile when any is `< band`.
  The test is deliberately **one-sided** (`phi < band`, not `|phi| < band`), so
  the seed set is a theorem about the FIM band rather than a property of V.
  This pass therefore reads phi over the **whole** bounding box, every step.
- `mgBuildFinestTopology` (`webgpu-uniform-pressure-multigrid.wgsl.ts:150`)
  reads phi to build the finest topology; `mgLiquid(p)` is `mgPhi(p) < 0`
  (`:94`). `mgDownsampleTopology` (`:186`) then carries that pyramid to every
  coarser level, `mgExtrapolatePhiOneCell` (`:310`) continues it one cell, and
  `mgBakeCoefficients` (`:322`) freezes per-level coefficients from it. Far-air
  phi enters the solve only as "not liquid", but it is *read* at every cell of
  every level.
- `uvTarget` (`uniform-volume.wgsl.ts:304`) takes 8 trilinear phi probes per
  cell inside `uvGather`; outside TRANSPORT the early-exit at `:291` skips it.
- `uvPublish` (surface publication, `reference:1926`) reads phi over the full
  grid to publish the render surface.
- The renderer and the grid overlays read the published surface, not phi
  directly.

**The released-walls lid read.** `uvReleasedWalls` is called only from
`uvAdvectPhi`. Its ceiling branch reads `velocity(cell)` at the top plane
(`upper` selects `velocity`, the lower side selects `boundaryVelocity`) through
`clampCell`, so every vertex in a tall column probes the **same** four ceiling
cells. `params.boundary.w > 0.5` marks an open top; the tall-air fixture uses
`top: "closed"`, so only the gravity-released side is evaluated.

**Extension far-field fill.** `webgpu-uniform-velocity-extrapolation.ts:234`
builds levels by `while (Math.max(...levelDims) > 1) levelDims = ceil(/2)`, so
height alone adds levels: 6 at 1x, 7 at 2x, 8 at 4x, 9 at 8x. The comment at
`:225-232` records why the loop must not stop at the short axis (the paper's 2D
slab topped out at 16x16x1, every face beyond packed zero, and a falling ball
was sliced flat at a block boundary). Restrict (`:416-426`) and prolong
(`:428-438`) each dispatch **full level**; only the FIM update is indirect.
Prolongation fills unknown values only, so levels above a fully-known level
change nothing but still launch. Readers of the filled field: `pack`
(`:438-448`) writes the transport shell over `(n+2)/4`, and
`encodeCoarseVelocityTable` publishes the 4h face table that the two-level
sampler (`uvCoarseVelocityComponent`, `uniform-volume.wgsl.ts:448`) reads for
every sample outside the FINE set. Transport (`uvBuildEdges`, `uvTrace`) and the
velocity advection both sample that field.

**Pressure topology / downsample.** Full-level dispatches at every level, listed
in section 3. The RHS pyramid (`mgBuildFinestRhs`, `:168`) reads face velocities
and V at every finest cell; `mgResidual` (`:210`), `mgRestrictResidual` (`:224`),
`mgProlongateAdd` (`:258`), `mgClearPressure` (`:279`) and `mgSmoothColour`
(`:351`) all cover their whole level. The consumer of the result is
`Uniform pressure projection` (`reference:1858`), which reads pressure over the
full grid; `pressureLiquid(p) = pressurePhi(p) < 0`
(`reference.wgsl.ts:275`) decides which faces it touches.

**Clears that cover the whole domain.**
- `clearBuffer(this.conditioningScratch, 0, nx*ny*nz*4)` four times per step
  (`reference:1577` and `:1579` inside a 3-round loop). Ranged to the donor-sum
  region only; words `[N,2N)` carry E1's restricted faces and class map, which
  the whole step samples, and the balance header and 4h map live above `2N`.
  4 MiB per clear at 4x, and they cost under one tick.
- `clearBuffer(this.reductions)` and `clearBuffer(this.rigidExchange)`
  (`:1707-1708`): fixed-size headers.
- `clearBuffer(this.conditioningScratch, this.sharpenTileCountWordOffset, 4)`
  (`:1610`): one word.
- `clearBuffer(this.convergence)` / `clearBuffer(this.dispatchArgs)`
  (`velocity-extrapolation:387-388`): fixed-size.

**Full-size texture copies per step.** `gammaB -> gammaA` (`reference:1556`) and
`volumeB -> volumeA` (`:1846`), both `[nx, ny, nz]`; plus
`vertexPhiScratch -> vertexPhiTexture` at `[nx+1, ny+1, nz+1]` only when
`redistance` is off (`:1570`). With rigid coupling active, `volumeB -> volumeA`
and `velocityB -> velocityA` again at `:1916-1917`.

**Publication / presentation.** `uvPublish` (`reference:1876`, also `:1057` on
the initial frame) runs over the full grid and writes the surface the renderer
and the overlays consume. `reduceDiagnostics` (`reference.wgsl.ts:1614`) sweeps
the full grid every step summing `surfaceOccupancy` and raw V; its output is
`volumeCellSum` / `rawVolumeDrift` in `readStats()` (`reference:1933`), which
the panels, the validation oracles and this probe read. It is the only stage
whose cost is pure measurement.

**What does not grow.** The FIM front (`#2`) is indirect and measured flat to
0.000 ms across a 4x domain. The tile-gated transport, gather and sharpening
stages (`#5-7`) grew by one tick each. Live tile counts are equal in absolute
terms between arms (FINE 1595 vs 1601, SHELL 1888 vs 1888), confirming the live
set is liquid-driven: `boxSolidVoxelShell` (`lib/core/solid-world.ts:485`)
places the tank's wall voxels at index -1 and n, **outside** the lattice, so no
in-domain cell is partially solid and the walls seed no tiles.

## 6. GPU commands run

Lease-wrapped, sequential, never concurrent with a browser or another Dawn
process. Tree state at every run: HEAD `743c2200`, working tree
`git diff --stat lib/methods/uniform lib/core` = 10 files changed, 335
insertions(+), 22 deletions(-), identical across all three runs -- the
concurrent "solve window" edits had not landed. (After the runs, and during the
write-up, `git diff --stat lib/methods/uniform` moved to 5 files, 422
insertions(+), 66 deletions(-); the measurements predate that.) No Dawn run
failed to compile, so no retry was needed.

```bash
# Run 1 -- arms A and B, 70 frames, lockstep in one process
FLUID_TALL_MULTIPLES=1,4 FLUID_TALL_FRAMES=70 \
FLUID_TALL_OUT=<tmp>/ab-1x-4x.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
npx tsx tools/run-webgpu-exclusive.ts -- \
  node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

# Run 2 -- height ladder, 60 frames
FLUID_TALL_MULTIPLES=1,2,8 FLUID_TALL_FRAMES=60 \
FLUID_TALL_OUT=<tmp>/ab-1x-2x-8x.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
npx tsx tools/run-webgpu-exclusive.ts -- \
  node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

# Run 3 -- gates-off dense control, 60 frames
FLUID_TALL_MULTIPLES=1,1d,4,4d FLUID_TALL_FRAMES=60 \
FLUID_TALL_OUT=<tmp>/ab-dense-control.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
npx tsx tools/run-webgpu-exclusive.ts -- \
  node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts
```

CPU-only, no lease needed:

```bash
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/cpu-census.mts
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/pressure-plan-census.mts \
  --validate=docs/research/uniform-geometric-tall-air-2026-09-19/ab-1x-4x.json,docs/research/uniform-geometric-tall-air-2026-09-19/ab-1x-2x-8x.json
```

## 7. What could not be verified

1. **Per-level pressure ms are derived, not measured.** Timing 948 passes
   individually needs a timestamp per pass, and prior work in this repo shows a
   per-pass census inflates the pressure bracket 3.6-6x, which would change the
   thing being measured. The apportionment is exact under "cost is linear in
   passes and workgroups"; the 2x and 8x cross-checks (0.7% and 1.6% error, the
   latter on a different hierarchy) are the evidence for that assumption, not a
   proof of it.
2. **6% of the A->B delta is unexplained** (+0.328 ms of +5.505). It is spread
   below the 65.5 us tick across seams, or sits between seams (submit overhead,
   the untimed clears and the two-level classify chain).
3. **No ceiling-stub measurement was taken.** It would have needed a fourth Dawn
   run; the gates-off control (section 4) answered the same question --- how much
   of the tall-air cost the existing work maps already remove --- from a run
   already budgeted.
4. **Single-tick deltas are not measurements.** See the caution under the
   attribution table.
5. **Only ~half of advances yield a trace sample** (`UNIFORM_PHYSICS_TRACE_CADENCE_MS`),
   so seam medians rest on n = 32-38 samples per arm out of 60-70 frames.
6. **8x was measured once** and in a different capture from arm B, per the
   one-run-per-arm rule. Its 2.62x GPU ratio is against that capture's own 1x
   control.
7. **Nothing here says what happens with liquid near the ceiling.** Every arm
   was built so the liquid never reaches the lid; a scene where it does would
   change the live tile set and is out of scope.
