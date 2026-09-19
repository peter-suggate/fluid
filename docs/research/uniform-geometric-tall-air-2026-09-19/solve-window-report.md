# Uniform Geometric: the solve window

A scene with a huge amount of empty air should cost what a small scene with the
same liquid costs. This is the implementation of that window, the measurement
of what it bought, and the audit of what could have gone stale.

Fixture, probe and baseline are the ones in this folder (`tall-air-scene.mts`,
`probe-tall-air-dawn.mts`, `ab-report.md`). Only the container height differs
between arms; the liquid is the identical 32x16x64-cell column in every arm.

## 1. Headline

Four arms, 70 frames, lockstep in one Dawn process. GPU figures are medians of
the solver's own hardware-timestamp seams (tick 65.5 us). "Window" is the
shipped path: the GPU computes the exact box, the host sizes the dispatches.

| | 1x (64^3) | 4x (64x256x64) |
|---|---|---|
| whole advance, window **off** | **11.600 ms** | **16.777 ms** |
| whole advance, window **on** | **11.731 ms** (+1.1%) | **12.911 ms** (**-23.0%**) |
| whole advance, window on, GPU-indirect launches | 16.319 ms (+40.7%) | 16.581 ms (-1.2%) |
| CPU encode, off -> on | 11.450 -> 11.478 ms | 11.356 -> 11.399 ms |
| CPU encode, on with indirect launches | 13.943 ms | 14.002 ms |
| work box, median | 62.5% of cells | **15.6%** of cells |
| dispatches per step, off | 1024 direct + 2 indirect | 1028 direct + 2 indirect |
| dispatches per step, window on | 1027 direct + 2 indirect | 1031 direct + 2 indirect |
| dispatches per step, window on, indirect arm | 25 direct + 1004 indirect | 25 direct + 1004 indirect |

**The tall scene now costs 1.11x the small one, down from 1.45x**, and the
small scene is unchanged within the lane's noise. The window's box is
*identical* in both heights -- `[0,0,0]..[64,40,64]` at frame 70 in both --
which is the property that was wanted: the cost of the air above the liquid is
gone, not merely reduced.

The four-arm capture with indirect launches is the reason WP2 exists, and it
confirms the prediction exactly: the window removes essentially all of the
tall-air work (every windowed seam is the same in the 1x and 4x arms), but
converting ~1000 launches a step from direct to indirect costs **+4.6 ms GPU**
and **+2.6 ms host encode**, which is more than the 4x scene had to save.

### Per-seam, all five arms

| seam | 1x off | 1x win | 4x off | 4x win | 4x win, indirect |
|---|---|---|---|---|---|
| Sec. 3.3 interface authority | 0.262 | 0.459 | 0.721 | 0.524 | 0.393 |
| Sec. 3.3 narrow-band FIM front | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 |
| Sec. 3.3 hierarchy fill + transport shell | 0.328 | 0.328 | 0.524 | 0.393 | 0.393 |
| Dense vertex phi transport and redistance | 0.459 | 0.459 | 1.245 | **0.590** | 0.328 |
| Dense geometric volume coupling | 1.835 | 1.835 | 1.901 | 1.835 | 1.901 |
| Dense liquid capacity balancing | 0.197 | 0.197 | 0.262 | 0.262 | 0.197 |
| Dense conservative volume sharpening | 1.180 | 1.180 | 1.311 | 1.180 | 1.311 |
| Velocity advection + body forces | 0.393 | 0.393 | 0.590 | 0.459 | 0.393 |
| CM11a topology + RHS pyramid | 0.655 | 0.655 | 1.704 | **0.852** | 0.590 |
| CM11a Full-Cycles | 5.243 | 5.177 | 7.274 | **5.636** | **9.830** |
| CM11a parity copy + fine residual | 0.066 | 0.066 | 0.066 | 0.066 | 0.066 |
| Pressure projection | 0.262 | 0.328 | 0.328 | 0.328 | 0.262 |
| Dense phi surface publication | 0.066 | 0.066 | 0.131 | 0.066 | 0.066 |
| Diagnostics reduction | 0.066 | 0.066 | 0.197 | 0.066 | 0.066 |
| **whole advance** | **11.600** | **11.731** | **16.777** | **12.911** | **16.581** |

Every seam the baseline named as height-scaling shrank: pressure L0 topology
halved, vertex phi more than halved, diagnostics and publication went back to
one tick. The 1.3 ms that separates `4x win` from `1x off` sits mostly in
Full-Cycles (5.64 vs 5.24) and topology (0.85 vs 0.66): the host's counts carry
lag padding the 1x arm does not need, the 4x arm runs two extra extension
levels, and its textures are 4x larger, so the same number of cell visits
touches a worse-behaved working set.

The `Sec. 3.3 interface authority` seam went **up** at 1x (0.262 -> 0.459). That
seam owns the window's own three passes (scan, reduce, finalize), which is also
where the +3 direct dispatches come from.

### Correctness across the arms

Zero WebGPU validation errors in every capture. No NaNs (every `maxSpeed`
finite, peak 3.499 m/s in all four arms).

| arm | volumeCellSum @70 | span over frames 5..70 | median maxSpeed | peak |
|---|---|---|---|---|
| 1x off | 32767.708 | 32767.573 .. 32768.243 | 1.760 | 3.499 |
| 1x window | 32767.576 | 32767.534 .. 32768.251 | 1.808 | 3.499 |
| 4x off | 32767.626 | 32767.553 .. 32768.249 | 1.786 | 3.499 |
| 4x window | 32767.641 | 32767.556 .. 32768.242 | 1.742 | 3.499 |

V agrees to 4e-6 relative across all four arms -- the same scale as the off
arms' drift against each other. Live tile counts stay liquid-sized
(fine tiles 1602-1613 in every arm). `pressurePassesEncoded` is **948 in every
arm**: the window never changes the encoded pass count.

## 2. What changed

### The toggle

- `lib/methods/uniform/uniform-volume-method.ts` -- `activeRegion` removed from
  the geometric `omitted` list and re-labelled for Uniform Geometric as
  **Solve window**, options **Liquid window** / **Whole domain**, default
  **off**. `createSolverAsync` passes `activeRegion: values.activeRegion === "on"`.
  It is a create-time (constructor) param, as on the paper method.
- `lib/methods/uniform/uniform-volume-pipeline.ts` -- the control and two
  readouts live on the `velocity-extension` stage, replacing the inherited
  paper controls: **Work box** (`cells / total · %`) and **Window launches**
  (`host-sized · N clipped · M dense`, or `whole domain · dust floor off`).
- `lib/methods/uniform/webgpu-uniform-reference.ts` -- the enable gate is
  `options.activeRegion === true && (!geometric || volumeDustThreshold > 0)`,
  plus `FLUID_UNIFORM_ACTIVE_REGION=0` as an off switch. **The dust floor is
  load-bearing**: `reduceDiagnostics` sums V over the window, which equals the
  domain sum only while every cell outside holds exactly zero.

Off is the old path exactly: no scan passes encoded, dense direct dispatches,
`activeRegion` published as `[0,dims)` so every window test folds to a plain
domain test.

### The window predicate (geometric)

`webgpu-uniform-reference.wgsl.ts`, `geometricActiveSeed`, gated on the
geometric variant. A cell seeds the window when

- `|V| > dust` (the dust floor, or 1e-6 when it is zero), **or**
- any of its 8 vertices has `phi < 4h` (the near-surface band
  `uvTwoLevelSeed` uses), **or**
- it carries a source this step (`scanExternalActiveSources`).

Solids and terrain are deliberately **not** seeds.

### Padding derivation

Reproduced from the comment at the finalize. The window must contain every cell
that can hold liquid this step plus every stencil such a cell reads, and be a
superset of every live tile set the step builds. Measured from the liquid:

| reach | cells |
|---|---|
| backward characteristics (`uvTrace`, `uvBuildEdges`, semi-Lagrangian advection) plus one trilinear tap | `travel + 1` |
| `uvRedistancePhi` closest-point search (`q` clamped to `p +- 4`, gradient +-0.25, one tap) | 6 |
| `uvAgreementShift` tent gather (`base + [-4,4)` plus a tap) | 5 |
| `uvSeedPhi` (`base + [-2,2)` plus a tap) | 3 |
| Sec. 3.5 sharpening: 8 sweeps of one-cell exchange inside the `|phi| < tuning.y*h` band | `ceil(tuning.y) + 1` |
| two-level SHELL tiles (the extension's finest working set) | `(fineReach + shellReach) * 4` |
| E3 live transport set: `ceil((ceil(D)+1)/4) + margin` tiles, capped at 16 | `m * 4` |
| 4h face-table trilinear tap | 8 |

`padding = max(all of the above) + 4` (the paper arm's slack). Then, for
geometric only: the origin is aligned **down** to a multiple of 4 and the far
edge **up** to a multiple of 4, so a 4x4x4 workgroup still coincides with one
4h tile (the sharpening work map, the two-level classes and the E3 live set all
index their tile by `cell/4` and exit whole-workgroup on that test, which is
only workgroup-uniform while the origin is a multiple of 4). Finally the window
snaps to any domain wall it has come within one padding of, so
`uvReleasedWalls` never reads a wall plane the window does not own.

### Kernels

`uniform-volume.wgsl.ts`: every geometric kernel dispatched through the window
now converts `gid` with `activeId(gid)` and keeps its domain bounds check --
`uvAgreementResidual`, `uvGather`, `uvPublish`, the four sharpening entries,
`uvBuildEdges`, `uvSumDonors`, `uvFallback`, `uvNormalizeRows`,
`uvNormalizeDonors`, `uvBalanceLiquidRows`, `uvBalanceLiquidDonors`,
`uvClassifySharpenTiles`. `uvAdvectPhi` / `uvRedistancePhi` run on a
window-sized **vertex** dispatch (`ceil((extent+1)/4)`, origin shared with the
cell box) whose record sits at word 176 of the active header.

Two window tests were added: `uvInWindow(cell)` and `uvTileInWindow(tile)`; both
fold to plain domain tests when the host publishes `[0,dims)`.

Two hazards found while doing this:

- **`uvSharpenTileActive` had to become window-aware.** `uvProposeSharpen` /
  `uvLimitedFlux` read a neighbour tile's proposals out of the shared `uvEdges`
  stencil arena. Outside the window that arena still holds this step's
  *transport* edges, so a stale-true tile class would have produced bogus fluxes
  right at the window boundary. `uvSharpenTileActive` now starts with
  `uvInWindow`.
- **`uvReleasedWalls` runs at every vertex** and its term is proportional to the
  vertex's distance from the wall plane. It now skips a plane the window does
  not contain, and the wall-snap above guarantees the window owns any plane a
  vertex it dispatches could actually reach.

`uvTwoLevelSeed` stays dense (it is a tile-grid pass) but exits immediately for
a tile outside the previous window when no source is present.

### WP2: direct dispatches sized by the host

The measurement above says why. The window is exact where it must be and lagged
where it is cheap:

- The GPU scan/reduce/finalize still runs every step and still publishes the
  **exact** box. Every kernel, every pressure level and every extension level
  still reads its **origin** out of `activeRegion`. No sample moves.
- Only the group **count** comes from the host. `planWindowDispatch`
  (`webgpu-uniform-reference.ts`) sizes this step's dispatches from the last box
  a never-awaited readback delivered:
  `extent = lagged extent + LAG_STEPS(3) * 2 * (ceil(D_lagged * 1.5) + 1) + 4`,
  aligned up to 4, clamped to the domain. Vertex counts are
  `ceil((extent+1)/4)`; per-level counts repeat finalize's own walk
  (`min(L+2, ceil(extent*L/d) + 6)`, the 2-cell low halo plus the 3-cell high
  one plus one for the floor/ceil pair the scaling can straddle).
- A count is only ever wrong by being too **small**. Threads past the exact
  extent are ordinary in-domain work; threads past the domain exit as always.
- `readWindowBox` is the `readPressureCycleDemand` pattern: its own 80-byte
  buffer, nothing in the frame path awaits it, a step skips the copy while an
  earlier map is outstanding.
- **Containment check on the GPU.** The host writes its chosen counts into the
  active header (words 180..234); `finalizeActiveRegion` compares its exact
  counts for the main grid, the vertex lattice and every level, and increments a
  violation counter (word 235) with an axis mask (word 236). On a violation the
  host goes whole-domain for 8 steps and the **Window launches** readout says so.
- Whole-domain counts are also forced before the first readback, after a reset,
  after a scene edit (`activeRegionRescanPending`) and on any step with a
  source -- each of those for `LAG_STEPS + 1` steps, not just one, because the
  lagged box stays stale for as long as the readback takes to catch up.
- The pressure plan is untouched: `setWindowLevelGroups` only replaces the
  launch size of dispatches the plan already carries, so the encoded pass count
  cannot move with the window (verified: 948 in all four arms).
- `FLUID_UNIFORM_WINDOW_DISPATCH=indirect` keeps the GPU's indirect records
  driving every launch. That is the A/B arm above; the UI toggle remains the
  single **Solve window: Liquid window | Whole domain**.

Observed at the default padding, both scenes, 60 frames: **max readback lag 1
step** (against the 3 assumed), **0 clipped steps**, **1 dense step** (start-up
only).

### One behaviour change outside the window path

`initializeActiveRegion` now seeds Uniform Geometric's window at the **whole
domain** and lets the first step's scan shrink it. The t=0 publication is a
dense dispatch, but every geometric kernel adds the window origin to its id, so
a t=0 window that did not start at the origin would leave the static
open-fraction plane unwritten below it -- for a droplet in a tall tank, that is
most of the domain. It costs two conservative steps. The paper arm's seed is
unchanged.

## 3. Staleness audit

"Outside" means outside the current window. The **clearing tail** referred to
below is the existing union-of-two-boxes rule (header words 7..12 are the union
of this step's padded box with the previous one), which gives a cell leaving the
window exactly one more full step during which `uvGather` writes V=0 and
`uvPublish` publishes air.

| field | who reads it outside the window | why it is safe / what changed |
|---|---|---|
| `vertexPhi` / `vertexPhiScratch` | `uvTwoLevelSeed` (all tiles), `mgBuildFinestTopology` (all level cells), `uvPublish` | Far-air phi is a large positive distance that does not change while the cell is far-air. Every consumer uses it as a sign test (`phi < 0`, `phi < band`). `uvTwoLevelSeed` now exits early outside the previous window unless a source exists. |
| `volumeA` / `volumeB` | `reduceDiagnostics`, `uvPublish`, pressure RHS | The dust floor makes V exactly 0 outside; the window is gated off when the floor is 0. The full-size `volumeB -> volumeA` copy means an undispatched cell keeps the value it had, so nothing is created or lost. |
| `gammaA` / `gammaB` | the gather and its full-size `gammaB -> gammaA` copy | Same: the copy is full-size, so an undispatched cell is copied unchanged. |
| `conditioningScratch` donor sums | `uvSumDonors` / `uvNormalizeDonors` inside the window | Cleared over the whole donor region every step; the clear was **not** shrunk to the window (explicitly out of scope). |
| `conditioningScratch` 4h velocity table | the two-level sampler, at every sample outside the FINE set | `publishCoarseVelocityTable` is **dense** and stays dense, so no sample reads a stale coarse face. |
| `conditioningScratch` class planes (E1/E2) | the whole step | Written by the dense tile classify chain over `ceil(tiles/4)`; unchanged. |
| sharpen work map | `uvSharpenTileActive` | Now ANDed with `uvInWindow`; this was the one real cross-window hazard found (see above). |
| transport shell / extension `up`/`down` textures | `uvBuildEdges`, `uvTrace` inside the live set | The live transport set is inside the window by the padding rule (`transportTiles` is one of the reaches padding takes the max over). |
| pressure textures, per level | `mgSmoothColour`, `mgResidual`, restrict/prolong | Each level's box is the scaled window plus a 2-cell low and 3-cell high halo. Outside it the topology says "air" (stale phi is still positive, stale V is still 0) and the smoother never relaxes there. |
| `volumePressureRows` rows | the coarse LCP | Default off. When on, rows are claimed only in cells the projection abandons, all inside the window. **Not measured with the window on.** |
| `mgBuildFinestTopology` | -- | Reads phi and V at every finest cell of its own (windowed) dispatch; both are sign tests. |
| `uvReleasedWalls` | every vertex of the phi passes | Guarded by `uvInWindow` on the wall plane, plus the wall snap. |
| `uvPublish` | renderer, overlays | Windowed; the clearing tail gives a departing cell one correct publication. t=0 is dense from origin 0 (see above). |
| `reduceDiagnostics` | panels, oracles, this probe | Windowed; correct only with the dust floor above zero, which the enable gate requires. |
| liquid capacity balancing's own indirect dispatch | -- | Now takes the window's main group counts (it previously had its own indirect record). Default off. |
| rigid coupling / solid passes | `coupleRigid` runs through the windowed `run()` | A rigid body whose cells lie outside the liquid window would be missed. **Unverified** -- no rigid scene was run. |
| MacCormack path | -- | **Unverified.** |

### What a clipped step actually does

Forced once on purpose (`FLUID_UNIFORM_WINDOW_LAG_PAD=0` starves the lag
allowance), 4x scene, 60 frames, paired with the off arm in the same process:
**3 clipped steps**, each followed by 8 whole-domain steps (25 dense steps in
total).

| | off arm | clipped arm |
|---|---|---|
| `volumeCellSum` @ 60 | 32767.635 | 32767.611 |
| front | 0.400 m | 0.400 m |
| max speed @ 60 | 1.261 | 1.353 |

V agrees to 7e-7 relative. A clipped step does not create or destroy volume: the
far edge of the window simply is not dispatched, an undispatched cell keeps the
value it already had, and the front stalls there for that step. Zero validation
errors.

## 4. Known gaps and unverified ground

1. **Rigid bodies** -- `coupleRigid` is a windowed dispatch. Never run with the
   window on. A body outside the liquid box is a plausible failure.
2. **Live insertion / inflow / drops** -- routed through the dense external-source
   scan and held at whole-domain counts for 4 steps, but never measured on Dawn.
3. **Scene edits** -- `applySceneUniforms` sets `activeRegionRescanPending`, which
   forces one dense census and 4 whole-domain steps. Not measured.
4. **Solids and terrain** -- deliberately not window seeds. A scene where solid
   geometry needs per-step work away from the liquid is not covered by this
   predicate.
5. **MacCormack velocity transport** -- not exercised.
6. **`volumePressureRows`, liquid capacity balancing, phi agreement, phi seeding**
   -- all default off and all unmeasured with the window.
7. **The 8x cliff** -- at 64x512x64 the pressure planner falls off lockstep into
   9 semi-coarsened levels. The extension reuses the shared level records as
   `record = extension level + 1`, which assumes both hierarchies halve in
   lockstep. Host-sized mode now falls back to dense counts for a level with no
   record instead of the zero-size indirect dispatch the indirect path would
   issue there, but **the window has not been run at 8x**.
8. **One AABB** -- two separated liquid bodies give one box spanning both.
9. **Only this fixture** -- one dam-break column in a closed tank, liquid never
   reaching the lid, 70 frames.
10. **Lane noise** -- this uniform Dawn lane is bimodal and the timestamp tick is
    65.5 us. The 1x off arm read 11.665 in one capture and 11.600 in the other
    (0.6%). Single-tick seam differences are not measurements.

## 5. Exact GPU commands

All under the repository WebGPU lease, sequential, never concurrent with a
browser or another Dawn process. The probe acquires the lease itself
(`acquireWebGPUExclusiveLock` at its top), so it is **not** wrapped in
`tools/run-webgpu-exclusive.ts` -- doing both would deadlock on the lock
directory.

```bash
# Smoke 1 -- does the geometric window shader compile and run at all (4 frames)
FLUID_TALL_MULTIPLES=1,1w FLUID_TALL_FRAMES=4 FLUID_TALL_WARMUP=1 \
FLUID_TALL_OUT=<tmp>/smoke.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

# Run A -- window off vs on with GPU-indirect launches, both heights, 70 frames
FLUID_TALL_MULTIPLES=1,1w,4,4w FLUID_TALL_FRAMES=70 \
FLUID_TALL_OUT=docs/research/uniform-geometric-tall-air-2026-09-19/solve-window-indirect.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts
# (this capture predates WP2, when the indirect path was the only one. To
#  reproduce it today, prefix FLUID_UNIFORM_WINDOW_DISPATCH=indirect. That flag
#  was verified on a 10-frame 4w run afterwards: 29 direct + 1004 indirect
#  dispatches a step, window fraction 15.6%, zero validation errors.)

# Smoke 2 -- host-sized direct window, 8 frames
FLUID_TALL_MULTIPLES=1w,4w FLUID_TALL_FRAMES=8 FLUID_TALL_WARMUP=1 \
FLUID_TALL_OUT=<tmp>/smoke2.json ...

# Run B -- window off vs on with host-sized direct launches, 70 frames
FLUID_TALL_MULTIPLES=1,1w,4,4w FLUID_TALL_FRAMES=70 \
FLUID_TALL_OUT=docs/research/uniform-geometric-tall-air-2026-09-19/solve-window-direct.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

# Run C -- forced containment violation (lag allowance starved to zero)
FLUID_UNIFORM_WINDOW_LAG_PAD=0 FLUID_TALL_MULTIPLES=4,4w FLUID_TALL_FRAMES=60 \
FLUID_TALL_OUT=docs/research/uniform-geometric-tall-air-2026-09-19/solve-window-forced-violation.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

# Run D -- lag and clipped-step census at the default padding, 60 frames
FLUID_TALL_MULTIPLES=1w,4w FLUID_TALL_FRAMES=60 \
FLUID_TALL_OUT=docs/research/uniform-geometric-tall-air-2026-09-19/solve-window-lag-census.json \
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts
```

CPU: `npx tsc --noEmit -p .` clean for `lib/`, `app/` and `components/`, and
`node --import tsx tools/run-feature-tests.ts` (1762 tests) shows the same 7
failures as before this work, none of them in the uniform method.

The probe gained three additive things and nothing else: the `w` arm suffix
(`activeRegion: "on"`), a direct/indirect dispatch counter wrapped around each
compute pass, and the window fields in the per-frame series.

## 6. Not attempted, deliberately

Shrinking the pressure or extension hierarchy depth to the window; planning the
pressure hierarchy on the window (the 8x cliff and the 2008-pass plan);
sub-ranged clears and copies; any memory work. The four full-range
`clearBuffer(conditioningScratch)` calls, the full-size `gammaB -> gammaA` and
`volumeB -> volumeA` copies and the dense tile classify chain all still scale
with the bounding box.
