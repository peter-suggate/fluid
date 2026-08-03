# Power Liquids — the droplet-in-a-vast-domain program (2026-08-03)

Handoff plan. Successor focus to `POWER_LIQUIDS_WORK_AND_DATA_20260803.md` §7 and
`POWER_LIQUIDS_LEAP_STATUS_20260803.md`'s ranked queue. Everything below is
grounded in a code sweep done 2026-08-03 at HEAD (post-`f85e502`); every claim
carries a file:line so a wrong one is checkable.

## 0. Thesis

The program's core insight — reduce GPU **work**, not dispatch count — survived
three falsification attempts this session (WORK_AND_DATA §4.3). The remaining
difficulty is that "work" has two components we keep conflating:

```
wall(scene) ≈ intercept(fluid footprint, solver constants) + slope · domain
```

Every existing lane moves both at once. The instrument that separates them is a
scene where the fluid term is pinned to ~zero: **~100 fluid cells, leaf 32, in a
container swept from 64³ to 240³.** With the footprint fixed, the measured
`ms/advance` vs domain-volume curve *is* the domain tax — its slope is the target
of this program, and its intercept is the solve-constant work the status doc's
item 0 already owns. The two programs stop stepping on each other.

Why this is the right lens and not another proxy:

- The four-cell matrix already proved the wall is **anti-correlated with fluid
  size** (tiny-still 361.6 ms vs large-hydro 143.6 ms). Fluid quantity is not
  the cost. What's left is constants and domain-shaped work.
- The code sweep below found the domain-shaped work is real and enumerable:
  ~84M thread-invocations per substep at 256³ that scale with `nx·ny·nz`
  against ~1,073 resident bricks, plus ~500 MB of dense allocations — **and a
  hard failure at exactly 256³** that nothing in the tree currently guards.
- Every term has a per-size prediction (§3). A profile that disagrees with the
  ledger falsifies the ledger, loudly — the WORK_AND_DATA lesson #4.

## 1. The hard wall first: 256³ does not run

`lib/webgpu-octree-fine-levelset-topology.ts:1437-1455` — the recurring fine-band
identity ladder computes `recurringBlocks = ceil(logicalBrickCount / 256)` and
issues it through `runIdentity` (`:1348-1353`), a **bare 1-D
`dispatchWorkgroups(x, y)`** not routed through `planFineLevelSetDispatch2D`
(unlike the `run` helper at `:1331-1341`). At the default
`globalFineLevelSetFactor = 4`, `logicalBrickCount === nx·ny·nz`
(`lib/octree-fine-levelset-bricks.ts:177-179`).

`maxComputeWorkgroupsPerDimension` floor is **65,535** and is *not* in
`requiredFluidDeviceLimits` (`lib/webgpu-device-limits.ts:27-54`). So the domain
cap is `65,535 × 256 = 16,776,960` cells:

- 255³ = 16,581,375 → runs (barely).
- **256³ = 16,777,216 → one workgroup over → Dawn validation error on every
  recurring step.**

`resolveGlobalFineBrickCapacity` clamps `maximumResidentBricks` against this
limit but never `logicalBrickCount`, so the existing clamp does not help, and
`FLUID_FINE_TOPOLOGY_INDIRECT_ASSIGN` converts a *sibling* dispatch, not these
five.

**Consequences for the plan:** start the sweep at 64³–240³ (240³ = 13.8M cells,
`recurringBlocks = 54,000`, under the limit). Making 256³+ *run at all* is
optimization target #1 (§4.1), and it is the same code that is the largest
per-step domain term — the wall and the tax have the same address.

## 2. The scene and lane, specified

Mirror `deep-power-hydrostatic` (commit `f85e502`) exactly — it is the freshest
scene+lane add and `tests/deep-power-hydrostatic.test.ts` pins the conventions.

### 2.1 Scene family: `power-droplet-<N>`

One parameterized factory, `createPowerDropletScene(edgeCells)`, registered as
(at least) two `defineScene` entries: **`power-droplet-64`** (control /
intercept anchor) and **`power-droplet-240`** (vast). Add 128/192 entries only
if the two-point slope needs resolving into a curve.

- **Container**: cube, `width_m = height_m = depth_m = N × 0.05`,
  `finestCellSize_m: 0.05`, `brickSize_cells: 8`, `top: "closed"`,
  `fluidWallMode: "free-slip"`, `surfaceTension_N_m: 0`,
  `fixedDt_s = maxDt_s = 0.004`, `rigidBodies: []`, no terrain, no inflow.
  Cell size 0.05 keeps the whole hydrostatic family's lattice so walls are
  comparable; N=64 is a 3.2 m cube (the large-dam width), N=240 is 12 m.
  Container extents must be exact multiples of `finestCellSize_m` — isotropy is
  enforced at `lib/webgpu-octree.ts:1663-1666`.
- **Fluid: a corner puddle, and only a corner puddle.**
  `initialCondition: "dam-break"`,
  `initialDamBreakDimensions_m: {x: 0.25, y: 0.2, z: 0.25}` → 5×4×5 = **100
  cells**, and **no `initialDamBreakOrigin_m`**. This is the only ~100-cell
  seeding that keeps `analyticSparseBootstrap === true`
  (`lib/webgpu-octree.ts:1649-1651`). Any offset origin or `initialBrickSeeds_m`
  flips the whole bootstrap dense: `Float32Array(nx·ny·nz)`, a full-domain
  signed-distance transform (`lib/volume-signed-distance.ts:30+`), a
  67–134 MB phi texture, and a persistent `cellCount·4 B` surface-state
  allocation (`lib/octree-surface-allocation.ts:27-41`). A free-floating
  droplet variant is a later, deliberate add with those costs budgeted — not
  the v1 scene. The puddle slumps briefly and settles; it wets the floor and
  two walls, which is physically boring and exactly what we want: near-zero
  fluid signal.
- **Method profile**: spread `LARGE_POWER_HYDROSTATIC_METHOD_PROFILE`
  (`lib/scenes.ts:98-106` — leaf **32**, `interfaceRefinementBandCells: 1`,
  `globalFineLevelSetFactor: "4"`) and author **both** capacities with written
  derivations in-source, the way `lib/scenes.ts:109-162` does for deep-hydro:
  - `pressureRowCapacity`: the planner's footprint-shaped default collapses to
    **256 rows** at 100 cells (`lib/webgpu-octree.ts:1086-1090`, headroom ×2,
    256-aligned) — too tight once the 3-cell closed-wall strip and 2:1 balance
    publish around the puddle, and an under-authored value dies at the
    `cold-topology` / `pressureCapacityOverflow` gates. Author **4,096** with
    the derivation written out (fluid rows + band + local wall strip + balance
    headroom), then *verify the derivation* with the t=0
    `pressureRequiredRows` readback and tighten.
  - `globalFineLevelSetMaximumBricks`: footprint default is ~1,073 resident
    bricks for this footprint; author **4,096** explicitly. There is **no env
    path** for this knob (`lib/methods/octree.ts:58-62`) — it must be in the
    smoke-catalog lane overrides, which is what the executor actually resolves
    (`tools/webgpu-smoke-executor.ts:962`, `:1088`).
- **Capacity-sensitivity gate** (the [[capacity-is-not-inert]] bug class): the
  1-step cold gate must be green at the authored capacities **and at 2×** — a
  lane that passes at one capacity and fails at another is reporting a Bet-1
  violation, whichever direction it fails.

### 2.2 Files to touch (mirror `f85e502`)

| file | change |
|---|---|
| `lib/scenes.ts` | capacity consts + derivations, method profile, factory, 2× `defineScene` |
| `lib/scene-webgpu-smoke-catalog.ts` | ids in `sceneWebGPUSmokeIds` (`:25-52`), overrides const, `suite(...)` per scene with a 1-step `default` lane and a `runtime-240` lane |
| `tools/power-dam-lane-environment.ts` | `"droplet-64"`, `"droplet-240"` in the union (`:11`) + lane records; restate `FLUID_PRESSURE_ROW_CAPACITY` and `FLUID_EXPECT_GRID` so the lane is self-describing and the environment record reads `clean: true` |
| `package.json` | 2 smoke scripts + benchmark scripts per lane |
| `tests/power-droplet.test.ts` | geometry, capacity derivations, preset, smoke lanes, lane table + npm scripts — clone the deep-hydro test's shape |
| `tests/scene-smoke-catalog-parity.test.ts` | check the parity gate picks the new ids up |
| `tools/octree-regression-artifact.ts:21` | widen `OctreeRegressionLane` **only if** using `--artifact=`; `--diagnostic-artifact=` (the measurement-floor shape) does not need it |

The 1-step `default` lane's acceptance should pin predictions, deep-hydro
style (`lib/scene-webgpu-smoke-catalog.ts:1089-1101`): `expected-grid
[N,N,N]`, `pressureRequiredRows at-most <derived>`,
`pressureCapacityOverflow equal false`. The `runtime-240` lane adds volume
drift ≤ 1e-4. First runs check against predictions, never accept what appears.

### 2.3 Run order per lane

1. 1-step cold gate (`npm run test:webgpu:power-droplet-240`). The three t=0
   killers: `cold-topology published no liquid-row frontier`
   (`lib/webgpu-uniform-eulerian.ts:1234-1241`), structured-authority rejection
   (`:1409-1433`), `initialPowerPressureReadiness` / `pressureCapacityOverflow`
   (`:1440-1448`). A 100-cell droplet in a vast box is the highest-risk
   configuration in the tree for the first of these.
2. `--lane=droplet-240 --steps=20` sizing run.
3. Full 240-step lane with
   `--diagnostic-artifact=artifacts/measurement-floor/droplet-240-240.json`.
   Check `clean: true` and record `comparisonKey` before quoting any number.

Watch the isolated-runner ceiling: 240 s
(`tools/run-webgpu-smoke-isolated.ts`), and cold start pays two unavoidable
O(domain) CPU loops — `planOctreeFluidFootprintBudget` triple-loops the whole
domain (`lib/webgpu-octree.ts:848-864`, 13.8M predicate calls at 240³) and
`createTallCellLayout` walks all columns (`lib/tall-cell-grid.ts:357-412`).
Seconds at 240³; budget them.

## 3. The domain-tax ledger (predictions to check the profile against)

Full inventory from the 2026-08-03 code sweep, ranked by expected cost at
240³ / 100 fluid cells. **P** = per-step, **1×** = one-time.

| # | term | site | scales with | 240³ prediction |
|---|---|---|---|---|
| 1 | **Recurring fine-band identity ladder** — 5 bare dispatches over the dense logical brick lattice, every substep | `webgpu-octree-fine-levelset-topology.ts:1437-1455` | `nx·ny·nz·(factor/4)³` **P** | ~69M invocations/substep vs ~1,073 live bricks; hard wall at 256³ |
| 2 | **Owner-page bitonic sort in one workgroup** — `SORT_CAPACITY` = pow2 of the full 8³-page lattice, ×2 per substep, 1 workgroup, device-scope barrier per stage | `webgpu-octree-owner-pages.ts:958-976`, capacity chain `:1295-1327` | `(nx/8)(ny/8)(nz/8)` **P** | 27,000 → 32,768 keys, 120 barrier stages, ~15,360 compare-exchanges *per lane* — domain-sized work wearing a compliant `(1,1,1)` launch; the exact §4.1-of-WORK_AND_DATA disease, undetected because the Bet-1 audit scores launches |
| 3 | **Owner-page arena forced to full domain** — `adaptiveCapacity` (8 pages at 256 rows) computed then discarded by a `Math.max` with the dense minimum | `webgpu-octree-owner-pages.ts:1292-1307`, `:191-252`; the `maximumArenaBytes` search at `:224-241` is never passed | `(nx/8)³ × 1028` words **1×** | ~111 MB (135 MB at 256³ — over the 128 MiB default binding limit) |
| 4 | **Fine level-set dense logical→physical directory** ×2 generations + 2× host `Uint32Array.fill` | `octree-fine-levelset-bricks.ts:190-196`, `:145`, `:225-229` | `nx·ny·nz` **1×** | ~110 MB GPU + 2×55 MB host fill+upload |
| 5 | **Fine-topology dense scan/scratch** (`desiredScan`, `topologyErrors`) | `webgpu-octree-fine-levelset-topology.ts:1011-1033` | `nx·ny·nz` **1×** | ~110 MB |
| 6 | **`structuredSeparationMask`** dense u32/cell | `webgpu-octree.ts:2924-2928` | `nx·ny·nz` **1×** | 55 MB (memory only; per-step writes are row-shaped) |
| 7 | **SPGrid dense brick/page directory sweep** — 6 dispatches (4 domain-shaped) on dirty epochs; the touched-radix-sort flag deletes them | `webgpu-octree-spgrid-vcycle.ts:2015-2027`, launch counts `:3926-3927` | `Σ_levels (d/2ˡ/4)³` **P** | ~250k + ~63k threads; ~4.5 MB directory |
| 8 | **Fine-seed candidate dense arm** — `includeLiquidInterior: true` (`webgpu-octree.ts:1778`) flips residency to the dense logical-key arm, 3 dispatches/substep of zero-fill over the brick lattice; also the input that drives #2 and #3 | `webgpu-fluid-brick-residency.ts:303-310`, `:1443-1460` | `(nx/8)³` **P** | ~500 workgroups/substep |
| 9 | **log2(domain) depth terms** — SPGrid levels 9 vs 5; the persistent MGPCG V-cycle walks `levelCount` down+up with barriers **per PCG iteration** inside its one workgroup; summary mip ladder 8 vs 4 | `spgrid-vcycle.ts:833-835`, `persistent-mgpcg.wgsl.ts:1034-1058`, `fine-levelset-summary-direct.ts:416-419` | `log2(max dim)` **P** | +80% barrier-serialized V-cycle depth per iteration even with ~0 live rows per level — this is where domain size leaks into the *solve constant*, coupling this program to status-doc item 0 |
| 10 | **Cold-start CPU** — footprint budget triple loop; tall-cell column walk | `webgpu-octree.ts:848-864`, `tall-cell-grid.ts:357-412` | `nx·ny·nz` **1×** | seconds |

Verified **not** domain-scaled (don't chase): air support (row-shaped via
`octreeAirSupportFootprintCapacity`, `webgpu-octree-air-velocity-support-gpu.ts:152-159`),
structured boundary/velocity, power descriptor, solid-vertex SDF, fine
transport/redistance, persistent-MGPCG buffers and its three ~1 KB staging
copies, and there is **no per-frame full-domain clear** anywhere on the octree
path. Dense solid field (134 MB+) and dense phi bootstrap are gated off by the
scene spec (no terrain/bodies, corner seeding).

**Coarse-only mode is not an escape hatch.** `globalFineLevelSetFactor: "1"`
deletes #1/#4/#5 but activates the dense complement summary: two
`(rowCap + nx·ny·nz)·32`-byte sample directories (~885 MB at 240³ — throws) and
13 domain-shaped dispatches/substep, ~218M invocations
(`webgpu-octree-power-coarse-levelset.ts:164-165`,
`webgpu-octree-coarse-summary.ts:82`, `:129-131`, `:247-275`). The Bet-1
audit's "unreachable on every matrix lane" claim about this module is true for
dam lanes and exactly wrong for this scene family. Stay at factor 4.

## 4. The optimization queue

Ordered by (expected slope reduction) × (confidence), each with its gate.

1. **Reshape the recurring fine-band ladder** (#1) — route the five
   `runIdentity` dispatches through `planFineLevelSetDispatch2D` (mechanical,
   removes the 256³ wall) and then shape them by **live counts, not the dense
   lattice**: the GPU-published `haloDispatch` (`:1425`) and `identityDispatch`
   (`:1480`) already exist as indirect sources two lines away. This is both
   the wall-remover and the predicted-largest slope term. Gate: D4-style
   bit-identity is not available on a new scene, so gate on the droplet lane's
   pinned t=0 counters + volume drift + the 64³/240³ pair, plus the existing
   dam lanes unchanged (the ladder runs on every scene — regression risk is
   real).
2. **Owner-page candidate capacity → fluid-shaped** (#2+#3 together, one root
   cause). The `adaptiveCapacity` already computes the right answer (8 pages)
   and a `Math.max` discards it (`webgpu-octree-owner-pages.ts:1298-1305`);
   `SORT_CAPACITY` and the 111 MB arena both follow from that discard. Fixing
   the capacity collapses the bitonic sort from 32,768 keys/120 stages to ~64
   keys/~21 stages *without touching the kernel*. This is also the test case
   for the missing **occupancy invariant** — the Bet-1 audit scores this file
   compliant while it does domain-sized work in one workgroup. Gate:
   capacity-sensitivity A/B (authored vs 2×) green on droplet *and* dam lanes.
3. **Score `FLUID_SPGRID_TOUCHED_RADIX_SORT` on the droplet lane** (#7). It
   was +2.72 ms/inconclusive on mini — where the dense directories are tiny.
   240³ is the lane it was built for; this is the cheapest possible re-test of
   a written-and-parked cutover. Note its ON-arm adds a `@workgroup_size(1)`
   kernel (`appendCandidateDirectoryIdentities`, `spgrid-vcycle.ts:4392`) —
   if the flag wins here, that kernel inherits queue position.
4. **Retire the fine-seed dense arm for small footprints** (#8) —
   `includeLiquidInterior: true` at `webgpu-octree.ts:1778` forces
   `sparseKeyPools = false`; the sparse arm already exists
   (`webgpu-fluid-brick-residency.ts:189-190`). Also the upstream feeder of #2/#3.
5. **Memory pass for 512³** (#4/#5/#6) — the dense directories become a hash or
   paged directory. Not wall-critical below 256³; *required* for the 512³
   milestone (1 GB+ as written). Do after the work terms, with the slope curve
   as the regression harness.
6. **V-cycle depth** (#9) — truncate the persistent MGPCG's level walk at the
   deepest level with live rows (the per-level `count(l)` is already
   GPU-published, `persistent-mgpcg.wgsl.ts:558`; the *depth* is the only
   dishonest part). Belongs jointly to this program and status item 0; hand it
   to whichever lands first.

**The standing rule this program adds** (WORK_AND_DATA learning #2, now with
two live specimens — `commitCandidateLevels` then, the owner-page sort now):
Bet 1's invariant gets an occupancy clause. *"Capacities may size buffers,
never launches"* becomes *"…never launches, and no kernel may iterate a
capacity-shaped range from a fixed-size launch."* An audit rule for
`@workgroup_size(1)`/`(1,1,1)` dispatches whose WGSL loops over a
capacity-derived bound is worth writing before optimization #2, so the fix is
provably a class fix and not a whack-a-mole.

## 5. Measurement protocol

- **The sweep is the deliverable.** For each lane in {64³, 240³} (+128/192 if
  needed): 3-round interleaved median, A/A noise floor first, `clean: true`
  and matching `comparisonKey` per artifact
  (`tools/power-dam-run-environment.ts`), artifacts in
  `artifacts/measurement-floor/droplet-*.json`. Plot wall vs `nx·ny·nz`.
  Report slope and intercept separately. Every optimization in §4 re-runs the
  sweep; success = slope falls, intercept unchanged (an intercept move means
  the change touched fluid-shaped work — investigate before celebrating).
- **Per-pass attribution**: `--pass-timestamps --isolate-pass-labels`, read the
  *ranking* never the wall (both knobs are contaminants —
  `power-dam-run-environment.ts:50-54`); raise
  `FLUID_GPU_PASS_TIMESTAMP_QUERY_CAPACITY` above the default 2048 and check
  `capacityOverflows == 0` in the printout. Prediction to check first: the
  fine-band ladder's passes should *grow* 64³→240³ by ~52× while
  `Octree persistent MGPCG` grows only via V-cycle depth.
- **Pass-boundary audit is free** — always on, in the diagnostic artifact as
  `gpuPassBoundaryAudit`. Prediction: closures ≈ 80 flat across the sweep
  (boundary structure is scene-independent; if it isn't, that's a finding).
- **Dispatch/workgroup census** (`WORK_AND_DATA` §7.3's tooling): per-size
  census of launched workgroups per pass family. This is the number the ledger
  in §3 predicts directly.
- **GPU frame profile**: once the 240³ lane is green and the sweep is
  captured, one `gpu-frame-profile` capture on droplet-240 to catch anything
  the ledger missed (the ledger is static analysis; occupancy sinks like #2
  hide from timestamps inside wider passes).

### Hazards (all previously paid for)

- A crashed run leaves `/tmp/fluid-webgpu-exclusive.lock`; every later run
  fails `EEXIST`. Check it before diagnosing anything.
- Never edit the tree under a running benchmark; `npx tsc --noEmit` misses
  import-time throws — `node --import tsx -e "import('./lib/…')"` belongs in
  the loop.
- `FLUID_TRIPWIRES` is pinned on after lane overlays
  (`benchmark-power-dam.ts:163`); only `failfast` costs ~27% — never compare a
  wall across tripwire modes.
- Any inherited shell `FLUID_*`/`WEBGPU_*` var → `clean: false`. Authored lane
  table values are clean; your shell is not.
- The mini lane is red at HEAD with no flags (scene-refactor fallout,
  WORK_AND_DATA §7.1). It does not block this program — new scenes, new lanes
  — but don't use mini as the control anchor until it's green; droplet-64 *is*
  the control.
- No stash/checkout/reset in this worktree, ever.

## 6. Milestones

- **M0** — scene family authored, both cold gates green at authored and 2×
  capacities, `runtime-240` green with volume drift ≤ 1e-4. *(No perf claims.)*
- **M1** — the sweep captured clean: wall, census, boundary audit, per-pass
  ranking for 64³ and 240³. Slope and intercept published. Ledger §3
  confirmed or corrected line by line.
- **M2** — recurring-ladder reshape landed behind a default-OFF flag, 256³
  wall removed, slope re-measured. Add `power-droplet-256` the day it runs —
  the scene that could not exist is the regression test.
- **M3** — owner-page capacity fix + occupancy audit rule. Slope re-measured.
- **M4** — radix-sort verdict on droplet-240; dense-arm retirement; defaults
  flipped for everything that held its gates.
- **M5** — memory pass; `power-droplet-512` runs; slope within ~2× of flat
  across 64³→512³ at fixed 100-cell footprint. That is the program's success
  criterion: *domain size becomes ~free*, which is what "sparse solver" was
  always supposed to mean.

## 6b. First results (2026-08-03) — M0/M1, and two ledger corrections

The family is authored (`power-droplet-{64,128,240,256}`, `lib/scenes.ts`),
lanes and scripts exist, and the sweep is captured. Every row below is
`environment: measurement-clean` on Apple M1 Max, 20 advances, warmup excluded.

| N | cells | ms/advance | dispatches/adv | passes/adv | active fine bricks |
|---|---|---|---|---|---|
| 64 | 262,144 | 169.35 / 160.00 / 123.55 (median **160.0**) | 501.4 | 80 | 565 |
| 128 | 2,097,152 | **163.50** | 502.4 | 80 | 565 |
| 240 | 13,824,000 | **230.95** | 503.4 | 80 | 565 |
| 256 | 16,777,216 | 233.95 / 207.15 / 224.25 (median **224.3**) | 503.4 | 80 | 565 |

The instrument works: 565 resident fine bricks and ~502 dispatches at every N,
so the fluid term really is pinned and only the container moves.

**The curve is a step, not a line.** 64 and 128 are indistinguishable — an 8x
volume increase costs nothing measurable — and 240 and 256 are indistinguishable
from each other at ~+65 ms. A linear "slope x domain" model is the wrong shape.
The step coincides exactly with the dense topology tile lattice reaching
8x8x8: `ceil(ceil(N/8)/4)` is 2, 4, 8, 8 for N = 64, 128, 240, 256. 240 and 256
differ by 21% in cells and 0% in that lattice — and 0% in wall. Whatever the
tax is, it is indexed by that lattice and not by volume.

### Correction 1 — the ladder is not the cost (ledger #1)

The recurring fine-band identity ladder is ~1.3M invocations/substep at 64 and
~84M at 256. If it drove the wall, 128 would sit measurably above 64. It does
not. The ranking-by-invocations in §3 is wrong on this hardware: the ladder is
early-exit-cheap. Its 2-D dispatch routing landed anyway
(`webgpu-octree-fine-levelset-topology.ts`) because it is what makes 256 cubed
*run at all* — that was a hard Dawn validation error before, and a scene that
cannot run cannot be measured. It is a wall-removal, not a scaling fix, and it
does not touch the O(domain) work or the dense directories. Tiling raises the
launch ceiling to ~1.1e12 cells; the memory wall (~537 MB per logical directory
copy at 512 cubed) arrives long before that.

### Correction 2 — the owner-page bitonic sort is not the cost either (ledger #2)

§3 predicted the single-workgroup 32,768-key / 120-barrier-stage sort dominated,
and the step's coincidence with `SORT_CAPACITY` (512, 4,096, 32,768, 32,768)
fits perfectly. It is still wrong. `FLUID_OWNER_PAGE_LIVE_SORT_SPAN=1`
(`webgpu-octree-owner-pages.ts`, default OFF) shapes the network by the live key
count instead of the arena bound — a 32-64x smaller network — and 256 cubed
measured **252.85 ms**, i.e. no improvement. The fix is correct and retained
behind the flag, but the sort is not the term.

What survives as the suspect is the *other* consumer of the same dense tile
lattice: the owner-page arena, forced to `topologyMaximumCandidatePages` physical
pages by the `Math.max` at the `minimumPages` clamp — 512 pages (2.1 MB) at 64,
4,096 (17 MB) at 128, and 32,768 (135 MB) at both 240 and 256. That reproduces
the step exactly, and it is memory traffic rather than launch count, which is
consistent with both corrections above. Root cause upstream is unchanged and is
ledger #8: `includeLiquidInterior: true` (`webgpu-octree.ts`) forces
`tileCapacity = logicalTileCount` in `planFineSeedCandidateResidencyPools`.
**Next experiment**: an A/B on that arena size alone, before any further
optimization.

### Correction 3 — the gates themselves are O(domain)

The 1-step cold gate is green at 64 cubed with every pinned counter
(`structuredAirSupportRows` 100, `globalFineActiveBricks` 624,
`pressureCapacityOverflow` false, grid `[64,64,64]`). At 240 cubed it exceeds
the 240 s isolated-runner ceiling *before its first advance*, while the same
scene's twenty measured advances take 4.6 s on the benchmark lane. Two smoke
collectors are domain-shaped: the spatial field / field-stats loop walks every
cell, and `compact-octree-field-readback`
(`tools/webgpu-smoke-readbacks.ts`) walks the **fine** lattice, `(N*4)^3` —
16.7M samples at 64 cubed, 885M at 240. The lanes now scope the first two above
2M cells (see the comment in `scene-webgpu-smoke-catalog.ts` for exactly what
the large members no longer check); the third is unconditional in the smoke
readback path and is the remaining blocker for a large cold gate. Until it is
shaped, the evidence for 240/256 is the benchmark lane, which is green and
clean at both (`validation errors: 0`, performance gates PASS).

`pressureRequiredRows` is also worth noting as a near-vacuous gate: it is the
solve's overflow *lower bound* and reads 0 whenever the arena holds, so
`deep-hydrostatic-interior-coarsening` is likewise not measuring what its name
suggests. The droplet lanes pin `structuredAirSupportRows` and
`globalFineActiveBricks` instead, which carry live counts.

### Regression status

`symmetric-expansion` reaches the documented window unchanged after the
dispatch reshape: first D4 loss at **step 68** (volume, velocity, pressure,
rhs), topology and diagonal at 69 — bit-identical to
`SYMMETRIC_EXPANSION_FRAME_PROFILE.md`. CPU suite shows no new failures against
a clean HEAD clone.

## 7. Open questions

1. Does the puddle stay 100±small cells over 240 steps, or does wetting/band
   growth inflate `pressureRequiredRows`? The t=0 readback answers this on day
   one; if it drifts, pin the count at settle-time and note it in the lane.
2. Is the fine-band ladder's ~69M-invocation term actually *time* on M1 Max,
   or is it early-exit-cheap and the owner-page sort (barrier-bound, not
   thread-bound) dominates? The ledger ranks by invocations; the M1 sweep
   ranks by ms. Expect surprises — that's the point of the instrument.
3. Interface band: the profile inherits band 1 (large-hydro). The tiny-still
   band-1-vs-3 A/B showed no difference at 16³ (WORK_AND_DATA §1); re-check
   once at 240³ before trusting that at scale.

## 6c. The domain tax, found and removed (2026-08-03)

Per-label GPU attribution (`--pass-timestamps --isolate-pass-labels`, matched
10-capture windows, both ends at 565 resident fine bricks) resolved the step to
a single pass:

| GPU pass | 64³ | 256³ | Δ |
|---|---|---|---|
| **Prepare inactive owner-page generation** | 1.02 | **52.91** | **+51.9** |
| Octree persistent MGPCG | 16.70 | 23.50 | +6.8 |
| Scan and compact recurring fine-band identity | 0.15 | 4.49 | +4.3 |
| Publish recurring sparse fine band | 0.09 | 4.30 | +4.2 |
| Octree resident grading closure | 55.59 | 57.49 | +1.9 |
| Advect structured families | 21.86 | 21.83 | −0.0 |
| **total attributed** | **117.6** | **188.7** | **+71.1** |

73% of the tax in one single-workgroup pass. Splitting its two dispatches into
two labels (free in production — `PassBroker` reuses the open pass) charged
60.4 ms to `buildOwnerPageCandidate` and 0.5 ms to the bank commit.

### Root cause: the occupancy grid was marked fully occupied

The topology tile lattice *is* a uniform occupancy grid — 8³ tiles of 32 cells
at 256³. `tileHasPressureBoundarySupport` marked the container's entire wall
shell, two tiles thick, unconditionally resident: **448 of 512 tiles**, each
carrying 64 owner pages, for 100 cells of water in one corner. The candidate
builder then sorted, deduped and re-materialized ~30,400 pages every advance in
*one workgroup* — ~124 MB of payload copy at 1/32 of the machine.

That retention was standing in for a refinement policy that no longer exists.
`refineLeaf` gates a closed-wall crossing on `minimumPhi <= band`
(`fluidGatedBoundaryRefinement`, default true since it landed), so a dry wall
tile far from liquid never splits. The scheduler was publishing topology for
leaves that do not exist, and the two policies had no shared source of truth.

**Fix**: `fluidGatedBoundarySupport` (residency scheduler flag bit 16), wired
from `this.fluidGatedBoundaryRefinement` at both construction sites so the two
can never disagree again. A dry wall tile is retained only when liquid is within
reach; wall tiles search one tile wider than liquid tiles, which is the 2:1
grading margin the unconditional form was standing in for. Both publishers are
covered — the dense `emitTopologyTiles` ring and the sparse publisher's
single-lane `pressureBoundaryTile` claim scan.

### Measured

Measurement-clean, M1 Max, 20 advances:

| N | before | after | Δ |
|---|---|---|---|
| 64³ | 160.0 | **118.50** | −26% |
| 128³ | 163.5 | **125.05** | −24% |
| 240³ | 231.0 | **147.85** | −36% |
| 256³ | 224.3 | **154.15** | −31% |

Domain tax (256³ − 64³) cut from +64.3 ms to +35.7 ms. The
`Prepare inactive owner-page generation` pass went 52.9 → 9.2 ms.

**It is also a correctness fix.** `power-droplet-256` at 80 steps tripped
`mgpcg-nonconvergence` 51 times from step 30 on the baseline; after the change,
0 trips over 80 steps. The 20-step lane never reached it, which is why the first
sweep looked green. Controlled: identical clean configuration, baseline clone at
`c1880f0`.

### Regression status

- `symmetric-expansion`: first D4 loss at **step 68** (volume/velocity/pressure/
  rhs), 69 (diagonal/topology) — bit-identical to the documented window.
- `minimal-power-dam-break` 500 advances, `large-power-dam-break` 500,
  `large-power-hydrostatic` 240: **0 tripwires** on all three.
  Wall: 245.6→246.8, 283.7→287.9, **156.0→146.3**. Neutral to better; the
  +1.5% on the large dam is a dense-fluid scene where the gate can only add the
  wider ring search, and is within this lane's run-to-run spread.
- `deep-power-hydrostatic` fails identically before and after (same
  `latchedFinalizeReason: 13`, `topologyPublished: 0`, same row count) — a
  pre-existing red lane, not a regression.
- Octree/residency CPU tests: 17 failures before, the same 17 after, by name.

### What the sort experiment was actually measuring

`FLUID_OWNER_PAGE_LIVE_SORT_SPAN` stays default OFF, and the earlier null result
is now explained rather than merely repeated: `sourceSlots` was (448+27)·64 =
30,400, so `pow2(sourceSlots)` only shrank the network 65,536→32,768 — one stage
group, and the measured saving was 7% (52.9→49.2). The flag was never wrong; the
live count simply was not live. With the occupancy fix the candidate stream is
small enough that the flag has nothing left to save.

### Next

The remaining 256³ frame is ~154 ms, of which only ~18 ms still scales with the
domain (owner-page prepare 9.1, the two recurring fine-band scans 8.9). The rest
is intercept: `Octree resident grading closure` ~55 ms and
`Advect structured families` ~22 ms, both flat from 64³ to 256³. That is now the
bigger number and a different problem — attack the intercept next, not the slope.

## 6d. The intercept, first cut: grading stopped re-materializing splits (2026-08-03)

Commit `0a839bf`. The `Octree resident grading closure` pass was the largest
single item in the frame and, unusually, **flat**: 55.6 ms at 64³ and 57.5 ms at
256³, ~47% of a 118.5 ms frame for 565 resident bricks. Flat means it is not
bounded by the domain, so the occupancy fix of §6c could not touch it.

### Attribution

Splitting the closure into one label per balance round (free in production —
PassBroker reuses the open pass — and exact under
`FLUID_GPU_ISOLATE_PASS_LABELS`) localized it immediately. At 64³ the ten rounds
cost **26.4, 13.2, 4.5, 1.8, then ~0.9 ms each**. A flat total that is
front-loaded like that is not a fixed per-round cost; it is split
materialization, concentrated where the topology is still moving.

### Root cause

Grading is a neighbour repair. Every leaf on the ring around a coarse neighbour
independently requests the **same** neighbour split, and `splitLeaf` then wrote
that neighbour's entire `size³` owner partition **serially in one lane, once per
asker**. The writes are idempotent `atomicMin`, so the duplicates never changed
published topology — they only multiplied a 32³ materialization by the ring
population and piled every copy onto the same words.

### Fix

1. `claimLeafSplit` claims each split on its own origin cell — the first cell
   `splitLeaf` would write anyway — so exactly one asker materializes the cube.
   The claim word is bit-identical to what the old first loop iteration wrote,
   so the winner performs every write the losers would have.
2. Materialize one owner page at a time, so the page directory is consulted once
   per page instead of once per cell.

A missing owner page is answered by materializing, not claiming: that path
latches the rejection flag inside `storeOwnerRequired`, and the loop must keep
visiting the pages that do exist.

### Measured

Paired, interleaved base/head, measurement-clean, 20 advances, M1 Max:

| lane | baseline | head | Δ |
|---|---|---|---|
| droplet-64 | 111.30 | **93.95** | −15.6% |
| droplet-256 | 140.50 | **121.90** | −13.2% |

Grading candidates, normalized against the untouched `Advect structured
families` pass in the same run: **51.69 → 21.87 ms (−58%)**.

### Gates

All measurement-clean. `power-droplet-256` at **80** advances: 0 tripwires with
an empty allow-list and `mode: end-of-run` — this is the lane that caught the
previous change diverging from step 30, so it is the result that matters.
`symmetric-expansion`: 68/68/68/68/69/69, the baseline window unchanged, and the
topology hook in particular still holds exact D4 symmetry through step 68 —
relevant because the dedup changes *which* invocation materializes a split.
`mini` 500, `large` 500, `large-hydrostatic` 240, `power-droplet-64` 80: 0
tripwires each. Walls 242.06 / 282.51 / 141.42 against the §6c references of
246.8 / 287.9 / 146.3. CPU tests: 13 failures by name at both `65b2427` and
`0a839bf` on the narrow glob; the wide glob adds the four un-prefixed
`tests/octree-*.test.ts` reds for the familiar 17.

**Note on `test:webgpu:symmetric-expansion`: it always exits rc=1.** Six
`hook.fluid-symmetry.octree.*` findings fail by construction. The criterion is
the step those hooks reach, never the exit code, and never a grep for check
names — `quadtreeMaximumNeighborRatio` and `powerDiagramAuthoritative` appear in
the same findings array as *passes*, and counting them as failures manufactures
a phantom 2:1-balance regression.

### Refuted, and kept anyway

Caching the owner-page arena header per invocation — it was re-read through five
device atomics on the same five words per owner lookup — measured **null**
(94.67 → 97.15 ms summed pass occupancy, with the unrelated `Advect` and `MGPCG`
passes moving by the same ~2%, i.e. run-to-run offset). It is retained only
because it is what makes the per-page hoist expressible. **No saving is claimed
for it**; do not credit it later.

### The seventh lane, and why a timeout there is not a regression

`deep-power-hydrostatic` timed out on the first gate run (`exceeded 240000 ms;
sending SIGTERM`, exit 124) instead of producing its usual red. That is a
*different failure mode* from the recorded one, so it was chased rather than
filed as expected.

Two facts make the raise-the-timeout reflex a dead end, and both are worth
knowing before anyone tries it again:

- `parseWebGPUSmokeTimeout` hard-caps `FLUID_WEBGPU_SMOKE_TIMEOUT_MS` at
  **240000**. A larger value aborts the run before Dawn starts.
- The benchmark overlay sets `FLUID_WEBGPU_SMOKE_TIMEOUT_MS: "240000"` *after*
  spreading `process.env`, so a shell value cannot reach it regardless.

The lane's own config comment already says it: 240 steps at ~1 s/advance against
a 240 s ceiling is zero margin by construction. Under any wall-clock pressure —
this run had already lost a lock race to a concurrent agent — it times out.

Resolved by shortening the lane instead, which fits under the cap and yields a
comparable verdict. At `--steps=120` both sides complete and fail identically:

| | HEAD `0a839bf` | BASELINE `65b2427` |
|---|---|---|
| restriction-unaccepted | 120 | 120 |
| fine-band-sentinel | 60 | 60 |
| air-support-failure | 120 | 120 |
| topology-rollback | 60 | 60 |
| first trips (steps) | 1 / 1 / 1 / 2 | 1 / 1 / 1 / 2 |

Bit-for-bit identical, so the lane is unchanged by the grading fix. Note the
comparison is on **trip counts and first-trip steps, not walls**: a tripwire
failure throws before the `ms/advance` line is printed, so a red lane never
reports a wall. The counts are the stronger equivalence claim anyway.

The failure itself is upstream of anything grading touches — it trips at step 1
with root `structured dynamics rejected at stage 1 (advect: advecting velocity
sample invalid at the face centroid)`, and `topology-rollback` reports
`published: false` with `downstreamFinalizeReason: 13`. (That is the reason-13
signature recorded elsewhere as `latchedFinalizeReason`; same value and same
not-published state, surfaced through a different field.)

### Next

`Octree resident grading closure` is now ~21.9 ms, still the largest intercept
item, and the serial single-lane materialization of one 32³ cube remains — the
ring duplication was the multiplier, not the base cost. Spreading the cube
across the workgroup's 256 lanes is the untried follow-up. `Advect structured
families` (~22 ms, flat from 64³ to 256³) is still unattacked.
