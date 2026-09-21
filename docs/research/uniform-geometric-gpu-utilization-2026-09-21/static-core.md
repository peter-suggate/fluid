# Uniform Geometric non-pressure core — static micro-architectural audit

Date 2026-09-21. Target: Apple M1 Max (32-core GPU, SIMD32, unified memory,
~400 GB/s). Runtime: WebGPU → Dawn → Tint → MSL.

Scope: `webgpu-uniform-reference.wgsl.ts`, `webgpu-uniform-velocity-extrapolation.wgsl.ts`
and their encode drivers. **Out of scope** (other agents): pressure multigrid,
`uniform-volume.wgsl.ts` kernel bodies. Volume-stage *encode* facts (clears,
copies, dispatch counts) are reported because they live in the encoder this
audit covers.

No GPU was run. Every number below is read off source. Claims are tagged
**[V]** verified in code (file:line) or **[I]** inference from verified code.

---

## 1. Configuration that selects the live set

| Flag | Value at geometric shipping defaults | Source |
|---|---|---|
| `geometricVolume` | `true` | `uniform-geometric-options.ts` |
| `pageDomain` | `true` | same |
| `nativeRootExecution` | `true` | `webgpu-uniform-reference.ts:795–805` |
| `nativePageCoordinates` | `true` | same |
| `activeRegionEnabled` | **false** (`pageDomain !== true` fails) | `webgpu-uniform-reference.ts:758` |
| `pageDomainDispatch` | **undefined** (guarded by `!nativeRootExecution`) | `:963` |
| `phiRegion` / `phiDispatch` | created (phi window ON) | `:937–945` |
| `velocityTransport` | `"semi-lagrangian"` | `uniform-geometric-parameters.ts:6` |
| `extensionFrontSweeps` | **2** (base default is 16) | `uniform-geometric-parameters.ts:10`, `parameters.ts:31` |
| `twoLevelVelocity/Extension/Advection` | `on`/`tiles`/`tiles` | `uniform-geometric-parameters.ts:28,34,40` |
| `gammaDiffusionIterations`, `densityPostProcessing`, `solidExcessCorrection` | `0`/`false`/`false` | `uniform-geometric-options.ts` |
| `volumePageConfig.work` | only when `max(nx,ny,nz) > 64` | `webgpu-uniform-reference.ts:974` |

**Consequence [V]:** `dispatch()` (`:1715–1724`) takes its *last* branch — a
**direct, dense `ceil(n/4)³` dispatch**. No indirect, no window, no active box.
`runVertex()` (`:1747–1761`) takes the `phiDispatch` branch — the **only
indirect dispatch on the non-pressure path** besides `runVolumeWork` above 64³
(`:2453–2458`).

---

## 2. Live/dead entry points

### 2a. `webgpu-uniform-reference.wgsl.ts` (32 `@compute` entries)

| Entry | Line | `@workgroup_size` | Firings/advance | Status |
|---|---|---|---|---|
| `scanExternalActiveSources` | 1782 | (4,4,4) | 1 (phi census) | **LIVE** |
| `reduceExternalActiveRegionSummaries` | 1839 | (256) | 1 | **LIVE** |
| `finalizeActiveRegion` | 1844 | **(1)** | 1 | **LIVE** |
| `buildExtrapolationAuthority` | 795 | (4,4,4) | 1 | **LIVE** |
| `semiLagrangianAdvection` | 1124 | (4,4,4) | 1 | **LIVE** |
| `project` | 1302 | (4,4,4) | 1 | **LIVE** |
| `coupleRigid` | 1385 | (4,4,4) | 1 *iff* `bodies.length>0` | **LIVE (cond.)** |
| `reduceDiagnostics` | 2044 | (4,4,4) | 1 | **LIVE** |
| `buildDenseExtrapolationAuthority` | 797 | (4,4,4) | t=0 only (`:1412`) | init |
| `advect` / `reverseAdvection` / `correctAdvection` | 1149/1164/1205 | (4,4,4) | 0 (MacCormack arm) | dead |
| `traceGammaAndBeta`, `scatterDensityDeficit`, `gatherConservativeDensity` | 930/967/995 | (4,4,4) | 0 (`geometricVolume` branch at `:2709`) | dead |
| `diffuseGammaX/Y/Z` | 1074–1076 | (4,4,4) | 0 (`gammaDiffusionIterations=0`) | dead |
| `sharpenCompute` / `sharpenScatter` / `sharpenResolve` | 1488/1496/1531 | (4,4,4) | 0 (geometric branch) | dead |
| `scatterSolidExcess` / `resolveSolidExcess` | 1563/1613 | (4,4,4) | 0 (`solidExcessCorrection=false`) | dead |
| `wallFilmResolve` | 1653 | (4,4,4) | 0 (geometric takes `uvPublish`, `:2881`) | dead |
| `postprocessBlurX/Y/Z`, `postprocessResolve` | 1674–1677 | (4,4,4) | 0 (`densityPostProcessing=false`) | dead |
| `scanActiveRegion`, `reduceActiveRegionSummaries` | 1770/1832 | (4,4,4)/(256) | 0 (`activeRegionEnabled=false`) | dead |
| `buildHeight` | 1220 | (8,8,1) | **no pipeline exists** | dead |
| `relaxSolidPhi` | 1414 | (4,4,4) | **no pipeline exists** | dead |

24 of 32 entry points are dead at shipping defaults. Two (`buildHeight`,
`relaxSolidPhi`) have no entry in the `PIPELINES` table at all (`:273–304`).

### 2b. `webgpu-uniform-velocity-extrapolation.wgsl.ts` (10 entries) — all live

| Entry | Line | `@workgroup_size` | Firings/advance @64³ | Dispatch |
|---|---|---|---|---|
| `clearExtrapolationState` | 207 | (4,4,4) | 1 | direct dense |
| `seedActiveFront` | 214 | (4,4,4) | 1 | direct dense |
| `updateActiveFront` | 353 | (4,4,4) | **2** (`extensionFrontSweeps`) | **indirect** off `activeDispatch` |
| `prepareActiveDispatch` | 406 | **(1)** | **2** (paired with each update) | direct `[1,1,1]` |
| `resolveConvergedFront` | 429 | (4,4,4) | 1 | direct dense |
| `restrictKnownVelocity` | 582 | (4,4,4) | 6 (one per hierarchy level) | direct per-level |
| `prolongUnknownVelocity` | 645 | (4,4,4) | 5 | direct per-level |
| `prolongAndPack` | 664 | (4,4,4) | 1 (finest, fused) | direct |
| `packTransportShell` | 671 | (4,4,4) | 1 | direct |
| `publishCoarseVelocityTable` | 698 | (4,4,4) | 1 | direct (4h tiles) |

`encodedPassCount = 4 + 2·sweeps + down + up − 1` (`:395–401`). At 64³:
6 hierarchy levels (32,16,8,4,2,1) → `4 + 4 + 6 + 6 − 1 = 19`, **+1** coarse
table = **20 passes**. At 128³: 7 levels → **22 passes**. **[V]**

`coarseVelocityLevel` requires `dims == 4 × level[1]` exactly (`:352–361`); at
64³ (16³ level) and 128³ (32³) it holds, so the two-level sampler is armed.
**If it did not hold, `twoLevelTileCount = 0` and the whole two-level scheme
silently reverts to all-fine** (`:1024–1025`). **[V]**

---

## 3. Per-advance pass and traffic census (non-pressure)

| Stage | Passes @64³ | Encoder source |
|---|---|---|
| Phi region census | 3 | `:2194–2211` |
| Two-level tile classes (4 entries) | 4 | `:2680–2689` |
| Extrapolation authority | 1 | `:2248` |
| Velocity extension | 20 | §2b |
| Phi advect + redistance (vertex, indirect) | 2 | `:2498–2499` |
| Transport page mark/compact | 2 | `:2441–2442` |
| Volume transport (build/finish/fallback + 3 donor rounds + gather) | 13 | `:2483–2515` |
| Surface-volume correction | 14 + 1 texture copy | `webgpu-uniform-surface-volume.ts:66–85` |
| Correction capacity/targets | 2 | `:2521,2525` |
| Sharpen tile classify + page prep + 8 rounds × 4 | 35 | `:2531–2545` |
| Semi-Lagrangian advection | 1 | `:2817` |
| Surface-deficit balance | 2 | `:2466–2481` |
| Projection | 1 | `:2858` |
| Surface publication `uvPublish` | 1 | `:2881` |
| Diagnostics reduction | 1 | `:2915` |
| **Total non-pressure compute passes** | **≈102** | |

### Clears and copies (live path only)

| Operation | Bytes @64³ | Bytes @128³ | Line |
|---|---|---|---|
| `clearBuffer(volumeDonorSums)` ×4 | 4 × 6,291,456 = **25.2 MB** | 4 × 50,331,648 = **201 MB** | `:2506, :2509` (×3 rounds) |
| `copyField(volumeB→volumeA)` | 1,048,576 | 8,388,608 | `:2795` |
| `copyField(gammaB→gammaA)` | 1,048,576 | 8,388,608 | `:2527` |
| `copyField(vertexPhiScratch→vertexPhiField)` | 1,098,500 | 8,586,756 | `:2500` |
| Surface-correction `copyTextureToTexture` | 1,098,500 | 8,586,756 | `webgpu-uniform-surface-volume.ts` |
| `clearBuffer(reductions)` / `(rigidExchange)` | 40 / 576 | same | `:2641–2642` |
| Phi-region header copies ×2 | 2 × 1024 | same | `:2199–2200` |
| Pressure cycle demand copy | 12 | same | `:2854` |
| `writeBuffer(params)` | 208 (52 f32) | same | `:1436` |

**The donor-sum clear alone is ~25 MB of write traffic per advance at 64³ and
~201 MB at 128³** — at 400 GB/s that is ≈63 µs and ≈503 µs of pure zero-fill.
**[V] for the bytes, [I] for the µs.**

The *full-buffer* `clearBuffer(conditioningScratch)` calls at `:2715`, `:2766`
and `:2789` are inside the **non-geometric else-branch** and never execute.
**[V]**

---

## 4. Answers to the six questions

### Q1 — Velocity advection

**Semi-Lagrangian** (`semiLagrangianAdvection`, 1124–1147). Per live cell it
calls `advectVelocityComponent` three times (one per positive MAC face), each
of which runs `departurePoint` (529–539): a **bounded 32-iteration RK2 loop**
with `stepSeconds = min(remaining, 1.5/max(rate,1e-6))`, so the trip count is
data-dependent on local CFL — **the loop is a divergence source within a
SIMD32 group** [I].

Per RK2 substep: 2 × `sampleVelocity` = 6 × `sampleVelocityComponent`
(476, 462–475). Each `sampleVelocityComponent` issues **8 `textureLoad(transportIn, …)`
on an `rgba32float`** (472). Sixteen bytes are fetched per tap; **one 4-byte
channel is used**. That is **32× overfetch** (128 B loaded, 4 B consumed per
corner set). **[V]**

Cost per cell at CFL≈1 (2 substeps): 3 faces × 2 substeps × 6 components × 8
taps = **288 taps ≈ 4.6 KB loaded for 3 output floats**. Worst case (32
substeps) is 4,608 taps ≈ 73.7 KB. **[V] counts, [I] the CFL assumption.**

Each `sampleVelocityComponent` also runs `uvTwoLevelFineAt(p)` first (466),
which is an **`atomicLoad` on binding 19** (`uniform-volume.wgsl.ts:612–615`).
An `atomicLoad` is a floating-point reassociation barrier and is not cached
like an ordinary load — **one per sample point, i.e. up to 4,608 per cell.**

Momentum interpolation uses `samplePhysicalVelocityComponent` (505–517), which
per component does 8 corners × (2 `velocityPhaseIn` loads + 2 `cellOpenFraction`
+ 1 `velocityIn` load). `cellOpenFraction` (364–367) calls
`staticSolidVoxelOccupied`, which is **5 storage-buffer loads including a magic-word
check and `u32(round(params.dropExtent.z))`** — a float round on a uniform to
recompute a constant base offset, **on every solidity test** (253–262). **[V]**

**Hardware trilinear:** binding 15 `transportSampler` is declared `filtering`
and `transportCoordinate()` (452) exists — **both are dead**: binding 14 is
`unfilterable-float` and no `textureSampleLevel` appears in the file. **[V]**
Wiring it would collapse 8 taps to 1 and would be *bit-exact only if* the
weights matched; they do not — the shader folds the 8 terms through `d4Sum8`
(265–276), a D4-canonical compensated sum that hardware filtering cannot
reproduce. So a filtered path is **not bit-exact** [I]. `rgba16float` storage
would also lose mantissa. A legal *exact* win is splitting `transportIn` into
three `r32float` planes: the same values, 1/4 the bytes moved.

`uniform-texture-pages.ts` **is live** — its `shader()` rewrites every
`textureDimensions(X)` into a uniform read from binding 34 (`:47–80`, line 55,
unconditional). `uniform-velocity-pages.ts` is applied only when
`!nativeStorage` and is **not live** (`:19`). `uniform-page-redistance.ts` is
referenced only by `tests/uniform-page-redistance-dawn.test.ts` — **dead**.
`uniform-field-loop-bounds.ts` is **not applied** because `literalLoops =
nativeRootExecution = true`.

### Q2 — Per-cell resident state

| Resource | Format | Bytes/cell | @64³ | @128³ |
|---|---|---|---|---|
| velocity A–D | `rgba32float` | 16 × 4 | 16.8 MB | 134.2 MB |
| transport A,B ((n+2)³) | `rgba32float` | 16 × 2 | 9.2 MB | 70.3 MB |
| FIM scratch ×6 ((n+2)³) | `rgba32float` | 16 × 6 | 27.6 MB | 210.9 MB |
| scalars (pressure A/B, volume A/B, surface A/B, gamma A/B, +1) | `r32float` | 4 × 9 | 9.4 MB | 75.5 MB |
| `conditioningScratch` | `u32` | 12 | 3.1 MB | 25.2 MB |
| `vertexPhiField` + scratch ((n+1)³) | `r32float` | 4 × 2 | 2.2 MB | 17.2 MB |
| `volumeDonorSums` | 6 × u32 | **24** | 6.3 MB | 50.3 MB |
| `volumeEdges` | packed | **80** | **21.0 MB** | **167.8 MB** |
| extension hierarchy (4 tex/level) | `rgba32float` | — | 2.4 MB | 19.2 MB |
| surface-volume correction | `r32float` + partials | — | 3.6 MB | 28.4 MB |
| **Total (excl. pressure multigrid)** | | | **≈97 MiB** | **≈763 MiB** |

**`volumeEdges` at 128³ is 167.8 MB, past WebGPU's default 128 MiB
`maxStorageBufferBindingSize`.** The repo requests the adapter value
(`webgpu-device-limits.ts:31–35`), so it works on M1 Max but is not portable.
**[V]**

Candidates for narrowing: the three FIM scratch pairs (extension distance +
provenance) and `transportA/B` are `rgba32float` with `.w` mostly unused —
**25% dead padding, ~9.2 MB at 64³ / 70 MB at 128³** [I]. `volumeEdges`
(80 B/cell) is the single largest structure. **Must stay f32:** pressure A/B,
`volumeA/B` and `volumeDonorSums` (the conservative accumulator — the method's
conservation identity is exact-sum based, `d4Sum*`), and `vertexPhiField`.

Per-advance host→GPU traffic is **208 bytes** (`writeBuffer(params)`, `:1436`).
No `mappedAtCreation` on any per-advance buffer.

### Q3 — Extension front

20 passes @64³, of which **only 2 are actual FIM sweeps** (`updateActiveFront`).
The other 18 are bookkeeping and hierarchy: 1 clear, 1 seed, 2
`prepareActiveDispatch` (**`@workgroup_size(1)`, a one-thread pass whose only
job is `atomicLoad`/`atomicStore` of the indirect args**, 405–424), 1 resolve,
6 restrict, 5 prolong, 1 `prolongAndPack`, 1 `packTransportShell`, 1
`publishCoarseVelocityTable`. **The sweep:bookkeeping ratio is 2:18.** **[V]**

Multiple sweeps per dispatch in threadgroup memory: a (4,4,4) group covering a
4³ tile plus a 1-cell apron is 6³ = 216 nodes × 16 B = 3.5 KB — comfortably
within the 32 KB M1 threadgroup budget, so 2 sweeps could fuse into one
dispatch with a `workgroupBarrier` between them [I]. This is **not bit-exact**:
`godunovDistance` (252–283) and `activatedByConvergedUpwindNeighbor` (331–350,
up to 7 `godunovDistance` calls) read across tile boundaries, and the halo
values after sweep 1 would be stale.

Hierarchy fill levels **cannot** merge: `restrictKnownVelocity` level *k*
consumes level *k+1*'s output, a strict serial chain of 6 (64³) or 7 (128³)
dispatches. The bottom three levels are 4³, 2³, 1³ — **each a full pass launch
for ≤64 threads**. Collapsing the bottom three into one resident pass removes
2 launches (~3–6 µs each direct) at zero numerical change if the reduction
order is preserved [I].

### Q4 — Uniform-guarded blocks that are `override` candidates

| Guard | Read from | Geometric default | Ever taken? |
|---|---|---|---|
| `params.twoLevel.z>0.5` | uniform | `on` | **always** — dead branch is the `off` arm |
| `params.physical.z>=0.0` (line 466) | uniform | `on` | **always** |
| `liquidOnlyVelocityAdvection` (16-probe recovery loop, 625–649) | uniform | off | **never** |
| `sigmaOverRho>0` curvature (1088–1122) | uniform | 0 unless surface tension | usually never |
| `params.boundary.w<=0.5` (open lid, 521) | uniform | scene-dependent | both |
| `gammaDiffusionIterations` | host loop | 0 | never (host-eliminated) |
| `MACCORMACK_AUDIT_ENABLED` | WGSL `const false` | — | Tint folds |

`override` is already used for the volume path (`UV_SHARPEN_TILE_WORK`,
`uniform-volume.wgsl.ts:433`) but **not for `twoLevel.z` / `physical.z`**,
which gate the hottest branch in the file — the per-sample choke at line 466.
Promoting those two to `override` lets Tint delete the `off` arm and, more
importantly, hoists the branch out of the RK2 inner loop [I].

### Q5 — Tint/MSL hazards

- **Runtime-sized arrays with robustness clamps:** `activeScratch`,
  `conditioningScratch` and `sharpenDeposits` (binding 19) are unsized
  `array<…>`. With Chrome robustness on, every index is clamped against a
  runtime length — a dependent load before each access. `staticSolidVoxelOccupied`
  performs 5 such accesses per call. **[V] for the accesses, [I] for the clamp cost.**
- **Non-constant div/mod:** grid dims come from `dims()` (129), which reads
  `textureDimensions` — rewritten by `uniform-texture-pages.ts:55` into a
  **uniform buffer read**, so all index arithmetic is against runtime values.
  Nothing is `override` or `const`. Metal cannot strength-reduce the
  multiplies/divides. **[V]**
- **`vec3` in structs:** the params uniform is 52 `f32` written as a flat array
  (`:1436`) and read as `vec4` groups — no vec3 alignment waste there.
- **f32↔u32 churn:** `volumeCorrectionDivergence` (1256–1271) does
  `bitcast<f32>(atomicLoad(&sharpenDeposits[…]))`; `reduceDiagnostics` (2044)
  does `bitcast<u32>(speed)` and float→u32 quantizations (`×2048.0+0.5`).
- **`select` vs branch:** `sampleVolume` (453–461) and
  `sampleVelocityComponent` (467–472) use `select` for corner weights — good,
  branch-free. But `sampleVolume` line 458 uses `select(0.0, weight*volume(donor), valid&&!solid)`,
  which **still evaluates `volume(donor)` unconditionally** (WGSL `select` is
  not short-circuiting) — that is the intent here, but it means the solid test
  buys nothing on the load path.
- **`shader-f16` / `subgroups`:** both are *requested* if the adapter exposes
  them (`lib/core/gpu-startup.ts:198`), but **no uniform WGSL module contains
  `enable f16` or any subgroup builtin** — the features are acquired and unused.
  **[V]** On M1 Max the f16 ALUs are therefore idle.

### Q6 — Diagnostics, reduction, readback

`reduceDiagnostics` (2044–2045) is one dense `ceil(n/4)³` pass doing **4
device-scope atomics per cell** (`atomicAdd` ×2, `atomicMax` ×2) into a single
40-byte buffer — **262,144 atomics × 4 = ~1.05 M atomic ops per advance at
64³, all contending on 4 words**. **[V] the count, [I] the contention cost.**

Per-advance CPU sync in the **app** path: **one 12-byte `mapAsync`**
(`readPressureCycleDemand`, 2334–2353) and **one 208-byte `writeBuffer`**
(`:1436`). `readStats()` (2960+) issues ten `copyBufferToBuffer` into a
256-byte readback plus a `mapAsync`, but it is called **only from
`setSimulationRunning(false)`** (`lib/core/webgpu-renderer.ts:2220`) — pause,
not per frame. **[V]** The `windowReadback` path (2902–2912) is gated on
`activeRegionEnabled` and is **dead** for geometric.

`symmetryStageAudit*` copyField pairs appear ~12 times in `advanceTo` and are
all harness-only (undefined at shipping defaults). **[V]**

---

## 5. Ranked candidate changes

| # | Mechanism | Removes | Bit-exact? | Confirming measurement |
|---|---|---|---|---|
| 1 | **Hoist the solid-voxel header out of `staticSolidVoxelOccupied`** (253–262): pass `solidVoxelScratchOffsetWords` and the shape as `override`s / a uniform vec4 instead of re-reading 4 words + `round()` per test | 4 of 5 storage loads and one float `round` per solidity test; the test runs ~16× per `samplePhysicalVelocityComponent` call, ~50–150× per advected cell | **Yes** — same bits, fewer loads | A/B the advection + projection stage medians in the stage-cost lane; expect the win to scale with fine-tile count |
| 2 | **Split `transportIn` into three `r32float` planes** (or `rg32float` + `r32float`) so a trilinear tap loads 4 B, not 16 B | 75% of advection gather bytes: ~3.4 KB/cell at CFL 1, ~55 KB/cell worst case | **Yes** — identical values, identical `d4Sum8` order | Bandwidth counter on the advection pass; frame-time A/B at 64³ and 128³ |
| 3 | **Range-limit or eliminate the 4 `clearBuffer(volumeDonorSums)`** — clear only the transport-tile subrange the pages touch (the machinery already exists: `prepareVolumePages` produces the compacted page list, `:2431–2451`) | 25.2 MB @64³ / 201 MB @128³ of zero-fill per advance | **Yes** if the cleared set is a superset of the written set | Stage-cost delta on the three donor rounds; verify conservation via the existing volume-drift diagnostic |
| 4 | **Promote `params.twoLevel.z` and `params.physical.z` to `override` pipeline constants** | The per-sample-point branch at line 466 (up to 4,608 evaluations/cell) plus the dead `off` arm; enables Tint to hoist `uvTwoLevelFineAt` out of the RK2 loop | **Yes** at the chosen setting | Compare MSL disassembly (branch count in the departure loop); stage A/B |
| 5 | **Replace the `atomicLoad` reads of the 4h class table** (`uvTwoLevelFineAt`, `uvCoarseFace`, `uvCoarseVelocityComponent` — 1, 1 and 8 atomics per call) with plain loads from a **second, non-atomic binding aliasing the same range**, published once per advance | Removes the FP-reassociation barrier and the uncached atomic path from the hottest inner loop | **Not guaranteed** — removing the barrier lets Metal reassociate the surrounding float math (see the `atomicLoad is an FP barrier` finding) | Bit-compare a 60-step mini run; if it diverges, measure the divergence magnitude against the dust floor |
| 6 | **Fuse the bottom three extension hierarchy levels (4³, 2³, 1³) into one resident pass** | 2 pass launches/advance (direct ≈3–6 µs each) plus 2 barrier round-trips | **Yes** if the restrict reduction order is preserved | Pass-count census + extension-stage median |
| 7 | **Drop `prepareActiveDispatch` (`@workgroup_size(1)`) by folding the indirect-arg write into the tail of `updateActiveFront`** | 2 single-thread passes/advance | **Yes** | Pass-count census; extension-stage median |
| 8 | **Two-stage the diagnostics reduction**: workgroup-local reduce (the file already has a 256-lane shared-memory tree at 1794–1831) then one atomic per workgroup | ~1.05 M device atomics → 4,096 at 64³ | **No** for `atomicAdd` of quantized u32 (order changes the u32 sum only if it overflows; in practice **yes** for `atomicMax`) | Compare reported volume/speed against the current values over 100 steps |
| 9 | **Narrow the six FIM scratch textures and `transportA/B` from `rgba32float` to `rgb`-packed or three planes** | ~9.2 MB @64³ / ~70 MB @128³ of resident padding, and 25% of every extension load | **Yes** if `.w` is genuinely unused (verify per texture) | Resident-bytes census + extension bandwidth |
| 10 | **Wire the declared `transportSampler`** (binding 15) with `rgba16float` transport | 8 taps → 1 filtered tap per sample | **No** — loses `d4Sum8` compensated ordering and f32 mantissa | Only worth measuring as an upper bound on the gather win; use it to size candidate 2 |
| 11 | **Delete the 24 dead entry points** from the geometric shader variant | Shader-module compile time only; no runtime effect | **Yes** | Pipeline-creation wall time at scene load |

**Highest-confidence pair:** #1 and #3. Both are pure bookkeeping removals with
no numerical exposure, and #3 is the single largest identifiable byte movement
on the non-pressure path at 128³.

**Biggest structural lever, highest risk:** #5 — the 4h class table is read
once per *sample point* inside the RK2 loop, so it is the most-executed
instruction on the advection path, but it is exactly the atomic that the
repo's prior finding says holds the float reassociation in place.
