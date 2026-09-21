# Activity and sparsity state audit — Uniform Geometric

Static read of `main`-tracked source at `codex/uniform-paging-2026-09-21`
(918fc273). No GPU was used. Everything below marked **V** is verified against
code at the cited `file:line`; everything marked **I** is inference and is
labelled as such. No ranked plan is offered.

Baseline configuration for every statement: `uniformGeometricSolverOptions`
defaults — `pageDomain:true`, `volumePages:32`, `activeRegion:false`,
`pressureWindow:false`, `transportTiles:true`, `transportReach:1`,
`twoLevelVelocity:true`, `twoLevelFineReach:2`, `twoLevelExtensionTiles:true`,
`twoLevelShellReach:1`, `twoLevelAdvectionTiles:true`, `geometricTileWork:true`,
`volumeDustThreshold:1e-3`
(`lib/methods/uniform/uniform-geometric-options.ts:12-38`,
`uniform-geometric-parameters.ts:20-51`).

Two residency facts gate the whole audit. **V** `activeRegionEnabled` is forced
false by `options.pageDomain !== true` (`webgpu-uniform-reference.ts:758`), so
the solve window is dead: `uvWindowMin/Max` read the host-published `[0,dims)`
and every `uvInWindow`/`uvTileInWindow` test folds to a domain test
(`uniform-volume.wgsl.ts:56-68`). **V** `volumePageConfig.work` requires
`max(nx,ny,nz) > 64` (`webgpu-uniform-reference.ts:974`), so at mini64 and below
the compact work list and its indirect dispatch do not exist, and
`runVolumeWork` falls through to the dense `this.run`
(`webgpu-uniform-reference.ts:2453-2455`).

---

## 1. Inventory of activity structures

### 1.1 Live under geometric defaults

| # | Structure | Granularity | Predicate (exact) | Reach | Producers | Storage | Lifetime | Consumers |
|---|---|---|---|---|---|---|---|---|
| A | Two-level class map: FINE(1) / SHELL(2) / TRANSPORT(4) | 4³ cell tile | seed: `abs(volume(id))>=dust` ∥ `dropSource>0` ∥ `inflowSweptPlugSource>0` ∥ `uvOpen(id)<0.99999` (FINE/SHELL only) ∥ any of the 5³ tile-corner vertices with `uvPhiIn < 4*max(h)` (`uniform-volume.wgsl.ts:666-684`) | FINE `k=2`; SHELL `k+max(twoLevel.x,1)=3`; TRANSPORT **measured** `clamp(ceil((ceil(D)+1)/4)+margin−8,0,16)` (`:710-715`) | 4 passes, all dense `@workgroup_size(4,4,4)` over `ceil(tiles/4)³` (`webgpu-uniform-reference.ts:2680-2689`): `uvTwoLevelSeed` (one thread per **tile**, serial 64-cell + 125-vertex loop, `:651-685`), `uvTwoLevelDilateX/Y/Z` (`:735-749`) | `conditioningScratch` word `cellCount()+4*t+3` for the class; planes at `cellCount()+4*cc+p*cc` are the dilation ping-pong; 3 counter words at plane 2 (`:607-611`, `uniform-volume.wgsl.ts:22-31`) | Per step, mid-step (after phi, before advection) | `uvTwoLevelFineAt` (1 `atomicLoad`, `:612-615`); `uvTransportSkip` (`:320-331`); `shellAt` in the extrapolator (`webgpu-uniform-velocity-extrapolation.wgsl.ts:108-113`); host counters via `readStats` |
| B | 4h coarse face-velocity table | 4³ tile, 3 words | published unconditionally from the extrapolator's `ceil(n/4)` hierarchy level | — | `publishCoarseVelocityTable` (`webgpu-uniform-velocity-extrapolation.wgsl.ts:697-715`) | same 4-word tile record, words 0..2 | Per step | `uvCoarseFace` → `uvCoarseVelocityComponent`, **8 `atomicLoad` per component per sample point** (`uniform-volume.wgsl.ts:617-639`) |
| C | Sharpen tile map | 4³ tile | `!(abs(uvPhi(cell+0.5)) >= tuning.y*h)` (2.1 h band), OR-reduced over the 64 cells by a workgroup atomic (`uniform-volume.wgsl.ts:450-474`) | none (0) | `uvClassifySharpenTiles`, 1 dense pass, `@workgroup_size(4,4,4)` over `ceil(n/4)³`; lane 0 stores and does a **contended global** `atomicAdd` into the count word | `conditioningScratch` word `2*cellCount()+8+t.x+d.x*(t.y+d.y*t.z)`; count at `2*cellCount()+7` (`:434-438`) | Per step, re-run before each sharpening block | `uvSharpenTileActive` — **2×** in `uvLimitedFlux` (`:578-594`), **6× + entry test** in `uvCommitSharpen`, entry test in `uvCacheSharpenCells` (`:478`); `uvPageSharpenActive` (`uniform-volume-pages.wgsl.ts:47-50`) |
| D | Phi region AABB + indirect dispatch | cell AABB → workgroup counts | `geometricActiveSeed(id)` = `abs(volume(id))>dust` ∥ any of 8 corners `uvPhiIn < 4*max(h)` (`webgpu-uniform-reference.wgsl.ts:236-245`) | padded by `VERTEX_PHI_REACH=6` (`:201`) | 3 direct passes — `scanExternalActiveSources` (dense), two 256-lane reducers, `finalizeActiveRegion` **@workgroup_size(1)** — plus 2× 1024 B `copyBufferToBuffer` (`webgpu-uniform-reference.ts:2194-2211`) | `activeRegion` buffer (binding 29, read-only) + `phiRegion`/`phiDispatch` | Per step head | `runVertex` takes `dispatchWorkgroupsIndirect(phiDispatch)` (`webgpu-uniform-reference.ts:1747-1761`) — **the only indirect dispatch on the geometric path** |
| E | 32³ page flags (transport phase / sharpen phase) | 32³ page | transport: `!uvTransportSkip(id)`; sharpen: `uvPageSharpenActive(id)` plus the 3-axis neighbour backing loop (`uniform-volume-pages.wgsl.ts:41-62`) | sharpen adds ±1 cell / +4 cells per axis | `uvMarkTransportPages` / `uvMarkSharpenPages` — dense `@workgroup_size(4,4,4)` where **63 of 64 lanes retire on `any(gid%4!=0)`**; then `uvCompactPages` **@workgroup_size(1)**, serial loop over `UV_PAGE_COUNT` (`:64-81`) | `conditioningScratch` from `p.base`: 8 header words, `count` flags, `count` slots (`webgpu-uniform-reference.ts:2431-2451`) | Twice per step (transport phase, sharpening phase); transport copy snapshotted into `volumeTransportPageView` before sharpening overwrites the flags (`:2449`) | `uvEdgeAddress` — **but identity under `nativeRecords`** (`uniform-volume-pages.wgsl.ts:33-39`); otherwise **only the field-view grid overlay** (e49add19) |
| F | Dust floor | per cell | `abs(V) < tuning.z` → store 0, accumulate discarded mass | — | inline in every transport/sharpening write, 2 contended device atomics (`uniform-volume.wgsl.ts:382-387`) | `reductions` | per write | It is the *enabling predicate* for A's TRANSPORT bit: "V = 0 outside the set" |
| G | Extension active-state texture / FIM front | cell | extrapolator-internal | front sweeps = 2 (`uniform-geometric-parameters.ts:10`) | extrapolator passes | own texture | per step | extension only |

### 1.2 Present but inert at defaults

| Structure | Why inert |
|---|---|
| Solve window `activeRegion[7..12]` | **V** `activeRegionEnabled=false` (`webgpu-uniform-reference.ts:758`); `encodeActiveRegion` returns immediately (`:2213-2215`). Host publishes `[0,dims)`. |
| `pageDomain` membership array + cell/vertex indirect dispatch words | **V** `nativeRootExecution` true → `pageDomainPublication?.encode` skipped (`:2643`); `activeId`/`activeVertexId` fold to `pageDomainCell/Vertex`, which is identity for a rectangular origin catalogue (`uniform-page-domain.ts`, `uniform-page-execution.ts`). |
| Compact 4³ work list + `volumeWorkDispatch` indirect args | **V** `work:false` at ≤64³ (`:974`); `uvAppendWork` and `uvWorkId`'s lookup compile to nothing (`uniform-volume-pages.wgsl.ts:16-30`). |
| `uvEdgeAddress` page→slot indirection | **V** identity when `nativeRecords` (`uniform-volume-pages.wgsl.ts:33-39`). |
| `uniform-page-support.ts` / `uniform-page-generation.ts` residency classify/emit | **V** no call site in `webgpu-uniform-reference.ts`. |
| `pressureWindow`, `mgActiveId` clipping | **V** `pressureWindow:false`; pressure multigrid is dense whole-level. |

---

## 2. Re-ask matrix — one advance

Rows are stages in encode order; cells say which structure the stage *reads*
(R) or *produces* (P).

| Stage | A class map | B 4h faces | C sharpen tiles | D phi AABB | E page flags |
|---|---|---|---|---|---|
| `encodePhiRegion` (step head) | — | — | — | **P** | — |
| phi advect / redistance (`runVertex`) | — | — | — | **R** (indirect) | — |
| `uvTwoLevelSeed` + 3 dilates | **P** | — | — | — | — |
| `publishCoarseVelocityTable` | — | **P** | — | — | — |
| extension finest passes | R (SHELL) | — | — | — | — |
| `semiLagrangianAdvection` | R (FINE, per sample point) | R | — | — | — |
| `prepareVolumePages(transport)` | R (TRANSPORT) | — | — | — | **P** |
| transport gather/normalize/commit | R (TRANSPORT) | — | — | — | R (identity) |
| `uvClassifySharpenTiles` | — | — | **P** | — | — |
| `prepareVolumePages(sharpen)` | — | — | R | — | **P** |
| sharpening sweeps ×8 | — | — | R (2–7× per invocation) | — | R (identity) |
| `project` | R (FINE) | — | — | — | — |

**Provable set relations (V).**

1. **C ⊂ A-SEED.** `uvClassifySharpenTiles` admits on `|φ_cell| < 2.1h`
   (`uniform-volume.wgsl.ts:458`, `tuning.y` default 2.1). `uvTwoLevelSeed`
   admits on the one-sided vertex test `φ_vertex < 4·max(h)`
   (`:684`). A cell centre is the mean of its eight vertices, so
   `|φ_cell| < 2.1h ⇒` some vertex `< 4h`. Every sharpen tile is therefore a
   SEED tile — before any dilation. C is recomputed by a separate dense pass
   that re-samples `uvPhi` 64× per tile.
2. **A-TRANSPORT ⊆ A-SEED**, by construction: `transportSeed` is set by the
   same liquid/source/vertex clauses minus the `uvOpen < 0.99999` solid clause
   (`:670-673`, `:684`), then dilated by a *different* reach (`m` vs `k`,`s`)
   in the same three scans (`:721-733`). One pass produces all three sets;
   this is the one place that already shares.
3. **E-transport is a 32³ dilation of A-TRANSPORT.** `uvMarkTransportPages`'s
   only predicate is `!uvTransportSkip(id)` (`uniform-volume-pages.wgsl.ts:44`),
   i.e. bit 4 of A, coarsened from 4³ to 32³ by an OR over 512 tiles. It is a
   pure re-derivation of a set already in memory.
4. **E-sharpen is a 32³ dilation of C** (`uvPageSharpenActive` reads
   `uvSharpenTileIndex`, `:47-50`), with a ±1 cell / +4 cell neighbour backing
   loop that in 32³ terms only ever adds pages adjacent along an axis.
5. **D-seed and A-SEED are the same predicate at different granularity.**
   `geometricActiveSeed` = `|V| > dust` ∨ 8 corners `φ < 4·max(h)`
   (`webgpu-uniform-reference.wgsl.ts:236-245`); `uvTwoLevelSeed` = `|V| ≥ dust`
   ∨ source ∨ `open<1` ∨ 5³ corners `φ < 4·max(h)` (`:670-684`). Same fields,
   same band constant, different `>`/`≥` on the dust test and a solid clause.
   D reduces them to an AABB in 3 passes; A reduces them to a tile map in 1.
6. **`uvTwoLevelFineAt` and `uvTransportSkip` read the same word** —
   `sharpenDeposits[uvCoarseBase()+4*uvCoarseIndex(t)+3]` — with masks `1` and
   `4` (`:612-615`, `:320-331`). Each is a separate `atomicLoad`.

---

## 3. Need matrix

Define the **seed set S** as the 4³ tiles satisfying A's `seed` clause: liquid
above the dust floor, a source cell this step, a partially-solid cell, or a tile
vertex with `φ < 4·max(h)`. `S⊕k` is the Chebyshev dilation by `k` tiles.

| Stage | Minimal need | Runs today | Gap |
|---|---|---|---|
| `encodePhiRegion` scan | dense (it is the census) | dense | — |
| phi advect / redistance | vertex lattice over `S⊕⌈6/4⌉` | indirect over D's AABB | AABB is a bounding box of a possibly disconnected set (**I**) |
| `uvTwoLevelSeed` | dense (it is the census) | dense, 1 thread/tile | 64+125 serial iterations per thread; 4³ workgroup covers 64 tiles |
| dilate ×3 | dense over tiles | dense over tiles | — |
| extension finest | `S⊕3` (SHELL) | dense launch, `shellAt` per-thread early-out | 10 call sites read SHELL (`…extrapolation.wgsl.ts:216,239,244,321,355,431,460,648,666,673`) |
| `semiLagrangianAdvection` | `S⊕2` (FINE) | dense launch, per-thread early-out that **still stores 3 textures** (`webgpu-uniform-reference.wgsl.ts:1124-1135`) | stores on the far-air arm |
| transport passes | `S_T⊕m` (TRANSPORT) | dense launch, `uvTransportSkip` early-out; `uvGather` **stores 0 to both ping-pong targets** outside the set (`uniform-volume.wgsl.ts:388-407`) | stores on the skip arm |
| `uvClassifySharpenTiles` | dense (census) | dense | duplicates §2.1 |
| sharpening ×8 | C | dense launch, early-out | 2–7 `uvSharpenTileActive` per invocation |
| `project` | `S⊕2` (FINE) | dense launch, early-out that **still stores velocity, boundary velocity and volume** (`:1302-1313`) | stores on the far-air arm |
| pressure multigrid | rows only | **dense whole-level**, `mgActiveId` doing 5 dead storage loads | 651 of 753 passes |

**Cost of the early-out form (V + I).** A per-thread early-out on a dead 4³
tile still launches 64 threads and issues at least one `atomicLoad` on binding
19 each; `uvGather` and `project` additionally issue their texture stores. Not
dispatching the tile at all costs zero. **I**: this is the difference between
"sparsity that removes arithmetic" and "sparsity that removes launches"; today
only the phi stage does the latter.

---

## 4. Pages → tiles

**V** The "page activity states" of e49add19 are exactly structure E: the
transport-phase and sharpening-phase flag arrays inside `conditioningScratch`,
snapshotted for the transport phase into `volumeTransportPageView`
(`webgpu-uniform-reference.ts:2449`) and surfaced as
`GPUFluidVolumePageSource.transportRecords` / `.workRecords`. Their only
consumer is the field-view grid overlay (`pageActivityAt`,
`layers.pageActivity`), copied per frame with `copyBufferToBuffer` when the
layer is visible. **No solver kernel reads them at defaults**, and they are not
read back to the host. Documented intent matches: "Pressure/interface work is
not inferred from these volume flags"
(`docs/research/uniform-page-coordinate-cost-2026-09-21/native-stages-results.md:103-108`).

**V** Host visibility: `readStats` copies `uniformVolumePagesActive` from word
59 of the stats readback (`webgpu-uniform-reference.ts:2960-3001`), i.e. the
header's `used` slot count, not the flag array. That readback is a lagged
diagnostic, so a host-sized launch derived from it would be **one advance
behind** — the same lag the shipped solve window and the P1 lagged pressure
budget already pad for.

**Launch arithmetic for host-sized direct per-page dispatch.**

| Scene | Dims | 32³ pages | 4³ tiles | Active pages (measured) |
|---|---|---|---|---|
| mini64 | 64³ | 8 | 4,096 | not censused |
| garden hose | 144×96×96 | 45 | 20,736 | **8 / 45** at hose start; **0** with inflow off |
| fig7 | 128³ | 64 | 32,768 | not censused |
| long dam | 192×96×32 | 18 | 9,216 | no census found |

**I** With ~30 volume-path dispatches per advance, per-page direct launches cost
`activePages × 30` launches at 3–6 µs each (memory: `indirect-dispatch-costs-3x`).
Garden at 8 active pages = 240 launches = 0.7–1.4 ms of pure launch overhead
against 30 dense launches (0.1–0.2 ms). mini64 at 8 pages total can never win.
fig7 at 64 pages is worse. The page grid is too coarse to remove much work and
too fine to be free — **V** garden's own census says the *tile* set is 210/20,736
(1.0%) while the *page* set is 8/45 (17.8%)
(`…/garden-transport-results.md:46-48`).

**What breaks under per-page dispatch (V).** (a) `uvLimitedFlux` reads the
neighbouring record, which is why `uvMarkSharpenPages` already backs every
axis-adjacent page (`uniform-volume-pages.wgsl.ts:56-61`). (b) The phi stage is
on the `(n+1)³` vertex lattice, which does not tile by 32³ cells without a
shared boundary plane. (c) `uvGather` and `project` write their ping-pong
targets on the *inactive* arm; skipping the launch leaves the destination
holding the previous step's bits unless the swap discipline changes. (d) The
`uvTwoLevelSeed`/classify censuses must stay dense or they cannot discover new
activity.

---

## 5. Existing scene sparsity numbers

All quoted; **most predate `pageDomain` default-on and the 1e-3 dust floor**.

| Scene | Quantity | Value | Source |
|---|---|---|---|
| fig7 128³ | live transport tiles | 10.6 % → 20.5 % (free fall), 44.6 % (impact), 41.6→52.6 % (spread) | `…/uniform-geometric-tile-first-2026-09-19/live-tile-audit.md:398-410` |
| fig7 128³ | live tiles of 32,768; max displacement D | 3,542 (10.8 %) / 6,328 (19.3 %) / 7,873 (24.0 %); D = 27.2 cells | `…/e3-transport-live-set-report.md:50-54` |
| mini64 | live transport tiles | **4,081 / 4,096 = 99.6 %** | `…/e3-transport-live-set-report.md:189-195` |
| mini32 | live transport tiles | 512 / 512 = 100 % | same |
| fig7 | FINE tiles at k=2 | 6.6 %–22.9 %, mean ≈ 14 % | `…/e1-implementation-report.md:224-232, 245` |
| fig7 / mini64 | dust-floor effect on the live set | 9,411 (28.7 %) → 3,159 (9.6 %) / 4,001 (97.7 %) → 2,828 (69.0 %) | `…/e1-implementation-report.md:67-88` |
| mini32 / mini64 / large-power 64×20×64 | band occupancy | 56.3/76.6/100 % · 47.3/56.3/76.2 % · 6.3/9.4/18.0 % | `docs/research/uniform-geometric-empty-air.md:68-76` |
| mini64 / large-power | extension far tiles | 43.4 % (1778/4096) / 79.2 % (1014/1280); 45 % / 19 % of ceiling realised | `docs/benchmarks/uniform-geometric-extension-tile-work-2026-09-19.md:51-58` |
| garden 144×96×96 | tiles / pages at hose start | **210 / 20,736 tiles and 8 / 45 pages**; zero of both with inflow off | `…/uniform-page-coordinate-cost-2026-09-21/garden-transport-results.md:46-48` |
| garden | sharpening pages | "grew from three to six of 45 logical pages" | `docs/research/uniform-volume-pages-production-2026-09-21.md:44-46` |
| mini64 | dense vs 32³ pages, full step | 66.65 ms vs 67.17 ms (99.2 %) | same, `:56-58` |
| long dam 192×96×32 | any live-set census | **none found** | — |
| tall-air | live-set census | **none found**; `…/uniform-geometric-tall-air-2026-09-19` records the pressure full-level share only | — |

**Staleness (V).** `docs/research/uniform-geometric-work-map-candidates.md`,
named in the brief, **does not exist** in the tree. The 09-19 censuses were
taken before `pageDomain:true` became the geometric default and before the
09-21 native-execution changes; the garden numbers are the only post-page
census.

---

## 6. Lookup cost today

| Site | Cost | Frequency |
|---|---|---|
| `uvCell(i)` | 3 runtime div/mod against `dims()` — **V** and `dims()` is itself a uniform-buffer load after `uniform-texture-pages.ts:55` rewrites `textureDimensions` | every kernel that starts from a linear id |
| `uvSharpenTileIndex` | `(dims()+3)/4` (3 divides) + `2*cellCount()+8+t.x+d.x*(t.y+d.y*t.z)` | 2× in `uvLimitedFlux`, 6×+1 in `uvCommitSharpen`, ×8 sweeps |
| `uvCoarseIndex` / `uvCoarseBase` | `(dims()+3)/4`, 3 mults, `cellCount()` | every `uvTwoLevelFineAt`, `uvCoarseFace` |
| `uvTwoLevelFineAt` | 1 `atomicLoad` on binding 19 | **once per sample point inside the RK2 loop** (`webgpu-uniform-reference.wgsl.ts:466`) |
| `uvCoarseVelocityComponent` | **8 `atomicLoad`** | per velocity component per far-air sample |
| `uvTransportSkip` | 1 `atomicLoad` | per transport-kernel invocation |
| `uvPageIndex` / `uvEdgeAddress` | 3 divides by `UV_PAGE_EDGE` (compile-time const) + 1 `atomicLoad` | **elided**: identity under `nativeRecords` |
| `staticSolidVoxelOccupied` | 5 storage loads incl. `u32(round(params.dropExtent.z))` | per solid test |

**Atomic reads are an FP barrier.** Repo memory `atomic-load-is-an-fp-barrier`
records that moving float reads off an atomic binding reassociates the maths.
`uvCoarseFace` `bitcast`s an `atomicLoad` into `f32` (`uniform-volume.wgsl.ts:617-619`)
— **I** these 8 loads per component are therefore both a latency chain and a
scheduling barrier inside the interpolation.

**What a precompiled per-frame structure could remove (I, with V budget).** A
tile-class 3D texture at `ceil(n/4)³` in `r32uint` would be addressed by
`id>>2` with `textureLoad`: no div/mod (shift by a compile-time 2), no atomic,
no FP barrier, and it is not against the storage-buffer limit. **V** the main
bind-group layout uses **10 storage buffers** (33, 9, 10, 11, 19, 26, 27, 28,
29, 30 — at the adapter's ten-per-stage limit, as the code comments at
`webgpu-uniform-reference.ts:1193-1194`), **12 sampled textures** (31, 0, 2, 4,
7, 12, 13, 14, 16, 20, 21, 24) against a default 16, and **6 storage textures**
(32, 1, 3, 5, 8, 25) against a default 8 (`:1065-1099`). **V**
`requiredFluidDeviceLimits` requests the *adapter's* values for all three
(`lib/core/webgpu-device-limits.ts:33,46,50`), so the real ceilings are
adapter-reported, not the spec defaults. There is headroom for one or two more
textures and none for another storage buffer.

**V** `workgroup_id` already *is* the tile id for every `@workgroup_size(4,4,4)`
pass — `uvClassifySharpenTiles` uses `activeId(tile*4u)` to address its own map
(`uniform-volume.wgsl.ts:472`). **I** What is *not* expressible in WGSL is a
data-dependent workgroup count without an indirect dispatch, and indirect was
measured at +40.7 % whole-advance; the only way to a data-dependent count that
has ever paid here is a host-sized direct launch from a lagged readback.

---

## 7. Facts that constrain any unified design

1. **V** The transport live set is only sound because of the dust floor:
   "every cell outside the set holds V=0 — which is what the dust floor
   guarantees" (`uniform-geometric-parameters.ts:48`). Any unified set inherits
   that dependency; `transportTiles` is forced dense when the dust floor or the
   two-level sampler is off.
2. **V** `uvSharpenTileActive` must stay window-aware: "The window bound is part
   of the predicate, not an optimization: the eight sweeps read a NEIGHBOUR
   tile's proposals out of the fixed stencil arena, and outside the window that
   arena still holds this step's transport edges. A tile the classify did not
   visit is not a sharpening tile" (`uniform-volume.wgsl.ts:440-444`).
3. **V** Vertex-path reach is 6 cells: `VERTEX_PHI_REACH:u32=6u`
   (`webgpu-uniform-reference.wgsl.ts:201`). Redistancing reads advected phi at
   that reach; a tighter phi set silently truncates it.
4. **V** The transport reach is *measured*, not authored: `uvTwoLevelSeed`
   `atomicMax`es the domain-max backward displacement D one dispatch before
   `uvTwoLevelDilate` reads it (`uniform-volume.wgsl.ts:678-679`, `:710-715`),
   and it is `clamp(...,0,16)` — a capped step is a real possibility the host is
   expected to observe. A unified structure must preserve that producer→consumer
   ordering within the same step.
5. **V** Stale ping-pong targets: `uvGather` stores 0 to *both* ping-pong
   targets outside the live set (`uniform-volume.wgsl.ts:388-407`), and
   `semiLagrangianAdvection` / `project` still store on their far-air arm
   (`webgpu-uniform-reference.wgsl.ts:1124-1135`, `:1302-1313`). Converting an
   early-out into a skipped launch changes what the untouched destination holds.
6. **V** `reductions` and `rigidExchange` are cleared at the step head
   (`webgpu-uniform-reference.ts:2641-2642`), and the two-level counter words
   get a ranged clear at `twoLevelShellCountOffset` (`:2680`). The unranged
   `clearBuffer(conditioningScratch)` calls at `:2715`, `:2766`, `:2789` sit in
   the **non-geometric** else-branch and never execute on this path — a unified
   map placed in `conditioningScratch` would be wiped by them if that branch
   were ever shared.
7. **V** The tile map's survival depends on ranged clears: the comment at
   `uniform-volume.wgsl.ts:598-601` states the geometric path's per-step clears
   are "ranged away from it so this survives the whole step."
8. **V** The sharpening phase *reuses* the transport page flags, which is why a
   metadata snapshot exists (`webgpu-uniform-reference.ts:2447-2449`). Two
   phases, one array.
9. **V** Transport cannot be bit-validated: `uvAddDonor` uses a float CAS, so
   the transport result is not deterministic across schedules. The owner's bar
   is V conservation and no NaNs, not bit-exactness.
10. **V** `uvReleasedWalls` / the one-sided vertex test is load-bearing for the
    theorem "the FIM accurate band lies inside SHELL": the seed's vertex test is
    `φ < band`, not `|φ| < band`, precisely so that every cell with a negative
    centre phi is in a seed tile (`uniform-volume.wgsl.ts:644-650`). A
    two-sided unification would break it.
11. **V** At ≤64³ the entire page-activity machinery is overhead: the two mark
    passes launch dense with 63/64 lanes retiring immediately, `uvCompactPages`
    runs a serial `@workgroup_size(1)` loop over `UV_PAGE_COUNT`, and the slot
    map it writes is never read because `uvEdgeAddress` is identity
    (`uniform-volume-pages.wgsl.ts:33-39`, `:41-81`,
    `webgpu-uniform-reference.ts:2431-2451`).
12. **V** Known negatives not to re-propose: GPU-indirect window launches
    +40.7 % whole advance; the L0 smoother 4³ tile map slower; the extension
    front work map null; the exact shared skip mask rejected on fig7 because
    reach `2D+7` made 45 % of tiles live (`…/live-tile-audit.md` §B.2–B.3, G).
13. **V** mini64 is not a sparsity scene: 99.6 % of its tiles are live
    (`…/e3-transport-live-set-report.md:189-195`), and dense vs paged measured
    99.2 % throughput. Any set-based win must be gated on a scene where the set
    is actually small — garden (1.0 % of tiles) and large-power (6–18 % band
    occupancy) are the only such censuses in the tree.
