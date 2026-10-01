# Levelset group: phi and surface stages (fig-9, HEAD c38adea3)

Read-only analysis. No repo edits, no GPU runs. Sources: lib/methods/uniform/*.ts at HEAD, the WGSL captured at
`$SCRATCH/wgsl/fig9` (module numbers below), and `$SCRATCH/runs/base/fig9.json`.

## 0. Ground rules for the numbers

**Modules (fig-9 capture, `pipelines.json` / `used.json`)**

- **#20 UniformMixedSurface** (100 KB, 38 compiled, 17 dispatched). It holds advect, traceCells, redistance and the evidence passes.
- **#15** phi resolve (counted fused). **#16** resolveListed (head only).
- **#18** momentum 4h sampling cache.
- **#19** hanging taps (`hanging`, `unitFaces`).
- **#24** surface geometry. The pressure variant is pipeline 185.
- **#25** sharpening (classify, compact, cacheGeometryPrepare, propose, limit, commit).

At fig-9 the solid scene is absent, so the odd-index solid-free twins run. The even solid twins, `solidClosed` and `solidClear` are compiled but never dispatched.

**fig-9 configuration (fig9.json `values`)**

- On: cubic (`flags.y`), drain (`flags.z`).
- Off: preserve (`flags.x` bit 1) and coarse surface travel (`flags.x` bit 2; `mixedCoarseningSurfaceTolerance` = 0).
- Sharpening: 8 sweeps, distance 2.1. Surface-volume rounds: 2.
- The surface is built as `UniformMixedSurface(…, hanging=true, resolved=true)` (uniform-mixed-frame.ts:285). So `vertexCache = hanging && !resolved = false`.

**Sizes**

- Lattice: n = 128×128×64 = 1,048,576 cells; (n+1)³ = 129·129·65 = 1,081,665 vertices.
- T = 16,384 tiles. Peak about 11k h tiles (704k h cells) and about 5k 4h tiles.

**Measured stage times** (windows.all mean; the GPU frame is 22.4 ms)

| Stage | Mean ms | Median ms | Range ms |
|---|---|---|---|
| Vertex phi transport + redistance (advect, resolve, traceCells, redistance, resolve) | 3.645 | 3.28 | 1.97 – 6.23 |
| Surface volume constraint + geometry (geometry is in here) | 1.731 | – | – |
| Conservative volume sharpening | 2.063 | 1.97 | 0.98 – 4.19 |
| 4h transport reach + sampling caches (cache, hanging, unitFaces, certificate) | 0.446 | – | – |

There is no per-kernel split. Every per-kernel ms below is an estimate built from the cost model that follows.

**Cost model (from memory: the frame is load-issue / latency bound and cache resident)**

- What is counted is load *instructions per SIMD group* (32 lanes), not bytes.
- The effective issue rate is R ≈ 2–4 G SIMD-load-instructions/s across the GPU. Prior audits measured 1.8–6.4 G/s, and atomics sit at the slow end.
- Dependent-load levels add latency that 19% occupancy cannot hide.
- 704k h vertices or cells = 22,000 SIMD groups. So each per-lane load removed from an h-wide kernel ≈ 22k issues ≈ **7–11 µs**. This is the yardstick used throughout.

**Atomics note (memory):** `atomicLoad` of a float bit pattern is an FP-optimisation barrier in Metal. De-atomising *integer* words (lists, flags, counts) is bitwise safe. De-atomising float words may let the compiler reassociate.

---

## 1. Kernels in frame order (uniform-mixed-frame.ts:486-503)

### 1.1 `cache` (#18, pipeline 141): 4h sampling cache. Every frame, 1 dispatch.
1. Fills the 4h MAC cache `coarseVelocity`, (T₃+2)³ = 34·34·18 = 20,808 texels rgba32f = **333 KB**. Channels xyz are live; **.w is dead**.
2. 405 workgroups of 4×4×4 lanes, one lane per cache texel.
3. Bindings:
   - Topology group 0 (read).
   - `extended` (velocityScratch, rgba32f n³, 16.8 MB, read).
   - `negative` (read).
   - Cache texture (write).
4. Per lane and per axis: `umOwnerAt` (1 topology load) → `umFace` (1–2) → 1 load, or 16 restriction loads when the face is split. Depth is 3. The lane count is tiny (20.8k), so the cost is about 1 launch.
5. Dead: the .w channel. It is not worth a packing change, because r/rg formats would need 3 textures and add load issues.

### 1.2 `hanging` + `unitFaces` (#19, pipelines 142, 143): every frame, 2 dispatches (uniform-mixed-momentum-cache.ts:130-160)

**What they do**

- `hanging` runs one 256-lane job per hanging slot (slots = fine seams + 4h seams, `UM_HANGING_SLOTS = T`, topology.wgsl.ts:261).
  - Lanes 0–191 evaluate `umVelocityTap1` for the tile's 64 cells × 3 axes into workgroup memory, hit a barrier, and store `unitVelocity` (rgba32f, w=0).
  - Lanes 192–239 evaluate the 48 negative-domain-plane taps into the slot record (only tiles with coord 0 on that axis).
  - Lanes 240–255 are always idle.
- `unitFaces` runs one 64-lane job per h tile and copies `extended` → `unitVelocity` (forces' viscosity reads every unit tile's texels, momentum-cache.ts:166).

**Bindings**

| Binding | Format | Size at fig-9 | Access |
|---|---|---|---|
| `unitVelocity` | rgba32f n³ | **16.8 MB**, .w dead | write |
| `umHanging` | u32 storage | (2T + 173·T)·4 = **11.47 MB** (velocity-sampling.wgsl.ts:12, 22; ownership.ts:180-184) | written by `hanging` |
| `extended` | rgba32f | 16.8 MB | read |
| `negative` | storage | – | read |
| coarse cache | rgba32f | 333 KB | read |

**Dead work (bitwise-safe to delete)**

- `umVelocityTap1` returns at its first case for any unit tile (`tileWidth==1u`, velocity-sampling.wgsl.ts:118-123) *before* it ever consults `umHanging[tile]`. The slot test at :126 is the only reader of the slot records, and the surface `vertexCache` (the only other reader, surface.ts:126, 913-920) is disabled.
- For **every unit (h-seam) slot**, the whole `hanging` job is therefore dead:
  - Lanes 0–191 compute exactly what `unitFaces` overwrites next (same texel, only .w differs and .w is unused).
  - Lanes 192–239 write plane taps that no reader can reach.
- Estimate: h-seam slots ≈ 3–4k of the slots. Per dead job: 192 lanes × (1 topology + 1 texel) + barrier + 64 rgba32f stores = 8 SIMD × about 3 issues. So ≈ 3.5k × 8 × 3 ≈ 84k issues ≈ **0.03–0.05 ms**, plus the barriers.

**Dead storage**

- The 125 vertex-cache words per slot: 125 × 16,384 × 4 = **8.19 MB**, only for the disabled `vertexCache`.
- The 48 plane-tap words per slot (3.15 MB) are read only for *4h* slots on a negative domain plane.
  - About 12% of tiles touch a negative plane (16384 − 31·31·15 = 1,969).

### 1.3 Surface `advect` pass (#20): every frame, 7 dispatches + 1 `clearBuffer` (surface.ts:951-993)

Order: clear claims (24.6 KB) → `wallReach` (384×256) → `advectOwners` regular (2048 claimed) → `advectFine` general (2048) → `advect` merged (2048) → `advectWalls` (1045×64) → *new pass* → `advectDeferred` ×2 twins (1024 each).

**Dispatch shapes**

- `advectFine` (surface.ts:739-747): one 64-lane job per h tile. Lane = positive vertex; there are 2 extra rounds on negative-wall tiles.
  - Lanes whose vertex is owned by a 4h neighbour idle (`umFineVertex` → `UM_NO_VERTEX`).
- `advect` merged (:822-850):
  - Either 8 seam 4h tiles × 8 corner lanes per job, with `umVertexAuthority` plus a canonical test per lane.
  - Or 64 regular coarse owners per job.
- `advectOwners` (:891-911): the regular (certified) owner list. The certificate requires no 4h tile within reach+2 tiles, so at fig-9 the regular h list is nearly empty and the launch is ≈ 2048 claims that exit.

**Bindings (group 1, surface.ts:107-130)**

| Binding | Resource | Size | Access |
|---|---|---|---|
| 0 | `phi` (phiScratch's source), r32f (n+1)³ | 4.33 MB | read |
| 1 | `outputPhi` | 4.33 MB | write |
| 2 | `velocity` = velocityScratch, rgba32f | 16.8 MB | read, x/y/z one at a time |
| 3 | coarse cache | 333 KB | read |
| 4 | `volume` r32f | 4.2 MB | read (drain) |
| 5 | `negative` | – | read |
| 6 | params | – | uniform |
| 8 | evidence (arena) | – | – |
| 10 | deferred | arena, 16 + 4·(n+1)³ B = 4.33 MB bound | atomics |
| 11 | `unitVelocity` | 16.8 MB | read |
| 12 | `umClaims` | 8 + 6 + 2 + 6,160 wall-tile words + 18,513 travel words | atomics |

**Per h vertex, `umAdvectStore` (surface.ts:394-421), solid-free and resolved**

| Step | Loads (per lane) | Dependent level |
|---|---|---|
| Wall-plane test | 0 | – |
| `umTrace` → `umSurfaceTrace` (:134-138): 2 `umSampleVelocity` calls, the 2nd at `mid` depending on the 1st | | |
| – interior sample (tile max width 1): `umFineStencilSample` 2 loads (support word, stencil word) → 3×8 rgba32f | 26 per sample | 2 per sample |
| – blend-zone sample: `umVelocitySamplingWeights` 3 loads → 24 `unitVelocity` + 24 coarse-cache loads | 51 per sample | 2 per sample |
| – domain-edge sample (base[axis] = −1): general path, `umTileMaximumWidth`, `umUnitTaps` (8 topology), `umVelocityTap1` slot chain | up to about 70 | 4–5 |
| `umSampleVertex(q)` resolved: 1 tile-width topology → 8 phi | 9 | +2 |
| Cubic when abs(v) < 2h: `umOwnerAt` 1 topology → 64 phi (`umCubicPhi` :241-269) | 65 | +2 |
| `umReleasedMayChange` (:373-392): 6 `atomicLoad` of `umClaims` per vertex (uniform addresses), +4 atomics per plane whose reach test passes | 6 (+4/plane) | +1 |
| `umDrain` when value < 0.5h (:271-292): tile width → volume; on a miss a box walk (≤ 8 topology + ≤ 64 volume, a 512-iteration ALU loop) that returns on the first hit | 2 (+ rare walk) | +2 |
| Store, or defer through `atomicAdd(deferred[0])` | – | – |

- Interior h vertex totals: 52 trace + 9 + 6 + about 2 ≈ **69 issues**, plus **65** whenever any lane of the SIMD group is within 2h.
- The cubic branch is SIMD-divergent: one near-surface lane makes the whole group pay 65 loads. Near the surface that is most groups.
- The chain is about 4 (trace) + 2 + 2 + 1 + 2 ≈ **11 dependent levels**.
- Fetches are rgba32f texels of which one channel is used, so 3× of each 16 B fetch is dropped on every velocity tap.

**Redundant work inside this chain**

- (a) **The `umOwnerAt` in `umCubicPhi` repeats the topology load `umSampleVertex` just did for the same cell** (`min(floor(q), D−1)`).
  - Tiles are width-uniform, so the owner width and origin are known from it.
  - This is one extra dependent level and 1 load per cubic lane.
- (b) **The 8 trilinear phi values are reloaded by the cubic** (offsets 0..1 of the 4³ cube, which are also the low/high clamp taps).
  - That is 8 of 65 loads per cubic lane.
- (c) `umReleasedMayChange` reads 6 *constant-per-dispatch* words through `atomicLoad` for every vertex.
  - They are final once `wallReach` finishes (an earlier dispatch).
  - Being atomics, they are neither cached nor CSE-able, and they are float bit patterns, so they also act as an FP-reordering barrier around the value test.
- (d) The surface's `umSampleVelocityFine` is **not given `regularTexture`**: surface.ts:105 passes `undefined`, while momentum passes `"extended"`.
  - Each of the 24 taps therefore runs the general path: transverse clamp, `anchor[axis]<0` branch into `negative[]` (a storage load inlined 24 times), and `select(0, weight*load, weight>0)`, which still issues the load.
  - Same load count, but 24 inlined branches and storage-load paths cost registers and code.
  - `git log -S` shows it was never passed (it has been absent since c9e9fe17), so this looks like an omission, not a decision.
- (e) The flags bit 1 (preserve) and bit 2 (travel) paths, `umSourceuvSourcePhi`, `umRecordTravel` (:342-346, a runtime early return) and the travel branch in `umRebuildBand` (:669-675) are runtime-uniform dead branches. They are compiled into every vertex kernel and so add registers and code.

**Other kernels in the pass**

- `advectWalls` (:443): 66,822 wall-plane vertex lanes. It runs `umAdvected` with the `umWallContact` continuation, which adds an extra `umSampleVelocity` per contact probe.
- `advectDeferred` ×2 (:424-441): each twin scans the whole deferred list and filters `code==3` against `umRegularFine`. That is 2 launches for a list that is tiny at fig-9.

**Rough cost**

- `advectFine`: 22k SIMD × (69 + about 0.6 × 65) ≈ 2.4M issues → **0.6–1.2 ms**.
- Merged, regular-empty, walls and deferred: about 0.15 ms.
- This fits the 3.6 ms phase together with the items below.

### 1.4 `phiResolve` on scratch (#15, pipeline 312): every frame, 1 of 3 resolves (uniform-mixed-phi-resolve.ts:55-85)

1. One 125-lane job per seam tile (fine seams, then 4h seams; header at 7T+16).
   - Lanes 0–7 stage 8 incident topology words, lanes 8–34 stage the 27 4h lattice phi values, then a barrier.
   - Each lane picks the lowest-index *non-unit* incident tile and stores the trilinear of its 8 LDS corners, if the value changed.
2. 125 lanes = 4 SIMD groups with 3 idle lanes.
   - **For an h-seam job, only local-0 face texels can be written.** Any texel with a local component ≠ 0 lies in this unit tile and is skipped.
   - So ≥ 64 of 125 lanes exit at once, and the whole job is a no-op when none of the 7 back neighbours is 4h (the 4h neighbour is only on + sides). That is plausibly about 40% of h-seam jobs.
3. Bindings: `field` r32f (n+1)³ read_write (4.33 MB); topology read.
4. Per lane: about 1 LDS-gated `textureLoad` (current value) + 8 LDS + store. Per job: 35 staged loads.
   - Est. 6.5k jobs × 4 SIMD × about 3 issues ≈ 80k issues + 6.5k barriers → **about 0.05–0.08 ms per resolve**, 3 per frame in this group (+1 at the head) ≈ **0.2 ms/frame**.
5. Copies: none (in place). Each resolve rewrites only hanging texels and skips equal values. The three resolves per frame are each required by the resolved invariant: after advect into scratch, after redistance into phi, and after the surface-volume shift.

### 1.5 Surface `traceCells` (#20, pipelines 181, 165): every frame, 2 dispatches + 1 clear
1. Writes the cell-centre RK2 departure point of every owner to `departures` (rgba32f n³, 16.8 MB; `.w` written 0; the texture is shared with momentum, so it stays).
   - Consumers are transport and momentum.
2. Regular list (about empty at fig-9) + merged.
   - The merged launch runs the general h list, then **seam 4h tiles as one 64-lane job each with only lane 0 active (63 idle)**, then 64 regular coarse owners per job.
3. Per lane: 2 `umSampleVelocity` = 52 issues interior / 102 in the blend zone, 4 levels, 1 store.
   - 22k SIMD × about 55 ≈ 1.2M issues → **0.3–0.6 ms**.
4. Seam 4h jobs: about 2–3k jobs × 64 lanes for 2–3k useful lanes. They are blend-zone samples (≈ 100 loads, deep chain), so one lane holds a SIMD slot for the whole chain.

### 1.6 Surface `redistance` (#20): every frame, 8 dispatches + 1 clear

Order: `retirementEvidence` (157) → `retirementEvidenceCoarse` (159) → `evidenceDistance` ×3 (151/153/155) → `redistanceFine` regular (179) → `redistanceFine` general (169) → `redistance` merged (163).

**Evidence passes (:552-597)**

- Per-tile "every closure vertex > 0" flags, then a 3-pass Chebyshev distance (cap 15).
- `retirementEvidence`: 128 lanes per h tile with 125 live, 2 barriers + atomic. `retirementEvidenceCoarse`: 8 corners per 4h tile.
- **The only consumer is `umNoNearbySurface`** (:601-635). It is called only when a Newton search *misses* and initial > 0 and value < band, plus the dead travel branch.
- `umEvidenceVertex` (:556-561) does `select(umVertexValue(v), umLoadVertex(v), maxWidth==1)`. `select` evaluates **both** arms, so a unit-stencil tile pays topology + phi + phi.
  - Under the resolved invariant every closure vertex is stored or resolved: phiScratch was resolved after advect (frame.ts:488) and the geometry module is also built `resolved=true` (frame.ts:291).
  - So it is just `umLoadVertex`.
- Cost ≈ 11k jobs × 4 SIMD × about 5 issues ≈ 220k issues + 22k barriers + 5 launches → **about 0.1–0.15 ms**.

**`redistanceFine` (:785-818): one 64-lane job per h tile**

- Pass 1 loads each owned vertex and tests `umRebuildBand` (|phi| < 4h). Off-band vertices are copied through.
- If any lane searches, the job **stages a 14³ = 2744-float LDS window (10.98 KB)**:
  - 43 iterations per lane.
  - Each entry is `umResolvedVertex` (:757-768), which is **1 tile-width topology load → 1 phi load** (8 for a 4h cell). That is 2 dependent global levels per entry.
- After a barrier, Newton runs in LDS: `umWindowGradient` = 6 samples × 8 LDS + `phiNext` 8 LDS per iteration, ≤ 8 iterations.
- **The window spans only 4×4×4 = 64 tiles** (tile−1 … tile+2). Yet the tile width is re-loaded 2744 times per job: **2680 redundant dependent topology loads per searching job**.
- 11 KB of LDS per 64-lane group also caps residency: about 32 KB / 11 KB ≈ 2–3 groups per core if the Apple threadgroup-memory budget is 32 KB.
- Cost (searching jobs ≈ 5k): staging 5k × 2 SIMD × 43 × 2 ≈ 860k issues, of which **430k are the topology half**.

**`redistance` merged (:853-890): seam 4h corners (8 tiles × 8 lanes) or 64 regular coarse owners**

- Band candidates are compacted into `umBandList[512]`, then `umRebuildSearch` (global) runs on the compacted lanes.
- Per Newton iteration:
  - `umSurfaceGradient` (:483-491) is 6 × resolved `umSampleVertex` = **6 × (1 topology + 8 phi) = 54 loads**, 2 levels.
  - `phiNext` for a cubic wide vertex is `umOwnerAt` → `umTileMinimumWidth` → `umCubicPhi`: **2 topology levels + another `umOwnerAt` inside `umCubicPhi` + 64 phi**.
- That is ≈ 120 loads and 5–6 levels per iteration, and ≤ 8 iterations (exit at 0.1 tol).
- Lane counts are small (4h band vertices ≈ 3–6k), so this is a latency tail (about 20–40 levels), not an issue-count cost. Est. 0.05–0.1 ms.

### 1.7 `phiResolve` on phi (#15): see 1.4.

### 1.8 Geometry, `{pressure:true}` (#24, pipeline 185): every frame, 1 dispatch (surface-geometry.ts:118-136; frame.ts:502)

1. `umAllOwner` over every owner (about 709k lanes).
   - `gcOwner` reads 8 corners through `umVertexValue`, runs `gcEvaluate`, and stores `targetFill` + `centerPhi` (r32f n³, 4.2 MB each).
   - The pressure variant has **lane 0 of each h tile** evaluate the tile's 4h corners again: 8 loads + `umSurfaceTarget`, a divergent lane-0 tail. It writes `pressureTarget` / `pressureCenterPhi`.
2. Resolved `umVertexValue` (vertex-sampling.wgsl.ts:51-62) per corner is `umTileAt` → `umTileMaximumWidth` (1 topology), plus `umTileMinimumWidth` (a 2nd) for a mixed stencil → phi.
   - Under the resolved invariant every owner corner is stored or resolved, so it equals `umLoadVertex`.
   - That is **8–16 topology loads + 1 dependent level per owner, removable bitwise**.
   - 22k SIMD × about 12 ≈ 264k issues → **about 0.07–0.12 ms**. The same applies at the head (183) and in `geometryChanged` (363).

### 1.9 Sharpening (#25): 3 + 24 launches per frame (uniform-mixed-sharpening.ts)

**Launches**

- `encodeGeometry`: clear (4·(8+T)) → `classify` (residentAll counted) → `compact` → `cacheGeometryPrepare` (192 lanes; prepares budgets, appends active regular owners).
- `encodeSweeps`: 8 × (`propose`, `limit`, `commit`), grid ≤ 1024, 192 lanes, with a barrier after each job (:394-405).

**Bindings**

| Binding | Size | Access |
|---|---|---|
| `work` (`array<atomic<u32>>`, read_write, :112) | workBytes = 4·(8 + 3T + 64T) = **4.39 MB** dedicated (flags T, lists 2T, active 64T) | atomics |
| `scratch` = arena `edgeBytes` (40N, borrowed) | raw proposals 3N floats (12.6 MB), budgets 6 per owner (≤ 25 MB), admission cache 1 word per cell (6 of 32 bits used) | read/write |
| `volume` ping-pong, `centerPhi`, `targetFill` | – | read |

**Per regular lane per sweep kernel**

- `shActiveOwner`: **atomicLoad** (active word) → `umTopology[tile]` (owner index). That is 2 levels.
- Per face: `umFace` → neighbour topology → **`shListed` = atomicLoad(flags)** → budgets/raw from `scratch` (indexed by neighbour index). That is 3–4 levels.
- propose touches 3 faces, limit and commit touch 6 each. So about 1 + 6 = 7 integer atomics per lane per kernel × 24 kernels.
- **The sweeps never write `work`**: the active list and flags are frozen after `cacheGeometryPrepare`.
- Every list and flag read in the 24 sweep kernels is an atomic load for no reason other than the binding type. These are integer words, so bitwise safe.
- Est. active owners ≈ 100k → 3.1k SIMD × 7 × 24 ≈ 520k atomic issues. At the atomic (L2) rate that is about **0.2–0.3 ms of the 2.06 ms**.

**Storage verdict**

- Raw proposals are position-indexed (`umRawAt` is pure arithmetic). Compacting them would add an index chase, so keep the layout.
- They are borrowed arena space, so shrinking them frees nothing unless sharpening sets the arena peak.
- The 64T active list is the required bound.

---

## 2. Storage audit (fig-9)

| Item | Size | Needed? |
|---|---|---|
| phi + phiScratch, r32f (n+1)³ | 2 × 4.33 MB | **Yes.** Advect and redistance read neighbours, so the ping-pong is required. Redistance cannot run in place (the window / Newton read the pre-pass values). |
| Surface claims travel words (n/4+1)³ | 74 KB | **Dead** with coarse travel off. Runtime option; lazy allocation is possible but trivial. |
| Deferred list | ≤ 4.33 MB arena | Borrowed; bound sizing is correct. |
| Evidence (T + solids words) | arena | Borrowed. |
| `umHanging` vertex-cache words | **8.19 MB** | **Dead** (vertexCache disabled). |
| `umHanging` plane-tap words | 3.15 MB | Replaceable by a 131 KB plane array (P2). |
| `umHanging` slot maps 2T | 131 KB | Keep (they drive the hanging launch). |
| `unitVelocity` rgba32f n³ | 16.8 MB | Needed (surface, momentum, forces). .w dead. It duplicates velocityScratch on every unit tile because forces read it everywhere. |
| Coarse cache | 333 KB | .w dead; negligible. |
| Sharpening `work` | 4.39 MB | Needed; 64T active is the bound. |
| Sharpening scratch | 40N arena | Borrowed; position-indexed by design. |
| `departures` .w | – | Written 0; shared format with momentum; keep. |

---

## 3. Proposals

### P1. Bitwise-safe trims in the hanging-taps stage (#19)

**Change**

- In `hanging`, exit before any work for unit tiles (`umTileWidth(tile)==1`, uniform per workgroup).
- Better: launch the hanging pass only over the 4h-seam part of the slot list (offset by the fine count) if slots follow the seam-list order (fine, then 4h). Check ownership.ts:206/277 and the builder.
- Then delete the dead 125 vertex-cache words per slot (record 173 → 48 words) together with surface `vertexCache` code (surface.ts:122-126, 913-920).

**Effect and cost**

- Removes about 3.5k dead 256-lane jobs: about 84k issues + barriers + 224k rgba32f stores → **about 0.03–0.06 ms**.
- Frees **8.19 MB** at fig-9 (**131 MB** at 256³: 125·262144·4).
- Risk: none, since readers return at the unit check first.
- Verify:
  - CPU: capture-uniform-wgsl, naga parse, and confirm `umHangingVertexAddress` no longer appears in #19/#20.
  - Dawn: one fig-9 A/B; per-frame phi digests must be identical.

### P2. Replace plane-tap records with a `unitNegative` plane array, and drop the general tap path from the surface's sampler

**Change**

- Add a buffer of the three negative domain planes: Dy·Dz + Dx·Dz + Dx·Dy = 8192 + 8192 + 16384 floats = **131 KB**. Write it in `unitFaces` (unit tiles: `negative[]`) and in `hanging` lanes 192–239 (4h slots).
- In `umSampleVelocity1`'s `unitTexture` fast path (velocity-sampling.wgsl.ts:151-160), take `base[axis]==-1` too, reading `unitNegative` for `bit[axis]==0` taps.
- `umVelocityTap1`, `umUnitTaps` and the `umVelocitySite`/`umFace` restriction become unreachable from the surface (and momentum) when hanging is on, so they drop out of #20 and #26.
- `umHanging` shrinks to the 2T slot maps.

**Effect and cost**

- Frees **3.15 MB** (+ 8.19 MB from P1 = 11.3 MB at fig-9, **181 MB at 256³**: 173·262144·4).
- Removes the deepest sampler path (4–5 levels) and its code from the largest module. The benefit is registers and occupancy in every vertex and trace kernel, because Metal allocates registers for the worst inlined path.
- ms: not groundable on CPU; plausibly 0–0.2 ms across advect, traceCells and momentum.
- Risk: low. Values are identical, because the same tap is computed by the same code once and then stored.
- Verify: CPU WGSL size and naga parse; check #20's byte count drops. Dawn A/B: digests identical; per-stage ms.

### P3. Give the surface sampler the `regularTexture` fast path

**Change**

- Pass `"velocity"` as `regularTexture` at surface.ts:105, as momentum does at momentum.ts:236.
- The 24 interior taps lose the transverse clamp, the `anchor[axis]<0` → `negative[]` branch and the `select`.
- The regular twin also gets the native 8-load `umSampleVelocity1` (velocity-sampling.wgsl.ts:139-149).

**Effect and cost**

- Same loads, fewer instructions and registers. `umLoadMixedFace`'s storage branch stays only on the `base[axis]<0` path.
- Bitwise identical. The skipped taps are exactly the zero-weight ones (0·x vs select(0,…) is the same +0 unless x is non-finite, and the texture is finite).
- Estimate: **0.02–0.1 ms** over advect, walls, deferred and traceCells. Trivial effort.
- Verify: CPU naga parse; Dawn A/B digests identical.

### P4. One LDS tile-width table for the `redistanceFine` window staging

**Change**

- Before staging, lanes 0–63 load the window's 4×4×4 tile words (`umTopology` width) into `var<workgroup> umWindowWidths: array<u32,64>`; then a barrier.
- `umResolvedVertex` in staging reads the width from LDS.
- The equivalent change to `umFineWide` could share the same table.

**Effect and cost**

- Per searching job: 2744 → 64 topology loads. Staging goes from 2 dependent global levels per entry to 1.
- 5k jobs × 2 SIMD × 43 ≈ **430k issues removed**, the bigger part of the staging chain → **about 0.1–0.25 ms**.
- Bitwise identical; low risk. 256 B of extra LDS.
- Verify: CPU naga parse, then a Dawn A/B on fig-9 + dam64, digests identical.

### P4b. Smaller window (follow-on, needs A/B)

**Change**

- Cut the 14³ window to the reach actually used: about 10³ = 1000 floats, 4 KB.
- Samples outside the window compute `umResolvedVertex` from global memory. That is the same function and the same value, so it is bitwise identical; it is an exact second path, not a fallback.

**Effect and cost**

- Staging work −64%, and LDS drops from 11 KB to 4 KB (occupancy).
- Risk: band vertices with |phi| near 4h step ≈ 4 cells and leave a 10³ window. That costs divergent global samples, so measure a "fraction of out-of-window samples" counter in a CPU replay first.
- Estimate: a further 0.1–0.2 ms if out-of-window samples stay ≲ 5%.

### P5. Resolved-invariant direct loads (geometry, evidence, `umNoNearbySurface`)

**Change**

- `gcOwner` (surface-geometry.ts:121): `umVertexValue` → `umLoadVertex` for owner corners. Every corner is width-aligned in the owner's own (mixed or unit) stencil, so it is stored or resolved.
- `umEvidenceVertex` (surface.ts:556-561): use plain `umLoadVertex`, removing the both-armed `select`.
- `umNoNearbySurface` (:621, :628): only for corners of width-aligned owners.

**Effect and cost**

- Geometry: about 8–16 topology loads + 1 level per owner → **0.07–0.12 ms** (pressure variant) + the same at the head / `geometryChanged`.
- Evidence: 2–3 loads per lane → **about 0.03 ms**.
- Bitwise identical given the invariant. Risk: the head geometry runs after the listed resolve (frame.ts:476 vs 683). Confirm the resolve precedes it on relayout frames.
- Verify: a CPU assertion lane, using the mock or Rust oracle, that `umVertexValue == umLoadVertex` at owner corners after each resolve. Then a Dawn A/B with identical digests.

### P6. De-atomise the sharpening sweeps

**Change**

- Give `propose`/`limit`/`commit` a layout whose `work` binding is `var<storage, read> work: array<u32>`. They never write it.
- `shListed`, `shActiveOwner`, `shSweepJobs` and `shSeamTile` read through a macro (`atomicLoad(&work[i])` in classify/compact/prepare, `work[i]` in sweeps).
- Optionally store the owner index in the active list (2 words per entry, `tile*64+lane` and `index`) to remove the `umTopology[tile]` chase. The 64T bound becomes 128T words (+4.2 MB), or the active list can pack tile|lane in 20 bits and keep index separate.

**Effect and cost**

- About 520k atomic issues become cached plain loads, and the compiler may CSE and hoist them.
- **About 0.15–0.3 ms** of 2.06.
- Bitwise safe: these are integer words, and the floats in `scratch` stay non-atomic.
- Risk: low. It needs a second pipeline layout for the sweep entries; same module, different binding access, so either 2 modules or a generated access mode.
- Verify: CPU naga parse; Dawn A/B on fig-9 with identical volume digests.

### P7. Fuse sample + cubic: reuse the tile width and the 8 trilinear values

**Change**

- In `umAdvectStore` and `umAdvected`, compute `umSampleVertex(q)` with its tile width, origin and 8 values exposed.
- `umCubicPhi` takes those, drops its `umOwnerAt` load, and fills the 8 inner taps from registers.
- In the redistance merged Newton, `umOwnerAt` + `umTileMinimumWidth` for `next` could be 1 stencil word: (2T+2t)/(2T+2t+1) are adjacent, so they form one vec2 load.

**Effect and cost**

- Per cubic lane: −9 loads and −1 dependent level. About 0.6 × 22k SIMD × 9 ≈ 120k issues → **about 0.04–0.08 ms**, plus the shorter chain.
- Bitwise identical (same values, same summation order).
- Verify: Dawn A/B, digests identical.

### P8. Hoist the wall-reach words

**Change**

- In the claimed advect entries, load `umClaims[UM_WALL_REACH+0..5]` once per workgroup into workgroup memory, or once per lane before the job loop.
- They are final after `wallReach`, which is an earlier dispatch in the same pass with a dispatch boundary in between.

**Effect and cost**

- −6 atomic issues per vertex lane → 22k SIMD × 6 ≈ 132k atomic issues → **about 0.03–0.06 ms**.
- Removes an FP barrier from the value test.
- Bitwise identical (the same bits are read).
- Verify: Dawn A/B, digests identical.

### P9. `traceCells`: pack seam 4h owners 64 per job, and/or fuse into the advect jobs

**Change**

- The merged `traceCells` currently gives each seam 4h tile a 64-lane job with one live lane. Pack those owners 64 per job, as regular coarse owners already are.
- Separately, the `advectFine` and `advectOwners` jobs could also write the cell departure (same owner decode, same tile-class loads).
  - The sample points differ (vertex vs centre), so the sampling itself does not merge.
  - The saving is the launch, the owner decode and the shared classification loads. A clear may then be needed.

**Effect and cost**

- Packing: about 2.5k jobs → about 40 → **about 0.02–0.05 ms**.
- Fusion: −2 launches (regular + merged traceCells) + decode → **about 0.03–0.05 ms**.
- Bitwise identical.
- Verify: Dawn A/B.

### P10. Resolve: skip no-op h-seam jobs, or resolve from 4h jobs only

**Change**

- An h-seam job can only write its local-0 face texels and only from a *back* non-unit neighbour.
- (a) Test the 7 back stencil bits from the tile's stencil word before staging, and broadcast with `workgroupUniformLoad`. Jobs with no back 4h neighbour then exit before staging and the barrier.
- (b) Stronger: let each 4h seam job also resolve its own upper (local-4) faces where the + neighbour is unit, applying the same "lowest non-unit index" tie-break from its 27 neighbour words. Then drop h-seam jobs from the resolve list entirely.

**Effect and cost**

- (a) about 25% of resolve work; (b) about 50–60%.
- 3 resolves/frame × about 0.06 ms × 0.25–0.55 → **0.05–0.1 ms**.
- (a) is bitwise identical. (b) is identical if the tie-break is reproduced exactly; CPU-check it on a mock layout.
- Verify: a CPU script enumerating the texels written per job on captured layouts, then a Dawn A/B with identical digests.

### P11. Specialise dead runtime branches with overrides (preserve, travel, source)

**Change**

- Make `flags.x` bits 1 and 2, `sourceParams` and drain/cubic override constants on #20 pipelines.
- Twins are only needed if a scene toggles them live. These are scene parameters, so a pipeline rebuild on change is acceptable. Fail fast if a flag mismatches the compiled pipeline.

**Effect and cost**

- Removes `umPreserved` (6 loads), the travel atomics paths (`umRecordTravel`, `umRebuildBand` travel, merged-redistance travel store) and their registers from every vertex kernel. Allows deleting the 74 KB travel tail when off.
- Est. 0–0.1 ms (registers only). Bitwise identical.
- Verify: CPU WGSL diff; Dawn A/B.

### P12. MAC-union texel sharing in `umSampleVelocityFine` ×3

**Change**

- The three components' 2×2×2 stencils overlap: per axis, `floor(p-1)` and `floor(p-0.5)` are equal or adjacent.
- The union is 8 / 12 / 16 / 20 texels for 0–3 differing axes, a **mean of 14 vs the 24 issued now**. One rgba32f fetch serves all components that need that texel; the per-component sums keep their order, so the result is bitwise identical.

**Effect and cost**

- SIMD divergence: lanes differ in which axes shift. A SIMD issues the union of paths, so expect about 20 issues, not 14.
- −4 to −10 issues per sample × 2 samples × (advect + traceCells) ≈ 22k × 2 × 2 × 7 ≈ 600k issues → **0.1–0.25 ms** in theory.
- Risk: medium. Index and weight bookkeeping raises register use, which might eat the gain.
- Verify: Dawn A/B, digests identical. Prototype in traceCells first; it is the simplest call site.

### P13 (semantic, high risk). Narrow-band advect for far h vertices

**Change**

- Skip the trace and sampling for h vertices with |phi| > band + 2 cells (cubic) + max travel. Store sign·max(|phi| − dt·|u|ₘₐₓ, band + margin).

**Effect and cost**

- Likely 40–60% of h vertices; −60 to −70 issues each → **0.3–0.6 ms**.
- Risk: high. It changes far values the census, coarsening and far-air certificates read. Memory notes (4h blobbiness, stale wide vertices) show far-value kinks matter.
- Only worth it after the bitwise items. It needs a quality gate, not digests.

### Not recommended / already covered

- Hardware-filtered sampling of rgba32f (8-bit weight quantisation, beyond ulp).
- A phi4 mirror (rejected) and f16 phi (rejected).
- Re-fusing propose/limit/commit: needs a grid barrier.
- Lazy evidence: the consumer needs global evidence before any search; its 0.1–0.15 ms is mostly fixed by P5's direct load. A further cut would need evidence only for 4h vertices, because h misses could scan the LDS window, but that is not bitwise at clipped coarse cells.
- The analytic cubic gradient for 4h Newton: a semantic change for a small (about 0.05 ms) latency tail. Low value.

---

## 4. RANKED experiment list (gain / effort)

| # | Experiment | Est. gain at fig-9 | Effort | Exact? |
|---|---|---|---|---|
| 1 | **P4** LDS tile-width table for `redistanceFine` window staging (surface.ts:757-768, 800-804) | 0.1–0.25 ms | S | bitwise |
| 2 | **P6** de-atomise sharpening sweeps (read-only `work` layout) + owner index in the active list | 0.15–0.3 ms | M | bitwise |
| 3 | **P5** resolved-invariant direct loads: `gcOwner`, `umEvidenceVertex` (drop both-armed `select`), `umNoNearbySurface` | 0.1–0.15 ms | S | bitwise (assert invariant) |
| 4 | **P1 + P2** skip dead unit-slot hanging jobs; `unitNegative` plane array; delete vertex-cache and plane records; drop the general tap path from the surface sampler | 0.03–0.25 ms + **11.3 MB** (181 MB at 256³) | M | bitwise |
| 5 | **P3** `regularTexture="velocity"` for the surface sampler (surface.ts:105) | 0.02–0.1 ms | XS | bitwise |
| 6 | **P7** fuse `umSampleVertex` + `umCubicPhi` (reuse width + 8 taps); vec2 stencil load in Newton | 0.04–0.08 ms | S | bitwise |
| 7 | **P12** MAC-union texel sharing in the fine sampler (prototype in traceCells) | 0.1–0.25 ms (theory) | M | bitwise |
| 8 | **P10** resolve: skip no-op h-seam jobs (a), or 4h-only jobs (b) | 0.05–0.1 ms | S / M | bitwise |
| 9 | **P8 + P9 + P11** hoist wall-reach words; pack seam-4h trace jobs / fuse traceCells; override-specialise preserve, travel and source | 0.05–0.15 ms combined | S each | bitwise |
| 10 | **P4b** shrink the redistance window to about 10³ (LDS 11 KB → 4 KB) | 0.1–0.2 ms | M | bitwise; perf risk |
| (11) | **P13** narrow-band far-vertex advect | 0.3–0.6 ms | L | semantic; high risk |

- The bitwise items 1–10 sum to roughly 0.7–1.6 ms of the about 7.4 ms these stages take (phi 3.65 + sharpening 2.06 + geometry share + caches).
- Suggested batching, at most 3 probe runs:
  - Batch A = 1 + 3 + 5 + 6 (all small, all in #20/#24).
  - Batch B = 2 (sharpening).
  - Batch C = 4 (hanging and storage).

## 5. Storage that can be deleted (fig-9 bytes; 256³ in brackets)

| Storage | Bytes | Note |
|---|---|---|
| `umHanging` vertex-cache words, 125 × T × 4 | **8,192,000** [131 MB] | dead: vertexCache is off with resolved=true |
| `umHanging` plane-tap records, 48 × T × 4 | **3,145,728** [50 MB] | replaced by a 131,072 B `unitNegative` array (P2) |
| Surface `umClaims` travel tail, (n/4+1)³ × 4 | **74,052** | dead while coarse travel is off (allocate on enable) |
| `unitVelocity` .w, coarse cache .w | 4 of 16 B per texel | dead channels; no cheaper WebGPU format, so keep |
| phi/phiScratch, sharpening `work`, deferred, evidence, sharpening scratch | – | needed at current size, or arena-borrowed |
