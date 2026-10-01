# Pressure group: two-stage pressure (all-4h root, then the h surface band)

This is a read-only analysis. No GPU work was run. The evidence comes from four sources:
- **Baseline timings**: `$SCRATCH/runs/base/fig9.json`, frames 5-130, GPU mean 22.42 ms, plus `dam64.json`.
- **Mock-device dispatch log**: my capture variant `$SCRATCH/pressure-work/capture.mts`, written to `pressure-work/fig9/{log.json,dlog.txt,pipelines.json,used.json}`.
- **WGSL reachability script**: `pressure-work/reach.py`.
- **Code reading.**

Scope of the paths below:
- Paths are under `lib/methods/uniform/` unless they are shown in full.
- "Module #N" means the capture's module number.

## 0. What pressure costs at fig-9 (measured)

| Stage, as the trace phases cut it (`uniform-mixed-frame.ts:509-556`) | fig-9 ms | dam64 ms |
|---|---|---|
| `pressureSetup`, "CM11a topology + RHS pyramid": surface census, sim authority #28, band prepare #36-39, transfer to pressure #32, split authority #33, warm copy, rhs #29, root setup #30 + 14 native setup dispatches #6, initial measure + checkpoint | 1.068 | 0.812 |
| `pressureVCycles`: 3 encoded slots, 1 executed | 1.513 | 1.213 |
| `pressureProjection`: gate + project #58 | 0.208 | 0.168 |
| `V.band`: transfer to simulation, band solve and projection (#40-50), presentation copies | 2.328 | 1.458 |
| **Total** | **5.12** (23% of 22.4) | 3.65 |

Per-slot means at fig-9:
- Executed slot: 0.900 ms. Closed slots: 0.308 and 0.304 ms.
- Every fig-9 frame encodes 3 slots and executes 1. The schedule floor is `UNIFORM_MIXED_SCHEDULE_FLOOR=3` (`uniform-mixed-pressure-schedule.ts:19`).
- **The two closed slots cost 0.61 ms per frame.**
- Each slot is 74 dispatches: gate 1, cycle 69, measure 1, checkpoint 3.
- A closed dispatch costs about **4.1 µs** (0.306 ms / 74). An open one averages about **12.2 µs** (0.900 / 74). Small coarse-level kernels run about 6 µs (the comment at `FUSED_VISIT_MAX_CELLS` cites twelve ~6 µs launches).
- **Rule of thumb:** removing one dispatch from the slot body saves about 6 µs (open) + 2 × 4.1 µs (closed) ≈ **14 µs per frame**.

Band at fig-9:
- About 100 dispatches: transfer 2, init 1, 4 cycles × 23, measure 1, copy/project/present 3.
- That averages about 23 µs per dispatch.
- The band always runs 4 cycles, because the residual target `bandParams.solve.x` is 0 (`uniform-mixed-frame.ts:402`). This is deliberate: the 1/h² target was the under-solve bug noted in memory.
- Band size in tiles: mean ≈ 2188, minimum 1240, maximum 3582 (from the earlier run log).

Steady-state pressure dispatches per fig-9 frame come to about **361**:

| Item | Dispatches |
|---|---|
| Surface census | 2 |
| Authority #28 | 4 |
| Band prepare | 4 |
| Transfer to pressure | 2 |
| Split authority | 6 |
| rhs | 1 |
| Root setup + native setup | 15 |
| Initial measure + checkpoint | 4 |
| Slots, 3 × 74 | 222 |
| Projection gate + project | 2 |
| Transfer to simulation | 2 |
| Band | 97 |

About 148 of these are closed-slot early exits.

Lattice sizes at fig-9 (used in the byte counts below):
- Lattice 128×128×64 = 1,048,576 cells. Tiles T = 32×32×16 = 16,384. Pages = 256.
- An rgba32f lattice texture is 16,777,216 B. An r32f lattice texture is 4,194,304 B.
- The negative wall buffer is 32,768 floats = 131,072 B.
- Pressure ownership storage `count` = 16,384 owners + 2·(32·32 + 32·16 + 32·16) = 20,480 words = **81,920 B**.
- Native levels with halo:

  | Level | Size | Cells |
  |---|---|---|
  | L2 | 34×34×18 | 20,808 |
  | L3 | 18×18×10 | 3,240 |
  | L4 | 10×10×6 | 600 |
  | L5 | 6×6×4 | 144 (32 interior unknowns) |

---

## 1. Kernels in frame order

### 1.1 Surface census: `uniform-pressure-surface-band.ts`, `classify` + `classifyCoarse`
1. **What it does:** builds the per-tile surface-crossing bitmask (tiles/32 words = 2,048 B). It runs every frame: a clearBuffer of 2,048 B, `classify` [4096], `classifyCoarse` [256], and a 2,048 B copy into the stage grids.
2. **Shape:**
   - `classify`: one 64-lane workgroup per h tile, striding over 125 closure vertices (2 loads per lane, 3 lanes idle on the second pass), then a 6-step barrier OR-reduce. The grid is the fixed counted 4096, and groups past `umCounts.x` exit.
   - `classifyCoarse`: 8 corner loads per 4h tile, one lane per tile.
3. **Bindings:** vertex phi r32f (129×129×65 × 4 B ≈ 4.33 MB, read) and the band bits (2 KB, atomic or).
4. **Access:** independent texture loads, no chains, and one atomicOr per surface tile.
5. **Dead:** **nothing in the solve reads this mask.** Its only reader is the grid overlay: `umStageBand` (`uniform-stage-grids.ts:145-147`) is called only at `lib/core/webgpu-grid-overlay.ts:1698`. The band builds its own list in `bandList`.
6. **Proposal P8:** encode the census only when the overlay layer that reads it is live, behind the same switch as `layoutViews` (`uniform-mixed-frame.ts:409-410`).
   - Saves 2 dispatches + clear + copy.
   - Estimate: classify over about 5-11k h tiles × 125 vertices with barriers ≈ 20-40 µs, plus classifyCoarse about 5 µs → **0.03-0.05 ms**.
   - Risk: the overlay layer goes silently empty when off. Make it fail loud instead (an explicit header flag in the stage-grid record).
   - Verify: CPU preflight, one fig-9 run, and check the overlay in the app.

### 1.2 Simulation authority #28 (`uniform-mixed-pressure-authority.ts`, sim ownership)
- Sequence: build [4096] → chunks [16] → reduce [1] → resolve [4096]. It is listed for context only; the volume-correction group owns it.
- It computes phi, phase and correction with the excess/deficit balance in simulation ownership.

### 1.3 Band prepare #36-#39 (`uniform-pressure-band.ts:873-878`)
1. **Sequence:**
   - `clearBuffer` of the index header (104 B) and the slot map (65,536 B).
   - `list` [4096×64]: classifies h tiles with a liquid row, compacting with atomics into the index list.
   - `prep` [4096 grid-stride over band slots ×64]: assembles the rows.
   - `middleBake` [1024×64]: 2h Galerkin aggregates.
   - `coarseBake` [256]: 4h aggregate rows, colour-ordered, plus neighbour-slot baking.
2. **Shape:** fixed grids striding the GPU band count. At ~2188 band tiles, `prep` keeps 2188 of 4096 groups busy.
3. **Bindings at fig-9:**

   | Buffer | Bytes | Access | Notes |
   |---|---|---|---|
   | rows | 4·21·16384·64 = **88,080,384 B (84 MiB)** | write | 21 fields, because solids are always compiled (`:141`, `:147`) |
   | coarse | 4·(8·9+23)·16384 = **6,225,920 B** | write | |
   | solve | (1,048,576 + 65,536)·4 = **4,456,448 B** | | |
   | index | (26 + 16384 + 16384)·4 = **131,176 B** | write | |
   | forced velocity | 16.8 MB | read | velocityScratch |
   | phi | 4.19 MB | read | |
   | correction | 4.19 MB | read | r32f |
   | vertex phi | 4.33 MB | read | |
   | solid record | 16·(20480+16384) = 589,824 B | read | |

4. **Access (prep, per row):**
   - 6 faces: kind classification (neighbour phi and the tile map), coefficient, forced u*, and in the S twin the V value.
   - Stores: `rows[(3+f)N]` coefficient, `rows[(9+f)N]` forced, and **`rows[(15+f)N]=volume`, which the solid-free twin still stores as 1.0** (`:439`). That is 6 dead stores per row: 6 × 4 B × 140k rows = 3.4 MB of writes per frame.
5. **Dead:**
   - The 6 V fields: 25,165,824 B of the 84 MiB.
   - Capacity is sized for 16,384 band tiles (`this.capacity=tiles`, `:139`, "simulation reserves every tile at h"). The observed maximum is 3582 tiles, 22%.
6. See P9 (storage) and the solid-free-twin item in P10.

### 1.4 Transfer to pressure #32 (`uniform-mixed-remap.ts:197-257`)
- Sequence:
  1. A 2,064 B residency-word copy.
  2. `toPressureCoarse`: a counted grid of resident pages (256), one lane per 4h tile.
  3. `toPressureFine`: `TRANSFER_GRID=4096`, one workgroup job per h tile, computing the mean of the h patches per 4h face and the h-cell volume mean.
- This is the first **representation hop**: simulation h/4h → all-4h pressure ownership.
- The writes (`velocity` 16.8 MB texture, `negative`, `volumeScratch`) are only the texels the pressure side reads. It was already slimmed ("transfer 3.55 → 1.09 ms f7").
- **Minor dead CPU work:** `toPressure` and `toSimulation` bind groups are built from identical arguments (`uniform-mixed-frame.ts:330-331`), so one group would serve both.

### 1.5 Split authority #33 (pressure ownership, resident mode)
1. **Sequence:** `cut` [256] → `absent` [256] → `build` [4096] → `chunks` [1] → `reduce` [1] → `resolve` [4096] (`uniform-mixed-pressure-authority.ts:283-293`).
2. **Shape:**
   - `cut`: grid-strides all 16,384 tiles, one lane per tile test. A cut tile gets a 64-lane vote. There are no cut tiles at fig-9, so every lane does one `umSolidCut` stub test.
   - `absent`: one group per page, early exit when the page is resident.
   - `build` and `resolve` are `residentAll` counted jobs. On the all-4h ownership, `umCounts.x = 0`, so jobs ≤ resident pages ≤ 256. The grid is `COUNTED_GRID=4096` (`uniform-mixed-ownership.ts:71,393-395`), so **at least 3840 of 4096 groups exit after a workgroupUniformLoad.**
   - `build` does a 64-lane vec4 barrier reduce per workgroup.
3. **Bindings:**

   | Binding | Bytes | Access |
   |---|---|---|
   | centerPhi, volume, targetFill, phase, correction | 4.19 MB each | read or write |
   | phi | 65,536 B, root conditioning prefix | write |
   | scratch | 8·(1 + 2·(256+1)) ≈ 4.1 KB | |
   | params | 16 B | read |

4. **Access:** one origin texel per owner plus up to 6 neighbour owner lookups when closed (solids only).
5. **Duplicated work:** this is a second full authority with its own excess/deficit balance reduction, done in 4h ownership after #28 did it in h ownership. It is needed because the RHS and projection read the 4h correction, and the 4h correction is not linear in the h ones.
6. **Proposal P11 (grid sizing):** size counted launches on the all-4h pressure ownership to their job bound, i.e. resident pages (256 here), not `min(4096, tiles)`.
   - This follows memory's "size fixed direct grids to the bound": here the cap is too large, not too small.
   - It applies to authority build/resolve, `buildRhs`, `project` and the acceptance `reduce` (×4 per frame).
   - About 8 open launches per frame plus 2 closed `reduce`s. At about 2-4 µs of idle-group cost per 3840 empty groups → **0.02-0.04 ms**.
   - Risk: low. The pressure ownership's fine capacity is 0 by construction (constructor throws otherwise), so the bound is layout-general.
   - Verify: preflight census plus one fig-9 run, which should be bit-identical.

### 1.6 Warm start
- One copy of 81,920 B per frame, presentation pressure → root pressure, plus its mirror after the solve (81,920 B + 65,536 B phi).
- Each costs about 2-3 µs. Ping-ponging the presentation buffer would remove both copies, but the gain is small (P14).

### 1.7 RHS #29 (`uniform-mixed-pressure-velocity.ts:184-209`, coarse mode)
1. **Run:** every frame, `residentAll` counted [4096] with at most 256 jobs. One lane per 4h owner.
2. **Per lane (liquid owner):**
   - phi.
   - 6 faces × (one face texel + `umCoarseCut` → `umSolidCut` stub).
   - The correction texel.
   - Writes rhs, minimum and pressure.
   - The boundary loop over 6 halo slots.
   - Air owners skip face loads.
3. **Bindings:**

   | Binding | Bytes | Access |
   |---|---|---|
   | velocity | 16.8 MB | read |
   | negative | 131 KB | read |
   | phi | 65,536 B | read |
   | correction | 4.19 MB | read |
   | rhs, minimum, pressure | 81,920 B each, arena | write |
   | fine velocity + negative | 16.8 MB + 131 KB | read, only on cut faces |

4. **Notes:**
   - `pressures[o]` is a read-modify-write (warm start clamp).
   - The 3840 idle groups are covered by P11.

### 1.8 Root setup #30 + native setup #6 (every frame)
1. **Sequence:**
   - `setup`: dense [9,9,5] groups of 4³ over the L2 lattice with halo. It writes native MG_PHI and MG_V (4 floats) per native cell plus far-field seeds for absent pages. This is the second **representation hop**: owner-indexed root → native L2 texels (arena-backed).
   - Native setup, 14 dispatches: `mgDownsampleTopology` ×3 (L2→L5), then per level `mgExtrapolatePhiOneCell`, `mgBakeCoefficients`, `mgBuildSmoothTiles` (L2, L3), and `mgBuildCycleTiles` (L2).
2. **Topology downsampling:** this is static-solid data that is rebuilt every frame **because the arena aliases it** (the extension's FIM fields overwrite the same words). Moving it out of the arena would cost about 0.06 MB of persistent storage (L3-L5 topology vec4 = 16·(3240+600+144) = 63.7 KB) and remove 3 dispatches (about 15 µs) (P13).
3. **Coefficient bake:** `mgBakeCoefficients` writes a vec4 per cell (16 B), with `.w` holding the mask and 3 coefficients. That is **24 B of stencil storage per native row**: 16 coefficient + 4 rhs + 4 min.

### 1.9 Initial measure + checkpoint #30/#31
- Sequence: `measure` [405 tiles ×64] writes `residual` (81,920 B), then `resetInitial` [1], `reduce` [4096 counted, ≤256 jobs; 64-lane max + atomicMax], `initial` [1].
- The verdict is a 1-lane kernel.

### 1.10 Slot loop (×3 encoded; fig-9 executes 1)

There are 74 dispatches per slot. Each one early-exits on `umSlotClosed()` or `mgSkipCycle()`, which reads `mgState.convergence[17]` or the support word `9n+24`.

| # | Kernel | Grid | Per lane |
|---|---|---|---|
| 1 | schedule gate #34 | [1] | 1 lane; reads the lagged plan and the last verdict; writes native word 17 + support word |
| 2 | `restrictRoot` #30 | [405 tiles ×64] | **`umRow`**: own p/phi/rhs, 6 neighbour phi + 6 neighbour p (owner lookups through `umTopology`), boundary halo values, and `umTheta` recomputed per liquid/air face (two abs, max, clamp, divide). No baked coefficients: 0 B coefficient storage, but about 16 dependent-ish loads and the theta math every cycle |
| 3-14 | `mgSmoothTilesJacobi` L2 ×12 (6 pre-sweeps, Jacobi, 2 dispatches per sweep) | [405] | tile list `atomicLoad` (chain depth 2) → p, min, coef vec4 (16 B), 3 × −neighbour coef vec4 (48 B fetched, 12 used), 6 p, rhs: **about 17 loads, 92 B fetched / 68 B used per row per sweep**, 4 B write |
| 15 | `mgResidualTiles` L2 | [405] | same stencil, writes residual |
| 16-18 | `mgRestrictResidual`, `mgClearPressure`, `mgDownsampleSubtract` (L2→L3) | [5,5,3] dense 4³ | 8 children each; three passes over the same coarse cells |
| 19-20 | `mgSmoothColour` L3 ×2 | [5,5,3] | first sweep dense red/black (L3 has no cycle list) |
| 21-30 | `mgSmoothTilesJacobi` L3 ×10 | [75] | as for L2 |
| 31-34 | `mgResidual`, `mgRestrictResidual`, `mgClearPressure`, `mgDownsampleSubtract` (L3→L4) | [5,5,3]/[3,3,2] | |
| 35 | `mgSmoothVisitLocalInPlace` L4 | [1] | 1 workgroup, fused visit (600 cells < `FUSED_VISIT_MAX_CELLS` = 1000) |
| 36-39 | residual/restrict/clear/downsampleSubtract (L4→L5) | [3,3,2]/[2,2,1] | |
| 40 | `mgSolveCoarsest` L5 | [1×256] | see below |
| 41-42 | `mgProlongateAdd` L4, L4 visit | | |
| 43-55 | `mgProlongateAdd` L3, 12 L3 post-sweeps | | |
| 56-68 | `mgProlongateAddTiles` L2, 12 L2 post-sweeps | | |
| 69 | **`mgCopyPressureTiles`** | [405] | copies native parity B → A because the post-smoothing count leaves the iterate in the odd buffer |
| 70 | `prolongRoot` #30 | [405] | `p_root += native[MG_P…]`: third **representation hop** per cycle |
| 71 | `measure` #30 | [405] | `umRow` again, with theta recomputed again |
| 72-74 | `resetCycle` [1], `reduce` [4096], `cycle` [1] (#31) | | |

**`mgSolveCoarsest`** (`uniform-coarse-solver.wgsl.ts:84-190`):
- One 256-lane workgroup over 144 rows (112 lanes idle) in **storage** rows of 48 B (`UniformCoarseRow`), with double-single arithmetic.
- Each iteration has 6 storage-barrier phases: red, black, clamp, reset, residual + 7 atomics, worst lane.
- The convergence check runs every iteration. The cap is `UNIFORM_CM11A_COARSE_SWEEP_CAP=4096` (`pressure-policy.ts`).
- The iteration count is not known from code. It is accumulated in `convergence[26]` (iterations) and `[27]` (calls). At an unverified ~1-2 µs per iteration, 30 iterations ≈ 30-60 µs per call.

**Representation hops per executed cycle:**
- `restrictRoot` (root → L2 rhs), `prolongRoot` (L2 → root), and `mgCopyPressureTiles` (L2 parity). That is 3 [405]-tile launches whose only purpose is crossing between two identical lattices: root owners = tiles = the L2 interior.
- The root exists because the native continuation alone is not consistent with the mixed operator. Memory LEAD: native-only blows up at long dam frame 3, root cause not found. The mixed root's residual corrects that each cycle.

### 1.11 Projection gate + project #58 (`uniform-mixed-pressure-velocity.ts:237-295`)
1. **Shape:** the gate is [1]. `project` is `residentAll` counted [4096] with ≤256 jobs, using the face dispatch helper (`uniformMixedFaceDispatchWGSL`), one lane per 4h owner.
2. **Per owner face:**
   - phi of owner and neighbour, the face velocity texel, the neighbour pressure, theta recompute, and the release tests.
   - Reconstruction slopes are 0 (all-4h root), so `umPressureFaceCorrection` returns early when widths are equal.
3. **Bindings:**

   | Binding | Bytes | Access |
   |---|---|---|
   | velocity | 16.8 MB | read |
   | output velocity | 16.8 MB | write |
   | pressure | 81,920 B | read |
   | centerPhi | 4.19 MB | bound but **unread** (the comment at `:32` says so) |
   | volume | 4.19 MB | read |
   | sourceParams | 176 B | read |

4. **Dead:** the `centerPhi` binding.

### 1.12 Transfer to simulation #32
- `toSimulationCoarse` [256] and `toSimulationFine` [4096]. This is the fourth hop: h faces interpolate the 4h faces.
- The h-tile faces in band tiles are then overwritten by the band projection, except Neumann faces on band boundaries, which `bandProject` reads through `bandCopy`. Skipping them is not worth the bookkeeping.

### 1.13 Band solve #40-#47 (`uniform-pressure-band.ts:882-893`)
1. **Sequence:**
   - `init` [4096 grid-stride over band slots]: trilinear 4h pressure start.
   - Per cycle, 23 dispatches:
     - `sweep` ×6 (3 red/black pre-sweeps).
     - `restrict@k`.
     - `middleSweep` ×3.
     - `middleRestrict`.
     - `coarseSolve`.
     - `middleSweepP`, `middleSweep` ×3.
     - `middleProlong`.
     - `sweep` ×6.
   - `measure@4` once at the end.
2. **`sweep`** (`:532-548`):
   - 32-lane workgroup per band tile per colour; grid 2048 strides the ~2188 slots.
   - Per lane:
     - `bTile(s)` index load.
     - kinds, diag, rhs (rows; field-major, coalesced).
     - For 6 faces: unit band faces skip the coefficient load (flag bits 18-23). A cross-tile neighbour costs `bNearSlot` (coarse buffer) then `solve`, chain depth 2. A 4³ tile has on average 1.5 crossing faces per cell. Wall faces load coefficient + halo.
     - Write `solve`, plus the halo follow.
   - Totals: **about 10.5 loads per row per half sweep (≈ 42 B)**, dependent depth about 3 (index → kinds → near slot → solve).
   - **Stencil storage per row: 60 B used (15 fields) of 84 B allocated (21 fields)**, plus 4 B solve and 0.375 B of neighbour slots.
   - Unique bytes per sweep: about 70k rows × 16 B + solve ≈ 1.1 MB. Over 48 sweeps that is about 54 MB per frame, which is not bandwidth-relevant (memory: "on Apple, compacting buys nothing").
3. **`restrict@k`:** a lane per h cell; residual into 2h aggregates; a `worst` atomicMax into the history word.
4. **Middle level:** 64-lane workgroups, 8 slots per group, 8 aggregates each. Each aggregate does 6 near loads through `bMiddleNear` (workgroup slot cache) and 6 coefficient loads.
5. **`coarseSolve`** (`:643-696`):
   - **One 1024-lane workgroup**, 16 sweeps × 2 colours = 32 phases.
   - The register path is used only when `n ≤ SOLVE_SLOTS = COARSE_SOLVE_SLOTS = 1024` (`:48`, `:190`).
   - **At fig-9, n ≈ 2188 (max 3582), so it always takes the streaming path.** Each phase loads 15 global words per row: 6 near, 6 weights, residual, diag, slot.
   - Each lane handles about 1.07 rows per phase (2188/2/1024). Corrections live in workgroup memory (`SHARED = 8160` ≥ n, so there is no storage fallback).
   - Each phase is therefore one global-latency round plus a barrier, about 1-1.5 µs (estimate): 32 phases ≈ 35-50 µs per call, **× 4 ≈ 0.14-0.2 ms per frame**.
   - The register path would take about 32 × ~0.2 µs + one load ≈ 8-10 µs per call.
6. **`measure@4`:** final residual into the index header.
7. **Dead:**
   - `measure@1..@3` are compiled (3 variants × 2 twins = 6 pipelines) and never encoded.
   - All `bConverged` / `bConvergedBefore` / `bLive` plumbing is evaluated (lane 0 or per lane, about 4-5 index loads) but can never be true while `solve.x = 0`.

### 1.14 Band projection #48-#50
1. **Sequence:**
   - `copy` [4096 ×128]: copies band texels plus the layer below into `velocityScratch`.
   - `project` [×128]: reads forced u* from rows, the solve and the copied field; writes the velocity texture and negative walls.
   - `present` [×64]: copies the band pressures into the stage grids.
2. **Why forced is stored per row:** forced u* lives in rows, and every interior face is stored twice, because `copy` overwrites `velocityScratch`, the u* source (`uniform-mixed-frame.ts:338`, `copy:f.velocityScratch`).
   - A distinct copy target would let `project` read u* directly and delete 6 fields (25.2 MB).
   - It needs another rgba32f lattice texture (16.8 MB) or an arena slice, so the net gain is about 8 MB. Low priority.
3. **Dead:** the solid-free twin's `project` still divides by `bVolume` (6 loads of the constant 1.0 per row). This is P10.

### 1.15 Presentation copies after the band
- Root pressure 81,920 B, root phi 65,536 B, band slot map 65,536 B, stage grid 65,536 B, plus receipts.
- That is about 5 copies at about 2-3 µs each.

---

## 2. Cross-cutting answers to the brief

**Why each band module (#36-#50, also #29-#33 and #58) is compiled twice and used once:**
- Every stage builds its pipelines through `uniformMixedSolidPipeline` → `UniformMixedSolid.compile`. That creates `create({})` and `create({umSolidsPresent:0})` (`uniform-mixed-solid.wgsl.ts:111-114`), so that "the first voxel edit switches variant without a compile".
- `select()` picks one per advance (`present`).
- fig-9 has no solids, so only the free twin runs.
- The band has 24 variants (sweep 2, restrict 4, middleSweep 3, measure 4, others 1 each) × 2 = 48 pipelines, of which 24 are ever used, and 18 at fig-9 because `measure@1-3` are dead.
- The cost is startup compile only, not frame time.
- Cheap fix (P15): build the unused twin after the first frame, in the background, and `await` it on the first `present` flip.

**How much of module #6 (333 KB raw, 21 pipelines) is reachable:**
- `reach.py` over the 21 compiled entries reaches **43.6 KB of 225 KB comment-free (19.3%), 82 of 513 functions**.
- The rest is the retired uniform reference path: sharpening, MacCormack, project, coupleRigid, active-region finalize, `mgBuildFinest*`, and others.
- In addition, `initialize` builds the full native plan (`this.plan = Object.freeze(this.buildPlan())`), which the mixed path never encodes.
- It also allocates smooth and cycle tile lists for L0 (18,513 tiles) and L1 (2,601), which are unused. That is about 21k × (16 + 8) B ≈ 0.5 MB.
- These are startup and memory costs, not frame time (P15).

**The gated schedule's encoded-but-skipped slots:** 2 slots × 74 dispatches × 4.1 µs = **0.61 ms per frame** (measured). The floor of 3 is justified by a dam128 impact. Every slot-body cut (P1-P4) is paid back three times.

**Measure and acceptance reductions:** each checkpoint is 4 dispatches: `measure` [405], `reset` [1], `reduce` [4096 grid, ≤256 jobs], `verdict` [1]. With 4 checkpoints that is 16 dispatches per frame, 9 of them in closed slots. See P4.

**Coarse solves on a single workgroup:**
- Band `coarseSolve` is 1 × 1024 lanes, streaming because n > 1024 (P2).
- Native `mgSolveCoarsest` is 1 × 256 lanes over 144 storage rows with 6 storage barriers per iteration and a check every iteration (P5).

**Per-sweep loads and stencil bytes per row:**

| Kernel | Loads per row per sweep | Stencil storage per row |
|---|---|---|
| Native L2 Jacobi | ~17 loads, 92 B fetched / 68 B used | 24 B (coef vec4 16 + rhs 4 + min 4) |
| Band h sweep | ~10.5 loads, ~42 B | 60 B used of 84 B allocated + 4 B solve |
| Band coarse (streaming) | 15 loads per phase | 23 fields × 4 B = 92 B |
| Mixed root `umRow` | ~16 loads | 0 B stored coefficients (theta recomputed on every restrict and measure) |

---

## 3. RANKED experiment list (gain / effort)

ms estimates are per fig-9 frame at the 1-open + 2-closed slot shape. "Bit-identical" means the same arithmetic order.

| # | Experiment | Expected ms | Effort | Risk / verification |
|---|---|---|---|---|
| **P2** | **Band `coarseSolve` register path up to 4096 rows.** Raise `COARSE_SOLVE_SLOTS` 1024 → 4096 (HELD = 4) and pack `near` as u16 pairs (n ≤ SHARED = 8160 < 65536) to keep about 48 registers per lane: 4 × (diag + residual + slot + 6 weights + 3 packed near). The operator loads once per call instead of 32 times. | 4 calls × (35-50 − ~10 µs) ≈ **−0.10 to −0.16** | S (one constant + packing in `coarseSolve`, `uniform-pressure-band.ts:48,643-676`) | **Bit-identical** (same update order). Risk: 1024-lane register pressure. Check pipeline creation and spills on Dawn/Metal; fall back to HELD = 3 / 3072. CPU preflight, then one fig-9 run comparing the band history words. |
| **P1** | **Fuse each level transition** (`mgResidual[Tiles]` + `mgRestrictResidual` + `mgClearPressure` + `mgDownsampleSubtract`) into one coarse-lane kernel computing the 8 child residuals in place. There are 3 transitions per V-cycle: −9 dispatches per slot. | 9 × 14 µs ≈ **−0.13** (−0.05 of it in closed slots) | M (new mg entry in #6, plan step change in `buildPlanSteps`) | ulp shifts only (Metal FMA). Verify with CPU preflight, one fig-9 run, then the uniform pressure lanes (dam, long-dam-front). |
| **P3** | **Resident L4 + L5 sub-cycle in one workgroup.** L4 (600 cells) and L5 (144) fit in workgroup memory (600 × 28 B + 144 × 48 B ≈ 23.7 KB). Fold L4 visit, transfer ops, `mgSolveCoarsest`, prolong and visit (8 dispatches) into 1 kernel. The coarse LCP iterations then run on workgroup memory instead of storage barriers. | 7 × 14 µs ≈ **−0.10**, plus the coarse-solve speed-up (see P5) | M-L | It must honour both minimum modes (memory WP-C "shifted-chain landmine"). Bitwise only if the DS arithmetic is kept. Verify as for P1, plus the long-dam-front lane. |
| **P0** | **Band residual target (measure first).** With `solve.x = 0` the band always runs 4 cycles, about 0.5 ms each. Set the target to the root's tolerance in the same units (5 s⁻¹ with 0.1 relative, `pressure-policy.ts`), not the buggy 1/h². First extend the receipt copy (`encodeReceipt`, 32 B) to cover the HISTORY words 8-23 (non-blocking, diagnostics only) to see how many cycles reach 5 s⁻¹ on fig-9. | Per cycle saved: −0.5 + 23 × 4 µs closed ≈ **−0.4**. If the mean drops 4 → 3: −0.4; 4 → 2: −0.8 | S (one param) + S (probe) | **High quality risk.** Memory: the under-solved band lost rigid-body mass, and the red-black asymmetry noise grows with looser tolerance. Gate on the long-dam-front, rigid-body mass and symmetry lanes. Biggest single lever if the history shows slack. |
| **P4** | **Checkpoint fusion.** Fold `resetCycle` into the gate kernel that precedes the cycle, and the `cycle` verdict into the next gate (both 1-lane). Have `measure` do the 64-lane max plus one `atomicMax` per workgroup, deleting `reduce` (only if non-listed owners' residuals are provably 0, which setup writes). From 4 checkpoint dispatches + 1 gate per slot to 1 + 1. | 3 per slot × 14 µs + 2 initial ≈ **−0.05** | S-M (#31/#34 share a binding group) | Verdict semantics unchanged. Verify with a fig-9 run comparing acceptance words and plan receipts. |
| **P8** | **Gate the surface census on the overlay.** It is dead in the solve (§1.1). | **−0.03 to −0.05** | S | Must keep the overlay layer served when on. Verify by preflight census and an overlay check in the app. |
| **P5** | **`mgSolveCoarsest`: measure iterations, then restructure.** Read `convergence[26]/[27]` (already accumulated) in an existing stats readback to get iterations per call. If it is ≳ 10: keep rows in workgroup memory (144 × 48 B = 6.9 KB) instead of `mgState.rows` with `storageBarrier`, and check the residual every 4 iterations (exit on the same criterion, just later). | Unknown: 30-60 µs per call → ~10, so **−0.02 to −0.05** per open slot (subsumed if P3 lands) | S (probe) / M | ulp-level; iterations change only by ≤ 3 extra. |
| **P6** | **Delete `mgCopyPressureTiles`.** `prolongRoot` reads parity B (MG_P1) directly when the post-sweep parity is odd, which it is every cycle. | 1 × ~(8 + 8.2) µs ≈ **−0.016** | S | Bit-identical. |
| **P11** | **Size all-4h counted grids to resident pages** (§1.5). | **−0.02 to −0.04** | S | Bit-identical, layout-general (pressure fine capacity = 0). |
| **P7** | **Root merge** (structural). Bake the mixed root operator (ghost-fluid theta, boundary coefficients, V) into L2's coefficient vec4 so the native continuation is the root operator. This deletes `restrictRoot`, `prolongRoot`, the copy, the root `measure`'s theta recompute and the root→native setup hop (3 [405] launches per slot + setup). | ≈ 3 × 14 µs + setup ≈ **−0.06 to −0.1** | L | **Blocked** by the unexplained native-vs-mixed inconsistency (memory LEAD, long-dam blow-up). Native theta already uses `UNIFORM_MIXED_THETA_MIN` (`webgpu-uniform-pressure-multigrid.wgsl.ts:663-702`), so the difference lies in boundary, V or wall-halo terms. Start with a CPU replay comparing the two operators row by row. |
| P10 | **Solid-free twin hygiene.** Skip the 6 `rows[(15+f)N]=1.0` stores in `prep` and the `bVolume` loads in `project` when `umSolidsPresent = 0`; or derive V from the solid record in the S twin and drop the fields. | ~3.4 MB writes per frame, ≈ **−0.01** | S | Bit-identical. |
| P13 | Move L3-L5 static topology out of the arena (63.7 KB) and rebuild only on solid edits: −3 dispatches. | ≈ −0.015 | S-M | Live-edit path must invalidate. |
| P14 | Ping-pong the presentation/root pressure buffers instead of 2 × 81,920 B copies. | ≈ −0.005 | S | |
| P15 | Startup: a mixed-only #6 module (19% of source), skip `buildPlan()` and the L0/L1 tile lists, defer unused solid twins, and drop the `measure@1-3` variants. | 0 frame; less compile and memory | S-M | |

**Order I would run:**
1. P2 (bit-identical, small).
2. P6 + P11 + P10 together (bit-identical).
3. P1 + P4 (slot body −12 dispatches; the closed-slot cost drops 0.61 → about 0.51 ms).
4. P0's probe, which decides the biggest lever.
5. P3/P5.
6. P7 only after a CPU operator comparison.

All arms together excluding P0: about −0.4 to −0.5 ms of the 5.1 ms. P0 could add −0.4 to −0.8 if the band history shows slack.

## 4. Storage that can be deleted (fig-9 bytes)

| Item | Bytes | Condition |
|---|---|---|
| Band rows V fields `rows[(15+f)N]` (6 × 4 B × 1,048,576 rows) | **25,165,824** | Derive V from the solid record in the S twin (it is constant 1 in the free twin) |
| Band capacity sized at 16,384 tiles vs observed max 3582 (rows 84 MiB, coarse 6.2 MB, solve 4.46 MB) | up to **≈ 76,000,000** at a 4096-tile bound (rows 21 → 15 fields: 4096 × 64 × 60 = 15.7 MB) | Needs a fail-fast capacity bound. Memory "capacity is not inert": it must abort loud on overflow, which the index overflow + fatal words already support |
| Band forced fields `rows[(9+f)N]` | 25,165,824 (net ≈ 8.4 MB after a 16.8 MB copy target) | Only if `bandCopy` gets its own target |
| Module #6 L0/L1 smooth and cycle tile lists | ≈ 500,000 | Mixed path never uses them |
| Project #58 `centerPhi` binding | 0 B (a binding slot, 4.19 MB texture stays) | Comment at `uniform-mixed-pressure-velocity.ts:32` says it is unread |
| Duplicate transfer bind group (`toSimulation` = `toPressure` args) | object only | |
| Surface band bitmask when the overlay is off | 2,048 | P8 |
| `measure@1..3` band pipelines (6 incl. twins) and unused solid twins (about half of every #29-#58 pipeline pair) | compile only | P15 |
