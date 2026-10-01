# Uniform Geometric deep analysis: "head" group

Scope: the frame head (census, layout builder, adopt and remap, resolve and solid widths, stage grids, plan, certificate, 4h cache, hanging taps), the ownership transfer, and the tail copies and receipt. It also covers a full GPU storage inventory and non-dispatch command inventory for cm12-figure-9, plus minimal-power-dam-break-64.

HEAD is c38adea3. All of this work was CPU-only.

**Tools (all in scratch):**
- `inv/capture-inventory.mts` and `inv/capture-passes.mts` are variants of `tools/capture-uniform-wgsl.mts`. Their mock device logs:
  - every createBuffer/createTexture, with label, size, format and call site;
  - every bind group;
  - every dispatch, with its bound groups;
  - every copy, clear, writeBuffer, pass and submit.
- `inv/table2.py` joins dispatches to bind groups and WGSL declarations. It decides "used" by **static reachability**: the binding's variable must be referenced in the call graph of the dispatched entry point, not merely bound.

**Runs:**
- `NOAWAIT=1 FRAMES=2 node --import tsx inv/capture-inventory.mts cm12-figure-9 inv/fig9b` (and `minimal-power-dam-break-64` into `inv/dam64b`).
- Frame 2 is the steady-state path: the relayout head, with the tail extension reused.

**Outputs:**
- Full tables: `inv/fig9-table2.txt`, `inv/dam64-table2.txt`, and `inv/fig9b/table2.json`.
- Commands: `inv/cmds.txt`.
- Frame-2 sequence: `inv/seq-fig9.txt`.

**Caveat.** The mock sees only the solver. The renderer and overlay readers of the presentation buffers are not visible to it. Every "DEAD" verdict below is "no solver reader in init or frames 1-2". Before deleting a DEAD item, grep the renderer and overlay for its label. The likely candidates are #22 and #130, both from the reference solver.

---

## 0. Where the head's time goes (baseline profile, `runs/base`)

| stage (profile label) | fig-9 mean ms | dam64 mean ms | what it contains |
|---|---:|---:|---|
| Resolution census + layout build + remap | **1.556** | 0.917 | census #52 (14 launches), builder #53 (7), adopts, remap #13/#12/#14 (5), generation copy |
| Solid record + liquid displacement | 0.103 | 0.065 | #16 resolveListed, #56 widthsListed |
| Support plan + interface phase | 0.350 | 0.215 | stage-grid copy, plan #10 seed and dilate×3, #24 compactChanged and geometryChanged, #28 build |
| 4h transport reach + sampling caches | 0.446 | 0.262 | certificate #10 (8 launches), #18 cache, #19 hanging and unitFaces |
| **group total in the head** | **2.455 (11% of 22.42)** | 1.459 (12% of 12.09) | |

The tail runs plan.encode again (seed and dilate×3) plus #28 build. That cost is priced inside "Mixed extension sweeps". The transfer #32 is priced inside the pressure stages.

**Fixed cost dominates.** fig-9 has 4× the tiles of dam64, but the census, build and remap stage grows only 1.556 / 0.917 = 1.7×. Fitting t = a + b·tiles:
- b = (1.556 − 0.917) / (16384 − 4096) = **0.052 µs per tile**
- a = 0.917 − 4096·0.052e-3 = **0.70 ms fixed**

That fixed 0.70 ms is spread over 26 dispatches and 11 encoder segments (6 compute passes and 5 blit runs), about 27 µs per dispatch. The head is therefore **latency/launch bound, not work bound**. The levers are fewer dependent launches, fewer compute↔blit transitions, and shorter per-lane dependent chains. That matches the prior M1 Max findings.

### Frame-2 command census (fig-9, `inv/fig9p`)

- **Whole frame:** 1214 dispatches, 79 compute passes, 20 blit runs (52 blit commands), 3 submits, 22 writeBuffer (704 B).
  - 82 launches use a fixed 4096-workgroup grid; 72 use 2048.
- **Head (census through hanging):** 46 dispatches in 14 compute passes, plus 7 blit runs.
- **dam64 frame 2:** 925 dispatches. Clears total 75,076 B; copies total 374,296 B.

---

## 1. Per-kernel analysis, in frame order (frame 2, relayout head)

Lattice: 128×128×64. That gives T = 32×32×16 tiles, n = 16384 tiles of 4³, and 256 pages of 4³ tiles. Live mix is about 11k h tiles and 5k 4h tiles.

**Topology buffer:** 4n words (256 KB).
- `[0,n)` tile words: bit 31 = h; low 30 bits = first owner index.
- `[n,2n)` the h list, then the 4h list.
- `[2n,4n)` stencils, two words per tile.

**Support buffer:** 9n + 32 + 2·pages words (578 KB).

### 1.1 Census #52 (`uniform-mixed-dynamic.ts`): 3 blit clears plus 14 launches in 3 compute passes, every frame

The clears are census 2128 + 4096 + 2052 B. They share the frame's first blit run with the dust clear, so they cost no extra transition.

| kernel (line) | grid × wg | one lane / one job | loads and chains | atomics / barriers |
|---|---|---|---|---|
| classify (513) | 4096 × 64; grid-stride over the h list (about 2.7 jobs per workgroup) | one h cell (`umClassifyOwner`) | list→tile→tile word. **12 `umFace` per cell**, each a `umOwnerAt` load on the neighbour's tile word, then a velocity texel: chain depth 4 (count→list→neighbour word→texel). 8 corners via `umVertexValue`: stencil load → branch → phi texel. When mixed, `umResolutionError` reads 8 + 125 vertices. | 3 workgroup barriers per job; 18 workgroup atomicMin/Max plus 2 atomicOr per lane; lane 0 serial `umFinishTile` with global atomics |
| classifyCoarse (541) | 256 × 64 | one 4h tile per lane | list→tile; up to 6×16 face parts with `umOwnerAt` each; 8 direct corners | global `atomicAdd(census[3|6|7])` per interface coarse owner (contended words) |
| prefix0/1/2 (556) | 9/9/18 × 64 | one line of the (T+1)³ table | **serial 32/32/16-step loop of `atomicLoad` + `atomicStore`** on census; dependent sum; no load pipelining | 0 |
| boundCube ×5 (593) | 256 × 64 each (cubeLevel override) | one start tile | gate: `interfaceTilesIn` (8 atomic SAT loads), then 8 × 6 loads (level 1 = atomics, from AoS `6t+k`; levels ≥2 = SoA cube planes) | 0 |
| decide (741) | 256 × 64 | one tile | early out via SAT (8 atomic loads). Otherwise an RK2 box trace of `steps` frames × 2 `sampledFlow` × `rangeKeys` (≤64 cubes × 6 keys), then `departureMeetsSurface` (SAT plus shell). Memory-driven trip counts; strongly divergent. | global atomicAdd to census[2], [4] and [5] per tile |
| [solidActive, solidPromote] | 256 × 64 | only with solids or bodies (not fig-9) | | |
| pageSeed (828), pageMark (845) | 4 × 64 | one page; 64-tile serial loop / (2r+1)³ page loop | atomic loads | |
| pageCompact (866) | **1 × 256** | 256 pages | Hillis-Steele: 8 steps × 2 barriers | |

**Dead or redundant:**
- On the h list `width` is always 1, yet it is loaded through `umTileWidth(tile)` at line 518. Every width-4 branch of `umClassifyOwner` is compiled into `classify`: the 16-part face loops and the `width==4` corner path.
- For h owners `umVertexValue` ≡ `umLoadVertex`. The resolved invariant (`uniform-mixed-vertex-sampling.wgsl.ts:51-60`) stores every vertex of a tile with a mixed or uniform-h stencil. An h cell's +plane corner that falls in the next tile lies in a tile whose stencil is mixed, because it has an h neighbour. So the 2-word stencil load and branch per corner are pure overhead, ×8 per lane, plus 8 in `umResolutionError`.
- The face anchors of an h owner are fixed: the positive face is at its origin, and the negative face is at origin − e_axis. Neither needs `umOwnerAt`. Only the *neighbour index* would need the tile word, and the census never reads `face.neighbor.index`.

**Proposals:**
- **C1 — h-owner specialisation.** In `classify`, pass the constant width 1, and replace both `umFace` and `umVertexValue` with direct anchors and `umLoadVertex`. Per lane this removes 12 dependent tile-word loads and 16 stencil loads, shortening the chain from 4 to 2: list→tile, then texels. `classifyCoarse` keeps the general path.
  - Expected ≈20–35% off `classify`. Not separately timed: I estimate classify at 0.15–0.25 ms of the 1.556, so the saving is **≈0.04–0.08 ms**.
  - Risk: low. The output must be bit-identical, since the same texels are loaded.
  - Verification:
    - CPU: capture the WGSL, naga parse it, and diff the census readback words on a CPU replay of `umClassifyOwner` for random layouts.
    - A/B: compare census words and frame time on both scenes.
- **C2 — non-atomic prefix table.** Move the SAT to its own non-atomic `array<u32>` binding. `classify` writes its own entry with a plain store, one writer per tile.
  - Each prefix lane then does plain loads that the compiler can hoist and pipeline (unroll by 4).
  - Readers (`boundCube`, `decide`, `departureMeetsSurface`) do plain loads too: 8 per query, previously atomic.
  - Each prefix pass currently runs about 32 serial device round trips (load, then dependent store) per lane: 32 × roughly 0.4 µs ≈ 13 µs, ×3 passes ≈ 40 µs. Pipelined plain loads should cut that to under 15 µs. Expected **≈0.03 ms**, plus cheaper SAT queries in `decide`.
  - Risk: low.
- **C3 — one compute pass for the census.** The three passes (classify / prefix / decide, line 950) are split only for labels. Merge them and remove 2 encoder boundaries: **≈0.01–0.03 ms**. The trace phase markers can stay as timestamps inside the pass, if the tracer allows, or be dropped for this stage.
- **C4 — `pageCompact` into `pageMark`.** Use a last-workgroup ticket (atomic counter; the last of the 4 workgroups compacts), instead of a separate 1-workgroup launch. That is −1 launch, ≈0.01–0.02 ms.

### 1.2 Builder #53 (`uniform-mixed-layout-builder.ts`): 1 blit run (clear 64 B + copy 80 B census→receipt) plus 7 launches in 1 pass, every frame

| kernel (line) | grid × wg | lane | access | sync |
|---|---|---|---|---|
| widths (187) | 256 × 64 | one tile | 2 bitmask loads; `current[t]` | atomicAdd to the changed list |
| classify (213) | 64 × 256 | one tile | **27 neighbour `atomicLoad`s of `work[FLAGS]`** (atomics, so no cache reuse across the 27-point stencil) | 5 workgroup atomicAdds per tile; 2 barriers |
| dilate (196) | 256 × 64 | one changed tile | 27 `atomicOr`s on flags | append atomics |
| scan (244) | **1 × 256** | 64 blocks × 5 categories | 5 × (serial loop + Hillis-Steele 8×2 barriers) = **80 barriers** in a single workgroup | |
| scatter (280) | 64 × 256 | one tile | **5 `scanPartial` = 80 barriers per workgroup**, then up to 5 scattered stores | |
| verifyWords (311) | 256 × 64 | one tile | re-reads `topology[t]`, the list entry (dependent) and the receipt | self-check of scatter's own output |
| sealBuild (320) | **1 × 1** | | 4 atomics | |

**Dead or redundant:**
- The 5 categories are not independent. Writing c0 = h:
  - c1 = 1 − c0 for valid tiles;
  - c4 = c1 − c3.

  So only c0, c2 and c3 need scanning. With per-block counts ≤ 256 (9 bits each), all three pack into one u32 with 10-bit fields.
- `verifyWords` re-derives what `scatter` just wrote, every frame. It is an internal-consistency assertion, not input validation.
- `sealBuild` is one lane of one workgroup.

**Proposals:**
- **B1 — packed scan.** Scatter does 1 `scanPartial` of a packed u32 (16 barriers instead of 80). `scan` does 2 passes (block totals ≤ 16384 fit 15 bits; pack two per word) instead of 5.
  - Barrier cost on Apple is about 0.1–0.2 µs, serialised per workgroup. Scatter saves 64 barriers ≈ 6–13 µs on the critical path of each of 2 waves (64 workgroups of 256 over 32 cores), so ≈12–25 µs. Scan saves about 48 barriers plus 3 serial loops ≈ 8 µs.
  - Total **≈0.02–0.03 ms**. Risk: low. CPU check: a TS emulation of `categories` + scatter versus the packed version over random width fields.
- **B2 — drop `verifyWords` from the hot path,** keeping it in the Dawn lane under a debug override. Fold `sealBuild` into scatter's last workgroup with an atomic ticket.
  - That is −2 launches ≈ **0.02–0.04 ms**.
  - Risk: low for correctness, but it is a policy question. Fail-fast doctrine wants loud failures. This check guards the builder's own determinism, not data, and the remap's `markBlocked` still latches every fatal the scan raises (tier sum, hanging capacity, residency).
- **B3 — non-atomic flags in classify.** Classify reads `work[FLAGS+t]` with 27 atomicLoads. After `widths`, the flags are read-only until classify's own write of bit 8. Write classify's result to a second plane (`FLAGS2`), and the 27 reads become plain loads that the L1 serves across the stencil. Expected ≈5–10 µs.

### 1.3 Generation adopt copies (`uniform-mixed-ownership.ts:271-277`, `uniform-mixed-remap.ts:125-133`): 3 blit runs, 13 commands

```
45..49  receipt 4+144 B; target.adoptGpu: topology 262,144 + counts 16 + support 196,640   (→ remap target)
50..51  clear worklist 16 + copy fatal 8
56..59  live.adoptGpu: topology 262,144 + counts 16 + support 196,640 + slots 131,072
62      generation word 4 B → frame status
```

These copy 1,049,200 B per frame. Of that, **458,800 B goes to the remap target, and the target's support (196,640 B) is never read.** Static reachability over #12, #13 and #14 shows the target's `umSupport` and `umCounts` are unread by `remapFaces`, `copyFaces` and `markListed`. `remapCells` reads only `umCounts.w`, through `umVertexAuthority`.

**Proposals:**
- **A1 — the remap target is the builder's own level.** Bind group 1 of the remap ("new" ownership) directly to the builder's `topology` / `work` / `support` buffers (`UniformMixedGenerationBuffers`).
  - The new generation is final in those buffers once `sealBuild` runs.
  - The new-ownership `umCounts` is a uniform; give the builder a 256-aligned 16-byte counts slot with UNIFORM usage, or make the remap's new-side counts a storage read.
  - This deletes the 3-copy target adopt (458,800 B) and the target ownership allocation:
    - topology 256 KB + support 578 KB + speeds 128 KB + counts ≈ **0.96 MB**;
    - plus the dead speeds of the other ownerships (see the storage list).
  - Per-frame: −3 blit commands. Combined with E1 below, one transition fewer. Expected **≈0.01–0.02 ms**.
  - Risk: low. Verification: the remap kernels see identical words, so CPU-check the WGSL; Dawn A/B.
- **A2 — live adopt without a blit run.** Optional. Replace the 4 live-adopt copies with one grid-stride compute "publish" kernel (≈147k words) inside the remap publish pass, *before* `copyFaces` (a separate dispatch, same pass). That removes the blit run at 56 and its 2 encoder transitions.
- **A3 — status word from the GPU.** Write the generation word into the status buffer from `sealBuild` (or the publish kernel) by binding status there. That removes the 1-copy blit run at 62.

### 1.4 Remap (`uniform-mixed-remap.ts`, modules #13/#12/#14): 2 compute passes, 5 launches, every frame

| launch | grid × wg | lane / job | notes |
|---|---|---|---|
| markListed #13 (12.wgsl:539) | 256 × 64 | one dilated changed tile | 27 neighbour width compares (old vs new tile words); atomic append |
| remapFaces (extension) #12 | **4096 × 192** | job = listed tile; lane = cell × axis | `remapSample` → `oldumOwnerAt`, `oldPatch` → `oldumFace` (2 more old tile words) → 2 texels. `remapFace` sums ≤16 LDS samples. `remapReleased` adds ≤16 old tile words + 16 texels. 2 barriers. |
| remapCells #14 | 4096 × 64 | job = listed tile | 125-vertex authority loops (`umVertexAuthority` = 8 incident owners × tile word), refine path, 4 barriers |
| remapFaces (velocity) #12 | 4096 × 192 | as above | |
| copyFaces ×2 #12 (publish pass) | 4096 × 192 | listed tile | copy texels from the staging texture back |

**Dead or redundant:**
- `#12` and `#13` are **byte-identical WGSL compiled as two modules** (38,021 B each; `cmp` equal). This is a one-time compile cost only, but it is free to share.
- Every launch is a fixed 4096-workgroup grid striding a list that is usually short (the dilated changed tiles).
- The staging texture "Uniform mixed remapped extension" (16 MB rgba32f, #347) is used only by these 4 launches. Its content is dead outside the head.

**Proposals:**
- **R1 — alias the staging texture.** Use the unit-taps texture (#353; same dims, format and usage) as the staging texture for the extension remap. Unit taps are fully rewritten by #19 (`hanging` + `unitFaces`) after the head and before any reader. That deletes **16 MB** (dam64: 4 MB) at zero time cost.
  - Risk: low. The only invariant is that nothing reads unit taps between head and #19. In frame 2 the readers are #19 onward (#20 advect, #26, #27), all after.
  - Verification: CPU — a static check of reader order in the capture.
- **R2 — shrink the grids.** Cut REMAP_GRID (line 13) from 4096 to about 512 (≥ resident capacity: 32 cores × ~8 workgroups of 192). Grid-stride makes any grid correct.
  - An early-exit 4096-workgroup launch costs dispatch and retire time for 4096 workgroups: on the order of 5–15 µs, by analogy with the measured ~12 µs empty indirect dispatch.
  - 5 launches → **≈0.02–0.05 ms**. A big relayout (thousands of listed tiles) loses nothing, because only about 300 workgroups are resident anyway.
  - The same applies frame-wide to the 82 fixed 4096-workgroup launches (`COUNTED_GRID`, `TRANSFER_GRID`, `CENSUS_TILE_GRID`). This is experiment E3.
- **R3 — fuse the two face launches per pass.** Each launch pair binds two field sets over the same list and lanes. One launch looping over both field groups (both bound) saves 2 launches and the duplicate `listed()` / `umOwnerAt` / `umPositiveFaceAtAnchor` work: **≈0.02 ms**.
  - The order constraint is `remapCells` between them. Check whether `remapCells` reads the extension; it does not bind it (#14 group 2 = volume and phi), so the extension remap can move next to the velocity remap.

### 1.5 `phiResolve` resolveListed #16, solid widthsListed #56: 1 + 1 launches, 2 passes

- `resolveListed`: 4096 × 64 over the changed list. Hanging vertices of tiles whose neighbourhood changed width.
- `widthsListed` (56.wgsl, 791 B): 256 × 64, one store of `record[20480+t].y = h?1:0`.
  - This is a **duplicate representation** of topology bit 31, kept in an f32 channel of the all-4h solid record.
  - The tile part of the solid record (16384 vec4f = 256 KB) has 2 of 4 channels live: x = cut, y = h. Both are booleans.

Proposal **S1**: put both bits into the pressure-side topology or a 2-bit-per-tile word, and delete `widthsListed`. That is −1 launch and −1 pass (≈0.01–0.02 ms) and frees 256 KB. Low priority. The all-4h readers are in the pressure group's modules (#29, #30, #33, #36).

### 1.6 Stage-grid record (`uniform-mixed-frame.ts:417-421`): 4 copies per frame plus 1 small one

| copy | bytes | where |
|---|---:|---|
| `transport` | 65,536 | head, line 690 / 468 |
| `band` | 2,048 | line 512 |
| `pressure` | 65,536 | line 551 |
| band slot map | 65,536 | line 554 |
| root pressure phi | 65,536 | line 550 |

These run unconditionally, even when no grid overlay is visible. `layoutViews` gates only the reasons, previous and certificate views.

Proposal **G1**: gate the stage-grid copies on overlay visibility, a host-known flag. The pressure-phi copy is needed by the overlay only, but check the renderer first. This removes 2 single-command blit runs (at seq 209 and inside 65) and trims 2 copies in the tail run: **≈0.01–0.03 ms**. Risk: overlay staleness when it is toggled on; it lags one frame.

### 1.7 Frame plan #10 seed + dilate×3 (`uniform-mixed-frame-plan.ts:104-139`): **twice per frame** (head after relayout; tail before extension)

Each run is 1 blit run (clear 64 B + 65,536 B of seed flags), 1 writeBuffer (policy), and 4 launches:
- seed: 4096 × 64, counted `residentAll`;
- dilate0/1/2: 256 × 64.

**seed:** lane = owner.
- Chain: `umResidentAllOwner` = count → h list entry → tile word. For 4h pages it is page count → page id → tile word → width.
- Then `umPositiveFaceSpeed`. For a non-unit tile that is `umOwnerAt` + 3 × `umFace` (+ part loops) → texels.
- 8 × (`umVertexValue` or `umLoadVertex`).
- `atomicMax` to workgroup memory, then lane 0 does a global `atomicMax(support[4n])`.

**dilate:** (2·reach+1) **atomic** loads per lane per pass.

**All `umSupport` reads in this module are rewritten as `atomicLoad` (line 36).** That includes the shared helpers:
- `umTileSupport`;
- the residency count and page list;
- the seam and certificate headers.

Atomic loads on Apple bypass L1, so every one is an L2 round trip. Only three things in the plan actually need atomics:
- `atomicMax(support[4n])` (seed speed);
- `atomicAdd(support[4n+1..2])` (certify list bases);
- the certify list appends.

The other writes are single-writer stores.

**Proposals:**
- **P1 — atomics only where needed.** Move the 4 atomic header words (speed, two list counts, spare) into a 16-byte atomic side buffer, or a separate binding of a small range of another buffer. Keep `umSupport` as plain `array<u32>` in the plan module.
  - Every topology helper read in 12 launches × 2 (head + tail plan + certificate) becomes a cacheable plain load.
  - Expected **≈0.02–0.05 ms** in total. Risk: low; it is a binding change only.
- **P2 — skip the head re-plan when the relayout changed nothing.** It already reuses the extension in that case. Alternatively make the head plan incremental: re-seed only owners of the dilated changed list, and re-dilate only the boxes around them.
  - The live support planes `[0,4n)` survive the adopt: `copyGeneration` with `planned=false` copies only `[6n+16, 9n+24)`. So the tail plan's planes are valid wherever the widths did not change.
  - A full plan is 4 launches + 1 blit run. The incremental version is 2 short listed launches with no clear (zero the listed tiles' seeds in-kernel).
  - Expected **≈0.05–0.1 ms**, if the head plan is about 0.1–0.15 ms of the 0.35 ms "support" stage.
  - Risk: medium. Dilation must be exact for every tile within reach of a changed tile. Use a 3D box re-evaluation per affected tile rather than separable passes.
  - CPU check: a TS replay of seed + dilate (full versus incremental) over recorded layouts.

### 1.8 Certificate #10 (localSpeed, spread×3, prefix×3, certify; lines 149-226): 8 launches, 1 pass, every frame

- **localSpeed:** 4096 × 64, `umAllOwner` over all owners.
  - Chain for h: count → h list → tile word. The tile word is only used for `.index`, which `localSpeed` never uses, so the compiler drops it.
  - `umSignedFaceExtent`: for unit tiles 1 texel per axis; otherwise `umOwnerAt` + `umFace` parts → texels.
  - 6 workgroup `atomicMax` per lane.
- **spread0-2:** 256 × 64, (2r+1) loads per lane per axis on `speeds`.
- **prefix0-2:** 8/8/16 × 64, serial 32/32/16-step loops, but on `speeds` (non-atomic, good).
- **certify:** 256 × 64. The SAT query is 8 loads. Appends to the support lists use workgroup aggregation (good).

**Dead or redundant:**
- The SAT of 4h tiles (prefix0-2) depends only on the layout, yet it is recomputed every frame after `spread2` (3 launches).
- The builder already scans widths. It could emit the 4h-tile SAT once per build, into the same `speeds[0,T)` range that `spread2` leaves dead, or its own range.
- On frames where the layout did not change, the SAT is identical to the previous frame's.

**Proposal K1.** Build the 4h-count SAT in the builder (1 extra pass inside the builder's compute pass, or fused into `verifyWords`'s replacement). Delete prefix0-2 from the certificate: −3 launches ≈ **0.03–0.05 ms**. Risk: low; it is the same table.

### 1.9 4h sampling cache #18 and hanging #19: 3 launches in 2 passes

**`cache` (18.wgsl:267):** 9×9×5 × (4,4,4), one lane per (n/4+2)³ entry.
- Per axis: `umOwnerAt` → `umFace` (neighbour tile word) → up to 16 texel loads.
- Writes rgba32f with w = 0, so 3 of 4 channels are live (325 KB; negligible).

**`hanging` (19.wgsl:449-466):** 4096 × 256, one job per slot. Slots are the h-seam tiles **and** the 4h-seam tiles (`uniform-mixed-layout-builder.ts:294-306`).
- 192 lanes compute `umVelocityTap1` per cell and axis, and 48 lanes compute the negative-plane taps.
- **For an h-seam slot every tap is `umLoadMixedFace`**, the stored face, because `tileWidth==1` returns at 19.wgsl:351. So the 192-lane fill writes exactly what `unitFaces` writes next.
- The plane record taps (lanes 192-239) of an h slot are never read: `umVelocityTap1` consults the record only for width-4 tiles (20.wgsl:585-590).
- So **all hanging work on h-seam slots is redundant**.

**`unitFaces`:** 4096 × 64, a verbatim `textureLoad(extended)` → `textureStore(unitVelocity)` for every h tile.
- About 11k × 64 × 16 B read + 16 B written ≈ 22.5 MB of texture traffic per frame.
- This is the deliberate "resolved sampler" representation: #20's fine sampler does 8 direct unit-texture loads with no topology chain (20.wgsl:603-611). So the copy itself is a fair trade.

**Proposals:**
- **H1 — restrict `hanging` to 4h-seam slots and fuse it with `unitFaces`.** The launch list becomes h tiles (copy) followed by 4h-seam tiles (interpolate), so h tiles are written once.
  - Removes the redundant h-seam jobs (192 lanes × several taps each) and 1 launch: **≈0.02–0.04 ms**.
  - Risk: low. Values are identical by construction. Verification: CPU — static proof above plus a capture diff; A/B on texture checksums.
- **H2 — shrink the hanging capacity.** Capacity = n slots: 11.2 MB = (2 + 173)·n words. Sizing it to a measured peak of seam tiles × 1.5 is legitimate, because the builder already raises `FATAL_HANGING` past capacity (fail-fast). It needs a seam-count measurement from the profile (receipt words `R_SEAMS`).
  - If peak seams are about 4k, this saves 8.5 MB at fig-9. Low priority; memory only.
- **H3 — unit taps at 3 channels instead of 4.** rgba32f is required for storage-texture writes of 3 values; rgba16f is ruled out for canonical fields. Leave it.

### 1.10 Ownership transfer #32 (`uniform-mixed-remap.ts:193-260`): 1 blit copy (2,064 B residency) plus 2 + 2 launches per frame

**`toPressureFine`** (4096 × 64, h list):
- Lane 0 sums 64 volumes **serially** from LDS.
- Lanes 0-2 then do `tMean` (16 serial LDS reads each, ×2 with walls).
- 61 lanes idle through that tail.

**`toPressureCoarse`:** resident pages, one lane per 4h tile, 3 + 16·split texels.

**`toSimulationFine`:** every lane loads the same 6 anchor texels (broadcast; cheap).

Proposals:
- **T1 — tree-reduce the 64-volume mean** (6 barriers) and use 48 lanes for the six 16-sample means. This shortens the serial tail by about 60 LDS steps per job, ≈0.5 µs per job. With about 2.7 jobs per workgroup on the critical path, roughly **0.01 ms**. Low priority.
- **T2 — copy the residency words in-kernel.** Have `toPressureCoarse` (lane 0) copy the 516 residency words instead of the 2,064 B blit. That removes a single-command blit run (seq 220): ≈0.01 ms.

### 1.11 Solid #7/#55/#57/#8

These are not dispatched in steady-state fig-9 frames: static solids are built once (`encodeCoarse` is gated by `built`), and `encodeSimulation` runs only via #56 on the changed list. There is nothing to cut per frame.

The record tile channels are covered in 1.5. A body-mirror copy (1,536 B) appears only in frame 1.

### 1.12 Tail copies (`uniform-mixed-frame.ts:523, 550-560`)

Each frame these do:
- the warm start (presented → arena, 81,920 B) and the return (arena → presented, 81,920 B);
- 3 presentation copies of 65,536 B;
- 5 receipt copies of 32 / 48 / 8 / 32 / 64 B into one readback.

Proposals:
- **Q1 — dedicated root-pressure buffer.** Give the root pressure iterate (`levels[0].pressure`, currently an arena range) its own persistent 80 KB buffer. That deletes both 81,920 B copies; the warm start is then in place. It also removes the single-copy blit run at seq 229. The renderer can bind that buffer, but check frame pairing: presentations are paired (one outstanding presentation).
- **Q2 — one contiguous receipt record.** The 5 receipt sources (`state`, `reductions`, schedule plan, band receipt, `status`) could become ranges of one buffer, giving one copy. The copies are already in one blit run, so this is CPU encode time only. Low.

---

## 2. Chained indirection: tile → owner → slot → neighbour

### What every stage pays today (topology ABI, `uniform-mixed-topology.wgsl.ts`, `uniform-mixed-faces.wgsl.ts`)

| lookup | chain (dependent loads) | used by |
|---|---|---|
| `umAllOwner` / `umOwner` tier path (119, 143) | count (uniform) → `topology[n+job]` (tile) → `topology[tile]` (first index) → field | owner-indexed kernels: #9, #21, #27, #29, #33, plan, transfer |
| `umResidentAllOwner` (129) | count → page id → tile id → `topology[tile]` (width) | plan seed, transfer coarse |
| certificate / seam lists (`umPlannedFine`, `umMergedTileJob`) | header count → list entry → `topology[tile]` | #20, #26, #27 |
| `umFace` (faces.wgsl:23) | owner → `umOwnerAt(probe)` = neighbour tile word → anchor → texel | every face reader: census, plan, cache, remap, transport, momentum, pressure |
| `umVertexValue` (vertex-sampling:51) | tile → stencil (2 words) → branch → texel | phi readers |
| hanging tap | tile → `umHanging[tile]` (slot) → record or texel | #20 general sampler |
| `umVelocitySamplingWeights` | tile → support word, then tile word, then stencil (3 separate arrays, same index) → mask loop | #20, #26 |

Two facts make this chain cheaper to resolve than it looks.

1. **For tier-list owners the tile word is arithmetic.** The builder assigns:
   - h: base = rank·64, and the h list is in rank order (`scatter`, builder:296);
   - 4h: word = 64·f + rank, list at n + f + rank.

   So for job j: h index = 64·j + lane, and 4h index = 64·f + j. The `topology[tile]` load in `umAllOwner` / `umOwner` (tier path) and in `umResidentAllOwner`'s h branch is redundant whenever `.index` is used. Where `.index` is unused, the compiler already drops it.
   - `.index` *is* used by #9 transport (27 uses), #21 surface band, #29/#33 pressure, #27 forces and #17 extension (grep counts in `inv/fig9b/*.wgsl`).
2. **A face's neighbour width is already in the owner tile's stencil.** The 27-bit h-neighbour mask holds the 6 face neighbours.
   - For an owner whose face lies inside its tile, the neighbour width is the owner's own width: no load.
   - For a tile-boundary face it is one bit of the owner tile's mask: one load per tile, shared by all of its faces and lanes.

   Today every `umFace` call does a fresh neighbour tile-word load. A 4h owner beside h pays 6 + 16 such loads.

### Proposal: a resolved per-job owner record plus stencil-resolved faces (experiment E2)

**(a) Job-ordered record `J[job]` (vec4u, 16 B per tile-job), written by the builder's scatter.** For the h list, the 4h list and each GPU list the builder emits (seam, regular 4h):

```
J.x = tile id
J.y = first owner index (tile word & 0x3fffffff) | h bit
J.z = stencil word 0 (27-bit h mask | max width << 27)
J.w = stencil word 1 (min width << 27) | hanging slot (24 bits; UM_NO_SLOT = 0xffffff)
```

- Size: 16 B × (n tier entries + seam entries + regular entries) ≤ 3n × 16 B = **768 KB at fig-9**. The planned certificate lists (regular and general h) are rebuilt per frame by `certify`; it would append the same 16-B record instead of a bare tile id, for +12 B per entry.
- `umOwner` / `umAllOwner` / `umTileJobOwner` become: count → `J[job]` → field, so **depth drops from 3 to 2**. The width, stencil and slot of the job's own tile come in the same 16-byte load instead of 3–4 separate arrays. One coalescing note: 64 lanes of one tile job load the same `J`, which is a broadcast.

**(b) Stencil-resolved `umFace`.** Compute `width = (local face inside tile) ? owner.width : bit(J.z, dir) ? 1 : 4`. Load the neighbour tile word only if the caller reads `face.neighbor.index`; that is a separate `umFaceNeighbourIndex(face)`.
- Removes 1 dependent load per face for every face-velocity reader.
- Census classify (12 faces per lane), plan seed/localSpeed, the cache, the transfer and the remap all read only `anchor` / `width`.

**(c) Per-tile record `R[t]` (vec4u) for spatial lookups (`umOwnerAt(p)`).** Pack {tile word, stencil0, stencil1 | slot, support} into 16 B at `topology[t]`. `support` changes per frame, so either leave .w as the slot, or have the plan write .w. `umVelocitySamplingWeights` and `umVertexValue` then do 1 load instead of 3. That changes cache lines touched (3 → 1) but not depth.

**Costs and risk:**
- Topology grows from 4n to 4n words + 3n × 4 words: +768 KB (+0.2%).
- The ABI is shared by about 45 modules. Introduce it behind `uniformAbOn("jobrecord")` (the A/B switch exists: `uniform-ab-switch`), changing only the helper bodies, not the callers.
- Bit identity holds: the same texels are read.

**Estimate, hedged.** Each removed dependent device load on a latency-bound lane costs roughly one L2 round trip, about 0.3–0.6 µs. A list-driven kernel's critical path is a few jobs deep per workgroup, so each kernel gains about 1–3 µs. About 100 owner-list or face-driven dispatches per frame (outside the pressure hierarchy) touch these helpers, giving **≈0.1–0.3 ms at fig-9**. The largest single beneficiaries are probably the extension sweeps (12 per frame; "bound by its dependent topology-load chain" in the brief) and transport. The win is uncertain, so pilot it:
1. (b) alone first. It is local to `uniform-mixed-faces.wgsl.ts`, about 15 lines. Measure on fig-9.
2. Then (a).

**Verification:**
- CPU: a TS emulation of builder scatter + `J` against `deriveOwnership` for random width fields, asserting `J[job]` equals (list, word, stencil, slot) for every job.
- CPU: capture + naga parse of all modules.
- Dawn: one A/B per scene.

---

## 3. GPU storage inventory

The full per-resource table for fig-9 is in **Appendix A** (every resource ≥ 16 KB individually; the rest aggregated by label). Raw tables: `inv/fig9-table2.txt` and `inv/dam64-table2.txt`.

| | fig-9 | dam64 |
|---|---:|---:|
| total allocated | **369.00 MiB** (388 resources) | **93.16 MiB** (379) |
| statically unused in init + frames 1-2 (no solver reader) | **50.85 MiB** (151 resources) | **12.96 MiB** (152) |
| of which also untouched by any command | 50.05 MiB (82) | 12.73 MiB (82) |

**Largest buckets (fig-9):**

| bucket | MiB |
|---|---:|
| pressure band rows | 84.0 |
| stage arena | 68.1 |
| 6 × rgba32f 128×128×64 textures (velocity A/B/D, remapped extension, unit taps, plus transport A 130×130×66) | 97.0 |
| the dead 16 MB front receipts buffer | 16.0 |
| hanging tap cache | 10.9 |

### Findings by verdict

**DEAD (no solver reader):**

| id | resource | bytes |
|---|---|---:|
| #22 | Reference transport A, rgba32f 130×130×66 (bound in 13 groups, read by none) | 17,846,400 |
| #29 | Front convergence receipts. `frontReceipts` is referenced only in legacy FIM functions (5.wgsl:248-469); the mixed entry points `mixedHierarchyTail` / `mixedProlongUnknownVelocity` / `mixedRestrictKnownVelocity` never reach it (call-graph check). It is sized at 16 B per fine cell (`webgpu-uniform-velocity-extrapolation.ts:158-160`). | 16,777,216 |
| #130 | CM11a L0 pressure A, r32f 130×130×66 | 4,461,600 |
| #57, #58 | Nearest-source origins 64×64×32, rgba32uint | 2 × 4,194,304 |
| #55, #56 | Sec 3.3 hierarchy down/up 64×64×32, rgba32f | 2 × 2,097,152 |
| #196, #197, #198, #199, #203 | Pressure liquid/cycle tile lists. Two of them are still **cleared every frame** (4 B each, `webgpu-uniform-pressure-multigrid.ts:762`). | 148,064 + 91,750 + 20,808 + 15,168 + 700 |
| #7 | Negative boundary velocity C | 131,072 |
| #344, #352 | `speeds` of the remap-target and pressure ownerships: allocated for every ownership (`uniform-mixed-ownership.ts:179`), used only on the live one | 2 × 131,072 |
| #375 | Builder receipt readback | 65,600 |
| #30 | Extension shell list | 65,536 |
| #19, #20 | Column base/occupancy rg32f 128×64 | 2 × 65,536 |
| #368 | Census readback | 2,176 |
| many | "Uniform field page layouts" (0.6 KB each) and nearest/hierarchy pyramids at 8×8×4 and smaller | ≈0.1 MiB |

**Dead content, live copy:**
- #343, the remap-target support (578 KB): receives 196,640 B per frame; never read (1.3, A1).

**Duplicate or staging:**
- #347, Mixed remapped extension (16 MB): alias onto #353 (R1).
- #341, the remap-target topology (256 KB): could be the builder's own topology (A1).
- The all-4h solid record's tile part `.y` duplicates topology bit 31 (1.5).

**Oversized:**
- **#0, stage arena, 71,386,112 B.** It is sized by `max(extensionEnd, pressureWords, donor)` (`uniform-scratch-arena.ts:58-65`). `extensionEnd` is the legacy FIM range: 4 padded vec4 planes = 4 × 130·130·66 × 16 B / 4. Bound ranges in the mixed frame:
  - #9 transport: edges 41.9 MB at 0 + rigidExchange 12.6 MB at 41.9 MB + sums 4.2 MB at 54.5 MB, ending at **58.7 MB**;
  - #17 extension stateIn/stateOut: 2 × 26.1 MB, ending at 52.2 MB;
  - #30 mixed 26.8 MB at 4.46 MB;
  - #30 native 1.0 MB at **63.6 MB**.

  The high-water mark is therefore 64.6 MB: **≥6.8 MB slack**. A mixed-only layout that puts the pressure ranges (#29/#30, CM11a L2+) below 58.7 MB would cap the arena at the transport's 58.7 MB, saving **≈12.7 MB**. Verify the #6 `fieldDims` offsets first; they are runtime metadata, not visible in the binding ranges.
- #336, Hanging tap cache, 11,468,800 B (capacity = n slots; H2).
- #364, band rows, 88 MB: for the pressure-band analyst.

**Fewer live channels than allocated:**

| resource | live channels |
|---|---|
| #353 unit taps rgba32f | 3 of 4 (w = 0) |
| #329 4h sampling cache | 3 of 4 |
| #331 solid record tile part | 2 of 4, both booleans |

Formats constrain the first two (no rgb32f storage textures; f16 rejected), so they are kept.

---

## 4. Non-dispatch command inventory (fig-9, frame 2)

Totals:
- 22 writeBuffer (704 B; queue-side, not in the stream);
- 27 clears (297,796 B);
- 25 copyB2B (1,479,192 B);
- **20 blit runs** interleaved with 79 compute passes.

| seq | run | commands (bytes) | owner | can go |
|---|---|---|---|---|
| 15 | 1 | clear dust 48; census 2,128 + 4,096 + 2,052 | frame / census | keep (frame-start run) |
| 36 | 2 | clear builder work 64; copy census→builder receipt 80 | builder | **yes**: `pageCompact` (or the census tail) zeroes the 16 words and writes the 20 receipt words directly |
| 45 | 3 | receipt 4 + 144; **target adopt topology 262,144, counts 16, support 196,640**; worklist clear 16 + fatal copy 8 | adopt / remap | **mostly**: A1 removes the target adopt; the worklist zero/fatal can be done by `sealBuild` |
| 56 | 4 | live adopt: topology 262,144, counts 16, support 196,640, slots 131,072 | adopt | A2 (compute publish) |
| 62 | 5 | generation word 4 | frame | A3 (`sealBuild` writes it) |
| 65 | 6 | stage grid "transport" 65,536; plan clears 64 + 65,536 | stage grid / plan | G1 + P2 (in-kernel zero) |
| 73 | 7 | clear geometry changed 16 | #24 | **yes**: hoist into run 6, or zero in `compactChanged`'s first lane |
| 89 | 8 | clear surface job claims 24,640 | #20 | hoist to the frame-start run (3 separate claim ranges) |
| 98, 101 | 9, 10 | clear claims 64, 64 | #20 | idem (per-substage claim slots) |
| 113 | 11 | clear transport live set 80 | #9 | hoist |
| 169 | 12 | clear sharpening tile list 65,568 | #25 | hoist |
| 197 | 13 | clear momentum claims 32 + deferred 16 | #26 | hoist |
| 206 | 14 | clear pressure surface band 2,048 | #21 | hoist |
| 209 | 15 | stage grid "band" 2,048 | stage grid | G1 |
| 214 | 16 | clear band tiles 104 + 65,536 | band | hoist |
| 220 | 17 | transfer residency 2,064 | #32 | T2 |
| 229 | 18 | warm start 81,920 | frame | Q1 |
| 232 | 19 | 6 × clear 4 B pressure tile lists (**2 of them dead buffers**) | CM11a | hoist; drop the dead 2 |
| 1249 | 20 | presented pressure 81,920; root phi 65,536; stage grids 65,536 × 2; receipt 32 / 48 / 8 / 32 / 64 | tail | keep one run |

**Proposal E1 — blit-run consolidation.** Hoist every clear whose buffer is not written between frame start and its clear point into the single frame-start blit run. For buffers cleared more than once (the surface claims ×3), give each sub-stage its own range and clear all ranges once.

Together with A1–A3, G1, T2 and Q1, this takes the frame from **20 blit runs to about 3** (frame start, the live adopt if A2 is not done, and the tail). That saves 17 compute→blit→compute transitions.

The per-transition cost on Dawn/Metal is not yet measured. Calibrate it first with a probe: insert 10 dummy 4-byte clears between head passes, one Dawn run. At 5–15 µs per transition the gain is **≈0.08–0.25 ms**. Risk: low, ordering only. The CPU check is the mock sequence: assert that no hoisted buffer is written between frame start and its old clear point.

---

## 5. Ranked experiments (fig-9 gain estimates; one Dawn A/B per scene each, CPU checks first)

| # | experiment | est. fig-9 gain | effort | risk | refs |
|---|---|---:|---|---|---|
| **0** | Split `A.resolutionCensus` into census / builder / adopt+remap trace phases, and calibrate one blit transition (10 dummy clears) | measurement only; makes the rest groundable | XS | none | `uniform-mixed-frame.ts:680`, `uniform-mixed-dynamic.ts:950` |
| **1** | E1: blit-run consolidation, 20 → about 3 runs. Hoist the clears; A1 (bind the builder's level as the remap "new" ownership, deleting the 459 KB target adopt); A3 (generation word from `sealBuild`); builder header zeroing in the census tail; G1 (gate stage grids on overlay) | **0.08–0.25 ms** | S–M | low | §1.3, §1.6, §4 |
| **2** | E2: resolved owner record `J[job]` + stencil-resolved `umFace` (pilot (b) first) | **0.1–0.3 ms** | M–L (shared ABI, about 45 modules, behind `uniformAbOn`) | medium | §2 |
| **3** | E3: shrink fixed 4096-workgroup grids to about 512–1024 where the list is usually short (REMAP_GRID; changed-list launches #13, #14, #16, #24; `hanging`). Survey the 82 fixed 4096 launches per frame. | 0.05–0.15 ms (5–15 µs × about 10 short-list launches) | XS per site | low (grid-stride is grid-independent) | `uniform-mixed-remap.ts:13`, `uniform-mixed-ownership.ts:71`, `uniform-mixed-remap.ts:196` |
| **4** | P2: no full head re-plan; incremental seed/dilate over the dilated changed list, since the planes survive the adopt | 0.05–0.10 ms | M | medium | `uniform-mixed-frame.ts:691`, `uniform-mixed-frame-plan.ts:104-139` |
| **5** | Census: C1 (h-owner specialisation of `classify`: width const, fixed face anchors, `umLoadVertex`) + C2 (non-atomic SAT) + C3 (one pass) + C4 (`pageCompact` folded) | 0.08–0.15 ms | S–M | low | `uniform-mixed-dynamic.ts:413-440, 513-563, 866, 950` |
| **6** | P1 + K1: plan module plain `umSupport` (atomics only for 4 header words); 4h-tile SAT built once per build instead of prefix0-2 every frame | 0.05–0.10 ms | S | low | `uniform-mixed-frame-plan.ts:36, 188-198` |
| **7** | Builder: B1 packed single scan (5 → 1 in scatter, 5 → 2 in scan); B2 `verifyWords` off the hot path, `sealBuild` folded; B3 non-atomic flags in classify | 0.04–0.07 ms | S | low (B2 is a policy call) | `uniform-mixed-layout-builder.ts:213-325` |
| **8** | H1: `hanging` over 4h-seam slots only, fused with `unitFaces` into one launch (h-seam fills are provably identical to `unitFaces`) | 0.02–0.04 ms | S | low | 19.wgsl:437-470; builder:294-306 |
| **9** | R1 + R3: alias the remapped-extension staging onto unit taps (−16 MB); fuse the ext/vel face remaps and the two `copyFaces` (−2 launches) | 0.02 ms; 16 MB | S | low | `uniform-mixed-frame.ts:259-260`, `uniform-mixed-remap.ts:125-160` |
| **10** | Q1 + T2: dedicated 80 KB root-pressure buffer (warm start in place, −2 × 81,920 B copies, −1 run); residency words copied in-kernel by the transfer | 0.01–0.03 ms | S | low–medium (presentation pairing) | `uniform-mixed-frame.ts:523, 550`, `uniform-mixed-remap.ts:251` |

The estimates overlap: items 1, 3, 5, 7 and 9 all remove launches or transitions from the same fixed 0.70 ms. A realistic combined target for this group is **0.3–0.6 ms at fig-9** (about 1.5–2.5% of 22.4 ms), with experiment 2 the largest uncertain upside frame-wide.

---

## 6. Storage that can be deleted (fig-9 bytes; dam64 in brackets)

| item | bytes | how |
|---|---:|---|
| #22 Reference transport A (rgba32f 130×130×66) | 17,846,400 [4,599,936] | not allocated on the mixed path (verify no overlay reader) |
| #29 Front convergence receipts | 16,777,216 [4,194,304] | allocate 16 B when the method is mixed (`cacheConvergence` false), or move it under the legacy FIM path |
| #347 Mixed remapped extension | 16,777,216 [4,194,304] | R1: alias the unit-taps texture |
| #130 CM11a L0 pressure A (r32f 130×130×66) | 4,461,600 [1,149,984] | not allocated (mixed root lives in the arena) |
| #57, #58 nearest-source origins 64×64×32 | 8,388,608 [2,097,152] | legacy FIM pyramid level |
| #55, #56 Sec 3.3 hierarchy 64×64×32 | 4,194,304 [1,048,576] | legacy FIM pyramid level |
| #0 arena slack | ≥6.8 MB (≈12.7 MB with a mixed layout) | mixed-specific `UniformScratchLayout` (verify #6 offsets) |
| remap target ownership (#341 topology, #343 support, #344 speeds, counts) | 262,144 + 591,968 + 131,072 + 16 ≈ 0.96 MB | A1 |
| #352 pressure-ownership speeds | 131,072 | allocate `speeds` only for the live ownership |
| #196–#199, #203 pressure tile lists (dead; 2 still cleared per frame) | ≈276,000 | drop, and drop their clears |
| #7 negative boundary C | 131,072 | drop |
| #375 builder receipt readback; #368 census readback; #30 shell list; #19/#20 column tex | 65,600 + 2,176 + 65,536 + 131,072 | drop |
| #336 hanging cache oversize | up to about 8.5 MB | H2: capacity from measured peak seams (fatal past it) |
| solid record tile part (vec4f → 2 bits) | about 256 KB | S1 |
| **sum, without H2/S1** | **≈77.0 MB = 73.4 MiB, about 20% of 369 MiB** (dam64 ≈ 17.3 MB before arena slack) | 17.85 + 16.78 + 16.78 + 4.46 + 8.39 + 4.19 + 6.8 + 0.96 + 0.13 + 0.28 + 0.13 + 0.26 MB; +8.5 MB with H2, +12 MB more with a mixed arena layout (12.7 instead of 6.8) |

---

## Appendix A: fig-9 allocation table

Columns:
- **f2 kernels/dispatches (static R/W):** the frame-2 entry points that statically reach the binding, and their dispatch count.
- **per-frame cmds:** frame-2 non-dispatch commands touching the resource (`<` = destination, `>` = source).

Total fig-9: **369.00 MiB in 388 resources** (mock-device capture, frames 1-2 encoded).

| id | bytes | kind | label | f2 kernels/dispatches (static R/W) | per-frame cmds | verdict | site |
|---:|---:|---|---|---|---|---|---|
| 364 | 88,080,384 | buf | Uniform pressure band rows | 8/61 RW |  | needed (band group); largest single allocation — size check belongs to the pressure-band report | uniform-pressure-band.ts:147 |
| 0 | 71,386,112 | buf | Uniform shared stage scratch | 93/1076 RW | copyB2B<81920×1, copyB2B>81920×1 | needed; sized by the legacy FIM range (4 padded rgba planes = 71.4 MB); mixed high-water ≈ 64.6 MB (#30.native ends at 61.6 MiB) → OVERSIZED ≈ 6.8 MB, ~12 MB if #30.native is packed below #9 | uniform-scratch-arena.ts:85 |
| 22 | 17,846,400 | rgba32float 130x130x66 | Uniform reference transport A | 0/0  |  | DEAD in the mixed frame (bound in 13 groups, no entry point reads it) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 1 | 16,777,216 | rgba32float 128x128x64 | Uniform reference velocity A | 23/25 RW |  | needed: live velocity (xyz faces + w released-wall bits: 4 live) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 2 | 16,777,216 | rgba32float 128x128x64 | Uniform reference velocity B | 35/40 RW |  | needed: velocity scratch = the extension (4 live) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 4 | 16,777,216 | rgba32float 128x128x64 | Uniform reference velocity D | 14/16 RW |  | needed: departure / D (check channel use in velocity report) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 29 | 16,777,216 | buf | Uniform front convergence receipts | 0/0  |  | DEAD: frontReceipts referenced only by legacy FIM entries; mixed #5 entries never touch it | webgpu-uniform-velocity-extrapolation.ts:158 |
| 347 | 16,777,216 | rgba32float 128x128x64 | Uniform mixed remapped extension | 2/2 RW |  | DUPLICATE staging: only remapFaces/copyFaces (2 dispatches); alias onto 353 (unit taps are dead at the head) | uniform-mixed-frame.ts:260 |
| 353 | 16,777,216 | rgba32float 128x128x64 | Uniform mixed unit velocity taps | 14/16 RW |  | needed but 3 of 4 channels live (w=0); 11 MB/frame of it is a verbatim copy of the extension (unitFaces) | uniform-mixed-momentum-cache.ts:102 |
| 336 | 11,468,800 | buf | Uniform mixed velocity tap cache | 9/11 RW | copyB2B<131072×1 | needed; sized for slot capacity = all tiles (175 words × n); seam slots in use are a fraction | uniform-mixed-ownership.ts:183 |
| 365 | 6,225,920 | buf | Uniform pressure band aggregates | 9/95 RW |  | needed (band) | uniform-pressure-band.ts:148 |
| 360 | 4,655,136 | buf | Uniform presented pressure phi and stage grids | 1/1 W | writeBuffer<32×1, copyB2B<65536×4, copyB2B<2048×1 | needed (renderer/overlay + warm start); 4 stage-grid copies/frame land here | buffer@uniform-mixed-frame.ts:241 |
| 28 | 4,522,500 | buf | Uniform reference compatibility scratch | 16/37 RW | copyB2B>65536×1 | needed (pressure compat scratch) | webgpu-uniform-reference.ts:985 |
| 130 | 4,461,600 | r32float 130x130x66 | Uniform CM11a L0 pressure A | 0/0  |  | DEAD (texture-pages L0 pressure A; mixed root uses the arena) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 366 | 4,456,448 | buf | Uniform pressure band iterate | 11/97 RW |  | needed (band) | uniform-pressure-band.ts:149 |
| 356 | 4,390,944 | buf | Uniform mixed sharpening tile list | 6/27 RW | clear<65568×1 | needed; 65,568 B cleared/frame | buffer@uniform-mixed-frame.ts:241 |
| 17 | 4,326,660 | r32float 129x129x65 | Uniform Geometric vertex phi | 30/43 RW |  | needed: canonical vertex phi | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 18 | 4,326,660 | r32float 129x129x65 | Uniform Geometric vertex phi scratch | 9/11 RW |  | needed: phi scratch | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 11 | 4,194,304 | r32float 128x128x64 | Uniform reference volume A | 29/47 RW |  | needed: V | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 12 | 4,194,304 | r32float 128x128x64 | Uniform reference volume B | 15/28 RW |  | needed: V scratch | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 13 | 4,194,304 | r32float 128x128x64 | Uniform reference smoothed surface A | 9/11 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 14 | 4,194,304 | r32float 128x128x64 | Uniform reference smoothed surface B | 13/15 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 15 | 4,194,304 | r32float 128x128x64 | Uniform reference transport gamma A | 4/4 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 16 | 4,194,304 | r32float 128x128x64 | Uniform reference transport gamma B | 5/7 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 57 | 4,194,304 | rgba32uint 64x64x64 | Uniform nearest-source origins 64x64x32 | 0/0  |  | DEAD (64³ nearest origins, legacy FIM) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 58 | 4,194,304 | rgba32uint 64x64x64 | Uniform nearest-source origins 64x64x32 | 0/0  |  | DEAD (64³ nearest origins, legacy FIM) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 241 | 4,194,304 | r32float 128x128x64 | Uniform mixed pressure surface target | 4/4 RW |  | needed (#24/#33) | scalar@webgpu-uniform-reference.ts:1330 |
| 242 | 4,194,304 | r32float 128x128x64 | Uniform mixed pressure centre phi | 4/4 RW |  | needed (#24/#33) | scalar@webgpu-uniform-reference.ts:1330 |
| 55 | 2,097,152 | rgba32float 64x64x32 | Uniform Sec. 3.3 hierarchy down 64x64x32 | 0/0  |  | DEAD (64×64×32 hierarchy, legacy) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 56 | 2,097,152 | rgba32float 64x64x32 | Uniform Sec. 3.3 hierarchy up 64x64x32 | 0/0  |  | DEAD (64×64×32 hierarchy, legacy) | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 373 | 1,966,080 | buf | Uniform dynamic ownership bound cubes | 4/8 RW |  | needed; 6 keys × 5 levels × n u32; levels above topCube() unused most frames | uniform-mixed-dynamic.ts:228 |
| 338 | 1,114,192 | buf | Uniform mixed transport live set | 14/28 RW | clear<80×1 | needed (transport) | uniform-mixed-transport.ts:60 |
| 26 | 992,512 | buf | Uniform reference active liquid census scratch and summaries | 34/61 R |  | needed (read-only census scratch) | webgpu-uniform-reference.ts:940 |
| 367 | 804,248 | buf | Uniform dynamic ownership census | 12/16 RW | clear<2128×1, clear<4096×1, clear<2052×1, copyB2B>80×1 | needed | uniform-mixed-dynamic.ts:219 |
| 334 | 592,000 | buf | Uniform shared frame support and certified work | 78/115 RW | copyB2B<4×1, copyB2B<196640×1, clear<64×2, clear<65536×2, copyB2B>2064×1 | needed: live ownership support | uniform-mixed-ownership.ts:178 |
| 343 | 592,000 | buf | Uniform shared frame support and certified work | 0/0  | copyB2B<196640×1 | DEAD content: remap target support — copied 196,640 B/frame, never read | uniform-mixed-ownership.ts:178 |
| 351 | 592,000 | buf | Uniform shared frame support and certified work | 14/65 RW | copyB2B<2064×1 | needed: pressure ownership support | uniform-mixed-ownership.ts:178 |
| 377 | 589,920 | buf | Uniform layout builder support | 2/2 RW | copyB2B>196640×2 | needed (builder) | storage@uniform-mixed-layout-builder.ts:120 |
| 331 | 589,824 | buf | Uniform mixed all-4h solid record | 11/27 RW |  | needed; tile part (16384 vec4) has 2 of 4 channels live, both booleans | uniform-mixed-solid.wgsl.ts:97 |
| 61 | 524,288 | rgba32uint 32x32x32 | Uniform nearest-source origins 32x32x16 | 3/3 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 62 | 524,288 | rgba32uint 32x32x32 | Uniform nearest-source origins 32x32x16 | 4/5 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 329 | 332,928 | rgba32float 34x34x18 | Uniform 4h sampling cache 0 | 14/16 RW |  | needed; rgba32f with 3 live (w=0) | <anonymous>@uniform-mixed-frame.ts:249 |
| 59 | 262,144 | rgba32float 32x32x16 | Uniform Sec. 3.3 hierarchy down 32x32x16 | 3/3 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 60 | 262,144 | rgba32float 32x32x16 | Uniform Sec. 3.3 hierarchy up 32x32x16 | 13/18 RW |  | needed | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 332 | 262,144 | buf | Uniform mixed owners and tier worklists | 89/140 R | copyB2B<262144×1, copyB2B>65536×1 | needed: live topology | uniform-mixed-ownership.ts:169 |
| 341 | 262,144 | buf | Uniform mixed owners and tier worklists | 4/6 R | copyB2B<262144×1 | needed only for 3 kernels; could be the builder's own buffer (no copy) | uniform-mixed-ownership.ts:169 |
| 349 | 262,144 | buf | Uniform mixed owners and tier worklists | 8/15 RW | copyB2B>65536×1 | needed: pressure topology | uniform-mixed-ownership.ts:169 |
| 376 | 262,144 | buf | Uniform layout builder topology | 3/3 RW | copyB2B>262144×2 | needed (builder output) | storage@uniform-mixed-layout-builder.ts:120 |
| 379 | 197,952 | buf | Uniform layout builder work | 10/10 RW | clear<64×1, copyB2B>16×2 | needed | storage@uniform-mixed-layout-builder.ts:120 |
| 358 | 196,624 | buf | Uniform mixed momentum deferred tiles | 2/2 RW | clear<16×1 | needed | uniform-mixed-momentum.ts:182 |
| 196 | 148,120 | buf | Uniform pressure liquid tile list | 0/0  |  | DEAD (legacy pressure tile list) | webgpu-uniform-pressure-multigrid.ts:473 |
| 363 | 131,176 | buf | Uniform pressure band tiles | 16/109 RW | clear<104×1, clear<65536×1, copyB2B>65536×1, copyB2B>32×1 | needed | uniform-pressure-band.ts:146 |
| 355 | 131,088 | buf | Uniform mixed geometry changed tiles | 2/2 RW | clear<16×1 | needed | uniform-mixed-surface-geometry.ts:57 |
| 5 | 131,072 | buf | Uniform reference negative boundary velocity A | 17/19 RW |  | needed | boundaryVelocity@webgpu-uniform-reference.ts:860 |
| 6 | 131,072 | buf | Uniform reference negative boundary velocity B | 29/34 RW |  | needed | boundaryVelocity@webgpu-uniform-reference.ts:860 |
| 7 | 131,072 | buf | Uniform reference negative boundary velocity C | 0/0  |  | DEAD (negative boundary C) | boundaryVelocity@webgpu-uniform-reference.ts:860 |
| 8 | 131,072 | buf | Uniform reference negative boundary velocity D | 7/7 RW |  | needed | boundaryVelocity@webgpu-uniform-reference.ts:860 |
| 335 | 131,072 | buf | Uniform local speed certificate | 8/8 RW |  | needed (live speeds) | uniform-mixed-ownership.ts:179 |
| 340 | 131,072 | buf | Uniform mixed cleanup fine seam tile summary | 2/3 RW |  | needed | uniform-mixed-cleanup.ts:23 |
| 344 | 131,072 | buf | Uniform local speed certificate | 0/0  |  | DEAD (speeds of remap-target ownership) | uniform-mixed-ownership.ts:179 |
| 348 | 131,072 | buf | Uniform mixed remapped extension walls | 2/2 RW |  | needed with 347 (staging walls) | buffer@uniform-mixed-frame.ts:241 |
| 352 | 131,072 | buf | Uniform local speed certificate | 0/0  |  | DEAD (speeds of pressure ownership) | uniform-mixed-ownership.ts:179 |
| 378 | 131,072 | buf | Uniform layout builder slots | 1/1 W | copyB2B>131072×1 | needed (builder slots) | storage@uniform-mixed-layout-builder.ts:120 |
| 354 | 98,692 | buf | Uniform mixed surface job claims | 8/11 RW | clear<24640×1, clear<64×2 | needed | uniform-mixed-surface.ts:70 |
| 197 | 91,760 | buf | Uniform pressure cycle tile list | 0/0  |  | DEAD (legacy pressure cycle tile list) | webgpu-uniform-pressure-multigrid.ts:481 |
| 359 | 81,920 | buf | Uniform presented pressure | 0/0  | copyB2B>81920×1, copyB2B<81920×1 | needed (warm-start home + presentation) | buffer@uniform-mixed-frame.ts:241 |
| 375 | 65,680 | buf | Uniform layout builder receipt | 0/0  |  | DEAD (builder receipt readback, unused on GPU path) | uniform-mixed-layout-builder.ts:128 |
| 30 | 65,552 | buf | Uniform extension shell list | 0/0  |  | DEAD (extension shell list, legacy) | webgpu-uniform-velocity-extrapolation.ts:167 |
| 345 | 65,552 | buf | Uniform mixed remap worklist | 4/6 RW | clear<16×1, copyB2B<8×1 | needed | uniform-mixed-remap.ts:50 |
| 19 | 65,536 | rg32float 128x64x1 | Uniform reference column base | 0/0  |  | DEAD (column base) | webgpu-uniform-reference.ts:898 |
| 20 | 65,536 | rg32float 128x64x1 | Uniform reference column occupancy | 0/0  |  | DEAD (column occupancy) | webgpu-uniform-reference.ts:899 |
| 65 | 65,536 | rgba32uint 16x16x16 | Uniform nearest-source origins 16x16x8 | 2/3 RW |  |  | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 66 | 65,536 | rgba32uint 16x16x16 | Uniform nearest-source origins 16x16x8 | 1/2 RW |  |  | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 337 | 65,536 | buf | Uniform mixed departure sampling widths | 8/12 RW |  |  | uniform-mixed-transport.ts:58 |
| 21 | 32,768 | r32float 128x64x1 | Uniform reference terrain | 34/61 R |  |  | webgpu-uniform-reference.ts:900 |
| 63 | 32,768 | rgba32float 16x16x8 | Uniform Sec. 3.3 hierarchy down 16x16x8 | 2/3 RW |  |  | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 64 | 32,768 | rgba32float 16x16x8 | Uniform Sec. 3.3 hierarchy up 16x16x8 | 1/2 RW |  |  | UniformTexturePages.createTexture@uniform-texture-pages.ts:38 |
| 198 | 20,824 | buf | Uniform pressure liquid tile list | 0/0  |  | DEAD | webgpu-uniform-pressure-multigrid.ts:473 |

Resources under 16 KB: 310, 152,488 B total, aggregated by label:

| label | count | bytes | of which statically unused |
|---|---:|---:|---:|
| Uniform field page layouts | 56 | 32,256 | 50 |
| Uniform CM11a dispatch parameters | 73 | 25,696 | 0 |
| Uniform pressure cycle tile list | 5 | 19,312 | 4 |
| Uniform nearest-source origins 8x8x4 | 2 | 16,384 | 0 |
| Uniform CM11a convergence status | 1 | 7,024 | 0 |
| Uniform Sec. 3.3 hierarchy down 8x8x4 | 1 | 4,096 | 0 |
| Uniform Sec. 3.3 hierarchy up 8x8x4 | 1 | 4,096 | 0 |
| Uniform dynamic ownership join tiles | 1 | 4,096 | 0 |
| Uniform layout builder static masks | 1 | 4,096 | 0 |
| Uniform pressure liquid tile list | 4 | 4,080 | 2 |
| Uniform reference active liquid region | 1 | 3,392 | 0 |
| Uniform dynamic ownership readback | 1 | 2,128 | 1 |
| Uniform nearest-source origins 4x4x2 | 2 | 2,048 | 1 |
| Uniform independent pressure surface band | 1 | 2,048 | 0 |
| Uniform dynamic ownership solid tiles | 1 | 2,048 | 0 |
| Uniform dynamic ownership static solid tiles | 1 | 2,048 | 1 |
| GPU authoritative rigid-body state | 1 | 1,536 | 1 |
| GPU rigid primitive motion sidecar | 1 | 1,536 | 1 |
| GPU rigid-body roster scratch | 1 | 1,536 | 1 |
| GPU rigid motion roster scratch | 1 | 1,536 | 1 |
| Uniform mixed solid body mirror | 1 | 1,536 | 0 |
| Uniform mixed solid record body mirror | 1 | 1,536 | 0 |
| Uniform reference active indirect dispatches | 1 | 1,024 | 1 |
| GPU rigid-body render records | 1 | 768 | 1 |
| GPU rigid render roster scratch | 1 | 768 | 1 |
| Uniform reference rigid exchange | 1 | 576 | 1 |
| Uniform Sec. 3.3 hierarchy down 4x4x2 | 1 | 512 | 1 |
| Uniform Sec. 3.3 hierarchy up 4x4x2 | 1 | 512 | 0 |
| Uniform Sec. 3.3 front parity 0->0 | 15 | 480 | 10 |
| Uniform reference parameters | 1 | 272 | 0 |
| Uniform nearest-source origins 2x2x1 | 2 | 256 | 2 |
| Uniform GPU pressure cycle dispatch | 1 | 252 | 1 |
| Uniform pressure receipt and mass accounting 0 | 1 | 184 | 1 |
| Uniform pressure receipt and mass accounting 1 | 1 | 184 | 1 |
| Uniform layout builder relayout receipt | 1 | 144 | 0 |
| Uniform mixed relayout receipt read 0 | 1 | 144 | 1 |
| Uniform mixed relayout receipt read 1 | 1 | 144 | 1 |
| Uniform mixed relayout receipt read 2 | 1 | 144 | 1 |
| Uniform mixed pressure schedule | 1 | 128 | 0 |
| GPU rigid-body step parameters | 1 | 112 | 1 |
| Uniform Sec. 3.3 hierarchy down 2x2x1 | 1 | 64 | 1 |
| Uniform Sec. 3.3 hierarchy up 2x2x1 | 1 | 64 | 1 |
| Uniform nearest-source origins 1x1x1 | 2 | 64 | 2 |
| Uniform mixed frame status | 1 | 64 | 0 |
| Uniform mixed remap idle status | 1 | 64 | 1 |
| Uniform dynamic ownership policy | 1 | 64 | 0 |
| Uniform reference diagnostics and volume control | 1 | 48 | 1 |
| GPU rigid immersed volumes | 1 | 48 | 1 |
| GPU rigid-body pick result | 1 | 48 | 1 |
| Uniform forces parameters | 1 | 48 | 0 |
| Uniform dust accounting | 1 | 48 | 0 |
| Uniform mixed work counts | 3 | 48 | 0 |
| Uniform pressure band parameters | 1 | 48 | 0 |
| Uniform Sec. 3.3 front parity 0->1 | 1 | 32 | 1 |
| Uniform Sec. 3.3 front parity 1->0 | 1 | 32 | 1 |
| GPU rigid-body pick ray | 1 | 32 | 1 |
| Uniform surface parameters | 1 | 32 | 0 |
| Uniform momentum parameters | 1 | 32 | 0 |
| Uniform sharpen parameters | 1 | 32 | 0 |
| Uniform projection parameters | 1 | 32 | 0 |
| Uniform pressure acceptance | 1 | 32 | 0 |
| Uniform shared support policy | 1 | 32 | 0 |
| Uniform mixed momentum job claims | 1 | 32 | 0 |
| Uniform reference unused velocity C | 1 | 16 | 1 |
| Uniform reference transport B | 1 | 16 | 1 |
| Extension unused source origins | 1 | 16 | 1 |
| Extension unused output origins | 1 | 16 | 1 |
| Uniform Sec. 3.3 FIM values A | 1 | 16 | 1 |
| Uniform Sec. 3.3 FIM values B | 1 | 16 | 1 |
| Uniform Sec. 3.3 FIM distances A | 1 | 16 | 1 |
| Uniform Sec. 3.3 FIM distances B | 1 | 16 | 1 |
| Uniform Sec. 3.3 resolved FIM values | 1 | 16 | 1 |
| Uniform Sec. 3.3 resolved FIM distances | 1 | 16 | 1 |
| Uniform Sec. 3.3 active-front convergence | 1 | 16 | 1 |
| Uniform Sec. 3.3 hierarchy down 1x1x1 | 1 | 16 | 1 |
| Uniform Sec. 3.3 hierarchy up 1x1x1 | 1 | 16 | 1 |
| Uniform disabled MacCormack audit binding | 1 | 16 | 1 |
| Uniform CM11a L0 V A | 1 | 16 | 1 |
| Uniform CM11a L0 coefficients | 1 | 16 | 1 |
| Uniform CM11a L1 V A | 1 | 16 | 1 |
| Uniform CM11a L1 coefficients | 1 | 16 | 1 |
| Uniform CM11a L2 V A | 1 | 16 | 0 |
| Uniform CM11a L2 coefficients | 1 | 16 | 0 |
| Uniform CM11a L3 V A | 1 | 16 | 0 |
| Uniform CM11a L3 coefficients | 1 | 16 | 0 |
| Uniform CM11a L4 V A | 1 | 16 | 0 |
| Uniform CM11a L4 coefficients | 1 | 16 | 0 |
| Uniform CM11a L5 V A | 1 | 16 | 0 |
| Uniform CM11a L5 coefficients | 1 | 16 | 0 |
| Pressure residual tolerance | 1 | 16 | 0 |
| Uniform extension parameters | 1 | 16 | 0 |
| Uniform authority parameters | 1 | 16 | 0 |
| Uniform acceptance parameters | 1 | 16 | 0 |
| Uniform layout builder params | 1 | 16 | 0 |
| Uniform mixed rigid body tiles horizon | 1 | 16 | 1 |
| Uniform extension shell dispatch | 1 | 12 | 1 |
| Uniform Sec. 3.3 indirect dispatch | 1 | 12 | 1 |
| Uniform Sec. 3.3 unused dispatch storage binding | 1 | 12 | 1 |
| Uniform reference pressure A | 1 | 4 | 1 |
| Uniform reference pressure B | 1 | 4 | 1 |
| Uniform CM11a L0 pressure B | 1 | 4 | 1 |
| Uniform CM11a L0 rhs A | 1 | 4 | 1 |
| Uniform CM11a L0 rhs B | 1 | 4 | 1 |
| Uniform CM11a L0 phi A | 1 | 4 | 1 |
| Uniform CM11a L0 phi B | 1 | 4 | 1 |
| Uniform CM11a L0 residual A | 1 | 4 | 1 |
| Uniform CM11a L0 p-min A | 1 | 4 | 1 |
| Uniform CM11a L0 p-min B | 1 | 4 | 1 |
| Uniform CM11a L1 pressure A | 1 | 4 | 1 |
| Uniform CM11a L1 pressure B | 1 | 4 | 1 |
| Uniform CM11a L1 rhs A | 1 | 4 | 1 |
| Uniform CM11a L1 rhs B | 1 | 4 | 1 |
| Uniform CM11a L1 phi A | 1 | 4 | 1 |
| Uniform CM11a L1 phi B | 1 | 4 | 1 |
| Uniform CM11a L1 residual A | 1 | 4 | 1 |
| Uniform CM11a L1 p-min A | 1 | 4 | 1 |
| Uniform CM11a L1 p-min B | 1 | 4 | 1 |
| Uniform CM11a L2 pressure A | 1 | 4 | 0 |
| Uniform CM11a L2 pressure B | 1 | 4 | 0 |
| Uniform CM11a L2 rhs A | 1 | 4 | 0 |
| Uniform CM11a L2 rhs B | 1 | 4 | 1 |
| Uniform CM11a L2 phi A | 1 | 4 | 0 |
| Uniform CM11a L2 phi B | 1 | 4 | 0 |
| Uniform CM11a L2 residual A | 1 | 4 | 0 |
| Uniform CM11a L2 p-min A | 1 | 4 | 0 |
| Uniform CM11a L2 p-min B | 1 | 4 | 1 |
| Uniform CM11a L3 pressure A | 1 | 4 | 0 |
| Uniform CM11a L3 pressure B | 1 | 4 | 0 |
| Uniform CM11a L3 rhs A | 1 | 4 | 0 |
| Uniform CM11a L3 rhs B | 1 | 4 | 0 |
| Uniform CM11a L3 phi A | 1 | 4 | 0 |
| Uniform CM11a L3 phi B | 1 | 4 | 0 |
| Uniform CM11a L3 residual A | 1 | 4 | 0 |
| Uniform CM11a L3 p-min A | 1 | 4 | 0 |
| Uniform CM11a L3 p-min B | 1 | 4 | 0 |
| Uniform CM11a L4 pressure A | 1 | 4 | 0 |
| Uniform CM11a L4 pressure B | 1 | 4 | 0 |
| Uniform CM11a L4 rhs A | 1 | 4 | 0 |
| Uniform CM11a L4 rhs B | 1 | 4 | 0 |
| Uniform CM11a L4 phi A | 1 | 4 | 0 |
| Uniform CM11a L4 phi B | 1 | 4 | 0 |
| Uniform CM11a L4 residual A | 1 | 4 | 0 |
| Uniform CM11a L4 p-min A | 1 | 4 | 0 |
| Uniform CM11a L4 p-min B | 1 | 4 | 0 |
| Uniform CM11a L5 pressure A | 1 | 4 | 0 |
| Uniform CM11a L5 pressure B | 1 | 4 | 0 |
| Uniform CM11a L5 rhs A | 1 | 4 | 0 |
| Uniform CM11a L5 rhs B | 1 | 4 | 0 |
| Uniform CM11a L5 phi A | 1 | 4 | 0 |
| Uniform CM11a L5 phi B | 1 | 4 | 0 |
| Uniform CM11a L5 residual A | 1 | 4 | 1 |
| Uniform CM11a L5 p-min A | 1 | 4 | 0 |
| Uniform CM11a L5 p-min B | 1 | 4 | 0 |
| Uniform CM11a Full-Cycle p_tmp | 1 | 4 | 1 |
| Uniform CM11a accepted pressure | 1 | 4 | 1 |
| dynamic trace marker storage | 1 | 4 | 1 |

dam64 per-resource table: `inv/dam64-table2.txt` (93.16 MiB in 379 resources; same DEAD set scaled: #22 4.6 MB, #29 4.2 MB, #347 4.2 MB, #130 1.15 MB).
