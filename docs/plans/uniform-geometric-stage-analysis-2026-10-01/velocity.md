# Velocity group: extension, momentum, forces, velocity storage (fig-9, HEAD c38adea3)

Paths are relative to lib/methods/uniform/. WGSL module numbers and pipeline ids come from $SCRATCH/wgsl/fig9.

## Timing

Timing comes from $SCRATCH/runs/base/fig9.json: 126 frames, GPU mean 22.4 ms. The fig-9 stage means in my scope are:

- "Mixed extension sweeps + hierarchy fill": 3.04 ms (p10 1.97, p90 4.85). This is the tail extension only; the head reuses it 59/60 frames. The window also includes the tail's `plan.encode` (support) and `authority.encode(phase)`.
- "Velocity advection + body forces": 2.22 ms. Momentum is about 1.95 of it and forces about 0.25 (prior per-dispatch: forces 0.13 + forcesRegular 0.05).
- "4h transport reach + sampling caches": 0.45 ms. This covers the certificate (localSpeed, spread×3, prefix×3, certify), cache, hanging and unitFaces.
- The earlier per-pass split of the tail extension (memory: uniform-census-chip-cost) was 3.23 ms in total:
  - h-interface sweeps: 1.04
  - sweepCoarse: 0.47
  - restrictBand: 0.24
  - remainder (seed, publish, hierarchy, plan, authority): about 1.5

Frame-1 cold head extension was 9.3 ms.

Fig-9 sizes used below:
- D = 128·128·64 = 1,048,576 cells
- T = 16384 tiles
- about 11000 h tiles, which is about 2.11 M h faces (11000·192)
- about 5000 4h tiles
- one native rgba32f D-texture = 16,777,216 B

---

## 0. Velocity storage census (fig-9)

| Resource | Format / size | Bytes | Role on the mixed path |
|---|---|---|---|
| velocityA (`f.velocity`) | rgba32f D | 16.78 MB | Physical canonical MAC. Needed. |
| velocityB (`f.velocityScratch`) | rgba32f D | 16.78 MB | Extension output, then the forces output. Needed. |
| velocityD (`f.departure`) | rgba32f D | 16.78 MB | Head: traceCells writes departure points (uniform-mixed-surface.ts:922) and transport reads them. Then the momentum output, which forces reads. Needed, and **dead at remap time** (see 9.2). |
| velocityC | 1³ (lazyMac) | 16 B | n/a |
| "Uniform mixed remapped extension" + walls (uniform-mixed-frame.ts:260-261) | rgba32f D + f32 | 16.78 MB + 131 KB | Staging for the extension remap: velocityB→this, then copied back after the velocity remap reuses velocityB as scratch (uniform-mixed-remap.ts:88-91, 156-162). **Can alias velocityD.** |
| hanging.unitVelocity (uniform-mixed-momentum-cache.ts:102) | rgba32f D | 16.78 MB | **Duplicate**: equals velocityB on every h tile (unitFaces copy), plus interpolated unit taps on seam-4h tiles. 4h non-seam texels are garbage. |
| 4h sampling cache 0 | rgba32f (n/4+2)³ = 34·34·18 | 333 KB | 4h-level face values with a halo. Small. |
| hanging records | (2T + 173·T)·4 (slots bound = T) | ~11.5 MB | 48 plane + 125 tap words per slot. Sized to the all-tiles bound. |
| extension scratch, 2 parities (arena) | vec2f slots: 3·D h slots + 3T coarse + walls + 2T mask | 2 × 26.1 MB (arena) | Sized for all-h. The bound is layout-general, so keep it. |
| extrapolator hierarchy level 0 (64×64×32) | down+up rgba32f + 2× origins rgba32uint [64,64,64] | 4.19 + 8.39 = **12.58 MB** | **Dead.** Mixed starts at level 1 (prepareMixedContinuation, webgpu-uniform-velocity-extrapolation.ts:443-460). |
| extrapolator levels ≥1 | 32·32·16 … 1 | ~1.6 MB | Level 1 is the root; levels 2+ hold the restrict/prolong/tail. |
| frontReceipts (webgpu-uniform-velocity-extrapolation.ts:155-162) | D·16 | **16.78 MB** | **Dead.** `cacheConvergence` is true (frontreceipts AB on, nativeFullGrid), but only legacy FIM encode reads it. Binding 17 can be a 16 B buffer. |
| transportA (webgpu-uniform-reference.ts:903) | rgba32f (n+2)³ = 130·130·66 | **17.85 MB** | **Never read or written on mixed.** Only legacy group bindings and `executionExtrapolatedVelocityTexture = present(transportA)` (line 1224). Verify no debug view reads it on mixed. |
| boundaryVelocityC (line 864) | f32 (ny·nz+nx·nz+nx·ny) | 131 KB | Only legacy groups. |
| boundaryVelocityA/B/D | same | 3 × 131 KB | Negative walls for physical / scratch / departure. Needed. |

Format waste:
- Every canonical velocity texel is rgba32f (16 B). An h owner uses x, y, z, which is 12 live bytes.
- A 4h owner uses one channel in each of its 3 anchor texels (origin+3e_a), so 12 live bytes out of 64·16 = 1024 B in each D-texture: 1.2% use.
- That is about 5000·1012 B ≈ 5.1 MB of dead texels in each of the five D-sized velocity textures.
- Because 3D textures are tiled, this costs memory and not bandwidth. Prior result: compacting 4h by itself bought nothing.

---

## 1. Tail: plan.encode (support) + authority.encode(phase) — inside the "extension" window

**What it does.** uniform-mixed-frame.ts:475-482 runs `plan.encode` (seed, then dilate0-2: support planes 0..3, about 5 dispatches plus clears), and then `authority.encode(encoder,group,false)`, one counted "all owners" launch of `build` with `umAuthorityBalance=0`, uniform-mixed-pressure-authority.ts:278/289. This writes `phi[o.index]` and `phase` for the extension's `umSource`.

**Frequency.** Every frame at the tail. After relayout, the head runs the same pair again (uniform-mixed-frame.ts:690-692) on the new layout, and the full balance authority runs before pressure (line 513).

**Data.**
- Reads: volume (r32f 4.19 MB), centerPhi (r32f 4.19 MB), targetFill.
- Writes: phi (root buffer), phase (r32f 4.19 MB).
- One lane per owner, 64-lane groups, 2-4 loads, no chain beyond umOrigin.

**Dead or unneeded.** Nothing dead. The phase is a 1-bit fact (`distance<0 || detached`) stored in an r32f D-texture. The extension (`umSource`, uniform-mixed-extension.ts:153-156) reads it twice per face.

**Proposal (low priority).** Seed could compute the source bit itself (centerPhi<0 at owner and neighbour), which deletes the tail phase launch. The cost is about 1 launch (≈0.03-0.06 ms). Keep it unless fused with something else: the build also writes `phi`, which pressure reads later, so the head and pressure authority would still have to rebuild it. Not ranked.

---

## 2. Extension: seed / sweep×2 / restrictBand / hierarchy / publish (module #17 + module #5)

Encode is at uniform-mixed-extension.ts:747-767. Everything is one compute pass:

| Phase | Dispatches | Shape |
|---|---|---|
| seed (314/320/349/350) | 4 counted | regular h tier, regular-4h list, seam h, seam 4h |
| sweep ×2 (315/321/351/352) | 8 | same 4 tiers |
| restrictBand (140) | 1 dense | (T/4)³ workgroups of 4³ = 16384 lanes, one per tile |
| hierarchy (module #5: 33, 35, 34) | 1 restrict (16·16·8 level, 32 wg×192) + 1 tail (1 wg) + 2 prolong (16·16·8 and the 32·32·16 root, 32 and 256 wg×192) | |
| publish (316/322/353/354) | 4 counted | |

That is **21 dispatches**, plus the about 6 of the tail plan/authority.

### 2.1 Sweep, per face (`umExtended`, umNeighbor at :163, umSlotState at :147)

Regular h path (`umRegularFine`):
- 6 neighbours, each one slot load (vec2f, 8 B) plus one support-word load (`umTileSupport(tile)&2`).
- They issue together; the comment at :151 says "Branch-free: the support word and the slot load together".
- Plus the own state. That is about 13 loads and about 100 B per face, with **chain depth 1**.
- The workgroup also stages 27 neighbour widths and masks (`ueStage`, :330) and builds fast/general queues with workgroup atomics. Per job this costs 2 barriers and 2 workgroup atomics per queued face.

General (seam) path:
- `umNeighbor` reads `umTileWidth` and `umTileMaximumWidth` (topology), then lowWidth/highWidth (2 more topology words, each dependent on the plane decision), then `umVelocitySite → umOwnerAt → umFace → umOwnerAt(probe)`, then up to 4 face slots.
- **Chain depth 4-5**: tile word → width → owner word → neighbour owner word → slot.
- This is the known lead "sweep bound by its dependent topology chain". It is unchanged at HEAD for seam faces.
- The regular path already removed it. That is consistent with seam h + sweepCoarse (1.04 + 0.47 ms) dominating while regular faces are cheap.

Re-loads:
- `umSlotState` loads the support word from global memory for each of the 6 neighbours, although `ueStage` has already staged the 27 neighbour tiles' masks and widths in workgroup memory.
- The six support words come from at most 27 distinct tiles. In the h tier, almost always the job's own tile or 3 face neighbours.

### 2.2 restrictBand (:553-583)

One lane per tile:
- 1 mask load decides `supported`. Unsupported tiles exit after 1 load.
- Then for 3 components × 16 plane anchors, plus 64 fallback anchors for y when gravity-flagged:
  - h tile: 1 slot load per anchor. There are 48 independent loads; the distance compare runs serially per anchor.
  - 4h tile: `umOwnerAt(p) → umPositiveFaceAtAnchor` (topology chain depth 2-3) per anchor, then the slot.
  - **For a 4h tile whose +neighbour is 4h, all 16 anchors resolve to the same width-4 face.** The same slot is reloaded 16 times and `count` accumulates 16 equal values.
- The 16384-lane grid has 64-lane groups, so supported tiles (about 8-10k) each run a serial 48-iteration loop on one lane. Parallelism is low.

Measured cost: 0.24 ms.

### 2.3 Hierarchy (module #5 mixed kernels, webgpu-uniform-velocity-extrapolation.wgsl.ts:943-1080)

- One lane per (cell, component).
- `nearestHierarchySample` makes 8 source loads plus origins.
- The tail is one workgroup that runs levels 8·8·4 → 1 in workgroup memory with barriers.
- 4 dispatches. Prior work already collapsed it from 12 passes. Not a lever.

### 2.4 Publish

- publishFine stages 27 coarse cells and source boxes per tile in workgroup memory, then writes one rgba32f texel per h owner (xyz plus w = physical.w).
- publishList / publishCoarse write 3 texels per 4h owner, one live channel each.

### 2.5 Dead code and storage in module #5 (webgpu-uniform-velocity-extrapolation.ts)

- 18 pipelines are compiled. The per-frame mixed path uses only 33, 34 and 35; pipeline 9 (`clearExtrapolationState`) runs at init.
- **Pipelines 7, 8 and 10-21 (legacy FIM, shell, pack and so on) are compiled and never dispatched.** They cost setup/compile time only.
- frontReceipts: 16.78 MB, dead. Hierarchy level 0: 12.58 MB, dead. Both are in the storage census above.

### Proposals

**E1. extensionSweeps 2→1.** This is parameter-only: `p.extensionSweeps`, uniform-mixed-frame.ts:482/671.
- One sweep ≈ (1.04 + 0.47)/2 ≈ **0.75 ms** of GPU time.
- Effect: faces 2 cells beyond the sources take the hierarchy's nearest-source value instead of a second FIM step.
- Risk: medium. The quality of the extension band changes and it is not ulp-level.
- Verify with the dam and fig-9 shape lanes and a single A/B.
- Effort: trivial. This is the cheapest large experiment in the group.

**E2. Use the staged 27-tile support/mask in umSlotState / umNeighbor** in place of the global `umTileSupport` load.
- This removes 6 global loads per regular face and 6+ per seam face (the support word can be read from `ueMasks`/`ueWidths`).
- It also replaces global `umTileWidth`/`umTileMaximumWidth` with the staged widths in the general path. That removes the first 1-2 links of the depth-4-5 chain for seam faces.
- Estimate: seam h + coarse sweeps are 1.51 ms for 2 sweeps. Cutting 2 of 5 chain links, if latency-bound, gives ≈ 0.2-0.4 ms. Regular faces gain little because their chain depth is already 1.
- Risk: low. Staging already exists, and staged and global values are identical within a frame.
- Verify: CPU naga parse plus capture; a bitwise-identical extension is expected; then one A/B.

**E3. restrictBand: lane per (tile, component)** (3× parallelism; 49152 lanes). For a 4h tile, resolve the +face once: if `umPositiveFaceAtAnchor(o,c,anchor0).width==4`, do one iteration instead of 16.
- Estimate: 0.24 → ≈ 0.10 ms, saving **0.14 ms**.
- Risk: low; the order-preserving tie average must keep z/y/x order (:567).
- Verify: restricted root bitwise equal (the CPU capture cannot check this; one Dawn A/B).

**E4. Fold the source test into seed** (see section 1). Small, not ranked.

---

## 3. Certificate (module with frame-plan: localSpeed, spread0-2, prefix0-2, certify)

See uniform-mixed-frame-plan.ts:136-226.

- **localSpeed**: one lane per owner. For an h tile it loads the origin texel once per axis. That is 3 loads of the same rgba32f texel; the compiler can CSE them. 4h tiles go through `umFace` (chain depth 2) per part.
- **spread**: per tile, a loop over ±reach (memory-independent).
- **prefix**: one lane per tile line, serial over 32/16 tiles.
- **certify**: 8 prefix loads.
- About 8 dispatches. Cheap: part of the 0.45 ms.

**C1 (enabler for M2). Fold fine-sampling support into the certificate.**
- In `prefix0` (:194), count `umTileWidth(t)!=1u || (umTileSupport(t)&1u)==0u` instead of `width!=1`.
- A tile certified regular is then guaranteed that every sample in its reach box has width 1 *and* fine support.
- The support plane is final before the certificate at both the reuse head and the relayout head (plan.encode runs first, :476/690).
- Population risk: support bit0 is "occupied, dilated by settings.x" (seed/dilate, :104-135). h tiles live at the surface, so expect ≈ 0 regular h tiles demoted. The stage view "certificate" reports the class counts.

---

## 4. 4h sampling cache (module #18, `cache`, uniform-mixed-momentum-cache.ts:~68)

- (n/4+2)³ lanes = 34·34·18 = 20808, workgroup 4³.
- Per lane and axis: `umOwnerAt → umFace` (chain 2-3), then the average of up to 16 anchors (16 independent loads).
- Writes 333 KB. Small. It is not a pure duplicate of the hierarchy root: restrictBand averages the 4 *nearest* plane faces, while the cache averages all 16. Not a lever.

---

## 5. Hanging taps + unitFaces (module #19)

See uniform-mixed-momentum-cache.ts:116-174.

- **hanging**: one 256-lane workgroup per slot. Slots are *all seam tiles, h and 4h* (uniform-mixed-ownership.ts:45-46).
  - Lanes 0-191: `umVelocityTap1` at every cell's positive face. This is a general-path site lookup, chain depth 3+ (`umOwnerAt → umFace → umOwnerAt`).
  - Lanes 192-239: domain-boundary plane taps, only where `tileCoord[axis]==0`.
  - Then 64 rgba32f stores into unitVelocity.
- **unitFaces**: one 64-lane job per h tile; it is a pure copy `unitVelocity[cell] = extended[cell]`.
  - About 11000·64 = 704k loads and stores of 16 B, about 22.5 MB of traffic per frame.

Dead or duplicated:
- **For a seam *h* tile, hanging's 192 taps recompute exactly the stored faces.** The source comment at :153 says "a slotted unit tile's fine taps are the same". unitFaces writes the same texels again in the next dispatch, so it is wasted general-path work per seam h tile.
- **unitVelocity duplicates `extended` on every h tile.** It differs only on seam 4h tiles (interpolated unit taps) and the plane taps. Readers:
  - momentum unit/deferred paths
  - forces viscosity (umLaplacian reads unitVelocity or the cache)
  - surface advect's unit sampler (surfaceFields.unitVelocity, uniform-mixed-frame.ts:286)

### Proposals

**H1. Skip lanes 0-191 in `hanging` for unit (h) slot tiles.** Add `if(umTileWidth(tile)==1u && lane<192u) {}` and store nothing from lanes <64 for those tiles. Lanes ≥192 (plane taps) still run.
- Estimate: if about 2-3k seam h tiles × 192 general taps at chain depth about 3, that is about 0.05-0.1 ms.
- Risk: very low; the result is bitwise identical, since the same values come from unitFaces.
- Effort: 2 lines.

**H2. Fuse unitFaces into localSpeed** (the certificate's per-owner lane already loads `extended` at each h origin texel).
- `textureStore(unitVelocity, origin, texel)` there deletes the unitFaces dispatch and its 704k loads.
- It needs unitVelocity bound in the plan module and ownership moved to the frame.
- Ordering is fine: localSpeed runs before hanging, and hanging overwrites seam h with identical values (or skips them under H1).
- Estimate: 0.05-0.1 ms plus 1 launch.
- Risk: low; bitwise identical.

**H3. (Bigger, medium risk) Delete unitVelocity (16.78 MB).**
- Write the seam-4h hanging taps into velocityB's *unused* texels of those 4h tiles; 61 of 64 per component are free.
- The unit samplers then read `extended` directly, and unitFaces disappears.
- Hazards:
  - The tap at a 4h anchor texel (origin+3e_a, channel a) must equal the stored patch value. It does when the +neighbour is h (unit patches); when the +neighbour is 4h, check `umVelocityTap1` on a width-4 patch.
  - In-dispatch read/write of the same texture needs a second pass (taps to workgroup or a buffer, then store), because read_write storage textures bypass the texture cache on Apple (remap note).
  - Every reader of velocityB in a 4h tile goes through anchors, but certificate, cache, remap and census must be audited.
- Payoff: 16.78 MB, plus about 0.1 ms, plus one texture binding fewer in the momentum, forces and surface groups.
- Rank it below H1/H2, which get most of the time.

---

## 6. Momentum advection (module #26, uniform-mixed-momentum.ts)

**Dispatch.** Four claimed launches (:321). Each has grid min(2048, tiles) and a 192-lane workgroup; a lane is (cell, axis) and computes one component:
1. regular (`momentumRegularStep`, umPlannedFine=1, umRegularFine): the certified list 1
2. unit (`momentumUnitStep`): list 2
3. deferred
4. merged `momentumStep`: seam 4h quads plus regular 4h, packed

### Per-lane access (regular h face, the dominant population)

1. **Cull** (`umCullAir`/`umPredictionCellLive`, :256): volume and centerPhi of the owner and neighbour. **4 loads**, depth 1. Faces with both sides air exit.
2. **Departure**, RK2 with `uniformVelocityDepartureWGSL` (uniform-velocity-departure.wgsl.ts:3-14). Substeps of at most 1.5 cells. Each substep makes **2 full-vector samples**: `first` at `point`, then the midpoint.
3. Each `umSampleVelocity(p)` (uniform-mixed-velocity-sampling.wgsl.ts:221):
   - `umVelocitySamplingWeights` (:181) **loads the support word** (`umTileSupport(umTileAt(tile))&1`, :184; `umTileAt` is arithmetic, so this is 1 load). Under umRegularFine it then returns 1.0 or 0.0.
   - `umSampleVelocityWeighted` (:202-203) then branches: `fine>0 ? umSampleVelocity1 : umSampleVelocity4`. **The 24 tap loads are control-dependent on the support load.**
   - `umSampleVelocity1`, regular branch (:142-150): 8 `textureLoad(extended)` of rgba32f per component, using 1 channel each. That is **24 loads, 384 B fetched, 96 B live** per vector sample. Zero-weight taps are still loaded (the known lead).
4. **Final** `umSampleVelocityComponent`: 8 loads (128 B fetched, 32 B live), also behind a support load.

Per face with 1 substep: 4 + 2·(1+24) + (1+8) = **63 loads, about 920 B fetched, about 250 B live**.
- At fig-9 speeds up to about 9 m/s: 9·(1/30)/0.05 = 6 cells, so up to ⌈6/1.5⌉ = 4 substeps for the fastest faces, giving 4 + 8·25 + 9 = 213 loads.
- **Dependent round trips per face (1 substep): cull → support → taps (first) → support → taps (midpoint) → support → taps (final) = 7.** 4 more per extra substep.
- Under umRegularFine, `cellWidth` is `select(f32(umOwnerAt(...).width),1.0,umRegularFine)` (:242). With the override folded to a constant, the load should be dead-code-eliminated; it was not confirmed in the MSL.

Unit and deferred paths (general):
- `umFineStencilSample`: tile word + stencil + support, then a `firstTrailingBit` loop over up to 27 stencil bits.
- `umUnitTaps`: 8 tile-width loads.
- `umVelocityTap1` via `umVelocitySite`: `umOwnerAt → umFace`, plus `umHanging[tile]` → unitVelocity.
- About 3-4 more dependent levels per sample than the regular path.

Population (h faces): about 2.11 M h faces at peak, before the air cull. Approximate load volume ≈ 1-2 M live faces × 63 ≈ 60-130 M texel loads per frame for about 1.9 ms. This is consistent with latency/issue-bound behaviour, not bandwidth (130 M × 16 B = 2 GB would be about 1 TB/s if it were really fetched from DRAM; it hits the texture cache).

### Proposals

**M1. Exact face-centre first sample.** The first `first` sample of a face's departure is at the face centre, where every trilinear fraction is 0 or ½:
- own component a: 1 tap (the face itself, which `umMomentum` already has or can load)
- each transverse component: 4 taps with weight ¼

Specialise substep 0's `first` under umRegularFine. **24 → 9 loads, and the floor/fract/weights math disappears.**
- Per face (1 substep): 63 → 48 loads (−24%). Chain depth is unchanged (still one round trip).
- Estimate: regular momentum ≈ 1.2-1.4 ms of the 1.95; 24% fewer loads and registers gives ≈ **0.2-0.3 ms**.
- Results differ only at the ulp level (exact ½/¼ weights against `1-f`/`f` products).
- Risk: low.
- Verify: CPU WGSL capture, then a momentum-only A/B.
- The same idea for the surface group: a cell-centre sample needs 2 taps per component, 6 instead of 24 (traceCells). Flag this to the level-set group.

**M2. Drop the per-sample support load and coarse branch in regular samplers (needs C1).**
- Under umRegularFine with C1, `umVelocitySamplingWeights` returns the constant 1.0. The `coarseCache` support test and the `umSampleVelocity4` branch (8 more taps and their registers) become dead code.
- **Chain per face: 7 → 4 round trips** (cull, taps, taps, taps), and 3 fewer loads.
- Estimate: regular momentum is latency-chain bound. Removing 3 of 7 serial round trips is up to 40% of the regular path's latency; with occupancy hiding half, ≈ **0.25-0.5 ms**.
- The same sampler source (`uniformMixedVelocitySamplingSource`) is used by surface advect and traceCells regular paths, which are other groups. They also gain if they run certified.
- Risk: low. The semantics are exact, since tiles that would have hit the coarse branch become general.
- Verify: CPU capture (check the support load is gone from the regular entry); compare certificate counts in the stage view; one A/B.

**M3. Hardware-filtered trace-only velocity copy (the allowed f16 exception, with a new argument).**
- The prior rejection was f16 for *canonical* fields. This is a trace-only copy, the allowed exception. The new argument is that **a filterable rgba16float 3D texture lets the sampler unit do the trilinear: one `textureSampleLevel` per component replaces 8 `textureLoad`s.**
- unitFaces and publish already write every h texel, so they also write `trace16[cell] = vec4h(extended[cell])`. That is +8 MB, but it replaces unitVelocity if H3 lands.
- In the regular RK2 (not the final value, which must stay f32), each vector sample becomes 3 filtered fetches instead of 24 loads. Per face (1 substep): 63 → 4 + 2·3 + 8 = 18 loads.
- Coordinates: q is clamped exactly as now; `texcoord = (q + 0.5)/dims` with clamp-to-edge. The `base[axis] < 0` (negative wall) case keeps the exact path.
- Accuracy:
  - f16 relative precision is 2⁻¹¹. At 9 m/s that is 0.0044 m/s, giving a departure error of 0.0044·(1/30)/0.05 = 0.003 cells.
  - Apple's linear-filter weight quantisation (8-bit sub-texel is typical) gives up to (1/256)·|Δu| per sample.
  - This is **not ulp-level**. It needs Peter's sign-off and quality lanes.
- Estimate: trace taps are 48 of 63 loads. Regular momentum ≈ 1.3 ms × (45/63 load reduction) × 0.5 latency-hiding discount ≈ **0.45 ms**. Larger again if surface advect / traceCells adopt it (other group, Vertex phi 3.65 ms).
- Effort: medium (new texture, sampler binding, `float16` not needed because the storage is rgba16float).

**M4. Substep CFL (1.5 cells) is an accuracy choice.** Not proposed.

---

## 7. Forces (module #27, uniform-mixed-forces.ts)

**Dispatch.** 3 used launches (:258-260): `forcesRegular` (h tier), `forcesRegularCoarse` (regular-4h list, far-air page check) and fused `forces`. Each lane makes about 10 loads:
- 1 advected (departure)
- 2-4 occupancy (centerPhi)
- `umLaplacian` (:171): 7 one-channel rgba32f site loads from unitVelocity or the cache
- capillarity is off (tension is 0)

Measured ≈ 0.18-0.25 ms.

**Dead.**
- 18 pipelines compiled, 3 used. The `umForceCached=1` set (364-369) and cacheNormals/cacheCurvature (198-201) are never dispatched.
- The `normals` binding (arena offset 0, 16·D) is bound but unread under inline curvature. It is arena-backed, so no allocation.
- Setup time only.

**Proposals.** Drop the cached-curvature variants from compilation when `cachedCurvature` is false (setup only). Nothing frame-relevant.

---

## 8. Authority before pressure (uniform-mixed-pressure-authority.ts)

`build` (balance), `chunks`, `reduce` (1 wg) and `resolve` run once per frame (:289-292). Per owner:
- 2-4 loads; no chain on the h path
- workgroup reduce with 6 barriers per 64
- one 1-workgroup reduce dispatch, which is a serial tail (≈ launch-bound)

It rebuilds `phi` and `phase` already built at the head (line 692). Both are needed: volume and centerPhi changed in between. No velocity lever; this belongs to the pressure group.

---

## 9. Module #1 "Uniform reference kernels" (254 KB) — init only

webgpu-uniform-reference.ts:1554-1624.
- With `geometricVolume` it compiles only `buildDenseExtrapolationAuthority` (pipeline 3) and `uvPublish` (pipeline 4).
- Both are dispatched **once**, in `encodeInitialPresentationSurface`.

### 9.1 Register allocation and dead code

- Dawn/Tint runs a single-entry-point prune before MSL generation, so the Metal compiler and register allocator see only each entry point's call graph. **The 254 KB does not affect frame time or occupancy.**
- It costs WGSL parse, resolve and validate on `createShaderModule`, **twice**: the same `this.shaderSource` is prepended to the CM11a pressure module (#6, line ~1605, `${this.shaderSource}\n…`). That is why #6 is 333 KB and contains `buildDenseExtrapolationAuthority`/`uvPublish` again (001.wgsl:1056, 006.wgsl:1064/4259).
- This is part of the 109 s setup. Check whether #6 references any helper from it (pressure group).

### 9.2 Init outputs and frame-time impact

- **buildDenseExtrapolationAuthority writes surfaceA (= mixed `phase`) and velocityD (= `departure`).** On the mixed path:
  - `phase` is rewritten by `authority.encode(phase)` before the first extension (uniform-mixed-frame.ts:388).
  - `departure` is rewritten by traceCells and momentum before any read.
  - So its outputs are dead. Verify that `uvPublish` doesn't read them.
- `uvPublish` writes centerPhi and target/openFraction for the t=0 presentation. The geometry stage rewrites both every frame.
- Proposal: on the geometric path, skip pipeline 3 and compile `uvPublish` from a small standalone module. Stop prepending `shaderSource` into #6 if #6 doesn't need it.
- Gain: setup and memory only, and compile time of one 254 KB module (plus the prepended copy). **0 ms frame time.**

### 9.3 Remap staging alias

The remapped extension texture can alias velocityD:
- At remap time (relayout head, uniform-mixed-frame.ts:675), velocityD holds last frame's momentum output, which forces already consumed.
- The next writer is traceCells after the remap, so it is dead there.
- Pass `{velocity: f.departure, negative: f.negativeDeparture}` to `remap.bindExtension` (:260-261) in place of the owned texture.
- Saves **16.78 MB + 131 KB**, 0 ms.
- Risk: low. Audit that nothing between the tail and the head remap reads velocityD; relayout/census do not bind it (grep shows departure only in surface, transport, momentum, forces, dynamic and layout).

---

## Bytes and loads per face (fig-9, h face, regular class)

| Stage | Loads/face | Fetched B | Live B | Dependent depth |
|---|---|---|---|---|
| extension seed | ~4 (physical, 2× phase, mask) | ~40 | ~16 | 1 |
| extension sweep (×2), regular | ~13 (6× slot 8 B + 6× support 4 B + own) | ~80 | ~56 | 1 |
| extension sweep, seam h | 6 neighbours × (2-4 topology + 1-4 slots) ≈ 30-40 | ~250 | ~60 | **4-5** |
| restrictBand (per tile) | 1 + 48 slot loads (+16× chains for 4h) | ~400 | ~400 | 1 (h), 3 (4h) |
| publish | ~3-6 + staged 27 | ~60 | ~20 | 1-2 |
| localSpeed + unitFaces | 3 (CSE→1) + 1 copy | 32 + 16 store | 12 | 1 |
| momentum, 1 substep | **63** | **~920** | ~250 | **7** |
| momentum, 4 substeps | 213 | ~3.3 KB | ~800 | 16 |
| forces | ~10 | ~130 | ~40 | 2 |

After the proposals (M1+M2), momentum with 1 substep is 4 + 9 + 24 + 8 = **45 loads, depth 4**. With M3 it is 4 + 3 + 3 + 8 = **18 loads, depth 4**.

---

## RANKED experiment list (fig-9, gain = GPU ms per frame)

| # | Experiment | Est. gain | Effort | Risk | Verification |
|---|---|---|---|---|---|
| 1 | **M2 + C1**: certificate also requires fine support (prefix0 counts `width!=1 ‖ !(support&1)`); regular samplers drop the support load and coarse branch. Chain 7→4 round trips per face. Surface advect and traceCells regular samplers also gain. | 0.25-0.5 (momentum) + more in surface | small (2 edits: frame-plan.ts:194, sampling.wgsl.ts:184/203 under umRegularFine) | low (exact) | capture WGSL; stage-view certificate counts; 1 A/B |
| 2 | **E1**: `extensionSweeps` 2→1 | ≈0.75 (half of 1.51) | trivial (param) | medium (quality) | dam + fig-9 shape lanes, 1 A/B |
| 3 | **M3**: rgba16float filtered trace copy for RK2 samples (final stays f32) | ≈0.45 (momentum), more if surface adopts | medium | medium (not ulp; needs Peter's OK) | quality lanes + A/B |
| 4 | **M1**: exact face-centre first sample (9 taps instead of 24) | 0.2-0.3 | small | low (ulp) | capture + A/B |
| 5 | **E2**: staged support/width in umSlotState/umNeighbor (cut 2 links of seam chain, 6 global loads per face) | 0.2-0.4 | small-medium | low (bitwise) | capture + A/B |
| 6 | **E3**: restrictBand lane per (tile, component) + single face for 4h/4h patches | ≈0.14 | small | low | A/B |
| 7 | **H1**: hanging skips 192 redundant taps on seam h tiles | 0.05-0.1 | 2 lines | very low (bitwise) | A/B (batch with 8) |
| 8 | **H2**: unitFaces copy fused into localSpeed (delete a dispatch and 704k loads) | 0.05-0.1 | small | low (bitwise) | A/B (batch with 7) |
| 9 | **H3**: delete unitVelocity; seam-4h taps into velocityB's free 4h texels | ~0.1 + 16.78 MB | medium-high | medium (anchor collision audit) | lanes + A/B |
| 10 | Setup hygiene: skip module-#1 dense authority, standalone uvPublish, drop the shaderSource prepend into #6, stop compiling legacy #5 pipelines 7, 8, 10-21 and forces' 15 unused variants | 0 ms frame; setup seconds | small | low | setup_ms in profile |

Suggested batching, at most 3 probe runs:
- Run A: 1+4+7+8 (exact or ulp-level).
- Run B: 5+6.
- Run C: 2 and 3 separately as quality-gated arms.

---

## Storage that can be deleted (fig-9)

| Item | Bytes | Where | Notes |
|---|---|---|---|
| frontReceipts | 16,777,216 | webgpu-uniform-velocity-extrapolation.ts:155-162 | Legacy FIM only; keep a 16 B buffer for binding 17 |
| transportA | 17,846,400 | webgpu-uniform-reference.ts:903 | Never read or written on mixed; check the `executionExtrapolatedVelocityTexture` view (line 1224) |
| hierarchy level 0 (64×64×32 down/up + origins) | 12,582,912 | webgpu-uniform-velocity-extrapolation.ts:330-350 | Mixed starts at level 1 |
| "Uniform mixed remapped extension" + walls | 16,777,216 + 131,072 | uniform-mixed-frame.ts:260-261 | Alias velocityD / negativeDeparture |
| unitVelocity | 16,777,216 | uniform-mixed-momentum-cache.ts:102 | Only with H3 (duplicates velocityB on h tiles) |
| boundaryVelocityC | 131,072 | webgpu-uniform-reference.ts:864 | Legacy groups only |
| **Total** | **≈ 64.1 MB without H3, ≈ 80.9 MB with H3** | | allocatedBytes is 386 MB → about 17-21% |
