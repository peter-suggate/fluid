# Static micro-architectural audit: Uniform Geometric volume shaders

Static read only. No GPU tool was run, no source file was modified, no git command
was issued. Every number below is either counted from the code (marked **V** for
verified, with `file:line`) or derived from those counts (marked **I** for
inference). Target: Apple M1 Max, 32 GPU cores, SIMD width 32, unified memory
~400 GB/s, WebGPU → Dawn → Tint → MSL.

Files audited: `lib/methods/uniform/uniform-volume.wgsl.ts` (803 lines, read
as-is, another session holds uncommitted edits),
`uniform-volume-donor-sum.wgsl.ts`, `uniform-volume-pages.wgsl.ts`,
`uniform-page-stencil.wgsl.ts`, `uniform-surface-volume.wgsl.ts`, and the host
`webgpu-uniform-reference.ts`, `webgpu-uniform-surface-volume.ts`,
`uniform-texture-pages.ts`, `uniform-field-loop-bounds.ts`.

## 0. Which configuration is actually live (V)

`uniformGeometricSolverOptions` sets `volumePages: 32` and `pageDomain: true`,
but the initial page domain is a complete rectangular grid, so
`uniformPageHasRectangularCoverage` → `UniformTexturePages.nativeStorage` →
`nativeRootExecution` (`webgpu-uniform-reference.ts:780-806`) →
`volumePageConfig.nativeRecords` (`:960-987`). With `nativeRecords`,
`uvEdgeAddress` is the identity (`uniform-volume-pages.wgsl.ts:34-35`) and the
record arena is dense `nx*ny*nz*80` (`webgpu-uniform-reference.ts:876-889`).

**Published docs describing a compacted ~78 MB paged record high-water are stale
at HEAD**: garden 144×96×96 allocates 106,168,320 B of records unconditionally.

`work` (the compact 4³ tile list) is enabled only when `max(nx,ny,nz) > 64`
(`webgpu-uniform-reference.ts:974`). **At 64³ and below the tile work list, the
tile-cached sharpening arm and the indirect dispatch are all off, yet the four
page bookkeeping passes still run.** mini64 — the gate lane — therefore runs the
*expensive* uncached sharpening arm.

## 1. Entry-point census — `uniform-volume.wgsl.ts`

All 25 `@compute` entries. "Hot" = interface cell on the live path; "cold" = the
early-out. Loads are **issued load instructions**, not DRAM bytes (see §7).
`uvOpen` (`:296`) = `cellOpenFraction` = 10 buffer loads + 0-1 terrain texel
(`webgpu-uniform-reference.wgsl.ts:364-367,253-261,686-691`); `uvPhi` (`:73`) =
8 texel loads.

| Entry | Line | WG | Dispatch | Hot loads / stores | Cold | Stencil | Notes |
|---|---:|---|---|---|---|---|---|
| `uvAgreementResidual` | 230 | 4,4,4 | 3D dense, direct | 16 texel + 1 st | — | 2×8 vtx | diagnostic only |
| `uvCorrectionCapacity` | 46 | 4,4,4 | 3D dense | ~20 | — | 1 cell | |
| `uvCorrectionTargets` | 50 | 4,4,4 | 3D dense | ~20 | — | 1 cell | |
| `uvAdvectPhi` | 272 | 4,4,4 | 3D vertex, direct | `uvTrace` ≈ 48 vel + 6 atomic + ~22 open + 8 phi ≈ **84**, 1 st | n/a (no early-out) | backward ray, CFL-bounded | two dead guarded blocks (`:280`, `:281`) |
| `uvRedistancePhi` | 284 | 4,4,4 | 3D vertex, direct | band: ≤8 Newton iters × (48 phi taps + grad) ≈ **460 texel**; `break` at `norm<1e-16` | out of band: 1 ld 1 st | 3×3×3 vtx per iter | data-dependent iteration count |
| `uvBuildEdges` | 332 | 4,4,4 | 3D or work-list, indirect | 1 tile atomic + `uvTrace` 84 + **24 `uvOpen` = 264** + 34 st + 8-16 atomicAdd ≈ **350 ld** | 1 atomic | 2×2×2 donors | writes all 80 B |
| `uvFinishDonorSums` | 345 | 4,4,4 | **3D dense ×4** | 6 atomicLoad + 1 st | none | none | not window- or work-restricted |
| `uvFallback` | 350 | 4,4,4 | 3D/work | 8 + `uvOpen` ×4 ≈ 52 | 1 atomic | 1 cell | |
| `uvNormalizeRows` | 356 | 4,4,4 | 3D/work **×3** | 18 rec ld/st + 9 `uvAddDonor` (9-18 atomicAdd) | 1 atomic | row of 9 | |
| `uvNormalizeDonors` | 365 | 4,4,4 | 3D/work **×3** | 9 rec ld + **9 scattered atomicLoad** + 9 st | 1 atomic | 9 scattered donors | worst gather pattern in the file |
| `uvGather` | 388 | 4,4,4 | 3D/work, indirect | 1 atomic + 9×(2 rec + 1 vol + `uvCell`) + 2 `uvOpen` + `uvTarget` (**64 phi taps** + 11) ≈ **125-180 ld**, 2 st | 1 atomic + 2 st | 9 donors + 8 vtx | 27 runtime div/mod |
| `uvClassifySharpenTiles` | 450 | 4,4,4 | 3D dense | 8 phi + 2 barriers + 1 wg atomic; lane 0: 1 store + 1 **global contended** atomicAdd | — | 8 vtx | compaction block (~114 ld) compiled, never taken |
| `uvCacheSharpenCells` | 477 | 4,4,4 | work-list only | `uvPhi` 8 + `uvOpen` 11 + 2 st | — | 1 cell | **not encoded at ≤64³** |
| `uvCacheSharpenFaces` | 485 | 4,4,4 | work-list only | 3×(`faceOpenFraction` ≈ 30) + 3 st | — | ±1 per axis | **not encoded at ≤64³** |
| `uvPrepareSharpen` | 505 | 4,4,4 | work/dense **×8** | 4 ld + 3 st | tile-inactive: 0 | 1 cell | |
| `uvProposeSharpen` | 529 | 4,4,4 | work/dense **×8** | cached arm ≈ **15 ld**, 3 st; uncached arm ≈ **225 ld** | tile-inactive: 0 | ±1 per axis | arm chosen by `uvPageWorkEnabled()` — a *compile-time* const |
| `uvLimitSharpen` | 567 | 4,4,4 | work/dense **×8** | 4 tile tests (24 activeRegion + 4 atomic) + 8 rec + 1 vec2 st | 0 | ±1 per axis | |
| `uvCommitSharpen` | 583 | 4,4,4 | work/dense **×8** | 6×`uvLimitedFlux` = **36 runtime div + 72 activeRegion ld + 12 tile atomic + 18 rec ld**, +1 vol ld, 1 st ≈ **95 ld / 13 atomic** | inactive: 2 ld 1 st | ±1 per axis | for four adds |
| `uvTwoLevelSeed` | 651 | 4,4,4 | 3D over **tiles** | per thread: 64 cells × ~14 + **125 phi texels** ≈ **1020 ld**; 1 atomicMax + 1 atomicStore | — | 4³ cells + 5³ vtx | one thread per *tile*: 189 serial iterations |
| `uvTwoLevelDilateX/Y/Z` | 731/736/741 | 4,4,4 | 3D over tiles | `-r..r` atomicLoads, r ≤ 16 → ≤ **33 atomic** | — | 1D line | data-dependent bound from an atomic |
| `uvPublish` | 750 | 4,4,4 | 3D dense | 8 phi + 11 open + 2 st | — | 8 vtx | publishes γ that `uvOpen` recomputes everywhere else |
| `uvBalanceMeasure` | 774 | 4,4,4 | 3D dense | `pressurePhi` + `uvSurfaceDeficit` ≈ 40 ld, wg reduce | — | 8 vtx | |
| `uvBalanceReduce` | 793 | 64 | 1 group | **serial loop over every partial** (324 at garden) | — | — | 63/64 lanes idle |

`uniform-volume-pages.wgsl.ts`: `uvMarkTransportPages` (`:43`) and
`uvMarkSharpenPages` (`:52`) dispatch `@workgroup_size(4,4,4)` over the whole
domain then `if(any(gid%4u!=0u)){return;}` — **63 of 64 lanes retire
immediately** (`:45`, `:54`); `uvCompactPages` (`:63`) is `@workgroup_size(1)`
with a serial loop over `UV_PAGE_COUNT` (`:71`).

`uniform-page-stencil.wgsl.ts` is a research probe used only by
`tools/benchmark-uniform-pages-dawn.ts`; it is **not** in the production advance.

`uniform-surface-volume.wgsl.ts`: 8 entries, `begin`@1 and seven @64;
`webgpu-uniform-surface-volume.ts:encode()` issues **14 passes plus a full
(n+1)³ texture copy per advance**.

## 2. Function-scope and workgroup arrays (V)

| Array | Site | Bytes | Dynamically indexed? |
|---|---|---:|---|
| `values:array<f32,8>` | `:76` (`uvPhi`) | 32 | literal loop — unrolled, **unless** the phi-redistance module's loop-bound rewrite (`uniform-field-loop-bounds.ts:1-20`) turns `8` into `uniformFieldPages[35][…]`, which forces a spill |
| `samples:array<f32,8>` | `:409` (`uvTarget`) | 32 | literal |
| `terms:array<f32,6>` | `:589` (`uvCommitSharpen`) | 24 | literal |
| `terms:array<f32,8>` | `:632` (`uvCoarseVelocityComponent`) | 32 | literal |
| `words:array<u32,6>` | donor-sum `:32` | 24 | literal |
| `uvTileAdmission:atomic<u32>` | `:449` | 4 | workgroup |
| `uvBalanceSums:array<vec2f,64>` | `:767` | **512** | workgroup |
| `sums:array<vec4f,320>` | surface-volume `:15` | **5,120** | workgroup — the occupancy risk |
| `raw/scale:array<f32,8>`, `result:array<vec4f,5>`, `v:array<f32,8>` | surface-volume `:65-76` | 32/32/80/32 | inside a 17-iteration sample loop |

No `array<f32,27>`. The one real occupancy hazard is the 5,120 B `sums` in
`measure`/`reduce`: at 5 KB/workgroup an M1 Max core's 32 KB threadgroup budget
admits ~6 groups, versus 32+ for the volume kernels (**I**).

## 3. The 80-byte record and every persistent field

`struct UVEdges { donor:array<u32,9>, weight:array<f32,9>, padding:vec2f }`
(`:43`) = 36 + 36 + 8 = **80 B exactly**, align 8, **no internal padding**; the
`vec2f` is not padding in practice — `uvLimitSharpen:567-577` writes it and
`uvLimitedFlux:578` reads it.

| Field | Phase | Read | Written | Recomputable? |
|---|---|---|---|---|
| `donor[0..8]` (36 B) | transport | gather, normalizeDonors | buildEdges, fallback | **Yes** — every donor is `base + corner` of the traced cell (`:334-344`); one packed base index + 9 implicit corners, or 3 bits/axis of CFL-bounded relative offset |
| `weight[0..8]` (36 B) | transport | gather, normalizeRows | buildEdges | partly: trilinear weights are a product of 3 fractions → **12 B** (3 f32 fractions) reproduces all 9 |
| `weight[5]`,`weight[6]`,`donor[4]`,`donor[0..2]` | sharpening | propose | cacheSharpenCells/Faces (`:477-504`) | reused slots — the record is a union of two disjoint lifetimes |
| `padding.xy` | sharpening | limitedFlux | limitSharpen | live |

**Minimal layout (I).** Transport phase: 1×u32 base + 3×f16 fractions =
**10 B** (vs 80) if the trilinear form is kept; 9 f16 weights + packed base =
**22 B** if arbitrary weights must survive normalization. Sharpening phase needs
φ, γ, open and three face flags + two limiter scalars ≈ **12 B** as a separate
SoA block. The two phases never overlap, so a hot/cold split lets the sharpening
rounds stream 12 B/cell instead of striding an 80 B AoS record — **at 80 B a
128 B cache line holds 1.6 records, so every sharpening load pulls ~68 wasted
bytes** (V: 80 B; I: line utilization).

Other persistent per-cell state (V, `webgpu-uniform-reference.ts:876-889`,
`uniform-host-allocation.ts:49`):

| Field | B/cell | Plausible min | 64³ total | garden total |
|---|---:|---:|---:|---:|
| records | 80 | 10-22 | 20.97 MB | 106.17 MB |
| donor sums (6 limbs) | 24 | 8 (f32 + 1 guard limb) | 6.29 MB | 31.85 MB |
| conditioning scratch (3 words) | 12 | 12 | 3.15 MB | 15.93 MB |
| volume / volumeB (r32f ×2) | 8 | 4 (f16 pair) | 2.10 MB | 10.62 MB |
| γ / γB | 8 | 2 (unorm8 ×2) | 2.10 MB | 10.62 MB |
| vertex φ in/out | 8/vtx | 4 | 2.20 MB | 11.14 MB |
| velocity (3 × r32f, MAC) | 12 | 6 | 3.15 MB | 15.93 MB |
| **per-cell total** | **152** | **~46** | | |

**Bytes touched per advance at 64³ (I, from the pass census):** compulsory
unique traffic ≈ **870 MB** — records 212 MB (write once, re-read 7×), sharpening
461 MB across 8 rounds × 4 passes, donor sums 87 MB, textures ~110 MB. At
400 GB/s that is ~2.2 ms against a ~66 ms mini64 advance. **The volume stage is
not bandwidth bound; it is load-issue, latency and atomic bound** — consistent
with the standing "rank by loads removed" finding. Issued loads tell the real
story: ~4 × 10⁹ load instructions per advance at 64³, over half of them in the
uncached `uvProposeSharpen` and `uvCommitSharpen`.

## 4. Gather vs scatter (V)

Transport is **gather with a scatter side-channel**. `uvBuildEdges:332-344`
scatters: it traces backward, writes its own 9 donor/weight pairs, and calls
`uvAddDonor` 8× to atomically accumulate each donor's outgoing total into
`rigidExchange` (donor-sum `:9-29`, 1-2 `atomicAdd` plus a bounded carry loop).
`uvGather:388-407` then gathers 9 donor volumes. Donor probes per receiver:
**exactly 9, every one a scattered `textureLoad(volume, uvCell(donor))` with 3
runtime divisions to rebuild the coordinate** (`:394-397`).

Wasted loads: the trace is CFL-bounded, so all 9 donors lie in one 2×2×2 block
plus the fallback centre — the coordinates are `base + corner` and **8 of the 9
`uvCell` calls, i.e. 24 integer divisions, recompute what the first one already
established**. `uvNormalizeDonors:365-372` re-reads the same 9 donor totals
through the *atomic* binding, which is both a scattered access and an
FP-reassociation barrier.

## 5. Paging with `pageDomain: true` (V)

Because `nativeRecords` is true, `uvEdgeAddress` is the identity and **paging
adds zero indirection loads per neighbour access**. What survives:

1. Four bookkeeping passes per advance — `uvMarkTransportPages`,
   `uvMarkSharpenPages` (63/64 lanes dead), `uvCompactPages` (`@workgroup_size(1)`
   serial), plus the work-list append — whose only consumers are the tile work
   list and a diagnostics readback.
2. `dims()` is a uniform-buffer load, not a constant: `uniform-texture-pages.ts:55`
   rewrites **every** `textureDimensions(X)` to `uniformFieldPages[binding].xyz`.
   Tint cannot constant-fold the `%`/`/` in `uvCell:72` or `uvWorkId:23-29`, and
   cannot fold robustness clamps.
3. There is no per-page halo and no per-sample page resolve in the production
   path. `uniform-page-stencil.wgsl.ts:neighbor()` — which *does* resolve a page
   per boundary sample — is research-only.

So ≤64³ takes a dense, identity-mapped single-submission path that **still pays
the bookkeeping passes and the runtime `dims()`**, and additionally loses the
cached sharpening arm.

## 6. Classification, compaction, indirect args (V)

Per advance the geometric volume stage issues (`webgpu-uniform-reference.ts:2483-2548`):

| Category | Passes | Notes |
|---|---:|---|
| physics | 1 buildEdges, 1 fallback, 3 normalizeRows, 3 normalizeDonors, 1 gather, 32 sharpening (8×4), 1 publish | 42 |
| bookkeeping | 4 two-level, 4 page mark/compact, 1 classify, 2 cache, 4 finishDonorSums, ~4 clears, ~3 copyField | **22** |
| phi | advect + redistance + 3 phi-region passes | 5 |
| surface volume | 14 + 1 texture copy | 15 |

**~34% of passes are bookkeeping.** At the ~13 µs per-firing floor that is
~0.29 ms of pure launch overhead, and indirect dispatch costs ~3× direct on the
CPU side. Fusion candidates: `uvClassifySharpenTiles` (dense, 8 phi taps) does
exactly what `uvGather` already does per cell; `uvMarkTransportPages` visits the
same tiles as `uvTwoLevelDilateZ`; the donor-sum clears can be folded into
`uvNormalizeDonors`'s existing write; the ping-pong `copyField` calls
(`:1684-1687`) are full-texture copies where a binding swap would do.
`uvFinishDonorSums` runs **dense ×4** while every consumer is work-restricted.

## 7. Tint / MSL hazards (V unless noted)

- **Runtime-sized arrays everywhere** (`array<UVEdges>`, `array<atomic<i32>>`):
  with Chrome robustness on, every `uvEdges[...]` and `sharpenDeposits[...]`
  access carries a clamp against a runtime length. Combined with runtime
  `dims()`, none of these fold.
- **Non-constant div/mod**: `uvCell:72` (3), `uvWorkId:23-29` (3),
  `uvPageIndex:30-33`, `uvSharpenTileIndex:436`. `dims()` is a uniform load, not
  `override`/`const`. `uvGather` alone issues 27 runtime divisions per
  invocation; `uvCommitSharpen` issues ~39.
- **Uniform-guarded dead code** (compiled, never executed at geometric shipping
  defaults): `uvAgreementShift:248-258` — a **512-iteration triple loop with a
  textureLoad**, guarded by `params.agreement.z>0.0` at `:280`;
  `uvSeedPhi:264-271` — 64 iterations × 8 taps, guarded at `:281`; the
  compaction block in `uvClassifySharpenTiles` (~114 loads);
  `uvEmbeddedContact:201-220` — up to 32 `cellOpenFraction` calls on the common
  no-solid path; the rigid-body loop in `cellSolidFraction`. The only `override`
  in the file is `UV_SHARPEN_TILE_WORK:bool=false` (`:433`). Register pressure
  is set by the union of paths, so the 512-iteration loop prices `uvAdvectPhi`
  even when never taken (**I**, but this exact mechanism is already documented
  in this repo).
- **Atomic bindings read as data**: `uvTransportSkip:328`, `uvSharpenTileActive:444`,
  `uvCoarseVelocityComponent:628-639` (8 `atomicLoad`), `uvNormalizeDonors` (9),
  `volumeCorrectionDivergence` (`webgpu-uniform-reference.wgsl.ts:1256-1258`
  bitcasts an `atomicLoad` to f32). Each is an FP-reassociation barrier and
  bypasses the normal read path.
- **Packing waste**: `padding:vec2f` forces 8 B align on an otherwise 72 B
  struct; `uvBalanceSums:array<vec2f,64>` and surface-volume's
  `array<vec4f,320>` are vec-padded.
- **f32↔u32 churn**: the donor-sum encode/decode does 6 `bitcast` per limb pair
  plus guard/sticky rounding (`donor-sum:9-50`); the tile maps store i32 and are
  bitcast to f32 at every read.
- **Hard constraint on fixes**: the main layout already binds **10 storage
  buffers** (9,10,11,19,26,27,28,29,30,33) and the code comments the limit
  (`webgpu-uniform-reference.ts:1193-1194`). Any fix that adds a read-only alias
  binding must displace an existing one.

## 8. Verified vs inference

**Verified in code**: every line citation, the 80 B layout, all pass counts and
dispatch shapes, all array sizes, the `nativeRecords` resolution, the
`max(nx,ny,nz)>64` work gate, the dead-code guards, the 10-buffer limit.
**Inference**: per-invocation load *totals* (summed from call graphs, assuming
the common no-solid / single-terrain case), the 870 MB/advance figure, the
occupancy estimate for the 5 KB threadgroup array, the claim that dead branches
cost register pressure, and all "plausible minimum" byte counts.

## 9. Ranked candidate changes

| # | Change | Removes | Bit-exact? | Confirming measurement |
|---|---|---|---|---|
| 1 | Bake `nx/ny/nz` as WGSL `const` (specialize the module per domain) instead of the `uniformFieldPages[4].xyz` rewrite (`uniform-texture-pages.ts:55`) | every runtime div/mod (27/invocation in gather, ~39 in commit), folds robustness clamps | **Yes** | whole-advance A/B at 64³ and garden; expect the win concentrated in gather + commit |
| 2 | Hoist `uvSharpenTileActive` out of `uvCommitSharpen`/`uvLimitedFlux` (`:578-594`) to one per-tile test | 12 `uvCell` + 12 tile atomics + 72 `activeRegion` loads per lane → ≤7 per workgroup | **Yes** | commit-stage timer, 8 rounds × dense |
| 3 | Enable the cached sharpening arm at ≤64³ (drop the `>64` gate, `:974`) | ~210 loads/axis/round × 8 rounds on the gate lane | **Yes** (same arm, already A/B'd at garden) | mini64 advance median |
| 4 | Cache γ as a field (already published by `uvPublish:750`) instead of `uvOpen` | ~11 loads × 24 calls in buildEdges, ×2 in gather; kills the double `staticSolidVoxelOccupied` | **Yes** if γ is identical | buildEdges stage timer |
| 5 | Restrict `uvFinishDonorSums` to the work list | 4 dense passes × 28 B/cell (~127 MB/advance at garden) | **Yes** | stage timer + workgroup count |
| 6 | Rewrite `uvTarget:408-420` to load its 8 vertices once | 64 phi taps → 8 | **Yes** (same values) | gather stage timer |
| 7 | `override`-specialize the dead blocks out (`uvAgreementShift`, `uvSeedPhi`, classify compaction, rigid loop) | compile-time code and register pressure in advectPhi/classify | **Yes** | advectPhi timer; MSL register count via xctrace |
| 8 | Shrink the record: packed base + 3 fractions, hot/cold SoA split | 80 B → ~22 B; ~68 wasted B per cache line in sharpening | **No** (f16 weights) — bit-exact only if base+corner reconstruction keeps f32 | records-arena size + gather/sharpen timers |
| 9 | Fold donor-sum clears into `uvNormalizeDonors` | 4 clears × 24 B/cell (31.85 MB at garden) | **Yes** | pass count |
| 10 | One thread per cell in `uvTwoLevelSeed:651` with a workgroup reduction | 189 serial iterations per thread → parallel | **Yes** (atomicMax/atomicOr are order-free) | two-level stage timer |
| 11 | Fuse classify into gather, mark-transport into dilateZ | 2 passes + 1 dense sweep | **Yes** | pass count + launch floor |
| 12 | Per-workgroup reduction for `uvDustFloor:382-387` diagnostics atomics | 2 contended device atomics per live cell | **Yes** (counters only) | gather timer with counters stubbed |
| 13 | Move the coarse face table off the atomic binding | 8 `atomicLoad` per `sampleVelocity`, unblocks FP reassociation | **Yes** | *blocked*: needs a free storage slot (§7) |
| 14 | Don't apply the loop-bound rewrite to `uvPhi`'s `array<f32,8>` in the redistance module (`uniform-field-loop-bounds.ts`) | stack-array spill in the hottest inner loop | **Yes** | redistance timer; check MSL for `alloca` |
