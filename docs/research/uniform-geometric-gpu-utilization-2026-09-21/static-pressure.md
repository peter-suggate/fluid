# Static micro-architectural audit: Uniform Geometric pressure multigrid

Target: Apple M1 Max (32-core TBDR, unified memory, SIMD32, ~400 GB/s, 48 MB SLC).
Runtime: WebGPU → Dawn → Tint → MSL. **No GPU was used.** Every number is either read
from source (*verified*), replayed from a pure-JS line-for-line transcription of
`buildPlanSteps` (`lib/methods/uniform/webgpu-uniform-pressure-multigrid.ts:616-833`)
(*verified*), quoted from an existing measured repo document, or labelled
**inference**. All `wgsl.ts:N` citations are
`lib/methods/uniform/webgpu-uniform-pressure-multigrid.wgsl.ts`; `mg*.ts:N` is the
host file above; `ref.ts` / `ref.wgsl.ts` are `webgpu-uniform-reference{,.wgsl}.ts`.

## 0. Which code path runs — verified

| question | answer | evidence |
| --- | --- | --- |
| Geometric options | `pageDomain:true`, `activeRegion:false`, `pressureWindow:false`, `pressureCycleBudget:"lagged"`, `pressureBudgetHeadroom:0` | `uniform-geometric-options.ts:13,34,35` |
| Paged pressure storage? | **No.** `pagedStorage = options.pressureStorageForQA !== undefined`, never set by geometric | `ref.ts:1107,1115` |
| Atlas-packed multigrid fields? | **No** — `texture()` falls to `device.createTexture` | `mg.ts:241-242` |
| Indirect dispatch? | **No.** needs `pressureCycleDispatch==="indirect"` or `pagedPressure` | `ref.ts:1113-1114`; `mg.ts:456` |
| Window lattice / active-region launch? | **No.** `activeRegionEnabled` forced false by `pageDomain===true` ⇒ `activeDispatch` undefined | `ref.ts:758,1112,1119-1120` |
| Branch taken in `encode` | final `else`: `pass.dispatchWorkgroups(...dispatch.workgroups)` — **dense, direct, whole-level** | `mg.ts:471-472` |

"Native multigrid workspaces for paged domains" means what the comment at
`ref.ts:1105-1107` says: residency pages the *fluid* fields, the pressure hierarchy
keeps **dense 3D textures and untranslated taps**. `uniform-pressure-pages.ts` is
QA-only. Everything below analyses the dense path. Hierarchy from
`pressure-plan.ts:127-140` (lockstep halving, stops when the thinnest axis hits 2):
64³ → 6 levels, 128³ → 7. Every level carries a one-cell halo (`mg.ts:250`).

## 1. The fact that reframes everything

| | 64³ | 128³ |
| --- | ---: | ---: |
| levels / plan passes | 6 / **2 187** | 7 / **2 840** |
| setup / cycles / finish | 22 / 2 002 / 163 | 25 / 2 652 / 163 |
| encoded at lagged budget 1 / 2 / 7 | 651 / 1 117 / 2 187 | 832 / 1 476 / 2 840 |
| distinct param buffers / bind groups | 61 / 191 | 70 / 222 |
| hierarchy working set | 30.2 MiB | 229.2 MiB |

Measured, same solver, existing repo documents:
128³ pressure = **2 661 passes, 65.60 ms, 47.4 % of frame, 25 µs/pass**
(`docs/research/uniform-geometric-tile-first-2026-09-19/fig7-census-report.md:136-152`,
`live-tile-audit.md:415-427`); 64³ pressure = **2 008 passes, 48–54 ms of a 73–78 ms
wall ⇒ ~25 µs/pass** (`docs/benchmarks/uniform-pressure-tolerance-2026-09-19.json`,
`uniformPipelineFacts.multigridPassesTotal`).

**µs/pass is identical across an 8× change in cell count: the solve is launch-bound,
not cell-bound.** A traffic bound agrees — the whole 64³ plan moves ≈4.3 GB assuming
*zero* stencil reuse, ≈11 ms at 400 GB/s against 51 ms measured, and the 30 MiB
working set fits SLC (inference). **Rank changes by passes removed first, loads
second, bytes third.**

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64³ haloed cells | 287 496 | 39 304 | 5 832 | 1 000 | 216 | 64 | — |
| 64³ passes (share) | 402 (18 %) | 293 (13 %) | 380 (17 %) | 467 (21 %) | **554 (25 %)** | 91 (4 %) | — |
| 64³ `mgSmoothColour` | 296 | 240 | 312 | 384 | **456** | 0 | — |
| 64³ workgroups/dispatch | 17³=4 913 | 9³=729 | 5³=125 | 3³=27 | 2³=8 | 1 | — |
| 128³ passes (share) | 402 (14 %) | 293 (10 %) | 380 (13 %) | 467 (16 %) | 554 (20 %) | **641 (23 %)** | 103 (4 %) |

At 64³ **1 112 passes (50.8 %) run on L3–L5, which hold 1 280 cells between them**;
1 492 (68.2 %) run on L2–L5 (7 112 cells). At 128³ the identical shape sits one level
down: 1 298 passes (45.7 %) over the same 1 280 cells. Pass count per level-from-the-
bottom depends on hierarchy *depth*, not size, so this worsens as the domain grows.

## 2. Entry-point table

All entries are `@workgroup_size(4,4,4)`, 3D direct, `ceil(haloedDim/4)` per axis,
`gid.x` → texture x, unless noted. Group-1 binding counts include binding 13
(`mgState`), forced onto **every** entry (`mg.ts:300`). Group 0 is a **29-binding**
whole-solver group re-bound on every pass (`ref.ts:1070-1104,1181`). "loads" = distinct
texels on the hot path (r32float 4 B, rgba32float 16 B). Every gated kernel *also*
runs `mgSkipCycle` (2 device `atomicLoad`s on binding 13) and `mgActiveId` (**5 scalar
loads from the runtime-sized `activeRegion` array**) per invocation
(`wgsl.ts:54-59,138-151`).

| entry | wgsl.ts | grp1 | hot loads (B) | early-out | stores | stencil | count 64³/128³ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mgSmoothColour` | 448-468 | 7 | 16 (**148**): 7 p, 7 coef, rhs, min | 3 (24 B) | 1 | 7-pt on p **and** coef | **1 688 / 2 216** |
| `mgResidual` | 307-316 | 6 | 15 (144) | 1 (16 B) | 1 | 7-pt p+coef | 68 / 90 |
| `mgMeasureFineResidual` | 472-494 | 8 | 16 distinct / **22 issued** (coef gathered at 484 *and* 485) + 1–2 device `atomicMax` on one word | 1 | 0 | 7-pt ×2 | 17 / 17 |
| `mgRestrictResidual` | 321-327 | 4 | 8 (32) | — | 1 | 8 children | 80 / 105 |
| `mgProlongateAdd` | 355-360 | 5 | 9 (36) | — | 1 | 8 taps + residual | 65 / 87 |
| `mgProlongateAssign` | 362-367 | 4 | 8 (32) | — | 1 | 8 taps | 15 / 18 |
| `mgDownsampleSubtract` | 432-438 | 5 | 16 (64) | — | 1 | 8 children ×2 | 65 / 87 |
| `mgDownsampleMinimum` | 440-446 | 4 | 8 (32) | — | 1 | 8 children | 15 / 18 |
| `mgDownsampleTopology` | 283-305 | 6 | 16 (160) | — | 2 (20 B) | 8 children ×(V,phi) | 5 / 6 |
| `mgClearPressure` | 376-381 | 3 | 0 | — | 1 | none | 68 / 90 |
| `mgCopyPressure` | 369-374 | 4 | 1 | — | 1 | none | 20 / 20 |
| `mgAddPressure` / `mgShiftMinimum` | 397-402 / 390-395 | 5 | 2 | — | 1 | none | 3 / 3 each |
| `mgSaveAccepted` | 103-108 | 4 | atomic + 1 | atomic only | 1 | none | 15 / 15 |
| `mgRestoreRejected` | 109-119 | 4 | 1–3 atomics + 1 | atomics | 1 | none | 16 / 16 |
| `mgExtrapolatePhiOneCell` | 407-417 | 5 | ≤19 (140) | 2 (20 B) | 1 | 6-pt (V,phi) | 6 / 7 |
| `mgBakeCoefficients` | 419-428 | 5 | ≈11 (≈90) | — | 1 (16 B) | +x/+y/+z faces | 6 / 7 |
| `mgBuildFinestTopology` | 229-254 | 4 | helper chain: `cellOpenFraction`, 3×`pressureFaceData` (8 solid samples each), `pressurePhi` | halo arm | 2 | — | 1 / 1 |
| `mgBuildFinestRhs` | 256-281 | 5 | `divergenceAt` → 6×`pressureFaceData` (48 solid samples), `volumeCorrectionDivergence` (`atomicLoad`), `nearAnyBody` | — | 3 | 6-face | 1 / 1 |
| `mgSolveCoarsest` | coarse-solver 77-166 | 9 | **`@workgroup_size(256)`, dispatch (1,1,1)** — 1 workgroup; 64 of 256 lanes active at a 4³ coarsest; 6 int div/mod per lane per phase; 7 barrier points per iteration; cap 4 096 | — | 1/cell | 6-pt over `mgState.rows` | 22 / 25 |
| `mgCheckCycleConvergence` | 78-101 | 3 | `@workgroup_size(1)`, ~8 atomics | — | — | — | 16 / 16 |
| `mgFinishSafety` | 120-123 | 2 | `@workgroup_size(1)` | — | — | — | 1 / 1 |
| `mgPublishCycleDispatch` / `mgClearMinimum` | 65-76 / 383-388 | 3 | compiled, **never emitted** on this path | — | — | — | 0 / 0 |

**Strides.** All fields are `texture_3d` / `texture_storage_3d`
(`wgsl.ts:23-35,49-50`), so addressing is Metal's 3D swizzle, *not* row-major: the
"±z is a page away" model does not apply and a 4×4×4 workgroup's 7-point footprint is
compact (inference). Row-major equivalents, 64³ L0 (66³): p/rhs/min ±x 4 B, ±y 264 B,
±z 17 424 B; coefficients 16 / 1 056 / 69 696 B. 128³ L0 (130³): 520 / 67 600 B and
2 080 / 270 400 B.

**Arithmetic intensity.** `mgSmoothColour` liquid path ≈ 6 mul + 10 add + 1 div over
148 B = **0.14 FLOP/B**; `mgResidual` 0.09; restriction/prolongation 0.25–0.3. Machine
balance ≈26 FLOP/B. Nothing here is ALU-bound, so f16 *arithmetic* buys nothing;
f16 *storage* would.

## 3. Data layout

Per haloed cell per level (`mg.ts:251-257`):

| field | format | B/cell | ping-pong | written by |
| --- | --- | ---: | --- | --- |
| pressure | r32float ×2 | 8 | yes | every smoother / transfer |
| rhs | r32float ×2 | 8 | slot 1 = V-cycle scratch | restriction |
| phi | r32float ×2 | 8 | parity frozen at 1 after setup | setup only |
| volume V | **rgba32float ×2** | **32** | allocated, **only slot 0 ever bound** (`mg.ts:632`) | setup only |
| residual | r32float ×2 | 8 | slot 1 mostly unused | `mgResidual` |
| minimum p_min | r32float ×2 | 8 | yes | setup, shift, downsample |
| coefficients | **rgba32float ×1** | **16** | no (immutable) | `mgBakeCoefficients` |
| **total** | | **88** (L0 96 with `p_tmp` + `accepted`) | | |

Totals **30.2 MiB** at 64³ (L0 26.3) and **229.2 MiB** at 128³ (L0 201.1); `mgState`
adds `coarsestCells×48 B` = 3 072 B + 112 B (`uniform-coarse-solver.wgsl.ts:5-7`).

| shrink | mechanism | saving | bit-exact |
| --- | --- | --- | --- |
| drop unused V / phi / residual parities | only `volume[0]` is ever bound | −16…−24 B/cell → ~23 MiB / ~175 MiB | yes |
| **per-cell AoS "row record"**: 6 coefficients + baked diagonal + 6-bit neighbour-liquid mask (2 rgba32f = 32 B) | today a face coefficient is owned by the lower cell (`wgsl.ts:213-220`) and the neighbour's liquid flag lives in the neighbour's texel (`:210-212`), forcing **7 gathered rgba32f texels** | smoother 16 loads/148 B → **11 loads/68 B**; +16 B/cell storage, paid for by the line above | yes (same bake, same D4 fold, divide kept) |
| bake the diagonal into `coefficients.w` (a 0/1 flag stored as f32 today, `:427`) | re-derived from 6 taps on *every* sweep (`:462-463`), twice in `mgMeasureFineResidual` | 5 adds/invocation; frees the flag slot | yes if `mgD4Sum6` order kept and the divide stays (`1/d`× is **not**) |
| merge rhs + p_min into one rg32float | read at the same cell in both the hot (`:464,467`) and early-out (`:459`) paths | −1 load on every smoother invocation | yes |
| coefficients as `rgba16float` | `texture-formats-tier1` and `shader-f16` are requested when advertised (`lib/core/gpu-startup.ts:196-199`); **no fluid WGSL uses f16** — the repo's only `enable f16;` is `lib/svo/features/shading/program.ts:5270` | 112 B → 56 B | **no** (10-bit mantissa) |
| red-black in place (storage buffer, no ping-pong) | see §4 | 50 % of smoother stores | no |

## 4. The red-black smoother

**Verified: each colour is a full-grid dispatch, and the wrong-colour lanes do not
return — they store.** `wgsl.ts:458-459`:

```
let colour=u32((id.x+id.y+select(id.z,0,depthSymmetry()))&1);
if(coarseDone||!mgBakedLiquid(id)||colour!=mg.control.z){textureStore(mgPressureOut,id,vec4f(max(old,textureLoad(mgMinimumIn,id,0).x)));return;}
```

The pass-through store is forced by the ping-pong: `pressureIn` and `pressureOut` are
different textures (`mg.ts:630`), flipped after each colour (`mg.ts:736`), so every
cell must be written. The thread→cell mapping is good — `4,4,4` with `gid.x` fastest,
and Metal's `local_index = x + 4y + 16z` makes a 32-lane SIMD group a compact 4×4×2
block. But **parity alternates every x step**, so each SIMD group is exactly 16 active
and 16 pass-through lanes: divergence is maximal and uniform, in *every* group of
*every* colour pass. `depthSymmetry()` (`ref.wgsl.ts:279`) drops z from the parity,
making the scheme Jacobi in z for symmetry-depth scenes — a convergence property, not
a bug.

A colour-compacted mapping is cheap: dispatch `ceil(dx/2)` in x and map
`x = 2*gid.x + ((gid.y+gid.z+colour)&1)`. Every lane then shares a colour and the
thread count per colour pass halves. It is not bit-exact alone — the pass-through
currently re-applies `max(old, p_min)` to air and opposite-colour cells each sweep
(idempotent except for halo rows whose `p_min` is 0, which would need one up-front
clamp). Removing the ping-pong entirely needs the field in a
`var<storage, read_write>` buffer rather than a write-only storage texture; a
red-black sweep in place is race-free by construction, and the buffer form is also
what makes §5(a) easy.

## 5. Pass structure and fusion

Per V-cycle: `V(l) = 12 smooth + mgResidual + mgRestrictResidual + mgClearPressure +
mgDownsampleSubtract + V(l+1) + mgProlongateAdd + 12 smooth` (`mg.ts:747-760`) — **29
passes of overhead per level** plus the child, bottoming out in one `mgSolveCoarsest`.
A Full-Cycle adds `3 + 2(M−1) + 2` plus a nested V-cycle per level (`mg.ts:761-783`);
checkpoints add 4–5 (`mg.ts:784-797`).

| lever | finding | size |
| --- | --- | --- |
| **(a) the pass split may not be required** | `encode` opens a `beginComputePass` per dispatch (`mg.ts:453`), justified by *"A WebGPU texture usage scope spans the whole compute pass"* (`:448-452`). **Inference: stale** — in a compute pass each *dispatch* is its own usage scope, and dispatches are ordered with memory visibility (Dawn/Metal serial dispatch). The plan already guarantees no dispatch aliases a sampled and a writable texture — `:648-655` throws if it does. | collapses ~2 180 encoders **and** 2 187 × (setPipeline + 29-binding grp0 rebind + grp1 rebind) |
| **(b) fuse adjacent dispatches** | `mgResidual`→`mgRestrictResidual`: the intermediate `residualOut` is read only by the restriction (`mg.ts:752-753`). `mgClearPressure` + `mgDownsampleSubtract` write disjoint outputs at one level. Both colours fit one dispatch over a 2-colour-safe blocking: a 4³ tile with a ±1 halo is 6³×4 B = **864 B** of threadgroup memory, nowhere near the 20 KB occupancy cliff this repo has hit — which also allows *k* sweeps per dispatch on the tile interior. | ≈ −4 passes per V-cycle-level ≈ **−260 at 64³** |
| **(c) the resident coarse solver stops one level too low** | `mgSolveCoarsest` (`uniform-coarse-solver.wgsl.ts:77-166`) is already the right structure — one workgroup, `mgState.rows` in global storage, `storageBarrier()+workgroupBarrier()` between colours, loop to tolerance — but it owns only the coarsest level (`mg.ts:741`). Extending it to levels `k..M−1` with a per-level row offset is the largest single structural win. | **−1 112 passes (50.8 %) at 64³ for k=3**; −1 298 (45.7 %) at 128³ for k=4; **−1 492 (68.2 %)** if extended to L2 (rows grow to 341 KB) |
| **(d) 163 always-encoded finish passes** | `RECOVERY_BATCHES=8 × RECOVERY_SWEEPS=8` (`pressure-policy.ts:10-11`) emits 8×(16 smooth + 5 checkpoint) at `gated=2`, which `mgSkipCycle` skips whenever recovery is inactive (`wgsl.ts:54-59`) — i.e. essentially every step. The lagged budget cannot truncate them (`mg.ts:446` cuts only `[prefixEnd, finishStart)`). The 16 `mgSaveAccepted`/`mgRestoreRejected` in those batches are **not** `mgSkipCycle`-gated (`:105,111`), so 8 full-grid L0 copies run every step. The 2026-09-19 census recorded `finish: 1` — this tail is new. | 144 skipped launches ≈ **0.82 ms/step** at the 5.7 µs floor, plus **18.4 MB** (64³) / 141 MB (128³) of redundant copies |
| **(e) 17 blit encoders in the compute stream** | `encoder.clearBuffer` at `mg.ts:435` and `:451` (per residual checkpoint) makes Dawn end the compute encoder and open an `MTLBlitCommandEncoder` to fill **four bytes**. The single-thread `mgCheckCycleConvergence` runs immediately before and could reset word 15 for free. | 16 encoder transitions/step |

## 6. Uniforms, bind groups, Dawn state

Per-level params are **separate 80-byte uniform buffers with no dynamic offsets**,
de-duplicated by cache key (`mg.ts:868-894`): 61 buffers / 191 groups at 64³, 70 / 222
at 128³. The rest of the repo does use dynamic offsets
(`uniform-page-redistance.ts:44-45`), so one buffer + `hasDynamicOffset` would cut 191
groups to ~24 and let a fused pass change level with an offset instead of a rebind.
**Group 0 (29 bindings) is re-bound on all 2 187 passes** (`mg.ts:455`) even though it
is the same object throughout — a fresh Metal encoder inherits no state, so a
single-encoder plan would bind it once. There are 24 layouts / pipelines / pipeline
layouts (`mg.ts:297-301,379-384`), two of them never dispatched here, and binding 13
is forced into all of them (`:300`). Lazy-clear: textures are
`COPY_DST|COPY_SRC|STORAGE|TEXTURE_BINDING` (`:236-237`) and Dawn cannot prove a
compute shader fully wrote a storage texture, so each is zero-initialised once at
startup (inference) — the recurring blits are (e) above, not this.

## 7. Tint/MSL codegen hazards visible from the WGSL

| hazard | where | cost / fix |
| --- | --- | --- |
| **`mgActiveId` is dead code that costs 5 storage loads per invocation** | `wgsl.ts:138-151` | `pressureWindowLattice()` is always false (§0), the packed-extent word is never written so no clip fires, the origin is always 0. The paged-logical build already specialises it to `fn mgActiveId(gid:vec3u)->vec3i{return vec3i(gid);}` (`uniform-pressure-pages.ts:40`). **Bit-exact.** ~1.4×10⁸ invocations × 5 loads at 64³ |
| **`mgSkipCycle` does 2 device `atomicLoad`s per invocation** | `wgsl.ts:56-57`, on 2 002 of 2 187 passes | the repo memory `atomic-load-is-an-fp-barrier` records that atomic-binding reads are expensive *and* block FP reassociation. The smoother never writes `mgState`, so it could read the same buffer through a `read-only-storage` binding — **not guaranteed bit-exact**, precisely because lifting the barrier may let Metal reassociate the D4 folds |
| runtime-sized arrays + robustness clamps | `activeRegion` (`ref.wgsl.ts:104`), `mgState.rows` (`coarse-solver:17`), `mgCycleDispatch` (`wgsl.ts:64`) | each dynamic index costs a size-table read + compare. `mgState.convergence` is `array<atomic<u32>,26>` — fixed, no clamp. `activeRegion`'s indices are provably in range (`base = 16+10·activeLevel`, level ≤ 6, header 256 words) |
| integer div/mod by non-constants | `mgSolveCoarsest` `%d.x`, `/d.x`, `/(d.x*d.y)` in six separate loops per iteration (`coarse-solver:84,95,107,112,116,142`); `mgFineChild`'s `finePhysical/coarsePhysical` (`wgsl.ts:162`) called 8× per restriction | Apple has no hardware integer divide. **No `override`/`const` grid dims exist in the pressure module** — every dimension is a `mg.*` uniform, so Tint can strength-reduce nothing. The extrapolator does use overrides (`webgpu-uniform-velocity-extrapolation.wgsl.ts:77-82`), so the pattern is established; one pipeline per level is cheap |
| arrays passed by value | `mgD4Sum6(array<f32,6>)`, `mgD4Sum8Vec4(array<vec4f,8>)` (`wgsl.ts:127-137`) copy 24 B / 128 B per call | Metal usually SROAs after inlining, but the WGSL *requires* it; a failure spills the hottest kernel's fold to thread-private memory. Cheap check: inline the fold and diff the MSL |
| `select` evaluates both arms | `select(0.0, a*mgP(q), mgBakedLiquid(q))` (`wgsl.ts:462`) | the neighbour load is unconditional. Right call here (predication beats divergence), but air neighbours are still fetched |
| redundant clamps | `mgP`/`mgPhi`/`mgTopology` apply `mgClamp` (`wgsl.ts:126`, 6 int ops) on top of WGSL's already-safe `textureLoad` | for the smoother's six neighbour pressures the clamped value is discarded by the `mgBakedLiquid(q)` select, so dropping it there is **bit-exact**: ~36 int ops × 1.1×10⁸ invocations |
| one-word global `atomicMax` fan-in | `mgMeasureFineResidual` (`wgsl.ts:490,493`) | 1–2 device `atomicMax` into a *single* `u32` per liquid cell ≈ 10⁵ serialised atomics/pass × 17 passes/step. A threadgroup reduction first cuts it 64× and is exact (max is associative) |
| small workgroups | `@workgroup_size(4,4,4)` = 64 threads everywhere | inference: 128–256 schedules better on Apple. At L4/L5 the whole dispatch is 8 or 1 workgroup — ≤1/32 of the GPU. `var<workgroup>` footprint is *not* a problem: only `mgCycleStopped` (4 B) and the coarse solver's 36 B exist, and Tint allocates per entry point by reachability |

## 8. Convergence-side levers (work, not codegen)

Schedule at HEAD (`pressure-policy.ts:2-20`): 3 Full-Cycles + 4 V-Cycles, 6 pre / 6
post PRBGS sweeps, coarse tolerance 1e-4 s⁻¹, coarse sweep cap 4 096, plus 8×8
recovery; default residual tolerance 10 (`:35`). The lagged budget (`:76-91`) already
truncates the encoded prefix and geometric runs it at `headroom = 0` with
`initialCycles = 1` (`uniform-geometric-options.ts:13`; `ref.ts:784,2305`) — encoded
passes at 64³ are 651 / 1 117 / 2 187 for budgets 1 / 2 / 7. It is already the biggest
working lever, and **it cannot touch the 163-pass finish tail or the 22-pass setup**.
Remaining moves, cheapest first: post-sweeps 6 → 4 (−4 per V-cycle-level ≈ −290 at
64³); three Full-Cycles → one (466-pass cycles replaced by 151-pass V-cycles);
`RECOVERY_BATCHES` 8 → 2 (−120 always-encoded passes). All change the answer and must
be judged on the residual, not the clock.

## 9. Ranked candidates

| # | change | mechanism | removes | bit-exact | confirming measurement |
| --- | --- | --- | --- | --- | --- |
| 1 | **One compute pass per V-cycle** instead of per dispatch | the usage-scope justification at `mg.ts:448-452` looks stale; per-dispatch scopes + in-order dispatch make the split unnecessary | ~2 180 encoder begin/ends; 2 187 × (setPipeline + 29-binding grp0 + grp1) | **yes** if Dawn accepts and orders them | 20-line Dawn probe: two dispatches in one pass, the second reading the first's storage-texture output; then A/B the real plan for pass count and a bit-identical pressure field |
| 2 | **Resident multi-level coarse solver** — extend `mgSolveCoarsest` to own L≥3 (64³) / L≥4 (128³) | one workgroup, `mgState.rows` per level, `storageBarrier` between colours; the structure already exists | **1 112 passes (50.8 %) at 64³**, 1 298 (45.7 %) at 128³; 68.2 % if extended to L2 | no (different sweep order; serial-in-workgroup coarse solve) | `planPassCount`, then pressure ms from the tolerance benchmark at fixed tolerance; `convergence[10]` must not regress |
| 3 | **Specialise `mgActiveId` to `vec3i(gid)`** | the window path is provably inert (§0); the paged-logical shader already does this | 5 storage loads + robustness clamps × 1.4×10⁸ invocations/step | **yes** | swap the fragment string, bit-compare the finest pressure texture, then stage ms |
| 4 | **Recovery batches 8 → 1–2**, and gate `mgSaveAccepted`/`mgRestoreRejected` on `mgSkipCycle` | they are encoded every step and skipped every step | 120–144 launches ≈ 0.7–0.8 ms, plus 18.4 MB (64³) / 141 MB (128³) of full-grid copies | yes for the gating, no for the batch count | `encodedPassCount` and the `finish` stage seam in the advance trace |
| 5 | **Per-cell AoS row record** | removes the 6 gathered neighbour-coefficient taps and the 6 `mgBakedLiquid(q)` taps | smoother 16 loads/148 B → **11 loads/68 B**; same for `mgResidual` | **yes** | L0 smoother pass time in the stage trace; bit-compare the field |
| 6 | **Delete the unused V / phi / residual parities** | only `volume[0]` is ever bound (`mg.ts:632`) | 30.2 → ~23 MiB (64³); 229 → ~175 MiB (128³) | yes | `allocatedBytes` in `stats`; no timing change expected below 128³ |
| 7 | **Red-black in place** (storage buffer + colour-compacted x) | no ping-pong ⇒ no pass-through store; half the threads per colour | 50 % of smoother threads and stores; enables #1/#2 cleanly | no (halo `p_min` clamp must be reproduced once) | thread count from the plan; then pressure ms + residual parity |
| 8 | **CSE `mgMeasureFineResidual`'s two coefficient gathers; threadgroup-reduce before the device `atomicMax`** | `wgsl.ts:484` and `:485` load the same 7 texels; 10⁵ atomics land on one word | 7 texels/invocation and 64× atomic contention, on 17 passes/step | yes (max is associative) | per-pass timing of the checkpoint seam |
| 9 | **Reset diagnostics word 15 inside `mgCheckCycleConvergence`** | removes a 4-byte blit between compute encoders | 16 encoder transitions/step | yes | encoder count in a capture; wall ms |
| 10 | **`rgba16float` coefficients** | halves the smoother's dominant load | 112 B → 56 B per liquid invocation | **no** | residual and volume-drift parity on the Dawn lanes first, then ms |
| 11 | **`override` grid dims (one pipeline per level) + fixed-size `activeRegion`** | lets Tint strength-reduce `mgFineChild`'s divides and drop robustness clamps | int div/mod in restriction/prolongation and the coarse solver | yes | MSL diff; restriction-pass timing |
