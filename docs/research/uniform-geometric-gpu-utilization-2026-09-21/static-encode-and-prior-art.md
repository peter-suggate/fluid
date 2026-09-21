# Uniform Geometric: the advance as a command stream, and what prior art already measured

2026-09-21 · **static audit** — no GPU device was created, no source was edited, no git command
was run. Every quantitative claim below carries a `file:line` or a doc path. A final section
separates **verified in code** from **inference**.

Configuration under audit: method `uniform-volume` ("Uniform Geometric",
`lib/methods/uniform/uniform-volume-method.ts:1-38`) at a 64³ lattice, shipping defaults from
`lib/methods/uniform/uniform-geometric-options.ts:8-38` and
`lib/methods/uniform/uniform-geometric-parameters.ts:1-94`, steady state (no rigid bodies, no
inflow, no drop, no symmetry audit, instrumentation off).

---

## Part A — the advance as a command stream

### A0. Which residency arms are live at 64³

This determines almost everything else, so it comes first.

| predicate | value at 64³ | where |
| --- | --- | --- |
| `pageDomain` | **on** (32³ pages, 2×2×2 = 8) | options `:13`; `initialUniformPageDomain` `uniform-page-domain.ts:18-31` |
| `nativePageCoordinates` (`count===1 && capacity===1`) | **false** (count 8) | `webgpu-uniform-reference.ts:797`, `uniform-page-execution.ts` |
| `fieldPages` paged storage | **false** (rectangular coverage ⇒ `pagedStorage=false`) | `webgpu-uniform-reference.ts:798-800` |
| `nativeRootExecution` | **true** (`fieldPages.nativeStorage`) | `:802` |
| `activeRegionEnabled` | **false** (`pageDomain !== true` fails) | `:758` |
| `pageDomainDispatch` | **undefined** (set only when `!nativeRootExecution`) | `:965` |
| `volumePageEdge` / `volumePageConfig.work` | 32 / **false** (`max(n) > 64` fails) | `:803-805`, `:971-976` |
| `phiWindow` (`phiRegion`/`phiDispatch`) | **true** | `:936-945` |
| pressure `pagedStorage`, `gpuCycleDispatch` | **false**, **false** | `:1107-1115` |

Consequences, all verified: the solve-window / indirect-window machinery is dead code here;
`UniformPageDomainPublication.encode` is skipped (`:2643`); `fieldPages.encodePublications` emits
**zero** passes because `publications` is only populated for paged textures
(`uniform-texture-pages.ts:100-101, 125-130`); and with `volumeWorkDispatch` undefined,
`runVolumeWork` falls through to `this.run` (`:2454`), so **every volume and sharpening pass is a
direct dense `dispatchWorkgroups(16,16,16)`**.

### A1. Full pass census, one steady-state advance at 64³

Encode order, from `advanceTo` (`webgpu-uniform-reference.ts:2554-2958`). One command encoder
(`:2631`), one `queue.submit` (`:2919`).

| # | stage | passes | dispatch geometry | direct / indirect |
| ---: | --- | ---: | --- | --- |
| 1 | `encodePhiRegion` census (`:2194-2211`) | 3 | (16,16,16), (1,1,1), (1,1,1) | direct |
| 2 | two-level 4h class map (`:2681-2690`) | 4 | (4,4,4) | direct |
| 3 | Sec. 3.3 authority (`:2242-2244`) | 1 | (16,16,16) | direct |
| 4 | Sec. 3.3 extension (`webgpu-uniform-velocity-extrapolation.ts:466-551`) | 19 | seed/resolve/prolong-to-base (16,16,16); prepare (1,1,1); restrict/prolong per level | **2 indirect** (FIM updates off `dispatchArgs`) |
| 5 | 4h face table publish (`:557-566`) | 1 | (9,9,9) | direct |
| 6 | `uvAdvectPhi`, `uvRedistancePhi` (`:2500-2502`) | 2 | vertex 65³ | **2 indirect** (off `phiDispatch`) |
| 7 | page mark + compact, transport (`:2431-2451`) | 2 | (16,16,16), (1,1,1) | direct |
| 8 | edges / donor finish / fallback (`:2507`) | 3 | (16,16,16) | direct |
| 9 | 3 × normalize rounds (`:2508-2512`) | 9 | (16,16,16) | direct |
| 10 | `uvGather` (`:2514`) | 1 | (16,16,16) | direct |
| 11 | total surface volume (`:2522-2527`, `webgpu-uniform-surface-volume.ts:68-86`) | 16 | capacity/targets (16,16,16); 14 reduction passes | direct |
| 12 | classify 4h sharpening map (`:2531-2532`) | 1 | (16,16,16) | direct |
| 13 | page mark + compact, sharpen (`:2533`) | 2 | (16,16,16), (1,1,1) | direct |
| 14 | 8 rounds × 4 sharpening sweeps (`:2544-2547`) | 32 | (16,16,16) | direct |
| 15 | semi-Lagrangian advection (`:2817`) | 1 | (16,16,16) | direct |
| 16 | surface-deficit balance (`:2466-2481`) | 2 | (16,16,16), (1,1,1) | direct |
| 17 | **CM11a pressure** (`webgpu-uniform-pressure-multigrid.ts:428-474`) | **651** at budget 1 | per level, see A1b | direct |
| 18 | projection (`:2859`) | 1 | (16,16,16) | direct |
| 19 | `uvPublish` (`:2877`) | 1 | (16,16,16) | direct |
| 20 | diagnostics reduction (`:2916`) | 1 | (16,16,16) | direct |
| | **non-pressure subtotal** | **102** | | 4 indirect |
| | **total, cycle budget 1** | **753** | | |
| | **total, full 7-cycle schedule** | **2 289** | | |

Host-side resource traffic per advance:

| primitive | count | bytes | citation |
| --- | ---: | ---: | --- |
| `createCommandEncoder` | 1 | — | `:2631` |
| `queue.submit` | 1 | — | `:2919` |
| `queue.writeBuffer` | **1** | **208** | `:1432-1471` (52 floats into `this.params`) |
| `clearBuffer` | **27** | **26 265 204** | see below |
| `copyBufferToBuffer` | 5 | 2 160 | `:2199-2200/2208-2209` (2×1024), `:2449` (96), `:2438` (4), `:2854` (12) |
| `copyTextureToTexture` | 3 | 3 195 652 | surface-volume output→phi (1 098 500), γB→γA (1 048 576), volumeB→volumeA (1 048 576) |
| `setPipeline` | 753 | — | one per pass |
| `setBindGroup` | 1 404 | — | 1 per non-pressure pass (`:1716`); 2 per pressure pass (`:454-455`) |
| distinct pipelines bound | ≈ 70 of 105 resident | — | see A6 |

Clears, by size: 4 × `volumeDonorSums` at **6 291 456 B** each (`:2506, :2509`; sized
`nx*ny*nz*24` at `:885`) = 25 165 824 B; surface-volume band 1 098 500 B
(`webgpu-uniform-surface-volume.ts:69`); `rigidExchange` 576 B (`:2642`); `reductions` 40 B
(`:2641`); multigrid diagnostics 104 B + 11 × 4 B (`webgpu-uniform-pressure-multigrid.ts:435, 451`);
the rest are 4–32 B counters. **96 % of all cleared bytes are the four donor-sum clears.**

### A1b. Pressure plan, exactly

Replayed from `buildPlanSteps()` (`webgpu-uniform-pressure-multigrid.ts:616-833`) with `p[]`/`min[]`
parity tracking, so the conditional `mgCopyPressure` in `checkpoint()` is exact. Policy
`FULL_CYCLES=3, V_CYCLES=4, PRE=POST=6, RECOVERY_BATCHES=RECOVERY_SWEEPS=8`
(`pressure-policy.ts:1-14`); lockstep hierarchy from `pressure-plan.ts`.

| lattice | levels | setup | per Full-Cycle | per V-Cycle | **finish** | full plan | **budget 1** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32³ | 5 | 19 | 317 | 122 | 163 | 1 621 | 499 |
| **64³** | **6** | **22** | **466** | **151** | **163** | **2 187** | **651** |
| 128³ | 7 | 25 | 644 | 180 | 163 | 2 840 | 832 |

At 64³: `mgSmoothColour` = 1 688 of 2 187 (77 %); level workgroup counts are
L0 17³=4 913, L1 9³=729, L2 5³=125, L3 3³=27, L4 2³=8, L5 1.

**The finding that matters most.** `encode()` truncates with
`if (index >= prefixEnd && index < this.finishStart) continue;` (`:446`), so everything at or after
`finishStart` — the **163-pass recovery finish** (8 batches × 8 sweeps × 2 colours = 128
`mgSmoothColour`, 8 checkpoints, a 4-pass tail) — **is encoded on every advance at every cycle
budget**. Its bodies early-return unless the GPU set recovery mode
(`webgpu-uniform-pressure-multigrid.wgsl.ts:57, 70`), so on an accepted step those 163 passes are
pure launch floor: **25 % of the pressure stream and 21.6 % of the whole advance** at budget 1.
The section post-dates the 2026-09-19 measurements (B5).

### A1c. Scaling

Non-pressure passes are **almost constant in grid size**: only the extension hierarchy grows, at
+2 per doubling (`encodedPassCount`, `webgpu-uniform-velocity-extrapolation.ts:395-400`:
`4 + 2·frontSweeps + down + up − 1`; 6 down + 6 up at 64³), so 32³→100, 64³→102, 128³→104.
Pressure grows as `3L+4 + 3F(L) + 4V(L) + 163` with `L = log2(min n)` — logarithmic, while work per
pass grows as the cube. Hence at 64³ the pressure stream sits **at** the launch floor and at 128³+
the volume stages become per-cell bound (Part B).

Page count does not change the 64³ pass count. Above `max(n) = 64`, `volumePageConfig.work` flips
true (`:974`), 20 volume/sharpening passes become indirect off `volumeWorkDispatch`, and 2 cache
passes plus a seeding `copyField` appear — a regime this census does not cover.

### A2. One pass per dispatch — and what forces it

**Verified: the solver opens one compute pass per dispatch, everywhere; there is no batching API.**
`run()` is literally `beginComputePass → dispatch → end` (`:1726-1730`), and `runDirect`
(`:1732-1738`), `runVertex` (`:1747-1761`), `runVolumeWork` (`:2454-2458`), the extrapolator
(`webgpu-uniform-velocity-extrapolation.ts:441, 477, 488, 495, 505, 518, 537, 561`), the surface
volume (`webgpu-uniform-surface-volume.ts:71-75`) and the multigrid (`:453`) all do the same.
**753 passes = 753 pass begins in one command buffer.**

What forces the split:

1. **Texture usage scope**, by far the dominant blocker. The multigrid's own comment: *"A WebGPU
   texture usage scope spans the whole compute pass. End the pass between hierarchy stages so
   storage outputs can become sampled inputs in the next stage."* (`:447-449`). The red/black
   smoother ping-pongs `pressure[p]`/`pressure[p^1]`, flipping parity after each colour (`:709`
   `flipPressure`, applied at `:736`), so colour 2 samples colour 1's output — this alone blocks all
   1 688 smoother passes.
2. **Storage-write → indirect-read.** A buffer cannot be writable STORAGE and INDIRECT in one usage
   scope, so FIM `prepare` (writes `dispatchArgs`,
   `webgpu-uniform-velocity-extrapolation.ts:146-150`) and `update` must alternate passes; likewise
   `uvCompactPages` → `copyBufferToBuffer` → `volumeWorkDispatch` (`:2444`).
3. **Nothing else.** Timestamp seams splice into the next real pass's `timestampWrites`
   (`:2605-2607`, `completeFinalPhaseOnNextPass` at `:2915`), and labels have no isolation effect.

Legal batching that exists today, all buffer-only or usage-compatible:

| candidate | passes now | batched | saving |
| --- | ---: | ---: | ---: |
| phi region census (scan → reduce → finalize, storage buffers only) | 3 | 1 | 2 |
| surface-volume reduction (begin, seed, 4×dilate, metric, 2×(measure,reduce,solve), apply — `band`/`next` are storage buffers, the output texture is write-only and never sampled) | 14 | 1 | 13 |
| page mark + compact, ×2 phases (`conditioningScratch` only) | 4 | 2 | 2 |
| surface-deficit measure + reduce | 2 | 1 | 1 |
| **total, no numerical change** | **23** | **5** | **18** |

18 of 753. Pass batching is **not** the lever at 64³ unless the pressure smoother is rewritten to
read through storage textures rather than sampled ones — which is the only way to touch the 1 688.

### A3. Bind groups, buffers, pipelines on the hot path

**Zero `createBindGroup`, zero `createComputePipeline`, and one conditional `createBuffer` on the
per-advance path.** Verified by walking every creation site:

- Main bind groups are built once in the constructor via `createPageAwareGroup` (`:1124-1234`,
  helper `:2414-2418`).
- Multigrid groups, views and 80-byte param buffers are cached by key at *plan* time (`groupCache`
  `:688-694`, `paramCache` `:864-894`, `viewCache`). The comment at `:169-178` records why: building
  them per dispatch cost 143 ms of a 178 ms re-plan on 64×512×64 (59k views, 4.2k param buffers,
  4.2k bind groups).
- The only per-advance `createBuffer` is `pressureCycleDemandReadback ??= …` (`:2851`), once on the
  first advance. `windowReadback` (`:2903`) is unreachable here; `statsReadback` (`:2964`) is in
  `readStats`.

**Param delivery is already minimal:** exactly **one** `queue.writeBuffer` per advance, 208 bytes,
into a persistent UNIFORM buffer (`:914`, written at `:1432-1471`). No dynamic offsets; per-dispatch
pressure parameters live in the cached 80-byte buffers written once at plan time (`:891`). There is
no staging-copy traffic to remove.

### A4. Indirect dispatch

Four per advance, and the repo has already measured that the direction of travel is *away* from
indirect.

| site | args written by | could be CPU-known? |
| --- | --- | --- |
| `uvAdvectPhi`, `uvRedistancePhi` (`runVertex`, `:1751`) | `finalizeActiveRegion` in the phi census, copied to `phiDispatch` (`:2209`) | Yes, with a host-lagged box — the machinery (`windowVertexGroups`, `planWindowDispatch`) exists but is disabled for geometric. |
| 2 × FIM update (`webgpu-uniform-velocity-extrapolation.ts:497`) | `prepare` (single workgroup) from the wavefront count | No — the front size is genuinely data-dependent. |

`docs/research/uniform-geometric-tall-air-2026-09-19/solve-window-report.md:38-39`: converting
~1000 launches a step from direct to indirect cost **+4.6 ms GPU and +2.6 ms host encode**; line 21
records the whole advance at **+40.7 %**. The kernel comment at `:328-334` puts an indirect launch
at 15–25 µs against 3–6 µs direct on this lane.

The asymmetry worth noting: the phi window **spends 3 passes and 2 × 1 024 B copies per advance to
shrink 2 passes**. `phiWindowForQA: false` (`:936`) is an existing kill switch, so the A/B is free.
Validation of the 4 indirect buffers is unavoidable (see Part C) but negligible at this count.

### A5. Buffer creation flags and clears

| buffer | size at 64³ | usage | note |
| --- | ---: | --- | --- |
| `volumeDonorSums` (`:885`) | 6 291 456 | STORAGE \| COPY_DST | cleared 4× per advance; COPY_DST exists only for the clear |
| `volumeEdges` (`:887`) | `nx·ny·nz·EDGE_BYTES` | STORAGE \| COPY_SRC \| COPY_DST | never cleared per advance |
| `conditioningScratch` (`:1011`) | 3 178 600 | STORAGE \| COPY_SRC \| COPY_DST | cleared in small ranges only |
| surface-volume band (`webgpu-uniform-surface-volume.ts:46`) | 1 098 500 | STORAGE \| COPY_SRC \| COPY_DST | cleared whole, once per advance |
| `params` (`:914`) | 208 | UNIFORM \| COPY_DST \| COPY_SRC | one `writeBuffer`/advance |
| `dispatchArgs` (extrapolation `:146`) | 12 | STORAGE \| INDIRECT \| COPY_DST | |
| `phiDispatch` (`:942`) | 1 024 | INDIRECT \| COPY_DST | |

Nothing forces non-private storage: no buffer on the advance path carries MAP_READ or MAP_WRITE.
The three readbacks are separate COPY_DST|MAP_READ buffers filled by explicit copies.

Can the four donor-sum clears fold into a first write? Not directly — `uvBuildEdges` /
`uvNormalizeRows` accumulate into 24-byte fixed-point records and a cell no donor touches must read
zero. The structural alternative is a generation tag (one u32 per cell holding the round index; a
stale tag reads as zero): 1 MiB and one compare per access, and all four clears go.

### A6. Pipelines and specialization

**105 compute pipelines** are resident for this configuration:

| group | count | citation |
| --- | ---: | --- |
| main `PIPELINES` | 30 | `:273-304`, compiled `:1305-1314` |
| `UNIFORM_VOLUME_ENTRIES` | 20 | `uniform-volume.wgsl.ts:4-11`, `:1315-1323` |
| `UNIFORM_VOLUME_PAGE_ENTRIES` | 3 | `uniform-volume-pages.wgsl.ts:6`, `:1324-1331` |
| tiled sharpening (4 sweeps + 2 cache), `UV_SHARPEN_TILE_WORK=1` | 6 | `:1334-1343` |
| 4h sharpening classify | 1 | `:1345-1351` |
| two-level E1 | 4 | `:1355-1363` |
| surface-volume correction | 8 | `webgpu-uniform-surface-volume.ts:5, 33-35` |
| velocity extension | 10 | `webgpu-uniform-velocity-extrapolation.ts:452-465` |
| CM11a multigrid | 22 | `webgpu-uniform-pressure-multigrid.ts:15-23` |
| page-domain publication | 1 | `uniform-page-domain-publication.ts` |

Field-page publication pipelines: **0**, because `publications` is empty when nothing is paged
(`uniform-texture-pages.ts:100-101, 111`).

Pipeline-overridable constants are used in exactly **two** places:
`UV_SHARPEN_TILE_WORK` (`uniform-volume.wgsl.ts:433`, set at `:1341`) and
`SOURCE_AWARE_HIERARCHY` / `ROOT_NX` / `ROOT_NY` / `ROOT_NZ`
(`webgpu-uniform-velocity-extrapolation.wgsl.ts:77-82`, set at `:420`). **Grid dimensions and every
level's dimensions arrive through uniform buffers, not overrides** — `mg.levelDims` in the
multigrid, `this.params` in the main module.

Specializing dims multiplies pipelines by the level count — 22 multigrid entries × 6 levels = 132,
the extension's 10 × 7 = 70, so roughly 105 → 270.
`docs/research/uniform-startup-compilation.md` (2026-09-21) records water-box startup falling
84.97 s → 68.23 → 47.29 → **7.21 s** and says explicitly it is *not* a cold-cache benchmark.
Tripling the pipeline count attacks that number directly.

---

## Part B — what has already been measured (so it is not re-proposed)

### B1. Per-stage GPU time

**fig7 (128³)** — `docs/research/uniform-geometric-tile-first-2026-09-19/fig7-census-report.md:75-152`.
GPU total median **138.54 ms** over 2 748 passes; CPU `physicsCPUTrace` **30.90 ms** (Full-Cycles
encoding 13.85, V-Cycles 5.09, capture-closure + submission 10.90; *every other stage under
0.21 ms*). Its conclusion: *"CPU encode is 22 % of the GPU span, so the frame is GPU-bound; the CPU
cost is essentially all command encoding for the ~2 650 pressure passes."*

| group | passes | median ms | share | µs/pass |
| --- | ---: | ---: | ---: | ---: |
| velocity extension (Sec. 3.3) | 35 | 24.18 | 17.5 % | 691 |
| conservative volume transport | 12 | 26.61 | 19.2 % | 2 217 |
| `uvGather` + γ copy | 1 | 6.36 | 4.6 % | 6 357 |
| velocity advection | 1 | 4.92 | 3.5 % | 4 915 |
| volume sharpening (tile map on) | 33 | 2.62 | 1.9 % | 79 |
| phi transport + redistance | 2 | 2.29 | 1.7 % | 1 147 |
| **pressure** | **2 661** | **65.60** | **47.4 %** | **25** |
| projection / publication / diagnostics | 3 | 2.56 | 1.8 % | — |

Line 151-152: *"Every non-pressure group runs at 79 µs to 6.4 ms per pass — far above the
~13–25 µs pass floor. Pressure runs at 25 µs per pass, i.e. **at** the floor."* Line 180-181:
8× the cells bought only 2.0× the pressure time; per pass 16.2 µs at 64³ → 24.7 µs at 128³.

### B2. Per-pass cost constants that are actually citable

| constant | value | source |
| --- | --- | --- |
| un-encoded pass, fig7 128³ | **7.4 µs GPU, 10.3 µs CPU** | `p1-lagged-cycle-budget-report.md:67` |
| un-encoded pass, large-power dam | 3.8 µs GPU, 8.6 µs CPU | same, `:69` |
| GPU fit, pressure seam | **4.758 µs/pass + 1.734 ns/workgroup**; launch floor = **85 % of arm A's pressure seam** (4.42 of 5.18 ms) | `.../tall-air-2026-09-19/ab-report.md:176, 185-186` |
| setup seam fit | 14.7 µs/pass + 16.8 ns/workgroup | same, `:187` |
| pressure µs/pass measured | 16.2 (64³) → 24.7 (128³) | `fig7-census-report.md:181` |
| direct → indirect delta | +4.6 ms GPU, +2.6 ms host encode for ~1 000 launches | `solve-window-report.md:38-39` |
| Dawn timestamp tick | 65.5 µs | `fig7-census-report.md:78, 133` |

The brief's unit of **~11 µs CPU + ~13 µs GPU per pass** is a reasonable central estimate but is
not stated in any repo doc; the citable bracket is 8.6–10.3 µs CPU and 3.8–25 µs GPU depending on
lattice and level. The candidate table below uses the brief's unit and gives the ab-report's
4.758 µs/pass as the conservative GPU floor.

### B3. Optimizations already measured null or negative — do not re-propose

| change | result | source |
| --- | --- | --- |
| GPU-indirect window launches (~1 000/step) | **+40.7 %** whole advance | `solve-window-report.md:21, 38-39` |
| L0 4³ smoother tile map | slower | tall-air campaign |
| extension-front 4h work map | inside noise, reverted | `docs/benchmarks/uniform-geometric-extension-tile-work-2026-09-19.md` |
| tight workgroup extents | no benefit | tall-air campaign |
| residual tolerance 1e-4 on mini64 | never converges; budget pinned at 7 | `p1-lagged-cycle-budget-report.md:50-54` |
| per-pass completion-frontier timing | 39.74 → 78.97 ms; rejected | tall-air campaign |
| extension front sweeps 16 → 8 | within 0.01 cell; default is now **2** | `docs/benchmarks/uniform-extension-front-sweeps-2026-09-19.md`, `uniform-geometric-parameters.ts:10` |
| lagged cycle budget (P1) | **landed**: fig7 −5.54 ms GPU / −2.58 ms CPU; large-power −2.29 / −5.11; mini64 −0.07 (null) | `p1-lagged-cycle-budget-report.md:29-37` |

### B4. Open items the prior art names

- **P2, resident coarse solve for levels ≤16³**: ~200 passes, ~1.5 ms GPU + ~2 ms CPU.
- **P3, fuse smoother + residual + restrict**: ~700 passes, ~5 ms GPU + ~7 ms CPU.
  Both are pass-count attacks on the multigrid; both are consistent with this audit's A2 finding
  that the smoother's sampled/storage ping-pong is what blocks batching.

### B5. Staleness flags — important

1. **Every 09-19/09-20 document predates the page domain.** `pageDomain:true` is now the shipping
   default (`uniform-geometric-options.ts:13`) and `activeRegionEnabled` is consequently false
   (`:758`), so the window results describe a path that no longer runs.
2. **Their pass counts are stale.** `p1-lagged-cycle-budget-report.md:75-79` gives 64³ = 23
   setup+finish, ~463/Full-Cycle, 149/V-Cycle, **2 008** total, and 128³ = **2 661**. HEAD is
   **2 187** and **2 840**: the 163-pass recovery finish (`pressure-policy.ts:10-11`) is new, and it
   is never truncated.
3. **Default residual tolerance is now 10, not 1e-4**, and geometric headroom is 0 (`:782-784`), so
   the V-cycle convergence behaviour in `fig7-census-report.md:141-145` no longer applies.
4. `docs/benchmarks/uniform-pressure-pass-floor-2026-09-19.md` does **not** exist (tree or
   `git log --all`); nor does `artifacts/xctrace-uniform-mini-dam-64/`.

---

## Part C — Dawn / Metal facts, only where groundable

- **`node_modules/webgpu` is v0.6.0**, `git+https://github.com/dawn-gpu/node-webgpu.git`, on
  `@webgpu/types ^0.1.72`. The package metadata records **no Dawn revision** and the binary is
  prebuilt, so the exact Dawn commit cannot be established statically.
- **Toggles** are `create([...])` strings, composed at
  `lib/harness/webgpu-smoke-executor.ts:436-439` (`enable-dawn-features=$FLUID_WEBGPU_DAWN_FEATURES`
  plus a `disable-dawn-features=` list). Uniform Dawn tools pass only `backend=metal`
  (`tools/benchmark-uniform-volume-pages-dawn.ts:18`,
  `tools/benchmark-uniform-pressure-tolerance-dawn.ts:21`); `tools/benchmark-uniform-pages-dawn.ts:73`
  adds `disable-dawn-features=timestamp_quantization`.
- **`skip_validation` is never enabled in a uniform lane** — only
  `tools/power-dam-run-environment.ts:85` and `tools/profile-svo-render-xctrace.ts:576`. Both the app
  and the uniform Dawn lanes therefore run full indirect-args validation (4 buffers/advance here).
- **App device** (`lib/core/webgpu-renderer.ts:1794-1803`):
  `requestAdapter({powerPreference:"high-performance"})`, then
  `fluidExecutionDeviceFeatures(adapter.features)` (`lib/core/gpu-startup.ts:189-202`) =
  `timestamp-query` (`:178-182`) plus `shader-f16`, `subgroups`, `texture-formats-tier1` when
  advertised.
- **Uniform Dawn harness device**: `requiredFeatures:["timestamp-query"]` only
  (`tools/benchmark-uniform-pressure-tolerance-dawn.ts:22`) — narrower than the app, so Dawn-lane
  A/Bs run without `shader-f16` and `subgroups`.
- **Nothing awaits the GPU on the advance hot path.** `advanceTo` ends with a fire-and-forget
  `void queue.onSubmittedWorkDone().then(...)` (`:2954`); both readbacks are `void mapAsync`
  (`:2168`, `:2338`). `readStats()` does `await this.awaitFrameCompletion()` (`:2961` → `:2551`),
  but the app calls it off the hot path (`webgpu-renderer.ts:2220`); the Dawn harness awaits it per
  step in some lanes (`webgpu-smoke-executor.ts:3388`) — a measurement hazard, not a production one.

---

## Verified in code vs inference

**Verified in code** (walked, cited): every count in Part A — 753 / 2 289 passes, 1 encoder,
1 submit, 1 `writeBuffer` of 208 B, 27 clears totalling 26 265 204 B, 5 buffer copies, 3 texture
copies, 4 indirect dispatches, 105 pipelines, 2 uses of `override` constants, zero hot-path
`createBindGroup`/`createComputePipeline`, the always-encoded 163-pass finish, the A0 residency
predicates, and the pressure census (replayed against `buildPlanSteps`).

**Inference** (reasoned, not observed here): that each WebGPU compute pass becomes its own
`MTLComputeCommandEncoder` region with its own resource-tracking, lazy-clear and barrier work — the
standard Dawn-on-Metal mapping, consistent with B2's per-pass floor, but Dawn is a prebuilt binary
here and no repo document states it; that Dawn duplicates/validates indirect-args buffers; the
bandwidth arithmetic for the donor clears (25.2 MB at ~200–400 GB/s ⇒ ~60–125 µs); the batching
legality in A2's table (follows from usage-scope rules and the code's binding lists, but no arm has
been built); and every per-advance µs figure below, which multiplies verified pass counts by the
brief's 11 µs CPU / 13 µs GPU unit.

---

## Ranked structural candidates

| # | candidate | removes per advance (64³, budget 1) | CPU / GPU at 11+13 µs (GPU floor at 4.758 µs) | confirming measurement |
| ---: | --- | --- | --- | --- |
| **1** | **Lag the recovery finish.** Encode the 8 recovery batches only when the previous step was *rejected*, using the same asynchronous evidence P1 already reads (`planPressureCycleBudget`, `:2294-2323`). Today `encode():446` exempts everything past `finishStart`. | **163 passes**, 163 pass-begins, 326 `setBindGroup`, 8 × 4 B clears | **−1.79 ms CPU, −2.12 ms GPU** (floor −0.78 ms) | Paired lockstep A/B with `probe-cycle-budget-dawn.mts` (`docs/research/uniform-pressure-granularity-2026-09-19/`), arms = finish-always vs finish-lagged, on `minimal-power-dam-break-64` (the null case, which never converges and so always needs recovery) **and** `large-power-dam-break` (converges every step). Paired because the lane is bimodal. |
| **2** | **Fold the four donor-sum clears** into generation-tagged records (1 u32/cell) or into an unconditional first write. | 4 `clearBuffer`, **25 165 824 B** of writes, 4 blit regions | ~−60 to −125 µs GPU (bandwidth); CPU negligible | Cheap upper bound first: an arm of `tools/benchmark-uniform-volume-work-dawn.ts` with the clears stubbed (numerically wrong, bandwidth honest) to price the prize before building the tag. |
| **3** | **Kill the phi window.** It spends 3 passes + 2 × 1 024 B copies to convert 2 vertex passes from direct to indirect — and indirect is 15–25 µs vs 3–6 µs on this lane (`:328-334`). | net **+1 pass** removed, 2 `copyBufferToBuffer`, 2 indirect → 2 direct launches | ~−11 µs CPU, ~−35 µs GPU | Existing kill switch: `phiWindowForQA: false` (`:936`). Single-flag paired A/B, no code change. |
| **4** | **Batch the 5 legal pass groups** (A2 table): surface-volume reduction 14→1, phi census 3→1, page prep 4→2, deficit 2→1. | **18 passes**, 18 pass-begins, 18 `setBindGroup` | −0.20 ms CPU, −0.23 ms GPU (floor −0.09 ms) | Same-scene bit-identical oracle (the batched arm must reproduce `volumeCellSum` exactly) plus the stage-seam trace; the seams are spliced, not separate passes, so the partition survives. |
| **5** | **P3 — fuse smoother + residual + restrict** so the pressure stream stops being one pass per colour. Requires the smoother to read pressure through storage textures instead of sampled ones, which is what `:447-449` currently forbids. | ~200–700 passes (prior-art estimate) | −2.2 to −7.7 ms CPU, −2.6 to −9.1 ms GPU | Prior art already sized it; the *precondition* to measure first is whether a storage-texture smoother is numerically identical (bit-exact `mgMeasureFineResidual` over 60 frames on mini64). |
| **6** | **P2 — resident coarse solve for levels ≤16³.** At 64³, levels L2–L5 are 125 / 27 / 8 / 1 workgroups and account for 1 092 of 2 187 planned passes; a single resident dispatch replaces the ladder. | ~200 passes | −2.2 ms CPU, −2.6 ms GPU | Level-resolved timestamp seam (the plan already tags `activeLevel`), then a resident-arm A/B on mini64. |
| **7** | **Turn off total surface volume** (15 passes + a 1 098 500 B clear + a 1 098 500 B texture copy for one global scalar). | 15 passes, 1 clear, 1 texture copy | −0.17 ms CPU, −0.20 ms GPU + copy bandwidth | `totalSurfaceVolume: "off"` is a shipping toggle (`uniform-geometric-parameters.ts:70-72`). Measure cost **and** the volume-drift regression it is there to prevent; this is a trade, not a free win. |
| **8** | **Specialize grid dims as `override` constants.** | 0 passes; possibly cheaper kernel bodies | unmeasured | Rank last: it roughly triples pipeline count (105 → ~270) against a startup budget that took a campaign to reach 7.21 s (`docs/research/uniform-startup-compilation.md`). Measure one entry point (`mgSmoothColour`, 1 688 of 2 187 passes) specialized per level before touching anything else. |

Candidates 1–4 together remove **186 of 753 passes (25 %)** and 25.2 MB of clear traffic, with no
change to the numerics and two of them behind flags that already exist. Nothing in this list
proposes weakening a lane, raising a timing ceiling, or reducing solver accuracy: candidate 1
preserves recovery exactly, it just stops paying for it on steps that were accepted.
