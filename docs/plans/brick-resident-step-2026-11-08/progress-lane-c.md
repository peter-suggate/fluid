# Lane C: fast pipeline compilation

Goal: from cold to first frame for seep and letters in well under 30 s, a warm reload in a few seconds, and no pipeline over about 5 s cold.

Outcome: no compile-time fix landed. The lane measured where the time goes and found the mechanisms; the recommendations at the end are ranked by the cold-build seconds each should save. The only code landed is diagnostic tooling (see "Tooling").

## COORDINATION

- 2026-11-08: Lane C owns the compile structure of surface advect in `uniform-mixed-surface.ts` (coordinator ruling). It covers `umAdvected` and the WGSL helpers it reaches, and the entry points `advectDeferred`, `advectOwners` and `advectWalls`, with their twins' compile list (`compile`/`keyed` in initialize).
  - The changes are to compile structure only: call sites, loops and specialisation. Step time and drift must be unchanged or better.
  - Lane T rebases S1 onto this structure.
- `uniform-mixed-surface-volume.ts` `clearBand` is in Lane C's scope as well.
- Lane C made no edits to either file. Both are back with their owners.
- 2026-11-08, phase 2 (coordinator): Lane C owns `uniform-mixed-surface.ts` and `uniform-mixed-surface-volume.ts` for A, the staged split of the solid-aware general advect (advectDeferred twins, advectOwners). Lane T builds the fused transport pass (D) on these stages; the stage boundaries and the record layout go to Lane T once settled. No GPU timing until the combined gate round.
- 2026-11-08, phase 2, H (coordinator): Lane C edits these for background-priority compilation:
  - `lib/core/gpu-compilation-manager.ts` (priority scope, background hold);
  - a new `lib/core/gpu-background-pipelines.ts`;
  - in `uniform-narrow-band-flip.ts`, only the pipeline compile callback (the line after `const entryPoint=`, not the peer-modified list or entryPoint lines), `compile`, `run` and a new frame predicate. Peer hunks (NB_CELL_MASS, nbSeedTarget, the removed extension and advectWalk lines) are untouched;
  - `uniform-surface-work-control.ts`, the surface geometry owner (`uniform-mixed-surface-geometry.ts`) and the forces owner (`uniform-mixed-forces.ts`), at their compile and launch sites only;
  - `uniform-mixed-frame.ts` (a readiness guard at the top of `advance`);
  - `webgpu-uniform-reference.ts` (the hold during `create` and a frame-level wait in `advanceTo` before its first queue write).
- 2026-11-08, phase 2, H follow-up (coordinator): Lane C makes the timing probes wait for compile idle before their timed steps: `tools/probe-reservoir-bulk.ts`, `tools/probe-tier-budget.ts` (only `construct` and its two `constructed` lines; the peer's extension hunks are untouched) and `tools/profile-solver-construction.ts`.

## Tooling

- `tools/profile-solver-construction.ts` records the phase in which each pipeline is first bound by `setPipeline` (`firstUse`: construction, sample or the step number) and how often it is bound (`uses`). It prints a `usage` summary by phase, and the report lists the pipelines from slowest to fastest.
- `tools/probe-pipeline-compile.ts --clone` gives every pipeline its own shader module, built from the same source.
- Scratch tools are in `~/code/me/fluid-scratch/lane-c/`:
  - `bin/inline-size.py`: the size of the WGSL that one entry point inlines, by callee.
  - `bin/wgsl-variants.py`: writes stubbed-helper variants of a dumped module for the probe.
  - `mtl/compile.m`: times one MSL compile in two phases, library and pipeline, at the default or the size optimisation level.
  - `mtl/conc.m`: N concurrent Metal compiles in one process, each into a fresh shader cache, with distinct code.

## Measurements

All runs used a cold Metal cache (`FLUID_METAL_SHADER_CACHE` set to a fresh directory) and the `timing` runner. Seep's pipeline set was dumped from the tree after S1 was reverted: 426 compute pipelines.

### Cold build of seep

| build | wall |
|---|---|
| app construction, S1 tree | 655 s (coordinator measurement) |
| app construction, S1 reverted | 217-229 s |
| probe, all 426 pipelines, N=3 | 174 s wall, 522 s summed |
| probe, all 426 pipelines, N=1 (true serial) | 327 s |
| probe, the 423 pipelines other than the three heaviest, N=3 | 82 s wall (163 s serial) |
| the same 423, N=8 | 81 s wall |

### True serial cost (N=1, cold), ranked

| pipeline | serial |
|---|---|
| `advectDeferred`, merged tiles + `umCubicDeferred` + `umSolidTwinned` | 81.4 s |
| `advectDeferred`, regular h + `umSolidTwinned` | 54.4 s |
| `advectOwners`, regular h, general (solid-aware, no twin) | 28.3 s |
| `advectParticlesDeferred` (NB-FLIP extended) | 6.5 s |
| pressure authority `surface` | 3.4 s |
| `pRowsG`, `advectDeferred` merged free, `pRows`, `advectFine`, `remapFaces`, `surfaceHeld`, ... | 2.3-3.3 s each |
| the remaining 410 or so | under 2.3 s each |

- The three surface pipelines cost 164 s of the 327 s.
- Their solid-free twins are cheap: 2.7 s merged and 1.4 s regular, with the fast twin at 1.6 s.
- `clearBand` takes 88 ms serial. Its 18-27 s in the parallel runs is pure waiting (see "Completion order"). The same applies to most of the "slow" tail of a parallel run.

### First-frame usage, seep, 60 steps

| first bound | pipelines | compile (contended) | serial estimate |
|---|---|---|---|
| construction only | 31 | 9.6 s | 7 s |
| step 1 | 311 | 570 s | n/a |
| steps 2-60 | 0 | 0 | 0 |
| never | 84 | 39.4 s | 26 s |

- Of the 84 never-bound pipelines, 28 are NB-FLIP stage entries (18 s contended) and 16 are surface work control.
- The letters usage run was stopped before it finished, so letters is not measured.

### Why more concurrency did not help

- **The cap is two compiles.** In the Dawn path (dawn.node 0.6.0), exactly two MTLCompilerService processes are busy at any time, at about 100% each. Node itself stays at about 30%, so it is not Tint-bound. This was sampled every 3 s through a whole run of the 423-pipeline set at N=8. The wall is therefore about serial/2 whatever N is: 327 s gives 174 s, and 163 s gives 81 s.
- **Metal is not the limit.** A plain Objective-C process (`mtl/conc.m`) ran 32 distinct 1 s compiles in 4.4 s, about 7x. Each thread made its own `newLibraryWithSource` plus an async `newComputePipelineState`. With 8 compiles the wall was 1.1 s. Setting `shouldMaximizeConcurrentCompilation` changed nothing. The two-compile cap is therefore in how Dawn issues the compiles: its worker pool, a lock, or its synchronous calls. The synchronous-mode and thread-stack checks did not run in time, so which of these it is remains unknown.
- **The earlier "flat at about 74 s" readings** came from the 81 s heaviest pipeline combined with the two-compile cap. They did not show a Metal limit.
- **Completion order.** A pipeline's promise does not resolve before those requested ahead of it:
  - With the twinned pipeline requested first, the free twin (2.7 s alone) resolved at 84 s, whether it shared the module or had its own (`--clone`).
  - With the free twin requested first, it resolved at 2.7 s.
  - So in any parallel run, per-pipeline milliseconds after a heavy request are waiting, not compiling. Rank pipelines by N=1 runs only.

### The advect compile mechanism

- **The time is in the Metal back end.** For the merged twinned `advectDeferred` MSL (244 KB), `newLibraryWithSource` takes 1.0 s and `newComputePipelineState` 120 s. At `MTLLibraryOptimizationLevelSize` the pipeline step drops to 55 s, but WebGPU and Dawn do not expose that option.
- **Register pressure is high.** The resulting pipeline reports `maxTotalThreadsPerThreadgroup` = 64, which fits the 1.5-1.9 ms "fat general kernels" of the deferred pass in earlier plans.
- **The cost is spread across the inlined body and grows faster than its size.** The entry point inlines 5.4 MB of WGSL before specialisation, and `umAdvected` alone 2.7 MB. Variants at N=1, against the 81 s base:

  | variant | serial |
  |---|---|
  | no `umWallContact` | 51 s |
  | no tail (wall contact, embedded air, released walls) | 48 s |
  | no `umCubicPhi` | 61 s |
  | no velocity sample | 72 s |
  | no trace and walk | 53 s |
  | no tail and no cubic | 37 s |
  | no foot and no tail | 21 s |
  | no `umAdvected` | 2.4 s |

  No single helper is responsible. A 4-5 way split of the body is needed to get each piece near 5 s.
- One variant is an outlier: dropping embedded air and released walls only measured 116 s. Treat it as noise, or as a back-end heuristic flip.

### Negative results

- **Loop unrolling.** Replacing every constant loop bound below 64 with a runtime-opaque bound changed nothing: 55, 78 and 70 s at N=3.
- **Separate shader module objects (`--clone`).** No change, in either compile time or completion order.
- **Workgroup size.** 32 or 128 instead of 64: 78 s and 79 s.
- **`shouldMaximizeConcurrentCompilation`.** No change in a standalone process.
- **Earlier N=3 "remove one helper" bisects** (`var-tw`, `var-free`) were confounded by the completion-order effect. Only the N=1 table above is valid.

### Not measured

- The warm reload.
- The effect of a shared-helper edit on the cache.
- The browser's compile concurrency.
- The letters cold build and its usage.
- Seep and letters step time and drift for any change: none was made.

## LEARNINGS

1. Cold seep is about 327 s of serial Metal back-end work: 655 s on the S1 tree, and 217-229 s of app construction with S1 reverted. Dawn runs it two compiles at a time, so the wall is about serial/2.
2. Three solid-aware surface pipelines are half the serial work: the two `umSolidTwinned` `advectDeferred` twins (81 s and 54 s) and the general `advectOwners` (28 s). Nothing else is over 6.5 s.
3. Concurrency did not help because Dawn's path caps compiles at two, not because of a per-core MTLCompilerService limit. Metal compiles about 7 at once from one plain process. Wall readings looked flat because the 81 s pipeline bounded each run and the cap hid the rest.
4. Pipeline promises resolve in request order. A slow pipeline delays every promise requested after it, and per-pipeline times from a parallel run mislead. Rank by N=1 runs only.
5. Seep binds 311 of its 426 pipelines in step 1 and no new ones in steps 2-60. 31 are bound only during construction and 84 are never bound (about 26 s serial).
6. The advect cost is in AGX code generation, which grows faster than the size of the fully inlined `umAdvected` body (the kernel is limited to 64 threads per threadgroup). It is not caused by loop unrolling, the workgroup size or shared modules. No single helper is responsible.

## Phase 2: the staged general advect (A) and the compile cap (B)

### A: the staged split, implemented, not yet gated

- **The stages.** The solid-aware general advect now runs as launches over a staged list.
  - advectDeferred's two general twins (`umSolidTwinned`) are compiled with `umStagedAdvect`. Each entry runs the preflight, the early and cheap transports, the foot and the Euler transport, then umAdvected's buried exit, and then it is staged (`umStageVertex`).
  - advectOwners no longer traces in place. After the foot and the Euler transport it stages into the regular-h list. Its old in-place path deferred hits, slow samples and released-wall changes to umAdvected, and for a non-wall vertex that gives the in-place value where the in-place value is valid. Staged entries go straight through umAdvected's own continuations, so the result is the same in intent.
  - `advectStagedTrace`:
    1. sets the wall flag and, under "clip", umOneStep;
    2. traces with `umVertexTrace` (the general sampler);
    3. walks with `umWalk`;
    4. computes the moved smoothing.
  - `advectStagedCell`:
    1. samples the cell with `umVertexCell`;
    2. takes `umCubicPhi`, unless the entry is on the merged list (`umCubicDeferred`) with no hit and no wall. Such a cubic is listed for advectCubicDeferred, exactly as umAdvected listed it.
  - `advectStagedTail`:
    1. applies `umWallContact` from the trace's travel;
    2. applies `umEmbeddedAir` at a hit;
    3. applies `umReleasedWalls`, then the source;
    4. records the region and stores.
  - The staged kernels have no solid twin, because advectOwners stages whether or not a solid is present. With no solid present, their solid tests answer as before (`umSolidEnabled`).
- **The record.** Each staged entry is 12 words at `UM_STAGE_BASE+12n`, after the trace list in binding 10.
  - The merged list counts from the start (`header[6]`) and the regular-h list from the end (`header[7]`). wallReach zeroes both.
  - Each canonical vertex is staged at most once, so the two lists together hold at most `deferredEntries`. deferredBytes grows by `48*deferredEntries`: 33 MB in seep, authored at capacity, with no runtime growth.
  - The words:

    | words | written by stage A | then |
    |---|---|---|
    | 0 | the vertex word (as umQueueVertex) | unchanged |
    | 1 | flags, set to 0 | HIT and WALL set by the trace; CUBIC and LISTED by the cell |
    | 2..4 | the foot | the walk's end q (trace), then the value in word 2 (cell) |
    | 5 | the smoothing | the moved smoothing (trace) |
    | 6..8 | the shift | the trace's end (trace) |
    | 9..11 | not written | the travel, `umLastTravel` (trace) |
- **Encode, deferred pass, per dispatch:**
  1. the fast twin (merged only);
  2. the free twin;
  3. the general twin (stage A);
  4. Trace, Cell, Tail, each at deferredGrid;
  5. advectCubicDeferred (merged only).
- **Cold serial compile** (seep, N=1, fresh Metal cache; `staged-cold.log`):

  | pipeline | before | after |
  |---|---|---|
  | advectDeferred, merged general (stage A) | 81.4 s | 2.99 s |
  | advectDeferred, regular-h general (stage A) | 54.4 s | 1.86 s |
  | advectOwners | 28.3 s | 2.33 s |
  | advectStagedTrace, merged / regular | n/a | 1.34 / 0.95 s |
  | advectStagedCell, merged / regular | n/a | 1.49 / 1.32 s |
  | advectStagedTail, merged / regular | n/a | 2.43 / 1.68 s |

  - The three heavy pipelines and their six new stages total 16 s serial, against 164 s before.
  - No surface pipeline is now over 3 s cold.
- **Checks so far:**
  - `check:types` is clean.
  - `test:unit` has the one known peer failure only.
  - A 3-step seep check built and stepped (`staged-seep.log`).
  - The combined gate round (seep and letters drift, step time, rest and drop, one drop profile, one cold construction) is pending with the coordinator.

### B: where Dawn caps compiles at two

- **The mechanism is Dawn's default platform worker pool.**
  - Dawn runs each `create*PipelineAsync` as a task on the device's worker task pool. The task makes the synchronous `newLibraryWithSource` and `newComputePipelineStateWithDescriptor` calls (`ComputePipelineMTL.mm`, `ShaderModuleMTL.mm`).
  - The pool comes from `Platform::CreateWorkerTaskPool()`. The default Platform returns `AsyncWorkerThreadPool(kDefaultTaskHandlingJobCount)`, and that constant is 2 (`src/dawn/platform/WorkerThread.h`). The pool is per device (`Device.cpp`: `mWorkerTaskPool = GetPlatform()->CreateWorkerTaskPool()`).
  - dawn.node passes no platform, so the Instance uses the default (`Instance.cpp`: `mDefaultPlatform`). The result is two compile threads per device.
  - Neither dawn.node nor a Dawn toggle exposes the pool size. Lifting it in Node needs a dawn.node build with its own Platform. A second device, or a second process, could compile the same descriptors into the shared Metal shader cache, but the layouts must match for the MSL to match.
- **Chrome is not capped this way.**
  - Chromium's `gpu/command_buffer/service/dawn_platform.cc` overrides `CreateWorkerTaskPool`. Every worker task goes to `base::ThreadPool::PostTask({MayBlock, USER_BLOCKING})`.
  - The foreground pool is `max(3, NumberOfProcessors()-1)` threads (`base/task/thread_pool/thread_pool_instance.cc`), so 9 on a 10-core M1 Max, shared with the GPU process's other foreground tasks.
  - In the browser UI, the floor was therefore the 81 s pipeline rather than the cap. A removes that floor.
  - This comes from reading the source only. It was not measured in a browser.

### H: background-priority compilation, implemented, not yet gated

What it does:
- `requestGPUCompilationAt(priority, compile)` issues the requests `compile` makes at that priority and can later raise them. A raise also pumps, so a raised job can start while background work is held.
- `holdGPUBackgroundCompilation()` keeps background jobs from starting. The reference holds them for all of `create()`, so they start only after every critical and visible request of construction.
- The scope and the hold are realm-wide, not per manager. A device behind a tool's Proxy (the kernel timer, the allocation audit) reaches its manager without being registered, so a per-manager lookup missed it: probes compiled the background pipelines at visible priority with no hold. Pipeline requests are issued synchronously, so the scope open when a request reaches a manager is the one that made it.
- Held background jobs are not counted in the construction progress total.
- `whenGPUCompilationIdle()` resolves once every manager in the realm is idle.
- `GPUBackgroundPipelines` (new, `lib/core/gpu-background-pipelines.ts`) is the owner-side helper:
  - `add` requests a pipeline at background priority and does not await it;
  - `require(keys)` returns undefined when the keys are compiled, otherwise raises them to critical and returns a wait;
  - `get` throws while a pipeline is still compiling or after it failed;
  - a failed background compile rejects every later `require`, so it fails the next advance explicitly.

How a frame waits:
- Encode writes to the queue before it binds, so the wait happens at frame level, before the first queue write.
- Each converted owner has a `pipelinesPending()` predicate over its current flags. It names every background pipeline the coming advance may bind.
- `UniformMixedFrame.pipelinesPending` joins the owners' predicates.
- `advanceTo` in the reference checks it first. If anything is pending, it admits the step but defers it until the wait settles. While it waits, `framePending` stays true and `awaitFrameCompletion` awaits the admission.
- `UniformMixedFrame.advance` refuses, before any queue write, a frame whose pipelines are still pending. There is no alternate path and no skipped stage.
- A failed compile fails the frame through `failMixedFrame`.

Owners converted, largest share first:
- **NB-FLIP:**
  - The air-classification pipelines go to background. The unplanned form is 8 pipelines; the planned form is airPlanSeeds, airPlanPropagate, airClearFine and airClearCoarse@planned. A frame needs one set or the other, or neither when the air clear is fused. `diagnoseBefore` also goes to background and is needed only with `measureTraces`.
  - Pipelines that no stage of the configuration can bind are now omitted instead of compiled. These are:
    - depthTiles and depthSeeds;
    - the surface-splat pair the fused or unfused path does not use;
    - under `previousCrossings`, the cold depth search, the nearest control and nearest entries, and depthReuseSelect@withX;
    - under a canonical coarse solid, arenaStore and arenaGather@derived.
  - A batch naming an omitted pipeline throws "is not compiled: no stage of this configuration binds it" if it ever runs.
  - The coarseOnly batches and airFused stay critical.
- **Work control:**
  - The pipelines the current flags bind are awaited. These are geometryRegional[Stats] and select, the snapshot or current geometry and metadata set, and, with surfaceEvidence, the 7 surface stages. The rest go to background.
  - The serial chain is gone: the frame plan, remap, init and stage pipelines are requested together and awaited once. Before, each await re-queued behind the whole build, stretching the chain from 77 s to 202 s.
  - Tests that flip flags after initialize await `compiled()`.
- **Surface geometry:**
  - Every compile is requested together and awaited once.
  - Under work control, the plain `geometryChanged` goes to background. It is needed only when `regionalGeometrySetup` is off.
- **Forces:**
  - Under cached curvature, the four capillarity pipelines (cacheCurvature, cacheNormals, forceSlopes, curvatureRegions) go to background. They were awaited serially before. They are needed only when surface tension is above zero.

Left critical, and why:
- The 31 construction-only pipelines are bound during construction itself, so construction must await them. Deferring them would only move the wait.
- Geometry's runtime-dependent `regionalPipeline` and `compactBodies` stay critical. Their selection depends on runtime state that the predicate cannot see before the frame.

Expected cold-build saving on seep: about 30 to 45 s out of about 207 s (construction_ms 225 s).
- About 17 s comes from removing the serial tails of work control, geometry and forces. They ended at 202 s, 207 s and 199 s.
- About 28 s comes from taking the never-bound pipelines off the critical path. That is 84 s of contended compile out of a 550 s pipeline sum, against 188 s of busy time.
- This is an estimate from the earlier attribution. It has not been measured.

Gate caveat: background compiles now run during the first frames. A timing gate must wait for them, or the first frames will include compile contention.
- `probe-reservoir-bulk` and `probe-tier-budget` now await `whenGPUCompilationIdle()` after construction, before any timed step. Both report `compileIdle_ms` beside the construction time.
- `profile-solver-construction` reports `construction_ms` (construction's return), `toFirstFrameReady_ms` (the first advance's pipelines compiled, as `advanceTo` waits for them) and `toCompileIdle_ms`. Its steps run after compile idle.

Two GPU tests in `tests/uniform-surface-work-control.test.ts` failed in the combined check, with model word 0 and slope 0. The cause is not H.
- 22d0bc3c (2026-10-20) turned the surface model evidence off by default (`surfaceEvidence=false`). `encode(…,"surface")` then returns before any pass by design: no coverage, no valid support, so advect takes its full paths.
- Both tests read the surface stage's models or slopes without enabling it. They are GPU-only, so the unit runs skip them, and they have failed since that commit.
- No pipeline readiness is involved. Every work-control bind goes through `pipeline(name)`, which throws for a background pipeline that is not ready, and `compiled()` awaits every background name.
- Fix: both tests set `surfaceEvidence=true` before initialize.
- With the fix, the slope test passes on HEAD and on the combined tree.
- The models test then gets past its model checks and fails later, at its first `compareRegionalGeometry` (after "geometry notices small current phi changes"). The regional path writes dirty certificates 1 where the full path writes 0 (512 words), and 3 where it writes 4 (88 words).
  - HEAD with only the test fix fails identically, word for word, so this is not H.
  - It is an older, separate defect in the regional-versus-full equivalence under cache overflow, hidden while the test stopped at line 68. It is parked and not fixed here.

Test: `tests/gpu-background-pipelines.test.ts` uses a fake manager with controllable compiles. It shows:
- a held background job starts only after every critical request and after the hold is released;
- a frame that may bind a pending pipeline waits before its first queue write, and its request is raised above queued visible work;
- `get` before the wait throws;
- a failure rejects every later wait;
- a device behind a forwarding Proxy still compiles at background priority, is held, is raised, and is seen by `whenGPUCompilationIdle`.

## Recommendations, ranked by expected cold-build saving on seep

Baseline: 327 s serial; about 174 s wall in the probe and 217-229 s in the app (S1 reverted).

1. **Lift Dawn's two-compile cap.** Find where dawn.node serialises: its worker pool, a lock, or synchronous calls into Metal. This needs a stack sample during a compile run. Then confirm that the browser (Chrome's GPU process) is or is not capped the same way.
   - At about 7-way, the 423 light pipelines take about 25 s.
   - The wall would then be bounded by the 81 s heaviest pipeline: about 174 s down to about 81 s, roughly 90 s saved.
   - Together with item 2, it reaches about 25-30 s.
2. **Split the solid-aware general advect into launches.** Use the existing `advectTrace`/`advectFinish`/`advectCubic` pattern for the `umSolidTwinned` `advectDeferred` twins and for `advectOwners`. The stages, as the variants suggest:
   - (a) preflight, foot and Euler (about 2.4 s);
   - (b) trace and walk;
   - (c) cell and cubic;
   - (d) wall contact, embedded air, released walls, source and store.

   The cost and the expected saving:
   - Each stage carries an 8-word record per deferred entry. The list holds 690k entries in seep, so that is about 22 MB, authored at capacity, with no runtime growth.
   - It should save about 150 s serial, which is about 75 s of wall at the current cap, and lower the floor set by the heaviest pipeline from 81 s to about 6.5 s.
   - It may also cut the deferred pass's 1.5-1.9 ms step time, through better occupancy.
   - Physics must stay identical, with seep and letters drift gated.
3. **Compile the 84 never-bound and 31 construction-only pipelines at background priority.** A frame that needs one waits for it. This saves about 26 s serial, about 13 s of wall at the cap or 4 s at 7-way.
   - Because of the completion order, the background requests must be issued after every critical one.
   - The encode-time wait needs care: encode writes to the queue before it binds.
4. **`advectParticlesDeferred` (6.5 s)** is next in line once items 1-3 have landed.

Scratch data: `~/code/me/fluid-scratch/lane-c/`. The main files:
- `probe-head-par1.log`: the serial ranking.
- `usage-seep.json`.
- `var-split.log` and `var-split2.log`: the N=1 bisect.
- `heavy/msl*.metal` with `heavy-run.log`: the phase split.
- `light-par3.log`, `light-par8.log` and `light-sample.log`: the two-compile cap.
- `conc.log`: Metal in a plain process.
- `pair-*.log`: completion order.
- `var-loop.log` and `var-ws.log`.
