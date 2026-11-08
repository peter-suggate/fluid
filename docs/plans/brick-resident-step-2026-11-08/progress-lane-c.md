# Lane C: fast pipeline compilation

Goal: from cold to first frame for seep and letters in well under 30 s, a warm reload in a few seconds, and no pipeline over about 5 s cold.

Outcome: no compile-time fix landed. The lane measured where the time goes and found the mechanisms; the recommendations at the end are ranked by the cold-build seconds each should save. The only code landed is diagnostic tooling (see "Tooling").

## COORDINATION

- 2026-11-08: Lane C owns the compile structure of surface advect in `uniform-mixed-surface.ts` (coordinator ruling). It covers `umAdvected` and the WGSL helpers it reaches, and the entry points `advectDeferred`, `advectOwners` and `advectWalls`, with their twins' compile list (`compile`/`keyed` in initialize).
  - The changes are to compile structure only: call sites, loops and specialisation. Step time and drift must be unchanged or better.
  - Lane T rebases S1 onto this structure.
- `uniform-mixed-surface-volume.ts` `clearBand` is in Lane C's scope as well.
- Lane C made no edits to either file. Both are back with their owners.

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
