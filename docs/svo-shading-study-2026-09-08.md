# SVO shading study — hero-garden-hose-x10, 1080p

**The measured 2× specialization is now enabled in the production renderer.** It retains soft shadows, four-sample AO, half-rate cones, full-resolution PBR material shading, and voxel-face edge shading. The timings below are the original controlled prototype measurements; production integration and its validation are documented below.

| Camera | Baseline shading | Candidate shading | Speedup | Baseline dry frame | Candidate dry frame |
|---|---:|---:|---:|---:|---:|
| Preset | 16.253 ms | 7.602 ms | **2.14×** | 21.627 ms | 12.714 ms |
| Orbit +0.25 rad azimuth, +0.10 rad elevation | 17.170 ms | 8.258 ms | **2.08×** | 22.675 ms | 13.566 ms |

These are pooled medians of two 16-sample runs per arm. Shading p95 improves from 17.564 to 8.126 ms in the preset view, and 18.547 to 8.978 ms in the orbit view. The dry-frame improvement is 1.70× and 1.67×, respectively. Source hashes for all recorded renderer, shader, mesh, scene and experiment modules match across the four final comparison runs.

The mesh rasterization draw itself costs about 3.2–3.3 ms; the broader raster preparation/visibility chain costs about 4.8–4.9 ms. Even after this optimization, shading remains more expensive than the mesh draw.

The candidate is not pixel-identical. Geometry and depth are unchanged. Against production cache-enabled images, mean clipped-gamma differences are 0.370/255 and 0.405/255, with about 1.12% and 1.18% of pixels differing by over 8/255. The narrower comparison against cache-disabled references isolates the guide change at 0.094/255 and 0.104/255. This is a measurable lighting-policy tradeoff, not a loss of shadows or AO.

**Evidence:** [acceptance summary](../artifacts/svo-shading-study/acceptance.json), [all raw timing samples](../artifacts/svo-shading-study/experiments.json), [baseline image](../artifacts/svo-shading-study/baseline.png), [candidate image](../artifacts/svo-shading-study/candidate.png), [orbit baseline](../artifacts/svo-shading-study/orbit-baseline.png), [orbit candidate](../artifacts/svo-shading-study/orbit-candidate.png). The four final HDR captures are retained as `.rgba16f.gz` alongside these files.

## Selected candidate

1. Compile out exact-mode visibility from the measured cone-only shader and voxel dielectric continuation from this opaque scene.
2. Specialize the verified single-directional-light roster: one light/sample loop and one shadow visibility channel, instead of the general eight-light closure.
3. Compile out the directional visibility cache, rather than merely turning off its runtime flag. This removes both its demand/population work and its shader branches/bindings. The half-resolution screen-space visibility prepass remains active.
4. For full-resolution relighting, let compatible static receivers share visibility across material IDs while retaining depth, normal, feature, field and motion checks. Materials and PBR are still evaluated at full resolution.
5. Replace exact-receiver recovery's exponential/power threshold calculations with equivalent depth/normal threshold comparisons.

The selected candidate does **not** remove specular/PBR shading or voxel-face edge darkening, reduce AO samples, reduce cone resolution, enable temporal reuse, or disable either requested lighting effect. Feature-reduced variants and the compute variant were measured but did not produce the best tradeoff.

The original experiment used fixed-scene shader transformations. Production now generates the same specialized shader through explicit generator options and selects it only when the publication and active lighting configuration are compatible. Production imports no experiment tool.

## Objective and acceptance

Halve the SVO shading-chain GPU time at 1920×1080. Preserve soft shadows and AO. The user explicitly permits removing other deferred-lighting features where it helps; still investigate execution and algorithm improvements rather than relying only on feature removal.

The target applies to the whole shading chain, not only the final deferred draw. Measure the complete frame separately. Require a complete depth-3 mesh, zero mesh overflow, unchanged geometry/depth for lighting changes, no nonfinite output, and no GPU validation errors. Compare stationary and changed-camera views before selecting a final candidate.

## Measurement plan

1. Build the production renderer and live scene at depth 3; fence maintenance and wait for complete mesh publication.
2. Warm the actual 1920×1080 graph. Record repeated GPU timestamp spans for complete frames, isolated shading, isolated raster preparation/visibility, deferred surfaces, cones/cache, and sky.
3. Ablate AO, shadows, visibility cache, cone resolution, and radiance reconstruction separately. These ablations identify costs; disabling AO or shadows is not a candidate.
4. Specialize unreachable shader branches, partition common/fallback shading, test receiver reconstruction, and test a compute implementation of the deferred closure.
5. Compare HDR captures, depth, finite values, and display-space differences. Repeat baselines to detect cache variability and drift. Use cache-disabled comparisons to separate shading arithmetic differences from nondeterministic cache population.
6. Re-run the winning candidate and baseline in reverse order and at another camera pose. Keep the best feature-preserving and feature-reduced options distinct.

## Test environment and scope

Apple M1 Max, Dawn/Metal, macOS 26.6.2. Production `mesh` primary, depth 3, `voxel-flat` surfaces, default preset camera, 1920×1080. Soft shadows on, AO on (four samples), half-rate cones, full-resolution material relighting, GI off as in production defaults. One authored light, five rigid bodies. The scene is frozen and dry: timing excludes fluid simulation, water optics, and the application final composite/present. The second camera is a fixed changed pose with the moving-camera flag, not continuous camera animation. Scene construction publishes 3,871,505 quads; the 64×64 publication check drew 1,739,623 before the benchmark resizes to 1080p. Scene and renderer buffer census approximately 1.83 GB; this is cumulative allocation accounting, not a peak-resident measurement.

Each scope has three warmup samples and sixteen measured samples, after eight full-frame warmups per arm. Final comparisons use two runs per arm (32 samples). The orbit candidate ran before its baseline, and a fresh preset baseline ran last to check for timing drift. Individual render-pass timestamp windows overlap on this tiler, so their sum is not a stage-cost measurement. The isolation tool records surrounding commands without submitting their passes and submits only the selected production pass set against warmed resources. It uses min-start/max-end timestamp span, never sum-of-pass-windows. Isolated scopes need not add exactly to the complete frame because isolation changes scheduling and cache state.

All GPU work takes the repository exclusive lease; no concurrent Dawn/browser tests. The existing depth-3 probe's pending-mesh assertion conflicts with in-progress editor changes permitting partial publication. The study uses its own entrypoint and requires the final complete mesh; it does not weaken the original probe or its assertion.

The initial older dry-frame benchmark was depth 0, not the production default. It measured approximately 60 ms but is excluded from the target baseline. The dedicated study avoids that default and explicitly waits for mesh publication before any timing.

## Initial results

Representative GPU medians in milliseconds. Full sample arrays and actual pass labels are in `artifacts/svo-shading-study/experiments.json`.

| Experiment | Complete frame | Shading chain | Deferred surface | Cone/cache | Raster |
|---|---:|---:|---:|---:|---:|
| Baseline | 21.89 | 15.99 | 12.71 | 3.15 | 4.78 |
| Baseline repeat | 21.17 | 16.38 | 13.37 | 3.15 | 4.78 |
| Cache disabled | 21.10 | 15.47 | 12.65 | 2.95 | 4.72 |
| AO disabled — attribution only | 15.79 | 10.75 | 8.52 | 1.97 | 4.78 |
| Shadows disabled — attribution only | 15.27 | 10.68 | 8.85 | 1.57 | 4.72 |
| Both disabled — attribution only | 9.37 | 4.39 | 4.00 | 0.33 | 4.78 |
| Quarter-rate cones | 21.96 | 16.25 | 15.47 | 1.25 | 4.78 |
| Bilateral radiance reconstruction | 16.45 | 11.14 | 7.86 | 3.15 | 4.85 |
| Cone-only specialization | 18.48 | 13.43 | 9.50 | 3.21 | 4.85 |
| Remove voxel dielectric continuation | 19.14 | 14.16 | 10.62 | 3.21 | 4.78 |
| Both specializations | 15.79 | 10.29 | 7.08 | 3.15 | 4.78 |
| Cached/exceptional separate draws | 20.32 | 15.07 | 11.27 | 3.21 | 4.78 |
| Both specializations + diffuse-only shading | 15.01 | 9.70 | 6.42 | 3.21 | 4.85 |
| Both specializations + visibility-only material guide | 14.35 | 9.18 | 5.90 | 3.21 | 4.78 |
| Combined lean shader | 13.63 | 8.52 | 5.31 | 3.21 | 4.72 |
| Compute closure + both specializations | 17.24 | 12.19 | 8.65 | 3.21 | 4.72 |
| Single-directional + both specializations | 14.29 | 8.72 | 5.70 | 3.21 | 4.72 |
| Single-directional + visibility guide | 13.70 | 8.39 | 4.98 | 3.21 | 4.78 |
| Same, runtime visibility cache disabled | 13.11 | 8.00 | 4.92 | 2.95 | 4.78 |
| Single-directional + Lambert | 13.96 | 8.45 | 5.18 | 3.21 | 4.72 |
| Selected: single-directional + guide + cache compiled out | 12.71 | 7.73 | 4.59 | 2.95 | 4.78 |

Sky is about 0.066 ms at the timestamp quantization floor. It is not the bottleneck in the production no-GI variant.

## Findings

- The premise is confirmed: the shading chain is roughly 3.4× raster time. About 80% of shading time is in the full-resolution deferred closure, not the explicitly named cone pass.
- Quarter resolution reduces cone work but increases exceptional full-resolution work. It is effectively a wash and degrades more of the image. Do not use it as the primary optimization.
- Removing specular/PBR costs alone yields a small saving. Most of the cost follows visibility and the large shader call graph.
- Specializing away exact-mode visibility and voxel dielectric continuation together saves about 5.8 ms. The likely mechanism is lower register/private-state pressure and a smaller call graph; no hardware occupancy counters have yet been collected, so this mechanism is an inference, not a measured counter result.
- Keeping two deferred draws for cached and exceptional pixels preserves the image closely but duplicates reconstruction/classification and leaves the expensive exceptional shader. The first prototype is not competitive with specialization.
- Full-resolution relighting can reuse visibility across material IDs without reusing material color: its surface material is evaluated at full rate. The prototype retains depth, normal, feature and motion checks. This is a changed reconstruction policy, not a bit-exact optimization, and it needs boundary-image review.
- The lean shader removes specular response and voxel-face edge darkening, specializes cone-only opaque shading, relaxes the material guide only for full-resolution relighting, and replaces exact-receiver threshold exponentials with equivalent depth/normal threshold comparisons. It retains cone sample counts and soft-shadow/AO algorithms. At 8.5 ms this intermediate feature-reduced variant did not demonstrate the target. The final PBR-preserving candidate above supersedes it.

The compute prototype executed successfully with unchanged depth, but lost against the equivalent fragment specialization (12.19 versus 10.29 ms for shading). It is rejected on this GPU rather than assuming compute is inherently faster.

Single-directional specialization retains the material model and narrows the light loop and visibility channel access. Relative to the cache-disabled reference it changed only 37 pixels, one over 8/255, and no depth values. Combining it with the material-independent guide and disabling the runtime visibility cache reaches approximately 8.0 ms. This intermediate runtime-cache-disabled result was borderline. Compiling the cache out brought the final result to 7.6–7.7 ms and passed the changed-camera check.

## Image evidence

All compared captures have finite RGB. Lighting-only specializations preserve depth exactly on this scene. With the visibility cache disabled, cone-only specialization changed one pixel and removal of dielectric continuation changed three pixels out of 2,073,600; the cached/exceptional split changed two. These are not claimed bit-exact.

Repeated baseline frames with the directional visibility cache enabled differed in 3,812 RGB pixels, including 1,005 over eight gamma-encoded byte levels. The cache deduplicates requests atomically and chooses a representative normal; this limits the meaning of cache-on bit-exact comparisons. A cache-disabled reference is needed for arithmetic/coverage validation.

The visibility-only material guide changed 53,675 pixels (2.59%), with a mean clipped-gamma difference of 0.094/255 and 5,602 pixels over 8/255. The lean shader's mean was 1.435/255 with 81,343 pixels over 8/255. These figures use a simple clipped gamma transform, not the application display grade. PNGs use the repository display-grade helper. Bilateral reconstruction changed depth at 896,712 pixels, so it is unsuitable as a drop-in lighting-only optimization without repairing its depth contract.

## Production integration

Implemented on 2026-09-08:

- The production factory disables the persistent directional-light cache at shader compilation and enables a second deferred-lighting pipeline. Primary visibility, cone prepasses and sky retain their generic pipelines; only the full-resolution surface closure is duplicated.
- `publishOpaqueSurfaceCapability` inspects the canonical SolidWorld material IDs, ordered fill patches, rigid-body material IDs and primitive material IDs against the published material table. Unused glass table slots do not disable the optimization. Glass edits and unknown material IDs do. Missing capability declarations conservatively use the generic shader.
- Draw-time selection requires that opaque proof, no thin/thick glass records, exactly one directional light, a ready derived-lighting hierarchy, cone mode, a reduced prepass, GI disabled and full-resolution relighting. Light roster, mode and readiness changes therefore fall back immediately; they do not wait on asynchronous shader recompilation.
- The specialized shader retains normal/depth/feature/field/motion guidance and moving-owner checks while reusing visibility across material IDs. Full-resolution materials and PBR remain intact. The generic shader retains dielectric continuation, arbitrary lights, exact visibility and material-sensitive reconstruction.
- Algebraically equivalent receiver threshold comparisons are shared by both shaders. The production render panel reports the removed directional cache as unavailable.

Validation: all 28 focused specialization, composition, frame-partition and material/glass tests pass. The specialization test compares the entire generated source, ignoring whitespace/comments, against the measured source transformation. The actual hero-garden-hose-x10 CPU publication passes the opaque proof and publishes one directional light and no glass. Workspace typechecking still reports unrelated existing errors, with none in the changed shading/integration files; `git diff --check` passes.

A new isolated production GPU timing run has not yet been performed: the editor thread owns the GPU for its canonical regression suite and browser acceptance. The table above must not be relabeled as a new production run. The integration changes rendering capability metadata only; it does not change simulation, topology, publication transactions, terrain or editor behavior.

## Reproduction

Winning prototype:

```sh
FLUID_PROBE_LIGHTING=1 \
FLUID_SHADING_DISABLE_CACHE=1 \
FLUID_SHADING_SHADER=cone-only-opaque-one-light-visibility-guide-cheap-guide \
FLUID_SHADING_ARMS=baseline,baseline-repeat \
FLUID_SHADING_OUT=/tmp/svo-shading/pruned-cache \
node --import tsx tools/run-webgpu-exclusive.ts \
  --expose-gc --import tsx tools/run-svo-shading-study.ts
```

Add `FLUID_SHADING_CAMERA=orbit` for the changed-camera check. Remove `FLUID_SHADING_DISABLE_CACHE`, set `FLUID_SHADING_SHADER=baseline` and `FLUID_SHADING_REFERENCE=1` for the old cache-enabled reference. With neither override, the harness uses the newly optimized production factory.

Full initial ablation matrix:

```sh
FLUID_PROBE_LIGHTING=1 \
FLUID_SHADING_REFERENCE=1 \
FLUID_SHADING_OUT=/tmp/svo-shading/depth3 \
node --import tsx tools/run-webgpu-exclusive.ts \
  --expose-gc --import tsx tools/run-svo-shading-study.ts
```

Set `FLUID_SHADING_ARMS=baseline,no-cache,baseline-repeat` to shorten an experiment. `FLUID_SHADING_SHADER` selects test-only source transformations (`baseline`, `cone-only`, `opaque`, `cone-only-opaque`, `split`, `cone-only-opaque-diffuse`, `cone-only-opaque-visibility-guide`, or `cone-only-opaque-visibility-guide-diffuse-cheap-guide-no-edge`). `FLUID_SHADING_COMPUTE=1` is the compute prototype. These tools are not imported by production.

```sh
node tools/compare-svo-shading-images.mjs \
  /tmp/svo-shading/depth3/no-cache.rgba16f \
  /tmp/svo-shading/visibility-guide/no-cache.rgba16f
```

All final GPU runs completed with no validation errors, no nonfinite RGB, complete mesh publication and zero mesh overflow. Both final camera comparisons preserve every depth half-word. The workspace typecheck still has unrelated pre-existing errors; none are reported in the shading study tools at the final check. The study and shading integration do not alter solver, sparse topology, publication transactions, terrain boundary or live editor behavior.
