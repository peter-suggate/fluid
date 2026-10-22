# Pool party rendering and picking

The pool uses a 50 mm fluid lattice and a separate 6.25 mm visual octree. Imported assets keep their native signed-distance samples and their complete PBR palettes; grass retains its 3.125 mm source sampling. These changes do not reduce render resolution, visual refinement, shadow samples, or material detail.

## Missing mesh asset during picking

The `Mesh asset diving-board-ladder has not loaded` exception came from the window's synchronous scenery picker. The render worker had loaded the volume, but the window had only received the scenery catalog.

The worker waits until all of the scene's scalar fields are loaded, then publishes each unique asset's field before publishing the catalog that makes it pickable. This also covers draws during initialization. In cross-origin-isolated browsers, brick views share the decoder's existing immutable `SharedArrayBuffer`; there is no second download or copy of the distance samples. Other hosts clone only the scalar channel once per asset. Spatial material IDs and normal channels remain in the renderer. Picks still intersect the actual surface, including gaps within the asset bounds.

CPU marching now evaluates only distance until it accepts a hit, then computes its normal. Already-occluded scenery is bounded by the nearest hit found so far.

## Rendering changes

- Imported proxy mesh faces were allocated, culled, transformed and then discarded in the fragment shader. The mesh extractor now rejects them in both its count and emit phases. Native sampled surfaces still draw them, and the voxel field remains available to lighting. The separate raster-shadow-map mode retains its proxy casters.
- Distance-only GPU queries skip source normal decoding and material voting. The accepted surface still uses the complete PBR sampler.
- Native opaque shadow rays stop after their first exact blocker. Camera rays continue searching for the nearest surface.

## Measured bottlenecks and results

Measured on an Apple M1 Max with 32 GiB unified memory, using Metal through Node 25.8.1 and Dawn, at 1000×720. Both captures use the same pool scene and camera, 6.25 mm visual cells, 50 mm fluid cells, surface meshing, split raster primary visibility, full PBR, and cone lighting at scale 0.5. Baseline is commit `86afa154`; the comparison includes the rendering changes above. Each run warms up four frames and measures a 12-second loop.

These are **dry scene rendering measurements**, excluding fluid simulation and water compositing. They are not live simulation FPS. WindowServer competed for GPU time in both captures; the results establish the improvement under these desktop conditions, not an isolated hardware ceiling.

| Metric | Before | After |
| --- | ---: | ---: |
| Median render frame | 187.34 ms | 136.69 ms |
| 95th percentile | 310.19 ms | 168.72 ms |
| Median JavaScript encoding | 0.568 ms | 0.565 ms |
| Live mesh records | 32,414,185 | 15,823,181 |
| Allocated GPU mesh arena | 2 GiB | 1 GiB |

Median frame time decreased **27.0%**, equivalent to **37.1% greater rendering throughput**. The mesh arena saves **1 GiB**, without reducing the retained native volume data.

The dominant bottlenecks are GPU work:

- Deferred dry lighting remains the largest isolated stage: **83.78 → 77.52 ms** average. Native shadow intersection and distance queries are part of this work.
- The composite bucket beginning at mesh rasterization fell from **95.77 → 39.02 ms**. This supported removing faces that the fragment shader already discarded.
- The composite bucket beginning at imported native surfaces fell from **32.58 → 24.98 ms**. Native volume tracing remains a substantial cost.
- CPU frame encoding is under 0.6 ms; GPU submission-to-completion accounts for almost all measured frame time.

Composite buckets include every stage until the next Metal pass boundary; they are **not isolated kernel timings**, and overlapping GPU intervals must not be added as exclusive costs. The reducer now restricts analysis to the final measured frame starts, excluding initialization and warmup with identical labels. It retains 51 complete baseline frames and 83 complete optimized frames after clipping boundary frames.

Local evidence: [baseline report](../artifacts/pool-timing-before/report.html), [optimized report](../artifacts/pool-timing-after/report.html), [baseline frame](../artifacts/pool-timing-before/frame.png), [optimized frame](../artifacts/pool-timing-after/frame.png), and [image comparison](../artifacts/pool-timing-after/image-comparison.json). Generated captures are ignored by Git.

Reproduce from the repository root (Metal and Xcode Instruments required):

```sh
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 \
  FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes \
  FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1 \
  node --import tsx tools/profile-svo-render-xctrace.ts \
  --scene=backyard-pool-party --scene-module=tools/preview/pool-party-live.ts \
  --resolution=1000x720 --refinement=3 --traversal=raster-primary \
  --shading=split --screen-space-pixels=0 --cone-scale=0.5 --warmups=4 \
  --timing-only --out=artifacts/pool-timing-repro
```

## Verification

The worker-handoff regression test picks the actual diving board with no full asset/PBR volume registered in the window and verifies that empty space inside its bounding box remains a miss. A shared-buffer test covers brick seams and confirms that structured cloning shares the scalar allocation.

`node --import tsx tools/blender/check-volume-distance-gpu.ts` compares 32,768 GPU queries across the diving board, travertine, turf and parasol. The largest scalar/shaded distance difference was 3.73e-9 metres (compiler floating-point rounding). Both the normal cone-lighting and raster-shadow shader variants compile.

The settled before/after images have **99.989% identical pixels** after the production AgX display transform: 78 of 720,000 pixels changed, and only one differed by more than two channel levels out of 255. Neither capture reported GPU validation errors. Chrome loaded the live pool and successfully selected `DIVING/ASSEMBLY`; [selection screenshot](../artifacts/pool-performance-live/diving-board-selected.jpg). `npm run check:types` passed; `npm run test:unit` passed with 900 passing tests and 45 skipped.

Full GPU-counter captures consumed excessive profiler memory and were discarded as performance evidence. Final measurements use Metal timing-only captures. The benchmark now reports actual visual cell size separately from fluid cell size, includes mesh memory and validation status, and can capture a final frame after the timing window for image comparison.

## Primary rasterization follow-up

An isolated replay of each production render pass, with GPU timestamps and warmed fixed-camera resources, confirms that the smooth mesh draw itself is expensive: **39.98 ms**, separately from **12.26 ms** for imported native fields. The earlier composite-bucket timing was not sufficient attribution on its own.

Removing all 128 turf instances leaves **7,907,840** drawn mesh records versus **7,891,528** with turf. Grass is therefore not the source of the mesh explosion. The separate imported-surface pass drops to **7.54 ms** without turf; the mesh pass remains expensive (49.15 ms in that run, with desktop contention). Removing grass alone does not restore the old rasterization cost.

A GPU census of the actual culled draw list attributes 3,939,763 records to foliage, 2,208,005 to SolidWorld pool/deck geometry, 935,754 to tree/shrub structure, 336,866 to the house wall and 297,672 to guests. No turf records occur in this draw; imported turf has its own surface pass.

The renderer previously submitted one four-vertex instance per record. The new triangle-list stream submits all visible records in one instance, with three vertices for each DMC triangle and six corners for ordinary quads. It preserves record order, strip winding, quad diagonals, quantized vertex positions, shading normals and material identities. The original indirect header remains intact for shadow-map consumers and diagnostics; a separate four-word argument block serves the continuous primary stream. Culling and the unculled diagnostic path both populate it.

The GPU regression test compares instanced strips against the continuous stream for triangles, quads, both windings and mixed quad/triangle records. It checks exact pixel coverage, interpolants and identities. The profiling helper also rebuilds untimed prerequisite passes for every sample, so a pass using depth `load` receives the real preceding depth buffer rather than its own previous output.

Sampled boundary queries now reject brick coordinates outside exact occupied bounds before probing the sparse hash table. Three previously reserved header words store these bounds, without increasing the field allocation. Empty, legacy or unrepresentably large fields keep the original lookup. This avoids unnecessary normal-sampling lookups across geometry far from the localized diving-board boundary.

With the corrected pass replay, the same M1 Max, 1000×720 camera, DMC mesher, 6.25 mm visual cells, 50 mm fluid cells and full PBR give:

| Isolated mesh draw | Median | 95th percentile |
| --- | ---: | ---: |
| Before continuous stream | 39.65 ms | — |
| Continuous stream | 16.19 ms | 16.32 ms |
| Stream and bounded boundary lookup | 15.60 ms | 15.79 ms |

The combined improvement is **60.7%**. Each isolated timing uses four warmups and twelve GPU timestamp samples. The additional 0.59 ms boundary-lookup difference is small relative to desktop variability; the continuous stream accounts for the clear majority of the gain. Final dry frame median is 111.13 ms over a short three-second window; this is still GPU-limited and excludes water and fluid simulation. It does not establish live simulation FPS or a return to the earlier two-millisecond mesh cost. The 7,891,528 drawn records, 15,823,181 live records and 1 GiB mesh arena remain unchanged.

GPU validation was enabled for the final captures and reported no errors. The final image is **bit-identical in linear color and depth**, with zero changed pixels after the display transform. Evidence: [final timing](../artifacts/pool-primary/final.log), [frame](../artifacts/pool-primary/final.png), and [comparison](../artifacts/pool-primary/final-comparison.json). Explicit GPU tests verify stream coverage/interpolation and bounded versus legacy boundary samples/normals. `npm run check:types` passed; `npm run test:unit` passed with 918 passing and 46 skipped tests. Grass appearance is unchanged; its separate roughly five-millisecond cost remains a possible later target.

For direct pass attribution, run `tools/benchmark-svo-dry-frame-gpu.ts` with the same scene/mesher settings and:

```sh
export FLUID_SVO_DRY_FRAME_PRIMARY_TIMING=1
export FLUID_SVO_DRY_FRAME_PRIMARY_TIMING_LABELS='Voxel surface mesh rasterization,Imported native voxel field surfaces'
export FLUID_SVO_DRY_FRAME_TIMING=gpu
export FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=3
```

The result JSON includes both isolated pass timings. Local diagnostic evidence is in `artifacts/pool-primary/`.

## Sample coverage culling

The 7.9 million records passing back-face and frustum checks are not all raster-visible. Most fine DMC triangles in the overview fall between the single-sample primary pass's pixel centres. The compute culler now projects each candidate triangle with the draw's camera transform and rejects it only when its expanded bounds or edge tests exclude every possible sample. Small triangles that cover even one sample remain drawable. Bounds spanning more than two by two samples, near-plane crossings and numerically large projections go to hardware without the edge test.

A 1/16-pixel guard in both axes conservatively accounts for rasterizer vertex snapping and floating-point projection differences. Vulkan requires at least [four fractional bits of subpixel precision](https://docs.vulkan.org/refpages/latest/refpages/source/Required_Limits.html); the regression also checks the actual Metal rasterizer used for these measurements. This is not an area threshold or mesh simplification. Each frame re-evaluates coverage for the current camera and viewport, including after movement or zoom. The full 6.25 mm SVO, DMC geometry, normals, PBR identities and shadow casters remain available. Neither the 50 mm fluid grid nor the mesh's 1 GiB retained allocation changes.

The overview's actual draw census changes as follows:

| Material group | Before | After |
| --- | ---: | ---: |
| Foliage | 3,939,763 | 213,322 |
| Pool/deck | 2,208,005 | 225,239 |
| Tree/shrub structure | 935,754 | 49,636 |
| House wall | 336,866 | 52,102 |
| Guests | 297,672 | 16,199 |
| Other | 173,468 | 11,975 |
| **Total drawn records** | **7,891,528** | **568,473** |

The first validated overview draw measured **1.64 ms** (95th percentile 1.70 ms), with bit-identical linear color and depth against the previous implementation. The retained live mesh still contains 15,823,181 records. This reduces the draw list rather than permanently deleting geometry. The SVO's spatial hierarchy and per-brick mesh ranges remain possible accelerators for candidate discovery; no coarse geometric LOD is enabled by this change.

The final paired runs use the same settings and validation-enabled Metal device as above, with four warmups and twelve timestamp samples per view:

| View / stage | Reference median | Optimized median | Optimized drawn records |
| --- | ---: | ---: | ---: |
| Overview mesh draw | 15.53 ms | 1.70 ms | 568,473 |
| Orbit mesh draw | 22.61 ms | 1.70 ms | 584,026 |
| Close mesh draw (camera distance 6.25 m) | 6.62 ms | 1.64 ms | 566,829 |
| Return to overview mesh draw | 15.27 ms | 1.70 ms | 568,473 |
| Overview culling compute | 2.69 ms | 2.69 ms | — |

The overview submits **92.8% fewer records** and its draw is **89.0% faster**. Culling cost did not increase measurably. The reference orbit has desktop-contention outliers; the return measurement confirms the overview result. All four paired captures have **zero changed linear-color, display-color or depth pixels**, and both runs report no GPU validation errors. [Paired comparison](../artifacts/pool-coverage/comparison.json), [reference timing](../artifacts/pool-coverage/reference.log), [optimized timing](../artifacts/pool-coverage/optimized.log), [overview](../artifacts/pool-coverage/optimized.png), [close view](../artifacts/pool-coverage/optimized.close.png).

Short three-second dry frame windows measured 119.38 → 99.38 ms median under variable desktop load. Lighting and imported-field rendering remain substantial; these measurements exclude fluid simulation and water compositing, and the roughly two-millisecond result refers specifically to the primary mesh draw at 1000×720. It is not a claim of two-millisecond total frames or all-camera/all-resolution performance.

The GPU coverage regression checks 37,548 triangles across 17×19, 64×64 and 1000×720 viewports against actual hardware fragment coverage. It includes both windings, perspective, tiny sample-covering triangles, triangles between samples, raster snapping boundaries, degenerate edges and near-plane/eye crossings. Every hardware-covered triangle must survive the compute test.

For paired checks, `FLUID_SVO_DRY_FRAME_MESH_SAMPLE_CULLING=0` keeps the reference draw and `=1` enables sample rejection. `FLUID_SVO_DRY_FRAME_MESH_CAMERA_CHECKS=1` saves orbit, close and return captures beside `FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT`. The isolated pass profiler also accepts compute labels; include `Voxel mesh culling` alongside `Voxel surface mesh rasterization` to account for the culling cost. Local evidence is in `artifacts/pool-coverage/`.

`npm run check:types` passed. `npm run test:unit` passed with 920 passing tests and 47 skipped. Explicit GPU checks for sample coverage, stream parity and raster AO passed six tests with no skips.

## Raster + AO attribution and shadow proxy culling

The live panel's roughly 31 ms Primary visibility total with only 5 ms attributed to primary rasterization was a reporting defect. Imported native fields execute in a separate pass, timed as `scene-primitive-visibility`. The mesh configuration marked that stage's row unavailable, suppressing its displayed duration while the band still included it. The panel now labels this row **Imported asset surfaces** and reports its cost. A regression test reproduces the hidden 25.6 ms interval. The former analytic-tier toggle does not control this pass and is no longer offered on this row.

Raster + AO also differs from the cone-lighting measurements above: its shadow cache retains native-asset proxy mesh triangles. Camera fragments discard these proxies because native fields provide their visible surfaces, but they previously still incurred camera culling and vertex work. The camera culler now rejects these records immediately. Shadow collection continues to use the complete arena, preserving its casters.

Validated Metal runs on the same M1 Max at 1000×720, 6.25 mm visual cells, 50 mm fluid cells and full PBR measured:

| Metric | Before | Final |
| --- | ---: | ---: |
| Camera draw records | 1,512,559 | 568,473 |
| Isolated mesh draw median | 3.41 ms | 1.84 ms |
| Isolated culling median | 5.57 ms | 5.11 ms |
| Isolated native imported surfaces median | 11.93 ms | 16.25 ms |
| Dry frame median | 25.21 ms | 27.64 ms |

Each isolated pass uses four warmups and twelve samples. The final run had substantial variability: culling ranged from 4.06 to 9.57 ms, and the unchanged native pass from 10.94 to 19.40 ms. The mesh draw reduction is clear, but these short runs do **not** demonstrate an overall frame-rate improvement. Imported field rendering remains the largest measured pass and needs further optimization; fixing its reporting does not remove that cost. These dry captures exclude water and fluid simulation and use a different viewport/refinement configuration from the user's screenshot.

The final capture is bit-identical to the baseline in linear color, display color and depth, with zero GPU validation errors. Visual resolution, source normals, PBR and shadows are unchanged. The retained 32,412,747 live mesh records and 2 GiB arena also remain unchanged; this improvement reduces camera work, not shadow storage. Local evidence: [baseline](../artifacts/pool-native/before.log), [final](../artifacts/pool-native/final.log), [comparison](../artifacts/pool-native/comparison.json), [frame](../artifacts/pool-native/final.png).

The benchmark now accepts `FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao`, selecting raster AO and disabling the incompatible voxel light cache. Use mesh primary, split shading and `FLUID_SVO_DRY_FRAME_CONE_FANOUT=0` to match production. Include `Voxel surface mesh rasterization,Voxel mesh culling,Imported native voxel field surfaces` in `FLUID_SVO_DRY_FRAME_PRIMARY_TIMING_LABELS` to expose all three costs. `npm run check:types` and `npm run test:unit` passed (921 passing, 48 skipped).

## Cached imported surfaces and moving-camera culling

Imported volumes now produce shared native-resolution triangle meshes and LOD index streams once per immutable volume. The full native mesh remains available; each frame selects a level targeting a quarter-pixel projected error. The simplifier error is an approximate geometric metric, not a certified maximum-distance bound. Surface normals and PBR materials still come from the original sampled fields. The 6.25 mm visual grid remains independent of the 50 mm fluid grid.

Geometry publication also builds a compact list of primary candidates, excluding shadow-only proxies. Camera culling and sample-coverage rejection run every frame, including during camera motion; no stationary-camera visibility cache is used. Supported devices compact visible records with subgroups. Background planes, imported meshes, voxel triangles and normal rigid rendering share one primary render pass. Diagnostic identity inspection retains the separate rigid bridge. Boundary normals are evaluated per fragment. Valid static shadow maps can be reused independently of camera motion.

The candidate list requires a second index stream: at the current 2 GiB mesh arena capacity, the combined candidate/visible allocation is 512 MiB instead of 256 MiB. Imported geometry and its LOD streams add shared per-asset buffers while eliminating repeated per-pixel volume raymarching.

The dense water fallback shares scalar samples within each workgroup. GPU parity tests compare its complete triangle and normal output against the original extractor, including nodal fields and walls. The live mixed-window extractor is unchanged.

The optional `FLUID_SVO_DRY_FRAME_PROFILE_WATER=1` benchmark currently uses a synthetic dense initial water field. Its forced-update modes measure dense fallback extraction, not the production solver's mixed surface windows. These results must not be used to infer a live water regression. With the latest M1 Max 1000×720 run, camera-motion rendering including this water field measured 7.27 ms median; forcing dense extraction raised it to 9.96 ms. These measurements do not establish the earlier requested sub-6 ms full-frame target. Per-pass intervals can overlap and must not be summed.

## Primary timing and host allocation audit (2026-10-19)

The previous **Primary rasterization** row combined three distinct stages. The
panel now gives **Mesh maintenance**, **Mesh culling**, and **Primary
rasterization** their own rows, each owning its existing GPU stage measurement.
Meshing settings and build status live under maintenance; the current draw count
lives under culling. Imported meshes and rigid bodies still share the raster
pass, so their rows describe that shared cost rather than inventing independent
GPU durations. Band totals count every stage once.

The audit found both real GPU work and avoidable host churn:

- The culler loaded the same immutable state words atomically for every
  candidate triangle. It now loads them once per 64-thread workgroup and shares
  them after a barrier. It still recomputes visibility for the current camera
  every frame and uses the same geometry, clipping and sample-coverage tests.
- Hundreds of imported draws were encoded from JavaScript every frame. A render
  bundle now retains those commands. Camera, poses, LOD and indirect draw counts
  are still read live from GPU buffers. Pipeline, binding, instance-list and
  rigid-mode changes rebuild the bundle; resize replaces attachment descriptors.
- Mesh status copied and mapped every frame, allocating copied buffers, promise
  chains and status objects. Pending builds still request prompt receipts; ready
  meshes poll status every 250 ms. Known maintenance starts extraction immediately,
  independently of a receipt. Unannounced GPU publication changes are discovered
  by the bounded poll. No primary encode waits for a map or queue completion.
- Moving rigid bounds rebuilt the entire scene parameter block and shadow-light
  metadata. They now update only their 16-byte block, keeping the parameter memo
  consistent. View/body upload storage, bounds scratch, tuning resolution,
  attachment descriptors, light-capability checks and command descriptors are
  reused on unchanged frames.

On the M1 Max at 1000×720, 6.25 mm visual cells, 50 mm fluid cells, smooth DMC
geometry and Raster + AO, the initial paired isolated GPU samples measured
culling **2.294 → 1.966 ms** (14% reduction). The shared draw remained **2.687 ms**.
Each arm used four warmups and sixteen samples. The complete overview capture
was bit-identical in all four half-float channels, including depth, with no GPU
validation errors. The roughly five milliseconds previously grouped into one
row therefore was not all raster drawing, nor a CPU timer masquerading as GPU
work.

A separate V8 sampling audit initially attributed approximately **81 KB/frame**
to application source before the changes and **4 KB/frame** after the first
optimization pass. These are sampled allocations, including native API wrappers
at application call sites, not an exact allocation census. The largest eliminated
site was primary command encoding. Follow-up cleanup also retains compute and
lighting descriptors and avoids rebuilding light-eligibility collections.

This does **not** establish zero allocations throughout the browser and solver.
WebGPU creates command/pass objects, completion uses promises, and sampled
telemetry still creates results. The Node Dawn binding's event polling also
allocates heavily. V8 allocation profiling itself increased GC pauses, so its
frame times are excluded from performance claims. It now runs after the clean
wall/timestamp measurements when `FLUID_SVO_DRY_FRAME_HOST_PROFILE` is set.
Likewise, a full scene's GPU completion timeline can vary with simulation,
desktop GPU contention and instrumentation; reducing host churn does not imply
that every fluctuation in a hardware timestamp was caused by GC.

Reproduction uses the existing dry-render benchmark and the settings above:

```sh
export WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js"
export FLUID_WEBGPU_BACKEND=metal
export FLUID_SVO_DRY_FRAME_SCENE=backyard-pool-party
export FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/pool-party-live.ts
export FLUID_SVO_DRY_FRAME_WIDTH=1000 FLUID_SVO_DRY_FRAME_HEIGHT=720
export FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=3
export FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary
export FLUID_SVO_DRY_FRAME_SURFACE_MESH=1
export FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes
export FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1
export FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao
export FLUID_SVO_DRY_FRAME_CONE_FANOUT=0 FLUID_SVO_DRY_FRAME_CONE_SCALE=1
export FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=8
export FLUID_SVO_DRY_FRAME_PRIMARY_TIMING=1
export FLUID_SVO_DRY_FRAME_PRIMARY_TIMING_LABELS='Voxel mesh culling,Voxel surface mesh rasterization'
export FLUID_SVO_DRY_FRAME_HOST_PROFILE=artifacts/pool-raster-host/audit
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

The optional host audit writes `.heapprofile`, `.cpuprofile`, and `.gc.json`
files after 512 warmed frames. These are dry rendering measurements, excluding
fluid simulation and water compositing. Local evidence lives in
`artifacts/pool-raster-host/`; the initial tiny-pass timing attempt rejected
unsampled/quantized timestamps and is not used as evidence.


Final verification retained the culling result (**1.966 ms median, 2.032 ms
95th percentile**) and draw result (**2.687 ms median, 2.818 ms 95th
percentile**). Overview, orbit, close and return captures are all bit-identical
in color and depth; see `artifacts/pool-raster-host/comparison.json`. Final host
sampling attributed **1,971 bytes/frame** to application call sites, about
**97.6% less** than the initial 80,935-byte sample. Remaining primary allocations
are predominantly WebGPU pass/command wrappers and occasional telemetry;
source-level reuse tests cover descriptors, bundles, upload buffers and tuning.

The final unprofiled loop recorded **0.270 ms median / 0.384 ms p95** JavaScript
encoding. Its submit/finish-to-completion median was 7.274 ms, but its maximum
was 609.864 ms, outside the encoding interval (maximum encoding: 2.897 ms).
That tail is not presented as solved or attributed conclusively to GC. Native
queue callbacks, initialization cleanup and competing desktop work remain
possible contributors. The separate allocation audit observes substantial Node
Dawn event-polling churn; this does not establish the browser's GC behavior.
The clean reference run overlapped CPU unit-test activity, so it is excluded
from end-to-end speedup claims; its images remain valid comparison inputs.

`npm run check:types` passes. `npm run test:unit` passes with 991 passing and
89 skipped tests. Focused regressions cover row ownership, no double counting,
mutable tuning inputs, partial rigid-bounds uploads, retained commands,
attachment changes, immediate edit extraction, and asynchronous status polling.

The explicit Metal GPU suite passes all seven coverage, mesh-stream and raster-AO
checks with no skips or validation errors.

### Interactive worker cadence (2026-10-19)

The dry-render audit above missed the interactive document adapter used by the
browser. `interactionScene` retained a derived document but dropped its revision
stamp. `FluidLabRenderer.publishRenderScene` therefore took its external-document
fallback on every draw: recursively sort, copy and stringify the entire scene.
Pool Party's canonical document is 3,213,094 characters. A standalone 120-call
check measured 58.2 ms median / 75.1 ms p95 for that fallback (Node); the browser's
full worker CPU reading was 24.47 ms. This was real work outside the GPU total.

Derived application documents now receive their own stable revision once, retain
identity across draws, and receive a new revision after an authored replacement.
Unversioned external documents retain their content-checking fallback. Both normal
and unsupported-interaction adapter paths preserve this contract.

Worker and window `performance.now()` clocks also had different origins. Frame
and physics traces were passed straight to the window, whose freshness filter
compared them against its own instrumentation start time. The attach handshake
now supplies `performance.timeOrigin`; the client translates each distinct cloned
trace once, including aliased presentation/stage traces. Durations are unchanged,
and message-delivery latency cannot change the clock offset.

A direct Chrome comparison used paused interactive Pool Party, refinement depth 1,
smooth surfaces, Raster + AO, and a 3456×1882 render surface:

| Reading | Before revision fix | After revision fix |
| --- | ---: | ---: |
| Completed-presentation FPS | 40.0 | approximately 60 (58.3–61.0 sampled) |
| CPU | 24.47 ms | 0.49–0.53 ms |
| GPU frame span | 11.37 ms | 13.10–13.53 ms |
| Primary rasterization | 4.98 ms | 5.61–5.65 ms |
| Queue wall | 24.73 ms | 31.67–31.71 ms |

These are live panel observations, not a controlled GPU speedup claim. Removing
the CPU bottleneck increases GPU submission throughput; queue wall includes two
in-flight presentations and callback delivery, so it is neither an additive GPU
stage nor the reciprocal of FPS. Completed-frame sampling can straddle batched
callbacks around the 60 Hz cadence. Browser evidence is saved locally under
`artifacts/pool-cadence/`, including the after screenshot. The remaining primary
GPU cost is investigated separately with draw-command ablations.

At the same resolution and refinement, isolated draw ablations measured the
shared pass at 4.784 ms median before and after the experiment. Omitting imported
asset draws reduced it to 2.228 ms; omitting voxel triangles reduced it to
3.604 ms; omitting the background reduced it to 4.588 ms. Planar/rigid omission
made no measurable difference in this view. These differences are marginal costs:
removing geometry changes occlusion, so they must not be summed as a partition.
`FLUID_SVO_DRY_FRAME_PRIMARY_ABLATION=1` records the production render bundle once,
then replays alternatives that omit only draws using the named pipeline. It does
not add production passes or change the shipped renderer.

### Primary shader follow-up

Two exact shader changes reduce the remaining primary cost:

- Imported volume material election uses constant array indices. The eight-corner
  weights, summation order, empty-material handling and first-wins tie rule stay
  unchanged. This avoids dynamically indexed local-array access in the nested
  voting loop. Register placement is a compiler decision, not a measured counter.
- Shadow-only imported voxel proxies now collapse to zero-area triangles in the
  vertex shader. Previously a material-dependent fragment `discard` rejected
  them, even though the normal culling path already filters these records. The
  vertex guard also covers unculled draws and removes that discard from the
  opaque fragment entry point. Better opaque-surface handling is the rationale;
  no hardware counter attribution is claimed.

Both were checked with interleaved ABBA reference/candidate draws on the same
Metal device, attachments and warmed scene resources: 16 warmups, 256 samples
(128 per arm), isolated production-pass GPU timestamps. The reference replaces
only the relevant shader pipeline; geometry, bindings and draw order stay fixed.

| Change | Reference median / p95 | Candidate median / p95 |
| --- | ---: | ---: |
| Constant-index material voting | 4.784 / 4.915 ms | 4.588 / 4.719 ms |
| Vertex rejection, with voting optimization in both arms | 4.522 / 4.784 ms | 4.325 / 4.522 ms |

Each change saves approximately 4% in its paired run. Different-run medians must
not be used to claim a combined speedup. An initial noisy vertex-rejection run
had 6.816/6.554 ms medians and worse candidate p95; the repeat above resolved the
median difference under steadier load. Oscillation and long-tail stalls are not
claimed eliminated. A uniform-material fast branch was tested and removed because
it did not improve performance.

Overview, orbit, close and return raw RGBA16F captures remain byte-identical to
before either shader change. A native GPU sampler regression checks 190 queries
covering mixed/repeated IDs, empty cells, ties, cell boundaries, distances and
normals. Logs and comparisons are under `artifacts/pool-cadence/`:
`sampler-paired.log`, `opaque-repeat.log`, and `opaque-comparison.json`.

For shader comparisons, set `FLUID_SVO_DRY_FRAME_PRIMARY_REFERENCE` to a JSON file
with `{ "pipeline": "exact pipeline label", "search": "candidate WGSL snippet",
"replacement": "reference WGSL snippet" }`. The benchmark builds both pipelines
and switches retained bundles in ABBA order. It rejects an unmatched shader or
undrawn reference pipeline. This instrumentation is confined to the benchmark.
Use the scene settings above with width 3456, height 1882 and refinement 1 for
this comparison. Local reference files and `run.sh` retain the exact experiments.

The panel now labels its total as **ms GPU** (or **ms queue** for the fallback),
rather than **ms/frame**. The tooltip explains why GPU span and presentation FPS
are not reciprocals. The earlier host audit still has a small measured allocation
remainder; neither these changes nor the cadence fix establish zero allocations
across browser/driver wrappers and telemetry.

Final follow-up validation: `npm run check:types` passes; `npm run test:unit`
passes with 996 passing, 90 skipped and zero failing tests. The explicit Metal
suite passes all 10 native sampler, imported-surface, mesh-stream, sample-coverage
and raster-AO tests with no skips. The mesh contract now checks proxy rejection
in both culling modes while retaining the separate LOD selection invariant.

## SVO construction and edit publication follow-up

The October 22 build/edit investigation retained these changes:

- Interaction-derived render scenes retain the accepted solid-world snapshot.
  Pool Party's robot adds bodies to a derived document but does not change voxel
  solids. Losing that cache caused each pen sample to replay all solid patches
  (59–152 ms in the traced run) and rescan all 6,458 pages (33–87 ms more).
  The derivative now shares the existing image in either direction, without
  triggering eager compilation on the input thread.
- Imported surface publication compares primitive membership and immutable source
  geometry before replacing instance buffers, indirect arguments and their bind
  group. Voxel, pose, material and lighting edits with unchanged membership keep
  these resources and the cached draw bundle. Selection still runs each frame;
  transforms and materials remain reads from the current scene arena. Reindexing,
  replacing an asset or removing the final instance invalidates the resources.
- Sampled solid-boundary publication compares only the 16 validity words per
  brick when its immutable source is unchanged. It no longer repacks or uploads
  the 729 distance samples alongside every mask edit. Changed source ordering,
  lost provenance and undo after provenance loss use a full-image diff.
- Surface mesh counting accumulates each extraction job's face count privately
  and adds it to the brick's scratch counter once. Emission and range allocation
  retain their existing contracts. In modes that omit sampled shadow casters,
  voxel meshing also rejects those proxy faces before exposure/contour work;
  neighbouring occupancy and raster shadow-map casters remain intact.
- Opaque-surface publication keeps the set of material IDs per immutable solid
  page. An edit rescans touched pages and rechecks the current material table;
  it no longer scans all 3.3 million material cells on each accepted sample.
- CPU fallback solid-reach queries prepare primitive descriptors once and skip
  field-gradient evaluation when only distance is required. Default GPU brick
  selection and DMC construction bypass this filter, so this is a fallback
  improvement rather than a measured default-build speedup.

### Isolated measurements

These measure the named work, not total build time or input-to-frame latency.
CPU samples ran alongside other checks and should be reproduced on an idle host
before using their absolute times as a budget.

| Work | Before | After |
| --- | ---: | ---: |
| Pool Party boundary, bytes per one-cell mask edit/undo | 154,044 | 4 |
| Pool Party opaque capability publication, median | 26.608 ms | 0.708 ms |
| Pool Party CPU fallback, 423,424 distance queries, median | 1,608.63 ms | 183.46 ms |
| GPU face-count overhead, 8 faces/job, median | 0.918 ms | 0.131 ms |
| GPU face-count overhead, 64 faces/job, median | 8.454 ms | 0.131 ms |

The boundary fixture is the actual 51-brick Pool Party diving board. An edit
outside that boundary uploads zero mask bytes. The distance fixture covers all
3,308 Pool Party primitives with 128 points per primitive and checks exact
reference parity before timing. GPU counting uses the production append helper
on 8,192 synthetic bricks with 48 jobs each, in paired ABBA order on Metal;
geometry traversal is excluded. At one face/job both paths measure 0.197 ms.
GPU timestamps here have 65.536-microsecond resolution.

Reproduce from the repository root:

```sh
node --import tsx tools/benchmark-svo-boundary-edits.ts
node --import tsx tools/benchmark-svo-solid-reach.ts
node --import tsx tools/benchmark-svo-opaque-publication.ts
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/benchmark-svo-mesh-count.ts
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx --test tests/svo-mesh-proxy-extraction.test.ts
```

The edit-latency probe now preloads the scene's real imported assets from disk,
reports their loading time separately, and records replayed render-bundle draws.
Its fake device supports bundles as well. Set `FLUID_EDIT_PROBE_PROGRESS=1` for
startup status logging. Raw component measurements and validation logs for this
investigation are under `artifacts/svo-build-edit/`.

### Live surface pen

Select **Pen** in the sculpt rail, choose **Draw** or **Carve**, and drag over a
surface. Diameter is adjustable from 1 to 64 voxels, including with brackets.
The pen publishes accepted samples during the drag, without a region selection;
release makes one undo entry, and Escape rolls the whole stroke back. It picks
and interpolates against the document at the start of the stroke so it does not
climb its own new geometry. A later stroke picks the newly edited surface.
Leaving the surface breaks the stroke rather than drawing across empty space.

Scenery carving is explicitly tagged in the saved patch history. Old voxel
clears retain their original meaning. The GPU subtracts rounded or box brush
records from scenery distances during occupancy, contour and DMC construction.
Cuts also affect analytic visibility. Affected imported instances use their
edited SVO mesh at the scene's voxel resolution; untouched instances keep their
native mesh. Independent mirrored material IDs preserve each imported PBR
palette without exposing the unedited proxies. The palette is prepared at scene
load, avoiding a large material upload on the first cut. The cut arena is fixed
capacity (4,096 stamps) and is diff-uploaded, including undo. Capacity refusal
happens before publication.

The live transaction holds one acceptance in flight and coalesces pending
proposals. CPU work still owns picking, the authored voxel-page edits used by
Draw/terrain Carve, validation, history and scene publication. Imported scenery
CSG and mesh reconstruction execute on the GPU. This is not yet a fully
GPU-resident brush editor.

The reproducible pen workloads send 16 surface samples through the actual pen
plugin and live transaction, render each accepted prefix, and validate undo:

```sh
FLUID_EDIT_PROBE_PROGRESS=1 WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/probe-edit-latency.ts --device=dawn \
  --scene=backyard-pool-party --edit=pen-draw,pen-carve --size=160x90 \
  --json=artifacts/svo-build-edit/pen-native.json
```

The report separates pointer geometry, acceptance, per-frame CPU stages and GPU
work. It checks one transaction finish and restoration to the starting patch
history, but does not capture images or assert first-visible-pixel latency.
Native regressions additionally execute imported-paver voxelization, verify cut
occupancy and normals, retain PBR materials and compare undo payloads byte for
byte. They also exercise ray hits on cut caps from inside a cavity.

The final native Pool Party run excluded 23.62 seconds of startup shader
specialization and produced these medians:

| Pen stage | Draw | Carve |
| --- | ---: | ---: |
| Scene publication | 3.91 ms | 3.48 ms |
| Acceptance | 6.35 ms | 5.96 ms |
| Frame host time | 9.52 ms | 8.64 ms |
| GPU completion | 99.96 ms | 121.43 ms |

Each stroke accepted 14 prefixes from 16 samples, finished once, and restored
the original 51 patches on undo. All builds were incremental, with zero GPU
validation errors and zero shader pipelines created during editing. Draw kept
all GPU objects; Carve replaced eight instance buffers and four bind groups as
affected props switched from native meshes to edited SVO meshes. No source
vertex or index geometry was reuploaded.

These are diagnostic measurements, not a controlled before/after comparison:
startup handling changed and CPU tests ran concurrently. GPU completion remains
too high to establish smooth realtime interaction, even though CPU publication
no longer replays the whole world. Raw results are in
`artifacts/svo-build-edit/pen-native-final.{json,log}`.

Validation for this follow-up: `npm run check:types` passes and
`npm run test:unit` passes with 1,074 passing, 106 skipped and zero failing tests.
Explicit native runs additionally pass all five sculpt-cut tests, two production
imported-renderer shader tests and three mesh extraction/counting tests without
skips. The live pen probe also completes on native Metal with no validation errors.

## Pen GPU latency investigation

The earlier 100–121 ms completion medians were measured under substantial
external contention. A subsequent quiet run at 160×90 measured 25.49 ms for
Draw and 30.65 ms for Carve, including host encoding and queue completion.
Actual dirty-payload GPU passes dominated at 9.67/11.89 ms; derived lighting was
3.01/3.08 ms. The first carve's payload pass still took 202.18 ms. These numbers
are from `artifacts/svo-build-edit/pen-gpu-baseline.json` and do not imply a
100-to-25 ms code speedup.

The probe now retains raw timestamp intervals and reports their enclosing span
and covered union. Summing per-pass timestamps overcounts overlapping GPU work.
It also records source-file hashes, the dirty-build plan, viewport dimensions,
and optional baseline/edited/undo image captures. Startup compilation and image
readback are excluded from stroke timings. Measurements at 800×450 provide a
more representative rendering workload than the original small profiling target.

The retained fixes target work that the traces identified:

- Native-instance cut classification uses a padded, quaternion-rotated box
  instead of a sphere-sized cube. A small ground cut no longer withdraws distant
  or vertically separated native geometry. Renderer publication, voxelization
  and dirty-region planning share this classification.
- GPU primitive evaluation reuses the cut-union distance for repeated queries
  at the same point, avoiding a fresh traversal of the stroke history for every
  candidate primitive.
- The DMC imported-volume sampler caches eight immutable scalar corners while
  consecutive probes remain in the same trilinear cell. It recomputes each
  point's interpolation weights and leaves material/normal sampling unchanged.
  Native scalar parity is bounded to four float32 ULPs; compiler reassociation
  means this scalar comparison is not bit-exact.
- DMC reconstruction omits source-centre normals that its fitted normal replaces.
  For authored page cells this avoids six neighbouring SDF reads and sometimes
  a 9³ Gaussian stencil. The terrain far shortcut, contour modes and later
  fitted-point/material normal overrides retain their original behavior.
  Native tests compare complete payload and fitted-vertex bytes against a
  shader using the prior normal route.
- Solid-only CPU publications reuse the terrain material model and bypass
  appearance-table work when scenery and lighting are unchanged.
- Appended cut samples retain the affected-instance classification and test
  only the new cuts against the remaining imported props. Undo, replacement
  and scenery changes recompute the classification.
- The first cut on an imported prop separates representation changes from
  geometry changes. Outside the brush's reconstruction halo, an initialized
  brick with one source primitive and no static terrain can preserve its
  geometry, fitted vertices, normals and baked palette while updating the
  sculpt material flag. Overlapping sources and fresh bricks take the full
  reconstruction path. Geometry invalidation always takes priority. Splitting
  a sampled terrain terminal also rebuilds its siblings, and preflight checks
  that work before adopting the edited document.
- Superseding a partially rebuilt revision retains its unfinished dirty bounds.
  They are stored without a sampling halo, so repeated interruptions do not
  expand the region indefinitely; capacity collapse remains conservative.

An experiment sizing derived-lighting dispatches indirectly on the GPU passed
native coverage checks but did not measurably reduce the roughly 3 ms stage.
It was removed. Moving the authoritative CPU patch/page model to GPU storage
would introduce synchronization for picking, physics and undo without addressing
the measured payload-reconstruction bottleneck. Canonical edits remain on the
CPU; voxel occupancy, brush CSG and surface reconstruction execute on the GPU.

A reverse-order sampler comparison used identical source hashes and the same
800×450 workload, with two 16-sample Draw strokes and two Carve strokes. The
reference arm replaces only the generated cached sampler with its uncached
equivalent. Active payload medians fell from 9.80 to 8.16 ms for Draw (16.7%)
and 11.01 to 9.37 ms for Carve (14.9%); derived lighting was 3.01 ms in both
arms. This supports retaining the cache. An earlier comparison had different
lighting timings despite unchanged lighting shaders, so it was inconclusive
because of possible scheduling, clock or overlapping-work differences.

The final first-Carve payload passes measured 17.69 and 17.89 ms, with total
frame completion of 34.58 and 33.32 ms. Earlier 800×450 investigative runs
measured 216.07 ms before tighter owner bounds, then 49.74 ms before geometry
reuse and sampler/normal improvements. Those staged runs are useful diagnostics,
not isolated ablations of every individual optimization.

Final accepted-prefix frames only (28 per mode; no no-op, settling or undo
frames) measured:

| 800×450 native probe | Draw median / p95 | Carve median / p95 |
| --- | ---: | ---: |
| Host encoding plus GPU completion | 21.47 / 23.24 ms | 23.00 / 33.32 ms |
| GPU timestamp span | 17.92 / 19.73 ms | 19.04 / 28.57 ms |
| Dirty payload pass | 8.16 / 9.11 ms | 9.37 / 17.69 ms |

These are serialized native completion timings, not browser pointer-to-photon
measurements. They do not establish sustained 60 fps. The final product and
reverse-reference runs have stable matching source hashes, zero GPU validation
errors, no full rebuild and passing transaction/undo checks. Raw reports are
`artifacts/svo-build-edit/pen-gpu-final-{product,reference}-800.json`;
active-frame comparison is `pen-gpu-final-active-comparison.json` in that folder.

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/probe-edit-latency.ts --device=dawn \
  --scene=backyard-pool-party --size=800x450 --fluid=off --max-frames=48 \
  --edit=pen-draw,pen-carve,pen-draw,pen-carve --sample-cache=product \
  --json=artifacts/svo-build-edit/pen-gpu-final-product-800.json
```

Run the same command with `--sample-cache=reference` and a different output
filename for the sampler counterfactual. Run timing probes without simultaneous
tests, shader compilation or other GPU benchmarks.

A larger-brush stress test used `lounger/one`, a close camera, 64 pointer
samples and an eight-voxel diameter. Grid-hit deduplication produced 30 accepted
stamps per stroke. Accepted-frame Draw median/p95 was 28.97/34.13 ms and Carve
35.19/40.53 ms. First Carve took 60.27 ms total, including 39.65 ms of payload
work over 504 planned bricks. Carve payload grew from a 12.94 ms median over
the first ten accepted samples to 19.10 ms over the last ten: cut-history cost
is still measurable and is not constant-time. The current pen therefore remains
below a sustained 60 fps target, especially for large brushes and longer cuts.

Both strokes finished as one transaction without rebuilding the scene or GPU
validation errors. Captures show a continuous rounded ridge and a trench through
the imported cushion; baseline-versus-undo comparison is exactly zero changed
pixels for both. Results are in `artifacts/svo-build-edit/pen-gpu-lounger-800.json`
and `pen-gpu-lounger-active-summary.json`; images are in `pen-gpu-lounger-images/`.
Acceptance budgets use accepted prefixes, excluding duplicate no-op samples.

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/probe-edit-latency.ts --device=dawn \
  --scene=backyard-pool-party --size=800x450 --fluid=off --max-frames=96 \
  --edit=pen-draw,pen-carve --pen-node=lounger/one --pen-size=8 \
  --pen-samples=64 --pen-camera=close \
  --pen-images=artifacts/svo-build-edit/pen-gpu-lounger-images \
  --json=artifacts/svo-build-edit/pen-gpu-lounger-800.json
```

Validation for the GPU-latency changes: `npm run check:types` passes;
`npm run test:unit` passes with 1,096 passing, 109 skipped and zero failing
tests. Explicit native runs cover terrain Draw/Carve, imported sculpting,
overlapping-source election, sampler parity, preserved DMC vertices and exact
undo/redo. The prior-normal and optimized-normal shader routes produce identical
tested payload and DMC bytes. `git diff --check` passes.

## Advancing simulation verification (2026-10-22)

The rendering measurements above do not exercise a running simulation. Even
`--fluid=scene` previously drew at time zero. The edit probe now accepts
`--simulation-steps=2`, which advances the authored `SceneInteractionHost`,
Box3D and the worker's NB-FLIP solver before, during and after Draw/Carve and
their undos. It checks completed GPU time against the interaction tick, finite
body poses, readiness, pipeline failures and GPU validation errors. Simulation
health and edit timing budgets are reported separately.

The first 160×90 coupled smoke completed 22 steps (0.366667 seconds), including four
accepted samples of each pen mode and both undos, with no runtime or GPU errors.
It also exposed roughly two-second accepted-prefix latency with physics enabled;
the earlier rendering-only timings must not be presented as coupled pen latency.
The report is `artifacts/svo-build-edit/pen-coupled-fluid-summary.json`.

CPU attribution found that Box3D regenerated every collision chunk before
checking its signature. The collision compiler now retains chunks whose immutable
pages, sampled source nodes and edit-validity masks are unchanged. Changes to
terrain, collider regions, lattice coordinates or coverage invalidate the relevant
cache. Overlap preflight skips remote edits only after comparing the source pages
under each body's current bounds; it does not reuse a stale body pose. Nearby
fills and undo still run penetration checks.

In the CPU-only four-prefix fixture, collision updates fell from 1,636–1,699 ms
to 7.8–9.3 ms. Preflight fell from 27–45 ms to 1.8–3.6 ms; commit validation is
0.24–0.51 ms. Initial collision construction remains about 1.5 seconds. These
are collision-stage timings, not full pen or render times. Cache tests compare
against uncached compilation through edits, removal, undo, sampled-field changes,
terrain changes, moving coverage and collider-region changes. Reproduce with
`node --import tsx tools/benchmark-box3d-solid-edits.ts`; evidence is in
`box3d-cpu-before.json` and `box3d-cpu-final.json` in the artifact directory.

The final combined 160×90 smoke, with collision reuse and the body-contact fix,
again completes all 22 synchronized steps through Draw, Carve and both undos,
with zero runtime/GPU errors and no full rebuild. Accepted-prefix medians are
186.4 ms for Draw and 143.8 ms for Carve, versus approximately two seconds in
the first coupled check. Both still fail the 5 ms acceptance budget; these changes
do not yet make the coupled pen realtime. The final report is
`artifacts/svo-build-edit/pen-coupled-fluid-final.json`.

The full splash checker now runs 960 coupled steps (16 simulated seconds) and
checks represented water volume using live mixed ownership, rather than treating
particle retirement as water loss. It samples water outside the ball and compares
wetness with the actual displaced-volume exchange. Its initial run retained at
least 99.66% of the water and had no runtime errors, but correctly failed because
buoyancy disappeared when the wet body contact region became coarse. Runtime
health alone is therefore insufficient to establish physical correctness.

The fix distinguishes rigid-body contact from static solid detail. Wet body
contact tiles and their neighbouring traction stencil remain at native `h`
resolution, even with surface-only static contact or coarse warm-region policies.
Dry body contact outside the predicted liquid band can still retire. The corrected
960-step run passes with zero coupling anomalies or GPU/pipeline errors. The ball
bottoms out at a centre height of 0.8731 m and returns to 1.7435 m near the 1.8 m
waterline; its final upward fluid impulse is 6.56 N·s. Minimum retained water is
99.218% and final retention is 99.261%. This verifies the authored splash and
settling interval, not indefinite volume conservation. Before/after reports are
`pool-party-splash-coupling-before.json` and `pool-party-splash.json`.

Native regression coverage also holds a sphere submerged after motion settles,
verifies real displaced-volume and pressure exchange, then moves it back into
air and verifies that fine contact ownership retires and displaced volume returns
to zero. An initially dry body remains coarse and seeds no particles. The policy
matrix and existing static-geometry tests pass on Dawn/Metal.

The existing native hydrostatic test also fails its unchanged 0.0001 m/s
threshold: its first frame reaches 0.000762446667 m/s. An isolated clean checkout
of commit `56ff3a510700f502ead2ff8cb351e4dd445fd529`, using the same Node 22.22.1
runtime and Dawn dependencies, fails with exactly the same value. This predates
the uncommitted pen changes. A 120-step diagnostic found no GPU errors, a
0.00491% volume increase and at most 85.45 micrometres of waterline error at the
end; switching the pressure-band schedule back to strict produced identical
results. The native rest assertion remains a known failure, with no tolerance
change. Evidence is in `pool-rest-head-native.{log,json}` and
`pool-hydrostatic.{log,json}` under `artifacts/svo-build-edit/`.

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/probe-edit-latency.ts --device=dawn \
  --scene=backyard-pool-party --size=160x90 --fluid=scene \
  --simulation-steps=2 --edit=pen-draw,pen-carve --pen-samples=4 \
  --max-frames=48 --json=artifacts/svo-build-edit/pen-coupled-fluid-smoke.json

WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx tools/check-pool-party-splash.ts
```

Repository checks after the simulation and collision fixes:
`npm run check:types` passes; `npm run test:unit` reports 1,116 passing,
115 skipped and zero failing tests. GPU-gated tests are run separately as
described above; the pre-existing native hydrostatic assertion is not included
in that unit-suite pass. `git diff --check` passes.

Before committing, the isolated task snapshot (excluding concurrent forest,
rendering and fluid-memory changes) also passed `npm run check:types` and
`npm run test:unit`: 1,076 passing, 111 skipped and zero failing tests.
