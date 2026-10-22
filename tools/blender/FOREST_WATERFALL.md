# Stillwater Grove — live forest waterfall

`forest-waterfall` is a standard scene in the application's Explore / Garden
catalog. Open it through the scene library or `/scene?scene=forest-waterfall`.
It uses the production
`uniform-narrow-band-flip` method, an initialized plunge pool, and a continuous
clifftop source. The falling water and pool surface are simulated by the app.
No Blender waterfall, water surface, foam sheet, or spray mesh enters the live
scene.

The fluid domain is 32 × 26 × 32 metres at a 12.5 cm fluid cell pitch
(256 × 208 × 256 authored cells). Its page pool is 5,140 MiB: 448 padded domain pages, 64 pages in an
open-top splash layer, and two reserved blocks. The initial pool and pinned
source occupy 200 pages. The shared terrain
heightfield spans 80 × 96 metres on a separate 25 cm grid.
The scene render profile uses refinement depth zero, preserving the previous
12.5 cm static-environment grid while fluid resolution increases. Native
Blender assets retain their own finer sampling. Explicit user render settings
override the authored profile.
The source runs for 60 seconds. The surrounding basin has room for that inflow;
there is no invented drain or hidden recirculation. Source and timestep values
are defined in `lib/core/forest-waterfall-scene.ts`.

## Static environment

`assets/forest-waterfall/forest-waterfall-runtime.blend` is the separate static
asset library. It contains shared botanical and rock prototypes, with no
modeled water. GLB and signed FVX files live in `public/assets/forest-waterfall`.
The manifest records their source names, native sampling pitches, discrete
size variants, and finite botanical envelopes. Imported instances are stored
in `lib/core/imported-assets/forest-waterfall-instances.json`.

The sampled runtime assets retain the authored branch and leaf geometry at
25–80 mm native pitch, with closed leaf envelopes for signed volume sampling.
Sub-cell leaf tips and the source's continuous size variation are finite LOD
approximations. The Cycles reference has more complete light transport and
finer geometry; it is not a screenshot of the app.

The five waterfall ledges have collision fields sampled from those same
exported mesh volumes. The terrain provides the basin and cliff collision.
Trees and small foliage are static scenery rather than fluid obstacles.
`scenery.solidBoundarySurface` assigns visible ledge surfaces to their imported
assets while retaining the sampled collision fields for the solver. Later
voxel edits remain visible.

## Rebuild

First create the full Blender art source:

```sh
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" \
  --background --python-exit-code 1 \
  --python tools/blender/forest_waterfall.py -- --no-render
```

Then export its static runtime library and bake the signed assets:

```sh
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" \
  --background --python-exit-code 1 \
  --python tools/blender/export-forest-waterfall.py
"/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender" \
  --background --python-exit-code 1 \
  --python tools/blender/bake_mesh_volume.py -- --collection forest-waterfall
npm run generate:assets
node --import tsx tools/blender/bake-forest-colliders.ts
```

Coordinates convert from Blender `(x, y, z)` to application `(x, z + 3, -y)`.
The runtime forest includes broadleaf, pale multi-stem alder, and evergreen
cedar forms with flared roots and curved trunks. Size variants share native
geometry and retain baked bark and leaf colors.

Assets are exported about their local origins; instance orientation rotates the
bake-bound centre offset as well as the geometry. Ground plants follow the
live terrain so their roots do not depend on the reference's noise function.

## Validate the application scene

```sh
npm run check:types
npm run test:unit
node --import tsx tools/check-forest-waterfall-live.ts --steps=360
node --import tsx tools/capture-forest-waterfall.ts --time=3 --width=1200
```

The live GPU check records the registered scene's physics digest, actual solver
particle positions and velocities, progress down the waterfall, pool presence,
capacity counters, completion time, and GPU errors. Its report is
`artifacts/forest-waterfall/live-physics.json`.
`tools/preview/forest-waterfall.ts` loads the same registered scene and assets
for production-renderer inspection. The capture tool writes the actual app
swap-chain PNG plus a JSON report of solver, renderer, and GPU status;
`--time=0` captures the initialized pool before the waterfall has developed.

## Art reference

`assets/forest-waterfall/stillwater-grove.blend` preserves the complete Cycles
look-development scene. Its water is explicitly art-directed still geometry,
useful for composition only. It is not the live fluid result. Render it with
`--width 2400 --samples 128` instead of `--no-render` above. Its output is
`artifacts/forest-waterfall/stillwater-grove.png`.

## Runtime draw grouping and measurements

Repeated native assets are prepared as shared asset/LOD draw groups. Each frame,
GPU screen-error selection compacts visible primitive IDs into bounded group
segments. Poses and materials still come from the scene publication; no optional
`indirect-first-instance` feature is needed. `importedSurfaceBatching: false`
retains the ungrouped reference path for profiling.

The first A/B used the original 25 cm scene and original tree assets, at 960 ×
640, with raster AO, shadows, and 20 GPU-timestamp samples. Both paths measured
14.55 ms median and their captured RGBA16F pixels were byte-identical. This
establishes parity and reduced draw-command count, **not** a measured GPU speedup
or the performance of the final 12.5 cm scene. Reports are
`artifacts/forest-waterfall/perf-before.json` and `perf-after.json`.

Imported assets now cast cached sun and spot shadows from their shared finest
native meshes. Their duplicate voxel shadow surfaces are omitted; edited assets
retain their voxel-derived surfaces. The renderer memory measurements below
cover this change and demand-driven radiance storage.

```sh
node --import tsx tools/check-imported-surface-batching.ts
node --import tsx tools/blender/check-forest-assets.ts
```

## Capture before memory optimizations

`artifacts/forest-waterfall/live-established-6s.png` is a historical production
app capture made before the solver and renderer memory optimizations below. At
six simulated seconds it records: exactly 720 fixed steps, no GPU errors, complete surface
mesh, 755,868 live-water vertices. Its JSON records the authored renderer profile,
solver state and timing. The final 24 assets / 2,390 instances issue 168 imported
draw calls. Twelve paused 1600 × 1067 frames have a median 23.02 ms host wall time
including the GPU fence; this is not interactive FPS or GPU-only timing.

The separate three-second solver check measures 37–75 ms per sampled step
(median 58.6 ms) and 10.14 GB allocated. Total water is 1.84% below the nominal
initial volume plus integrated nozzle inflow at that time. This comparison is
not a GPU budget-ledger readback, and the full 60-second run is not certified.
`artifacts/forest-waterfall/validation-summary.json` links that historical evidence.
The allocation and timing figures in this section are not the optimized result.

## Solver memory and stage profiling

The following paired runs use the registered forest scene at 12.5 cm, its
unchanged 514-block page pool, 262,144 fine-tile capacity, and 16,842,752-particle
capacity. Foliage, source, timestep, pressure settings, and fluid quality are
unchanged. Each run advances 360 fixed steps to three seconds and measures
frames 13–360. Reports, resource labels, source hashes, and per-frame counters
are in `artifacts/forest-waterfall/performance/forest-stage-baseline.json` and
`forest-stage-optimized.json`; `solver-optimization-comparison.json` contains
the calculated differences. The baseline was captured before these changes.

| Requested GPU allocation | Before (bytes) | After (bytes) | Saved (bytes) |
| --- | ---: | ---: | ---: |
| Deposit / surface metrics / regional masks | 1,045,778,320 | 642,076,560 | 403,701,760 |
| Remap worklist | 495,225,588 | 2,979,288 | 492,246,300 |
| Separate momentum trace list | 335,544,320 | 0 | 335,544,320 |
| Shared stage scratch | 1,268,222,976 | 1,054,596,096 | 213,626,880 |
| **Total live solver resources** | **10,116,492,104** | **8,671,372,844** | **1,445,119,260** |

The 14.3% reduction comes from reusing storage after its previous contents are
no longer needed, retaining remap partial sums in workgroup memory, and bounding
transient surface queues. An overflow trace takes the full deferred transport
path; an overflow cubic evaluates inline. No fluid sample is discarded. The
free/solid cubic partitions fit because each admitted trace emits at most one
cubic. Surface queue reduction alone saves more than a gigabyte in that stage,
but extension still requires 1.055 GB, limiting the shared allocation saving.
Peak requested solver allocation falls from 10,132,466,388 to 8,687,347,128 bytes.
These are buffer/texture sizes requested from WebGPU, not process RSS or physical
GPU residency; they exclude renderer resources.

| Measurement | Before | After |
| --- | ---: | ---: |
| Surface-advection GPU pass mean | 4.324 ms | 2.996 ms |
| Deferred surface GPU pass mean | 5.181 ms | 5.189 ms |
| Whole solver-step wall mean | 51.88 ms | 50.21 ms |
| Whole solver-step wall median | 50.16 ms | 48.97 ms |
| Whole solver-step wall p95 | 74.43 ms | 71.20 ms |
| Final-third wall median | 64.30 ms | 62.23 ms |

The surface pass improves 30.7% by rejecting noncanonical coarse-wall vertices
before expensive ownership and far-air checks. An exhaustive GPU regression
checks the production listing over all 35,937 vertices of a page, including
mixed h/4h faces, shared edges/corners, and cold/far exits. Whole-step mean falls
3.2% in this pair. GPU timestamps preserve production pass boundaries; wall
times include host encoding, submission, and completion, with diagnostic
readback outside the measured interval. These are instrumented simulation
measurements, **not app FPS**, and do not include rendering. Both runs finish
without GPU errors or particle reseed clipping. Parallel scheduling and changed
remap reduction order produce different trajectories/adaptive work, so this is
not a bitwise state-parity claim.

Reproduce the current solver profile and focused wall check with:

```sh
node --import tsx tools/benchmark-nbflip-ui.ts forest-stage-recheck \
  --scene=forest-waterfall --steps=360 --from=13 --profile --memory \
  --out=artifacts/forest-waterfall/performance
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx --test tests/uniform-surface-wall-list.test.ts
```

For hotspot diagnosis, add `--dispatch-probe` and
`--dispatch-labels="Uniform mixed surface advect|Uniform mixed surface advect deferred"`.
This splits selected dispatches into timestamped passes and perturbs scheduling;
its timings must not be substituted for the preserved-pass comparison above.

Allocated capacity remains much larger than active state: the optimized run has
440,039 live particles, 7,084 fine tiles, and 295 resident pages at three seconds.
Its two particle buffers still reserve 1.617 GB for worst-case capacity, and
pressure, extension, and field backing remain major allocations. The changes
reduce scratch/storage duplication; they do not make the whole solver allocate
only for live particles, and they do not certify the complete 60-second take.


## Renderer memory and shadow performance

These isolated production-renderer runs use the registered 12.5 cm static
lattice, all 2,390 imported instances / 24 assets, a fixed 960 × 640 camera,
raster AO, shadows, and GI off on an Apple M1 Max / Metal. Each records four
warmup frames and twelve GPU-timestamp frame samples. They contain no live
fluid solver; their allocation totals must not be presented as whole-app memory
or added to process RSS. The ledger counts explicit WebGPU buffer sizes and
logical texture payloads, excluding driver alignment, caches, and executable
code. The peak includes transient resources during construction.

| Renderer arm | Live bytes | Peak bytes | Cached-frame GPU median | Whole-shadow-refresh GPU median |
| --- | ---: | ---: | ---: | ---: |
| Original voxel shadow proxies, full radiance allocation | 5,584,197,936 | 6,255,286,576 | 13.893632 ms | 49.053696 ms |
| Native shadows, full radiance allocation | 4,838,628,392 | 4,922,514,472 | 14.221312 ms | 52.461568 ms |
| Native shadows, full radiance allocation, second pair | 4,838,628,392 | 4,922,514,472 | 13.959168 ms | 51.838976 ms |
| Native shadows, opacity-only allocation | 2,869,476,408 | 2,953,362,488 | 14.024704 ms | 52.035584 ms |

The two changes save **2,714,721,528 live bytes** and **3,301,924,088 peak
bytes** relative to the original renderer. Reports and PNGs are
`artifacts/forest-waterfall/shadows-reference.*`, `shadows-native.*`,
`shadows-radiance-full.*`, and `shadows-opacity-only.*`.
`renderer-memory-comparison.json` collects the allocations, timings, and frame
hashes. Every run completed without GPU validation errors.

Native shadows reuse 284,426,628 bytes of source vertex/index buffers already
needed for primary drawing. All instances together contain 113,147,042 finest
mesh triangles. Shadow groups use that full geometry, independently of the
camera's visibility and LOD selection, with conservative sun/spot bounds and
partial-redraw bounds. A GPU cache gate sets indirect instance counts to zero
when no map needs updating. Scene edits invalidate the affected maps; sculpted
instances yield to their edited voxel surfaces. The additional immutable
instance records occupy 19,120 bytes.

Removing duplicate imported shadow proxies reduces required voxel quads from
15,222,690 to 2,287,242 while primary drawn quads remain exactly 290,496. The
voxel arena falls from 730,689,120 to 134,217,728 bytes and its candidate buffer
from 182,672,280 to 33,554,432 bytes. Including the small new draw state, this
saves 745,569,544 live bytes. Visual inspection shows finer native leaf and fern
shadows with retained caster coverage; this representation change is not a
pixel-parity claim. Six alternating cached/forced-invalidation samples measured
a complete shadow refresh about 3.41 ms slower in the first pair. This is a
worst-case full-map refresh, not the cost of every live frame or a measurement
of a typical local edit. Cached frames remain around 14 ms; these measurements
do not establish a rendering speedup or interactive app FPS.

Raster AO with GI disabled does not consume exitant-radiance textures. Its
normal coarse canopy AO **does** consume opacity, so the complete 244,047,872-byte
opacity atlas and its maintenance remain unchanged. Only the four unused
488,095,744-byte radiance lobes and excess radiance scratch are replaced by
inert, nonpublished bindings. This saves another 1,969,151,984 bytes. Enabling GI
or switching to cone lighting rebuilds the disposable scenery sidecar with full
radiance storage while retaining the running fluid solver. Cone lighting keeps
its paired hierarchy even with GI off, preserving its existing visibility path.
Dispatching a radiance build against inactive storage is rejected.

The radiance-full and opacity-only runs produce byte-identical complete
960 × 640 RGBA16F frames, including canopy AO and shadows. Both raw frames have
SHA-256 `d13c2b4e9c75700f2446c8a0b961b49c319ee0c27753df7d01b2ec057c527377`.
This parity applies to the isolated fixed-camera dry renderer and the radiance
allocation change; it does not claim bitwise fluid trajectories or parity
between voxel-proxy and native shadow representations. Fluid resolution,
foliage sampling, asset geometry, and shadow-map resolution are unchanged.

Reproduce all four renderer arms from the repository root. The explicit
radiance override selects the former allocation policy for comparison; the
opacity-only arm uses the application's current raster-AO/GI-off demand rule.
The commands write separate `-recheck` reports and images:

```sh
export WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal
export FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/forest-waterfall.ts
export FLUID_SVO_DRY_FRAME_WIDTH=960 FLUID_SVO_DRY_FRAME_HEIGHT=640
export FLUID_SVO_DRY_FRAME_WARMUPS=4 FLUID_SVO_DRY_FRAME_CYCLES=12
export FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_SURFACE_MESH=1
export FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes FLUID_SVO_DRY_FRAME_CONE_SCALE=1
export FLUID_SVO_DRY_FRAME_GI=0 FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao
export FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1 FLUID_SVO_DRY_FRAME_TIMING=gpu
export FLUID_SVO_DRY_FRAME_PHASE_TRACE=1 FLUID_SVO_DRY_FRAME_NATIVE_SHADOW_TIMING=1
forest_renderer_profile() (
  export FLUID_SVO_DRY_FRAME_NATIVE_SHADOWS="$2"
  export FLUID_SVO_DRY_FRAME_DERIVED_RADIANCE="$3"
  export FLUID_SVO_DRY_FRAME_OUT="artifacts/forest-waterfall/shadows-$1-recheck.json"
  export FLUID_SVO_DRY_FRAME_DUMP="artifacts/forest-waterfall/shadows-$1-recheck.png"
  export FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT="artifacts/forest-waterfall/shadows-$1-recheck.raw"
  node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
)
forest_renderer_profile reference 0 1
forest_renderer_profile native 1 1
forest_renderer_profile radiance-full 1 1
forest_renderer_profile opacity-only 1 0
```

Focused regressions cover full-instance shadow grouping, sun/spot bounds,
GPU indirect cache gating, light invalidation, sculpt/undo resource lifetime,
inactive radiance publication, dispatch guards, and sidecar toggles retaining
the solver:

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
  node --import tsx --test tests/imported-surface-shadow.test.ts
node --import tsx --test tests/svo-imported-surface-publication.test.ts \
  tests/svo-radiance-storage-demand.test.ts tests/svo-derived-lighting-demand.test.ts
```


## Whole-app capture after memory optimizations

`artifacts/forest-waterfall/performance/live-optimized-6s.png` is the final
1600 × 1067 production capture. Its adjacent JSON records exactly 720 fixed
steps to six seconds, no GPU errors, a complete terrain/foliage mesh, and
784,332 live-water vertices. All 24 assets and 2,390 instances remain present,
with 168 primary imported draw groups and 24 native shadow groups. Visual
inspection confirms the continuous waterfall, splashes, pond, and foliage.

The all-resource ledger records 11,908,014,937 bytes live and 12,025,455,185
bytes peak, including the fluid solver, scenery, water rendering, and capture
swap chain. The peak also includes diagnostic volume-readback allocations.
There is no matched whole-app baseline ledger; the solver and dry-renderer
A/B measurements above are separate workloads. Requested resource payloads
are not physical residency or process RSS.

Direct production phi and solid-capacity readback measures 529.260 m³ initially
and 548.682 m³ at six seconds. Of the final water, 496.690 m³ remains below
the initialized pond level plus 0.5 m. This checks that the settled Eulerian
pond remains present; it is not a conservation certification.

Twelve paused frames have a median 22.978 ms fenced host wall time. During
the 720 simulated frames, the median is 134.983 ms and p95 is 171.858 ms,
including solver work, renderer maintenance, host encoding, queue completion,
and callback yields. These serialized capture costs are not interactive FPS,
and the six-second workload is not directly comparable with the isolated
three-second solver profile. The remaining live-frame cost is substantial.

Final checks: `npm run check:types` passed; `npm run test:unit` passed
1,119 tests, with 115 skipped and zero failures. Focused native GPU checks
separately passed for deposit scratch clearing, exhaustive mixed-wall
ownership, and native shadow coverage/cache behavior.
`artifacts/forest-waterfall/performance/optimization-summary.json` links the
measurements and their scope. Reproduce the full capture with:

```sh
node --import tsx tools/capture-forest-waterfall.ts \
  --time=6 --width=1600 --height=1067 \
  --out=artifacts/forest-waterfall/performance/live-optimized-6s-recheck.png
```
