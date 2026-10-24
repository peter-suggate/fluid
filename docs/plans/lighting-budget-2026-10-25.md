# Lighting at 120 Hz on M1 Max

The target is a complete presented frame below 8.33 ms. At the screenshot's
65.2 FPS, the frame interval is about 15.34 ms; reaching 120 FPS requires 1.84×.
The proposed 2–3× improvement is an engineering target, not a measured result.
The initial investigation below recorded the original renderer. The implementation
and paired measurements are recorded in the final section.

## Findings in the current renderer

1. **The reservoir is not a sun-only scene.** `buildSvoSceneLights` publishes a
   directional light and `default/light/softbox`, a rectangular area emitter.
   Raster AO supports cached visibility for the first directional light and
   four eligible spotlights. Rectangles, points, sphere lights, additional
   directionals and excess spots use exact SVO visibility. The reservoir's
   softbox therefore traces up to two full-resolution rays per eligible pixel.
   Any unsupported light also prevents selection of the lean opaque raster
   shader for the whole deferred draw. This is a general capability cliff,
   not a problem specific to the wall geometry.
2. **Sun/AO are already reduced.** They run at half width and height. At
   3474 × 1882, that is 1,634,517 invocations versus 6,538,068 full pixels.
   A valid receiver evaluates 36 horizon depth taps, up to six coarse voxel AO
   samples, and sun PCSS: 12 blocker taps plus 24 comparison taps when blockers
   are found, potentially on two sun maps at their transition. Full-resolution
   reconstruction has four depth/normal-guided taps and evaluates visibility
   again where all four candidates fail. Moving the existing sun pass to half
   resolution is therefore not a new optimization.
3. **Spot filtering repeats work.** A mapped spot performs 12 blocker and up to
   12 comparison taps in the full-resolution light loop. Its cached solid-map
   visibility depends on receiver and light slot, not area sample index, so the
   two emitter samples can repeat that same map calculation. Analytic body
   corrections do depend on the sample ray and must remain distinct.
4. **The light budget changes the lighting.** Publication retains at most 32
   records. Deferred shading defaults to eight *sample* slots; the same uniform
   also bounds light count. Area lights consume two slots, even when a sample
   subsequently contributes nothing. Later emitters can disappear. Increasing
   the limit alone makes the full-resolution loop more expensive.
5. **Static shadow maps are already cached.** The settled scene skips their
   redraw. Re-caching those same maps is not the main opportunity; receiver
   filtering, unsupported-light tracing and AO still repeat each frame.

Code: `lib/svo/features/lighting-visibility/{svo-raster-ao,raster-spot-lights}.ts`,
`lib/svo/features/shading/{program,deferred-specialization}.ts`,
`lib/svo/contracts/svo-light-abi.ts`, and
`lib/svo/pipeline/webgpu-svo-dry-scene.ts`.

## Exploit empty and unaffected regions before doing expensive work

There are several distinct kinds of regions. Treating dark pixels as empty is
incorrect: they can still have environment light, reflections and contributions
from other lights.

| Region | Current behavior | General acceleration |
| --- | --- | --- |
| No opaque receiver | Deferred surface draw already rejects sky with reversed-Z depth before fragment shading; separate cheap sky draw shares the same render pass. AO dispatches everywhere and returns early for a miss. | Classify 8×8 reduced-resolution tiles (16×16 full pixels), compact occupied tiles, and dispatch AO/visibility indirectly. Measure whether saved miss loads/writes exceed queue overhead. |
| Receiver outside a light's influence | Each pixel visits the light, constructs a sample, then rejects zero radiance, range, cone or facing. | Build conservative per-tile/cluster light lists using depth bounds, emitter extent, range and cone. Keep directional lights separate. An empty list bypasses direct light work while retaining ambient/material terms. |
| Entire shadow footprint has no blocker | PCSS still performs blocker search; it already skips filtering when no blocker is found. | Add a cached min/max shadow-depth hierarchy. Prove visibility for the whole footprint with a few hierarchy reads; filter only ambiguous regions. |
| Entire footprint is occluded | PCSS or exact visibility still resolves it. | A conservative max-depth proof over the complete filter footprint can return zero visibility for that light. It must cover receiver-plane depth variation, biases, cascade boundaries and the largest possible PCSS footprint. |
| Ray cannot meet any caster | Exact SVO fallback still runs in many cases; some scene-exit rejection already exists. | Reject using conservative caster bounds/occupancy for the entire ray segment; traverse occupied hierarchy nodes only. Include backdrop, native surfaces and moving bodies in the proof. |
| No possible AO occluder in its support | Horizon taps visit a large flat region. | Use a depth hierarchy to reject empty AO support or skip empty intervals. A flat center tile alone is insufficient: a wall in the surrounding AO radius can still occlude it. Coarse offscreen AO needs a separate world-space proof/cache. |

Build the depth bounds, active mask and normal bounds once and share them between
AO, light assignment and visibility. Do not add three independent fullscreen
classification passes. Preserve neutral AO/shadow values for inactive tiles, or
carry an explicit validity mask: indirect dispatch must not expose stale texels.
Use GPU indirect counts without a CPU readback. A dense-screen path should remain
available when classification/compaction costs more than the work it eliminates.

Normal-cone rejection must be conservative for the complete finite emitter and
the shading normals; disable it for unsupported two-sided/transmissive materials.
Tile/light-list overflow must use a correct fallback, never silently drop lights.
Do not count sky savings twice: deferred surface shading already rejects it.
Sampling a few shadow-map corners is not a proof that the intervening footprint
is empty; the hierarchy must conservatively bound its complete support.

Instrument occupied tile count, lights per occupied cluster, zero-contribution
rejections, proven-lit/proven-shadowed fractions, actual ray count and traversal
steps, and full-rate reconstruction fallbacks. These distinguish a useful
classification pass from one that adds overhead to an already cheap early exit.

## Implementation order

**1. Separate visibility from full-resolution material shading for all lights.**
Produce per-light (and, where needed, per-emitter-sample) visibility at reduced
rate on occupied receiver tiles. Initially retain the current exact SVO query
for unsupported light types, but run it there instead of inside the full-rate
BRDF shader. This is the shortest route to preserving current lighting semantics
while cutting the expensive ray population by roughly four away from edges.
Reconstruct with depth, geometric normal and surface identity; use an explicit
edge/disocclusion queue where reduced samples cannot represent the receiver.
Retain full-resolution geometry, texture evaluation and specular shading.
Moving the rays out also permits a lean material shader independent of light mix.

Do not allocate a full-screen visibility layer for every possible light. Store
the compact tile's contributing lights, or process bounded batches and accumulate
their lighting. Reuse reconstruction weights across AO and light visibility.
Hoist identical mapped-spot filtering outside the area-sample loop.

**2. Share conservative classification and shadow proofs.** Introduce the active
tile/light lists and min/max hierarchy above. For supported shadows, choose
per-light atlas resolution from projected contribution and cache static casters
separately from dynamic changes. A light move should invalidate that light's
shadow allocation, not all maps. Extending map support to point/area emitters
requires an explicit quality design: one central shadow map does not reproduce
the visibility integral of a large rectangle.

**3. Make AO cheaper without removing it.** Prepare view-depth/ray data once,
use a depth pyramid for distant horizon taps, and vary sample density by local
depth/normal complexity. Keep detailed contact AO near the wall, seams and thin
geometry. Cache the broader static world AO separately where source revisions
prove validity. Temporal reuse is optional and requires depth/normal/motion and
geometry/light revision rejection; do not rely on long accumulation to hide
low-quality moving frames.

**4. Scale light selection correctly.** Replace the eight-sample truncation with
visible clustered lists backed by a storage buffer. All supported lights must
remain eligible. With many overlapping emitters, exact per-light visibility is
still linear work; use a bounded importance-sampled estimator with correct
weighting and spatial/temporal reuse if the product truly needs arbitrary light
counts. That is a later, higher-risk step with noise and disocclusion costs.
No renderer can promise fixed cost and identical deterministic quality for an
unbounded number of overlapping lights.

## Expected headroom and quality contract

For a fraction `f` of frame time made four times cheaper, total speedup is
`1 / ((1 - f) + f / 4)`. At 75% affected work this is 2.29×; at 85% it is 2.76×.
These are conditional arithmetic examples, not a profile claiming those fractions.
Empty-space elimination, reduced visibility and caching overlap; their claimed
speedups must not be multiplied independently.

Target dry rendering at roughly 3–4 ms to leave room for water, compositing and
submission inside 8.33 ms. Measure the complete live frame before concluding
that this allocation is sufficient. The attachment is not a capture of its exact
scene document, render settings, browser timings or simulation state.

Acceptance scenes must include the reservoir, a mostly empty view, a floor
filling the screen, foliage/thin silhouettes, strong normal maps, area-light
penumbrae, many non-overlapping and overlapping lights, moving objects/lights,
camera motion and live edits. Require unchanged light inclusion, stable exposure,
contact grounding, shadow softness and sharp material/specular detail. Inspect
full-resolution images and motion clips, especially disocclusions and thin edges.
Use GPU spans plus browser presented-frame median/p95; never sum overlapping
Metal render-pass timestamp windows as if they were independent costs.

## Diagnostic measurements

M1 Max was confirmed by Dawn's adapter information. The catalog reservoir was
rendered at 3474 × 1882 with raster primary/mesh, Raster + AO, GI off and normal
quality defaults. The dry renderer excludes fluid simulation, water rendering,
compositing and browser presentation. Other jobs were using the machine/GPU:
these first numbers are **contention-affected diagnostic observations**, not a
release gate or proof of any proposed speedup.

An initial clean-submission loop measured 8.21 ms median submit-to-fence plus
0.38 ms median JS encoding. Timestamped stationary frames measured 9.11 ms median;
orbit frames 6.95 ms. The combined half-resolution sun/AO compute median was
2.62 ms stationary and 1.97 ms during the orbit. These are different sequences,
not additive timings.

The new alternating-ablation run and a subsequent coverage-validation repeat
produced the following. Both had potential external contention:

| Diagnostic control | First median (ms) | Repeat median (ms) |
| --- | ---: | ---: |
| Baseline, shadows and AO on | 10.55 | 7.67 |
| AO off | 8.85 | 5.57 |
| Shadows off | 8.45 | 6.09 |
| Both off | 5.57 | 3.87 |
| Coarse AO off only | 11.21 | 7.47 |
| One emitter sample | 11.01 | 7.27 |
| One-light budget | 8.39 | 6.68 |
| Half output width and height | 4.98 | 3.01 |

The non-monotonic controls and severe outliers demonstrate why fine-grained
speedup claims would be unjustified from this run. Disabled-effect controls are
cost probes, not proposed solutions; half output resolution also changes geometry
and material resolution. The results support investigating pixel/visibility cost
but do not establish visual parity or a 2× optimized renderer.

The repeat's depth census found **51.52% empty pixels and 50.86% completely
empty 16×16 tiles**. Thus compact AO dispatch can eliminate about half its
workgroups, but those groups currently take the cheap miss branch. This does
not imply halving AO time, and deferred lighting already skips those pixels.

The repeat also provides a useful budget constraint: even deleting visibility
entirely leaves 3.87 ms of dry rendering. Treating the difference as an approximate
affected cost, making visibility four times cheaper would yield about 4.82 ms,
or 1.59× overall. Reaching 2–3× therefore also requires reducing the remaining
material shading/primary/presentation costs. Removing exact traversal from the
material shader and eliminating irrelevant light work are candidates, not yet
measured wins. The same requirement becomes stricter when water is included.

Raw repeat samples and coverage are saved beside this report in
`lighting-budget-2026-10-25.json`. GPU validation passed. Type checking passed;
the unit suite passed 1,183 tests, with 120 skipped and zero failures.

## Reproduce

The new `tools/benchmark-raster-lighting-budget.ts` is called from the existing
production harness. It runs four alternating forward/reverse blocks, with two
warmup and four timestamped frames per arm per block, and reports an untimed
depth-buffer census of empty pixels and completely empty 16×16 tiles. This census
describes opaque geometry coverage, not all future transparent/water work.

```sh
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" \
FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE=reservoir-seep \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary \
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 \
FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao \
FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1 \
FLUID_SVO_DRY_FRAME_GI=0 \
FLUID_SVO_DRY_FRAME_WIDTH=3474 FLUID_SVO_DRY_FRAME_HEIGHT=1882 \
FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=1 \
FLUID_SVO_DRY_FRAME_LIGHTING_BUDGET=1 \
FLUID_SVO_DRY_FRAME_OUT=/tmp/fluid-lighting-study/report.json \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

Run without competing GPU work. Current camera, material and mesh details are
part of the measured fixture; substitute other catalog scene IDs to test generality.

## Primary references

- [Clustered Deferred and Forward Shading](https://research.chalmers.se/publication/161725):
  spatial/depth/normal grouping for reducing per-receiver light candidates.
- [Percentage-Closer Soft Shadows](https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf):
  blocker search and contact-hardening filtering; a cached map does not make its
  per-receiver sampling free.
- [FidelityFX CACAO](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/combined-adaptive-compute-ambient-occlusion/):
  reference for depth preparation, adaptive sampling and AO reconstruction;
  porting its implementation is not itself an M1 performance guarantee.
- [ReSTIR direct lighting](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct):
  an option for bounded many-light sampling, not evidence of a drop-in WebGPU/M1
  speedup or unchanged temporal quality.


## Implemented: current-frame light visibility (2026-10-25)

Opaque Raster AO now computes visibility for every admitted light type before
material shading. It retains the existing light/sample budget and two area-light
samples; it does not introduce a new light limit. Transmissive solid publications
and GI retain the reference path, since a primary-pixel cache cannot represent
secondary refracted or indirect receivers.

The reduced producer evaluates each emitter sample at half width and height,
rejecting empty pixels, zero-energy/range/beam failures and receiver backfaces
before shadow queries. AO also rejects taps outside its world radius by radial
distance, and taps behind the receiver plane before normalization. The existing
AO taps, PCSS settings, light energy, material evaluation and output resolution
are retained.

A compute resolve matches depth, normal and surface identity, writes a compact
reconstruction guide, and evaluates unsupported edge receivers exactly. This
includes thin features and clamped odd viewport edges. The fragment shader reads
that guide and RGB10 visibility; its specialized variant contains no visibility
query call path. Dynamic-body and fluid shadow factors remain in the visibility
producer. There is no temporal history to trail behind edits or camera motion.

An optional GPU tile queue tests all pixels, including thin features missed by
the reduced lattice. Measurements showed that its separate classification pass
cost more than coherent empty-pixel exits in these views. Consequently direct
dispatch is the default; `FLUID_VISIBILITY_COMPACT_TILES=1` retains the comparison.

### Paired measurements

Apple M1 Max, 32 GiB, Metal/Dawn, **3474 × 1882**. Each pair uses the same renderer,
scene, camera, AO/shadow controls and sample budgets; order alternates. Four
paired warmups precede 24 catalog or 16 close-camera pairs. Both final runs held
the shared exclusive GPU-runner lock. Values below are median **dry GPU frame
spans**, not browser FPS. Water rendering, physics and presentation are excluded.

| View / lights | Reference | Candidate | Median paired speedup |
| --- | ---: | ---: | ---: |
| Catalog, authored sun + rectangle | 7.67 ms | 7.27 ms | 1.05× |
| Catalog orbit, authored lights | 7.60 ms | 7.27 ms | 1.05× |
| Close, authored lights | 11.27 ms | 8.98 ms | 1.25× |
| Close orbit, authored lights | 11.21 ms | 8.85 ms | 1.27× |
| Catalog, five light types / eight samples | 176.95 ms | 46.27 ms | 3.82× |
| Catalog orbit, five light types | 177.34 ms | 46.40 ms | 3.82× |
| Close, five light types | 181.34 ms | 46.07 ms | 3.93× |
| Close orbit, five light types | 182.65 ms | 46.33 ms | 3.97× |

The five-light fixture adds a point, spot and spherical emitter to the authored
sun and rectangle. These are deliberately demanding exact-visibility cases.
The large relative gain does **not** make them 120 Hz: reduced exact traversal
still costs about 35–37 ms. The authored close view also remains above 8.33 ms.
A general 2–3× gain for the simple screenshot scene is not established by this
change. Further work needs cheaper finite-emitter visibility and the remaining
primary/material costs, not just more empty-pixel tests.

### Fidelity and costs

All eight comparisons have identical geometry depth. Receiver-only display RMSE
is 0.050–0.057 on a 0–255 scale. At worst 0.0324% of receiver pixels differ by more
than two levels; at most two pixels exceed eight levels in the final catalog
captures. Close-view maximum differences are below 6.1 levels. The harness writes
reference/candidate PNGs and linear RGBA16F captures, and rejects depth changes,
RMSE ≥ 1, or differences above eight levels on ≥ 1% of receivers.

The atlas uses four bytes per pixel per admitted emitter sample, with reduced
representatives and exact-edge samples sharing one full-size address space.
The reconstruction guide adds eight bytes per pixel. Additional allocation at
this Retina size is approximately **125 MiB for the authored three samples** or
**249 MiB for eight samples**, plus a small optional tile queue. This is a real
memory tradeoff; sparse edge storage is a useful follow-up.

Machine-readable samples, percentiles and source hashes are in
`raster-visibility-2026-10-25.json`. Reproduce with:

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
FLUID_WEBGPU_BACKEND=metal \
FLUID_SVO_DRY_FRAME_SCENE=reservoir-seep \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary \
FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 \
FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao \
FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1 \
FLUID_SVO_DRY_FRAME_GI=0 \
FLUID_SVO_DRY_FRAME_WIDTH=3474 FLUID_SVO_DRY_FRAME_HEIGHT=1882 \
FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=1 \
FLUID_SVO_DRY_FRAME_VISIBILITY_COMPARE=1 \
FLUID_VISIBILITY_CYCLES=24 \
FLUID_SVO_DRY_FRAME_OUT=/tmp/fluid-visibility/report.json \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

Set `FLUID_VISIBILITY_CAMERA_SCALE=.25` for the close stress view. The exact user
camera is not recoverable from the screenshot; these use the catalog camera and
a documented distance change.

Validation: `npm run check:types` passes. `npm run test:unit` passes with 1,195
passing, 125 skipped and no failures. Six focused tests also pass with Metal
enabled, including actual GPU queue reset/thin-receiver and RGB10 error checks.
A separate 833 × 461 compact-queue run passes all image/depth checks with
receiver RMSE below 0.175/255; its concurrent timings are not used above. All
production paired runs completed without GPU validation errors.
