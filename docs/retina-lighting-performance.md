# Adaptive Retina lighting

Measured on Apple M1 Max / Dawn Metal, 3448 × 1892, `backyard-pool-party`,
native imported assets, smooth dual marching cubes, 25 mm environment detail.
These are production **dry-render GPU spans**, excluding water, physics and
browser presentation. They should not be equated with the screenshot's total
frame time.

## What was expensive

Lighting scales with physical pixels and visible coverage. Doubling device pixel
ratio produces four times as many output pixels. The existing reduced visibility
grid helps, but complex geometry and identity boundaries still require exact
full-resolution receivers. In the close view, 4,677,812 pixels contain surfaces;
716,137 require exact fallback (15.3%). Each fallback previously combined horizon
AO, coarse AO, sun visibility and any unsupported light-ray traversal in one
shader. It also appended independently to a single global queue counter.

The close baseline's exact fallback pass was about 4.46 ms, on top of roughly
2.75 ms for reduced sun/AO and 1.38 ms for resolve. Per-pass timestamps describe
individual windows; whole-frame paired measurements are the performance gate.

## Changes

The **Lighting → Minimum AO taps** slider selects 0, 4, 8 … 36 taps;
24 remains the default used for the measurements below. It updates live and is
saved in links as `svoFarAO`. Zero removes screen-space contact AO at the
minimum-importance receivers; coarse voxel AO remains separately controlled.
Lower budgets retain the innermost taps first, alternating opposite directions.

1. **Original projected-size AO budget (superseded below).** The world-space AO radius is projected using
   camera depth, vertical FOV and physical viewport height. At a projected radius
   of 16 pixels or less, retain 24 of the original 36 taps. Above 24 pixels, retain
   all 36. Between these bounds the two outer radial samples per direction fade
   in with `smoothstep`. The transition still executes 36 taps. All six directions
   and the four innermost radial taps remain, preserving thin contact detail.
   Distance alone would ignore zoom and resolution; projected size handles both.
2. **Separate exact AO from shadow traversal.** A dedicated compute pass writes
   unquantized AO into the receiver queue; the subsequent shadow pass consumes
   it. This keeps AO out of the shader that carries ray traversal state.
3. **Compact exact receivers by 16 × 16 tile.** Workgroups gather at most 256
   receivers locally and reserve one contiguous global range. This reduces
   global atomic contention and keeps nearby receivers together for AO texture
   reads and shadow traversal. Every original fallback receiver remains present.

Geometry, material identity compatibility, direct-shadow sample budgets and
shadow-map resolution are unchanged. The AO queue payload costs one extra
32-bit word per possible output pixel: about 26.1 MB at this resolution.

## Scene-normalized budgets and tap overlay

The RHS **AO diagnostics** section contains **AO tap budget** and its 0/12/24/36
colour legend. **Lighting → Minimum AO taps** controls the minimum. The overlay
shares the shader's budget function, importance texture and current-frame GPU
range. It shows selected horizon taps per surface receiver, before border
rejection and receiver sharing; coarse voxel AO is separate. It also appears as
a Lighting visibility stage view and persists as `svoStage=raster-ao-taps`.

The fixed 16–24 pixel thresholds were ineffective in the garden hero at its
default Retina camera: zero receivers reached the reduced tier, 93.93% were at
full quality, and the other 6.07% faded weights while still executing 36 taps.
Changing the minimum from 36 to zero barely changed the image.

Normalizing projected footprint against the visible frame made the minimum
attainable, but still spent 36 taps on near, featureless ground. With minimum 24,
that version averaged 31.67 taps: the horizon pass saved about 0.33 ms, which its
normalization pass consumed. It did not produce a net gain.

The replacement uses **local geometric variation times projected AO footprint**.
A 16×16 workgroup loads each reduced receiver once, then reuses those positions
and normals in local quads. Normal changes and departure from the receiver's
plane estimate where losing samples is more likely to matter. Coplanar regions
get zero importance, are eligible for the slider minimum, and do not determine
the range of detailed receivers. Classification adds no neighbourhood texture
reads.
A half-resolution R32F importance texture with max mips costs about 8.7 MB at
3448×1892; AO and the overlay read that same publication. There is no production CPU readback.

The base budget maps the lowest 10% of positive importance to the slider count;
the highest 10% uses 36. Between them, `smoothstep` grades the desired count.
Nested samples are retained with at most one fractional sample, so intermediate
budgets skip real texture reads. Flat/equal-importance views use the minimum.
For settings below 24, a max-mip pyramid inspects the AO sampling footprint
using four bounded reads. It supplies a smoothly weighted contact floor up to
24 taps, independent of the slider, protecting flat receivers near plants or
rocks while keeping slider counts monotonic. Minimum positive-importance
receivers retain the requested minimum even in a view filled with detail.
Point-only importance at minimum 4 failed the quality gate (RMSE 3.15/255)
because local planarity alone missed these contacts, so that variant is not used.
Setting 36 bypasses classification. The existing still-view reuse also reuses
this publication when receiver inputs are unchanged.

This is a cheap visual-loss proxy, not a conservative proof that contact AO is
absent. The default remains 24, preserving four inner samples in all six
directions; low settings intentionally make a larger tradeoff. No temporal
history is used. A changed view can change the normalization range.

At minimum 24, the garden's stationary view averages **25.10 taps**: 4,517,072
locally flat surface pixels use 24, while 1,364,683 detailed pixels average
28.74. Against fixed 36 taps, 24 alternating pairs measured **11.67 → 11.47 ms**
for the dry GPU frame (about 1.7%). The combined sun/horizon pass was
3.28 → 2.62 ms, with 0.39 ms for classification. Depths and exact-receiver
counts match. Display RMSE is **0.35/255**, with 0.042% of surface pixels differing
by more than eight levels in any channel. The orbit comparison was effectively
even at 12.65 ms for both arms (paired median 0.990×), with RMSE 0.36. These are
aggregate errors, not a claim that every contact is identical or that the
default produces a consistent frame-time win.

With footprint protection and minimum **4**, the same garden view averages
**17.05 taps**. 2,259,433 surface pixels use exactly 4; contact/detail regions
average 25.15. Twenty-four alternating pairs give:

| Garden, minimum 4 | Fixed 36 taps | Adaptive | Reduction | Display RMSE / 255 |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 11.67 ms | 11.14 ms | 4.5% | 0.409 |
| Orbiting | 11.60 ms | 11.08 ms | 4.5% | 0.421 |

Both comparisons preserve every depth and exact receiver. 0.065–0.069% of
surface pixels differ by more than eight display levels in any channel; local
maxima reach 26, so low budgets still trade some contact detail. Classification
plus footprint bounds costs about 0.39 ms, while the sun/horizon pass falls
from 3.34 to 2.29 ms. Coarse AO and direct shadows remain; reduced horizon taps
do not translate proportionally into whole-frame savings. The default stays 24.

`tools/benchmark-raster-ao-census.ts` measures the same importance/budget planes
outside timed frames and separates locally flat and detailed pixel counts.
The paired runner accepts `FLUID_SVO_DRY_FRAME_FAR_AO_TAPS` for slider sweeps.

## Results with the earlier fixed thresholds

Alternating reference/candidate order, four warmup pairs, 24 measured pairs per
close fixture. Reference shaders restore the fixed 36-tap AO, original queue
layout and combined exact pass; the new AO pass is omitted entirely in that arm.
Both arms share scene resources and mesh data. Receiver counts match.

| Close view | Original GPU frame | Adaptive + tiled fallback | Reduction |
| --- | ---: | ---: | ---: |
| Authored lights, stationary | 20.05 ms | 18.42 ms | 8.2% |
| Authored lights, orbiting | 20.25 ms | 18.61 ms | 8.1% |
| Added point, spot and sphere-area lights, stationary | 183.89 ms | 146.28 ms | 20.5% |
| Added point, spot and sphere-area lights, orbiting | 180.55 ms | 147.19 ms | 18.5% |

The distant view (camera distance 25 m, 32 pairs) measured 11.01 → 10.16 ms
stationary and 10.88 → 10.03 ms orbiting, about 7.7–7.8% lower median frame time.
That run had occasional timing spikes in both arms; paired median speedups were
1.077× and 1.086×. It retained all 266,077 stationary exact receivers. Distant
display RMSE was 0.52–0.53, with zero depth mismatches and 0.099–0.103% of surface
pixels differing by more than eight levels in any channel.

The mixed-light stress fixture remains expensive and has wider timing spread;
unsupported exact rays still dominate. It is not the normal authored scene.
Adaptive taps alone had little measurable effect in the close view and about a
2% improvement in the distant view. The larger close-view gain comes from
scheduling: exact AO plus exact shadows fell from about 4.46 ms combined to
1.70 + 1.11 ms in the stationary authored case.

Full-resolution close-view comparisons have zero depth mismatches, display RMSE
0.31–0.33 on a 0–255 scale, and only 0.021–0.026% of surface pixels differing by
more than eight levels in any channel. Local maxima reach 48 during the orbit,
so this is a small aggregate AO tradeoff, not a claim of identical images.
More aggressive sample removal and sharing across material boundaries were
rejected because they visibly weakened contact shading.

## Reproduction

`tools/benchmark-svo-dry-frame-gpu.ts` accepts
`FLUID_SVO_DRY_FRAME_CAMERA_DISTANCE_SCALE` and
`FLUID_SVO_DRY_FRAME_LIGHTING_REFERENCE` (a JSON file). The latter contains
`pipelines`, exact WGSL `replacements: [{search, replacement}]`, and optional
`omitPasses`. Stale source matches, missing pipeline labels and accidentally
submitted omitted passes fail the comparison. Reference source for this study
is commit `767fc07f`.

Generate the reference JSON from current and original functions:

```sh
mkdir -p artifacts/retina-lighting
python3 - <<'PY'
import json, pathlib, subprocess
root = pathlib.Path('lib/svo/features/lighting-visibility')
def fn(source, name):
    start = source.index('fn ' + name + '(')
    return source[start:source.index('\n}', start) + 2]
changes = []
for file, names in {
    'svo-raster-ao.ts': ['rasterHorizonVisibility'],
    'raster-light-visibility.ts': ['rasterLightEdgeAt', 'rasterExactEdgesMain', 'rasterLightEdgesMain'],
}.items():
    path = root / file
    current = path.read_text()
    original = subprocess.check_output(['git', 'show', '767fc07f:' + str(path)], text=True)
    changes += [dict(search=fn(current, name), replacement=fn(original, name)) for name in names]
pathlib.Path('artifacts/retina-lighting/reference.json').write_text(json.dumps(dict(
    pipelines=['Tiled horizon contact AO', 'Resolve visibility and exact edges', 'Exact visibility exceptions'],
    replacements=changes, omitPasses=['Exact receiver contact AO', 'Normalize visible AO footprint'])))
PY

env WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
  FLUID_SVO_DRY_FRAME_SCENE=backyard-pool-party \
  FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/pool-party-live.ts \
  FLUID_SVO_DRY_FRAME_WIDTH=3448 FLUID_SVO_DRY_FRAME_HEIGHT=1892 \
  FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=1 \
  FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_SURFACE_MESH=1 \
  FLUID_SVO_DRY_FRAME_MESHER=dual-marching-cubes \
  FLUID_SVO_DRY_FRAME_SPECIALIZED_LIGHTING=1 \
  FLUID_SVO_DRY_FRAME_CONE_TRACING=raster-ao FLUID_SVO_DRY_FRAME_CONE_FANOUT=0 \
  FLUID_SVO_DRY_FRAME_CONE_SCALE=1 FLUID_SVO_DRY_FRAME_TIMING=gpu \
  FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=1 FLUID_SVO_DRY_FRAME_VISIBILITY_COMPARE=1 \
  FLUID_SVO_DRY_FRAME_CAMERA_DISTANCE_SCALE=.55 FLUID_VISIBILITY_CYCLES=24 \
  FLUID_SVO_DRY_FRAME_LIGHTING_REFERENCE=artifacts/retina-lighting/reference.json \
  FLUID_SVO_DRY_FRAME_PRIMARY_TIMING=0 FLUID_SVO_DRY_FRAME_WARMUPS=4 \
  FLUID_SVO_DRY_FRAME_MESH_CAMERA_CHECKS=0 \
  FLUID_SVO_DRY_FRAME_OUT=artifacts/retina-lighting/close.json \
  node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

Use camera scale `1` for the distant view. Isolate timed runs from other GPU
jobs. Captures use fixed names beside the report; use separate output directories
when retaining images from multiple runs. `FLUID_SVO_DRY_FRAME_LIGHTING_BUDGET=1`
runs AO/shadow/sample/resolution ablations instead of the paired comparison.

GPU regression tests exercise adaptive weights, exact queue compaction with
empty/sparse/full tiles across repeated frames, and preservation of AO payload
words. Run them with `WEBGPU_NODE_MODULE` set in addition to repository checks:

```sh
npm run check:types
npm run test:unit
WEBGPU_NODE_MODULE="$PWD/node_modules/webgpu/index.js" FLUID_WEBGPU_BACKEND=metal \
  node --import tsx --test tests/svo-raster-ao.test.ts tests/svo-raster-visibility.test.ts
```
