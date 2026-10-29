# Rain in Shinjuku

`createTokyoRainScene()` authors a live narrow-band FLIP flooded alley. The
simulation uses a 6.25 cm grid in an 8 × 2 × 16 m domain, a 17–20 cm deep
central gutter, raised curbs, a roof runoff source, and repeated resolved
roof drops. These drops are deliberately larger than millimetre rain: this
resolution cannot resolve individual real raindrops.

The Blender set uses metres and the export transform `(x, y, z) → (x, -z, y)`.
Its street bed is at Z=.18 m and nominal waterline at Z=.38 m. Water must be
exported from the production solver to OBJ. The builder never synthesizes
replacement liquid. Thin airborne rain streaks and subpixel shader bump are
photographic detail, independent of the solved liquid silhouette.

```sh
node --import tsx tools/blender/export-nbflip-water.ts --scene=tokyo-rain --time=0.8
'/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --python tools/blender/tokyo_rain.py -- \
  --water-mesh artifacts/water-scenes/tokyo-rain/water.obj \
  --output artifacts/water-scenes/tokyo-rain/final.png \
  --width 1600 --height 1000 --samples 128
```

The editable `.blend` is saved alongside the PNG. `--no-render` saves only
the project. Running without a water path is an explicitly dry architectural
preview. The set includes modeled shopfronts, enamel blade signs, lanterns,
a city bicycle and cyclist beneath transparent vinyl, utilities, AC units,
and wet material detail. Native Japanese lettering uses the macOS Hiragino
font when present; a different machine can substitute a Japanese font in the
builder. Cycles provides a photographic lighting reference; the primary
deliverable is the live application with these Blender-authored assets and
the running narrow-band FLIP solver.

## Live application assets

`export-tokyo-rain.py` publishes fourteen reusable native assets to
`public/assets/tokyo-rain/`, preserving procedural albedo as UV texture data.
`bake_mesh_volume.py -- --collection tokyo-rain` converts those exact closed
meshes to FVX; `npm run generate:assets` registers their metadata. The scene
uses 55 instances with scene-owned placements, surrounding masonry,
asphalt and raised pavement. No baked water is registered as scenery.

Native static materials currently lack transmission. The clear umbrella
therefore contributes its wet rim, ribs and shaft in the live app; the
transparent film is present in the Cycles reference. Narrow geometry gets a
finite sampling envelope so its visual structure survives the native volume
resolution. Imported shopfronts use a 6 cm sampling pitch, signs 6.25 mm,
the outdoor lanterns 6.25 mm and interior apertures 5 cm, and the cyclist and vending machine 3.5 cm.
The small lantern volume includes the source's twelve physical bamboo hoops;
its previous paper-only export omitted these ribs. The updated volume retains
both paper and dark cedar surface samples without changing the Cycles source.
Transparent shop panes are optically empty in the live asset; visible warm
apertures and paper lanterns are separate emissive scene nodes.
Neon letters and rails likewise form separate emissive assets; their dark
enamel backing remains part of the unlit storefront.

The current shopfronts contain actual recessed rooms with cedar counters,
bottle shelves and hanging light strips. Alternating right-hand storefronts
use a separate closed roller-shutter asset. The cyclist is in a three-quarter
pose so both wheels and the riding posture remain legible.

Generated base-colour grain uses `base_color_image.py` to encode linear optical
values into sRGB PNGs before glTF export. The earlier Non-Color image export
stored linear bytes which glTF decoded as sRGB, darkening cedar from intended
mean red .108 to .0116. The corrected five assets preserve their production
GLB geometry and transforms exactly; only embedded colour images were replaced,
then their voxel volumes were rebaked. The v16 geometry and volume audit JSONs
record this preservation. There is no runtime brightness compensation.

The authored scene now seeds `mathutils.noise` as well as Python randomness.
Earlier refuse-bag deformation used an unseeded noise table, so regenerating
the editable full-scene `.blend` could change those small shapes. The latest
editable source is an authoring reference; it must not be described as an
exact geometry match to the preserved production GLBs or earlier Cycles frame.

To rebuild only a subset without losing the complete manifest, pass an asset
ID or quoted glob: `export-tokyo-rain.py -- --asset 'tokyo-shopfront-*'`.
Follow it with the corresponding `bake_mesh_volume.py --asset` calls and
`npm run generate:assets`. The full export and bake commands rebuild all assets.

Bicycle tyre bounds are grounded at Z=.18 m, putting the lower .20 m below
the simulated .38 m waterline. Bike, clothed rider, folded canvas backpack,
cranks and pedals share the same three-quarter transform in both outputs.

Native neon lettering has a finite sampling envelope and 8 cm extra mounting
clearance from the 6 cm storefront field, preventing the coarser backing from
burying thin letters. Each outdoor lantern is its own point fixture; rear shop
strips remain visible emissive geometry rather than an aggregate exterior lamp.
The lantern instance records carry the paper's authored emission color. The
point light must not inherit the first FVX palette entry: adding wooden hoops
put cedar first and accidentally reduced the lamp's emitted light by roughly
two orders of magnitude. Surface shading still samples the voxel palette.

The live rig contains 31 light records: one directional, fourteen neon point
fixtures, fourteen outdoor lanterns, and two mounted spotlights matching the
source blue far fill and warm ramen-shop practical. Both have 18 cm apertures
in opaque housings; the warm practical lights the front of the cyclist.
Distant office windows have frames and mullions in both representations. Point and spot fixtures approximate the source area lights;
their intensities are renderer radiance values, not Blender watts. The shared renderer now defaults to Lighting → Direct lights = 32, which
includes all31 records. Area samples is independent: one sample per area
uses31 visibility slots and two samples uses33, without losing the final
warm spotlight. Explicit lower light limits still deliberately omit fixtures. The cool ambient palette
uses approximately the source world's colour multiplied by its strength.

The inspected `live-color-v16-2x.png` capture advances actual NB-FLIP water to
0.8 seconds, uses World reflections and 200% render resolution, and clearly
resolves the lantern's physical hoops. Its capture reports no WebGPU errors or
failed World samples; the 100% frame recovers 17,107 off-screen/occluded
reflection samples. Corrected texture encoding and the explicit paper light
color restore visible woodwork and warm reflections. This is not a claim of
photographic equivalence. Storefront
repetition, the simplified rider, empty transparent umbrella membrane and
limited indirect illumination remain visible. The live scene lacks the
Cycles reference's volumetric rain/mist and multi-bounce light transport.


## Shared lighting contract correction

The October29 shared lighting fix removes automatically invented fixture ranges.
Generated point/area/spot lights now use range zero (unbounded physical falloff),
while explicitly supplied finite light-record ranges retain their authored
window. The previous point-light cap was4.5m with an additional squared fade;
at3m that fade retained only11.1% of the inverse-square contribution and at4m
only1.2%. This suppressed cross-alley spill independently of actual occlusion.
No Tokyo colors, exposure, powers, geometry or solver settings changed.
The current spot shadow-map implementation uses finite light range as its far
plane. Unbounded spots therefore take exact finite source-distance visibility
rays; this preserves correctness but can cost more until shadow-map coverage
is independently bounded by scene geometry with a proven fallback.

The environment diffuse helper now returns irradiance rather than normalized
cosine-weighted radiance: a constant sky L yieldsπL, and the existing Lambert
consumers divide byπ once. This is a shared energy correction affecting every
scene, not a night-scene gain. `environment-irradiance-units.test.ts` checks
constant-sky energy, independent gradient quadrature, and CPU/GPU agreement.
`scene-light-budget.test.ts` executes the production shader loop with32 area
lights/64 samples and preserves explicitly reduced record limits and finite
ranges. Visual/performance comparisons must use a new capture; v16 predates
these shared corrections.

The subsequent GI injection correction removes its separate four-light cap.
The feedback builder now visits all published records up to the shared ABI limit
of 32, evaluates both finite-source samples for each area or spot light, and
averages their energy. Fractional opacity-pyramid visibility multiplies incident
radiance; it is no longer treated as a Boolean that admits full light whenever
transmittance is nonzero. The CPU proxy reference uses the same two source
samples, while retaining its independent analytic solid-occlusion model.
`tests/radiance-injection-light-contract.test.ts` exercises the production GPU
functions. The four-tap opacity-pyramid visibility and low-rate diffuse feedback
remain approximations; these corrections do not establish Cycles-equivalent GI.

Native shadow rays now share their bounded receiver-to-FVX projection within
one shader invocation. The cache key contains the exact position, normal,
source owner and sculpt-cut count; invalid results are retained unchanged.
Every emitter sample still traces its own visibility ray, including farther
blockers belonging to the receiver's owner. The production GPU fixture in
`tests/sampled-surface-projection-gpu.test.ts` verifies bitwise result parity,
64 calls using one projection, key invalidation and the retained far blocker.
This removes repeated projection work without lowering light or ray counts;
its end-to-end performance benefit requires a successful exclusive capture.

## Exclusive live capture validation

`cone-gi-quality/cones-v5-validation.json` audits the 800×560, 0.8-second v5
capture. All ten independent GPU execution sentinels match, there are no GPU
errors, and World water reflections recover 11,069 samples with zero reported
failures. Exact dry-frame reuse changes no displayed, dry-HDR or World-HDR
bytes in the frozen forward/reverse comparison; the final advancing-water
cached frame also equals its frozen recomputation. Across 32 GPU-fenced host
samples per variant, median latency falls from 1,259.61 to 195.46 ms (6.44×).
These are serialized host timings, not interactive FPS. The scene still shows
repeated architecture, a simplified rider and coarse small-object detail.

GI/specular convergence remains unverified in v5: a capture harness bug treated
a void diagnostic-copy method as a Boolean and skipped submission; reused dry
frames also do not produce a new GI gather. The image, execution sentinels and
byte-parity checks are independent of that diagnostic error. Five subsequently
edited UI/persistence files differ from the capture's recorded source hashes;
its shader/backend hashes, including receiver projection reuse, match. The
validation JSON lists those files and records a +1.46% water-volume change,
without claiming exact volume conservation or Cycles-equivalent lighting.

## Matched production-input reference

For renderer diagnosis, `export-live-reference-layout.ts` exports the registered
scene camera, physical terrain, resolved primitive materials, production GLB
instance transforms and selected light records. `render-live-reference.py`
uses these inputs with an actual solver OBJ. It does not add the artistic
reference's transparent umbrella film, rain, mist, water bump, depth of field,
or bloom. The default 800×560 framing matches the current cone/GI captures.

```sh
node --import tsx tools/blender/export-live-reference-layout.ts tokyo-rain \
  artifacts/water-scenes/tokyo-rain/cone-gi-quality/matched-layout.json
tools/gpu/fluid-gpu-run.sh timing \
  '/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender' -b --python-exit-code 1 \
  --threads 6 --python tools/blender/render-live-reference.py -- \
  --layout artifacts/water-scenes/tokyo-rain/cone-gi-quality/matched-layout.json \
  --water-mesh artifacts/water-scenes/tokyo-rain/cone-gi-quality/solver-control.obj \
  --output artifacts/water-scenes/tokyo-rain/cone-gi-quality/cycles-matched-v1.png \
  --width 800 --height 560 --samples 64
```

This is a diagnostic reference, not a pixel oracle. Cycles uses source GLB
triangles; the interactive app remains voxel geometry. The sidecar records
source hashes, the supplied water's simulation report, each light conversion,
and remaining differences. Point/spot conversion matches far-field
inverse-square irradiance, while the live 1m attenuation cap and spot beam
interpolation differ. Only fixtures with explicit light records suppress their
duplicate diffuse mesh emission; other emissive surfaces remain full emitters.
Each explicit light excludes its own source object from shadow blocking, matching
the live owner exclusion. Terrain uses its published base color and roughness;
procedural material modulation and sky angular interpolation remain approximate.
The AgX implementations and finite cone gathering also differ from Cycles.

The shared water-reflection reconstruction now preserves subpixel footprints:
its former minimum one-pixel binomial filter broadened even very small authored
roughness. A fractional footprint blends adjacent integer-radius kernels by
second moment, including the zero-radius impulse; the existing depth/normal
edge rejection remains. `tests/water-reflection-footprint.test.ts` verifies the
production GPU response at the supported .04 roughness minimum, constant
radiance, impulse energy and continuous variance. The zero-footprint test is
an analytic limit, not a new zero-roughness UI option. Noninteger footprints
above one pixel need two nine-tap kernels; this is a quality correction, not a
performance optimization. Visual benefit is pending the next production capture.

For optical comparisons, export with `--normals=optical`. This optional mode
writes the production vertex-normal lane as OBJ `vn` records; the default
remains the legacy position-only export. The matched renderer retains these
vectors as float32 corner attributes and normalizes their interpolation into
the water BSDF. This avoids Blender's compressed custom-normal representation
(maximum measured vector error .00591 on a few sliver faces), while retaining
the original triangle positions and winding. Every face corner and attribute
value is checked on import; the sidecar records the measured errors. Without
`vn` records the renderer explicitly reports its legacy welded, averaged
triangle-normal mode. Those two optical treatments must not be described as
the same water shading, even when their mesh positions are identical.

`cone-gi-quality/cycles-matched-v7.png` uses the exact `cones-v7.obj` vertex
buffer exported by the initial live frame at .8 seconds: 89,400 triangles and
268,200 optical-normal corners. No second simulation or extraction is used.
The captured normals include the shared quadratic B-spline derivative
reconstruction, which reproduces affine field gradients without the previous
subcell-phase bias. The matched renderer validates every corner position and
preserves all optical vectors with zero measured float32 attribute error.
Blender's own compressed custom normals differed by up to .00612, so the BSDF
uses the validated corner attribute instead. The live frame also uses the
fractional reflection-filter footprint correction described above.

The initial v7 image and mesh export passed their GPU execution sentinels.
A later uncached performance stress variant failed its sentinel; the complete
v7 capture is unsuccessful, and no performance or 2× comparison is claimed
from it. The Cycles sidecar records this distinction. Future references read
`<mesh>.obj.json` when present and verify the mesh hash and captured export
sentinel before rendering.

Inspection of this exact-water pair still shows broader, paler blue reflections
and fine stippling in the live image, while Cycles has more continuous saturated
neon and lantern reflections. This is a remaining fidelity gap, not evidence
that photorealism has been reached. Matching water positions and optical normals
does not remove the documented differences in static voxel reconstruction,
light attenuation, indirect transport, environment, grading and optional bloom.
