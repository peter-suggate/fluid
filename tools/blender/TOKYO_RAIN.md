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
their intensities are renderer radiance values, not Blender watts. For the
full rig, use Lighting → Direct light samples = 32 and Area samples = 1.
The general renderer default remains eight samples. The cool ambient palette
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
